"""
The forward rigid-body pass, in Warp.

Four sub-passes, in MuJoCo's order -- `fwd_position`, `fwd_velocity`, `actuation` and
`fwd_acceleration` (`fwd_rbd`). The mass matrix is built and factored here, and the
contact solve reads it from here too (`lodos.cd.driver.gather_M`, `lodos.cd.rowspace`).
"""

import warp as wp

from lodos.rbd import implicit as rbd_implicit
from lodos.rbd import passive as rbd_passive
from lodos.rbd import smooth
from lodos.rbd.types import (AFFINE, DYN_FILTER, DYN_FILTEREXACT,
                                 DYN_INTEGRATOR, MINVAL, RbdData, RbdModel)


#######################################################################################
# Actuation
#######################################################################################


@wp.func
def _clamp(x: float, limited: int, rng: wp.vec2):
    """
    Clamps only where the model says the quantity is limited.

    Called by `_actuator_force` and `_qfrc_actuator`.
    In: x float, the value; limited int, nonzero if it is limited; rng wp.vec2, its range (lower, upper). Out: float.
    """
    if limited == 0:
        return x
    return wp.clamp(x, rng[0], rng[1])


@wp.kernel
def _actuator_force(m: RbdModel, d: RbdData):  # type: ignore
    """
    Scalar force per actuator, and the activation derivative that goes with it.

    `ctrl` is clamped first and everything downstream uses the clamped value --
    including `act_dot`, so a stateful actuator integrates what the model would
    actually apply rather than what was asked for.

    Launch: dim=(B, nu), a thread per world and actuator, by `actuation`.
    Reads: d.ctrl (B, nu) float32, d.act (B, na) float32, d.actuator_length and d.actuator_velocity (B, nu) float32;
    the actuators' dynamics, gains, biases and ranges. Writes: d.act_dot (B, na) float32, d.actuator_force (B, nu)
    float32.
    """
    b, u = wp.tid()  # int, int: the world, the actuator

    ctrl = d.ctrl[b, u]  # float: the control, clamped to its range
    if m.disable_clampctrl == 0:
        ctrl = _clamp(ctrl, m.actuator_ctrllimited[u], m.actuator_ctrlrange[u])

    adr = m.actuator_actadr[u]  # int: the actuator's first activation, -1 if it has none

    dyntype = m.actuator_dyntype[u]  # int: DYN_NONE, DYN_INTEGRATOR, DYN_FILTER or DYN_FILTEREXACT
    if adr != -1:
        act = d.act[b, adr]  # float: the activation
        if dyntype == DYN_INTEGRATOR:
            d.act_dot[b, adr] = ctrl
        elif dyntype == DYN_FILTER or dyntype == DYN_FILTEREXACT:
            d.act_dot[b, adr] = (ctrl - act) / wp.max(m.actuator_dynprm[u, 0], MINVAL)
        else:
            d.act_dot[b, adr] = 0.0

    # A stateful actuator drives its force from the last of its activations, a
    # stateless one from `ctrl`.
    ctrl_act = ctrl  # float: what the gain multiplies, the control or the last activation
    if adr != -1:
        ctrl_act = d.act[b, adr + m.actuator_actnum[u] - 1]

    length = d.actuator_length[b, u]  # float
    vel = d.actuator_velocity[b, u]  # float

    gain = m.actuator_gainprm[u, 0]  # float
    if m.actuator_gaintype[u] == AFFINE:
        gain += m.actuator_gainprm[u, 1] * length + m.actuator_gainprm[u, 2] * vel

    bias = float(0.0)  # float
    if m.actuator_biastype[u] == AFFINE:
        bias = (
            m.actuator_biasprm[u, 0]
            + m.actuator_biasprm[u, 1] * length
            + m.actuator_biasprm[u, 2] * vel
        )

    d.actuator_force[b, u] = _clamp(
        gain * ctrl_act + bias, m.actuator_forcelimited[u], m.actuator_forcerange[u]
    )


@wp.kernel
def _qfrc_actuator(m: RbdModel, d: RbdData):  # type: ignore
    """
    `actuator_moment^T force`, plus any gravity compensation routed here.

    Launch: dim=(B, nv), a thread per world and dof, by `actuation`.
    Reads: d.actuator_moment (B, nu, nv) float32, d.actuator_force (B, nu) float32, d.qfrc_gravcomp (B, nv) float32;
    the joints' actuator force limits. Writes: d.qfrc_actuator (B, nv) float32.
    """
    b, v = wp.tid()  # int, int: the world, the dof

    qfrc = float(0.0)  # float: the dof's actuator force
    for u in range(m.nu):  # u: int, an actuator
        qfrc += d.actuator_moment[b, u, v] * d.actuator_force[b, u]

    j = m.dof_jntid[v]  # int: the dof's joint
    if m.ngravcomp:
        qfrc += d.qfrc_gravcomp[b, v] * float(m.jnt_actgravcomp[j])

    d.qfrc_actuator[b, v] = _clamp(
        qfrc, m.jnt_actfrclimited[j], m.jnt_actfrcrange[j]
    )


def actuation(m: RbdModel, d: RbdData):  # type: ignore
    """
    Actuator forces. Leaves `qfrc_actuator` at zero for an unactuated model.

    Called by `fwd_rbd`.
    In: m, d, B worlds' model and state; reads the controls, activations and transmission (`_actuator_force`).
    Out: d.act_dot (B, na) float32, d.actuator_force (B, nu) float32, d.qfrc_actuator (B, nv) float32.
    """
    if not m.nu or m.disable_actuation:
        d.act_dot.zero_()
        d.qfrc_actuator.zero_()
        return

    wp.launch(_actuator_force, dim=(m.B, m.nu), inputs=(m, d))
    wp.launch(_qfrc_actuator, dim=(m.B, m.nv), inputs=(m, d))


@wp.kernel
def _qfrc_smooth(m: RbdModel, d: RbdData):  # type: ignore
    """
    Every non-constraint force, summed.

    Launch: dim=(B, nv), a thread per world and dof, by `fwd_acceleration`.
    Reads: d.qfrc_passive, d.qfrc_bias, d.qfrc_actuator and d.qfrc_applied (B, nv) float32.
    Writes: d.qfrc_smooth (B, nv) float32.
    """
    b, v = wp.tid()  # int, int: the world, the dof

    d.qfrc_smooth[b, v] = (
        d.qfrc_passive[b, v]
        - d.qfrc_bias[b, v]
        + d.qfrc_actuator[b, v]
        + d.qfrc_applied[b, v]
    )


def fwd_position(m: RbdModel, d: RbdData):  # type: ignore
    """
    Everything that depends on `qpos` alone, mass matrix included. Under `implicitfast`
    with `M - dt D` throughout, M is factored only once `D` is in it (`fwd_acceleration`).

    Called by `fwd_rbd`.
    In: m, d, B worlds' model and state; reads d.qpos (B, nq) float32 and normalizes its quaternions in place.
    Out: the poses (`smooth.kinematics`), the center-of-mass frames (`smooth.com_pos`), d.crb (B, nbody) vec10,
    d.M_blk (nblk, B) float32 and, unless with `M - dt D` throughout, its factor d.L_blk (nblk, B) float32;
    d.actuator_length (B, nu) and d.actuator_moment (B, nu, nv) float32.
    """
    smooth.kinematics(m, d)
    smooth.com_pos(m, d)
    if m.implicit and m.implicit_constraints:
        smooth.build_m(m, d)
    else:
        smooth.factor_m(m, d)
    smooth.transmission(m, d)


def fwd_velocity(m: RbdModel, d: RbdData):  # type: ignore
    """
    Everything that additionally depends on `qvel`.

    Called by `fwd_rbd`.
    In: m, d, B worlds' model and state; reads d.qvel (B, nv) float32 and what `fwd_position` wrote.
    Out: d.actuator_velocity (B, nu) float32, d.cvel (B, nbody) and d.cdof_dot (B, nv) wp.spatial_vector,
    d.qfrc_passive, d.qfrc_gravcomp and d.qfrc_bias (B, nv) float32.
    """
    smooth.actuator_velocity(m, d)
    smooth.com_vel(m, d)
    rbd_passive.passive(m, d)
    smooth.rne(m, d)


def fwd_acceleration(m: RbdModel, d: RbdData):  # type: ignore
    """
    The unconstrained force and the acceleration it produces. Under `implicitfast`, also
    the implicit matrix (`lodos.rbd.implicit.prepare`): with `M - dt D` throughout,
    `qacc_smooth` is that matrix's.

    Called by `fwd_rbd`.
    In: m, d, B worlds' model and state; reads d.qfrc_passive, d.qfrc_bias, d.qfrc_actuator and d.qfrc_applied
    (B, nv) float32, and the factor. Out: d.qfrc_smooth and d.qacc_smooth (B, nv) float32; under `implicitfast`,
    what `lodos.rbd.implicit.prepare` writes.
    """
    wp.launch(_qfrc_smooth, dim=(m.B, m.nv), inputs=(m, d))
    if m.implicit and m.implicit_constraints:
        rbd_implicit.prepare(m, d)
    smooth.solve_m(m, d)
    if m.implicit and not m.implicit_constraints and rbd_implicit.active(m):
        rbd_implicit.prepare(m, d)


def fwd_rbd(m: RbdModel, d: RbdData):  # type: ignore
    """
    Everything unconstrained rigid-body dynamics produces, for one state.

    Nothing here synchronizes or touches the host, so the whole pass captures
    into a CUDA graph alongside collision and the contact solver.

    Called by `lodos.step.forward`, `lodos.rbd.io.load_state` and
    `tools.performance_tuning.profile_scene.stages_of`.
    In: m, d, B worlds' model and state: d.qpos (B, nq), d.qvel (B, nv), d.ctrl (B, nu), d.act (B, na) and
    d.qfrc_applied (B, nv), float32. Out: every derived field of d through d.qacc_smooth (B, nv) float32, as
    `fwd_position`, `fwd_velocity`, `actuation` and `fwd_acceleration` write them.
    """
    fwd_position(m, d)
    fwd_velocity(m, d)
    actuation(m, d)
    fwd_acceleration(m, d)
