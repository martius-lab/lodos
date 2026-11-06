"""
Semi-implicit Euler, in Warp.

Semi-implicit means velocity is advanced first and position is advanced with the
*new* velocity; `_integrate_pos` therefore has to run after `_integrate_vel`, and
they are separate launches for exactly that reason.

`integrate` is the step's entry point: under `implicitfast` (`lodos.rbd.implicit`) the
acceleration it advances the velocity by is the implicit one.
"""

import warp as wp

from lodos.rbd import implicit as rbd_implicit
from lodos.rbd.types import (BALL, DYN_FILTEREXACT, FREE, MINVAL, RbdData,
                                 RbdModel)
from lodos.utils import math as lmath


@wp.kernel
def _integrate_vel(m: RbdModel, d: RbdData, qacc: wp.array(dtype=float, ndim=2)):  # type: ignore
    """
    `qvel += dt * qacc`.

    Launch: dim=(B, nv), a thread per world and dof, by `symplectic_euler`.
    In: qacc (B, nv) float32, the acceleration. Writes: d.qvel (B, nv) float32, added to.
    """
    b, v = wp.tid()  # int, int: the world, the dof
    d.qvel[b, v] += m.dt * qacc[b, v]


@wp.kernel
def _integrate_pos(m: RbdModel, d: RbdData):  # type: ignore
    """
    Advances each joint's coordinates by the velocity `_integrate_vel` just wrote.

    A rotational coordinate is a quaternion and its velocity a rotation vector, so
    the two cannot simply be added: `quat_integrate` turns the angular velocity
    into a rotation of `dt * |w|` about its own axis and composes it on the right,
    in the joint's frame.

    Launch: dim=(B, njnt), a thread per world and joint, by `symplectic_euler`.
    Reads: d.qvel (B, nv) float32. Writes: d.qpos (B, nq) float32.
    """
    b, j = wp.tid()  # int, int: the world, the joint

    jtype = m.jnt_type[j]  # int: FREE, BALL, SLIDE or HINGE
    qadr = m.jnt_qposadr[j]  # int: the joint's first qpos, a free joint's quaternion's below
    adr = m.jnt_dofadr[j]  # int: the joint's first dof, a free joint's first angular dof below

    if jtype == FREE:
        for c in range(3):  # c: int, a coordinate
            d.qpos[b, qadr + c] += m.dt * d.qvel[b, adr + c]
        qadr += 3
        adr += 3

    if jtype == FREE or jtype == BALL:
        q = lmath.quat_integrate(  # wp.vec4: the advanced orientation
            wp.vec4(
                d.qpos[b, qadr], d.qpos[b, qadr + 1],
                d.qpos[b, qadr + 2], d.qpos[b, qadr + 3],
            ),
            wp.vec3(d.qvel[b, adr], d.qvel[b, adr + 1], d.qvel[b, adr + 2]),
            m.dt,
        )
        for c in range(4):
            d.qpos[b, qadr + c] = q[c]
    else:
        d.qpos[b, qadr] += m.dt * d.qvel[b, adr]


@wp.kernel
def _integrate_act(m: RbdModel, d: RbdData):  # type: ignore
    """
    Advances actuator activation state, clamped to its range.

    `FILTEREXACT` integrates its first-order lag in closed form over the step
    instead of taking an Euler step, which is what makes it stable at a `tau`
    below the timestep.

    Launch: dim=(B, nu), a thread per world and actuator, by `symplectic_euler`.
    Reads: d.act_dot (B, na) float32; the actuators' dynamics and activation ranges. Writes: d.act (B, na) float32.
    """
    b, u = wp.tid()  # int, int: the world, the actuator

    adr = m.actuator_actadr[u]  # int: the actuator's activation, -1 if it has none
    if adr == -1:
        return

    act = d.act[b, adr]  # float: the activation
    if m.actuator_dyntype[u] == DYN_FILTEREXACT:
        tau = wp.max(m.actuator_dynprm[u, 0], MINVAL)  # float: the filter's time constant (s)
        act += d.act_dot[b, adr] * tau * (1.0 - wp.exp(-m.dt / tau))
    else:
        act += d.act_dot[b, adr] * m.dt

    rng = m.actuator_actrange[u]  # wp.vec2: the activation range, (lower, upper)
    if m.actuator_actlimited[u] != 0:
        act = wp.clamp(act, rng[0], rng[1])

    d.act[b, adr] = act


@wp.kernel
def _advance_time(m: RbdModel, d: RbdData):  # type: ignore
    """
    Advances each world's time by one timestep.

    Launch: dim=(B,), a thread per world, by `symplectic_euler`.
    Writes: d.time (B,) float32, added to.
    """
    d.time[wp.tid()] += m.dt


def symplectic_euler(m: RbdModel, d: RbdData, qacc=None):  # type: ignore
    """
    Advances the state by one timestep, given `qacc` (or the acceleration passed) and
    `act_dot`.

    In this order: activations, then velocity, then position from the new velocity, then time.

    Called by `integrate`.
    In: m, d, B worlds' model and state; qacc (B, nv) float32, the acceleration, or None for d.qacc; reads d.act_dot
    (B, na) float32. Out: d.act (B, na), d.qvel (B, nv), d.qpos (B, nq) and d.time (B,) float32, advanced a step.
    """
    if m.na:
        wp.launch(_integrate_act, dim=(m.B, m.nu), inputs=(m, d))

    wp.launch(_integrate_vel, dim=(m.B, m.nv), inputs=(m, d, d.qacc if qacc is None else qacc))
    wp.launch(_integrate_pos, dim=(m.B, m.njnt), inputs=(m, d))
    wp.launch(_advance_time, dim=(m.B,), inputs=(m, d))


def integrate(m: RbdModel, d: RbdData):  # type: ignore
    """
    The step's integration, after the constraint solve: semi-implicit Euler with `qacc`,
    the model's integrator either way. Under `implicitfast` in MuJoCo's form the
    acceleration integrated is the implicit one `lodos.rbd.implicit.correct` solves for,
    and `qacc` stays the solver's, as MuJoCo leaves `d->qacc`; with `M - dt D` throughout
    the solver's `qacc` already is it.

    Called by `lodos.step.step_sim`, `viewer.session.Runner.capture` and
    `tools.performance_tuning.profile_scene.stages_of`.
    In: m, d, B worlds' model and state; reads d.qacc (B, nv) float32 and d.act_dot (B, na) float32.
    Out: d.act, d.qvel, d.qpos and d.time advanced a step (`symplectic_euler`); in MuJoCo's form of `implicitfast`,
    d.qacc_ma and d.qacc_int (B, nv) float32 too.
    """
    if m.implicit and not m.implicit_constraints and rbd_implicit.active(m):
        rbd_implicit.correct(m, d)
        symplectic_euler(m, d, d.qacc_int)
    else:
        symplectic_euler(m, d)
