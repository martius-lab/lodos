"""
MuJoCo's `implicitfast` integrator, in Warp: velocity-implicit in the actuators' and the
dofs' damping, as MJWarp's `forward.implicit` and `derivative.deriv_smooth_vel` have it.

The step's acceleration solves `(M - dt D) qacc = qfrc_smooth + qfrc_constraint`
instead of `M qacc = ...`. `D` is the derivative of the smooth forces in `qvel` without
the bias forces' (implicitfast's choice, which keeps it symmetric): each actuator's
`moment moment^T dforce/dvelocity`, and `-damping` on the diagonal, on the mass matrix's
own sparsity -- a dof and its ancestors -- as MuJoCo keeps it. A free body standing
alone (a tree of its own 6 dofs, nothing massive below) solves its own 6x6 system
instead, with its gyroscopic force's derivative as well, as MuJoCo's `mjd_freeMhat`
does; that block is not symmetric, and is solved by LU.

`RbdModel.implicit_constraints` picks the matrix the constraint solve sees:

  0  MuJoCo's form. The solvers see M, as under Euler; from their acceleration,
     `qacc_int = (M - dt D)^-1 M qacc` (`correct`), the free bodies by their own
     blocks, is what the step integrates. M is factored twice a step.
  1  `M - dt D` throughout. It replaces M before the factorization (`prepare`), so
     `qacc_smooth`, the mass solves and the contact solve's gathered M are the
     implicit matrix's, and the solver's acceleration is integrated as it is: the
     contact impulse sees the damping that acts in the same step. One factorization.
     The free bodies' gyroscopic term, not being symmetric, is left out.

Without constraints the two are the same step, the gyroscopic term apart.
"""

import warp as wp

from lodos.rbd import smooth
from lodos.rbd.smooth import mul_M_tree, solve_M_tree
from lodos.rbd.types import AFFINE, DYN_NONE, RbdData, RbdModel
from lodos.utils import math as lmath


# Launched (B, nu)
@wp.kernel
def _actuator_dfdv(m: RbdModel, d: RbdData):  # type: ignore
    """
    Each actuator's force derivative in its own velocity, MJWarp's
    `_qderiv_actuator_passive_vel` for the gain and bias types `lodos.rbd` admits: the
    affine bias's velocity coefficient, plus the affine gain's times the control (the
    last activation, for a stateful actuator); none where the force is clamped.

    Adapted from MuJoCo Warp 3.14.0's `mujoco_warp/_src/derivative.py`, `_qderiv_actuator_passive_vel`.
    """
    b, u = wp.tid()

    gain = float(0.0)
    if m.actuator_gaintype[u] == AFFINE:
        gain = m.actuator_gainprm[u, 2]
    bias = float(0.0)
    if m.actuator_biastype[u] == AFFINE:
        bias = m.actuator_biasprm[u, 2]

    dfdv = float(0.0)
    if gain != 0.0 or bias != 0.0:
        f = d.actuator_force[b, u]
        rng = m.actuator_forcerange[u]
        if m.actuator_forcelimited[u] == 0 or (f > rng[0] and f < rng[1]):
            dfdv = bias
            if gain != 0.0:
                if m.actuator_dyntype[u] != DYN_NONE:
                    dfdv += gain * d.act[b, m.actuator_actadr[u] + m.actuator_actnum[u] - 1]
                else:
                    dfdv += gain * d.ctrl[b, u]

    d.actuator_dfdv[b, u] = dfdv


@wp.kernel
def _subtract_dtD(m: RbdModel, d: RbdData, A: wp.array(dtype=float, ndim=2)):  # type: ignore
    """
    `A -= dt D` at one entry of `D`'s pattern (`types.implicit_pairs_np`) and its mirror:
    the moments of the actuators reaching both dofs times their force derivatives, and a
    damped dof's `-damping` on the diagonal. Each pair is its own thread: no two write the
    same entry.

    Launch: dim=(npair, B), a thread per pair and world, the batch fastest as `M_blk` is laid out, by `prepare`.
    In: A (nblk, B) float32, the per-tree blocks: d.M_blk with `M - dt D` throughout, d.Mi_blk in MuJoCo's form.
    Reads: d.actuator_moment (B, nu, nv) float32, d.actuator_dfdv (B, nu) float32; the pairs and the dofs' damping.
    Writes: A at the pair's entry and its mirror.
    """
    p, b = wp.tid()  # int, int: the pair, the world
    i = m.dpair_i[p]  # int: the pair's dof
    j = m.dpair_j[p]  # int: i or an ancestor of i

    dij = float(0.0)  # float: D[i, j]
    if m.disable_actuation == 0:
        for k in range(m.dpair_beg[p], m.dpair_beg[p + 1]):  # k: int, a slot in dpair_act
            u = m.dpair_act[k]  # int: an actuator reaching both dofs
            dij += d.actuator_moment[b, u, i] * d.actuator_moment[b, u, j] * d.actuator_dfdv[b, u]
    if i == j and m.disable_damper == 0:
        dij -= m.dof_damping[i]

    if dij != 0.0:
        t = m.dof_tree[i]  # int: the dofs' tree
        beg = m.tree_beg[t]  # int: the tree's first dof
        n = m.tree_beg[t + 1] - beg  # int: the tree's dof count
        o = m.blk_beg[t]  # int: the tree's block offset
        r = i - beg  # int: i's row in the block
        c = j - beg  # int: j's column in the block
        v = A[o + r * n + c, b] - m.dt * dij  # float: the entry, dt D[i, j] subtracted
        A[o + r * n + c, b] = v
        A[o + c * n + r, b] = v


@wp.kernel
def _implicit_qacc(m: RbdModel, d: RbdData):  # type: ignore
    """
    MuJoCo's form: `qacc_ma = M qacc`, then `qacc_int = (M - dt D)^-1 qacc_ma`, per tree.

    Launch: dim=(B, ntree), a thread per world and tree, the batch slowest, as `smooth._solve_qacc_smooth`, by
    `correct`.
    Reads: d.M_blk and d.Li_blk (nblk, B) float32, d.qacc (B, nv) float32. Writes: d.qacc_ma and d.qacc_int (B, nv)
    float32.
    """
    b, t = wp.tid()  # int, int: the world, the tree
    beg = m.tree_beg[t]  # int: the tree's first dof
    n = m.tree_beg[t + 1] - beg  # int: the tree's dof count
    o = m.blk_beg[t]  # int: the tree's block offset
    mul_M_tree(d.M_blk, d.qacc, d.qacc_ma, b, beg, n, o)
    solve_M_tree(d.Li_blk, d.qacc_ma, d.qacc_int, b, beg, n, o)


# Launched (B, nfree)
@wp.kernel
def _free_bodies(m: RbdModel, d: RbdData):  # type: ignore
    """
    A free body standing alone, MuJoCo's form: `qacc_int` over its 6 dofs from its own
    block -- M, `dt damping` on the diagonal, and the derivative of its gyroscopic
    force -- by LU. MJWarp's `_implicit_free_body_solve`, without the fluid forces
    `lodos.rbd` refuses. A singular block leaves `_implicit_qacc`'s answer.

    Adapted from MuJoCo Warp 3.14.0's `mujoco_warp/_src/forward.py`, `_implicit_free_body_solve`.
    """
    b, k = wp.tid()
    body = m.free_body[k]
    adr = m.body_dofadr[body]
    o = m.blk_beg[m.dof_tree[adr]]

    A = lmath.mat66()
    for r in range(6):
        for c in range(6):
            A[r, c] = d.M_blk[o + r * 6 + c, b]
    if m.disable_damper == 0:
        for r in range(6):
            A[r, r] += m.dt * m.dof_damping[adr + r]

    mass = m.body_mass[body]
    w_local = wp.vec3(d.qvel[b, adr + 3], d.qvel[b, adr + 4], d.qvel[b, adr + 5])
    lin, rot = lmath.free_bias_vel_blocks(
        mass, d.xmat[b, body], d.ximat[b, body], m.body_inertia[body], d.xipos[b, body] - d.xpos[b, body], w_local
    )
    for r in range(3):
        for c in range(3):
            A[r, 3 + c] += -m.dt * mass * lin[r, c]
            A[3 + r, 3 + c] += m.dt * rot[r, c]

    LU, pivot, ok = lmath.lu_factor_6x6(A)
    if ok:
        y = lmath.vec6()
        for r in range(6):
            y[r] = d.qacc_ma[b, adr + r]
        x = lmath.lu_solve_6x6(LU, pivot, y)
        for r in range(6):
            d.qacc_int[b, adr + r] = x[r]


def active(m: RbdModel) -> bool:  # type: ignore
    """
    Whether the step is implicit: `implicitfast`, unless actuation, springs and damping are all off, as MuJoCo tests.

    Called by `lodos.rbd.forward.fwd_acceleration` and `lodos.rbd.integrate.integrate`.
    In: m, the model: m.implicit and the disable flags. Out: bool.
    """
    return bool(m.implicit) and not (m.disable_actuation and m.disable_spring and m.disable_damper)


def prepare(m: RbdModel, d: RbdData):  # type: ignore
    """
    Once the actuator forces are known: `D` into the matrix the step solves with. Under
    `M - dt D` throughout, into `M_blk` itself, which `smooth.build_m` left unfactored, and
    its factor into `L_blk`; under MuJoCo's form, into `Mi_blk` beside M, factored into
    `Li_blk` -- after `smooth.solve_m`, whose tiled solve reads the scratch this reuses.

    Called by `lodos.rbd.forward.fwd_acceleration`.
    In: m, d, B worlds' model and state; reads d.actuator_force and d.ctrl (B, nu), d.act (B, na), d.actuator_moment
    (B, nu, nv) and d.M_blk (nblk, B), float32. Out: d.actuator_dfdv (B, nu) float32; d.M_blk and d.L_blk, or in
    MuJoCo's form d.Mi_blk and d.Li_blk, (nblk, B) float32; d.M_big and d.L_big (B * nbig, big_np, big_np) float32.
    """
    if m.nu and not m.disable_actuation:
        wp.launch(_actuator_dfdv, dim=(m.B, m.nu), inputs=(m, d))
    if m.implicit_constraints:
        if m.npair:
            wp.launch(_subtract_dtD, dim=(m.npair, m.B), inputs=(m, d, d.M_blk))
        smooth.factor_blocks(m, d, d.M_blk, d.L_blk)
    else:
        wp.copy(d.Mi_blk, d.M_blk)
        if m.npair:
            wp.launch(_subtract_dtD, dim=(m.npair, m.B), inputs=(m, d, d.Mi_blk))
        smooth.factor_blocks(m, d, d.Mi_blk, d.Li_blk)


def correct(m: RbdModel, d: RbdData):  # type: ignore
    """
    MuJoCo's form, after the constraint solve: the acceleration the step integrates, into `qacc_int`.

    Called by `lodos.rbd.integrate.integrate`.
    In: m, d, B worlds' model and state; reads d.qacc (B, nv) float32, d.M_blk and d.Li_blk (nblk, B) float32.
    Out: d.qacc_ma and d.qacc_int (B, nv) float32.
    """
    wp.launch(_implicit_qacc, dim=(m.B, m.ntree), inputs=(m, d))
    if m.nfree:
        wp.launch(_free_bodies, dim=(m.B, m.nfree), inputs=(m, d))
