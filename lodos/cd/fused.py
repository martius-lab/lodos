"""
The NLSP solver's Newton solve in one kernel per world.

MJWarp's Newton solver (`lodos.cd.solver.nlsp`) takes about ten launches an
iteration -- the line search, the constraint update, the gradient, the Hessian
`M + J^T D J` (dense, in global memory), the elliptic cone's curvature, the Cholesky
factorization, the convergence test -- under a graph-conditional loop, and the prox
solver (`nlsp.solve_prox`) wraps it in rounds of a few more. On a small model that is
its cost: a Newton iteration is nearly all launches and memory traffic, for a
problem one block of threads solves in microseconds.

This is the same solve as one kernel, one block of 1 or 2 warps per world, for small
models (nv <= 16):

- the world's rows -- J, `J qacc - aref`, `J search`, D, the friction loss, the elliptic
  contacts' line-search quadratics, the prox state -- and its dof vectors live in shared
  memory for the whole solve, behind plain per-thread loads and stores
  (`_shared_accessors`), the threads taking the rows in turn;
- the line search is MJWarp's: its row functions (`_eval_pt_direct_*`,
  `_eval_frictionloss_*`, `_eval_elliptic_*`), the same Newton step from zero, the same
  bracketing, the same `gtol`; its sums over rows are the warps' shuffles, then the warps'
  partials in shared memory;
- the constraint update is MJWarp's `_eval_constraint`;
- the Hessian `M + S^T S`, S the rows scaled by their curvature (below): a chunk at a time,
  each thread stages its row's rows of S -- none, one, or two on an elliptic contact in the
  cone's middle zone -- packed by a prefix sum over the block, and each thread owning
  entries of the packed lower triangle sums them over the staged rows; thread 0 factors
  it in shared memory and solves for the search direction;
- the convergence test is MJWarp's `_solve_done`: the improvement, the gradient or half
  the Newton decrement below the tolerance, each scaled by `meaninertia nv`, or the
  iteration budget spent;
- under the prox solver, its rounds are `nlsp._prox_round`'s: the Newton solve of the
  round's rows, then per contact `z <- f`, the De Saxce step on `Gamma` at the rounds
  `_prox_update` takes it, the rows' `aref` rewritten, and `_prox_check`'s test on the
  largest changes of `z` and `Gamma`. `nlsp._prox_init` sets the rows up before the
  kernel, and its cache fill carries `z` to the next step after it;
- a contact group (`lodos.cd.groups`, a `GROUP` contact) is solved as the contact its
  rows make: under mj as any other, and under the prox solver with its members' De Saxce
  term, `nlsp._group_gamma`, at the round's `qvel + dt qacc`, which the kernel reads from
  `qacc` in global memory, written there before the De Saxce step.

The problem is strictly convex: both solvers reach the same acceleration, and their
iterates differ by rounding alone.

The rows of S. A quadratic row is `sqrt(D) J`; a satisfied or linear one is zero. An
elliptic contact in the cone's middle zone has, with `N = mu jaref_0`,
`u_j = friction_j jaref_j`, `T = |u|`, `z_0 = mu J_0`, `z_j = friction_j J_j`,
`p = sum_j (u_j / T) z_j` and `dm = D_0 / (mu^2 (1 + mu^2))`, MJWarp's curvature
(`_update_gradient_JTCJ_dense`)

    dm [ (z_0 - mu p)(z_0 - mu p)^T + (mu^2 - mu N / T)(Z^T Z - p p^T) ],

and at condim 3 `Z^T Z - p p^T = s s^T` with `s = sum_j e_j z_j`, `e = (-u_2, u_1) / T`:
two rows, `sqrt(dm) (z_0 - mu p)` and `sqrt(dm (mu^2 - mu N / T)) s`, which the thread of
the contact's normal row stages. `mu^2 - mu N / T > 0` in the middle zone, where
`N < mu T`.

What it takes: Newton (not CG), a dense model of nv <= 16, no sleeping, elliptic
contacts of condim 3 at most, a world's rows within 48 KB of shared memory; under the
prox solver, its exact rows (not `prox_soft`) at impratio 1. `FusedNewton` refuses the
rest. It uses no Warp shared tiles: their element store carries a block barrier, and
their tile matmul and Cholesky are slow at these shapes (a 12 x 384 x 12 product took about
30k cycles on one warp).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp


from lodos.cd.solver import nlsp
from lodos.cd.solver import types as solver_types
from lodos.mjcf import types as mjcf_types
from lodos.utils import math as lmath

wp.set_module_options({"enable_backward": False})

# MJWarp's constraint states and types, a contact group's type flag, its float bounds and its line search's noise floor.
QUADRATIC = wp.constant(int(solver_types.ConstraintState.QUADRATIC.value))
CONE = wp.constant(int(solver_types.ConstraintState.CONE.value))
CONTACT_ELLIPTIC = wp.constant(int(solver_types.ConstraintType.CONTACT_ELLIPTIC.value))
CONTACT_FRICTIONLESS = wp.constant(int(solver_types.ConstraintType.CONTACT_FRICTIONLESS.value))
GROUP = wp.constant(int(mjcf_types.ContactType.GROUP))
MAXVAL = wp.constant(float(mjcf_types.MAXVAL))
MINVAL = wp.constant(float(mjcf_types.MINVAL))
ALPHA_NOISE_EPS = wp.constant(float(nlsp._ALPHA_NOISE_EPS))

# A row's kind, as the line search, the constraint update and the prox update dispatch on it.
EQUALITY = wp.constant(0)
FRICTION = wp.constant(1)
ONESIDED = wp.constant(2)  # limits, frictionless and pyramidal contacts
ELLIPTIC0 = wp.constant(3)  # an elliptic contact's normal row: it carries the contact's cone
ELLIPTIC1 = wp.constant(4)  # its tangent rows
CONTACT1 = wp.constant(5)  # a frictionless contact under the prox solver: one-sided, and a prox anchor
IGNORED = wp.constant(6)  # an elliptic contact `njmax` cut short: MJWarp skips it

LANES = 32  # a warp
# Warps per world, by default: 1 for at most 128 rows a world, 2 above. Fewer rows leave the block's barriers and
# sums the cost, and a smaller block lets more worlds share a multiprocessor.
WARPS = None
NV_MAX = 16  # every thread holds dof vectors of nv floats in registers; thread 0's factorization is unrolled
SHARED_MAX = 48 * 1024  # static shared memory a block may take
NPROF = 11  # profiled sections: setup, jv, p0 + lo_in, bracketing, step, rows, reduce, all, mv, prox, Cholesky


@wp.func_native(snippet="__syncthreads();")
def _syncthreads():
    """
    The block's barrier, CUDA's `__syncthreads()`.

    Called by `newton_kernel`'s `kernel`, `bsum5`, `bsum9` and 3 others.
    """
    pass


@wp.func_native(snippet="return __shfl_up_sync(0xffffffff, v, d);")
def _shfl_up_i(v: int, d: int) -> int:
    """
    `v` from the lane `d` below, across the full warp (`__shfl_up_sync`); a lane below `d` keeps its own.

    Called by `newton_kernel`'s `bscan`. In: v int; d int, a lane distance. Out: int.
    """
    ...


@wp.func_native(snippet="return __shfl_sync(0xffffffff, v, src);")
def _shfl_idx_i(v: int, src: int) -> int:
    """
    Lane `src`'s `v`, in every lane of the warp (`__shfl_sync`).

    Called by `newton_kernel`'s `bscan`. In: v int; src int, a lane. Out: int.
    """
    ...


@wp.func_native(snippet="return __shfl_xor_sync(0xffffffff, v, m);")
def _shfl_xor(v: float, m: int) -> float:
    """
    `v` from lane `lane ^ m`, across the full warp (`__shfl_xor_sync`): a butterfly step of a warp's sum or maximum.

    Called by `_wsum33`, `_wmax4`, `newton_kernel`'s `bsum5` and one other. In: v float; m int, the lane mask.
    Out: float.
    """
    ...


@wp.func_native(snippet="return (long long)clock64();")
def _clock() -> wp.int64:
    """
    The multiprocessor's cycle counter, `clock64()`.

    Called by `newton_kernel`'s `kernel`, under `profile`. Out: int64.
    """
    ...


@wp.func
def _wsum33(a: wp.vec3, b: wp.vec3, c: wp.vec3):
    """
    The sums of three vectors over the warp, in every lane; their shuffles interleaved, step by step.

    Called by `newton_kernel`'s `bsum9`. In: a, b, c wp.vec3, the lane's. Out: three wp.vec3.
    """
    for k in range(5):  # k: int, a butterfly step
        m = 16 >> k  # int: the lane mask, 16 down to 1
        a = wp.vec3(a[0] + _shfl_xor(a[0], m), a[1] + _shfl_xor(a[1], m), a[2] + _shfl_xor(a[2], m))
        b = wp.vec3(b[0] + _shfl_xor(b[0], m), b[1] + _shfl_xor(b[1], m), b[2] + _shfl_xor(b[2], m))
        c = wp.vec3(c[0] + _shfl_xor(c[0], m), c[1] + _shfl_xor(c[1], m), c[2] + _shfl_xor(c[2], m))
    return a, b, c


@wp.func
def _wmax4(v: wp.vec4) -> wp.vec4:
    """
    The componentwise maximum of `v` over the warp, in every lane.

    Called by `newton_kernel`'s `bmax4`. In: v wp.vec4, the lane's. Out: wp.vec4.
    """
    for k in range(5):  # k: int, a butterfly step
        m = 16 >> k  # int: the lane mask
        v = wp.vec4(wp.max(v[0], _shfl_xor(v[0], m)), wp.max(v[1], _shfl_xor(v[1], m)),
                    wp.max(v[2], _shfl_xor(v[2], m)), wp.max(v[3], _shfl_xor(v[3], m)))
    return v


@wp.func
def _pt_alpha_zero(kind: int, jaref: float, jv: float, D: float, fl: float, q0: wp.vec3, q1: wp.vec3, q2: wp.vec3,
                   mu: float) -> wp.vec3:
    """
    A row's (cost, slope, curvature) at alpha 0: MJWarp's _compute_efc_eval_pt_alpha_zero_*.

    Called by `newton_kernel`'s `kernel`, in the line search. In: kind int, the row's kind (`EQUALITY` .. `IGNORED`);
    jaref, jv, D, fl float, its `J qacc - aref`, `J search`, D and friction loss; q0, q1, q2 wp.vec3, an elliptic
    normal row's quadratics (`row_inputs`); mu float, its contact's friction. Out: wp.vec3, (cost, slope, curvature).
    """
    if kind == ONESIDED or kind == CONTACT1:
        if jaref < 0.0:
            return nlsp._eval_pt_direct_alpha_zero(jaref, jv, D)
        return wp.vec3(0.0)
    if kind == ELLIPTIC0:
        return nlsp._eval_elliptic_alpha_zero(mu, q0, q1, q2)
    if kind == FRICTION:
        rf = lmath.safe_div(fl, D)  # float: the edge of the row's quadratic zone, |jaref| < rf
        return nlsp._eval_frictionloss_pt(jaref, fl, rf, jv, D)
    return wp.vec3(0.0)  # equality rows are in the Gauss quadratic; tangent rows in their normal row


@wp.func
def _pt(kind: int, alpha: float, jaref: float, jv: float, D: float, fl: float, q0: wp.vec3, q1: wp.vec3,
        q2: wp.vec3, mu: float) -> wp.vec3:
    """
    A row's shifted point at alpha: MJWarp's _compute_efc_eval_pt_*.

    Called by `newton_kernel`'s `kernel`, in the line search. In: alpha float, the step; the rest as `_pt_alpha_zero`'s.
    Out: wp.vec3, (cost, slope, curvature), the cost less the row's at alpha 0.
    """
    if kind == ONESIDED or kind == CONTACT1:
        x = jaref + alpha * jv  # float: the row's jaref at alpha
        quad0 = nlsp._eval_pt_direct_cost_alpha_zero(jaref, D)  # float: the quadratic's cost at alpha 0
        cost0 = wp.where(jaref < 0.0, quad0, 0.0)  # float: the row's cost at alpha 0
        offset = quad0 - cost0  # float
        if x < 0.0:
            return nlsp._eval_pt_direct_shifted(jaref, jv, D, alpha, offset)
        return wp.vec3(-cost0, 0.0, 0.0)
    if kind == ELLIPTIC0:
        # float, float, float, int: the cost, tangential norm, cone residual and zone at alpha 0
        cost0, T0, r0, state0 = nlsp._eval_elliptic_reference(mu, q0, q1, q2)
        return nlsp._eval_elliptic_shifted(mu, q0, q1, q2, alpha, cost0, T0, r0, state0)
    if kind == FRICTION:
        x = jaref + alpha * jv
        rf = lmath.safe_div(fl, D)  # float: the edge of the row's quadratic zone
        return nlsp._shift_cost(nlsp._eval_frictionloss_pt(x, fl, rf, jv, D), nlsp._eval_frictionloss_cost(jaref, fl, rf, D))
    return wp.vec3(0.0)


@wp.func
def _pt3(kind: int, lo_alpha: float, hi_alpha: float, mid_alpha: float, jaref: float, jv: float, D: float, fl: float,
         q0: wp.vec3, q1: wp.vec3, q2: wp.vec3, mu: float):
    """
    A row's shifted points at three alphas: MJWarp's _compute_efc_eval_pt_3alphas_*.

    Called by `newton_kernel`'s `kernel`, in the line search's bracketing. In: lo_alpha, hi_alpha, mid_alpha float;
    the rest as `_pt_alpha_zero`'s. Out: three wp.vec3, as `_pt`'s.
    """
    if kind == ONESIDED or kind == CONTACT1:
        x_lo = jaref + lo_alpha * jv  # float
        x_hi = jaref + hi_alpha * jv  # float
        x_mid = jaref + mid_alpha * jv  # float
        quad0 = nlsp._eval_pt_direct_cost_alpha_zero(jaref, D)  # float: the quadratic's cost at alpha 0
        cost0 = wp.where(jaref < 0.0, quad0, 0.0)  # float: the row's cost at alpha 0
        offset = quad0 - cost0  # float
        # wp.vec3 x 3: the quadratic's points
        pt_lo, pt_hi, pt_mid = nlsp._eval_pt_direct_shifted_3alphas(jaref, jv, D, lo_alpha, hi_alpha, mid_alpha, offset)
        inactive = wp.vec3(-cost0, 0.0, 0.0)  # wp.vec3: a satisfied row's point
        return (wp.where(x_lo < 0.0, pt_lo, inactive), wp.where(x_hi < 0.0, pt_hi, inactive),
                wp.where(x_mid < 0.0, pt_mid, inactive))
    if kind == ELLIPTIC0:
        cost0, T0, r0, state0 = nlsp._eval_elliptic_reference(mu, q0, q1, q2)  # float, float, float, int: as in `_pt`
        return (nlsp._eval_elliptic_shifted(mu, q0, q1, q2, lo_alpha, cost0, T0, r0, state0),
                nlsp._eval_elliptic_shifted(mu, q0, q1, q2, hi_alpha, cost0, T0, r0, state0),
                nlsp._eval_elliptic_shifted(mu, q0, q1, q2, mid_alpha, cost0, T0, r0, state0))
    if kind == FRICTION:
        x_lo = jaref + lo_alpha * jv
        x_hi = jaref + hi_alpha * jv
        x_mid = jaref + mid_alpha * jv
        rf = lmath.safe_div(fl, D)  # float: the edge of the row's quadratic zone
        cost0 = nlsp._eval_frictionloss_cost(jaref, fl, rf, D)
        lo, hi, mid = nlsp._eval_frictionloss_pt_3alphas(x_lo, x_hi, x_mid, fl, rf, jv, D)  # wp.vec3 x 3
        return (nlsp._shift_cost(lo, cost0), nlsp._shift_cost(hi, cost0), nlsp._shift_cost(mid, cost0))
    return (wp.vec3(0.0), wp.vec3(0.0), wp.vec3(0.0))


def _shared_accessors(nfloat: int, nint: int):
    """
    Per-lane loads and stores on a block's static shared memory: `shf(i, v, store)` and `shi(i, v, store)` store `v`
    at `i` when `store`, and return the element. Warp's own shared tiles put a barrier in every element store. (Each
    is a decorated definition: Warp takes a native stub's return type only from one.)

    Called by `newton_kernel`. In: nfloat, nint int, the floats and ints of a world's layout (`shared_layout`).
    Out: shf, shi, the two native functions.
    """
    fsnip = f"__shared__ float buf[{max(nfloat, 1)}];\nif (store) buf[i] = v;\nreturn buf[i];"  # str: shf's CUDA source
    isnip = f"__shared__ int buf[{max(nint, 1)}];\nif (store) buf[i] = v;\nreturn buf[i];"  # str: shi's

    @wp.func_native(snippet=fsnip)
    def shf(i: int, v: float, store: int) -> float:
        """
        Float `i` of the block's shared memory, after storing `v` there when `store` is nonzero.

        Called by `newton_kernel`'s `kernel`, `jrow`, `jget` and 8 others. In: i, store int; v float. Out: float.
        """
        ...

    @wp.func_native(snippet=isnip)
    def shi(i: int, v: int, store: int) -> int:
        """
        Int `i` of the block's shared memory, after storing `v` there when `store` is nonzero.

        Called by `newton_kernel`'s `kind_of`, `a0_of` and `kernel`. In: i, v, store int. Out: int.
        """
        ...

    return shf, shi


def shared_layout(nv: int, njmax: int, elliptic: bool, prox: bool, warps: int, jshared: bool = True):
    """
    A world's layout in shared memory: the rows rounded up to whole chunks, J's row stride, float offsets,
    the floats, the ints, and the bytes it takes. Per row: J (unless `jshared` is false: the kernel then reads
    J from global memory, where L1 holds it); `jaref`, `jv`, D; P, the friction loss on a
    friction row and `aref0` on a contact row under prox; elliptic: the line-search quadratic (3 floats) and X, a
    contact's `mu` on its normal row and its frictions on its tangent rows; prox: `aref`, the anchor `z`, the
    velocity of a tangent row and `Gamma` on the normal row. An elliptic contact's rows are consecutive
    (`make_constraint`), so a row keeps only its contact's normal row, packed with its kind into one int.

    Called by `newton_kernel` and `FusedNewton.__init__`. In: nv, njmax, warps int; elliptic, prox, jshared bool.
    Out: nj int, nvs int, o dict of str to int, the floats int, the ints int (nj), the bytes int.
    """
    nj = (njmax + LANES - 1) // LANES * LANES  # int: njmax rounded up to whole warps
    nvs = nv if nv % 2 else nv + 1  # int: J's row stride: odd, so the lanes' rows fall in distinct banks
    o = dict(J=0)  # dict of str to int: each array's offset, in floats
    o["jaref"] = o["J"] + (nj * nvs if jshared else 0)
    o["jv"] = o["jaref"] + nj
    o["D"] = o["jv"] + nj
    o["P"] = o["D"] + nj
    o["M"] = o["P"] + nj
    o["dof"] = o["M"] + nv * nv  # qacc, Ma, qfrc_smooth, H^-1 grad: nv each
    o["H"] = o["dof"] + 4 * nv  # the Hessian's packed lower triangle
    o["S"] = o["H"] + nv * (nv + 1) // 2  # the Hessian's nonzero rows, a chunk at a time, packed
    o["red"] = o["S"] + (warps * LANES + 1) * nvs  # the block sums' warp partials: two buffers of 16 a warp
    end = o["red"] + 2 * warps * 16  # int: the floats so far
    if elliptic:
        o["quad"] = end
        o["X"] = o["quad"] + 3 * nj
        end = o["X"] + nj
    if prox:
        o["aref"] = end
        o["z"] = o["aref"] + nj
        o["vel"] = o["z"] + nj  # a tangent row's velocity; Gamma on a normal row
        end = o["vel"] + nj
    return nj, nvs, o, end, nj, 4 * (end + nj)


_KERNELS = {}  # newton_kernel's kernels, by its arguments


def newton_kernel(nv: int, njmax: int, elliptic: bool, warmstart: bool, prox: bool, profile: bool = False,
                  warps: int = 4, jshared: bool = True, groups: bool = False):
    """
    The fused Newton kernel for `nv` dofs, `njmax` rows a world, the model's cone and the prox loop or not, launched
    `(nworld, warps x LANES)` in blocks of `warps x LANES`, one world a block. `profile` compiles in clock64 counters
    per section (`NPROF`), written to `prof` by thread 0. `jshared` keeps J in shared memory; otherwise the kernel reads
    it from `efc.J`, which leaves room for more rows. `groups`: some contacts may be contact groups (`GROUP`), whose
    De Saxce term under the prox loop is their members' (`nlsp._group_gamma`).

    Called by `FusedNewton.__init__`. In: nv, njmax, warps int; elliptic, warmstart, prox, profile, jshared, groups
    bool. Out: the Warp kernel, built once per argument set (`_KERNELS`).
    """
    key = (nv, njmax, elliptic, warmstart, prox, profile, warps, jshared, groups)  # tuple: the cache key
    if key in _KERNELS:
        return _KERNELS[key]
    NV = nv  # int: the dofs
    NP = nv * (nv + 1) // 2  # int: the entries of the Hessian's packed lower triangle
    IS_ELLIPTIC = elliptic  # bool
    WARMSTART = warmstart  # bool
    PROX = prox  # bool
    PROFILE = profile  # bool
    GROUPS = prox and groups  # bool: without the prox loop a group's rows are a contact's: nothing to add
    W = warps  # int: warps a block
    THREADS = warps * LANES  # int: threads a block
    JSH = jshared  # bool
    # int, int, dict, int, int: `shared_layout`'s rows, J's row stride, offsets, floats and ints
    NJ, NVS, o, nfloat, nint, _ = shared_layout(nv, njmax, elliptic, prox, warps, jshared)
    # int: the float offsets of J, jaref, jv, D, P and M
    OJ, OJA, OJV, OD, OP, OM = o["J"], o["jaref"], o["jv"], o["D"], o["P"], o["M"]
    OQA, OMA, OQS, OMG = o["dof"], o["dof"] + nv, o["dof"] + 2 * nv, o["dof"] + 3 * nv  # qacc, Ma, qfrc_smooth, H^-1 g
    OH, OS = o["H"], o["S"]  # int, int: the Hessian's and S's offsets
    OQ, OX = o.get("quad", 0), o.get("X", 0)  # int, int: the quadratics' and X's offsets, 0 if absent
    OAR, OZ, OV = o.get("aref", 0), o.get("z", 0), o.get("vel", 0)  # int: aref's, z's and vel's offsets, 0 if absent
    OG = OV  # int: Gamma: on the normal row, whose own velocity the prox update does not read
    ORED = o["red"]  # int: the block sums' scratch offset
    KO = (NP + THREADS - 1) // THREADS  # int: Hessian entries a thread owns
    shf, shi = _shared_accessors(nfloat, nint)  # the float and int accessors of shared memory
    vecv = wp.types.vector(length=NV, dtype=float)  # Warp vector type, NV floats: a dof vector
    veco = wp.types.vector(length=KO, dtype=float)  # Warp vector type, KO floats: the Hessian entries a thread owns
    veci = wp.types.vector(length=KO, dtype=int)  # Warp vector type, KO ints: their rows or columns

    @wp.func
    def jrow(J: wp.array3d(dtype=float), worldid: int, r: int):
        """
        Row `r` of the world's J: from shared memory under `jshared`, else from `efc.J`.

        Called by `kernel`. In: J (nworld, njmax_pad, nv_pad) float32, `efc.J`; worldid, r int. Out: vecv.
        """
        v = vecv()  # vecv: the row
        for i in range(NV):  # i: int, a dof
            if wp.static(JSH):
                v[i] = shf(OJ + r * NVS + i, 0.0, 0)
            else:
                v[i] = J[worldid, r, i]
        return v

    @wp.func
    def jget(J: wp.array3d(dtype=float), worldid: int, r: int, i: int) -> float:
        """
        Entry `(r, i)` of the world's J: from shared memory under `jshared`, else from `efc.J`.

        Called by `kernel`. In: J as `jrow`'s; worldid, r, i int. Out: float.
        """
        if wp.static(JSH):
            return shf(OJ + r * NVS + i, 0.0, 0)
        return J[worldid, r, i]

    @wp.func
    def kind_of(r: int) -> int:
        """
        Row `r`'s kind (`EQUALITY` .. `IGNORED`), the low three bits of its int in shared memory.

        Called by `row_inputs`, `eval_row` and `kernel`. In: r int. Out: int.
        """
        return shi(r, 0, 0) & 7

    @wp.func
    def a0_of(r: int) -> int:
        """
        The normal row of row `r`'s elliptic contact, the upper bits of its int in shared memory; `r` itself on any
        other row.

        Called by `eval_row`. In: r int. Out: int.
        """
        return shi(r, 0, 0) >> 3

    @wp.func
    def bsum5(a: wp.vec2, b: wp.vec3, lane: int, warp: int, buf: int):
        """
        The block's sums of a and b, in every thread: the warps' shuffles, then their partials in shared memory.

        Called by `kernel`, for the line search's point at alpha 0. In: a wp.vec2, b wp.vec3, the thread's terms; lane,
        warp int; buf int, the scratch buffer, 0 or 1. Out: wp.vec2, wp.vec3.
        """
        for k in range(5):  # k: int, a butterfly step
            m = 16 >> k  # int: the lane mask
            a = wp.vec2(a[0] + _shfl_xor(a[0], m), a[1] + _shfl_xor(a[1], m))
            b = wp.vec3(b[0] + _shfl_xor(b[0], m), b[1] + _shfl_xor(b[1], m), b[2] + _shfl_xor(b[2], m))
        if wp.static(W > 1):
            base = ORED + buf * W * 16 + warp * 16  # int: this warp's partials
            if lane == 0:
                shf(base, a[0], 1)
                shf(base + 1, a[1], 1)
                shf(base + 2, b[0], 1)
                shf(base + 3, b[1], 1)
                shf(base + 4, b[2], 1)
            _syncthreads()
            a = wp.vec2(0.0)
            b = wp.vec3(0.0)
            for w in range(W):  # w: int, a warp
                base = ORED + buf * W * 16 + w * 16
                a += wp.vec2(shf(base, 0.0, 0), shf(base + 1, 0.0, 0))
                b += wp.vec3(shf(base + 2, 0.0, 0), shf(base + 3, 0.0, 0), shf(base + 4, 0.0, 0))
        return a, b

    @wp.func
    def bsum9(a: wp.vec3, b: wp.vec3, c: wp.vec3, lane: int, warp: int, buf: int):
        """
        The block's sums of a, b and c, in every thread.

        Called by `kernel`, for the line search's points. In: a, b, c wp.vec3, the thread's terms; lane, warp, buf as
        `bsum5`'s. Out: three wp.vec3.
        """
        a, b, c = _wsum33(a, b, c)
        if wp.static(W > 1):
            base = ORED + buf * W * 16 + warp * 16  # int: this warp's partials
            if lane == 0:
                for k in range(3):  # k: int, a component
                    shf(base + k, a[k], 1)
                    shf(base + 3 + k, b[k], 1)
                    shf(base + 6 + k, c[k], 1)
            _syncthreads()
            a = wp.vec3(0.0)
            b = wp.vec3(0.0)
            c = wp.vec3(0.0)
            for w in range(W):  # w: int, a warp
                base = ORED + buf * W * 16 + w * 16
                a += wp.vec3(shf(base, 0.0, 0), shf(base + 1, 0.0, 0), shf(base + 2, 0.0, 0))
                b += wp.vec3(shf(base + 3, 0.0, 0), shf(base + 4, 0.0, 0), shf(base + 5, 0.0, 0))
                c += wp.vec3(shf(base + 6, 0.0, 0), shf(base + 7, 0.0, 0), shf(base + 8, 0.0, 0))
        return a, b, c

    @wp.func
    def bsumv(v: Any, lane: int, warp: int, buf: int):
        """
        The block's sum of a dof vector, in every thread.

        Called by `kernel`, for `J^T f`. In: v vecv, the thread's terms; lane, warp, buf as `bsum5`'s. Out: vecv.
        """
        for k in range(5):  # k: int, a butterfly step
            m = 16 >> k  # int: the lane mask
            for i in range(NV):  # i: int, a dof
                v[i] += _shfl_xor(v[i], m)
        if wp.static(W > 1):
            base = ORED + buf * W * 16 + warp * 16  # int: this warp's partials
            if lane == 0:
                for i in range(NV):
                    shf(base + i, v[i], 1)
            _syncthreads()
            v = vecv()
            for w in range(W):  # w: int, a warp
                base = ORED + buf * W * 16 + w * 16
                for i in range(NV):
                    v[i] += shf(base + i, 0.0, 0)
        return v

    @wp.func
    def bmax4(v: wp.vec4, lane: int, warp: int, buf: int):
        """
        The block's componentwise maximum of v, in every thread.

        Called by `kernel`, for the prox round's test. In: v wp.vec4, the thread's; lane, warp, buf as `bsum5`'s.
        Out: wp.vec4.
        """
        v = _wmax4(v)
        if wp.static(W > 1):
            base = ORED + buf * W * 16 + warp * 16  # int: this warp's partials
            if lane == 0:
                for k in range(4):  # k: int, a component
                    shf(base + k, v[k], 1)
            _syncthreads()
            for w in range(W):  # w: int, a warp
                base = ORED + buf * W * 16 + w * 16
                v = wp.vec4(wp.max(v[0], shf(base, 0.0, 0)), wp.max(v[1], shf(base + 1, 0.0, 0)),
                            wp.max(v[2], shf(base + 2, 0.0, 0)), wp.max(v[3], shf(base + 3, 0.0, 0)))
        return v

    @wp.func
    def accumulate(hown: Any, own_i: Any, own_j: Any, tid: int, nstaged: int):
        """
        The staged rows of S into the Hessian entries this thread owns.

        Called by `kernel`. In: hown veco, the owned entries' sums so far; own_i, own_j veci, their rows and columns;
        tid int, the thread; nstaged int, the rows staged. Out: veco, hown added to.
        """
        for mo in range(KO):  # mo: int, an owned entry
            if tid + THREADS * mo < NP:
                ii = own_i[mo]  # int: its row
                jj = own_j[mo]  # int: its column
                acc = hown[mo]  # float: its sum
                for t in range(nstaged):  # t: int, a staged row
                    acc += shf(OS + t * NVS + ii, 0.0, 0) * shf(OS + t * NVS + jj, 0.0, 0)
                hown[mo] = acc
        return hown

    @wp.func
    def bscan(n: int, lane: int, warp: int, buf: int):
        """
        The block's exclusive prefix sum of n at this thread, and its total.

        Called by `kernel`, to pack the rows of S. In: n int, the thread's count; lane, warp, buf as `bsum5`'s.
        Out: int, int.
        """
        x = n  # int: the inclusive prefix sum at this lane, once scanned
        for k in range(5):  # k: int, a scan step
            d = 1 << k  # int: the lane distance
            y = _shfl_up_i(x, d)  # int: the sum d lanes below
            if lane >= d:
                x += y
        excl = x - n  # int: the exclusive sum in the warp
        off = int(0)  # int: the lower warps' total
        tot = _shfl_idx_i(x, LANES - 1)  # int: the warp's total, then the block's
        if wp.static(W > 1):
            base = ORED + buf * W * 16  # int: the warps' totals
            if lane == LANES - 1:
                shf(base + warp * 16, float(x), 1)
            _syncthreads()
            tot = int(0)
            for w in range(W):  # w: int, a warp
                cw = int(shf(base + w * 16, 0.0, 0))  # int: warp w's total
                if w < warp:
                    off += cw
                tot += cw
        return excl + off, tot

    @wp.func
    def row_inputs(r: int):
        """
        A row's kind, jaref, jv, D and friction loss, and for an elliptic normal row its quadratics and mu.

        Called by `kernel`, in the line search. In: r int, a row. Out: int, float, float, float, float, wp.vec3,
        wp.vec3, wp.vec3, float, as `_pt`'s arguments.
        """
        kind = kind_of(r)  # int
        q0 = wp.vec3(0.0)  # wp.vec3: the contact's quadratic cost coefficients
        q1 = wp.vec3(0.0)  # wp.vec3: (u0, v0, uu)
        q2 = wp.vec3(0.0)  # wp.vec3: (uv, vv, dm)
        mu = float(0.0)  # float: the contact's mu
        if wp.static(IS_ELLIPTIC):
            if kind == ELLIPTIC0:
                q0 = wp.vec3(shf(OQ + 3 * r, 0.0, 0), shf(OQ + 3 * r + 1, 0.0, 0), shf(OQ + 3 * r + 2, 0.0, 0))
                q1 = wp.vec3(shf(OQ + 3 * r + 3, 0.0, 0), shf(OQ + 3 * r + 4, 0.0, 0), shf(OQ + 3 * r + 5, 0.0, 0))
                q2 = wp.vec3(shf(OQ + 3 * r + 6, 0.0, 0), shf(OQ + 3 * r + 7, 0.0, 0), shf(OQ + 3 * r + 8, 0.0, 0))
                mu = shf(OX + r, 0.0, 0)
        return (kind, shf(OJA + r, 0.0, 0), shf(OJV + r, 0.0, 0), shf(OD + r, 0.0, 0), shf(OP + r, 0.0, 0),
                q0, q1, q2, mu)

    @wp.func
    def eval_row(r: int):
        """
        MJWarp's _eval_constraint on row r: (force, state), and its contact's (normal row, mu, u1, u2, TT, f1, f2).

        Called by `kernel`, in the constraint update and after an unconverged prox loop. In: r int, a row. Out: force
        float, state int, efcid0 int, then jaref0, D0, mu, u1, u2, TT, f1 and f2 float.
        """
        kind = kind_of(r)  # int
        is_elliptic = kind == ELLIPTIC0 or kind == ELLIPTIC1  # bool
        efcid0 = int(-1)  # int: the contact's normal row
        jaref0 = float(0.0)  # float: its jaref
        D0 = float(0.0)  # float: its D
        mu = float(0.0)  # float: the contact's mu
        u1 = float(0.0)  # float: friction_1 jaref_1
        u2 = float(0.0)  # float: friction_2 jaref_2
        f1 = float(0.0)  # float: the first tangent's friction
        f2 = float(0.0)  # float: the second's
        ufrictionj = float(0.0)  # float: u_j friction_j on tangent row j
        TT = float(0.0)  # float: u1^2 + u2^2
        if wp.static(IS_ELLIPTIC):
            if is_elliptic:
                efcid0 = a0_of(r)
                mu = shf(OX + efcid0, 0.0, 0)
                f1 = shf(OX + efcid0 + 1, 0.0, 0)
                f2 = shf(OX + efcid0 + 2, 0.0, 0)
                jaref0 = shf(OJA + efcid0, 0.0, 0)
                D0 = shf(OD + efcid0, 0.0, 0)
                u1 = shf(OJA + efcid0 + 1, 0.0, 0) * f1
                u2 = shf(OJA + efcid0 + 2, 0.0, 0) * f2
                TT = u1 * u1 + u2 * u2
                if r == efcid0 + 1:
                    ufrictionj = u1 * f1
                elif r == efcid0 + 2:
                    ufrictionj = u2 * f2
        res = wp.vec3(0.0)  # wp.vec3: (force, state, cost)
        if kind != IGNORED:
            res = nlsp._eval_constraint(
                kind == EQUALITY, kind == FRICTION, is_elliptic, shf(OJA + r, 0.0, 0), shf(OD + r, 0.0, 0),
                shf(OP + r, 0.0, 0), r, efcid0, jaref0, D0, mu, ufrictionj, TT,
            )
        return res[0], int(res[1]), efcid0, jaref0, D0, mu, u1, u2, TT, f1, f2

    # Loops over dofs and over the Hessian's entries unroll (their indices constants: registers, fixed shared
    # addresses); the line search's loop does not (its bound is an argument).
    @wp.kernel(module="unique", enable_backward=False, module_options={"max_unroll": NV_MAX * (NV_MAX + 1) // 2})
    def kernel(
        # Model:
        opt_tolerance: wp.array(dtype=float),
        opt_ls_tolerance: wp.array(dtype=float),
        opt_iterations: int,
        opt_ls_iterations: int,
        stat_meaninertia: wp.array(dtype=float),
        opt_timestep: wp.array(dtype=float),
        opt_prox_tol_z: wp.array(dtype=float),
        opt_prox_tol_sigma: wp.array(dtype=float),
        opt_prox_sigma_relax: wp.array(dtype=float),
        nround: int,  # prox rounds, outer x inner; 1 without prox
        inner: int,  # prox rounds a De Saxce step
        desaxce: int,
        z_only: int,  # Gamma is not stepped within the step: z alone is tested, every round
        M_entry: wp.array2d(dtype=int),  # (nv, nv) CSR entry of M[i, j], -1 where it is zero
        # Data in:
        ne_in: wp.array(dtype=int),
        nf_in: wp.array(dtype=int),
        nefc_in: wp.array(dtype=int),
        M_in: wp.array2d(dtype=float),
        qacc_warmstart_in: wp.array2d(dtype=float),
        qacc_smooth_in: wp.array2d(dtype=float),
        qfrc_smooth_in: wp.array2d(dtype=float),
        contact_friction_in: wp.array(dtype=mjcf_types.vec5),
        contact_efc_address_in: wp.array2d(dtype=int),
        contact_impratio_invsqrt_in: wp.array(dtype=float),
        efc_type_in: wp.array2d(dtype=int),
        efc_id_in: wp.array2d(dtype=int),
        efc_J_in: wp.array3d(dtype=float),
        efc_D_in: wp.array2d(dtype=float),
        efc_aref_in: wp.array2d(dtype=float),
        efc_frictionloss_in: wp.array2d(dtype=float),
        efc_vel_in: wp.array2d(dtype=float),
        prox_aref0_in: wp.array2d(dtype=float),
        prox_gamma_in: wp.array(dtype=float),
        njmax_in: int,
        # A GROUP contact's members and the trees that move them, for its De Saxce term (read under GROUPS):
        body_rootid: wp.array(dtype=int),
        body_weldid: wp.array(dtype=int),
        body_dofnum: wp.array(dtype=int),
        body_dofadr: wp.array(dtype=int),
        dof_parentid: wp.array(dtype=int),
        geom_bodyid: wp.array(dtype=int),
        qvel_in: wp.array2d(dtype=float),
        cdof_in: wp.array2d(dtype=wp.spatial_vector),
        subtree_com_in: wp.array2d(dtype=wp.vec3),
        contact_type_in: wp.array(dtype=int),
        contact_geom_in: wp.array(dtype=wp.vec2i),
        contact_group_adr_in: wp.array(dtype=int),
        contact_group_num_in: wp.array(dtype=int),
        group_member_pos_in: wp.array(dtype=wp.vec3),
        group_member_frame_in: wp.array(dtype=wp.mat33),
        group_member_weight_in: wp.array(dtype=float),
        group_member_slip_in: wp.array(dtype=float),
        # Data out:
        qacc_out: wp.array2d(dtype=float),
        qfrc_constraint_out: wp.array2d(dtype=float),
        efc_force_out: wp.array2d(dtype=float),
        efc_state_out: wp.array2d(dtype=int),
        solver_niter_out: wp.array(dtype=int),
        prox_z: wp.array2d(dtype=float),  # in: the anchors _prox_init set; out: the last round's, for the cache
        prof: wp.array2d(dtype=wp.int64),
    ):
        """
        The fused solve of one world, a block of `warps x LANES` threads: its rows into shared memory, then in each prox
        round (one without prox) MJWarp's Newton iterations -- the line search, the constraint update, the gradient, the
        Hessian and its Cholesky solve, the convergence test -- and the round's prox update and test.

        Launch: dim=(nworld, warps x LANES), block_dim=warps x LANES, a block per world, by `FusedNewton.solve`.
        Reads: the opt_* arrays and stat_meaninertia (1 or nworld,) float32; M_entry (nv, nv) int32; ne_in, nf_in and
        nefc_in (nworld,) int32; M_in (nworld, nC) float32; qacc_warmstart_in, qacc_smooth_in, qfrc_smooth_in (nworld,
        nv) float32; contact_friction_in (naconmax,) vec5, contact_efc_address_in (naconmax, nmaxpyramid) int32,
        contact_impratio_invsqrt_in (naconmax,) float32; efc_type_in, efc_id_in (nworld, njmax) int32; efc_J_in (nworld,
        njmax_pad, nv_pad) float32; efc_D_in (nworld, njmax_pad) float32; efc_aref_in, efc_frictionloss_in, efc_vel_in
        and prox_aref0_in (nworld, njmax) float32; prox_gamma_in (naconmax,) float32 (prox_* (1, 1) and (1,) without
        prox); under GROUPS body_rootid, body_weldid, body_dofnum, body_dofadr (nbody,), dof_parentid (nv,), geom_bodyid
        (ngeom,) int32, qvel_in (nworld, nv) float32, cdof_in (nworld, nv) wp.spatial_vector, subtree_com_in (nworld,
        nbody) wp.vec3, the contacts' type, geom, group_adr and group_num (naconmax,), the members' pos, frame, weight
        and slip (ngroupmember,).
        Writes: qacc_out (read back under GROUPS) and qfrc_constraint_out (nworld, nv) float32; efc_force_out (nworld,
        njmax) float32; efc_state_out (nworld, njmax_pad) int32; solver_niter_out (nworld,) int32; prox_z (nworld,
        njmax) float32, read and written; prof (nworld, NPROF) int64 under `profile`.
        """
        worldid, tid = wp.tid()  # int, int: the world, the thread in its block
        lane = tid % LANES  # int: the thread's lane
        warp = tid // LANES  # int: its warp
        rbuf = int(0)  # int: the block sums' scratch buffer: they alternate, one barrier each
        tstart = wp.int64(0)  # int64: the clock at the start, under profile
        tclk = wp.int64(0)  # int64: the clock at the last section's end
        pc0 = wp.int64(0)  # int64: the setup's cycles
        pc1 = wp.int64(0)  # int64: J search's and the cone quadratics'
        pc2 = wp.int64(0)  # int64: p0's and lo_in's
        pc3 = wp.int64(0)  # int64: the bracketing's
        pc4 = wp.int64(0)  # int64: the step's
        pc5 = wp.int64(0)  # int64: the rows': constraint update, S, H's sums
        pc6 = wp.int64(0)  # int64: J^T f's, the gradient's and H's assembly
        pc8 = wp.int64(0)  # int64: M search's
        pc9 = wp.int64(0)  # int64: the prox updates'
        pc10 = wp.int64(0)  # int64: the Cholesky solve's
        if wp.static(PROFILE):
            tstart = _clock()
            tclk = tstart
        ne = ne_in[worldid]  # int: the world's equality rows, first
        nf = nf_in[worldid]  # int: its friction rows, next
        nefc = wp.min(njmax_in, nefc_in[worldid])  # int: its rows, within njmax
        nchunk = (nefc + THREADS - 1) // THREADS  # int: chunks of THREADS rows
        tolerance = opt_tolerance[worldid % opt_tolerance.shape[0]]  # float
        ls_tolerance = opt_ls_tolerance[worldid % opt_ls_tolerance.shape[0]]  # float
        meaninertia = stat_meaninertia[worldid % stat_meaninertia.shape[0]]  # float
        scale = meaninertia * float(NV)  # float: MJWarp's _rescale
        own_i = veci()  # veci: the Hessian entries this thread owns: row and column
        own_j = veci()  # veci
        for mo in range(KO):  # mo: int, an owned entry
            k = tid + THREADS * mo  # int: its index in the packed lower triangle
            ii = int(0)  # int: its row
            for i in range(NV):  # i: int, a dof
                if i * (i + 1) // 2 <= k:
                    ii = i
            own_i[mo] = ii
            own_j[mo] = wp.min(k - ii * (ii + 1) // 2, NV - 1)

        # ---- the world's rows into shared memory
        for c in range(nchunk):  # c: int, a chunk of rows
            r = c * THREADS + tid  # int: this thread's row in it
            if r < nefc:
                if wp.static(JSH):
                    for i in range(NV):
                        shf(OJ + r * NVS + i, efc_J_in[worldid, r, i], 1)
                shf(OD + r, efc_D_in[worldid, r], 1)
                kind = ONESIDED  # int: the row's kind
                p = float(0.0)  # float: P, a friction row's loss or, under prox, a contact row's aref0
                if r < ne:
                    kind = EQUALITY
                elif r < ne + nf:
                    kind = FRICTION
                    p = efc_frictionloss_in[worldid, r]
                etype = efc_type_in[worldid, r]  # int: its ConstraintType
                a0 = r  # int: its elliptic contact's normal row, else itself
                if wp.static(IS_ELLIPTIC):
                    if kind == ONESIDED and etype == CONTACT_ELLIPTIC:
                        conid = efc_id_in[worldid, r]  # int: its contact
                        a0 = contact_efc_address_in[conid, 0]
                        if a0 + 2 >= nefc:
                            kind = IGNORED
                        else:
                            kind = wp.where(r == a0, ELLIPTIC0, ELLIPTIC1)
                            friction = contact_friction_in[conid]  # vec5
                            x = friction[0] * contact_impratio_invsqrt_in[conid]  # float: the normal row: mu
                            if r == a0 + 1:
                                x = friction[0]
                            elif r == a0 + 2:
                                x = friction[1]
                            shf(OX + r, x, 1)
                            if wp.static(PROX):
                                if r == a0:
                                    shf(OG + r, prox_gamma_in[conid], 1)
                                else:
                                    shf(OV + r, efc_vel_in[worldid, r], 1)
                if wp.static(PROX):
                    if kind == ONESIDED and etype == CONTACT_FRICTIONLESS:
                        kind = CONTACT1
                        shf(OG + r, prox_gamma_in[efc_id_in[worldid, r]], 1)
                    if kind == ELLIPTIC0 or kind == ELLIPTIC1 or kind == CONTACT1:
                        p = prox_aref0_in[worldid, r]
                        shf(OZ + r, prox_z[worldid, r], 1)
                    shf(OAR + r, efc_aref_in[worldid, r], 1)
                shf(OP + r, p, 1)
                shi(r, kind | (a0 << 3), 1)
        for e in range(tid, NV * NV, THREADS):  # e: int, an element of M, row-major
            k = M_entry[e // NV, e % NV]
            mij = float(0.0)  # float: element e of M
            if k >= 0:
                mij = M_in[worldid, k]
            shf(OM + e, mij, 1)
        # The dof vectors qacc, Ma and qfrc_smooth live in shared memory too; thread i writes entry i.
        qacc = vecv()  # vecv: the world's qacc, in every thread
        for i in range(NV):
            if wp.static(WARMSTART):
                qacc[i] = qacc_warmstart_in[worldid, i]
            else:
                qacc[i] = qacc_smooth_in[worldid, i]
            if tid == i:
                shf(OQA + i, qacc[i], 1)
                shf(OQS + i, qfrc_smooth_in[worldid, i], 1)
        _syncthreads()
        for c in range(nchunk):
            r = c * THREADS + tid
            if r < nefc:
                aref = float(0.0)  # float: the row's aref
                if wp.static(PROX):
                    aref = shf(OAR + r, 0.0, 0)
                else:
                    aref = efc_aref_in[worldid, r]
                shf(OJA + r, wp.dot(jrow(efc_J_in, worldid, r), qacc) - aref, 1)
        for i in range(NV):
            if tid == i:
                s = float(0.0)  # float: (M qacc)_i
                for j in range(NV):  # j: int, a dof
                    s += shf(OM + i * NV + j, 0.0, 0) * qacc[j]
                shf(OMA + i, s, 1)
        _syncthreads()
        timestep = float(1.0)  # float: the timestep (s), read under prox
        tol_z = float(0.0)  # float
        tol_sigma = float(0.0)  # float
        relax = float(0.0)  # float: the De Saxce update's relaxation
        if wp.static(PROX):
            timestep = opt_timestep[worldid % opt_timestep.shape[0]]
            tol_z = opt_prox_tol_z[worldid % opt_prox_tol_z.shape[0]]
            tol_sigma = opt_prox_tol_sigma[worldid % opt_prox_tol_sigma.shape[0]]
            relax = opt_prox_sigma_relax[worldid % opt_prox_sigma_relax.shape[0]]

        search = vecv()  # vecv: the search direction
        qfrc = vecv()  # vecv: J^T f, qfrc_constraint
        niter = int(0)  # int: the round's Newton iterations
        prox_niter = int(0)  # int: all rounds' Newton iterations
        prox_converged = bool(False)  # bool
        if wp.static(PROFILE):
            tnow = _clock()  # int64
            pc0 += tnow - tclk
            tclk = tnow
        for rnd in range(nround):  # rnd: int, a prox round
            improvement = float(0.0)  # float: the last line search's cost decrease
            niter = int(0)
            for it in range(opt_iterations + 1):  # it: int, a Newton iteration; 0 forms the first search
                if it > 0:
                    # ---- the line search: MJWarp's _linesearch_iterative_kernel
                    mv = vecv()  # vecv: M search
                    for i in range(NV):
                        s = float(0.0)
                        for j in range(NV):
                            s += shf(OM + i * NV + j, 0.0, 0) * search[j]
                        mv[i] = s
                    if wp.static(PROFILE):
                        tnow = _clock()
                        pc8 += tnow - tclk
                        tclk = tnow
                    for c in range(nchunk):
                        r = c * THREADS + tid
                        if r < nefc:
                            shf(OJV + r, wp.dot(jrow(efc_J_in, worldid, r), search), 1)
                    _syncthreads()
                    if wp.static(IS_ELLIPTIC):
                        for c in range(nchunk):
                            r = c * THREADS + tid
                            if r < nefc:
                                if kind_of(r) == ELLIPTIC0:
                                    Jaref = shf(OJA + r, 0.0, 0)  # float: the normal row's jaref
                                    jv0 = shf(OJV + r, 0.0, 0)  # float: its jv
                                    efc_D = shf(OD + r, 0.0, 0)  # float: its D
                                    mu = shf(OX + r, 0.0, 0)  # float: the contact's mu
                                    jvD = jv0 * efc_D  # float
                                    # wp.vec3: the quadratic's coefficients, summed over the contact's rows
                                    q = wp.vec3(0.5 * Jaref * Jaref * efc_D, jvD * Jaref, 0.5 * jv0 * jvD)
                                    u0 = Jaref * mu  # float
                                    v0 = jv0 * mu  # float
                                    jv1 = shf(OJV + r + 1, 0.0, 0)  # float: the first tangent row's jv
                                    jaref1 = shf(OJA + r + 1, 0.0, 0)  # float: its jaref
                                    d1 = shf(OD + r + 1, 0.0, 0)  # float: its D
                                    DJ1 = d1 * jaref1  # float
                                    q += wp.vec3(0.5 * jaref1 * DJ1, jv1 * DJ1, 0.5 * jv1 * d1 * jv1)
                                    jv2 = shf(OJV + r + 2, 0.0, 0)  # float: the second tangent row's jv
                                    jaref2 = shf(OJA + r + 2, 0.0, 0)  # float
                                    d2 = shf(OD + r + 2, 0.0, 0)  # float
                                    DJ2 = d2 * jaref2  # float
                                    q += wp.vec3(0.5 * jaref2 * DJ2, jv2 * DJ2, 0.5 * jv2 * d2 * jv2)
                                    f1 = shf(OX + r + 1, 0.0, 0)  # float: the first tangent's friction
                                    f2 = shf(OX + r + 2, 0.0, 0)  # float: the second's
                                    u1 = jaref1 * f1  # float
                                    v1 = jv1 * f1  # float
                                    u2 = jaref2 * f2  # float
                                    v2 = jv2 * f2  # float
                                    mu2 = mu * mu  # float
                                    shf(OQ + 3 * r, q[0], 1)
                                    shf(OQ + 3 * r + 1, q[1], 1)
                                    shf(OQ + 3 * r + 2, q[2], 1)
                                    shf(OQ + 3 * r + 3, u0, 1)
                                    shf(OQ + 3 * r + 4, v0, 1)
                                    shf(OQ + 3 * r + 5, u1 * u1 + u2 * u2, 1)
                                    shf(OQ + 3 * r + 6, u1 * v1 + u2 * v2, 1)
                                    shf(OQ + 3 * r + 7, v1 * v1 + v2 * v2, 1)
                                    shf(OQ + 3 * r + 8, efc_D / (mu2 * (1.0 + mu2)), 1)
                        _syncthreads()
                    if wp.static(PROFILE):
                        tnow = _clock()
                        pc1 += tnow - tclk
                        tclk = tnow

                    snorm = wp.sqrt(wp.dot(search, search))  # float: |search|
                    gtol = wp.max(tolerance * ls_tolerance * snorm * scale, 1.0e-6)  # float: the slope tolerance
                    local_gauss = wp.vec2(0.0)  # wp.vec2: this thread's Gauss slope and curvature
                    if tid == 0:
                        g1 = float(0.0)  # float: search . (Ma - qfrc_smooth)
                        for i in range(NV):
                            g1 += search[i] * (shf(OMA + i, 0.0, 0) - shf(OQS + i, 0.0, 0))
                        local_gauss = wp.vec2(g1, 0.5 * wp.dot(search, mv))
                    local_p0 = wp.vec3(0.0)  # wp.vec3: this thread's rows' point at alpha 0
                    for c in range(nchunk):
                        r = c * THREADS + tid
                        if r < nefc:
                            kind, ja, jvr, Dr, flr, q0, q1, q2, mu = row_inputs(r)  # int, float x 4, wp.vec3 x 3, float
                            if kind == EQUALITY:
                                jvD = jvr * Dr
                                local_gauss += wp.vec2(jvD * ja, 0.5 * jvr * jvD)
                            else:
                                local_p0 += _pt_alpha_zero(kind, ja, jvr, Dr, flr, q0, q1, q2, mu)
                    gauss, rows = bsum5(local_gauss, local_p0, lane, warp, rbuf)  # wp.vec2, wp.vec3: the block's sums
                    rbuf = 1 - rbuf
                    quad_gauss = wp.vec3(0.0, gauss[0], gauss[1])  # wp.vec3: the Gauss term's quadratic in alpha
                    p0 = wp.vec3(quad_gauss[0], quad_gauss[1], 2.0 * quad_gauss[2]) + rows  # wp.vec3: the point at 0
                    p0_delta = wp.vec3(0.0, p0[1], p0[2])  # wp.vec3: p0 with its cost 0
                    q1_abs = wp.sqrt(2.0 * wp.max(rows[0], 0.0) * wp.max(rows[2], 0.0)) + wp.abs(quad_gauss[1])  # float
                    gtol_accept = wp.max(gtol, ALPHA_NOISE_EPS * q1_abs)  # float: the slope tolerance, above rounding

                    lo_alpha_in = -lmath.safe_div(p0[1], p0[2])  # float: the Newton step from 0
                    local_lo_in = wp.vec3(0.0)  # wp.vec3
                    for c in range(nchunk):
                        r = c * THREADS + tid
                        if r < nefc:
                            kind, ja, jvr, Dr, flr, q0, q1, q2, mu = row_inputs(r)
                            local_lo_in += _pt(kind, lo_alpha_in, ja, jvr, Dr, flr, q0, q1, q2, mu)
                    # wp.vec3 x 3: the rows' point at lo_alpha_in, two unused sums
                    lo_in_sum, unused0, unused1 = bsum9(local_lo_in, wp.vec3(0.0), wp.vec3(0.0), lane, warp, rbuf)
                    rbuf = 1 - rbuf
                    lo_in = nlsp._eval_pt(quad_gauss, lo_alpha_in) + lo_in_sum  # wp.vec3: the point at lo_alpha_in
                    if wp.static(PROFILE):
                        tnow = _clock()
                        pc2 += tnow - tclk
                        tclk = tnow

                    alpha = float(0.0)  # float: the step taken
                    improvement = float(0.0)
                    initial_converged = wp.abs(lo_in[1]) < gtol_accept and lo_in[0] < 0.0  # bool
                    if not initial_converged:
                        lo_less = lo_in[1] < p0[1]  # bool
                        lo = wp.where(lo_less, lo_in, p0_delta)  # wp.vec3: the bracket's low point
                        lo_alpha = wp.where(lo_less, lo_alpha_in, 0.0)  # float
                        hi = wp.where(lo_less, p0_delta, lo_in)  # wp.vec3: its high point
                        hi_alpha = wp.where(lo_less, 0.0, lo_alpha_in)  # float
                        for _ in range(opt_ls_iterations):  # _: int, a line-search iteration
                            lo_next_alpha = lo_alpha - lmath.safe_div(lo[1], lo[2])  # float: Newton's step from lo
                            hi_next_alpha = hi_alpha - lmath.safe_div(hi[1], hi[2])  # float: from hi
                            mid_alpha = 0.5 * (lo_alpha + hi_alpha)  # float
                            local_lo = wp.vec3(0.0)  # wp.vec3
                            local_hi = wp.vec3(0.0)  # wp.vec3
                            local_mid = wp.vec3(0.0)  # wp.vec3
                            for c in range(nchunk):
                                r = c * THREADS + tid
                                if r < nefc:
                                    kind, ja, jvr, Dr, flr, q0, q1, q2, mu = row_inputs(r)
                                    # wp.vec3 x 3: the row's points
                                    r_lo, r_hi, r_mid = _pt3(kind, lo_next_alpha, hi_next_alpha, mid_alpha, ja, jvr,
                                                             Dr, flr, q0, q1, q2, mu)
                                    local_lo += r_lo
                                    local_hi += r_hi
                                    local_mid += r_mid
                            gauss_lo, gauss_hi, gauss_mid = nlsp._eval_pt_3alphas(  # wp.vec3 x 3: the Gauss term's
                                quad_gauss, lo_next_alpha, hi_next_alpha, mid_alpha
                            )
                            # wp.vec3 x 3: the rows' points, summed over the block
                            sum_lo, sum_hi, sum_mid = bsum9(local_lo, local_hi, local_mid, lane, warp, rbuf)
                            rbuf = 1 - rbuf
                            lo_next = gauss_lo + sum_lo  # wp.vec3
                            hi_next = gauss_hi + sum_hi  # wp.vec3
                            mid = gauss_mid + sum_mid  # wp.vec3
                            conv_lo = wp.abs(lo_next[1]) < gtol_accept and lo_next[0] < 0.0  # bool
                            conv_hi = wp.abs(hi_next[1]) < gtol_accept and hi_next[0] < 0.0  # bool
                            conv_mid = wp.abs(mid[1]) < gtol_accept and mid[0] < 0.0  # bool
                            converged = conv_lo or conv_hi or conv_mid  # bool
                            swap_lo = False  # bool: lo moved
                            swap_hi = False  # bool: hi moved
                            if converged:
                                conv_pt = wp.vec3(MAXVAL, 0.0, 0.0)  # wp.vec3: the best converged point
                                conv_alpha = float(0.0)  # float
                                if conv_lo and lo_next[0] < conv_pt[0]:
                                    conv_pt = lo_next
                                    conv_alpha = lo_next_alpha
                                if conv_hi and hi_next[0] < conv_pt[0]:
                                    conv_pt = hi_next
                                    conv_alpha = hi_next_alpha
                                if conv_mid and mid[0] < conv_pt[0]:
                                    conv_pt = mid
                                    conv_alpha = mid_alpha
                                lo = conv_pt
                                lo_alpha = conv_alpha
                                hi = conv_pt
                                hi_alpha = conv_alpha
                            else:
                                swap_lo_lo_next = nlsp._in_bracket(lo, lo_next)  # bool: lo moves to lo_next
                                lo = wp.where(swap_lo_lo_next, lo_next, lo)
                                lo_alpha = wp.where(swap_lo_lo_next, lo_next_alpha, lo_alpha)
                                swap_lo_mid = nlsp._in_bracket(lo, mid)  # bool: lo moves to mid
                                lo = wp.where(swap_lo_mid, mid, lo)
                                lo_alpha = wp.where(swap_lo_mid, mid_alpha, lo_alpha)
                                swap_lo_hi_next = nlsp._in_bracket(lo, hi_next)  # bool: lo moves to hi_next
                                lo = wp.where(swap_lo_hi_next, hi_next, lo)
                                lo_alpha = wp.where(swap_lo_hi_next, hi_next_alpha, lo_alpha)
                                swap_lo = swap_lo_lo_next or swap_lo_mid or swap_lo_hi_next
                                # bool: hi moves to hi_next
                                swap_hi_hi_next = nlsp._in_bracket(hi, hi_next) or (hi[1] < 0.0 and hi_next[1] > 0.0)
                                hi = wp.where(swap_hi_hi_next, hi_next, hi)
                                hi_alpha = wp.where(swap_hi_hi_next, hi_next_alpha, hi_alpha)
                                swap_hi_mid = nlsp._in_bracket(hi, mid)  # bool: hi moves to mid
                                hi = wp.where(swap_hi_mid, mid, hi)
                                hi_alpha = wp.where(swap_hi_mid, mid_alpha, hi_alpha)
                                swap_hi_lo_next = nlsp._in_bracket(hi, lo_next)  # bool: hi moves to lo_next
                                hi = wp.where(swap_hi_lo_next, lo_next, hi)
                                hi_alpha = wp.where(swap_hi_lo_next, lo_next_alpha, hi_alpha)
                                swap_hi = swap_hi_hi_next or swap_hi_mid or swap_hi_lo_next
                            ls_done = (  # bool: the line search is done
                                converged
                                or (not swap_lo and not swap_hi)
                                or (lo[0] < 0.0 and lo[1] < 0.0 and lo[1] > -gtol)
                                or (hi[0] < 0.0 and hi[1] > 0.0 and hi[1] < gtol)
                            )
                            improved = lo[0] < 0.0 or hi[0] < 0.0  # bool
                            lo_better = lo[0] < hi[0]  # bool
                            best_alpha = wp.where(lo_better, lo_alpha, hi_alpha)  # float
                            best_delta = wp.where(lo_better, lo[0], hi[0])  # float: its cost change
                            alpha = wp.where(improved, best_alpha, alpha)
                            improvement = wp.where(improved, -best_delta, improvement)
                            if ls_done:
                                break
                    else:
                        alpha = lo_alpha_in
                        improvement = -lo_in[0]
                    if wp.static(PROFILE):
                        tnow = _clock()
                        pc3 += tnow - tclk
                        tclk = tnow

                    for i in range(NV):
                        if tid == i:
                            shf(OQA + i, shf(OQA + i, 0.0, 0) + alpha * search[i], 1)
                            shf(OMA + i, shf(OMA + i, 0.0, 0) + alpha * mv[i], 1)
                    for c in range(nchunk):
                        r = c * THREADS + tid
                        if r < nefc:
                            shf(OJA + r, shf(OJA + r, 0.0, 0) + alpha * shf(OJV + r, 0.0, 0), 1)
                    _syncthreads()
                    if wp.static(PROFILE):
                        tnow = _clock()
                        pc4 += tnow - tclk
                        tclk = tnow

                # ---- the constraint update (MJWarp's _update_constraint), the gradient and the Hessian
                # The Hessian's rows S (H = M + S^T S) pass through shared memory a chunk at a time; thread t owns the
                # entries t, t + threads, ... of H's packed lower triangle and sums them over the staged rows.
                hown = veco()  # veco: this thread's entries of S^T S
                local_qfrc = vecv()  # vecv: this thread's rows' J^T f
                for c in range(nchunk):
                    r = c * THREADS + tid
                    # This thread's rows of S -- none on a satisfied or linear row, one on a quadratic row, two on an
                    # elliptic contact in the cone's middle zone -- go from J in shared memory to the staging rows,
                    # packed by a prefix sum over the block, so the owners run over the chunk's nonzero rows alone.
                    nt = int(0)  # int: this thread's rows of S, 0, 1 or 2
                    wq = float(0.0)  # float: a quadratic row's sqrt(D)
                    wa = float(0.0)  # float: sqrt(dm) mu, the first cone row's scale
                    wb = float(0.0)  # float: sqrt(dm (mu^2 - mu N / T)), the second's
                    c1 = float(0.0)  # float: f1 u1 / T
                    c2 = float(0.0)  # float: f2 u2 / T
                    e1 = float(0.0)  # float: -f1 u2 / T
                    e2 = float(0.0)  # float: f2 u1 / T
                    if r < nefc:
                        # float, int, int, then floats, as `eval_row`'s
                        force, state, efcid0, jaref0, D0, mu, u1, u2, TT, f1, f2 = eval_row(r)
                        efc_force_out[worldid, r] = force
                        efc_state_out[worldid, r] = state
                        for i in range(NV):
                            local_qfrc[i] += force * jget(efc_J_in, worldid, r, i)
                        if state == QUADRATIC:
                            wq = wp.sqrt(shf(OD + r, 0.0, 0))
                            nt = 1
                        elif wp.static(IS_ELLIPTIC):
                            if state == CONE and r == efcid0:
                                nt = 2
                                T = wp.max(wp.sqrt(TT), MINVAL)  # float: the tangential norm |u|
                                c1 = f1 * u1 / T  # z_j = friction_j J_j enters as (u_j / T) friction_j J_j
                                c2 = f2 * u2 / T
                                e1 = -f1 * u2 / T  # e = (-u_2, u_1) / T against z_1 and z_2
                                e2 = f2 * u1 / T
                                mu2 = mu * mu
                                dm = lmath.safe_div(D0, mu2 * (1.0 + mu2))  # float: D0 / (mu^2 (1 + mu^2))
                                wa = wp.sqrt(dm) * mu
                                wb = wp.sqrt(wp.max(dm * (mu2 - mu2 * jaref0 / T), 0.0))
                    off, nstaged = bscan(nt, lane, warp, rbuf)  # int, int: its first staged row, the rows staged
                    rbuf = 1 - rbuf
                    if nt == 1:
                        for i in range(NV):
                            shf(OS + off * NVS + i, wq * jget(efc_J_in, worldid, r, i), 1)
                    elif nt == 2:
                        for i in range(NV):
                            j0 = jget(efc_J_in, worldid, r, i)  # float: J[r, i]
                            j1 = jget(efc_J_in, worldid, r + 1, i)  # float: J[r + 1, i]
                            j2 = jget(efc_J_in, worldid, r + 2, i)  # float: J[r + 2, i]
                            shf(OS + off * NVS + i, wa * (j0 - c1 * j1 - c2 * j2), 1)
                            shf(OS + (off + 1) * NVS + i, wb * (e1 * j1 + e2 * j2), 1)
                    _syncthreads()
                    hown = accumulate(hown, own_i, own_j, tid, nstaged)
                    _syncthreads()
                if wp.static(PROFILE):
                    tnow = _clock()
                    pc5 += tnow - tclk
                    tclk = tnow
                qfrc = bsumv(local_qfrc, lane, warp, rbuf)  # J^T f
                rbuf = 1 - rbuf
                grad = vecv()  # vecv: the gradient, Ma - qfrc_smooth - J^T f
                for i in range(NV):
                    grad[i] = shf(OMA + i, 0.0, 0) - shf(OQS + i, 0.0, 0) - qfrc[i]
                # H = M + S^T S in shared memory; thread 0 factors it in place (packed lower triangle, the diagonal
                # holding the reciprocals) and solves for H^-1 grad, which every thread then reads. (Held in every
                # thread's registers, the factor would spill.)
                for mo in range(KO):
                    if tid + THREADS * mo < NP:
                        shf(OH + tid + THREADS * mo, hown[mo] + shf(OM + own_i[mo] * NV + own_j[mo], 0.0, 0), 1)
                _syncthreads()
                if wp.static(PROFILE):
                    tnow = _clock()
                    pc6 += tnow - tclk
                    tclk = tnow
                if tid == 0:
                    for j in range(NV):
                        djj = shf(OH + j * (j + 1) // 2 + j, 0.0, 0)  # float: the pivot, H[j, j] - sum_k L[j, k]^2
                        for k in range(NV):
                            if k < j:
                                ljk = shf(OH + j * (j + 1) // 2 + k, 0.0, 0)  # float: L[j, k]
                                djj -= ljk * ljk
                        inv = 1.0 / wp.sqrt(wp.max(djj, MINVAL))  # float: 1 / L[j, j]
                        shf(OH + j * (j + 1) // 2 + j, inv, 1)
                        for i in range(NV):
                            if i > j:
                                s = shf(OH + i * (i + 1) // 2 + j, 0.0, 0)
                                for k in range(NV):
                                    if k < j:
                                        s -= shf(OH + i * (i + 1) // 2 + k, 0.0, 0) * shf(OH + j * (j + 1) // 2 + k, 0.0, 0)
                                shf(OH + i * (i + 1) // 2 + j, s * inv, 1)
                    y = vecv()  # vecv: L^-1 grad, then H^-1 grad
                    for i in range(NV):
                        s = grad[i]
                        for k in range(NV):
                            if k < i:
                                s -= shf(OH + i * (i + 1) // 2 + k, 0.0, 0) * y[k]
                        y[i] = s * shf(OH + i * (i + 1) // 2 + i, 0.0, 0)
                    for ii in range(NV):
                        i = NV - 1 - ii
                        s = y[i]
                        for k in range(NV):
                            if k > i:
                                s -= shf(OH + k * (k + 1) // 2 + i, 0.0, 0) * y[k]
                        y[i] = s * shf(OH + i * (i + 1) // 2 + i, 0.0, 0)
                    for i in range(NV):
                        shf(OMG + i, y[i], 1)
                _syncthreads()
                mgrad = vecv()  # vecv: H^-1 grad
                for i in range(NV):
                    mgrad[i] = shf(OMG + i, 0.0, 0)
                search = -mgrad
                if wp.static(PROFILE):
                    tnow = _clock()
                    pc10 += tnow - tclk
                    tclk = tnow

                if it > 0:  # MJWarp's _solve_done
                    niter += 1
                    done = (  # bool
                        improvement / scale < tolerance
                        or wp.sqrt(wp.dot(grad, grad)) / scale < tolerance
                        or 0.5 * wp.dot(grad, mgrad) / scale < tolerance
                    )
                    if done or niter == opt_iterations:
                        break

            if wp.static(not PROX):
                break
            # ---- the prox round's update and test: nlsp's _prox_update and _prox_check, per contact by the thread
            # of its normal row, then the rows' Jaref against the new aref (_refresh_after_aref).
            prox_niter += niter
            update_sigma = ((rnd + 1) % inner == 0) and desaxce != 0  # bool: the round takes a De Saxce step
            if wp.static(GROUPS):
                if update_sigma:
                    # A group's De Saxce term is its members' (`_group_gamma`), at qvel + dt qacc read from global
                    # memory: the round's qacc goes there first, for every thread of the block to see.
                    if tid < NV:
                        qacc_out[worldid, tid] = shf(OQA + tid, 0.0, 0)
                    _syncthreads()
            local_max = wp.vec4(0.0)  # wp.vec4: dz, znorm, dsigma, signorm
            for c in range(nchunk):
                r = c * THREADS + tid
                if r < nefc:
                    kind = kind_of(r)
                    if kind == ELLIPTIC0 or kind == CONTACT1:
                        f0 = efc_force_out[worldid, r]  # float: the normal row's force
                        D0 = shf(OD + r, 0.0, 0)
                        dz = wp.abs(f0 - shf(OZ + r, 0.0, 0))  # float: the anchor's change
                        znorm = wp.abs(f0)  # float: the anchor's size
                        shf(OZ + r, f0, 1)
                        gamma = shf(OG + r, 0.0, 0)  # float: the contact's De Saxce term
                        is_group = False  # bool: a GROUP contact
                        conid = int(-1)
                        if wp.static(GROUPS):
                            conid = efc_id_in[worldid, r]
                            is_group = (contact_type_in[conid] & GROUP) != 0
                        if kind == ELLIPTIC0:
                            f1 = efc_force_out[worldid, r + 1]
                            f2 = efc_force_out[worldid, r + 2]
                            dz = wp.max(dz, wp.abs(f1 - shf(OZ + r + 1, 0.0, 0)))
                            dz = wp.max(dz, wp.abs(f2 - shf(OZ + r + 2, 0.0, 0)))
                            znorm = wp.max(znorm, wp.max(wp.abs(f1), wp.abs(f2)))
                            shf(OZ + r + 1, f1, 1)
                            shf(OZ + r + 2, f2, 1)
                            if update_sigma:
                                target = float(0.0)  # float: the De Saxce term's new target
                                if is_group:
                                    if wp.static(GROUPS):
                                        target = nlsp._group_gamma(
                                            body_rootid, body_weldid, body_dofnum, body_dofadr, dof_parentid,
                                            geom_bodyid, qvel_in, qacc_out, cdof_in, subtree_com_in,
                                            contact_group_adr_in, contact_group_num_in, group_member_pos_in,
                                            group_member_frame_in, group_member_weight_in, group_member_slip_in,
                                            worldid, conid, contact_geom_in[conid], timestep, shf(OX + r, 0.0, 0),
                                            False)
                                else:
                                    # J qacc = Jaref + aref; impratio is 1 under the prox loop: mu is nlsp's mu_eff
                                    sig1 = shf(OV + r + 1, 0.0, 0) + timestep * (  # float: sigma_t[0]
                                        shf(OJA + r + 1, 0.0, 0) + shf(OAR + r + 1, 0.0, 0))
                                    sig2 = shf(OV + r + 2, 0.0, 0) + timestep * (  # float: sigma_t[1]
                                        shf(OJA + r + 2, 0.0, 0) + shf(OAR + r + 2, 0.0, 0))
                                    target = shf(OX + r, 0.0, 0) * wp.sqrt(sig1 * sig1 + sig2 * sig2)
                                gamma_new = gamma + relax * (target - gamma)  # float: the relaxed update
                                local_max = wp.vec4(local_max[0], local_max[1],
                                                    wp.max(local_max[2], wp.abs(gamma_new - gamma)),
                                                    wp.max(local_max[3], gamma_new))
                                shf(OG + r, gamma_new, 1)
                                gamma = gamma_new
                            shf(OAR + r + 1, shf(OP + r + 1, 0.0, 0) + f1 / shf(OD + r + 1, 0.0, 0), 1)
                            shf(OAR + r + 2, shf(OP + r + 2, 0.0, 0) + f2 / shf(OD + r + 2, 0.0, 0), 1)
                        elif is_group:
                            # a ray group (condim 1): its members' slips along their friction directions
                            if wp.static(GROUPS):
                                if update_sigma:
                                    target = nlsp._group_gamma(
                                        body_rootid, body_weldid, body_dofnum, body_dofadr, dof_parentid, geom_bodyid,
                                        qvel_in, qacc_out, cdof_in, subtree_com_in, contact_group_adr_in,
                                        contact_group_num_in, group_member_pos_in, group_member_frame_in,
                                        group_member_weight_in, group_member_slip_in, worldid, conid,
                                        contact_geom_in[conid], timestep, 0.0, True)
                                    gamma_new = gamma + relax * (target - gamma)
                                    local_max = wp.vec4(local_max[0], local_max[1],
                                                        wp.max(local_max[2], wp.abs(gamma_new - gamma)),
                                                        wp.max(local_max[3], gamma_new))
                                    shf(OG + r, gamma_new, 1)
                                    gamma = gamma_new
                        shf(OAR + r, shf(OP + r, 0.0, 0) - gamma / timestep + f0 / D0, 1)
                        local_max = wp.vec4(wp.max(local_max[0], dz), wp.max(local_max[1], znorm), local_max[2],
                                            local_max[3])
            world_max = bmax4(local_max, lane, warp, rbuf)  # wp.vec4: the world's dz, znorm, dsigma, signorm
            rbuf = 1 - rbuf
            if update_sigma or z_only != 0:
                z_ok = world_max[0] <= tol_z * (1.0 + world_max[1])  # bool
                sigma_ok = True  # bool
                if update_sigma:
                    sigma_ok = world_max[2] <= tol_sigma * (1.0 + world_max[3])
                if z_ok and sigma_ok:
                    prox_converged = True
            _syncthreads()
            if wp.static(PROFILE):
                tnow = _clock()
                pc9 += tnow - tclk
                tclk = tnow
            if prox_converged:
                break
            qacc = vecv()
            for i in range(NV):
                qacc[i] = shf(OQA + i, 0.0, 0)
            for c in range(nchunk):
                r = c * THREADS + tid
                if r < nefc:
                    shf(OJA + r, wp.dot(jrow(efc_J_in, worldid, r), qacc) - shf(OAR + r, 0.0, 0), 1)
            _syncthreads()

        if wp.static(PROX):
            # The budget spent unconverged: MJWarp's last refresh leaves the forces of the next round's rows.
            if not prox_converged:
                for c in range(nchunk):
                    r = c * THREADS + tid
                    if r < nefc:
                        force, state, efcid0, jaref0, D0, mu, u1, u2, TT, f1, f2 = eval_row(r)
                        efc_force_out[worldid, r] = force
                        efc_state_out[worldid, r] = state
            for c in range(nchunk):
                r = c * THREADS + tid
                if r < nefc:
                    kind = kind_of(r)
                    if kind == ELLIPTIC0 or kind == ELLIPTIC1 or kind == CONTACT1:
                        prox_z[worldid, r] = shf(OZ + r, 0.0, 0)
            niter = prox_niter
        if tid == 0:
            for i in range(NV):
                qacc_out[worldid, i] = shf(OQA + i, 0.0, 0)
                qfrc_constraint_out[worldid, i] = qfrc[i]
            solver_niter_out[worldid] = niter
            if wp.static(PROFILE):
                prof[worldid, 0] = pc0
                prof[worldid, 1] = pc1
                prof[worldid, 2] = pc2
                prof[worldid, 3] = pc3
                prof[worldid, 4] = pc4
                prof[worldid, 5] = pc5
                prof[worldid, 6] = pc6
                prof[worldid, 7] = _clock() - tstart
                prof[worldid, 8] = pc8
                prof[worldid, 9] = pc9
                prof[worldid, 10] = pc10

    _KERNELS[key] = kernel
    return kernel


def dense_M_entries(mjm):
    """
    (nv, nv): the entry of MuJoCo's CSR `M` holding each element, both triangles; -1 where M is zero.

    Called by `FusedNewton.__init__`. In: mjm, MuJoCo's `MjModel` (`M_rowadr`, `M_rownnz`, `M_colind`).
    Out: (nv, nv) int32.
    """
    nv = int(mjm.nv)  # int
    out = -np.ones((nv, nv), dtype=np.int32)  # (nv, nv) int32
    for i in range(nv):  # i: int, a dof: a row of M
        for e in range(mjm.M_rowadr[i], mjm.M_rowadr[i] + mjm.M_rownnz[i]):  # e: int, a CSR entry of row i
            j = int(mjm.M_colind[e])  # int: its column
            out[i, j] = out[j, i] = e
    return out


def _max_condim(mjm):
    """
    The largest condim a contact of the model's own geoms and pairs can take.

    Called by `FusedNewton.__init__`. In: mjm, MuJoCo's `MjModel` (`geom_condim` (ngeom,), `pair_dim` (npair,)).
    Out: int.
    """
    c = [int(x) for x in mjm.geom_condim] + [int(x) for x in mjm.pair_dim]  # list of int: the condims
    return max(c) if c else 1


class FusedNewton:
    """
    MJWarp's Newton solve -- and, under `opt.prox_enable`, the prox solver around it -- fused into one kernel per
    world, for small dense models.

    `solve(m, d)` stands in for `lodos.cd.solver.nlsp.solve(m, d)`: it reads the rows `make_constraint`
    assembled and writes `qacc`, `qfrc_constraint`, `efc.force`, `efc.state` and `solver_niter` (under prox the
    Newton iterations of all its rounds), and under prox `efc.D`, `efc.aref` (`_prox_init`'s) and the anchor cache.
    Every array is allocated here; the solve is capturable.

    Built by `lodos.cd.driver.SolverDriver.__init__`.
    """

    def __init__(self, mjm, m, d, profile=False, warps=WARPS, jshared=None):
        """
        Checks that the model is one the kernel takes, builds the kernel for its sizes and allocates the solve's arrays.

        Called by `lodos.cd.driver.SolverDriver.__init__`. In: mjm, MuJoCo's `MjModel`; m, d, MJWarp's `Model` and
        `Data`; profile bool, the kernel's clock counters; warps int, or None: 1 up to 128 rows a world, 2 above;
        jshared bool, or None: False.
        """
        nv = int(mjm.nv)  # int
        if warps is None:
            warps = 1 if d.njmax <= 128 else 2
        if m.opt.solver != mjcf_types.SolverType.NEWTON:
            raise ValueError("the fused Newton solve is Newton's: the model asks for CG")
        if m.is_sparse or nv > NV_MAX:
            raise ValueError(f"the fused Newton solve takes small dense models (nv <= {NV_MAX}, a dense Jacobian); "
                             f"this one has nv {nv}, {'sparse' if m.is_sparse else 'dense'}")
        if m.opt.enableflags & mjcf_types.EnableBit.SLEEP:
            raise ValueError("the fused Newton solve does not take sleeping")
        elliptic = m.opt.cone == mjcf_types.ConeType.ELLIPTIC  # bool
        if elliptic and _max_condim(mjm) > 3:
            raise ValueError("the fused Newton solve takes elliptic contacts of condim 3 at most")
        if d.njmax == 0:
            raise ValueError("the fused Newton solve needs constraint rows (njmax > 0)")
        prox = bool(m.opt.prox_enable)  # bool: the prox loop around the solve
        if prox:
            nlsp._validate(m)
            if m.opt.prox_soft:
                raise ValueError("the fused Newton solve runs the prox loop on its exact rows: not prox_soft")
            if not np.allclose(m.opt.impratio_invsqrt.numpy(), 1.0):
                raise ValueError("the fused Newton solve runs the prox loop at impratio 1")
        if jshared is None:  # J read from efc.J, where L1 holds it: the smaller block lets more worlds share an SM
            jshared = False
        nbytes = shared_layout(nv, int(d.njmax), elliptic, prox, warps, bool(jshared))[-1]  # int: a world's bytes
        if nbytes > SHARED_MAX:
            raise ValueError(f"the fused Newton solve keeps a world's rows in shared memory: njmax {d.njmax} at nv {nv} "
                             f"takes {nbytes} bytes, more than {SHARED_MAX}")
        self.nv, self.prox = nv, prox  # int, bool
        self.groups = d.contact_group_adr.shape[0] > 0  # bool: MJWarp's group arrays: contact groups may come
        self.warmstart = not (m.opt.disableflags & mjcf_types.DisableBit.WARMSTART)  # bool
        self.threads = int(warps) * LANES  # int: threads a block
        self.jshared = bool(jshared)  # bool
        # Warp kernel: the fused solve
        self.kernel = newton_kernel(nv, int(d.njmax), elliptic, bool(self.warmstart), prox, bool(profile), int(warps),
                                    self.jshared, self.groups)
        self.M_entry = wp.array(dense_M_entries(mjm), dtype=int)  # (nv, nv) int32: each element's CSR entry
        self.prof = wp.zeros((d.nworld, NPROF) if profile else (1, NPROF), dtype=wp.int64)  # (nworld or 1, NPROF) int64
        nw, nj, nc = d.nworld, d.njmax, d.naconmax  # int, int, int
        if prox:
            self.outer = max(1, int(m.opt.prox_outer_iterations))  # int: the prox loop's outer rounds
            self.inner = max(1, int(m.opt.prox_inner_iterations))  # int: rounds a De Saxce step
            self.desaxce = bool(m.opt.prox_desaxce)  # bool
            self.carry = float(m.opt.prox_z_carry) > 0.0  # bool: z carried to the next step
            self.Jaref = wp.zeros((nw, nj), dtype=float)  # (nworld, njmax) float32: J qacc - aref, for _prox_init
            self.prox_z = wp.zeros((nw, nj), dtype=float)  # (nworld, njmax) float32: the anchors z; (1, 1) without prox
            # (nworld, njmax) float32: the rows' stabilization-only reference aref0; (1, 1) without prox
            self.prox_aref0 = wp.zeros((nw, nj), dtype=float)
            self.prox_gamma = wp.zeros(nc, dtype=float)  # (naconmax,) float32: the De Saxce terms; (1,) without prox
            self.prox_rho_i = wp.zeros(nc, dtype=float)  # (naconmax,) float32: the proximal weights rho_i
            self.prox_bdamp = wp.zeros(nc, dtype=float)  # (naconmax,) float32: prox_soft's damping b
            self.prox_done = wp.zeros(nw, dtype=bool)  # (nworld,) bool: no world retired, for _prox_jaref_kernel
        else:
            self.outer = self.inner = 1
            self.desaxce = self.carry = False
            self.prox_z = self.prox_aref0 = wp.zeros((1, 1), dtype=float)
            self.prox_gamma = wp.zeros(1, dtype=float)

    def _prox_init(self, m, d):
        """
        nlsp's solve_prox up to its rounds: qacc's warm start, Jaref against MuJoCo's aref, _prox_init.

        Called by `solve`, under prox. Reads: d.qacc_warmstart and d.qacc_smooth (nworld, nv) float32, d.efc's J and
        aref, the contacts and the anchor cache. Writes: d.qacc and d.qfrc_constraint (nworld, nv) float32, Jaref
        (nworld, njmax) float32, d.efc.D (nworld, njmax_pad) and d.efc.aref (nworld, njmax) float32, prox_z and
        prox_aref0 (nworld, njmax) float32, prox_gamma, prox_rho_i and prox_bdamp (naconmax,) float32.
        """
        wp.launch(nlsp._solve_init_dof(self.warmstart, False), dim=(d.nworld, m.nv),
                  inputs=[d.nefc, d.qacc_warmstart, d.qacc_smooth], outputs=[d.qacc, d.qfrc_constraint])
        wp.launch(nlsp._prox_jaref_kernel(False, m.nv, 50), dim=(d.nworld, d.njmax, 1),
                  inputs=[d.nefc, d.qacc, d.efc.J_rownnz, d.efc.J_rowadr, d.efc.J_colind, d.efc.J, d.efc.aref,
                          self.prox_done],
                  outputs=[self.Jaref])
        wp.launch(
            nlsp._prox_init(bool(m.opt.prox_baumgarte), self.carry, self.desaxce, False, self.groups,
                            bool(m.flg_speculative)),
            dim=d.contact.dist.shape[0],
            inputs=[
                m.opt.timestep, m.opt.impratio_invsqrt, m.opt.prox_rho, m.opt.prox_baumgarte_alpha,
                m.opt.prox_baumgarte_slop, m.opt.prox_z_carry, m.opt.disableflags, m.body_rootid, m.body_weldid,
                m.body_dofnum, m.body_dofadr, m.body_invweight0, m.dof_parentid, m.geom_bodyid, m.geom_rbound,
                d.contact.pos, d.contact.frame, d.contact.efc_address, d.efc.pos, d.efc.margin, d.efc.vel, d.nacon,
                d.prox_zcache_key, d.prox_zcache_val, d.prox_zcache_z, d.prox_zcache_pos, d.prox_zcache_n, d.qvel,
                d.qacc, d.cdof, d.subtree_com, d.contact_group_adr, d.contact_group_num, d.group_member_pos,
                d.group_member_frame, d.group_member_dist, d.group_member_weight, d.group_member_slip,
                d.contact.dim, d.contact.worldid, d.contact.geom, d.contact.friction, d.contact.type, self.Jaref,
                d.contact.solref, d.contact.solimp,
            ],
            outputs=[d.efc.D, d.efc.aref, self.prox_z, self.prox_aref0, self.prox_gamma, self.prox_rho_i,
                     self.prox_bdamp],
        )

    def solve(self, m, d):
        """
        The solve: the contacts' impratio, under prox `_prox_init`, the fused kernel, and with the carry the anchor
        cache filled and sorted for the next step.

        Called by `lodos.cd.driver.SolverDriver.contact_dynamics`. Reads: the rows `make_constraint` assembled in d.efc,
        d.M (nworld, nC) float32, d.qacc_warmstart, d.qacc_smooth and d.qfrc_smooth (nworld, nv) float32, the contacts.
        Writes: d.contact_impratio_invsqrt (naconmax,) float32, d.qacc and d.qfrc_constraint (nworld, nv) float32,
        d.efc.force (nworld, njmax) float32, d.efc.state (nworld, njmax_pad) int32, d.solver_niter (nworld,) int32,
        prof; under prox what `_prox_init` writes, and with the carry d.prox_zcache_key (2 naconmax,) int64,
        d.prox_zcache_val (2 naconmax,) int32, d.prox_zcache_z and d.prox_zcache_pos (naconmax,) wp.vec3 and
        d.prox_zcache_n (1,) int32.
        """
        wp.launch(nlsp._fill_contact_impratio, dim=d.contact.dist.shape[0],
                  inputs=[m.opt.impratio_invsqrt, d.contact.worldid], outputs=[d.contact_impratio_invsqrt])
        if self.prox:
            self._prox_init(m, d)
        z_only = int(self.outer == 1 or not self.desaxce)  # int: 1 where Gamma is not stepped within the step
        wp.launch(
            self.kernel, dim=(d.nworld, self.threads), block_dim=self.threads,
            inputs=[
                m.opt.tolerance, m.opt.ls_tolerance, int(m.opt.iterations), int(m.opt.ls_iterations),
                m.stat.meaninertia, m.opt.timestep, m.opt.prox_tol_z, m.opt.prox_tol_sigma, m.opt.prox_sigma_relax,
                self.outer * self.inner, self.inner, int(self.desaxce), z_only, self.M_entry,
                d.ne, d.nf, d.nefc, d.M, d.qacc_warmstart, d.qacc_smooth, d.qfrc_smooth, d.contact.friction,
                d.contact.efc_address, d.contact_impratio_invsqrt, d.efc.type, d.efc.id, d.efc.J, d.efc.D,
                d.efc.aref, d.efc.frictionloss, d.efc.vel, self.prox_aref0, self.prox_gamma, int(d.njmax),
                m.body_rootid, m.body_weldid, m.body_dofnum, m.body_dofadr, m.dof_parentid, m.geom_bodyid, d.qvel,
                d.cdof, d.subtree_com, d.contact.type, d.contact.geom, d.contact_group_adr, d.contact_group_num,
                d.group_member_pos, d.group_member_frame, d.group_member_weight, d.group_member_slip,
            ],
            outputs=[d.qacc, d.qfrc_constraint, d.efc.force, d.efc.state, d.solver_niter, self.prox_z, self.prof],
        )
        if self.prox and self.carry:
            wp.launch(
                nlsp._prox_cache_fill, dim=d.contact.dist.shape[0],
                inputs=[d.contact.pos, d.contact.frame, d.contact.efc_address, d.nacon, d.contact.dim,
                        d.contact.worldid, d.contact.geom, d.contact.type, self.prox_z],
                outputs=[d.prox_zcache_key, d.prox_zcache_val, d.prox_zcache_z, d.prox_zcache_pos],
            )
            wp.utils.radix_sort_pairs(d.prox_zcache_key, d.prox_zcache_val, d.contact.dist.shape[0])
            wp.copy(d.prox_zcache_n, d.nacon)
