"""
The NLSP solver's Newton system solved in the space of the constraint rows (`row_space`).

MJWarp's Newton (`lodos.cd.solver.nlsp`) forms, every iteration, the dof-space Hessian
`H = M + J^T C J` -- C the rows' curvature: `D` on a quadratic row, the cone's on an elliptic contact
in its middle zone -- and factors it, nv x nv a world, whatever the number of rows. On a model of many
dofs that can be most of the solve, so contact groups, which take rows away, cannot shorten it. Here the
same system is solved through the rows instead.

Write the curvature as `S^T S`, S's rows those of `lodos.cd.fused`: a quadratic row's `sqrt(D) J`, and
an elliptic contact in the cone's middle zone two, `sqrt(dm) (z_0 - mu p)` and
`sqrt(dm (mu^2 - mu N / T)) s` (`lodos.cd.fused`'s docstring) -- one to three efc rows combined, `S = W J`.
With Lodos's factor of each tree's block of M, `M = L L^T` (`L_blk`), the Woodbury identity gives

    H^-1 g = L^-T (w - Z q),   q = W^T z,   w = L^-1 g,   (I + W G W^T) z = W Z^T w,
    g . H^-1 g = w . w - (W Z^T w) . z,

with `Z = L^-1 J^T` and `G = Z^T Z = J M^-1 J^T` formed once a step: the rows do not change within it,
in any of the prox rounds; G only where two rows share a kinematic tree, as M is block diagonal by
tree. Each row of S takes the slot of an efc row -- a quadratic row its own, a middle-zone contact's
two the slots of its normal and first tangent row, an inactive row none (an identity row of the
system) -- so an iteration builds `A = I + W G W^T` over the world's rows, rounded up to blocks of 16,
from a few entries of G each, and factors it by MJWarp's blocked Cholesky: with contact groups a few
dozen rows a world, fewer than a large model's dofs. Everything else -- the gradient, the line search, the
constraint update, the convergence test, the prox rounds -- is `nlsp`'s, which calls this in place of
forming H (`nlsp.solve`'s `newton_solve`).

Capacity: the system has a world's rows, fewer than `rows` (MJWarp's `njmax` plus one, rounded up to
16, at most `ROWS_MAX`: the last row of the system is padding, where MJWarp's augmented factorization
keeps the right-hand side); G, A and the factor take rows^2 floats a world, Z rows x nv, so a large B
wants a tuned `njmax`. A world past it takes `M^-1 g` for that iteration -- a descent direction the line search still
makes safe -- and sets `overflow`, which the step's overflow check reports. Takes Newton, condim 3 at
most on the elliptic cone, and Lodos's per-tree factor of M.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from lodos.cd.solver import types as solver_types
from lodos.cd.solver.block_cholesky import create_blocked_cholesky_augmented_factorize_solve_func
from lodos.mjcf import types as mjcf_types

wp.set_module_options({"enable_backward": False})

TILE = mjcf_types.TILE_SIZE_JTDAJ_DENSE  # MJWarp's Cholesky block edge
ROWS_MAX = 1024  # rows a world at most: the solve's right-hand side lives in shared memory
LANES = 128  # threads a world builds its system with
QUADRATIC = wp.constant(int(solver_types.ConstraintState.QUADRATIC))  # a row in its quadratic zone
CONE = wp.constant(int(solver_types.ConstraintState.CONE))  # an elliptic contact in the cone's middle zone
MINVAL = wp.constant(float(mjcf_types.MINVAL))  # MuJoCo's mjMINVAL
NONE = wp.constant(wp.vec3i(-1, -1, -1))  # a slot of S that combines no efc rows


@wp.func
def _npad(ne: int, nrow: int):
    """The system's size: a world's rows plus one, rounded up to whole blocks -- the last row is padding, where
    MJWarp's augmented factorization keeps the right-hand side; none past the capacity.

    Called by `_system` and the `_cholesky_kernel` kernel.
    In: ne int, the world's efc rows; nrow int, the capacity (`RowSpaceNewton.rows`). Out: int, a multiple of
    TILE, or 0 past the capacity."""
    if ne + 1 > nrow:
        return 0
    return ((ne + 1 + wp.static(TILE) - 1) / wp.static(TILE)) * wp.static(TILE)


@wp.func
def _span(tree_beg: wp.array(dtype=int), trees: wp.vec2i, wide: int, slot: int, nv: int):  # type: ignore
    """
    The dof range of a row's `slot`-th tree (0 or 1); a row over more than two trees (`wide`) spans every dof.

    Called by `_gram_rows` and `_system`.
    In: tree_beg (ntree + 1,) int32; trees wp.vec2i, the row's trees, -1 for none; wide int, 1 for a row over more
    than two; slot int; nv int. Out: int, int: the first dof and one past the last, (0, 0) for none.
    """
    if wide != 0:
        if slot == 0:
            return 0, nv
        return 0, 0
    t = trees[slot]  # int: the tree, -1 for none
    if t < 0:
        return 0, 0
    return tree_beg[t], tree_beg[t + 1]


@wp.func
def _share(a: wp.vec2i, wa: int, b: wp.vec2i, wb: int):
    """
    Whether two rows share a kinematic tree, where M couples them.

    Called by `_gram_rows` and `_system`.
    In: a, b wp.vec2i, the rows' trees (-1 for none); wa, wb int, their `wide` flags. Out: bool.
    """
    if wa != 0 or wb != 0:
        return True
    if a[0] >= 0 and (a[0] == b[0] or a[0] == b[1]):
        return True
    return a[1] >= 0 and (a[1] == b[0] or a[1] == b[1])


@wp.func
def _has(trees: wp.vec2i, wide: int, t: int):
    """
    Whether a row lives on tree t.

    Called by `_lift`.
    In: trees wp.vec2i, the row's trees; wide int; t int. Out: bool.
    """
    return wide != 0 or trees[0] == t or trees[1] == t


def _rows_kernel(sparse: bool):
    """
    Z = L^-1 J^T, one efc row a thread: its trees, then a forward substitution in each.

    Called by `RowSpaceNewton.__init__`.
    In: sparse bool, MJWarp's sparse Jacobian layout (`m.is_sparse`). Out: the kernel, compiled for that layout.
    """
    SPARSE = sparse  # bool: the layout, a compile-time constant

    @wp.kernel(module="unique", enable_backward=False)
    def kernel(
        nv: int,
        tree_beg: wp.array(dtype=int),  # type: ignore
        blk_beg: wp.array(dtype=int),  # type: ignore
        dof_tree: wp.array(dtype=int),  # type: ignore
        L_blk: wp.array2d(dtype=float),  # (nblk, B) # type: ignore
        nefc: wp.array(dtype=int),  # type: ignore
        J_rownnz: wp.array2d(dtype=int),  # type: ignore
        J_rowadr: wp.array2d(dtype=int),  # type: ignore
        J_colind: wp.array3d(dtype=int),  # type: ignore
        J: wp.array3d(dtype=float),  # type: ignore
        Z: wp.array3d(dtype=float),  # (B, rows, nv) out # type: ignore
        rtree: wp.array2d(dtype=wp.vec2i),  # (B, rows) out # type: ignore
        rwide: wp.array2d(dtype=int),  # (B, rows) out # type: ignore
    ):
        """
        Row r of Z = L^-1 J^T in world b, and the trees the row lives on.

        Launch: dim=(B, rows), a thread per world and row slot, by `RowSpaceNewton.prepare`.
        Reads: nefc (B,) int32; J_rownnz and J_rowadr (B, njmax) int32, J_colind (B, 1, njmax_nnz) int32 and
        J (B, 1, njmax_nnz) float32 when sparse, J (B, njmax_pad, nv_pad) float32 when dense; L_blk (nblk, B)
        float32; tree_beg and blk_beg (ntree + 1,) int32; dof_tree (nv,) int32.
        Writes: Z (B, rows, nv) float32, rtree (B, rows) wp.vec2i and rwide (B, rows) int32, at the rows
        r < min(nefc, rows).
        """
        b, r = wp.tid()  # int, int: the world, the efc row
        if r >= wp.min(nefc[b], Z.shape[1]):
            return
        for i in range(nv):  # i: int, a dof
            Z[b, r, i] = 0.0
        t0 = int(-1)  # int: the row's first tree, -1 for none
        t1 = int(-1)  # int: its second
        wide = int(0)  # int: 1 for a row over more than two trees
        if wp.static(SPARSE):
            adr = J_rowadr[b, r]  # int: the row's first nonzero
            for k in range(J_rownnz[b, r]):  # k: int, a nonzero of the row
                dof = J_colind[b, 0, adr + k]  # int: its dof
                Z[b, r, dof] = J[b, 0, adr + k]
                t = dof_tree[dof]  # int: the dof's tree
                if t != t0 and t != t1:
                    if t0 < 0:
                        t0 = t
                    elif t1 < 0:
                        t1 = t
                    else:
                        wide = 1
        else:
            for dof in range(nv):
                v = J[b, r, dof]  # float: the row's entry at the dof
                if v != 0.0:
                    Z[b, r, dof] = v
                    t = dof_tree[dof]
                    if t != t0 and t != t1:
                        if t0 < 0:
                            t0 = t
                        elif t1 < 0:
                            t1 = t
                        else:
                            wide = 1
        rtree[b, r] = wp.vec2i(t0, t1)
        rwide[b, r] = wide
        for t in range(tree_beg.shape[0] - 1):
            if wide == 0 and t != t0 and t != t1:
                continue
            beg = tree_beg[t]  # int: the tree's first dof
            n = tree_beg[t + 1] - beg  # int: its dofs
            o = blk_beg[t]  # int: its block's offset in L_blk
            for i in range(n):
                s = Z[b, r, beg + i]  # float
                for j in range(i):  # j: int
                    s -= L_blk[o + i * n + j, b] * Z[b, r, beg + j]
                Z[b, r, beg + i] = s / L_blk[o + i * n + i, b]

    return kernel


@wp.kernel
def _gram_rows(
    nv: int,
    tree_beg: wp.array(dtype=int),  # type: ignore
    nefc: wp.array(dtype=int),  # type: ignore
    rtree: wp.array2d(dtype=wp.vec2i),  # type: ignore
    rwide: wp.array2d(dtype=int),  # type: ignore
    Z: wp.array3d(dtype=float),  # type: ignore
    G: wp.array3d(dtype=float),  # (B, rows, rows) out # type: ignore
):
    """
    Row r of G = Z^T Z: over r's trees, for the rows that share one; zero for the others.

    Launch: dim=(B, rows), a thread per world and row slot, by `RowSpaceNewton.prepare`.
    Reads: nefc (B,) int32, rtree (B, rows) wp.vec2i, rwide (B, rows) int32, Z (B, rows, nv) float32, tree_beg
    (ntree + 1,) int32. Writes: G (B, rows, rows) float32, its first nefc rows and columns.
    """
    b, r = wp.tid()  # int, int: the world, the row
    n = wp.min(nefc[b], Z.shape[1])  # int: the world's rows, at most the capacity
    if r >= n:
        return
    tr = rtree[b, r]  # wp.vec2i: r's trees
    wr = rwide[b, r]  # int: its wide flag
    for s in range(n):  # s: int, another row
        g = float(0.0)  # float: G[r, s]
        if _share(tr, wr, rtree[b, s], rwide[b, s]):
            for slot in range(2):  # slot: int, r's first or second tree
                beg, end = _span(tree_beg, tr, wr, slot, nv)  # int, int: its dof range
                for dof in range(beg, end):  # dof: int
                    g += Z[b, r, dof] * Z[b, s, dof]
        G[b, r, s] = g


@wp.kernel
def _forward(
    tree_beg: wp.array(dtype=int),  # type: ignore
    blk_beg: wp.array(dtype=int),  # type: ignore
    L_blk: wp.array2d(dtype=float),  # type: ignore
    done: wp.array(dtype=bool),  # type: ignore
    grad: wp.array2d(dtype=float),  # type: ignore
    w: wp.array2d(dtype=float),  # (B, nv) out: L^-1 grad # type: ignore
    ww: wp.array2d(dtype=float),  # (ntree, B) out: each tree's w . w # type: ignore
):
    """
    w = L^-1 grad, a tree a thread.

    Launch: dim=(ntree, B), a thread per tree and world, the worlds fastest as `L_blk` is laid out, by
    `RowSpaceNewton.__call__`.
    Reads: tree_beg and blk_beg (ntree + 1,) int32, L_blk (nblk, B) float32, done (B,) bool, grad (B, nv_pad) float32
    (MJWarp's `ctx.grad`). Writes: w (B, nv) float32, ww (ntree, B) float32.
    """
    t, b = wp.tid()  # int, int: the tree, the world
    if done[b]:
        return
    beg = tree_beg[t]  # int: the tree's first dof
    n = tree_beg[t + 1] - beg  # int: its dofs
    o = blk_beg[t]  # int: its block's offset in L_blk
    acc = float(0.0)  # float: w . w over the tree
    for k in range(n):  # k: int, a dof of the tree
        s = grad[b, beg + k]  # float
        for j in range(k):  # j: int
            s -= L_blk[o + k * n + j, b] * w[b, beg + j]
        s = s / L_blk[o + k * n + k, b]
        w[b, beg + k] = s
        acc += s * s
    ww[t, b] = acc


def _stage_kernel(elliptic: bool):
    """
    Each efc row's slot of S, from the rows' states (MJWarp's last constraint update). A middle-zone contact's
    normal row fills its own slot and its tangents'.

    Called by `RowSpaceNewton.__init__`.
    In: elliptic bool, MJWarp's elliptic cone. Out: the kernel, compiled for that cone.
    """
    ELLIPTIC = elliptic  # bool: the cone, a compile-time constant

    @wp.kernel(module="unique", enable_backward=False)
    def kernel(
        nrow: int,  # rows a world at most
        nefc: wp.array(dtype=int),  # type: ignore
        efc_state: wp.array2d(dtype=int),  # type: ignore
        efc_D: wp.array2d(dtype=float),  # type: ignore
        efc_id: wp.array2d(dtype=int),  # type: ignore
        contact_efc_address: wp.array2d(dtype=int),  # type: ignore
        contact_friction: wp.array(dtype=mjcf_types.vec5),  # type: ignore
        contact_impratio_invsqrt: wp.array(dtype=float),  # type: ignore
        Jaref: wp.array2d(dtype=float),  # type: ignore
        done: wp.array(dtype=bool),  # type: ignore
        srow: wp.array2d(dtype=wp.vec3i),  # (B, rows) out: the efc rows a slot combines # type: ignore
        scoef: wp.array2d(dtype=wp.vec3),  # (B, rows) out: their coefficients # type: ignore
        q: wp.array2d(dtype=float),  # (B, rows) out: zeroed for _scatter # type: ignore
        overflow: wp.array(dtype=int),  # (1,) out # type: ignore
    ):
        """
        Slot i of S in world b: the efc rows it combines and their coefficients; q's slot zeroed.

        Launch: dim=(B, rows), a thread per world and row slot, by `RowSpaceNewton.__call__`.
        Reads: nefc (B,) int32; efc_state (B, njmax_pad) int32 and efc_D (B, njmax_pad) float32; efc_id (B, njmax)
        int32; contact_efc_address (naconmax, nmaxpyramid) int32; contact_friction (naconmax,) vec5;
        contact_impratio_invsqrt (naconmax,) float32; Jaref (B, njmax) float32 (MJWarp's `ctx.Jaref`); done (B,)
        bool. Writes: srow (B, rows) wp.vec3i, scoef (B, rows) wp.vec3 and q (B, rows) float32, at the rows
        r < min(nefc, rows); overflow (1,) int32, set to 1 for a world of `nrow` rows or more.
        """
        b, i = wp.tid()  # int, int: the world, the efc row
        if done[b]:
            return
        ne = nefc[b]  # int: the world's efc rows
        if i >= wp.min(ne, nrow):
            if i == 0 and ne >= nrow:
                overflow[0] = 1
            return
        if ne >= nrow:  # past the capacity: no rows of S, the iteration takes M^-1 grad
            if i == 0:
                overflow[0] = 1
            srow[b, i] = NONE
            q[b, i] = 0.0
            return
        q[b, i] = 0.0
        state = efc_state[b, i]  # int: the row's ConstraintState
        if state == QUADRATIC:
            srow[b, i] = wp.vec3i(i, -1, -1)
            scoef[b, i] = wp.vec3(wp.sqrt(efc_D[b, i]), 0.0, 0.0)
            return
        if wp.static(ELLIPTIC):
            if state == CONE:
                conid = efc_id[b, i]  # int: the row's contact
                if contact_efc_address[conid, 0] != i:
                    return  # a tangent row: its contact's normal row fills its slot
                # the middle zone's curvature as two rows of S (lodos.cd.fused; MJWarp's _update_gradient_JTCJ_dense)
                r1 = contact_efc_address[conid, 1]  # int: the contact's first tangent row
                r2 = contact_efc_address[conid, 2]  # int: its second
                srow[b, r2] = NONE
                fri = contact_friction[conid]  # vec5: the contact's friction
                mu = fri[0] * contact_impratio_invsqrt[conid]  # float: the cone's coefficient, as MJWarp's
                mu2 = mu * mu  # float
                dm = float(0.0)  # float: the middle zone's curvature weight, D / (mu^2 (1 + mu^2))
                if mu2 * (1.0 + mu2) > 0.0:
                    dm = efc_D[b, i] / (mu2 * (1.0 + mu2))
                if dm == 0.0:
                    srow[b, i] = NONE
                    srow[b, r1] = NONE
                    return
                u1 = fri[0] * Jaref[b, r1]  # float: the first tangent's friction-scaled residual
                u2 = fri[1] * Jaref[b, r2]  # float: the second's
                T = wp.max(wp.sqrt(u1 * u1 + u2 * u2), MINVAL)  # float: their norm
                c1 = fri[0] * u1 / T  # float: the first row of S's tangent coefficients, along (u1, u2)
                c2 = fri[1] * u2 / T  # float
                e1 = -fri[0] * u2 / T  # float: the second row's, across it
                e2 = fri[1] * u1 / T  # float
                wa = wp.sqrt(dm) * mu  # float: the first row's scale
                wb = wp.sqrt(wp.max(dm * (mu2 - mu2 * Jaref[b, i] / T), 0.0))  # float: the second row's
                srow[b, i] = wp.vec3i(i, r1, r2)
                scoef[b, i] = wp.vec3(wa, -wa * c1, -wa * c2)
                srow[b, r1] = wp.vec3i(r1, r2, -1)
                scoef[b, r1] = wp.vec3(wb * e1, wb * e2, 0.0)
                return
        srow[b, i] = NONE

    return kernel


@wp.kernel
def _system(
    nv: int,
    nrow: int,
    tree_beg: wp.array(dtype=int),  # type: ignore
    done: wp.array(dtype=bool),  # type: ignore
    nefc: wp.array(dtype=int),  # type: ignore
    rtree: wp.array2d(dtype=wp.vec2i),  # type: ignore
    rwide: wp.array2d(dtype=int),  # type: ignore
    srow: wp.array2d(dtype=wp.vec3i),  # type: ignore
    scoef: wp.array2d(dtype=wp.vec3),  # type: ignore
    Z: wp.array3d(dtype=float),  # type: ignore
    G: wp.array3d(dtype=float),  # type: ignore
    w: wp.array2d(dtype=float),  # type: ignore
    A: wp.array3d(dtype=float),  # (B, rows, rows) out # type: ignore
    rhs: wp.array3d(dtype=float),  # (B, rows, 1) out: W Z^T w # type: ignore
):
    """
    A = I + W G W^T and the right-hand side W Z^T w, LANES threads a world striding over the entries.

    Launch: dim=(B, LANES), LANES threads a world, by `RowSpaceNewton.__call__`.
    Reads: tree_beg (ntree + 1,) int32; done (B,) bool, nefc (B,) int32; rtree (B, rows) wp.vec2i, rwide (B, rows)
    int32; srow (B, rows) wp.vec3i, scoef (B, rows) wp.vec3; Z (B, rows, nv), G (B, rows, rows) and w (B, nv) float32.
    Writes: A (B, rows, rows) and rhs (B, rows, 1) float32, the world's first `_npad` rows and columns.
    """
    b, lane = wp.tid()  # int, int: the world, the lane
    if done[b]:
        return
    ne = nefc[b]  # int: the world's efc rows
    n = _npad(ne, nrow)  # int: the system's size, 0 past the capacity
    for i in range(lane, n, wp.static(LANES)):  # i: int, a slot of S
        v = float(0.0)  # float: rhs[i]
        if i < ne:
            ri = srow[b, i]  # wp.vec3i: the efc rows the slot combines, -1 for none
            if ri[0] >= 0:
                ti = rtree[b, ri[0]]  # wp.vec2i: their trees
                wi = rwide[b, ri[0]]  # int: their wide flag
                ci = scoef[b, i]  # wp.vec3: their coefficients
                for k in range(3):  # k: int, one of the slot's rows
                    r = ri[k]  # int: an efc row, -1 for none
                    if r >= 0:
                        zw = float(0.0)  # float: (Z^T w)[r]
                        for slot in range(2):  # slot: int, the first or second tree
                            beg, end = _span(tree_beg, ti, wi, slot, nv)  # int, int: its dof range
                            for dof in range(beg, end):  # dof: int
                                zw += Z[b, r, dof] * w[b, dof]
                        v += ci[k] * zw
        rhs[b, i, 0] = v
    for e in range(lane, n * n, wp.static(LANES)):  # e: int, an entry of A, row-major
        i = e / n
        j = e - i * n  # int: its column
        a = float(0.0)  # float: A[i, j]
        if i == j:
            a = 1.0
        if i < ne and j < ne:
            ri = srow[b, i]
            rj = srow[b, j]  # wp.vec3i: the efc rows slot j combines
            if ri[0] >= 0 and rj[0] >= 0:
                if _share(rtree[b, ri[0]], rwide[b, ri[0]], rtree[b, rj[0]], rwide[b, rj[0]]):
                    ci = scoef[b, i]
                    cj = scoef[b, j]  # wp.vec3: their coefficients
                    for k in range(3):
                        if ri[k] >= 0:
                            for l in range(3):  # l: int, one of slot j's rows
                                if rj[l] >= 0:
                                    a += ci[k] * cj[l] * G[b, ri[k], rj[l]]
        A[b, i, j] = a


def _cholesky_kernel(rows: int):
    """
    MJWarp's blocked Cholesky factorization and solve (its Newton's augmented one), over a world's own rows.

    Called by `RowSpaceNewton.__init__`.
    In: rows int, the system's capacity (`RowSpaceNewton.rows`). Out: the tiled kernel, compiled for it.
    """
    ROWS = rows  # int: the capacity, a compile-time constant

    @wp.kernel(module="unique", enable_backward=False, module_options={"enable_mathdx_gemm": False})
    def kernel(
        done: wp.array(dtype=bool),  # type: ignore
        nefc: wp.array(dtype=int),  # type: ignore
        A: wp.array3d(dtype=float),  # type: ignore
        rhs: wp.array3d(dtype=float),  # type: ignore
        U: wp.array3d(dtype=float),  # type: ignore
        sol: wp.array3d(dtype=float),  # type: ignore
    ):
        """
        Factors world b's A and solves A z = rhs, over its first `_npad` rows.

        Launch: `wp.launch_tiled`, dim=B, a block of `m.block_dim.update_gradient_cholesky_blocked` threads a world,
        by `RowSpaceNewton.__call__`.
        Reads: done (B,) bool, nefc (B,) int32, A (B, rows, rows) and rhs (B, rows, 1) float32.
        Writes: U (B, rows, rows) float32, the factor; sol (B, rows, 1) float32, z.
        """
        b = wp.tid()  # int: the world
        if done[b]:
            return
        n = _npad(nefc[b], ROWS)  # int: the system's size
        if n == 0:
            return
        wp.static(create_blocked_cholesky_augmented_factorize_solve_func(TILE, ROWS))(A[b], rhs[b], n, U[b], sol[b])

    return kernel


@wp.kernel
def _scatter(
    nrow: int,
    done: wp.array(dtype=bool),  # type: ignore
    nefc: wp.array(dtype=int),  # type: ignore
    srow: wp.array2d(dtype=wp.vec3i),  # type: ignore
    scoef: wp.array2d(dtype=wp.vec3),  # type: ignore
    sol: wp.array3d(dtype=float),  # type: ignore
    q: wp.array2d(dtype=float),  # (B, rows) out: W^T z # type: ignore
):
    """
    q = W^T z, each slot's z onto its efc rows: at most two slots a row, so the sums are exact in any order.

    Launch: dim=(B, rows), a thread per world and slot, by `RowSpaceNewton.__call__`.
    Reads: done (B,) bool, nefc (B,) int32, srow (B, rows) wp.vec3i, scoef (B, rows) wp.vec3, sol (B, rows, 1)
    float32. Writes: q (B, rows) float32, added to atomically (`_stage` zeroed it).
    """
    b, i = wp.tid()  # int, int: the world, the slot
    if done[b]:
        return
    ne = nefc[b]  # int: the world's efc rows
    if ne >= nrow or i >= ne:
        return
    ri = srow[b, i]  # wp.vec3i: the efc rows the slot combines
    if ri[0] < 0:
        return
    z = sol[b, i, 0]  # float: the slot's z
    ci = scoef[b, i]  # wp.vec3: its rows' coefficients
    for k in range(3):  # k: int, one of its rows
        if ri[k] >= 0:
            wp.atomic_add(q, b, ri[k], ci[k] * z)


@wp.kernel
def _lift(
    nrow: int,
    dof_tree: wp.array(dtype=int),  # type: ignore
    done: wp.array(dtype=bool),  # type: ignore
    nefc: wp.array(dtype=int),  # type: ignore
    rtree: wp.array2d(dtype=wp.vec2i),  # type: ignore
    rwide: wp.array2d(dtype=int),  # type: ignore
    Z: wp.array3d(dtype=float),  # type: ignore
    w: wp.array2d(dtype=float),  # type: ignore
    q: wp.array2d(dtype=float),  # type: ignore
    y: wp.array2d(dtype=float),  # (B, nv) out: w - Z q # type: ignore
):
    """
    y = w - Z q, a dof a thread over the rows on its tree.

    Launch: dim=(B, nv), a thread per world and dof, by `RowSpaceNewton.__call__`.
    Reads: dof_tree (nv,) int32; done (B,) bool, nefc (B,) int32; rtree (B, rows) wp.vec2i, rwide (B, rows) int32;
    Z (B, rows, nv), w (B, nv) and q (B, rows) float32. Writes: y (B, nv) float32, w itself past the capacity.
    """
    b, dof = wp.tid()  # int, int: the world, the dof
    if done[b]:
        return
    ne = nefc[b]  # int: the world's efc rows, none past the capacity
    if ne >= nrow:
        ne = 0
    t = dof_tree[dof]  # int: the dof's tree
    yy = w[b, dof]  # float: y at the dof
    for r in range(ne):  # r: int, an efc row
        if _has(rtree[b, r], rwide[b, r], t):
            yy -= Z[b, r, dof] * q[b, r]
    y[b, dof] = yy


@wp.kernel
def _back(
    tree_beg: wp.array(dtype=int),  # type: ignore
    blk_beg: wp.array(dtype=int),  # type: ignore
    L_blk: wp.array2d(dtype=float),  # type: ignore
    done: wp.array(dtype=bool),  # type: ignore
    y: wp.array2d(dtype=float),  # type: ignore
    search: wp.array2d(dtype=float),  # (B, nv) out: -L^-T y # type: ignore
    sd: wp.array2d(dtype=float),  # (ntree, B) out: each tree's search . search # type: ignore
):
    """
    H^-1 grad = L^-T y, a tree a thread: the search direction and its square norm.

    Launch: dim=(ntree, B), a thread per tree and world, by `RowSpaceNewton.__call__`.
    Reads: tree_beg and blk_beg (ntree + 1,) int32, L_blk (nblk, B) float32, done (B,) bool, y (B, nv) float32.
    Writes: y (B, nv) float32, overwritten with L^-T y; search (B, nv) float32 (MJWarp's `ctx.search`);
    sd (ntree, B) float32.
    """
    t, b = wp.tid()  # int, int: the tree, the world
    if done[b]:
        return
    beg = tree_beg[t]  # int: the tree's first dof
    n = tree_beg[t + 1] - beg  # int: its dofs
    o = blk_beg[t]  # int: its block's offset in L_blk
    acc = float(0.0)  # float: search . search over the tree
    for i in range(n - 1, -1, -1):  # i: int, a dof of the tree, the last first
        s = y[b, beg + i]  # float
        for j in range(i + 1, n):  # j: int
            s -= L_blk[o + j * n + i, b] * y[b, beg + j]
        s = s / L_blk[o + i * n + i, b]
        y[b, beg + i] = s
        search[b, beg + i] = -s
        acc += s * s
    sd[t, b] = acc


@wp.kernel
def _finish(
    ntree: int,
    nrow: int,
    done: wp.array(dtype=bool),  # type: ignore
    nefc: wp.array(dtype=int),  # type: ignore
    srow: wp.array2d(dtype=wp.vec3i),  # type: ignore
    rhs: wp.array3d(dtype=float),  # type: ignore
    sol: wp.array3d(dtype=float),  # type: ignore
    ww: wp.array2d(dtype=float),  # type: ignore
    sd: wp.array2d(dtype=float),  # type: ignore
    search_dot: wp.array(dtype=float),  # (B,) out # type: ignore
    newton_decrement: wp.array(dtype=float),  # (B,) out # type: ignore
):
    """
    The sums, in a fixed order: search . search, and the Newton decrement w . w - (W Z^T w) . z.

    Launch: dim=B, a thread per world, by `RowSpaceNewton.__call__`.
    Reads: done (B,) bool, nefc (B,) int32, srow (B, rows) wp.vec3i, rhs and sol (B, rows, 1) float32, ww and sd
    (ntree, B) float32. Writes: search_dot and newton_decrement (B,) float32 (MJWarp's `ctx.search_dot` and
    `ctx.newton_decrement`).
    """
    b = wp.tid()  # int: the world
    if done[b]:
        return
    a = float(0.0)  # float: w . w, then the Newton decrement
    s = float(0.0)  # float: search . search
    for t in range(ntree):  # t: int, a tree
        a += ww[t, b]
        s += sd[t, b]
    ne = nefc[b]  # int: the world's efc rows
    if ne < nrow:
        for i in range(ne):  # i: int, a slot of S
            if srow[b, i][0] >= 0:
                a -= rhs[b, i, 0] * sol[b, i, 0]
    search_dot[b] = s
    newton_decrement[b] = a


class RowSpaceNewton:
    """
    The Newton system of MJWarp's solve in the rows' space, for `lodos.cd.solver.nlsp.solve(m, d, newton_solve=...)`:
    `prepare` once the rows are assembled (after `make_constraint`), then MJWarp calls the object every time it
    needs a search direction. Capturable: every array is allocated here.

    mjm, m, d: MuJoCo's model and MJWarp's; mrbd, drbd: Lodos's rigid bodies, whose `L_blk` factors the M that
    MJWarp's solve reads (`lodos.cd.driver.gather_M`).

    Built by `lodos.cd.driver.SolverDriver.__init__` (`row_space`). B below is `d.nworld`, rows `self.rows`.
    """

    def __init__(self, mjm, m, d, mrbd, drbd):
        """
        Every array of the system for `d`'s worlds, and its kernels.

        Called by `lodos.cd.driver.SolverDriver.__init__`.
        In: mjm (MjModel); m, d, MJWarp's model and data (d.nworld, d.njmax); mrbd (`RbdModel`): tree_beg and
        blk_beg (ntree + 1,) int32; drbd (`RbdData`), kept for its L_blk (nblk, B) float32.
        Raises ValueError without the Newton solver, under sleeping, on an elliptic contact of condim above 3, or
        where Lodos's trees do not cover the dofs.
        """
        if m.opt.solver != mjcf_types.SolverType.NEWTON:
            raise ValueError("row_space solves the Newton solver's system: it needs the Newton solver")
        if m.opt.enableflags & mjcf_types.EnableBit.SLEEP:
            raise ValueError("row_space does not support sleeping")
        self.elliptic = m.opt.cone == mjcf_types.ConeType.ELLIPTIC  # bool: the elliptic cone
        if self.elliptic and int(m.nmaxcondim) > 3:
            raise ValueError("row_space takes elliptic contacts of condim 3 at most")
        B, nv, njmax = int(d.nworld), int(mjm.nv), int(d.njmax)  # int: worlds, dofs, efc rows a world at most
        tree_beg = mrbd.tree_beg.numpy().astype(np.int32)  # (ntree + 1,) int32: each tree's dof range
        if nv < 1 or int(tree_beg[-1]) != nv:
            raise ValueError("row_space: Lodos's trees do not cover the model's dofs")
        dof_tree = np.zeros(nv, np.int32)  # (nv,) int32: each dof's tree
        for t in range(len(tree_beg) - 1):  # t: int, a tree
            dof_tree[tree_beg[t]:tree_beg[t + 1]] = t
        self.rows = min(-(-(njmax + 1) // TILE) * TILE, ROWS_MAX)  # int: njmax + 1 rounded up to TILE, ROWS_MAX at most
        R = self.rows  # int
        # int: worlds, dofs, trees, MJWarp's efc rows a world at most
        self.B, self.nv, self.ntree, self.njmax = B, nv, int(len(tree_beg) - 1), njmax
        self.sparse = bool(m.is_sparse)  # bool: MJWarp's sparse Jacobian
        self.tree_beg, self.blk_beg = mrbd.tree_beg, mrbd.blk_beg  # (ntree + 1,) int32 each: mrbd's
        self.dof_tree = wp.array(dof_tree, dtype=int)  # (nv,) int32: each dof's tree
        self.drbd = drbd  # RbdData: L_blk (nblk, B) float32, the step's factor of M
        # A world past `rows` rows uses none of them (it overflows), so Z and G stop there too.
        self.Z = wp.zeros((B, R, nv), dtype=float)  # (B, rows, nv) float32: L^-1 J^T, by row
        self.G = wp.zeros((B, R, R), dtype=float)  # (B, rows, rows) float32: Z^T Z
        self.rtree = wp.zeros((B, R), dtype=wp.vec2i)  # (B, rows) wp.vec2i: each row's trees, -1 for none
        self.rwide = wp.zeros((B, R), dtype=int)  # (B, rows) int32: 1 for a row over more than two trees
        self.srow = wp.zeros((B, R), dtype=wp.vec3i)  # (B, rows) wp.vec3i: the efc rows each slot of S combines
        self.scoef = wp.zeros((B, R), dtype=wp.vec3)  # (B, rows) wp.vec3: their coefficients, the slot's row of W
        self.q = wp.zeros((B, R), dtype=float)  # (B, rows) float32: W^T z
        self.ww = wp.zeros((self.ntree, B), dtype=float)  # (ntree, B) float32: each tree's w . w
        self.sd = wp.zeros((self.ntree, B), dtype=float)  # (ntree, B) float32: each tree's search . search
        self.w = wp.zeros((B, nv), dtype=float)  # (B, nv) float32: L^-1 grad
        self.y = wp.zeros((B, nv), dtype=float)  # (B, nv) float32: w - Z q, then L^-T of it
        self.A = wp.zeros((B, R, R), dtype=float)  # (B, rows, rows) float32: I + W G W^T
        self.U = wp.zeros((B, R, R), dtype=float)  # (B, rows, rows) float32: A's factor
        self.rhs = wp.zeros((B, R, 1), dtype=float)  # (B, rows, 1) float32: W Z^T w
        self.sol = wp.zeros((B, R, 1), dtype=float)  # (B, rows, 1) float32: z
        self.overflow = wp.zeros(1, dtype=int)  # (1,) int32: 1 once a world had `rows` rows or more
        self._rows = _rows_kernel(self.sparse)  # wp.Kernel: Z and the rows' trees
        self._stage = _stage_kernel(self.elliptic)  # wp.Kernel: the slots of S
        self._cholesky = _cholesky_kernel(R)  # wp.Kernel: A's factor and solve
        self.block_dim = int(m.block_dim.update_gradient_cholesky_blocked)  # int: the Cholesky's threads a world

    def prepare(self, d):
        """
        Z = L^-1 J^T and G = Z^T Z for this step's rows (after `make_constraint`; L the step's factor of M).

        Called by `lodos.cd.driver.SolverDriver.contact_dynamics`, every step, before `lodos.cd.solver.nlsp.solve`.
        In: d.nefc (B,) int32; d.efc.J_rownnz and J_rowadr (B, njmax) int32, d.efc.J_colind (B, 1, njmax_nnz) int32
        and d.efc.J (B, 1, njmax_nnz) float32 when sparse, d.efc.J (B, njmax_pad, nv_pad) float32 when dense;
        drbd.L_blk (nblk, B) float32. Out: self.Z, rtree, rwide and G.
        """
        wp.launch(self._rows, dim=(self.B, self.rows),
                  inputs=[self.nv, self.tree_beg, self.blk_beg, self.dof_tree, self.drbd.L_blk, d.nefc, d.efc.J_rownnz,
                          d.efc.J_rowadr, d.efc.J_colind, d.efc.J],
                  outputs=[self.Z, self.rtree, self.rwide])
        wp.launch(_gram_rows, dim=(self.B, self.rows),
                  inputs=[self.nv, self.tree_beg, d.nefc, self.rtree, self.rwide, self.Z], outputs=[self.G])

    def __call__(self, m, d, ctx):
        """
        ctx.search = -H^-1 ctx.grad, ctx.search_dot, ctx.newton_decrement for every world not done.

        Called by `lodos.cd.solver.nlsp._update_gradient`, as `ctx.newton_solve`, every Newton iteration.
        In: ctx.grad (B, nv_pad) float32, ctx.Jaref (B, njmax) float32, ctx.done (B,) bool; d.nefc (B,) int32,
        d.efc.state (B, njmax_pad) int32, d.efc.D (B, njmax_pad) float32, d.efc.id (B, njmax) int32,
        d.contact.efc_address (naconmax, nmaxpyramid) int32, d.contact.friction (naconmax,) vec5,
        d.contact_impratio_invsqrt (naconmax,) float32. Out: ctx.search (B, nv) float32, ctx.search_dot and
        ctx.newton_decrement (B,) float32; self.overflow (1,) int32, set where a world is past the capacity.
        """
        B, R, nv = self.B, self.rows, self.nv  # int: worlds, rows, dofs
        L = self.drbd.L_blk  # (nblk, B) float32: the per-tree factor of M
        wp.launch(_forward, dim=(self.ntree, B), inputs=[self.tree_beg, self.blk_beg, L, ctx.done, ctx.grad],
                  outputs=[self.w, self.ww])
        wp.launch(self._stage, dim=(B, R),
                  inputs=[R, d.nefc, d.efc.state, d.efc.D, d.efc.id, d.contact.efc_address, d.contact.friction,
                          d.contact_impratio_invsqrt, ctx.Jaref, ctx.done],
                  outputs=[self.srow, self.scoef, self.q, self.overflow])
        wp.launch(_system, dim=(B, LANES),
                  inputs=[nv, R, self.tree_beg, ctx.done, d.nefc, self.rtree, self.rwide, self.srow, self.scoef, self.Z,
                          self.G, self.w],
                  outputs=[self.A, self.rhs])
        wp.launch_tiled(self._cholesky, dim=B, inputs=[ctx.done, d.nefc, self.A, self.rhs],
                        outputs=[self.U, self.sol], block_dim=self.block_dim)
        wp.launch(_scatter, dim=(B, R), inputs=[R, ctx.done, d.nefc, self.srow, self.scoef, self.sol],
                  outputs=[self.q])
        wp.launch(_lift, dim=(B, nv),
                  inputs=[R, self.dof_tree, ctx.done, d.nefc, self.rtree, self.rwide, self.Z, self.w, self.q],
                  outputs=[self.y])
        wp.launch(_back, dim=(self.ntree, B), inputs=[self.tree_beg, self.blk_beg, L, ctx.done, self.y],
                  outputs=[ctx.search, self.sd])
        wp.launch(_finish, dim=B,
                  inputs=[self.ntree, R, ctx.done, d.nefc, self.srow, self.rhs, self.sol, self.ww, self.sd],
                  outputs=[ctx.search_dot, ctx.newton_decrement])
