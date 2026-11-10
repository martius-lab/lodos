"""
Newton's contact reduction, on Lodos's contacts, for MJWarp's solvers.

MJWarp's Newton solver and the prox solver make constraint rows of every contact they are handed,
and a Lodos pair holds one contact per penetrating vertex: thousands a world between
closely fitting, finely meshed surfaces, where a few hundred placed well hold the same
bodies. Newton's collision pipeline reduces its mesh contacts before its solvers see
them (`newton/_src/geometry/contact_reduction.py`). This is that reduction, per (world,
segment) -- a segment being one direction of a Lodos pair -- instead of per shape pair:

    per normal bin (20, an icosahedron's faces, as Newton's; `lodos.cd.bins`):
        the deepest contact, and the most extreme one in each of 6 directions 60
        degrees apart in the bin's plane -- the corners of that patch
    per cell of a grid over the vertex geometry's bounding box (at most 100):
        the deepest contact, for cover inside a large patch

At most 240 contacts per segment, each one Lodos found; `lodos.cd.driver.export_lodos`
skips the others. As in Newton, only contacts nearer than `BETA` (0.1 mm, Newton's
`BETA_THRESHOLD`) compete for the extremes, the outline of what touches: a `toa` pair's
speculative contacts (`lodos.col.sdf`, at a positive gap) stay out of it and compete only
for the depth slots, the nearest first. Every other Lodos contact penetrates and competes
for all of them, as before the threshold existed.

**Deterministic.** A slot's winner is the atomic max of one 64-bit word -- the score,
quantized at 0.1 um, in the high half and the contact's index inverted in the low half
-- so it does not depend on thread order, and ties go to the lower index. Newton packs
its `float_flip(score)` the same way.

**Where the slots live.** A world's active segments, those with contacts this step,
take rows `0 .. rows-1` of a `(B, rows, 240)` table in segment order. A world with more
active segments than `rows` exports the rest unreduced: no contact is lost, the solver
just sees more of them. A grouped segment (`lodos.cd.groups`) is reduced like any other:
its groups are made of the contacts kept.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from lodos.cd.bins import bin_normals, bin_tangents

wp.set_module_options({"enable_backward": False})

NBIN = 20  # normal bins, Newton's icosahedron
NDIR = 6  # scan directions per bin, Newton's NUM_SPATIAL_DIRECTIONS
NVOX = 100  # voxel slots per segment, Newton's NUM_VOXEL_DEPTH_SLOTS
NSLOT = NBIN * (NDIR + 1) + NVOX  # slots per reduced segment, 240: each bin's 6 corners and deepest, then the voxels
QUANT = 1.0e7  # scores per metre: 0.1 um, over +-214 m
BETA = 1.0e-4  # m: the largest gap that competes for a bin's extremes, Newton's BETA_THRESHOLD
EMPTY_VALUE = -(2**63)  # an empty slot: below every packed word
EMPTY = wp.constant(wp.int64(EMPTY_VALUE))  # EMPTY_VALUE, for the kernels


@wp.struct
class ReduceModel:
    """
    What the reduction does not change from step to step: the slot rows a world has, the normal bins' frames, and
    each vertex geometry's OBB and the voxel grid over it.

    Built by `make_reduce_context`.
    """

    rows: int  # segments reduced per world
    bin_n: wp.array(dtype=wp.vec3)  # (NBIN,)
    bin_t1: wp.array(dtype=wp.vec3)  # (NBIN,) the plane each bin's corners are found in
    bin_t2: wp.array(dtype=wp.vec3)  # (NBIN,)
    seg_g_v: wp.array(dtype=int)  # (nSEG,) the segment's vertex geometry
    g_obb_c: wp.array(dtype=wp.vec3)  # (nB,) OBB centre, body coordinates
    g_obb_A: wp.array(dtype=wp.mat33)  # (nB,) OBB axes as rows
    g_obb_h: wp.array(dtype=wp.vec3)  # (nB,) OBB half-extents
    g_vox: wp.array(dtype=wp.vec3i)  # (nB,) voxel grid over the OBB


@wp.struct
class ReduceData:
    """
    A step's reduction: each (world, segment)'s slot row, the slots' packed winners, and the contacts kept.

    Built by `make_reduce_context`; written by `reduce`.
    """

    seg_row: wp.array2d(dtype=int)  # (B, nSEG) slot row, -1 for one exported unreduced
    slot: wp.array3d(dtype=wp.int64)  # (B, rows, NSLOT) packed winners
    keep: wp.array(dtype=wp.uint8)  # (nCa3_max,) 1 for a contact to export


def voxel_resolution(h, nvox=NVOX):
    """
    Cells per OBB axis, in proportion to its extent, at most `nvox` in all.

    Called by `make_reduce_context`, per Lodos geometry.
    In: h (3,) float, the OBB's half-extents (m); nvox int. Out: (3,) int64, cells per axis, at least 1 each.
    """
    h = np.maximum(np.asarray(h, dtype=float), 0.0)
    live = h > 1e-9  # (3,) bool: the axes with an extent
    dims = np.ones(3, dtype=int)  # (3,) int64: cells per axis
    if not live.any():
        return dims
    scale = (nvox / np.prod(h[live])) ** (1.0 / live.sum())  # float: cells per metre of half-extent
    dims[live] = np.maximum(1, np.floor(h[live] * scale)).astype(int)
    while dims.prod() > nvox:
        dims[np.argmax(dims)] -= 1
    grew = True  # bool: an axis took another cell this pass
    while grew:  # fill what is left, longest axis first
        grew = False
        for a in np.argsort(-h):  # a: int, an axis
            if live[a] and dims.prod() // dims[a] * (dims[a] + 1) <= nvox:
                dims[a] += 1
                grew = True
    return dims


def make_reduce_context(mcol, dcol, rows=None):
    """
    The reduction's model and data for a Lodos collision context.

    Called by `lodos.cd.driver.SolverDriver.__init__`.
    In: mcol (`CollisionModel`): B, nSEG, nB, seg_g_v (nSEG,) int32, g_obb_c (nB,) wp.vec3, g_obb_A (nB,) wp.mat33,
    g_obb_h (nB,) wp.vec3; dcol (`CollisionData`): nCa3_max; rows int, the segments reduced per world, or None for
    `min(nSEG, 16)`. Out: (`ReduceModel`, `ReduceData`), the model sharing mcol's arrays.
    """
    B, nSEG, nB = int(mcol.B), int(mcol.nSEG), int(mcol.nB)  # int: worlds, segments, Lodos geometries
    rows = min(nSEG, 16) if rows is None else int(rows)
    d = bin_normals(NBIN)  # (NBIN, 3) float64: the bins' unit normals
    t1, t2 = bin_tangents(d)  # (NBIN, 3), (NBIN, 3) float64: each bin's plane
    rm = ReduceModel()  # ReduceModel
    rm.rows = max(1, rows)
    rm.bin_n = wp.array(d.astype(np.float32), dtype=wp.vec3)
    rm.bin_t1 = wp.array(t1.astype(np.float32), dtype=wp.vec3)
    rm.bin_t2 = wp.array(t2.astype(np.float32), dtype=wp.vec3)
    rm.seg_g_v = mcol.seg_g_v
    rm.g_obb_c, rm.g_obb_A, rm.g_obb_h = mcol.g_obb_c, mcol.g_obb_A, mcol.g_obb_h
    h = mcol.g_obb_h.numpy().reshape(nB, 3)  # (nB, 3) float32: the OBBs' half-extents
    rm.g_vox = wp.array(np.array([voxel_resolution(h[g]) for g in range(nB)], dtype=np.int32), dtype=wp.vec3i)
    rd = ReduceData()  # ReduceData
    rd.seg_row = wp.zeros((B, nSEG), dtype=int)
    rd.slot = wp.zeros((B, rm.rows, NSLOT), dtype=wp.int64)
    rd.keep = wp.zeros(int(dcol.nCa3_max), dtype=wp.uint8)
    return rm, rd


@wp.func
def _pack(score: float, i: int):
    """
    The word a slot's winner is the atomic max of: `score` quantized at 1 / QUANT m in the high half, contact `i`
    inverted in the low half, so the higher score wins and a tie goes to the lower index.

    Called by `score_contacts`.
    In: score float (m); i int, the contact. Out: wp.int64.
    """
    q = wp.clamp(score * QUANT, -2147483647.0, 2147483647.0)  # float: the score in units of 0.1 um, clamped to int32
    return wp.int64(int(q)) * wp.int64(4294967296) + wp.int64(2147483647 - i)


@wp.kernel
def assign_rows(
    rm: ReduceModel,  # type: ignore
    rd: ReduceData,  # type: ignore
    idul: wp.array2d(dtype=int),  # (B, nSEG + 1) each segment's first contact # type: ignore
):
    """
    Each segment's slot row in a world: the active segments, those with contacts this step, take rows 0 to
    `rm.rows - 1` in segment order; the others, and the active ones past the rows, -1 (exported unreduced).

    Launch: dim=(B,), a thread per world, by `reduce`.
    Reads: idul (B, nSEG + 1) int32. Writes: rd.seg_row (B, nSEG) int32.
    """
    b = wp.tid()  # int: the world
    r = int(0)  # int: the active segments so far
    for sg in range(rd.seg_row.shape[1]):  # sg: int, a segment
        row = int(-1)  # int: the segment's slot row
        if idul[b, sg + 1] > idul[b, sg]:
            if r < rm.rows:
                row = r
            r += 1
        rd.seg_row[b, sg] = row


# Contacts per scoring thread. Lodos lists a segment's contacts contiguously and its
# vertices in Hilbert order, so neighbours in the list mostly share a bin and a voxel: a
# thread walking RUN of them keeps each slot's best word in registers and bids only when
# its (world, row, bin) or voxel changes. The winners are maxima, which do not depend on the
# order bids arrive in, so the kept set does not depend on RUN.
RUN = 16
_slots_t = wp.types.vector(length=NDIR + 1, dtype=wp.int64)  # a bin's slots, merged over a run


@wp.func
def _bin_of(rm: ReduceModel, n: wp.vec3):  # type: ignore
    """
    The normal bin of the unit normal `n`: the bin whose direction is closest to it.

    Called by `score_contacts`.
    In: n wp.vec3. Out: int, 0 to NBIN - 1.
    """
    kn = int(0)  # int: the closest bin so far
    best = wp.dot(rm.bin_n[0], n)  # float: its direction's dot with n
    for k in range(1, NBIN):  # k: int, a bin
        dk = wp.dot(rm.bin_n[k], n)  # float
        if dk > best:
            best = dk
            kn = k
    return kn


@wp.func
def _voxel_of(rm: ReduceModel, g: int, x: wp.transformf, p: wp.vec3):  # type: ignore
    """
    The cell of geometry `g`'s voxel grid that the world point `p` falls in, clamped to the grid, as a flat index.

    Called by `score_contacts`.
    In: g int, the vertex geometry; x wp.transformf, its world pose; p wp.vec3, the point. Out: int, 0 to the
    product of `rm.g_vox[g]` minus 1.
    """
    q = rm.g_obb_A[g] * (wp.transform_point(wp.transform_inverse(x), p) - rm.g_obb_c[g])  # wp.vec3: p in the OBB's axes
    h = rm.g_obb_h[g]  # wp.vec3: the OBB's half-extents
    res = rm.g_vox[g]  # wp.vec3i: cells per axis
    cell = wp.vec3i(0)  # wp.vec3i: p's cell
    for k in range(3):  # k: int, an axis
        t = float(0.5)  # float: p's place along the axis, 0 to 1
        if h[k] > 0.0:
            t = (q[k] + h[k]) / (2.0 * h[k])
        cell[k] = wp.clamp(int(t * float(res[k])), 0, res[k] - 1)
    return (cell[0] * res[1] + cell[1]) * res[2] + cell[2]


@wp.kernel
def score_contacts(
    rm: ReduceModel,  # type: ignore
    rd: ReduceData,  # type: ignore
    ct: wp.array(dtype=int),  # type: ignore
    b_nCa3: wp.array(dtype=int),  # type: ignore
    seg_nCa3: wp.array(dtype=int),  # type: ignore
    p_nCa3: wp.array(dtype=wp.vec3),  # type: ignore
    n_nCa3: wp.array(dtype=wp.vec3),  # type: ignore
    sdf_nCa3: wp.array(dtype=float),  # type: ignore
    T_w: wp.array2d(dtype=wp.transformf),  # (B, nB) geometry poses # type: ignore
):
    """
    A run of contacts' bids for their bins' corners (those nearer than `BETA`) and depth, and their voxels' depth.

    Launch: dim=(ceil(nCa3_max / RUN),), a thread per run of RUN contacts, by `reduce`.
    Reads: ct (5,) int32 (ct[2] the contacts), b_nCa3 and seg_nCa3 (nCa3_max,) int32, p_nCa3 and n_nCa3
    (nCa3_max,) wp.vec3, sdf_nCa3 (nCa3_max,) float32, T_w (B, nB) wp.transformf, rd.seg_row (B, nSEG) int32;
    rm's bins and voxel grids. Writes: rd.slot (B, rows, NSLOT) int64, by atomic max; rd.keep (nCa3_max,) uint8,
    1 for each contact of a segment exported unreduced.
    """
    t = wp.tid()  # int: the run
    n = ct[2]  # int: the contacts this step
    i0 = t * RUN  # int: the run's first contact
    if i0 >= n:
        return
    i1 = wp.min(i0 + RUN, n)  # int: one past its last

    # The run being merged: its (world, row, bin) and its best word per slot; and the voxel's.
    cb = int(-1)  # int: the merged bin's world
    crow = int(-1)  # int: its slot row
    ckn = int(-1)  # int: the bin, -1 before the first
    best = _slots_t(wp.int64(EMPTY))  # _slots_t: the bin's best word per slot
    vb = int(-1)  # int: the merged voxel's world
    vrow = int(-1)  # int: its slot row
    vvox = int(-1)  # int: the voxel, -1 before the first
    vbest = wp.int64(EMPTY)  # wp.int64: the voxel's best word

    for i in range(i0, i1):  # i: int, a contact
        b = b_nCa3[i]  # int: its world
        sg = seg_nCa3[i]  # int: its segment
        row = rd.seg_row[b, sg]  # int: the segment's slot row, -1 if unreduced
        if row < 0:
            rd.keep[i] = wp.uint8(1)
            continue
        p = p_nCa3[i]  # wp.vec3: the contact point
        depth = -sdf_nCa3[i]  # float: its penetration (m), > 0 but for a speculative contact's negated gap
        kn = _bin_of(rm, n_nCa3[i])  # int: its normal bin
        if b != cb or row != crow or kn != ckn:
            if ckn >= 0:
                for j in range(NDIR + 1):  # j: int, a slot of the bin
                    if best[j] != EMPTY:
                        wp.atomic_max(rd.slot, cb, crow, ckn * (NDIR + 1) + j, best[j])
            cb = b
            crow = row
            ckn = kn
            for j in range(NDIR + 1):
                best[j] = EMPTY
        if sdf_nCa3[i] < BETA:  # a speculative contact does not outline what touches
            u = wp.dot(p, rm.bin_t1[kn])  # float: p's coordinates in the bin's plane (m)
            v = wp.dot(p, rm.bin_t2[kn])  # float
            for j in range(NDIR):
                a = float(j) * (2.0 * wp.pi / float(NDIR))  # float: the scan direction's angle (rad)
                best[j] = wp.max(best[j], _pack(u * wp.cos(a) + v * wp.sin(a), i))
        packed_depth = _pack(depth, i)  # wp.int64: the contact's depth bid
        best[NDIR] = wp.max(best[NDIR], packed_depth)

        vox = _voxel_of(rm, rm.seg_g_v[sg], T_w[b, rm.seg_g_v[sg]], p)  # int: its voxel
        if b != vb or row != vrow or vox != vvox:
            if vvox >= 0:
                wp.atomic_max(rd.slot, vb, vrow, NBIN * (NDIR + 1) + vvox, vbest)
            vb = b
            vrow = row
            vvox = vox
            vbest = wp.int64(EMPTY)
        vbest = wp.max(vbest, packed_depth)

    if ckn >= 0:
        for j in range(NDIR + 1):
            if best[j] != EMPTY:
                wp.atomic_max(rd.slot, cb, crow, ckn * (NDIR + 1) + j, best[j])
    if vvox >= 0:
        wp.atomic_max(rd.slot, vb, vrow, NBIN * (NDIR + 1) + vvox, vbest)


@wp.kernel
def mark_winners(rd: ReduceData):  # type: ignore
    """
    Marks each filled slot's winner kept: the contact whose index is inverted in the low half of its word.

    Launch: dim=(B, rows, NSLOT), a thread per world, slot row and slot, by `reduce`.
    Reads: rd.slot (B, rows, NSLOT) int64. Writes: rd.keep (nCa3_max,) uint8.
    """
    b, r, s = wp.tid()  # int, int, int: the world, the slot row, the slot
    v = rd.slot[b, r, s]  # wp.int64: the slot's packed winner
    if v == EMPTY:
        return
    i = 2147483647 - int(v & wp.int64(4294967295))  # int: the winning contact
    rd.keep[i] = wp.uint8(1)


def reduce(rm, rd, dcol):  # type: ignore
    """
    Marks this step's contacts to export in `rd.keep`. Capturable.

    Called by `lodos.cd.driver.SolverDriver.contact_dynamics`, every step.
    In: rm, rd (`make_reduce_context`); dcol (`CollisionData`): idul_B_nSEG_to_nCa3 (B, nSEG + 1) int32, ct (5,)
    int32, b_nCa3 and seg_nCa3 (nCa3_max,) int32, p_nCa3 and n_nCa3 (nCa3_max,) wp.vec3, sdf_nCa3 (nCa3_max,)
    float32, T_w (B, nB) wp.transformf. Out: rd.keep (nCa3_max,) uint8, 1 for a contact to export; rd.seg_row
    (B, nSEG) int32 and rd.slot (B, rows, NSLOT) int64.
    """
    B = int(rd.seg_row.shape[0])  # int: worlds
    wp.launch(assign_rows, dim=(B,), inputs=(rm, rd, dcol.idul_B_nSEG_to_nCa3))
    rd.slot.fill_(EMPTY_VALUE)
    rd.keep.zero_()
    wp.launch(score_contacts, dim=((int(dcol.nCa3_max) + RUN - 1) // RUN,),
              inputs=(rm, rd, dcol.ct, dcol.b_nCa3, dcol.seg_nCa3, dcol.p_nCa3, dcol.n_nCa3, dcol.sdf_nCa3, dcol.T_w))
    wp.launch(mark_winners, dim=(B, rm.rows, NSLOT), inputs=(rd,))
