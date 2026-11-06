"""
Device-side stream compaction for the collision pipeline.

Every routine here runs entirely on the GPU: survivor counts live in
`dcol.ct` and are never read back to the host, so no stage stalls the pipeline
and all launch dimensions are compile-time constants. That is what makes the
Warp section CUDA-graph capturable.

Stage 1 (bounding-box) is special: a segment's *tiles* are contiguous, so its
offsets follow from a scan over per-generator tile counts -- there is never a
dense (B, nC) array. An nCa1 index is a primitive slot of a surviving tile, and
`resolve_nCa1_tiled` recovers (batch, contact) from it with one indexed load
shared by the tile's threads.

Stages 2 and 3 share the generic flag -> exclusive-scan -> scatter pattern,
which preserves the segment-contiguous ordering the `idul_B_nSEG_to_nCa*` maps
depend on.

Activity is decided per *pair* -- that is what the OBB filter writes into
`cpr_active` -- so all of a pair's segments switch on and off together.
"""

import warp as wp

from lodos.utils import scan as lscan
from lodos.col.sdf.types import CollisionModel, CollisionData, PARTITION_BLK


#######################################################################################
# Stage T: tiles -> stage-1 primitive slots
#######################################################################################
#
# The stage-1 index space is the primitive slots of the *surviving tiles*: slot
# `i` belongs to compacted tile `i / tile_K` where every segment shares a tile
# size, and to the tile `tile_off` says owns it where they differ.
#
# The tiles worth testing are produced by the *generator* level above them
# (`broadphase.gen_weights`, and `CollisionModel.gen_seg` for what a generator
# is). Generators are enumerated (batch, generator) and each expands to a run of
# consecutive tiles of one segment, so the candidate list stays ordered
# (batch, segment, tile) whether or not the segment has super-tiles -- which is
# why the two levels need no merge between them. That ordering is what keeps a
# segment's contacts one run, which `lodos.cd` requires, and what lets
# `build_ul_nCa1_tiled` recover a segment's stage-1 start as the scan evaluated
# at its first generator.
#
# The generators are also what the tile survivors are *scanned over*: they are a
# property of the scene where the candidate list is a capacity. `compact_nTa`.


@wp.kernel
def set_ct_nTb(mcol: CollisionModel, dcol: CollisionData):
    """
    The tile-candidate count: the end of the generator scan, clamped to `nTb_max`, with its peak and overflow flag.

    Launch: dim=1, a single thread, by `compact_nTb`.
    Reads: dcol.gen_s and gen_w (B*nG_world,) int32. Writes: dcol.ct[4] and ct_raw[4], ct_peak[4] (raised to),
    overflow[4] (set on overflow).
    """
    last = mcol.B * mcol.nG_world - 1  # int: the last (world, generator) slot
    total = dcol.gen_s[last] + dcol.gen_w[last]  # int: the tile candidates, unclamped

    dcol.ct_raw[4] = total
    wp.atomic_max(dcol.ct_peak, 4, total)
    dcol.ct[4] = wp.min(total, dcol.nTb_max)
    if total > dcol.nTb_max:
        dcol.overflow[4] = 1


@wp.func
def resolve_nTb(i: int, mcol: CollisionModel, dcol: CollisionData):  # type: ignore
    """A compacted tile-candidate index to (batch, segment, tile-within-segment).

    A branch-coherent binary search over `gen_s`, which holds each generator's
    first tile slot. Consecutive threads walk one generator's tiles in order, so
    the search is warp-coherent for all but the boundary warp of each.

    A generator that contributed nothing -- an inactive pair, or a super-tile the
    filter above rejected -- shares its successor's offset, and the search lands
    on the last of any such run, which is the one that owns slots.

    The tile is then named within its *segment*, not within its generator, which
    is what leaves everything downstream unable to tell the two kinds apart.

    The generator itself comes back as well. `tile_filter_seg` counts its
    survivors there, so it would otherwise recompute a number the search already
    holds -- see `compact_nTa` for why that count is what gets scanned.

    Called by `broadphase.tile_filter_seg`.
    In: i int, the tile candidate; mcol, read for gen_seg and gen_tile_beg (nG_world,) int32; dcol, read for gen_s
    (B*nG_world,) int32. Out: int, int, int, int: the world, the segment, the tile within the segment, and the
    (world, generator) slot.
    """
    lo = int(0)  # int: the search's lower bound
    hi = mcol.B * mcol.nG_world  # int: its upper bound

    while lo < hi:
        mid = (lo + hi) >> 1  # int
        if dcol.gen_s[mid] <= i:
            lo = mid + 1
        else:
            hi = mid

    k = lo - 1  # int: the (world, generator) slot owning `i`, b * nG_world + g
    b = k / mcol.nG_world  # int: the world
    g = k - b * mcol.nG_world  # int: the generator

    return b, mcol.gen_seg[g], mcol.gen_tile_beg[g] + (i - dcol.gen_s[k]), k


@wp.func
def resolve_nTa(tl: int, mcol: CollisionModel, dcol: CollisionData):  # type: ignore
    """A compacted tile index to (batch, segment, tile-within-segment).

    Slots are dense -- `b * nT_world + j`, with `j` the tile's index among the
    ones a world has -- so the segment comes from `tile_seg` and the
    tile-within-segment from subtracting that segment's start. Both are the
    inverse of how `make_collision_context` laid `tile_c`/`tile_r` out.

    Called by `resolve_nCa1_tiled`, `resolve_nCa1_partition` and `build_tile_sz`.
    In: tl int, the surviving tile; mcol, read for tile_seg (nT_world,) int32 and seg_nT_beg (nSEG,) int32; dcol,
    read for idm_nTa_to_slot (nTa_max,) int32. Out: int, int, int: the world, the segment, the tile within it.
    """
    slot = dcol.idm_nTa_to_slot[tl]  # int: the dense tile slot, b * nT_world + j

    b = slot / mcol.nT_world  # int: the world
    j = slot - b * mcol.nT_world  # int: the dense tile
    s = mcol.tile_seg[j]  # int: its segment
    t = j - mcol.seg_nT_beg[s]  # int: the tile within the segment

    return b, s, t


@wp.func
def resolve_nCa1_tiled(i: int, mcol: CollisionModel, dcol: CollisionData):  # type: ignore
    """
    A stage-1 primitive slot to (batch, contact, segment, tile).

    One indexed load and some arithmetic, shared by the `tile_K` threads of a
    tile rather than paid per primitive. A per-primitive binary search, over the
    widest launch in the pipeline, measured a large share of the whole collision
    section.

    The tile comes back too: the filter counts its survivors per tile, and
    recomputing `i / tile_K` there is a second integer division for a number the
    thread is already holding.

    Called by `broadphase._tiled_sdf_filter_one` and `resolve_nCa1`.
    In: i int, the stage-1 slot; mcol, read for tile_K and idul_nSEG_to_nC (nSEG+1,) int32; dcol, through
    `resolve_nTa`. Out: int, int, int, int: the world, the candidate (an index into nC), the segment and the
    surviving tile.
    """
    tl = i / mcol.tile_K  # int: the surviving tile
    k = i - tl * mcol.tile_K  # int: the slot within it

    b, s, t = resolve_nTa(tl, mcol, dcol)  # int, int, int: the world, the segment, the tile within it

    # The tile's primitives are a run of its *segment's* list, so the local index
    # is the candidate's offset within that segment.
    id_nC = mcol.idul_nSEG_to_nC[s] + t * mcol.tile_K + k  # int: the candidate

    return b, id_nC, s, tl


@wp.func
def resolve_nCa1_partition(i: int, mcol: CollisionModel, dcol: CollisionData):  # type: ignore
    """`resolve_nCa1_tiled`, but without assuming every tile is the same size.

    The division `i / tile_K` is what a uniform stride buys, and the only part
    of this pipeline that needs one. This resolve does without it: a thread
    reads the tile its block of `PARTITION_BLK` slots starts in, then walks
    forward over `tile_off` until it finds its own.

    It measures slower than the division, on this kernel and over the whole
    step, so a context whose segments share a tile size keeps the division.

    The walk is `PARTITION_BLK` over the mean tile size, so one warp per block
    keeps it to a step or two at `tile_K` 16. That is why it walks tiles: a
    walk over whole segments would be long, and a long walk is divergent, where
    a binary search is warp-uniform.

    Called by `broadphase._tiled_sdf_filter_one` and `resolve_nCa1`.
    In: i int, the stage-1 slot; mcol, read for seg_K (nSEG,) int32 and idul_nSEG_to_nC (nSEG+1,) int32; dcol,
    read for blk_base (nCa1_max // PARTITION_BLK + 2,) int32 and tile_off (nTa_max + 1,) int32, and through
    `resolve_nTa`. Out: as `resolve_nCa1_tiled`.
    """
    tl = dcol.blk_base[i / PARTITION_BLK]  # int: the tile owning the block's first slot, walked on to `i`'s
    while dcol.tile_off[tl + 1] <= i:
        tl += 1
    k = i - dcol.tile_off[tl]  # int: the slot within its tile

    b, s, t = resolve_nTa(tl, mcol, dcol)  # int, int, int: the world, the segment, the tile within it
    # `seg_K[s]`, not `tile_K`: here tile sizes differ between segments.
    id_nC = mcol.idul_nSEG_to_nC[s] + t * mcol.seg_K[s] + k  # int: the candidate
    return b, id_nC, s, tl


@wp.func
def resolve_nCa1(i: int, mcol: CollisionModel, dcol: CollisionData):  # type: ignore
    """Whichever of the two resolves this context's filter ran, chosen at runtime.

    The narrowphase resolves the same slots the stage-1 filter did and has to
    agree with it, but it runs over `nCa2`, a small fraction of stage 1's
    slots, so the difference between the two is not measurable there. A branch
    on a model scalar, uniform across the launch, is the cheaper trade than
    doubling every narrowphase specialization; stage 1 keeps the compile-time
    split, where the difference is measurable.

    Called by `narrowphase.compute_nsdf` and `narrowphase.gather_contacts`.
    In: i int, the stage-1 slot; mcol, read for partition; dcol. Out: as `resolve_nCa1_tiled`.
    """
    b = int(0)  # int: the world
    id_nC = int(0)  # int: the candidate
    s = int(0)  # int: the segment
    tl = int(0)  # int: the surviving tile

    if mcol.partition == 1:
        b, id_nC, s, tl = resolve_nCa1_partition(i, mcol, dcol)
    else:
        b, id_nC, s, tl = resolve_nCa1_tiled(i, mcol, dcol)

    return b, id_nC, s, tl


@wp.kernel
def build_tile_sz(mcol: CollisionModel, dcol: CollisionData):
    """Each surviving tile's own size, for the scan that gives `tile_off`.

    Zero past the live count and in the extra trailing slot, so the exclusive
    scan over the whole capacity ends with the stage-1 total at `tile_off`'s last
    entry -- which is what `ct[3] * tile_K` gives when the sizes agree.

    Launch: dim=(nTa_max + 1,), a thread per surviving-tile slot and one more, by `compact_nTa` where `partition`
    is 1.
    Reads: dcol.ct[3], the surviving tiles through `resolve_nTa`; mcol.seg_K (nSEG,) int32.
    Writes: dcol.tile_sz (nTa_max + 1,) int32.
    """
    t = wp.tid()  # int: the surviving tile

    if t >= dcol.ct[3]:
        dcol.tile_sz[t] = 0
        return

    b, s, k = resolve_nTa(t, mcol, dcol)  # int, int, int: the world, the segment, the tile within it
    dcol.tile_sz[t] = mcol.seg_K[s]


@wp.kernel
def build_blk_base(mcol: CollisionModel, dcol: CollisionData):
    """The tile owning each block's first slot: the last one starting at or below it.

    A binary search, but launched over *blocks* -- `PARTITION_BLK` times shorter
    than stage 1 -- so the per-slot cost is the walk from here, not the search.
    That is what partitioning buys: one warp-uniform search per 32 slots
    instead of a divergent one per slot.

    Launch: dim=(nCa1_max // PARTITION_BLK + 2,), a thread per block of `PARTITION_BLK` stage-1 slots, by
    `compact_nTa` where `partition` is 1.
    Reads: dcol.ct[3], dcol.tile_off (nTa_max + 1,) int32. Writes: dcol.blk_base (nCa1_max // PARTITION_BLK + 2,)
    int32.
    """
    k = wp.tid()  # int: the block
    target = k * PARTITION_BLK  # int: its first stage-1 slot

    lo = int(0)  # int: the search's lower bound
    hi = dcol.ct[3]  # int: its upper bound, the surviving tiles

    while lo < hi:
        mid = (lo + hi) >> 1  # int
        if dcol.tile_off[mid] <= target:
            lo = mid + 1
        else:
            hi = mid

    # Blocks past the live slots resolve to the last tile; their threads exit on
    # `id_nCa1 >= ct[0]` before the walk ever reads this.
    dcol.blk_base[k] = wp.clamp(lo - 1, 0, wp.max(dcol.ct[3] - 1, 0))


@wp.kernel
def scatter_nTa(mcol: CollisionModel, dcol: CollisionData):
    """Surviving tiles, in order, as dense slots. One thread per *generator*.

    A generator owns a contiguous run of tile candidates, so its survivors are a
    contiguous run of the output too: the thread starts at its scanned offset and
    walks its own candidates, ranking them locally. That is `compact_nCa2`'s
    trick one level up, and it is what lets the scan run over generators rather
    than over the candidate capacity -- `compact_nTa` says why that matters.

    Where the run *is* differs between the launches, and only here. Compacted, it
    is what `gen_weights` handed down (`gen_s`, `gen_w`), bounded by the live
    candidate count in case the list overflowed; densely there is no candidate
    list, so it is the generator's own tiles at their dense slots.

    What lands in `idm_nTa_to_slot` is a *dense* slot either way, which is what
    keeps `resolve_nTa` and everything past it identical under both launches.
    Densely a thread's index already is that slot; compacted, the filter wrote
    it to `tslot` when it resolved the thread, which is cheaper than resolving
    it a second time here.

    Launch: dim=(B*nG_world,), a thread per world and generator, by `compact_nTa`.
    Reads: dcol.goff (B*nG_world,) int32, dcol.tflag (nTb_max,) int32; compacted, dcol.gen_s and gen_w
    (B*nG_world,) int32, ct[4] and tslot (nTb_max,) int32; dense, mcol.seg_nT_beg (nSEG,) int32, gen_seg,
    gen_tile_beg and gen_nT (nG_world,) int32. Writes: dcol.idm_nTa_to_slot (nTa_max,) int32.
    """
    k = wp.tid()  # int: the (world, generator) slot
    b = k / mcol.nG_world  # int: the world
    g = k - b * mcol.nG_world  # int: the generator

    beg = int(0)  # int: the generator's first tile candidate, an index into tflag
    end = int(0)  # int: one past its last
    if mcol.seg_compact == 1:
        beg = dcol.gen_s[k]
        # A generator whose candidates were cut off by an overflowing list has
        # nothing past `ct[4]` to scatter; one starting past it walks nothing.
        end = wp.min(beg + dcol.gen_w[k], dcol.ct[4])
    else:
        # `gen_tile_beg` counts tiles from the start of the generator's own
        # *segment*, which is what `resolve_nTb` hands back and what the dense
        # slot needs `seg_nT_beg` added to.
        beg = (b * mcol.nT_world + mcol.seg_nT_beg[mcol.gen_seg[g]]
               + mcol.gen_tile_beg[g])
        end = beg + mcol.gen_nT[g]

    o = dcol.goff[k]  # int: the output slot of its first survivor
    for i in range(beg, end):  # i: int, a tile candidate
        if dcol.tflag[i] == 1:
            if o < dcol.nTa_max:
                if mcol.seg_compact == 1:
                    dcol.idm_nTa_to_slot[o] = dcol.tslot[i]
                else:
                    dcol.idm_nTa_to_slot[o] = i
            o += 1


@wp.kernel
def set_ct_nTa(mcol: CollisionModel, dcol: CollisionData):
    """
    The surviving-tile count: the end of the generator survivor scan, clamped to `nTa_max`, with its peak and
    overflow flag; and, where every segment shares a tile size, the stage-1 count `ct[3] * tile_K`.

    Launch: dim=1, a single thread, by `compact_nTa`.
    Reads: dcol.goff and gcnt (B*nG_world,) int32. Writes: dcol.ct[3], ct_raw[3], ct_peak[3] and overflow[3];
    where `uniform_K`, ct[0], ct_raw[0] and ct_peak[0].
    """
    # The scan is over generators, and every world has all of them, so its last
    # entry is the total in either launch -- with nothing to clear past a live
    # count, because there is no live count to be past.
    last = mcol.B * mcol.nG_world - 1  # int: the last (world, generator) slot
    total = dcol.goff[last] + dcol.gcnt[last]  # int: the surviving tiles, unclamped

    dcol.ct_raw[3] = total
    wp.atomic_max(dcol.ct_peak, 3, total)
    dcol.ct[3] = wp.min(total, dcol.nTa_max)
    if total > dcol.nTa_max:
        dcol.overflow[3] = 1

    # Stage 1 is the surviving tiles' primitive slots. Every kept tile
    # contributes exactly its own size, padding included: a tile past the end of
    # its segment's primitives is masked in the primitive kernel, not skipped
    # here, so that this count stays a plain product where the sizes agree and
    # the segment bounds below stay a plain scan.
    #
    # Where they differ it is not a product at all -- which tiles survived
    # decides how many slots they want -- so `set_ct_nCa1` takes it from the scan
    # of their sizes instead, and stage 1 can then overflow on its own.
    if mcol.uniform_K == 1:
        dcol.ct_raw[0] = dcol.ct_raw[3] * mcol.tile_K
        wp.atomic_max(dcol.ct_peak, 0, dcol.ct_raw[0])
        dcol.ct[0] = dcol.ct[3] * mcol.tile_K


@wp.kernel
def set_ct_nCa1(dcol: CollisionData):
    """The stage-1 count where tile sizes differ: the last entry of `tile_off`.

    Unlike the uniform case this is a capacity of its own -- `nTa_max` tiles of
    the largest segment ask for more slots than `nTa_max` of the smallest -- so
    it clamps and flags like every other stage.

    Launch: dim=1, a single thread, by `compact_nTa` where `partition` is 1 and `uniform_K` is 0.
    Reads: dcol.ct[3], dcol.tile_off (nTa_max + 1,) int32. Writes: dcol.ct[0], ct_raw[0], ct_peak[0] and
    overflow[0].
    """
    # At the live end, not the capacity's: `tile_sz` is zero past `ct[3]`, so the
    # two agree wherever the scan covered the whole array -- and only this one is
    # right when it did not. Same for the two counts below.
    total = dcol.tile_off[dcol.ct[3]]  # int: the stage-1 slots, unclamped

    dcol.ct_raw[0] = total
    wp.atomic_max(dcol.ct_peak, 0, total)
    dcol.ct[0] = wp.min(total, dcol.nCa1_max)
    if total > dcol.nCa1_max:
        dcol.overflow[0] = 1


@wp.kernel
def build_ul_nCa1_tiled(mcol: CollisionModel, dcol: CollisionData):
    """
    Segment bounds in the tiled stage-1 space.

    A segment owns a contiguous run of *generators*, and each of those owns a
    contiguous run of tiles, so the generator scan read at the segment's first
    generator counts every surviving tile before that segment -- and the stage-1
    slots those contributed are `tile_K` each where the sizes agree and
    `tile_off` read at that rank where they do not.

    That the scan is over generators rather than over the filter's own domain is
    what collapses this to one expression: a generator contributing nothing --
    an inactive pair, a rejected super-tile, or an entire dense world -- shares
    its neighbour's offset, so a compacted candidate list and a dense grid give
    the same answer without the launch having to know which it ran over.

    The rank itself is written out for `build_ul_nCa2_tiled`, so that it need not
    be recovered by dividing the stage-1 bound by `tile_K`.

    The (nSEG+1)-th bound is the *next world's* first generator, which for the
    last batch runs off the end of the scan and is the total.

    Launch: dim=(B, nSEG+1), a thread per world and segment bound, by `compact_nTa`.
    Reads: dcol.goff (B*nG_world,) int32, ct[0] and ct[3], dcol.tile_off (nTa_max + 1,) int32 where tile sizes
    differ; mcol.seg_nG_beg (nSEG,) int32. Writes: dcol.idul_B_nSEG_to_nCa1 and idul_B_nSEG_to_nTa (B, nSEG+1)
    int32.
    """
    b, s = wp.tid()  # int, int: the world, the segment bound

    off = mcol.nG_world  # int: the segment's first generator within its world; nG_world for the end bound
    if s < mcol.nSEG:
        off = mcol.seg_nG_beg[s]
    k = b * mcol.nG_world + off  # int: that generator's slot

    if k >= mcol.B * mcol.nG_world:
        dcol.idul_B_nSEG_to_nCa1[b, s] = dcol.ct[0]
        dcol.idul_B_nSEG_to_nTa[b, s] = dcol.ct[3]
        return

    rank = wp.min(dcol.goff[k], dcol.ct[3])  # int: the surviving tiles before the segment
    dcol.idul_B_nSEG_to_nTa[b, s] = rank

    if mcol.uniform_K == 1:
        dcol.idul_B_nSEG_to_nCa1[b, s] = rank * mcol.tile_K
    else:
        dcol.idul_B_nSEG_to_nCa1[b, s] = wp.min(dcol.tile_off[rank], dcol.ct[0])


#######################################################################################
# Stages 2 and 3: generic flag -> scan -> scatter
#######################################################################################


@wp.kernel
def zero_tcnt(dcol: CollisionData):
    """Clear the per-tile survivor counts before `tiled_sdf_filter` adds to them.

    `nTa_max` entries, where `flag1` has `nTa_max * tile_K` -- and the counts
    they hold spare a whole counting pass over `flag1`.

    Only up to the live tile count where the scan that follows is bounded by it:
    the tail is then neither summed nor read, so writing it is the pure waste
    this stage's own docstring warns about. With CUB the tail *is* summed, and
    the whole capacity has to be cleared.

    Launch: dim=(nTa_max,), a thread per surviving-tile slot, by `collide.broadphase_sdf`.
    Reads: dcol.ct[3] and dyn_scan. Writes: dcol.tcnt (nTa_max,) int32, zeroed (up to ct[3] where `dyn_scan`).
    """
    t = wp.tid()  # int: the surviving tile
    if dcol.dyn_scan == 1 and t >= dcol.ct[3]:
        return
    dcol.tcnt[t] = 0


@wp.kernel
def set_ct_nCa2_tiled(dcol: CollisionData):
    """
    The stage-2 count: the end of the tile survivor scan, clamped to `nCa2_max`, with its peak and overflow flag.

    Launch: dim=1, a single thread, by `compact_nCa2`.
    Reads: dcol.ct[3], dcol.toff1 and tcnt (nTa_max,) int32. Writes: dcol.ct[1], ct_raw[1], ct_peak[1] and
    overflow[1].
    """
    total = int(0)  # int: the stage-2 survivors, unclamped
    if dcol.ct[3] > 0:
        last = dcol.ct[3] - 1  # int: the last surviving tile
        total = dcol.toff1[last] + dcol.tcnt[last]

    dcol.ct_raw[1] = total
    wp.atomic_max(dcol.ct_peak, 1, total)
    dcol.ct[1] = wp.min(total, dcol.nCa2_max)
    if total > dcol.nCa2_max:
        dcol.overflow[1] = 1


@wp.kernel
def scatter_nCa2_tiled(mcol: CollisionModel, dcol: CollisionData):
    """One thread per tile, writing its survivors in order.

    The rank within a tile is a running counter over the tile's lanes, which is
    what a primitive-granularity scan would compute globally.

    Where tile sizes differ both the tile's first slot and how many it has come
    from `tile_off`, so the loop bound varies between threads -- but only across
    a segment boundary, since a segment's tiles are all one size.

    Launch: dim=(nTa_max,), a thread per surviving-tile slot, by `compact_nCa2`.
    Reads: dcol.ct[3], dcol.toff1 (nTa_max,) int32, dcol.flag1 (nCa1_max,) int32, dcol.tile_off (nTa_max + 1,)
    int32 where tile sizes differ. Writes: dcol.idm_nCa2_to_nCa1 (nCa2_max,) int32.
    """
    t = wp.tid()  # int: the surviving tile
    if t >= dcol.ct[3]:
        return

    beg = int(0)  # int: its first stage-1 slot
    n = int(0)  # int: its stage-1 slots
    if mcol.uniform_K == 1:
        beg = t * mcol.tile_K
        n = mcol.tile_K
    else:
        beg = dcol.tile_off[t]
        n = dcol.tile_off[t + 1] - beg

    o = dcol.toff1[t]  # int: the stage-2 slot of its first survivor
    for k in range(n):  # k: int, a slot within the tile
        i = beg + k  # int: the stage-1 slot
        if dcol.flag1[i] == 1:
            if o < dcol.nCa2_max:
                dcol.idm_nCa2_to_nCa1[o] = i
            o += 1


@wp.kernel
def build_ul_nCa2_tiled(mcol: CollisionModel, dcol: CollisionData):
    """
    Segment bounds in nCa2 space, from the tile scan.

    The tile scan evaluated at a segment's first surviving tile counts every
    survivor before that segment -- the same statement as stage 1, one level up.
    `build_ul_nCa1_tiled` hands that tile's rank over directly, which works
    whether or not every tile is `tile_K` in size and costs no division.

    Launch: dim=(B, nSEG+1), a thread per world and segment bound, by `compact_nCa2`.
    Reads: dcol.idul_B_nSEG_to_nCa1 and idul_B_nSEG_to_nTa (B, nSEG+1) int32, dcol.toff1 (nTa_max,) int32, ct[0]
    and ct[1]. Writes: dcol.idul_B_nSEG_to_nCa2 (B, nSEG+1) int32.
    """
    b, s = wp.tid()  # int, int: the world, the segment bound

    idx = dcol.idul_B_nSEG_to_nCa1[b, s]  # int: the segment's first stage-1 slot
    if idx >= dcol.ct[0]:
        dcol.idul_B_nSEG_to_nCa2[b, s] = dcol.ct[1]
    else:
        dcol.idul_B_nSEG_to_nCa2[b, s] = wp.min(
            dcol.toff1[dcol.idul_B_nSEG_to_nTa[b, s]], dcol.ct[1]
        )


@wp.kernel
def scatter_nCa3(dcol: CollisionData):
    """
    Writes each penetrating stage-2 candidate's index at its scanned stage-3 slot.

    Launch: dim=(nCa2_max,), a thread per stage-2 slot, by `compact_nCa3`.
    Reads: dcol.ct[1], dcol.flag2 and off2 (nCa2_max,) int32. Writes: dcol.idm_nCa3_to_nCa2 (nCa3_max,) int32.
    """
    i = wp.tid()  # int: the stage-2 slot
    if i >= dcol.ct[1]:
        return

    if dcol.flag2[i] == 1:
        o = dcol.off2[i]  # int: its stage-3 slot
        if o < dcol.nCa3_max:
            dcol.idm_nCa3_to_nCa2[o] = i


@wp.kernel
def set_ct_nCa3(dcol: CollisionData):
    """
    The stage-3 count: the end of the `flag2` scan, clamped to `nCa3_max`, with its peak and overflow flag.

    Launch: dim=1, a single thread, by `compact_nCa3`.
    Reads: dcol.ct[1], dcol.off2 and flag2 (nCa2_max,) int32. Writes: dcol.ct[2], ct_raw[2], ct_peak[2] and
    overflow[2].
    """
    total = int(0)  # int: the stage-3 survivors, unclamped
    if dcol.ct[1] > 0:
        last = dcol.ct[1] - 1  # int: the last live stage-2 slot
        total = dcol.off2[last] + dcol.flag2[last]

    dcol.ct_raw[2] = total
    wp.atomic_max(dcol.ct_peak, 2, total)
    dcol.ct[2] = wp.min(total, dcol.nCa3_max)
    if total > dcol.nCa3_max:
        dcol.overflow[2] = 1


@wp.kernel
def build_ul_nCa3(dcol: CollisionData):
    """
    Segment bounds in nCa3 space: the `flag2` scan read at each segment's first stage-2 slot.

    Launch: dim=(B, nSEG+1), a thread per world and segment bound, by `compact_nCa3`.
    Reads: dcol.idul_B_nSEG_to_nCa2 (B, nSEG+1) int32, dcol.off2 (nCa2_max,) int32, ct[1] and ct[2].
    Writes: dcol.idul_B_nSEG_to_nCa3 (B, nSEG+1) int32.
    """
    b, s = wp.tid()  # int, int: the world, the segment bound

    idx = dcol.idul_B_nSEG_to_nCa2[b, s]  # int: the segment's first stage-2 slot
    if idx >= dcol.ct[1]:
        dcol.idul_B_nSEG_to_nCa3[b, s] = dcol.ct[2]
    else:
        dcol.idul_B_nSEG_to_nCa3[b, s] = wp.min(dcol.off2[idx], dcol.ct[2])


#######################################################################################
# Drivers
#######################################################################################


def _scan(dcol: CollisionData, src, dst, slot: int, extra: int = 0):
    """An exclusive scan over a *capacity*, bounded by what is live in it.

    `lodos.utils.scan` when `dyn_scan` is on, CUB over the whole allocation
    when it is not. The two differ only past `ct[slot]`, which nothing reads --
    every `set_ct_*` takes its total at the live end and every `build_ul_*`
    clamps -- so this is a performance switch and not a semantic one.

    The scans over `B * nG_world` do not come through here: every entry of those
    is live, so bounding them would buy nothing and cost launches.

    Called by `compact_nTa`, `compact_nCa2` and `compact_nCa3`.
    In: dcol, read for dyn_scan, ct (5,) int32 and scan_buf; src, dst (n,) int32, the capacity-sized input and its
    exclusive scan; slot int, the `ct` entry that bounds it; extra int, added to that count. Out: dst, written.
    """
    if int(dcol.dyn_scan):
        lscan.scan(src, dst, dcol.ct, slot, dcol.scan_buf, extra=extra)
    else:
        wp.utils.array_scan(src, dst, inclusive=False)


def compact_nTb(mcol: CollisionModel, dcol: CollisionData):
    """Generator weights -> the tile candidates worth filtering. All device-side.

    The weights themselves are `broadphase.gen_weights`, which is where the
    super-tile test lives; this is the scan over them. Only runs when
    `capacity1 < 1.0`; at 1.0 the tile filter is launched densely over every tile
    every world has. See `make_collision_context`.

    Called by `collide.broadphase_bbox` where `seg_compact` is 1.
    Reads: dcol.gen_w (B*nG_world,) int32. Writes: dcol.gen_s (B*nG_world,) int32 and the tile-candidate count
    dcol.ct[4], with ct_raw, ct_peak and overflow at slot 4.
    """
    wp.utils.array_scan(dcol.gen_w, dcol.gen_s, inclusive=False)
    wp.launch(set_ct_nTb, dim=1, inputs=(mcol, dcol))


def compact_nTa(mcol: CollisionModel, dcol: CollisionData):
    """Surviving tiles -> stage-1 segment offsets and count. All device-side.

    The scan is over **generators**, not over the tile candidates themselves.
    Both orderings are (batch, segment, tile), so a candidate's rank is its
    generator's offset plus its rank within that generator -- the same identity
    `compact_nCa2` uses between tiles and their primitives.

    That matters because the candidate list is a *capacity* and the generator
    list is not: `B * nG_world` is a property of the scene, so this scan costs
    the same whatever `capacity1` is set to. It also spares the tile filter
    clearing its tail, as `compact_nCa2` spares stage 1.

    What this saves, against a scan over the tile candidates, is a small share
    of the step end to end; timed on its own, `array_scan` overstates it,
    carrying per-call host and allocation overhead that graph replay does not.
    Shortening the scans further would buy little.

    The middle block is what a stage-1 slot's tile costs when it is not
    `i / tile_K`: a launch over the surviving tiles, a scan over them, and a
    search per 32 slots. A context whose segments share a tile size skips all
    three and keeps the division.

    Called by `collide.broadphase_bbox`.
    Reads: dcol.gcnt (B*nG_world,) int32, tflag and tslot (nTb_max,) int32. Writes: dcol.goff (B*nG_world,)
    int32, idm_nTa_to_slot (nTa_max,) int32, ct[3] and ct[0]; where `partition`, tile_sz and tile_off
    (nTa_max + 1,) int32 and blk_base (nCa1_max // PARTITION_BLK + 2,) int32; and the segment bounds
    idul_B_nSEG_to_nCa1 and idul_B_nSEG_to_nTa (B, nSEG+1) int32.
    """
    wp.utils.array_scan(dcol.gcnt, dcol.goff, inclusive=False)
    wp.launch(set_ct_nTa, dim=1, inputs=(mcol, dcol))
    wp.launch(scatter_nTa, dim=(mcol.B * mcol.nG_world,), inputs=(mcol, dcol))

    if int(mcol.partition):
        wp.launch(build_tile_sz, dim=(dcol.nTa_max + 1,), inputs=(mcol, dcol))
        # `ct[3] + 1`: the extra trailing slot is what holds the stage-1 total.
        _scan(dcol, dcol.tile_sz, dcol.tile_off, 3, extra=1)
        if not int(mcol.uniform_K):
            wp.launch(set_ct_nCa1, dim=1, inputs=(dcol,))
        wp.launch(build_blk_base,
                  dim=(int(dcol.nCa1_max) // PARTITION_BLK + 2,),
                  inputs=(mcol, dcol))

    # After `tile_off`, which it reads for the segment starts when the sizes
    # differ, and after `ct[0]`, which it clamps against.
    wp.launch(build_ul_nCa1_tiled, dim=(mcol.B, mcol.nSEG + 1), inputs=(mcol, dcol))


def compact_nCa2(mcol: CollisionModel, dcol: CollisionData):
    """
    Stage-1 -> stage-2 compaction, at tile granularity.

    Stage 1 is `tile_K` contiguous slots per surviving tile, and that structure
    is worth exploiting: a vertex's stage-2 offset is its tile's offset plus its
    rank among the survivors of that tile. The scan runs over *tiles* rather than
    vertex slots, which is `tile_K` times shorter, and the rank is `tile_K` lanes
    of local work in the thread that owns the tile.

    A per-vertex scan would run over stage 1's slots, the largest arena in the
    pipeline. Prefix scan is bandwidth-bound and linear in its length, so the
    ratio of the two lengths is the saving.

    It also spares zeroing the stage-1 tail: nothing past `ct[0]` is ever read,
    because the tile loop never runs past `ct[3]`.

    Called by `collide.broadphase_sdf`.
    Reads: dcol.tcnt (nTa_max,) int32, flag1 (nCa1_max,) int32. Writes: dcol.toff1 (nTa_max,) int32, ct[1],
    idm_nCa2_to_nCa1 (nCa2_max,) int32 and idul_B_nSEG_to_nCa2 (B, nSEG+1) int32.
    """
    # `tcnt` was accumulated by `tiled_sdf_filter`, which only ever adds --
    # `zero_tcnt` in `broadphase_sdf` is what makes that valid, and it also
    # clears the entries past `ct[3]` so the scan sees a valid array over the
    # whole capacity.
    _scan(dcol, dcol.tcnt, dcol.toff1, 3)
    wp.launch(set_ct_nCa2_tiled, dim=1, inputs=(dcol,))
    wp.launch(scatter_nCa2_tiled, dim=(dcol.nTa_max,), inputs=(mcol, dcol))
    wp.launch(build_ul_nCa2_tiled, dim=(mcol.B, mcol.nSEG + 1),
              inputs=(mcol, dcol))


def compact_nCa3(mcol: CollisionModel, dcol: CollisionData):
    """
    Stage-2 -> stage-3 compaction.

    Called by `collide.narrowphase_sdf`.
    Reads: dcol.flag2 (nCa2_max,) int32. Writes: dcol.off2 (nCa2_max,) int32, ct[2], idm_nCa3_to_nCa2 (nCa3_max,)
    int32 and idul_B_nSEG_to_nCa3 (B, nSEG+1) int32.
    """
    _scan(dcol, dcol.flag2, dcol.off2, 1)
    wp.launch(set_ct_nCa3, dim=1, inputs=(dcol,))
    wp.launch(scatter_nCa3, dim=(dcol.nCa2_max,), inputs=(dcol,))
    wp.launch(build_ul_nCa3, dim=(mcol.B, mcol.nSEG + 1), inputs=(dcol,))
