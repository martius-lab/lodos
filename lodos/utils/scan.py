"""
An exclusive prefix scan whose *length* is read from device memory.

`wp.utils.array_scan` is CUB underneath and takes its length as a host integer,
so every compaction in this pipeline scans its whole allocation -- the last entry
has to be valid, because that is where the total is read from. The capacity is
therefore not only a memory bound but a per-step cost, and a scene calibrated at
2x its true peak pays for the other half on every step.

This module's scan is bounded by a count that lives in `dcol.ct`. Launch
dimensions are host-side constants derived from the capacity, so the section is
one CUDA graph, and a block whose tile starts past the live count exits without
reading anything.

**Reduce-then-scan, not decoupled look-back.** The single-pass chained scan
(b0nes164/GPUPrefixSums, and what CUB itself does) carries a partition's
running sum forward through device memory and is roughly 2N of traffic against
this design's 3N -- one extra read of the input. It also needs release/acquire
ordering, a spin on a neighbour's flag, and either a forward-progress guarantee
from the scheduler or that repository's fallback path for devices without one.
Warp exposes no memory fence and no volatile load, so writing it here would mean
inline CUDA through `@wp.func_native`, and a scan that is subtly wrong under
contention produces wrong *contact indices* silently. The extra read is the price
of not needing any of that: this version wins whenever the live count is under
about two thirds of the capacity, which is the regime the capacities are
in: each is sized with a margin over its peak.

Past the live count the output is not maintained: entries beyond
it hold whatever the last step left. An exclusive scan is a prefix, so an entry's
value depends only on entries before it -- everything at or below the live count
is exact, and nothing in the pipeline reads past it. That is the same contract
`compact_nCa2` keeps for stage 1, whose tail it does not clear.
"""

import warp as wp

# Elements per block. 1024 at 256 threads is four per thread, which is where
# `tile_scan_exclusive` stops being launch-bound without making the ragged last
# tile expensive. The level count follows from it: 34M elements is three levels.
TILE = 1024
BLOCK = 256  # threads per block


@wp.func
def _live(ct: wp.array(dtype=int), slot: int, extra: int, div: int):  # type: ignore
    """How many entries of this level are live, from the count one level 0 has.

    Level `l` holds one entry per `TILE^l` entries of the input, so its live
    length is the input's rounded up by that -- `div` is that power, passed in
    rather than recomputed, since the caller knows the level.

    Called by `_reduce` and `_scan_add`.
    In: ct (5,) int32, the device-side counts (`dcol.ct` in the pipeline); slot int, the count bounding the scan;
    extra int, added to it; div int, `TILE**l`. Out: int, the level's live length.
    """
    n = ct[slot] + extra  # int: the input's live length
    return (n + div - 1) / div


@wp.kernel
def _reduce(src: wp.array(dtype=int), part: wp.array(dtype=int),  # type: ignore
            ct: wp.array(dtype=int), slot: int, extra: int, div: int):  # type: ignore
    """
    Each block's own sum, for the level above it.

    Launch: `wp.launch_tiled`, dim=(sizes[l + 1],) = ceil(sizes[l] / TILE) blocks of BLOCK threads, a block per
    tile of level `l`, by `scan`, once per level below the top. A block whose tile starts past the live length
    returns.
    In: src (sizes[l],) int32, level `l`; ct, slot, extra, div as `_live`.
    Out: part (sizes[l + 1],) int32, each block's tile sum at the block's index.
    """
    b = wp.tid()  # int: the block, which is the tile it sums
    if b * TILE >= _live(ct, slot, extra, div):
        return

    t = wp.tile_load(src, shape=TILE, offset=b * TILE, bounds_check=True)  # tile (TILE,) int32
    wp.tile_store(part, wp.tile_sum(t), offset=b)


@wp.kernel
def _scan_top(src: wp.array(dtype=int), dst: wp.array(dtype=int)):  # type: ignore
    """The last level, which is one tile by construction, scanned in one block.

    No bound to test: at most `TILE` entries exist, and the ones past the live
    count are stale rather than wrong -- they contribute only to prefixes that
    are themselves past it.

    Launch: `wp.launch_tiled`, dim=(1,), one block of BLOCK threads, by `scan`: on the input itself when it fits
    one tile, else in place on the top level.
    In: src (<= TILE,) int32. Out: dst, the same length, int32, its exclusive scan; `dst` may be `src`.
    """
    t = wp.tile_load(src, shape=TILE, offset=0, bounds_check=True)  # tile (TILE,) int32
    wp.tile_store(dst, wp.tile_scan_exclusive(t), offset=0, bounds_check=True)


@wp.kernel
def _scan_add(src: wp.array(dtype=int), dst: wp.array(dtype=int),  # type: ignore
              base: wp.array(dtype=int),  # type: ignore
              ct: wp.array(dtype=int), slot: int, extra: int, div: int):  # type: ignore
    """A block's own exclusive scan, offset by everything before its tile.

    Safe in place (`src is dst`): a tile is loaded cooperatively before it is
    stored, and blocks own disjoint ranges.

    Launch: `wp.launch_tiled`, dim=(sizes[l + 1],) blocks of BLOCK threads, a block per tile of level `l`, by
    `scan`, once per level from the one below the top down to the input. A block whose tile starts past the live
    length returns.
    In: src (sizes[l],) int32, level `l`; base (sizes[l + 1],) int32, the scanned level above; ct, slot, extra,
    div as `_live`. Out: dst (sizes[l],) int32, level `l`'s exclusive scan; `dst` is `src` above the input level.
    """
    b = wp.tid()  # int: the block, which is the tile it scans
    if b * TILE >= _live(ct, slot, extra, div):
        return

    t = wp.tile_load(src, shape=TILE, offset=b * TILE, bounds_check=True)  # tile (TILE,) int32
    s = wp.tile_scan_exclusive(t)  # tile (TILE,) int32: the tile's own exclusive scan
    off = wp.tile_load(base, shape=1, offset=b)  # tile (1,) int32: the sum of everything before the tile
    wp.tile_store(dst, s + wp.tile_broadcast(off, shape=(TILE,)),
                  offset=b * TILE, bounds_check=True)


def plan(n: int):
    """The level sizes for a scan over `n` elements: `[n, ceil(n/TILE), ...]`.

    Host-side arithmetic on a capacity, so the launch count and every launch
    dimension are compile-time constants and the whole thing captures.

    Called by `scan` and `workspace`.
    In: n int, the capacity. Out: list of int, ending at the first size <= TILE.
    """
    sizes = [int(n)]  # list of int: the level sizes
    while sizes[-1] > TILE:
        sizes.append((sizes[-1] + TILE - 1) // TILE)
    return sizes


def workspace(*n_max):
    """Elements of scratch a shared workspace needs to serve every given site.

    One buffer for all of them: the scans run one after another in the pipeline,
    so nothing overlaps, and the largest site's partials fit every other. Tiny in
    absolute terms -- 9.2M elements need 9009 of scratch.

    Called by `lodos.col.sdf.types.make_collision_context`.
    In: n_max, the sites' capacities, ints. Out: int, a count of int32 elements; 0 for no site.
    """
    return max(sum(plan(n)[1:]) for n in n_max) if n_max else 0


def scan(src, dst, ct, slot: int, buf, extra: int = 0):
    """Exclusive scan of `src[:ct[slot] + extra]` into `dst`.

    The length of `src` is the capacity, and it is what sizes the launches; only
    the work is bounded by the count. Entries of `dst` past that bound are left
    as they were -- see the module docstring for why that is the contract.

    src, dst: (n,) int32. May not alias.
    ct: the device-side count array (`dcol.ct`).
    slot: which of its entries bounds this scan.
    extra: added to that count, for a scan of `n + 1` entries whose count names
        the `n` -- `tile_off` is the one such site.
    buf: shared int32 scratch, at least `workspace(n)` long.

    Called by `lodos.col.sdf.compact._scan`.
    In: ct (5,) int32, [nCa1, nCa2, nCa3, nTa, nTb] in the pipeline. Out: dst, exact up to the live length.
    """
    sizes = plan(len(src))  # list of K int: the level sizes, sizes[0] = n
    K = len(sizes)  # int: the number of levels

    if K == 1:
        wp.launch_tiled(_scan_top, dim=(1,), inputs=(src, dst), block_dim=BLOCK)
        return

    # Slice the shared buffer into one array per level above the input.
    part = [None]  # list of K: part[l] a (sizes[l],) int32 view of buf for l >= 1; part[0] unused
    o = 0  # int: the next level's offset into buf
    for n in sizes[1:]:  # n: int, a level's size
        part.append(buf[o:o + n])
        o += n

    # Up: every level's block sums become the level above it.
    lo = src  # (sizes[l],) int32: the level being reduced
    for l in range(K - 1):  # l: int, the level
        wp.launch_tiled(
            _reduce, dim=(sizes[l + 1],),
            inputs=(lo, part[l + 1], ct, slot, extra, TILE ** l),
            block_dim=BLOCK,
        )
        lo = part[l + 1]

    # The top level fits one tile by construction.
    wp.launch_tiled(_scan_top, dim=(1,), inputs=(part[K - 1], part[K - 1]),
                    block_dim=BLOCK)

    # Down: each level's own tiles, offset by the scanned level above it.
    for l in range(K - 2, -1, -1):
        wp.launch_tiled(
            # `sizes[l + 1]` is that count by construction -- it is how the
            # level above was sized.
            _scan_add, dim=(sizes[l + 1],),
            inputs=(src if l == 0 else part[l],
                    dst if l == 0 else part[l],
                    part[l + 1], ct, slot, extra, TILE ** l),
            block_dim=BLOCK,
        )
