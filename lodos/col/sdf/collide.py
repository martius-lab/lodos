import numpy as np
import warp as wp

from lodos.col.sdf import broadphase as sdf_broadphase
from lodos.col.sdf import narrowphase as sdf_narrowphase
from lodos.col.sdf import compact as sdf_compact
from lodos.col.sdf.types import CollisionModel, CollisionData

#######################################################################################
# Stages
#######################################################################################


def broadphase_bbox(mcol: CollisionModel, dcol: CollisionData):
    """
    OBB pair filter and stage-1 compaction.

    Also refreshes the geometry poses and the per-pair relative poses that both
    later stages read, so the stage functions remain safe to call in sequence
    after the rigid-body pass without a separate preparation step.

    The pair filter is followed by a *tile* filter, so stage 1 is the vertex
    slots of the surviving tiles rather than every vertex of an active pair.

    Called by `collide` and `tools.performance_tuning.profile_scene.stages_of`.
    In: mcol, the collision model; dcol, B worlds' collision state, read for dcol.xpos (B, nbody) wp.vec3 and
    dcol.xquat (B, nbody) wp.vec4. Out: dcol.T_w (B, nB) and dcol.T_rel (B, nSEG) wp.transformf, dcol.cpr_active
    (B, nCPR) bool, the surviving tiles dcol.idm_nTa_to_slot (nTa_max,) int32 and their count dcol.ct[3], the
    stage-1 count dcol.ct[0] and segment bounds dcol.idul_B_nSEG_to_nCa1 and idul_B_nSEG_to_nTa (B, nSEG+1) int32.
    """
    wp.launch(sdf_broadphase.compute_T_w, dim=(mcol.B, mcol.nB), inputs=(mcol, dcol))
    wp.launch(sdf_broadphase.compute_T_rel, dim=(mcol.B, mcol.nCPR), inputs=(mcol, dcol))
    # A context with a `toa` pair runs the `toa` variants of the filters, which also keep what the
    # pair's motion could bring within reach over the horizon; without one, the kernels as before.
    toa = int(mcol.has_toa)  # int: 1 when some pair is `toa`
    if toa:
        wp.launch(sdf_broadphase.compute_twist_rel, dim=(mcol.B, mcol.nSEG), inputs=(mcol, dcol))
    wp.launch(sdf_broadphase.obb_filter_toa if toa else sdf_broadphase.obb_filter, dim=(mcol.B, mcol.nCPR),
              inputs=(mcol, dcol))

    # The level above the tiles, which decides how many of them each generator
    # hands down, and clears the counter each accumulates its survivors into --
    # on both paths, since `compact_nTa` scans those counters whichever filter
    # produced them.
    wp.launch(sdf_broadphase.gen_weights_toa if toa else sdf_broadphase.gen_weights, dim=(mcol.B * mcol.nG_world,),
              inputs=(mcol, dcol))

    # `capacity1 < 1.0` compacts the box filter's survivors first, so the tile
    # filter runs over the tiles of *active* segments rather than over every tile
    # a world has. At 1.0 there is nothing to compact -- the list would be the
    # whole grid -- so the dense launch runs unchanged.
    if mcol.seg_compact:
        sdf_compact.compact_nTb(mcol, dcol)
        wp.launch(sdf_broadphase.tile_filter_seg_toa if toa else sdf_broadphase.tile_filter_seg,
                  dim=(dcol.nTb_max,), inputs=(mcol, dcol))
    else:
        # Densely there is no candidate list to shorten, so the level's weights
        # are read only where a super-tile could reject something -- the tile
        # filter reads that verdict rather than a scan of it.
        wp.launch(
            sdf_broadphase.tile_filter_toa if toa else sdf_broadphase.tile_filter,
            dim=(mcol.B, mcol.nT_world),
            inputs=(mcol, dcol),
        )
    sdf_compact.compact_nTa(mcol, dcol)


def broadphase_sdf(mcol: CollisionModel, dcol: CollisionData):
    """
    Voxel-SDF prefilter and stage-2 compaction.

    Called by `collide` and `tools.performance_tuning.profile_scene.stages_of`.
    In: mcol, dcol, after `broadphase_bbox`: the surviving tiles and dcol.ct[0], ct[3]. Out: dcol.flag1 (nCa1_max,)
    int32, dcol.tcnt and toff1 (nTa_max,) int32, the stage-2 candidates dcol.idm_nCa2_to_nCa1 (nCa2_max,) int32,
    their count dcol.ct[1] and segment bounds dcol.idul_B_nSEG_to_nCa2 (B, nSEG+1) int32.
    """
    # The filter accumulates each tile's survivor count as it goes, so the
    # counters have to start at zero. See `compact.zero_tcnt`.
    wp.launch(sdf_compact.zero_tcnt, dim=(dcol.nTa_max,), inputs=(dcol,))
    # `nCa1_grid`, not `nCa1_max`: the launch is sized by the hardware and the
    # threads walk. Equal to the allocation when `grid_stride` is off.
    wp.launch(sdf_broadphase.sdf_filter_kernel(mcol), dim=(dcol.nCa1_grid,),
              inputs=(mcol, dcol))
    sdf_compact.compact_nCa2(mcol, dcol)


def narrowphase_sdf(mcol: CollisionModel, dcol: CollisionData):
    """
    Spline-SDF evaluation, stage-3 compaction and the world-space contacts.

    The narrowphase kernel is a specialization picked once per context rather
    than a kernel with a runtime branch on the SDF representation.
    `narrowphase.nsdf_kernel` holds the reason.

    Called by `collide` and `tools.performance_tuning.profile_scene.stages_of`.
    In: mcol, dcol, after `broadphase_sdf`: the stage-2 candidates and dcol.ct[1]. Out: dcol.sdf_nCa2 (nCa2_max,)
    float32, dcol.n_nCa2 (nCa2_max,) wp.vec3, dcol.flag2 and off2 (nCa2_max,) int32; the contacts dcol.p_nCa3 and
    n_nCa3 (nCa3_max,) wp.vec3, sdf_nCa3 (nCa3_max,) float32, b_nCa3, seg_nCa3 and v_nCa3 (nCa3_max,) int32, their
    count dcol.ct[2] and segment bounds dcol.idul_B_nSEG_to_nCa3 (B, nSEG+1) int32.
    """
    wp.launch(sdf_narrowphase.nsdf_kernel(mcol), dim=(dcol.nCa2_max,), inputs=(mcol, dcol))
    sdf_compact.compact_nCa3(mcol, dcol)
    wp.launch(sdf_narrowphase.contact_kernel(mcol), dim=(dcol.nCa3_max,),
              inputs=(mcol, dcol))


def collide(mcol: CollisionModel, dcol: CollisionData):
    """
    Full Warp-side pipeline. Free of host synchronization end to end.

    Called by `lodos.step.forward` (every step), `make_collide_graph`, `calibrate_capacity` and 2 others.
    In: mcol, the collision model; dcol, B worlds' collision state, read for dcol.xpos (B, nbody) wp.vec3 and
    dcol.xquat (B, nbody) wp.vec4. Out: the contacts dcol.p_nCa3 and n_nCa3 (nCa3_max,) wp.vec3, sdf_nCa3
    (nCa3_max,) float32, b_nCa3, seg_nCa3 and v_nCa3 (nCa3_max,) int32, live up to dcol.ct[2], and their segment
    bounds dcol.idul_B_nSEG_to_nCa3 (B, nSEG+1) int32; each stage's arrays on the way (`broadphase_bbox`,
    `broadphase_sdf`, `narrowphase_sdf`).
    """
    broadphase_bbox(mcol, dcol)
    broadphase_sdf(mcol, dcol)
    narrowphase_sdf(mcol, dcol)


#######################################################################################
# CUDA Graph Capture
#######################################################################################


def make_collide_graph(mcol: CollisionModel, dcol: CollisionData, device=None):
    """
    Captures `collide` into a CUDA graph and returns a callable that replays it.

    Valid only while the buffers in `dcol` keep their addresses: use
    `lodos.rbd` (whose `xpos`/`xquat` these are) and do not reassign
    any preallocated array afterwards.

    Called by `tools.performance_tuning.tune.time_collision`.
    In: mcol, dcol, as `collide`; device, the CUDA device to capture on, the current one if None.
    Out: `replay`, a callable taking no arguments.
    """
    device = device or wp.get_device()

    # Warm up so kernel modules are loaded before capture.
    collide(mcol, dcol)
    wp.synchronize()

    with wp.ScopedCapture(device) as capture:  # wp.ScopedCapture
        collide(mcol, dcol)

    graph = capture.graph  # wp.Graph: one `collide`

    def replay():
        """
        Replays the captured `collide`: one graph launch, no host synchronization.

        Called by the callers of `make_collide_graph`, which returns it.
        """
        wp.capture_launch(graph)

    return replay


#######################################################################################
# Diagnostics
#######################################################################################


def get_counts(dcol: CollisionData):
    """
    Survivor counts [nCa1, nCa2, nCa3]. Synchronizes -- keep off the hot path.

    Three, the stage counts callers unpack: `ct` also carries the tile count,
    in a fourth slot. Read `dcol.ct.numpy()[3]` for the tiles.

    Not called in the repository.
    In: dcol, read for ct (5,) int32. Out: (3,) int32 numpy, [nCa1, nCa2, nCa3].
    """
    return dcol.ct.numpy()[:3]


def get_contacts(mcol: CollisionModel, dcol: CollisionData):
    """
    The final contact list as numpy arrays, trimmed to the live count.

    `seg` is the (pair, direction) segment the contact came from; `cpr` and `dir`
    unpack it. `p` lies on `cprs[cpr][0]`, the primitive geometry, and `n` is the
    outward normal of `cprs[cpr][1]`, the SDF geometry. With every pair
    asymmetric, `seg == cpr`.

    `radius` is the primitive's: 0 for a vertex, positive for an inner sphere,
    which is also what `sphere` reports as a bool. `sdf` is the *gap*, so it is
    already `phi - radius` and needs no adjusting to compare the two kinds.

    Synchronizes. Downstream Warp code should read the `*_nCa3` device arrays
    directly and bound its loops by `dcol.ct[2]` instead.

    Called by `viewer.scenes.nist.thread_phase`.
    In: mcol, read for prim_r (nV,) float32, seg_cpr and seg_dir (nSEG,) int32; dcol, read for ct (5,) int32 and
    the nCa3 arrays. Out: a dict of numpy arrays over the n = ct[2] contacts: p and n (n, 3) float32, sdf (n,)
    float32, batch, seg, cpr, dir and vertex (n,) int32, radius (n,) float32 and sphere (n,) bool.
    """
    n = int(dcol.ct.numpy()[2])  # int: the live contact count
    seg = dcol.seg_nCa3.numpy()[:n]  # (n,) int32: each contact's segment
    radius = mcol.prim_r.numpy()[dcol.v_nCa3.numpy()[:n]]  # (n,) float32: each contact's primitive radius

    return {
        "p": dcol.p_nCa3.numpy()[:n],  # world contact position
        "n": dcol.n_nCa3.numpy()[:n],  # world surface normal
        "sdf": dcol.sdf_nCa3.numpy()[:n],  # penetration depth (< 0)
        "batch": dcol.b_nCa3.numpy()[:n],
        "seg": seg,
        "cpr": mcol.seg_cpr.numpy()[seg],
        # 0: `cprs[cpr][0]` supplied the vertices, 1: `cprs[cpr][1]` did. Only a
        # symmetric pair ever reports both.
        "dir": mcol.seg_dir.numpy()[seg],
        "vertex": dcol.v_nCa3.numpy()[:n],
        # Provenance, not a solver input: nothing in `lodos.cd` reads either,
        # and a sphere contact is indistinguishable to it.
        "radius": radius,
        "sphere": radius > 0.0,
    }


def check_overflow(dcol: CollisionData, raise_on_overflow=True):
    """
    Reports stages whose survivors exceeded their preallocated capacity.

    Synchronizes, so this is a debugging aid, not a per-step check. On overflow
    the offending stage kept the first `nCa*_max` survivors and dropped the rest;
    re-create the context with a larger `capacity1`/`capacity2`.

    `nTb` is the tile filter's launch, sized by `capacity1` when tiling
    compacts; it cannot overflow at `capacity1=1.0`, where the launch is the
    whole grid.

    Not called in the repository.
    In: dcol, read for overflow and ct_peak (5,) int32 and the capacities; raise_on_overflow bool.
    Out: None if no stage overflowed, else the message (str), raised as a RuntimeError if `raise_on_overflow`.
    """
    flags = dcol.overflow.numpy()  # (5,) int32: 1 where a stage overflowed
    if not flags.any():
        return None

    # The *peak* each stage reached, not the latest step's count: the flags are
    # sticky, so pairing them with a live count reports whatever happened to be
    # there when this was called rather than what overflowed.
    raw = dcol.ct_peak.numpy()  # (5,) int32: each stage's peak
    stages = [  # 5 (name, capacity) pairs, in `ct`'s order
        ("nCa1", dcol.nCa1_max),
        ("nCa2", dcol.nCa2_max),
        ("nCa3", dcol.nCa3_max),
        ("nTa", dcol.nTa_max),
        ("nTb", dcol.nTb_max),
    ]
    detail = ", ".join(  # str
        f"{name}: {raw[i]} > capacity {cap}"
        for i, (name, cap) in enumerate(stages)
        if flags[i]
    )
    msg = f"collision stage capacity exceeded ({detail})"  # str

    if raise_on_overflow:
        raise RuntimeError(msg)
    return msg


def reset_overflow(dcol: CollisionData):
    """
    Clears the overflow flags and the peaks they are reported against.

    Called by `resize_context` and `Autosizer.tick`.
    Out: dcol.overflow and dcol.ct_peak (5,) int32, zeroed.
    """
    dcol.overflow.zero_()
    dcol.ct_peak.zero_()


#######################################################################################
# Adaptive capacity
#######################################################################################


def suggest_capacity(mcol, dcol, safety=1.5):
    """The capacity fractions the peaks reached so far would justify.

    Synchronizes. `safety` multiplies each observed peak, and the result is a
    fraction of the same worst case `make_collision_context` takes -- so it can
    be passed straight back to it, or to `resize_context`.

    Peaks, not the latest counts: `ct_peak` is a running maximum since the last
    `reset_overflow`, which is the only thing an allocation may be sized by.

    Called by `Autosizer.tick`.
    In: mcol, read for n_tile_slots, uniform_K, tile_K, B, n_slot_world and seg_compact; dcol, read for ct_peak
    (5,) int32; safety float. Out: a dict of float fractions, "capacity_tile" and "capacity2", and "capacity1"
    where the context compacts.
    """
    peak = dcol.ct_peak.numpy()  # (5,) int32: each stage's peak since the last `reset_overflow`
    worstT = max(1, int(mcol.n_tile_slots))  # int: B * nT_world, every tile of every world

    capT = min(1.0, safety * max(1, int(peak[3])) / worstT)  # float: the capacity_tile fraction
    if int(mcol.uniform_K):
        nCa1 = max(1, int(np.ceil(capT * worstT))) * int(mcol.tile_K)  # int: the stage-1 size `capT` gives
    else:
        nCa1 = max(1, int(np.ceil(capT * int(mcol.B) * int(mcol.n_slot_world))))

    out = {  # dict, knob name -> fraction
        "capacity_tile": capT,
        "capacity2": min(1.0, safety * max(1, int(peak[1])) / nCa1),
    }
    # `capacity1` sizes the tile filter's launch, which only exists when the
    # context compacts; at 1.0 the launch is the dense grid and there is no
    # fraction to move.
    if int(mcol.seg_compact):
        out["capacity1"] = min(
            1.0 - 1e-9, safety * max(1, int(peak[4])) / worstT
        )
    return out


def resize_context(mcol, dcol, capacity1=None, capacity2=None,
                   capacity_tile=None, contacts=False):
    """Reallocate the capacity-sized buffers in place, at new fractions.

    **Every CUDA graph capturing this context is invalid afterwards** -- the
    buffers have new addresses, and replaying an old graph writes into freed
    memory. Re-capture before the next step; `Autosizer` does that for you.

    **`capacity2` is a fraction of `nCa1_max`**, so moving `capacity_tile` alone
    leaves stage 2 at its old *absolute* size and a fraction that no longer
    means what it did. That is the safe direction -- the arena is merely wider
    than the fraction says -- but it is not the intended one, so pass both.
    The old fraction is not re-applied, which would be unsafe:
    shrinking the tile capacity, by any factor, does not shrink stage 2's live
    count at all.
    `Autosizer` passes both together for exactly this reason.

    Sizes may go up as well as down. Two things never move:

    - **`seg_compact`**, since crossing 1.0 switches which tile filter runs.
    - **`mcol`**, which holds no capacity at all.

    A third, **`nCa3_max` and the contact arrays sized by it**, moves only if
    `contacts=True`. `lodos.cd` builds its own `(nCa3_max,)` arrays from
    this one at solver-context time and has no way to hear about a change, so
    moving it behind the solver's back hands it a stale arena -- shorter, and
    then written past. So:

    - `contacts=False` (the default) pins `nCa3_max` and **caps `nCa2_max` at
      it**. Safe in the direction that matters, since stage 3 is a subset of
      stage 2. The contact arena is then whatever the context was built with,
      for the life of the run.
    - `contacts=True` resizes it with stage 2, and **the caller must rebuild the
      contact solver before the next step** -- `lodos.sim.make_sim` again, and
      re-capture. That is what lets a run start at a capacity nobody measured
      and converge all the way down instead of only part of the way.
      `Autosizer`'s `resize_contacts` is the supported way to ask for it.

    Everything else is free to grow, which is what lets an undersized context
    recover instead of having to be rebuilt: `scan_buf` is sized for the worst
    case at build time so that growing a capacity cannot outrun it.

    Called by `Autosizer.tick`.
    In: mcol, read for n_tile_slots, seg_compact, uniform_K, tile_K, B, n_slot_world and partition; dcol;
    capacity1, capacity2 and capacity_tile, float fractions, or None to leave one alone; contacts bool.
    Out: dcol's capacity-sized arrays reallocated with their capacities -- tflag and tslot (nTb_max,), tcnt, toff1
    and idm_nTa_to_slot (nTa_max,), flag1 (nCa1_max,), tile_sz and tile_off (nTa_max + 1,), blk_base
    (nCa1_max // PARTITION_BLK + 2,), the stage-2 arrays (nCa2_max,), and with `contacts` the contact arrays
    (nCa3_max,) -- and dcol.ct, ct_raw, overflow and ct_peak (5,) int32 zeroed.
    """
    from lodos.col.sdf.types import PARTITION_BLK, STAGE1_GRID, _stage1_grid

    # Every buffer below is about to be freed, and a caller cannot be relied on
    # to know what still holds them: work is dispatched asynchronously, so a caller
    # that has *returned* from a rollout may have it still running.
    # Freeing under it is an illegal access inside a graph, reported later and
    # somewhere else. A device-wide synchronize is the only thing that covers
    # streams this module never sees, and a resize is rare enough to afford one.
    wp.synchronize()

    n_tile_slots = int(mcol.n_tile_slots)  # int: B * nT_world
    seg_compact = bool(int(mcol.seg_compact))  # bool: whether the tile filter runs over a compacted list

    if capacity1 is not None:
        assert seg_compact and 0.0 < capacity1 < 1.0, (
            "capacity1 may only move within (0, 1) and only where the context "
            "already compacts; 1.0 is a different launch"
        )
        nTb_max = max(1, int(np.ceil(capacity1 * n_tile_slots)))  # int: the new tile-candidate capacity
        dcol.nTb_max = wp.int32(nTb_max)
        dcol.tflag = wp.zeros(nTb_max, dtype=wp.int32)
        dcol.tslot = wp.zeros(nTb_max, dtype=wp.int32)

    if capacity_tile is not None:
        assert 0.0 < capacity_tile <= 1.0, "capacity_tile must be in (0, 1]"
        nTa_max = max(1, int(np.ceil(capacity_tile * n_tile_slots)))  # int: the new surviving-tile capacity
        if int(mcol.uniform_K):
            nCa1_max = nTa_max * int(mcol.tile_K)  # int: the new stage-1 capacity
        else:
            nCa1_max = max(1, int(np.ceil(
                capacity_tile * int(mcol.B) * int(mcol.n_slot_world))))

        dcol.nTa_max = wp.int32(nTa_max)
        dcol.nCa1_max = wp.int32(nCa1_max)
        # A context that grid-strides keeps grid-striding. Its width is capped
        # at `STAGE1_GRID`, and where the allocation is below that the two
        # answers coincide -- so the cap is the test.
        dcol.nCa1_grid = wp.int32(_stage1_grid(
            nCa1_max, int(dcol.nCa1_grid) <= STAGE1_GRID))
        dcol.tcnt = wp.zeros(nTa_max, dtype=wp.int32)
        dcol.toff1 = wp.zeros(nTa_max, dtype=wp.int32)
        dcol.idm_nTa_to_slot = wp.zeros(nTa_max, dtype=wp.int32)
        dcol.flag1 = wp.zeros(nCa1_max, dtype=wp.int32)
        if int(mcol.partition):
            dcol.tile_sz = wp.zeros(nTa_max + 1, dtype=wp.int32)
            dcol.tile_off = wp.zeros(nTa_max + 1, dtype=wp.int32)
            dcol.blk_base = wp.zeros(
                nCa1_max // PARTITION_BLK + 2, dtype=wp.int32
            )

    if capacity2 is not None:
        assert 0.0 < capacity2 <= 1.0, "capacity2 must be in (0, 1]"
        want = max(1, int(np.ceil(capacity2 * int(dcol.nCa1_max))))  # int: the stage-2 size `capacity2` asks for
        nCa2_max = want if contacts else min(int(dcol.nCa3_max), want)  # int: the new stage-2 capacity
        dcol.nCa2_max = wp.int32(nCa2_max)
        dcol.idm_nCa2_to_nCa1 = wp.zeros(nCa2_max, dtype=wp.int32)
        dcol.sdf_nCa2 = wp.zeros(nCa2_max, dtype=wp.float32)
        dcol.n_nCa2 = wp.zeros(nCa2_max, dtype=wp.vec3)
        dcol.flag2 = wp.zeros(nCa2_max, dtype=wp.int32)
        dcol.off2 = wp.zeros(nCa2_max, dtype=wp.int32)

        if contacts:
            # `make_collision_context`'s own rule, `nCa3_max = nCa2_max`. The
            # solver's arrays are now the wrong length until the caller rebuilds
            # it, which is the contract `contacts=True` carries.
            dcol.nCa3_max = wp.int32(nCa2_max)
            dcol.idm_nCa3_to_nCa2 = wp.zeros(nCa2_max, dtype=wp.int32)
            dcol.p_nCa3 = wp.zeros(nCa2_max, dtype=wp.vec3)
            dcol.n_nCa3 = wp.zeros(nCa2_max, dtype=wp.vec3)
            dcol.sdf_nCa3 = wp.zeros(nCa2_max, dtype=wp.float32)
            dcol.b_nCa3 = wp.zeros(nCa2_max, dtype=wp.int32)
            dcol.seg_nCa3 = wp.zeros(nCa2_max, dtype=wp.int32)
            dcol.v_nCa3 = wp.zeros(nCa2_max, dtype=wp.int32)

    # The counts describe an arena that no longer exists, and the flags describe
    # overflows of it. `scan_buf` is not resized: `make_collision_context` sized
    # it for every capacity at its largest, so it covers every level.
    dcol.ct.zero_()
    dcol.ct_raw.zero_()
    reset_overflow(dcol)


class Autosizer:
    """Shrink a context's capacities onto what the run actually reaches.

    The loop a caller drives:

        sizer = Autosizer(mcol, dcol, recapture=lambda: rebuild_the_graph())
        for _ in range(n_steps):
            step()
            sizer.tick()

    `tick` is a device sync, so it only looks every `every` steps, and only
    after `warmup` of them -- a peak measured over three steps of a settling
    scene is not the peak.

    It shrinks when the allocation exceeds the justified size by more than
    `slack`, and grows by `grow` when a stage overflowed, which is the half that
    makes shrinking safe to attempt at all. The only thing growth cannot pass is
    `nCa3_max`; `resize_context` says why.

    A knob that has overflowed never shrinks back to that size: it keeps a clear
    `grow` above it for the rest of the run. Without that ratchet the two halves
    fight -- a grow resets the peaks, the next window looks over-sized, and the
    sizing hunts instead of settling.

    **It sizes to what it has seen, and an overflow is not merely slow.** There
    is no scene here and no offline pass -- `ct_peak` is whatever the live run
    reached -- so a context shrunk during a quiet phase can be overrun by a busy
    one, and an overrun stage **keeps the first `nCa*_max` survivors and drops
    the rest**. That is missing contacts, not a slower step, and it lasts until
    the next `tick` sees the flag: up to `every` steps.

    `safety` and `slack` multiply, so nothing shrinks until the allocation
    exceeds 2.25x the observed peak at the defaults, and a state has to beat the
    window's worst by that much to fall through. It is not zero risk.

    **Prefer it as an instrument where a dropped contact is not recoverable.**
    Run it once, read `suggest_capacity` off the result, and build the real
    context with those numbers -- `calibrate_capacity`'s answer without needing
    a trace. Running it live is the other supported use, and the one that lets a
    run start at a capacity nobody measured; shorten `every` and raise `safety`
    where a bad window would cost more than a rebuild.

    `resize_contacts` moves `nCa3_max` with stage 2 instead of pinning it. That
    is the difference between converging part of the way down and all the
    way, because a contact arena chosen before the run is otherwise frozen for
    it -- but it obliges `recapture` to rebuild the **solver** context as well,
    since `lodos.cd` sized its arrays from `nCa3_max` and would otherwise
    write past the new one. `resize_context` states the contract.

    Note also which peak it is: a capacity is a fraction of a *per-world* worst
    case, and `ct_peak` is the batch *total*, so this settles lower than an
    offline calibration of the same scene. That is the reason it is faster,
    but it means every world is being sized by the batch's mean behaviour rather
    than by its own worst.

    Not called in the repository.
    """

    def __init__(self, mcol, dcol, recapture=None, warmup=64, every=256,
                 safety=1.5, slack=1.5, grow=2.0, resize_contacts=False):
        """
        Reads the fractions the context is allocated at; nothing is resized before `tick`.

        Called when an `Autosizer` is built (see the class).
        In: mcol, dcol, the context to size; recapture, a callable taking no arguments, or None; warmup and every,
        ticks (int); safety, slack and grow, multipliers (float); resize_contacts bool.
        """
        self.mcol, self.dcol = mcol, dcol  # CollisionModel, CollisionData: the context it sizes
        self.recapture = recapture  # callable or None: re-captures the caller's graphs after a resize
        self.warmup, self.every = int(warmup), int(every)  # int, int: ticks before the first look, and between looks
        self.safety, self.slack, self.grow = safety, slack, grow  # float, float, float: see the class
        self.resize_contacts = bool(resize_contacts)  # bool: move `nCa3_max` with stage 2
        self.n = 0  # int: ticks so far
        self.resizes = 0  # int: resizes so far
        # What growth may reach: the worst case, not the size the context was
        # built at -- the arrays are reallocated either way, and a context that
        # started too small has to be able to recover. `capacity2` has a real
        # ceiling and `resize_context` applies it, in `nCa3_max`.
        self.ceiling = {"capacity_tile": 1.0, "capacity2": 1.0}  # dict, knob name -> fraction
        if int(mcol.seg_compact):
            self.ceiling["capacity1"] = 1.0 - 1e-9
        # Sticky, because the device flags are not once this has ticked: both
        # the resize and the window reset clear them, so a caller checking
        # `check_overflow` at the end of a run would find nothing. This is the
        # record that a capacity was ever short.
        self.overflowed = False  # bool: whether any stage has overflowed since construction
        # The largest fraction each knob has been seen to overflow at. A shrink
        # never goes back to one: without that the loop oscillates, because a
        # grow resets the peaks and the next window, over a quiet phase,
        # looks over-sized and shrinks straight back into the size that
        # just failed.
        self.floor = {}  # dict, knob name -> fraction
        self.current = {}  # dict, knob name -> the fraction the context is allocated at
        self._read_current()

    def _read_current(self):
        """
        The fractions the context is actually allocated at.

        Called by `__init__` and `tick`.
        Reads: dcol.nTa_max, nCa2_max, nCa1_max and nTb_max; mcol.n_tile_slots and seg_compact.
        Writes: self.current.
        """
        mcol, dcol = self.mcol, self.dcol  # CollisionModel, CollisionData
        self.current["capacity_tile"] = (
            int(dcol.nTa_max) / max(1, int(mcol.n_tile_slots))
        )
        self.current["capacity2"] = (
            int(dcol.nCa2_max) / max(1, int(dcol.nCa1_max))
        )
        if int(mcol.seg_compact):
            self.current["capacity1"] = (
                int(dcol.nTb_max) / max(1, int(mcol.n_tile_slots))
            )

    def tick(self):
        """
        One step's worth of bookkeeping. Returns True if it resized.

        Not called in the repository; the loop a caller drives calls it once per step.
        Reads: dcol.overflow and ct_peak (5,) int32, on a tick it looks at. Out: bool; on a resize the context's
        capacity-sized arrays are reallocated (`resize_context`) and `recapture` is called.
        """
        self.n += 1
        if self.n < self.warmup or self.n % self.every:
            return False

        over = self.dcol.overflow.numpy()  # (5,) int32: 1 where a stage overflowed
        self.overflowed = self.overflowed or bool(over.any())
        want = suggest_capacity(self.mcol, self.dcol, self.safety)  # dict, knob name -> justified fraction

        # Grow first, and only the knob that overflowed: an overflowed stage has
        # already dropped contacts, so its peak is clamped and `want` understates
        # it -- but the stages that did not overflow are sized correctly and
        # inflating them with the one that did would undo the shrinking.
        #
        #   overflow[0] nCa1 and [3] nTa   <- capacity_tile
        #   overflow[1] nCa2               <- capacity2
        #   overflow[4] nTb                <- capacity1
        #   overflow[2] nCa3               <- nothing here can move it
        new = {}  # dict, knob name -> the fraction to resize to
        if over.any():
            for key, flags in (("capacity_tile", (0, 3)),  # str, tuple of int: a knob, its overflow slots
                               ("capacity2", (1,)),
                               ("capacity1", (4,))):
                if key in self.current and any(over[i] for i in flags):
                    # This size was too small; remember it before leaving it.
                    self.floor[key] = max(self.floor.get(key, 0.0),
                                          self.current[key])
                    new[key] = min(self.ceiling[key],
                                   self.current[key] * self.grow)
        else:
            for key, w in want.items():  # w: float, the knob's justified fraction
                if key in self.current and self.current[key] > self.slack * w:
                    new[key] = w
            # Stage 2 is sized as a fraction of stage 1, so a change to the
            # tiles changes what its fraction means. Set it from the peak
            # instead of letting it stand: the peak is an absolute count and so
            # says the same thing at either stage-1 size.
            if "capacity_tile" in new and "capacity2" in want:
                new["capacity2"] = want["capacity2"]
            # The ratchet. A knob that has overflowed keeps a clear `grow`
            # above the size it failed at, so the sizing is monotone from the
            # first overflow on rather than hunting around it.
            for key in list(new):
                floor = self.floor.get(key, 0.0) * self.grow  # float: the smallest fraction the knob may shrink to
                new[key] = max(new[key], floor)
                if new[key] >= self.current[key]:
                    del new[key]

        new = {k: v for k, v in new.items() if abs(v - self.current[k]) > 1e-12}
        if not new:
            # Nothing to do, but the peaks have to start again from here or a
            # single busy step keeps the arena wide for the rest of the run.
            reset_overflow(self.dcol)
            return False

        resize_context(self.mcol, self.dcol, contacts=self.resize_contacts,
                       **new)
        # Re-read rather than trusting what was asked for: `resize_context`
        # clamps `capacity2` against `nCa3_max`, and a `current` that drifted
        # from the allocation would compare the next shrink against a size that
        # is not there.
        self._read_current()
        self.resizes += 1
        if self.recapture is not None:
            self.recapture()
        return True


def calibrate_capacity(mcol, dcol, states, safety=1.5):
    """
    Suggests `capacity1`/`capacity2` for `make_collision_context` by measuring
    the survivor counts actually reached over a set of representative states.

    The offline route. `suggest_capacity` and `Autosizer` are the online one, which
    needs no states and no second pass; this is what to use when the sizes have to
    be *known* before the run rather than found during it.

    Over-sizing is cheap but not free: the scans are bounded by the live count, so
    what is left is launch width, a real cost with `grid_stride` off
    (`make_collision_context`).

    states: iterable of callables that each install one state via
        the rigid-body pass and return nothing.
    safety: multiplier applied to the observed peak.

    Returns a dict with the recommended fractions; synchronizes, so this is a
    setup-time utility.

    Not called in the repository.
    Reads: dcol.ct_raw (5,) int32, after each state's `collide`. Out: the dict's "capacity1" (always 1.0),
    "capacity2" and "capacity_tile" are floats; "peak_nCa1", "peak_nCa2", "peak_nTa", "worst_case_nCa1" and
    "worst_case_nTa" are ints.
    """
    peak1 = peak2 = peakT = 0  # int, int, int: the peak nCa1, nCa2 and nTa
    for install in states:  # install: callable, installs one state
        install()
        collide(mcol, dcol)
        raw = dcol.ct_raw.numpy()  # (5,) int32: this state's unclamped counts
        peak1, peak2 = max(peak1, int(raw[0])), max(peak2, int(raw[1]))
        peakT = max(peakT, int(raw[3]))

    # Stage 1 follows from the tile capacity: every surviving tile contributes
    # exactly its own size in slots. `capacity1` sizes a different thing -- the
    # tile filter's own launch -- and `lodos.utils.calibrate.settings`
    # derives it from the peak `nTb`, which this pass does not measure.
    worstT = int(mcol.n_tile_slots)  # int: B * nT_world, every tile of every world
    capT = min(1.0, safety * peakT / worstT) if worstT else 1.0  # float: the capacity_tile fraction

    worst1 = int(mcol.B) * int(mcol.nC)  # int: B * nC, every candidate of every pair
    # The same rule `make_collision_context` sizes stage 1 by, and for the same
    # reason: which tiles survive decides how many slots they want once the sizes
    # differ, so the fraction goes on the slot total rather than on the tiles.
    if int(mcol.uniform_K):
        nCa1_max = max(1, int(np.ceil(capT * worstT))) * int(mcol.tile_K)  # int: the stage-1 size `capT` gives
    else:
        nCa1_max = max(
            1, int(np.ceil(capT * int(mcol.B) * int(mcol.n_slot_world)))
        )
    cap1 = 1.0  # float: the capacity1 fraction
    cap2 = min(1.0, safety * peak2 / nCa1_max) if peak2 else 1e-3  # float: the capacity2 fraction

    return {
        "capacity1": cap1,
        "capacity2": cap2,
        "capacity_tile": capT,
        "peak_nCa1": peak1,
        "peak_nCa2": peak2,
        "peak_nTa": peakT,
        "worst_case_nCa1": worst1,
        "worst_case_nTa": worstT,
    }
