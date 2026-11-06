"""
How full every buffer of a `lodos.sim.Sim` gets, and what to size it to.

Every capacity in a scene is fixed at construction -- CUDA graphs need stable
addresses -- and each one, when exceeded, *drops* work rather than failing: Lodos's
collision stages clamp their survivor lists, MJWarp drops contacts past `naconmax` and
rows past `njmax`. A run that overflowed simulated a different
problem, and nothing about the trajectory says so.

`peaks(sim, nsteps)` rolls the scene forward and records, for each capacity, the
largest demand seen and whether it overflowed. `suggest(report)` turns that into
`make_sim` arguments with a safety margin. Calibrate on a representative rollout --
peaks come with contact-rich phases, so include the impact and the settled state --
and at the batch size you will run: per-world demand is a distribution, and a larger
batch samples its tail.
"""

from __future__ import annotations

import math

import warp as wp

from lodos import step as lstep


def _reset(sim):
    """
    Zeroes Lodos's collision peaks and overflow flags, when the scene has Lodos surfaces.

    Called by `peaks` and `Tracker.reset`.
    In: sim, a `lodos.sim.Sim`. Writes: sim.dcol.ct_peak and sim.dcol.overflow (5,) int32, zeroed.
    """
    if sim.dcol is not None:
        sim.dcol.ct_peak.zero_()
        sim.dcol.overflow.zero_()


def peaks(sim, nsteps, every=1, graph=True):
    """
    Rolls `sim` forward `nsteps` and returns the demand on each capacity.

    The state advances: snapshot it first (`lodos.rbd.io.save_state`) if the rollout
    should start where it was. Counters Lodos keeps on the device (the collision
    stages' peaks) see every step; MJWarp's contact, broadphase pair and row counts are
    read every `every` steps, every step by default: a peak can last only a few steps, so
    sparser reads can miss it, and each read synchronizes. The broadphase's pairs go in a
    buffer of `naconmax` too, and past it are dropped.

    Called by `notebooks/04_batched_rollouts.ipynb`.
    In: sim, a `lodos.sim.Sim`; nsteps int; every int, the steps between reads of MJWarp's counts; graph bool, whether
    to replay a captured step (`lodos.step.make_step_graph`) rather than call `lodos.step.fwd_step`.
    Reads: d.nacon and d.ncollision (1,) int32, d.nefc (nworld,) int32; dcol.ct_peak and dcol.overflow (5,) int32.
    Out: dict: "B", "nsteps"; "runtime": "nacon_peak", "ncollision_peak", "naconmax", "overflow",
    "nefc_peak_per_world", "njmax", "efc_overflow"; with Lodos surfaces, "lodos_col": "stages", and "peak", "capacity"
    and "overflow", five each in the stages' order; "overflowed" bool.
    """
    _reset(sim)
    replay = lstep.make_step_graph(sim) if graph else (lambda: lstep.fwd_step(sim))  # callable: one step
    d = sim.d  # lodos.mjcf.types.Data: MJWarp's data
    # int, int, int, bool, bool: the peaks read so far, and whether a read overflowed
    nacon_peak, ncollision_peak, nefc_peak, runtime_overflow, efc_overflow = 0, 0, 0, False, False
    for i in range(nsteps):  # i: int, the step
        replay()
        if (i + 1) % every == 0 or i == nsteps - 1:
            # int, int: the batch's contacts and broadphase pairs
            n, c = int(d.nacon.numpy()[0]), int(d.ncollision.numpy()[0])
            nacon_peak, ncollision_peak = max(nacon_peak, n), max(ncollision_peak, c)
            runtime_overflow |= max(n, c) > d.naconmax
            e = int(d.nefc.numpy().max())  # int: the most rows in one world
            nefc_peak = max(nefc_peak, e)
            efc_overflow |= e >= d.njmax
    wp.synchronize()
    out = dict(B=sim.B, nsteps=nsteps,  # dict: the report
               runtime=dict(nacon_peak=nacon_peak, ncollision_peak=ncollision_peak, naconmax=int(d.naconmax),
                            overflow=bool(runtime_overflow)))
    out["runtime"].update(nefc_peak_per_world=nefc_peak, njmax=int(d.njmax), efc_overflow=bool(efc_overflow))
    if sim.dcol is not None:
        out["lodos_col"] = dict(
            stages=("nCa1", "nCa2", "nCa3", "nTa", "nTb"),
            peak=sim.dcol.ct_peak.numpy().tolist(),
            capacity=[int(sim.dcol.nCa1_max), int(sim.dcol.nCa2_max), int(sim.dcol.nCa3_max),
                      int(sim.dcol.nTa_max), int(sim.dcol.nTb_max)],
            overflow=sim.dcol.overflow.numpy().astype(bool).tolist(),
        )
    out["overflowed"] = bool(out["runtime"]["overflow"] or out["runtime"].get("efc_overflow")
                             or any(out.get("lodos_col", {}).get("overflow", [])))
    return out


def suggest(report, safety=1.5):
    """
    `make_sim` arguments from a `peaks` report: MJWarp's contact and row capacities, each
    `safety` times the observed peak; the contacts' capacity holds the broadphase's pairs
    too, so it follows the larger of the two peaks. Lodos's collision capacities have their
    own calibration, `lodos.col.sdf.collide.calibrate_capacity`; the report's `lodos_col`
    entry says whether they need it.

    Called by `notebooks/04_batched_rollouts.ipynb`.
    In: report, a `peaks` dict; safety float. Out: dict: "nconmax" (per world) and "njmax", int, each where its peak
    is nonzero.
    """
    B = report["B"]  # int: the batch measured
    out = {}  # dict: make_sim argument -> int
    m = report["runtime"]  # dict: MJWarp's peaks
    if n := max(m["nacon_peak"], m["ncollision_peak"]):  # int: the larger of the contacts' and the pairs' peaks
        out["nconmax"] = max(1, math.ceil(safety * n / B))
    if "nefc_peak_per_world" in m and m["nefc_peak_per_world"]:
        out["njmax"] = max(1, math.ceil(safety * m["nefc_peak_per_world"]))
    return out


#######################################################################################
# Calibration over a state set
#######################################################################################
#
# `peaks` follows one rollout from wherever the scene is. What a tuning file is made of
# is the demand over a *state set* (`lodos.utils.states`): every state rolled out under the
# controls and the layout the scene will be run with, the peaks read on the device after
# every step rather than every `every`, and each capacity sized to its own peak.
#
# A peak read in an arena that overflowed is not the demand: the stage that overflowed
# kept its first survivors, so every stage after it saw fewer than it would have. Every
# counter below is the raw one -- Lodos's `ct_peak`, MJWarp's `nacon`, the hybrid's
# counts -- so the stage that overflowed reports its true demand,
# but the stages downstream of it do not. `calibrate` therefore repeats the measurement,
# grown, until nothing overflows; only then are the peaks the demand.

# What `Tracker` keeps, in its `peak` array.
TRACKED = ("nacon", "nefc_world", "runtime_contacts_world", "lodos_contacts_world", "epa", "manifold",
           "hybrid_contacts", "ncollision")


@wp.kernel
def _track_batch(nacon: wp.array(dtype=int), ncollision: wp.array(dtype=int), peak: wp.array(dtype=int)):  # type: ignore
    """
    Raises the peaks of MJWarp's contacts and broadphase pairs, batch totals, to this step's counts.

    Launch: dim=1, a single thread, by `Tracker.launch`.
    In: nacon (1,) int32, ncollision (1,) int32, d's. Out: peak (8,) int32, slots 0 ("nacon") and 7 ("ncollision").
    """
    wp.atomic_max(peak, 0, nacon[0])
    wp.atomic_max(peak, 7, ncollision[0])


@wp.kernel
def _track_rows(nefc: wp.array(dtype=int), peak: wp.array(dtype=int)):  # type: ignore
    """
    Raises the peak of MJWarp's constraint rows in one world to this step's.

    Launch: dim=B, a thread per world, by `Tracker.launch`.
    In: nefc (nworld,) int32, d's. Out: peak (8,) int32, slot 1 ("nefc_world").
    """
    wp.atomic_max(peak, 1, nefc[wp.tid()])


@wp.kernel
def _count_by_world(n: wp.array(dtype=int), slot: int, cap: int, world: wp.array(dtype=int),  # type: ignore
                    cnt: wp.array(dtype=int)):  # type: ignore
    """
    Counts, per world, the first `min(n[slot], cap)` entries of a contact list.

    Launch: by `Tracker.launch`: dim=naconmax, a thread per MJWarp contact slot (n d.nacon, slot 0, world
    d.contact.worldid), and dim=nCa3_max, a thread per Lodos contact slot (n dcol.ct, slot 2, world dcol.b_nCa3).
    In: n (1,) or (5,) int32, the list's count at `slot`; slot int; cap int, the list's capacity; world (cap,) int32,
    each entry's world. Out: cnt (max(1, B),) int32, added to.
    """
    i = wp.tid()  # int: the entry
    if i < wp.min(n[slot], cap):
        wp.atomic_add(cnt, world[i], 1)


@wp.kernel
def _max_by_world(cnt: wp.array(dtype=int), k: int, peak: wp.array(dtype=int)):  # type: ignore
    """
    Raises `peak[k]` to the largest count one world has in `cnt`, and zeroes `cnt` for the next count.

    Launch: dim=B, a thread per world, by `Tracker.launch`, with k=2 for MJWarp's contacts and k=3 for Lodos's.
    In: cnt (max(1, B),) int32; k int, a slot of `TRACKED`. Out: peak (8,) int32, slot k; cnt zeroed.
    """
    w = wp.tid()  # int: the world
    wp.atomic_max(peak, k, cnt[w])
    cnt[w] = 0


@wp.kernel
def _track_counts(epa: wp.array(dtype=int), man: wp.array(dtype=int), con: wp.array(dtype=int),  # type: ignore
                  peak: wp.array(dtype=int)):  # type: ignore
    """
    Raises the peaks of the hybrid narrowphase's EPA pairs, manifolds and contacts to this step's counts.

    Launch: dim=1, a single thread, by `Tracker.launch`.
    In: epa, man, con (1,) int32, the hybrid's `epa_count`, `man_count` and `con_count`.
    Out: peak (8,) int32, slots 4 ("epa"), 5 ("manifold") and 6 ("hybrid_contacts").
    """
    wp.atomic_max(peak, 4, epa[0])
    wp.atomic_max(peak, 5, man[0])
    wp.atomic_max(peak, 6, con[0])


class Tracker:
    """
    Running maxima, on the device, of every demand a step of `sim` makes that its own
    counters do not already keep: MJWarp's contacts (the batch total and the most in one
    world), its constraint rows per world, Lodos's contacts per world, and the
    hybrid narrowphase's EPA, manifold and contact counts. `launch` after each step; it is
    capturable, so `capture` puts it in the step's graph.

    Built by `measure`, `viewer.session.Runner.build`, `tools.performance_tuning.common.throughput` and 7 others.
    """

    def __init__(self, sim):
        """
        Allocates the maxima and the per-world counts, zeroed.

        Called when a `Tracker` is built (see the class). In: sim, a `lodos.sim.Sim`.
        """
        self.sim = sim  # lodos.sim.Sim: the scene tracked
        self.peak = wp.zeros(len(TRACKED), dtype=int)  # (8,) int32: the running maxima, in `TRACKED`'s order
        self.cnt = wp.zeros(max(1, sim.B), dtype=int)  # (max(1, B),) int32: per-world tallies, zeroed after use
        # HybridNarrowphase or None: the exact hull narrowphase, when MJWarp's collision is active and has one
        self.hy = getattr(sim.prim, "hy", None) if sim.prim is not None and sim.prim.active else None

    def launch(self):
        """
        Raises the maxima to this step's demands: launches the counting kernels, without synchronizing.

        Called by `capture` and `viewer.session.Runner.capture`.
        Reads: d.nacon and d.ncollision (1,), d.nefc (nworld,), d.contact.worldid (naconmax,) int32; dcol.ct (5,)
        and dcol.b_nCa3 (nCa3_max,) int32; the hybrid's epa_count, man_count and con_count (1,) int32.
        Writes: self.peak (8,) int32; self.cnt (max(1, B),) int32, left zeroed.
        """
        sim, d = self.sim, self.sim.d  # lodos.sim.Sim, lodos.mjcf.types.Data
        wp.launch(_track_batch, dim=1, inputs=(d.nacon, d.ncollision), outputs=(self.peak,))
        wp.launch(_track_rows, dim=sim.B, inputs=(d.nefc,), outputs=(self.peak,))
        if int(d.naconmax):
            wp.launch(_count_by_world, dim=int(d.naconmax),
                      inputs=(d.nacon, 0, int(d.naconmax), d.contact.worldid), outputs=(self.cnt,))
            wp.launch(_max_by_world, dim=sim.B, inputs=(self.cnt, 2), outputs=(self.peak,))
        if sim.dcol is not None:
            wp.launch(_count_by_world, dim=int(sim.dcol.nCa3_max),
                      inputs=(sim.dcol.ct, 2, int(sim.dcol.nCa3_max), sim.dcol.b_nCa3), outputs=(self.cnt,))
            wp.launch(_max_by_world, dim=sim.B, inputs=(self.cnt, 3), outputs=(self.peak,))
        if self.hy is not None:
            st = self.hy.st  # lodos.col.prim.hybrid.kernels.PipeState: the hybrid's counters
            wp.launch(_track_counts, dim=1, inputs=(st.epa_count, st.man_count, st.con_count), outputs=(self.peak,))

    def reset(self):
        """
        Zeroes the maxima, and Lodos's collision peaks and overflow flags with them.

        Called by `measure`, `tools.performance_tuning.common.throughput`, `tools.performance_tuning.common.artifacts`
        and 4 others.
        Writes: self.peak (8,) int32; sim.dcol.ct_peak and sim.dcol.overflow (5,) int32.
        """
        self.peak.zero_()
        _reset(self.sim)

    def capture(self):
        """
        A replayable step that also tracks: `fwd_step` and `launch`, in one CUDA graph.

        Runs one step uncaptured first, so the state advances a step. Called by `measure`,
        `tools.performance_tuning.common.throughput`, `tools.performance_tuning.common.artifacts` and 6 others.
        Out: a callable taking no arguments, which replays the graph.
        """
        sim = self.sim  # lodos.sim.Sim
        lstep.fwd_step(sim)
        self.launch()
        wp.synchronize()
        with wp.ScopedCapture() as cap:  # cap: wp.ScopedCapture
            lstep.fwd_step(sim)
            self.launch()
        graph = cap.graph  # wp.Graph: the step and the tracking
        return lambda: wp.capture_launch(graph)

    def report(self):
        """
        Every capacity's peak demand, its allocation and whether it overflowed. Synchronizes.

        Called by `measure`, `viewer.session.Runner.demand`, `tools.performance_tuning.common.throughput` and 5 others.
        Out: dict: "B"; "runtime": "nacon", "ncollision", "naconmax", "contacts_per_world", "overflow",
        "nefc_per_world", "njmax", "efc_overflow", and with the hybrid narrowphase "ccd": "epa", "manifold",
        "contacts", "naccdmax", "overflow"; with Lodos surfaces, "lodos_col": "peak", "allocated" and "overflow",
        dicts by stage, "tile_slots", "slots_per_world", "uniform_K", "tile_K", "compacts", "contacts_per_world";
        "overflowed" bool. Counts are int.
        """
        sim, d = self.sim, self.sim.d  # lodos.sim.Sim, lodos.mjcf.types.Data
        wp.synchronize()
        p = dict(zip(TRACKED, self.peak.numpy().tolist()))  # dict: `TRACKED` name -> its maximum, int
        out = dict(B=sim.B)  # dict: the report
        # MJWarp's broadphase writes its candidate pairs into arrays of the same `naconmax`, and
        # a scene of many hulls has far more overlapping boxes than contacts: a pair dropped
        # there is a contact never looked for.
        m = dict(nacon=p["nacon"], ncollision=p["ncollision"], naconmax=int(d.naconmax),  # dict: MJWarp's demands
                 contacts_per_world=p["runtime_contacts_world"],
                 overflow=p["nacon"] > int(d.naconmax) or p["ncollision"] > int(d.naconmax))
        m.update(nefc_per_world=p["nefc_world"], njmax=int(d.njmax), efc_overflow=p["nefc_world"] >= int(d.njmax))
        if self.hy is not None:
            hy = self.hy  # HybridNarrowphase
            m["ccd"] = dict(epa=p["epa"], manifold=p["manifold"], contacts=p["hybrid_contacts"],
                            naccdmax=int(d.naccdmax),
                            overflow=p["epa"] > hy.cap_epa or p["manifold"] > hy.cap_man
                            or p["hybrid_contacts"] > hy.cap_con)
        out["runtime"] = m
        if sim.dcol is not None:
            dcol, mcol = sim.dcol, sim.mcol  # CollisionData, CollisionModel: Lodos's collision context
            stages = ("nCa1", "nCa2", "nCa3", "nTa", "nTb")  # tuple of str: `dcol.ct`'s order
            # tuple of 5 int: the stages' capacities
            alloc = (dcol.nCa1_max, dcol.nCa2_max, dcol.nCa3_max, dcol.nTa_max, dcol.nTb_max)
            out["lodos_col"] = dict(
                peak=dict(zip(stages, dcol.ct_peak.numpy().tolist())),
                allocated=dict(zip(stages, (int(a) for a in alloc))),
                overflow=dict(zip(stages, dcol.overflow.numpy().astype(bool).tolist())),
                tile_slots=int(mcol.n_tile_slots), slots_per_world=int(mcol.n_slot_world),
                uniform_K=int(mcol.uniform_K), tile_K=int(mcol.tile_K), compacts=int(mcol.seg_compact),
                contacts_per_world=p["lodos_contacts_world"],
            )
        out["overflowed"] = overflowed(out)
        return out


def which_overflowed(rep):
    """
    The names of the capacities that overflowed in a `Tracker.report`.

    Called by `tools.performance_tuning.common.throughput`, `tools.performance_tuning.common.artifacts`,
    `bench.lodos_mjwarp.lodos_point.point` and 3 others.
    In: rep, a `Tracker.report` dict. Out: list of str: "naconmax", "njmax", "nccdmax" and "lodos <stage>".
    """
    out = []  # list of str
    m = rep["runtime"]  # dict: MJWarp's demands
    if m["overflow"]:
        out.append("naconmax")
    if m.get("efc_overflow"):
        out.append("njmax")
    if m.get("ccd", {}).get("overflow"):
        out.append("nccdmax")
    out += [f"lodos {k}" for k, v in rep.get("lodos_col", {}).get("overflow", {}).items() if v]
    return out


def overflowed(rep):
    """
    Whether any capacity in a `Tracker.report` overflowed -- and with it, which peaks are not demands.

    Called by `Tracker.report`. In: rep, a `Tracker.report` dict. Out: bool.
    """
    m = rep["runtime"]  # dict: MJWarp's demands
    return bool(m["overflow"] or m.get("efc_overflow") or m.get("ccd", {}).get("overflow")
                or any(rep.get("lodos_col", {}).get("overflow", {}).values()))


def measure(sim, sset, layout="independent", horizon=200, controls=None, assign=None, cover=True,
            max_batches=None, rng=None, progress=None):
    """
    The peaks of `sim` over the state set `sset`: every batch `lodos.utils.states.batches` makes
    of it, rolled out `horizon` steps under `controls`. Returns `Tracker.report` plus how
    it was measured. The state of `sim` is overwritten.

    Called by `calibrate`.
    In: sim, a `lodos.sim.Sim`; sset, a `lodos.utils.states.StateSet`; layout str; horizon int, steps per batch;
    controls, a `lodos.utils.states.Controls` or None (hold); assign, cover and rng as `lodos.utils.states.batches`
    takes them; max_batches int or None, the most batches measured; progress, None or a callable taking
    (batches done, batches in all).
    Out: dict, `Tracker.report`'s, with "layout", "horizon", "batches", "states" and "controls" added.
    """
    from lodos.utils import states as lstates

    tracker = Tracker(sim)  # Tracker
    replay = tracker.capture()  # callable: one tracked step
    tracker.reset()
    # list of (B,) int64: each batch's states, one per world
    groups = lstates.batches(len(sset), sim.B, layout, assign, sset.weights, rng, cover)
    if max_batches is not None:
        groups = groups[:max_batches]
    for k, idx in enumerate(groups):  # k: int, the batch; idx: (B,) int64, its states
        lstates.rollout(sim, sset, idx, horizon, controls, replay)
        if progress is not None:
            progress(k + 1, len(groups))
    rep = tracker.report()  # dict: the peaks over every batch
    rep.update(layout=layout, horizon=int(horizon), batches=len(groups), states=len(sset),
               controls=(controls or lstates.Controls()).describe())
    return rep


def settings(rep, safety=1.5, njmax_floor=16):
    """
    A tuning file's section from a `Tracker.report` that did not overflow: every capacity
    `safety` times its peak, in the units `make_sim` takes it in.

        collision     `capacity_tile`, `capacity2` -- fractions of the batch's tiles and of
                      stage 1's arena, batch totals -- and `capacity1`, the tile filter's
                      compacted launch, when the report measured it
        runtime       `naconmax` (batch), `nconmax` (its per-world share), `njmax` (per world,
                      at least `njmax_floor`) and `nccdmax` (per world, the hybrid's EPA and
                      manifold scratch)

    Called by `calibrate`, `tools.performance_tuning.calibrate.calibrate_section`,
    `tools.performance_tuning.common.start_section` and `viewer.session.Runner.tune_capacities`.
    In: rep, a `Tracker.report` dict; safety float; njmax_floor int.
    Out: dict: "collision" (capacity_tile, capacity2 and capacity1, float), "runtime" (naconmax, nconmax, njmax and
    nccdmax, int), "sim" and "solver_kwargs" (empty).
    """
    if rep["overflowed"]:
        raise ValueError("the report overflowed, so its peaks understate the demand: grow and measure again")
    B = rep["B"]  # int: the batch measured
    out = dict(collision={}, runtime={}, sim={}, solver_kwargs={})  # dict: the section
    col = rep.get("lodos_col")  # dict or None: Lodos's collision stages
    if col is not None:
        p = col["peak"]  # dict: stage -> its peak, int
        worstT = max(1, col["tile_slots"])  # int: the batch's tile slots, at least 1
        capT = min(1.0, safety * max(1, p["nTa"]) / worstT)  # float: capacity_tile
        if col["uniform_K"]:
            nCa1 = max(1, math.ceil(capT * worstT)) * col["tile_K"]  # int: stage 1's arena at capT
        else:
            nCa1 = max(1, math.ceil(capT * B * col["slots_per_world"]))
        out["collision"] = dict(capacity_tile=capT, capacity2=min(1.0, safety * max(1, p["nCa2"]) / nCa1))
        if col["compacts"]:
            out["collision"]["capacity1"] = min(1.0, safety * max(1, p["nTb"]) / worstT)
    m = rep["runtime"]  # dict: MJWarp's demands
    naconmax = max(B, math.ceil(safety * max(m["nacon"], m.get("ncollision", 0))))  # int: the batch's contacts
    out["runtime"] = dict(naconmax=naconmax, nconmax=max(1, math.ceil(naconmax / B)))
    out["runtime"]["njmax"] = max(njmax_floor, math.ceil(safety * m["nefc_per_world"]))
    ccd = m.get("ccd")  # dict or None: the hybrid narrowphase's demands
    if ccd is not None:
        need = max(ccd["epa"], ccd["manifold"], math.ceil(ccd["contacts"] / 4))  # int: the batch's CCD slots
        out["runtime"]["nccdmax"] = min(out["runtime"]["nconmax"], max(1, math.ceil(safety * need / B)))
    return out


def grow(current, rep, factor=2.0, safety=1.5):
    """
    `current` (a `settings` section) with every capacity that overflowed in `rep` raised: to
    `safety` times the raw demand it reported, and at least `factor` times what it was --
    the raw demand of a stage behind one that overflowed is an underestimate.

    Called by `calibrate`.
    In: current dict, a `settings`-style section; rep, a `Tracker.report` dict; factor float; safety float.
    Out: dict, a new section; `current` is left as it was.
    """
    out = {k: dict(v) for k, v in current.items()}  # dict: a copy of `current`, two levels deep
    B = rep["B"]  # int: the batch measured
    col = rep.get("lodos_col")  # dict or None: Lodos's collision stages
    if col is not None:
        c = out.setdefault("collision", {})  # dict: the collision section, grown in place
        # dict, dict, dict: per stage, whether it overflowed, its peak and its allocation
        of, p, alloc = col["overflow"], col["peak"], col["allocated"]
        worstT = max(1, col["tile_slots"])  # int: the batch's tile slots, at least 1
        if of["nCa1"] or of["nTa"]:
            c["capacity_tile"] = min(1.0, max(factor * c.get("capacity_tile", alloc["nTa"] / worstT),
                                              safety * p["nTa"] / worstT))
        if of["nCa2"] or of["nCa3"]:
            c["capacity2"] = min(1.0, max(factor * c.get("capacity2", alloc["nCa2"] / max(1, alloc["nCa1"])),
                                          safety * p["nCa2"] / max(1, alloc["nCa1"])))
        if of["nTb"]:
            c["capacity1"] = min(1.0, max(factor * c.get("capacity1", alloc["nTb"] / worstT), safety * p["nTb"] / worstT))
    m = rep["runtime"]  # dict: MJWarp's demands
    mw = out.setdefault("runtime", {})  # dict: the runtime section, grown in place
    if m["overflow"]:
        mw["naconmax"] = max(math.ceil(factor * m["naconmax"]),
                             math.ceil(safety * max(m["nacon"], m.get("ncollision", 0))))
        mw["nconmax"] = max(1, math.ceil(mw["naconmax"] / B))
    if m.get("efc_overflow"):
        mw["njmax"] = math.ceil(factor * m["njmax"])
    ccd = m.get("ccd")  # dict or None: the hybrid narrowphase's demands
    if ccd is not None and ccd["overflow"]:
        mw["nccdmax"] = math.ceil(factor * ccd["naccdmax"] / B)
    if "nccdmax" in mw and "nconmax" in mw:
        mw["nccdmax"] = min(mw["nccdmax"], mw["nconmax"])
    return out


def calibrate(build, sset, B, start, layout="independent", horizon=200, controls=None, safety=1.5,
              max_rounds=5, progress=None, **measure_kwargs):
    """
    Measures `build(section, B)` -- a `Sim` at the capacities of a `settings`-style
    section -- over `sset`, growing whatever overflowed until nothing does, and returns
    `(settings(report, safety), report, rounds)`. `start` is the first section tried:
    generous, since every round that overflows is a full measurement thrown away.

    Called by `tools.performance_tuning.calibrate.calibrate_section`.
    In: build, a callable `(section, B) -> lodos.sim.Sim`; sset, a `lodos.utils.states.StateSet`; B int; start dict,
    a section; layout, horizon, controls, progress and measure_kwargs as `measure` takes them; safety float;
    max_rounds int. Out: (dict, dict, int): the settings, the last report and the rounds it took.
    """
    if max_rounds < 1:
        raise ValueError(f"max_rounds must be at least 1, not {max_rounds}")
    section = {k: dict(v) for k, v in start.items()}  # dict: the capacities this round builds with
    for r in range(1, max_rounds + 1):  # r: int, the round
        sim = build(section, B)  # lodos.sim.Sim
        rep = measure(sim, sset, layout, horizon, controls, progress=progress, **measure_kwargs)  # dict: its report
        del sim
        if not rep["overflowed"]:
            return settings(rep, safety), rep, r
        section = grow(section, rep, safety=safety)
    raise RuntimeError(f"still overflowing after {max_rounds} rounds: {rep}")
