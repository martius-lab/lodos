"""
Tunes one scene for throughput, end to end, and records the settings before and after.

    python tools/performance_tuning/tune.py arm_5_sdf [--states S.npz] \\
        [--B 2048] [--out-dir tools/performance_tuning/tuned] [--copy-from SCENE] [--keep-tiles] [--mode mj]

1. **States.** A state set (`explore.py`, with the scene family's defaults below), unless
   `--states` names one.
2. **Baseline.** The scene's own settings -- its per-world MJWarp capacities, the
   collision defaults (`capacity_tile`/`capacity2` 1.0, tile size 8), its solver
   defaults -- at the tuning batch size: world-steps/s, and the artifact metrics
   every candidate is held to (`common.artifacts`). A baseline that does not fit in memory
   is recorded as such and measured at the largest batch that does.
3. **Tiles** (Lodos surfaces only).
   Segments are grouped by their (vertex surface, SDF surface) assets, and each group's
   tile size is searched over 4, 8, 16, 32 with the others held, timing the collision
   pipeline alone on frozen states, each candidate at 1.5x its own demand.
4. **Capacities.** `calibrate.calibrate_section` at the tile sizes found: every capacity
   1.5x its peak over the state set.
5. **Budgets** -- the knobs that change the answer: the solver's proximal iterations and
   contact reduction. A candidate is kept only if it is faster by 3% and
   its artifacts are no worse than the baseline's (`common.no_worse`).
6. **Batch size.** The tuned settings at 2x, 4x, 8x and 16x the tuning batch, each recalibrated
   at its own size, while throughput grows by more than 5% and memory allows; the fastest
   becomes the file's.

The tuning file goes to `<out-dir>/<scene>.tuning.json` with a `baseline` section, and
the comparison to `<out-dir>/<scene>.report.json`. `--copy-from` makes a variant that differs
from another scene only in its joint rows follow that scene. `--keep-tiles` takes the scene's file's tile sizes
rather than searching them again -- a re-tune after a change that leaves the collision as
it was; it refuses a file that is missing, holds no tile record or is stale.

`--mode` tunes under a mode other than the scene's own (`exact` unless the scene says): every
step runs under it, from a state set explored under it, and the file -- which records the
mode in its `solver_kwargs`, so that a build from it runs that mode -- goes beside the
scene's own, in `<out-dir>/<mode>/`, its states in `<states-dir>/<mode>/`. The proximal
budgets are the exact mode's: under mj, which runs no proximal loop, only contact reduction
is tried.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibrate as CAL  # noqa: E402
import common as C  # noqa: E402
import explore as EX  # noqa: E402

from lodos.utils import calibrate as lcalibrate  # noqa: E402
from lodos.utils import states as lstates  # noqa: E402
from lodos.utils import tuning as ltuning  # noqa: E402
from lodos.cd import driver as cd_driver  # noqa: E402
from lodos.col.sdf import collide as sdf_collide  # noqa: E402
from lodos.col.sdf import types as sdf_types  # noqa: E402
from lodos.rbd import smooth  # noqa: E402

# Per scene family: how to explore it, and the batch to tune at (the smallest the batch-size
# sweep tries). The first prefix that matches a scene's name wins.
FAMILIES = [
    ("sharpa_armadillo", dict(strategy=["initial", "perturb", "actuation"], starts=32, horizon=1000, stride=50,
                              jitter=0.002, sigma=0.3), 1024),
    ("shadow_grasp", dict(strategy=["initial", "perturb", "actuation"], starts=32, horizon=1000, stride=50,
                          jitter=0.002, sigma=0.3), 1024),
    # twenty hands: four times the five's bodies, tuned from a quarter of their batch
    ("sharpa_walled_20", dict(strategy=["scatter"], starts=32, horizon=2000, stride=50), 128),
    ("sharpa_20", dict(strategy=["scatter"], starts=32, horizon=2000, stride=50), 128),
    ("shadow_walled_20", dict(strategy=["scatter"], starts=32, horizon=2000, stride=50), 128),
    ("shadow_20", dict(strategy=["scatter"], starts=32, horizon=2000, stride=50), 128),
    ("sharpa_walled_5", dict(strategy=["scatter"], starts=32, horizon=2000, stride=50), 512),
    ("sharpa_5", dict(strategy=["scatter"], starts=32, horizon=2000, stride=50), 512),
    ("shadow_walled_5", dict(strategy=["scatter"], starts=32, horizon=2000, stride=50), 512),
    ("shadow_5", dict(strategy=["scatter"], starts=32, horizon=2000, stride=50), 512),
    ("sharpa", dict(strategy=["scatter"], starts=32, horizon=2000, stride=50), 1024),
    ("arm_walled_20", dict(strategy=["scatter"], starts=32, horizon=2500, stride=50), 1024),
    ("arm_20", dict(strategy=["scatter"], starts=32, horizon=2500, stride=50), 1024),
    ("arm_box_20", dict(strategy=["scatter"], starts=32, horizon=2500, stride=50), 1024),
    ("cube_walled_20", dict(strategy=["scatter"], starts=32, horizon=2500, stride=50), 2048),
    ("cube_20", dict(strategy=["scatter"], starts=32, horizon=2500, stride=50), 2048),
    ("nutbolt", dict(strategy=["scatter"], starts=32, horizon=3000, stride=100), 1024),
    # the NIST nut-bolts and board scatter by their own jittered starts, as Newton's nut-bolt; the threading
    # and the driven nuts have none, and scattering their free nut would throw it about the scene
    ("nist_nutbolt", dict(strategy=["scatter"], starts=32, horizon=3000, stride=100), 1024),
    ("nist_board", dict(strategy=["scatter"], starts=32, horizon=3000, stride=100), 1024),
    ("nist_threading", dict(strategy=["initial", "perturb", "actuation"], starts=32, horizon=1000, stride=50,
                            jitter=0.0005, sigma=0.3), 1024),
    ("nist_driven", dict(strategy=["initial", "perturb"], starts=32, horizon=2000, stride=50, jitter=0.0002), 1024),
    ("house_of_cards", dict(strategy=["initial", "perturb"], starts=32, horizon=1500, stride=50, jitter=0.0005), 2048),
    ("masonry_arch", dict(strategy=["initial", "perturb"], starts=32, horizon=1500, stride=50, jitter=0.0005), 2048),
    ("incline", dict(strategy=["initial", "perturb"], starts=32, horizon=1500, stride=50, jitter=0.002), 8192),
    ("backspin_ball", dict(strategy=["initial", "perturb"], starts=32, horizon=1500, stride=50, jitter=0.002), 8192),
    ("painleve_box", dict(strategy=["initial", "perturb"], starts=32, horizon=1500, stride=50, jitter=0.002), 8192),
    ("arm_", dict(strategy=["scatter"], starts=64, horizon=2000, stride=50), 2048),
    ("cube_", dict(strategy=["scatter"], starts=64, horizon=2000, stride=50), 2048),
    # anything else, an MJCF file among them: every strategy that applies to it (`make_states`
    # skips scatter without a sampler or free bodies, and actuation without actuators)
    ("", dict(strategy=["initial", "scatter", "perturb", "actuation"], starts=32, horizon=1500, stride=50,
              jitter=0.002, sigma=0.3), 1024),
]

TILE_SIZES = (4, 8, 16, 32)
BUDGETS = [dict(prox_iters_per_qp=5), dict(prox_iters_per_qp=3)]


def family(name):
    for prefix, ex, B in FAMILIES:
        if name.startswith(prefix):
            return dict(ex), B
    raise AssertionError


class Explore:
    """`explore.py`'s arguments, as an object its functions can be given."""

    def __init__(self, **kw):
        self.strategy = ["scatter"]
        self.starts, self.horizon, self.stride = 64, 2000, 50
        self.seed, self.jitter, self.sigma, self.max_depth = 0, 0.01, 0.5, 0.01
        self.__dict__.update(kw)


def make_states(src, name, path, ex):
    """A state set for `src` by the family's exploration, saved to `path`."""
    a = Explore(**ex)
    parts = []
    for strat in a.strategy:
        if strat == "initial":
            parts.append(EX._rollout_samples(src, src.qpos0[None], src.qvel0[None], a.horizon, a.stride, None,
                                             "initial"))
        elif strat == "scatter":
            if src.sampler is None and not (src.mjm.jnt_type == 0).any():
                C.log(f"{name}: no sampler and no free bodies, skipping scatter")
                continue
            q0 = EX.scatter_starts(src, a.starts, a.seed)
            parts.append(EX._rollout_samples(src, q0, np.tile(src.qvel0, (len(q0), 1)), a.horizon, a.stride, None,
                                             "scatter"))
        elif strat == "perturb":
            q0 = EX.perturb_starts(src, a.starts, a.jitter, a.seed)
            parts.append(EX._rollout_samples(src, q0, np.tile(src.qvel0, (len(q0), 1)), a.horizon, a.stride, None,
                                             "perturb"))
        elif strat == "actuation" and src.mjm.nu:
            q0 = np.tile(src.qpos0, (a.starts, 1))
            parts.append(EX._rollout_samples(src, q0, np.tile(src.qvel0, (a.starts, 1)), a.horizon, a.stride,
                                             lstates.Controls.random(a.sigma, seed=a.seed), "actuation"))
    sset = lstates.concat(*parts)
    finite = np.isfinite(sset.qpos).all(1) & np.isfinite(sset.qvel).all(1)
    sset = sset.subset(np.nonzero(finite)[0])
    _, deep = EX.loads(src, sset)
    keep = deep <= a.max_depth
    sset = sset.subset(np.nonzero(keep)[0])
    sset.source = dict(kind="explore", scene=name, **{k: v for k, v in a.__dict__.items()},
                       dropped_nonfinite=int((~finite).sum()), dropped_deep=int((~keep).sum()))
    sset.save(path)
    return sset


def timing_batch(sset, B, seed=0):
    """The `B` states every timing of a scene runs from: a weighted draw, the same for every candidate."""
    return lstates.batches(len(sset), B, "independent", weights=sset.weights, rng=np.random.default_rng(seed),
                           cover=False)[0]


def guarded(fn, *a, **kw):
    """`fn(...)`, or `None` and the error's text if it raised (out of memory, an overflow...)."""
    try:
        return fn(*a, **kw), None
    except Exception as e:  # noqa: BLE001
        wp.synchronize()
        return None, f"{type(e).__name__}: {str(e).splitlines()[0][:300] if str(e) else ''}"


# Scenes whose motion is not chaotic, so that a setting's paths can be held to the baseline's.
PATHS = ("incline", "backspin_ball", "painleve_box")
PATH_TOL = 5e-3


# The guard's batch: its metrics are means and percentiles over worlds, which 512 measure as
# well as thousands, and the rollouts at the baseline's full batch would be most of a tuning.
GUARD_B = 512


def guard(src, Bg, sset, section, rest, horizon=400):
    """The artifact metrics of one configuration at the guard's batch; `(artifacts, error)`."""
    def run():
        sim = C.build(src, Bg, section=section)
        art = C.artifacts(sim, sset, timing_batch(sset, Bg), horizon=horizon, rest=rest)
        C.free(sim)
        return art
    return guarded(run)


def measure(src, B, sset, idx, section, solver_kwargs=None, guard_horizon=400, steps=100, art=True, rest=None):
    """Throughput and artifacts of one configuration; `(result, error)`. An overflow is an error."""
    def run():
        sim = C.build(src, B, section=section, solver_kwargs=solver_kwargs)
        thr = C.throughput(sim, sset, idx, steps=steps)
        if thr["overflowed"]:
            C.free(sim)
            raise RuntimeError(f"overflowed during the timing: {', '.join(thr['overflow'])}")
        res = dict(throughput=thr)
        if art:
            res["artifacts"] = C.artifacts(sim, sset, idx, horizon=guard_horizon, rest=rest)
        C.free(sim)
        return res
    return guarded(run)


#######################################################################################
# Tiles
#######################################################################################


def tile_groups(src, share=0.8, most=6):
    """
    `{(vertex asset, SDF asset): [(v, s), ...]}` over the scene's directional pairs, the
    largest first (by the vertices they test). Only the largest groups that together test
    `share` of the vertices -- at most `most` of them -- are searched one by one; the rest
    are one group, `("rest", "rest")`: the SHARPA hand has some 250 distinct pairs of links,
    a search over each would take twenty minutes, and no one of them moves the total
    beyond the timing's noise.
    """
    keys = []
    for s in src.surfaces.surfaces:
        fp = ltuning.surface_fingerprint(s)
        keys.append("plane" if fp["kind"] == "plane" else fp["sha"])
    groups = {}
    for v, s in src.surfaces.pairs:
        groups.setdefault((keys[v], keys[s]), []).append((int(v), int(s)))
    nv = lambda pairs: sum(len(np.asarray(src.surfaces.surfaces[v].v)) for v, _ in pairs)  # noqa: E731
    ordered = sorted(groups.items(), key=lambda kv: -nv(kv[1]))
    total = sum(nv(p) for _, p in ordered) or 1
    out, seen = {}, 0
    for k, pairs in ordered:
        if len(out) < most and seen < share * total:
            out[k] = pairs
            seen += nv(pairs)
        else:
            out.setdefault(("rest", "rest"), []).extend(pairs)
    return out


def tile_dict(groups, sizes):
    """A `tile_size` dict from a size per group (8 where a group has none)."""
    out = {None: 8}
    for g, pairs in groups.items():
        for p in pairs:
            out[p] = int(sizes.get(g, 8))
    return out


def time_collision(src, sim, sset, idx, col_kwargs, reps=5, nrep=20):
    """The Lodos collision pipeline alone on the frozen states `idx`, at 1.5x its own demand there: ms."""
    surf = src.surfaces
    body_ids = [int(sim.mjm.geom_bodyid[g]) for g in surf.geoms]
    lstates.install(sim, sset, idx)
    smooth.kinematics(sim.mrbd, sim.drbd)

    def ctx(kw):
        return sdf_types.make_collision_context(sim.mjm, sim.drbd, surf.pairs, surf.surfaces, body_ids, **kw)

    kw = dict(col_kwargs)
    for _ in range(6):
        mcol, dcol = ctx(kw)
        sdf_collide.collide(mcol, dcol)
        wp.synchronize()
        if not dcol.overflow.numpy().any():
            break
        kw["capacity_tile"] = min(1.0, 2 * kw.get("capacity_tile", 1.0))
        kw["capacity2"] = min(1.0, 2 * kw.get("capacity2", 1.0))
        del mcol, dcol
    else:
        raise RuntimeError("collision still overflows at capacity 1.0")
    peak = dcol.ct_peak.numpy()
    worstT = max(1, int(mcol.n_tile_slots))
    capT = min(1.0, 1.5 * max(1, int(peak[3])) / worstT)
    nCa1 = (max(1, int(np.ceil(capT * worstT))) * int(mcol.tile_K) if int(mcol.uniform_K)
            else max(1, int(np.ceil(capT * int(mcol.B) * int(mcol.n_slot_world)))))
    tight = dict(col_kwargs, capacity_tile=capT, capacity2=min(1.0, 1.5 * max(1, int(peak[1])) / nCa1))
    del mcol, dcol
    C.free()
    mcol, dcol = ctx(tight)
    replay = sdf_collide.make_collide_graph(mcol, dcol)
    for _ in range(5):
        replay()
    wp.synchronize()
    best = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(nrep):
            replay()
        wp.synchronize()
        best.append((time.perf_counter() - t0) / nrep)
    over = bool(dcol.overflow.numpy().any())
    del mcol, dcol, replay
    C.free()
    if over:
        raise RuntimeError("the timing context overflowed")
    return 1e3 * float(np.median(best))


def tune_tiles(src, sset, B, start, log):
    """Coordinate descent over the groups' tile sizes; returns `(tile_size dict, record)`."""
    groups = tile_groups(src)
    sim = C.build(src, B, section=start)
    idx = timing_batch(sset, B)
    base_kw = dict(start.get("collision", {}))
    base_kw.pop("tile_size", None)
    sizes = {g: 8 for g in groups}
    record = dict(groups={f"{k[0]}|{k[1]}": len(v) for k, v in groups.items()}, trials=[])
    t_best = time_collision(src, sim, sset, idx, dict(base_kw, tile_size=tile_dict(groups, sizes)))
    t_base = t_best
    record["trials"].append(dict(sizes={f"{k[0]}|{k[1]}": 8 for k in groups}, ms=t_best))
    log(f"  tiles: all 8: {t_best:.3f} ms")
    for g in groups:
        for size in TILE_SIZES:
            if size == sizes[g]:
                continue
            trial = dict(sizes)
            trial[g] = size
            t, err = guarded(time_collision, src, sim, sset, idx, dict(base_kw, tile_size=tile_dict(groups, trial)))
            record["trials"].append(dict(sizes={f"{k[0]}|{k[1]}": v for k, v in trial.items()}, ms=t, error=err))
            if t is not None and t < 0.97 * t_best:
                t_best, sizes = t, trial
        log(f"  tiles: group {g[0][:6]}|{g[1][:6]} ({len(groups[g])} segments) -> {sizes[g]}: {t_best:.3f} ms")
    C.free(sim)
    record.update(base_ms=t_base, best_ms=t_best, sizes={f"{k[0]}|{k[1]}": v for k, v in sizes.items()})
    if all(v == 8 for v in sizes.values()):
        return None, record
    return tile_dict(groups, sizes), record


#######################################################################################
# The driver
#######################################################################################


def fmt_art(a):
    out = [f"deep {a['deep_mean'] * 1e3:.2f}/{a['deep_p99'] * 1e3:.2f} mm"]
    if "creep" in a:
        out.append(f"creep {a['creep'] * 1e3:.3f} mm/s")
    if "rest_speed" in a:
        out.append(f"rest {a['rest_speed']:.3g}")
    if a["nan"] or a["escaped"]:
        out.append(f"nan {a['nan']} escaped {a['escaped']}")
    return ", ".join(out)


def section_without_peaks(sec):
    """A section's settings and the batch it was measured at, without its record of how."""
    return {k: copy.deepcopy(v) for k, v in sec.items() if k in ltuning.SECTIONS or k == "B"}


def tune_section(src, name, sset, B, idx, rest, path_tol, tiles, copied, a, tuned, rec, log):
    """
    Steps 2 and 4-6 for `src`: its baseline and tuned settings into `tuned` (the file) and
    what was tried into `rec` (the report). A step that fails is recorded, and ends it.
    """
    t0 = time.time()

    # 2. baseline
    base = C.baseline_section(src)
    b_res, b_err = measure(src, B, sset, idx, base, art=False)
    B_base = B
    while b_res is None and B_base > 64:
        log(f"{name}: baseline at B={B_base}: {b_err}")
        B_base //= 2
        b_res, b_err = measure(src, B_base, sset, timing_batch(sset, B_base), base, art=False)
    Bg = min(B_base, GUARD_B)
    if b_res is not None:
        a_base, g_err = guard(src, Bg, sset, base, rest)
        if a_base is None:
            b_res, b_err = None, f"the baseline's guard rollout: {g_err}"
        else:
            b_res["artifacts"] = a_base
    pub = dict(throughput=b_res["throughput"], artifacts=C.public(b_res["artifacts"]), guard_B=Bg) if b_res else {}
    rec["baseline"] = dict(B=B_base, settings=base, error_at_B=b_err if B_base != B else None, **pub)
    tuned["baseline"] = dict(B=B_base, settings=base, **pub)
    if b_res is None:
        log(f"{name}: baseline failed even at B={B_base}: {b_err}")
        return
    log(f"{name}: baseline B={B_base}: {b_res['throughput']['world_steps_per_s']:.4g} world-steps/s "
        f"art {fmt_art(b_res['artifacts'])}")
    art_base = b_res["artifacts"]

    # A variant that differs from the scene it copies only in its joint rows follows it: that
    # scene's budgets, tiles and batch size, with the capacities measured again for the joint
    # rows, held to this scene's own baseline. Only its constraint rows differ, so its choices
    # are the other scene's.
    follow = copied is not None
    Bc, idxc = B, idx
    keep = dict(collision=dict(tile_size=tiles)) if tiles else None
    if follow:
        Bc = int(copied.get("B", B))
        idxc = timing_batch(sset, Bc)
        keep = {k: {n: v for n, v in copied.get(k, {}).items() if n not in ltuning.CAPACITIES.get(k, ())}
                for k in ltuning.SECTIONS}
        if copied.get("collision", {}).get("capacity1", 1.0) < 1.0:
            keep["collision"]["capacity1"] = 0.999999
        rec["followed"] = a.copy_from
        log(f"{name}: following {a.copy_from}: B={Bc}, {keep}")

    # 4. capacities, at the tiles found
    (cal, err) = guarded(CAL.calibrate_section, src, sset, Bc, safety=a.safety, keep=keep, log=log)
    if cal is None:
        log(f"{name}: calibration failed: {err}")
        rec["error"] = err
        return
    sec, peaks, rounds = cal
    sec = section_without_peaks(sec)
    res, err = measure(src, Bc, sset, idxc, sec, art=False)
    if res is None:
        log(f"{name}: tuned capacities failed: {err}")
        rec["error"] = err
        return
    best_thr = res["throughput"]["world_steps_per_s"]
    rec["capacities"] = dict(throughput=res["throughput"], rounds=rounds)
    log(f"{name}: capacities: {best_thr:.4g} world-steps/s")
    # The guard's noise floor: capacities change no physics, only the order of sums, so
    # how far their artifacts land from the baseline's is the run-to-run spread.
    a_n, err = guard(src, Bg, sset, sec, rest)
    rn = None if a_n is None else dict(artifacts=a_n)
    if rn is not None and rn["artifacts"].get("overflow"):
        # a run that dropped work says nothing about the noise
        log(f"{name}: the capacities overflow in the guard's rollouts: {rn['artifacts']['overflow']}")
        rn = None
    nz = C.noise(rn["artifacts"], art_base) if rn is not None else {}
    ptol = path_tol
    if path_tol is not None and rn is not None and art_base["final"].size:
        # the paths' own spread: the capacities change nothing but the order of sums,
        # and that alone has moved a sliding box's path by a centimetre
        with np.errstate(invalid="ignore"):
            dev = float(np.nanmax(np.linalg.norm(rn["artifacts"]["final"] - art_base["final"], axis=2)))
        nz["path"] = dev
        ptol = max(path_tol, 2.0 * dev)
    rec["noise"] = nz

    # the context compacts the tile filter's launch only if that is faster
    if src.surfaces is not None and "capacity1" not in sec["collision"] and not follow:
        trial = copy.deepcopy(sec)
        trial["collision"]["capacity1"] = 0.999999
        r1, _ = guarded(CAL.calibrate_section, src, sset, B, safety=a.safety, keep=trial, log=lambda *x: None)
        if r1 is not None:
            s1 = section_without_peaks(r1[0])
            m1, _ = measure(src, B, sset, idx, s1, art=False)
            if m1 is not None:
                rec["compacted"] = m1["throughput"]
                if m1["throughput"]["world_steps_per_s"] > 1.03 * best_thr:
                    sec, peaks, best_thr = s1, r1[1], m1["throughput"]["world_steps_per_s"]
                    log(f"{name}: compacting the tile filter: {best_thr:.4g}")

    # 5. budgets, guarded
    kwargs = {}
    if follow:
        kwargs = dict(keep.get("solver_kwargs", {}))
        if "contact_reduction" in keep.get("sim", {}):
            kwargs["contact_reduction"] = keep["sim"]["contact_reduction"]
    if not a.no_budgets and not follow:
        cands = list(BUDGETS) if src.defaults().get("mode", "exact") != "mj" else []  # mj runs no loop to trim
        if src.surfaces is not None and not src.sim_kwargs.get("contact_reduction"):
            cands.append(dict(contact_reduction=True))
        rec["budgets"] = []
        for cand in cands:
            kw = dict(kwargs, **cand)
            trial = copy.deepcopy(sec)
            trial["solver_kwargs"].update({k: v for k, v in kw.items() if k != "contact_reduction"})
            if "contact_reduction" in kw:
                trial["sim"]["contact_reduction"] = kw["contact_reduction"]
            a_c, err = guard(src, Bg, sset, trial, rest)
            rB, errB = measure(src, B, sset, idx, trial, art=False)
            if a_c is None or a_c.get("overflow") or (rB is None and "overflowed" in (errB or "")):
                # a budget changes the dynamics, and with them the demand: size the candidate for itself
                cal, err = guarded(CAL.calibrate_section, src, sset, B, safety=a.safety, keep=trial,
                                   log=lambda *x: None)
                if cal is not None:
                    trial = section_without_peaks(cal[0])
                    a_c, err = guard(src, Bg, sset, trial, rest)
                    rB, errB = measure(src, B, sset, idx, trial, art=False)
            if a_c is None or rB is None:
                rec["budgets"].append(dict(candidate=cand, error=err or errB))
                continue
            r = dict(artifacts=a_c)
            ok, why = C.no_worse(r["artifacts"], art_base, path_tol=ptol, noise=nz)
            # the speed is judged at the tuning batch
            thr = rB["throughput"]["world_steps_per_s"]
            rec["budgets"].append(dict(candidate=cand, throughput=thr, artifacts=C.public(r["artifacts"]), ok=ok,
                                       why=why))
            if ok and thr > 1.03 * best_thr:
                kwargs, sec, best_thr = kw, trial, thr
                log(f"{name}: {cand}: {thr:.4g} world-steps/s, kept")
            else:
                log(f"{name}: {cand}: {thr:.4g} world-steps/s, {'not faster' if ok else why}")

    # 6. batch size: each size calibrated at its own batch, from the last one's settings
    B_best = Bc
    if not a.no_batch_sweep and not follow:
        rec["batch"] = [dict(B=B, world_steps_per_s=best_thr)]
        for mult in (2, 4, 8, 16):
            Bm = B * mult
            cal, err = guarded(CAL.calibrate_section, src, sset, Bm, safety=a.safety, keep=sec, log=lambda *x: None)
            r = thr = None
            if cal is not None:
                secm = section_without_peaks(cal[0])
                r, err = measure(src, Bm, sset, timing_batch(sset, Bm), secm, art=False)
                thr = r["throughput"]["world_steps_per_s"] if r else None
            rec["batch"].append(dict(B=Bm, world_steps_per_s=thr, error=err))
            log(f"{name}: B={Bm}: {thr if thr is None else f'{thr:.4g}'} {err or ''}")
            if thr is None or thr < 1.05 * best_thr:
                break
            B_best, best_thr, sec, peaks = Bm, thr, secm, cal[1]

    # the final check: the tuned settings' artifacts against the baseline's, at the baseline's batch
    a_f, err = guard(src, Bg, sset, sec, rest)
    fin = None if a_f is None else dict(artifacts=a_f)
    ok, why = (C.no_worse(fin["artifacts"], art_base, path_tol=ptol, noise=nz) if fin else (False, err))
    if not ok and kwargs:
        # The budgets passed one rollout and failed this one: fall back to the capacities
        # alone, which change nothing, recalibrated at the batch found.
        log(f"{name}: final check failed with {kwargs} ({why}); falling back to the solver's own budget")
        rec["fallback"] = dict(budgets=kwargs, why=why)
        plain = copy.deepcopy(sec)
        for k in kwargs:
            plain["solver_kwargs"].pop(k, None)
            plain["sim"].pop(k, None)
        cal, err = guarded(CAL.calibrate_section, src, sset, B_best, safety=a.safety, keep=plain, log=lambda *x: None)
        if cal is not None:
            plain = section_without_peaks(cal[0])
            r, err = measure(src, B_best, sset, timing_batch(sset, B_best), plain, art=False)
            a_f2, err2 = guard(src, Bg, sset, plain, rest)
            fin2 = None if a_f2 is None else dict(artifacts=a_f2)
            if r is not None and fin2 is not None:
                sec, peaks, kwargs = plain, cal[1], {}
                best_thr, fin = r["throughput"]["world_steps_per_s"], fin2
                ok, why = C.no_worse(fin["artifacts"], art_base, path_tol=ptol, noise=nz)
    tuned.update(copy.deepcopy(sec))
    tuned["peaks"] = peaks
    tuned["B"] = B_best
    tuned["throughput"] = dict(world_steps_per_s=best_thr, B=B_best)
    tuned["artifacts"] = dict(tuned=C.public(fin["artifacts"]) if fin else None, baseline=C.public(art_base), ok=ok,
                              why=why)
    base_thr = b_res["throughput"]["world_steps_per_s"]
    rec.update(tuned=dict(B=B_best, world_steps_per_s=best_thr, settings=sec, artifacts_ok=ok, why=why),
               speedup=best_thr / base_thr)
    log(f"{name}: {base_thr:.4g} -> {best_thr:.4g} world-steps/s ({best_thr / base_thr:.2f}x) at B={B_best}"
        f"{'' if ok else ' -- ARTIFACTS: ' + why} ({time.time() - t0:.0f} s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene")
    ap.add_argument("--states")
    ap.add_argument("--B", type=int)
    ap.add_argument("--out-dir", default="tools/performance_tuning/tuned")
    ap.add_argument("--states-dir", default="results/tuning/states")
    ap.add_argument("--safety", type=float, default=1.5)
    ap.add_argument("--no-budgets", action="store_true")
    ap.add_argument("--no-batch-sweep", action="store_true")
    ap.add_argument("--copy-from", help="a tuned scene to follow: its tiles, budgets and batch size (jlimit variants)")
    ap.add_argument("--keep-tiles", action="store_true",
                    help="the scene's tuning file's tile sizes, rather than searching them again")
    ap.add_argument("--mode", choices=cd_driver.MODES,
                    help="tune under this mode rather than the scene's own: to <out-dir>/<mode>/")
    a = ap.parse_args()

    t_start = time.time()
    name = C.scene_name(a.scene)
    src = C.scene(a.scene)
    ex, B_family = family(name)
    B = a.B or B_family
    out_dir = Path(a.out_dir)
    states_dir = Path(a.states_dir)
    log = C.log
    if a.mode is not None and a.mode != src.defaults().get("mode", "exact"):
        # Another mode: every build of the scene runs it, and its file and states go beside the scene's own
        src = dataclasses.replace(src, solver_defaults=dict(src.solver_defaults, mode=a.mode))
        out_dir, states_dir = out_dir / a.mode, states_dir / a.mode
        log(f"{name}: tuning under {a.mode} -> {out_dir}")

    # 1. states
    spath = Path(a.states) if a.states else states_dir / f"{name}.states.npz"
    if spath.exists():
        sset = lstates.load(spath, src.mjm)
        log(f"{name}: {len(sset)} states from {spath}")
    else:
        sset = make_states(src, name, spath, ex)
        log(f"{name}: explored {len(sset)} states -> {spath} ({time.time() - t_start:.0f} s)")
    idx = timing_batch(sset, B)
    rest = C.settled(sset)
    path_tol = PATH_TOL if name.startswith(PATHS) else None
    log(f"{name}: {len(rest)} settled states for the rest test{'; paths held to the baseline' if path_tol else ''}")

    tuned = ltuning.new(name, B, "independent", ltuning.scene_fingerprint(src.mjm, src.surfaces),
                        states=dict(path=str(spath), digest=sset.digest(), size=len(sset), source=sset.source,
                                    horizon=200, controls=dict(kind="hold")),
                        safety=a.safety)
    report = dict(scene=name, B=B, states=len(sset))
    copied = ltuning.load(out_dir / f"{a.copy_from}.tuning.json") if a.copy_from else None
    kept = None
    if a.keep_tiles:
        path = out_dir / f"{name}.tuning.json"
        if not path.exists():
            raise SystemExit(f"--keep-tiles: there is no {path} to keep the tile sizes of")
        kept = ltuning.load(path)
        diff = ltuning.stale(kept, src.mjm, src.surfaces)
        if diff:
            raise SystemExit(f"--keep-tiles: {path} is stale ({'; '.join(diff[:3])}): search the tiles again")

    # 3. tiles
    tiles = None
    if src.surfaces is not None and any(len(np.asarray(s.v)) for s in src.surfaces.surfaces):
        earlier = copied if copied is not None else kept
        if earlier is not None and earlier.get("tiles"):
            tiles_rec = earlier["tiles"]
            tiles = earlier.get("collision", {}).get("tile_size")
            log(f"  tiles: kept from {a.copy_from or 'the file'}: {tiles if tiles else 'all 8'}")
        elif kept is not None:
            raise SystemExit(f"--keep-tiles: {out_dir / f'{name}.tuning.json'} holds no tile sizes to keep")
        else:
            start, _, _ = CAL.calibrate_section(src, sset, B, safety=3.0, log=lambda *x: None)
            start = section_without_peaks(start)
            res, err = guarded(tune_tiles, src, sset, B, start, log)
            tiles, tiles_rec = res if res is not None else (None, dict(error=err))
            if err:
                log(f"  tiles: failed: {err}")
        tuned["tiles"] = tiles_rec

    tune_section(src, name, sset, B, idx, rest, path_tol, tiles, copied, a, tuned, report, log)
    if a.mode is not None:
        tuned.setdefault("solver_kwargs", {})["mode"] = a.mode  # a build from the file runs the mode it was tuned under
    report["seconds"] = time.time() - t_start
    ltuning.save(tuned, out_dir / f"{name}.tuning.json")
    C.write_json(out_dir / f"{name}.report.json", report)
    log(f"{name}: done in {time.time() - t_start:.0f} s -> {out_dir / f'{name}.tuning.json'}")


if __name__ == "__main__":
    # every guard run at the baseline's batch, below a section's own, says so
    warnings.filterwarnings("ignore", message="the tuning file was measured at")
    try:
        main()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
