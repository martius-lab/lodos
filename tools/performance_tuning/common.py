"""
What the performance tools share: scenes by name, builds at a batch size from a settings
section, timing, and the artifact guard.

A scene is a built-in viewer scene's name (`viewer.scenes.catalog.BUILTIN`) or an MJCF
file the viewer can load (`lodos:` surfaces in its `<custom>` section). Every tool takes
a state set (`lodos.utils.states`) and runs the same states for everything it measures.
"""

from __future__ import annotations

import gc
import json
import math
import statistics
import sys
import time
import weakref
from pathlib import Path

import numpy as np
import warp as wp

wp.config.quiet = True  # no banner or kernel-load lines in the tools' logs

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lodos.utils import calibrate as lcalibrate  # noqa: E402
from lodos.utils import states as lstates  # noqa: E402
from lodos import step as lstep  # noqa: E402
from lodos.utils import tuning as ltuning  # noqa: E402
from viewer.scenes import catalog as vscenes  # noqa: E402


def scene(name, seed=0):
    """A `SceneSource`: a built-in scene's name, or a path to an MJCF file."""
    p = Path(name)
    if p.suffix.lower() == ".xml" and p.exists():
        return vscenes.from_mjcf(str(p))
    return vscenes.builtin(name, seed)


def scene_name(name):
    return Path(name).stem if Path(name).suffix.lower() == ".xml" else name


def as_tuning(src, B, section, layout="independent"):
    """
    A tuning file in memory, holding `section` -- how a tool builds from settings it is
    still measuring. The file's `B` is the section's own when it has one (the batch it was
    calibrated at), so that a build at another batch scales it as a file would.
    """
    B = int(section.get("B", B))
    t = ltuning.new(scene_name(src.name), B, layout, _fingerprint(src))
    t.update(section)
    return t


_FP = {}


def _fingerprint(src):
    """
    The scene's fingerprint, once per scene: hashing its surfaces is not free. Held with a
    weak reference to the scene, since a later scene in the same process can reuse a
    collected one's `id` (and would be handed its fingerprint: a stale-file warning on a
    multi-scene run).
    """
    hit = _FP.get(id(src))
    if hit is None or hit[0]() is not src:
        hit = _FP[id(src)] = (weakref.ref(src), ltuning.scene_fingerprint(src.mjm, src.surfaces))
    return hit[1]


def build(src, B, section=None, tuning=None, solver_kwargs=None, overrides=(), qpos=None):
    """`B` worlds of `src`, with a settings section or a tuning file (not both)."""
    if section is not None:
        tuning = as_tuning(src, B, section)
    return src.make_sim(solver_kwargs=solver_kwargs, overrides=overrides, B=B, qpos=qpos, tuning=tuning)


def free(sim=None):
    """Drops a `Sim` and returns its memory to the pool, so the next build is not measured beside it."""
    del sim
    gc.collect()
    wp.synchronize()


def mempool_mb():
    """What Warp's memory pool holds for this process, MB."""
    try:
        dev = wp.get_device()
        return wp.get_mempool_used_mem_current(dev) / 2**20
    except Exception:  # noqa: BLE001
        return None


#######################################################################################
# Timing
#######################################################################################


def throughput(sim, sset, idx, steps=200, warm=30, reps=3):
    """
    World-steps per second of `sim` from the states `idx`, its step captured as one CUDA
    graph and replayed: `warm` steps, then `steps` timed, `reps` times from the same
    states (the median). The same steps then run once more with every capacity's demand
    tracked (`lodos.utils.calibrate.Tracker`): a run that overflowed timed a different problem,
    and says so in `overflowed`.
    """
    replay = lstep.make_step_graph(sim)
    times = []
    for _ in range(reps):
        lstates.install(sim, sset, idx)
        for _ in range(warm):
            replay()
        wp.synchronize()
        t0 = time.perf_counter()
        for _ in range(steps):
            replay()
        wp.synchronize()
        times.append(time.perf_counter() - t0)
    t = statistics.median(times)
    # The same steps once more, untimed, with every counter tracked: MJWarp's contacts and
    # rows raise no flag of their own, and the tracker's launches would cost the timing.
    tracker = lcalibrate.Tracker(sim)
    checked = tracker.capture()
    tracker.reset()
    lstates.install(sim, sset, idx)
    for _ in range(warm + steps):
        checked()
    rep = tracker.report()
    return dict(world_steps_per_s=sim.B * steps / t, ms_per_step=1e3 * t / steps, B=sim.B, steps=steps, reps=reps,
                spread=(max(times) - min(times)) / t, overflowed=rep["overflowed"],
                overflow=lcalibrate.which_overflowed(rep), mempool_mb=mempool_mb())


#######################################################################################
# The artifact guard
#######################################################################################


def _deepest_per_world(sim):
    """`(B,)`: each world's deepest penetration now (m, positive; 0 without contact), synchronizing."""
    deep = np.zeros(sim.B)
    if sim.dcol is not None:
        n = int(sim.dcol.ct.numpy()[2])
        if n:
            np.maximum.at(deep, sim.dcol.b_nCa3.numpy()[:n], -sim.dcol.sdf_nCa3.numpy()[:n])
    d = sim.d
    n = min(int(d.nacon.numpy()[0]), int(d.naconmax))
    if n:
        np.maximum.at(deep, d.contact.worldid.numpy()[:n], -d.contact.dist.numpy()[:n])
    return np.maximum(deep, 0.0)


def _free_positions(sim, q):
    adr = [int(sim.mjm.jnt_qposadr[j]) for j in range(sim.mjm.njnt) if sim.mjm.jnt_type[j] == 0]
    return np.stack([q[:, a:a + 3] for a in adr], 1) if adr else np.zeros((len(q), 0, 3))


def settled(sset, speed=0.05, least=32, fallback=64):
    """
    The states of `sset` at rest: no dof faster than `speed` (m/s or rad/s). Fewer than
    `least` of them, and it is the `fallback` slowest states instead -- a scene that never
    quite stops (a rocking pile) is still held to its baseline's drift from there.
    """
    if not sset.qvel.shape[1]:
        return np.arange(len(sset))
    fastest = np.abs(sset.qvel).max(1)
    out = np.nonzero(fastest < speed)[0]
    if len(out) >= least:
        return out
    return np.argsort(fastest)[:min(fallback, len(sset))]


def artifacts(sim, sset, idx, horizon=400, every=20, rest=None, rest_horizon=500):
    """
    What a setting must not make worse.

    Over a rollout of `horizon` steps from `idx` (the scene in motion):

        nan         worlds whose state went non-finite
        escaped     worlds a body left: further than 100 m from the origin, or more than
                    1 m below where the lowest body started
        deep_mean   each world's deepest penetration, sampled every `every` steps, averaged
                    over worlds and samples (m)
        deep_p99    its 99th percentile (m)
        final       every world's free bodies' positions at the end (not saved): for
                    scenes that are not chaotic, `no_worse` compares paths

    Over `rest_horizon` steps from the settled states `rest` (indices into `sset`; the
    batch draws from them), when given:

        creep       the mean over worlds of the furthest a free body moved, per second (m/s)
        rest_speed  the mean over worlds of the fastest dof at the end
    """
    B = sim.B
    tracker = lcalibrate.Tracker(sim)
    replay = tracker.capture()
    tracker.reset()
    lstates.install(sim, sset, idx)
    z0 = sset.qpos[idx]
    depths = []
    for t in range(horizon):
        replay()
        if (t + 1) % every == 0:
            depths.append(_deepest_per_world(sim))
    q = sim.drbd.qpos.numpy().astype(np.float64)
    v = sim.drbd.qvel.numpy().astype(np.float64)
    bad = ~(np.isfinite(q).all(1) & np.isfinite(v).all(1))
    pos = _free_positions(sim, q)
    escaped = np.zeros(B, dtype=bool)
    if pos.shape[1]:
        low = float(_free_positions(sim, z0)[:, :, 2].min())
        with np.errstate(invalid="ignore"):
            escaped = (np.linalg.norm(pos, axis=2).max(1) > 100.0) | (pos[:, :, 2].min(1) < low - 1.0)
    dep = np.stack(depths) if depths else np.zeros((1, B))
    dep = dep[:, ~bad] if (~bad).any() else dep
    out = dict(nan=int(bad.sum()), escaped=int((escaped & ~bad).sum()),
               deep_mean=float(dep.mean()), deep_p99=float(np.percentile(dep, 99)), final=pos)
    if rest is not None and len(rest):
        # The settled states in some of the worlds and the batch's own in the rest: a batch of
        # the settled states alone is the shared layout, whose totals the capacities were not
        # measured for, and it can overflow them.
        k = min(len(rest), max(1, B // 2))
        ridx = np.asarray(idx).copy()
        ridx[:k] = np.asarray(rest)[np.random.default_rng(1).permutation(len(rest))[:k]]
        lstates.install(sim, sset, ridx)
        p0 = _free_positions(sim, sset.qpos[ridx])
        for _ in range(rest_horizon):
            replay()
        q = sim.drbd.qpos.numpy().astype(np.float64)[:k]
        v = sim.drbd.qvel.numpy().astype(np.float64)[:k]
        ok = np.isfinite(q).all(1) & np.isfinite(v).all(1)
        secs = rest_horizon * float(sim.mjm.opt.timestep)
        if p0.shape[1] and ok.any():
            moved = np.linalg.norm(_free_positions(sim, q) - p0[:k], axis=2).max(1)
            out["creep"] = float(moved[ok].mean() / secs)
        out["rest_speed"] = float(np.abs(v[ok]).max(1).mean()) if ok.any() else math.inf
        out["rest_nan"] = int((~ok).sum())
    out["overflow"] = lcalibrate.which_overflowed(tracker.report())
    return out


NOISY = ("deep_mean", "deep_p99", "creep", "rest_speed")


def noise(a, b):
    """
    How far two runs of the same physics land apart, per metric: `a` and `b` from
    `artifacts` on the same states under settings that change nothing but the order of
    floating-point sums (capacities). A chaotic scene's speed at rest varied twofold this
    way on the SHARPA grasp, which is the guard's own noise floor.
    """
    return {k: abs(a[k] - b[k]) for k in NOISY if k in a and k in b and np.isfinite(a[k]) and np.isfinite(b[k])}


def no_worse(cand, base, depth_ratio=1.25, depth_slack=2e-4, creep_ratio=1.5, creep_slack=2e-4,
             speed_ratio=2.0, speed_slack=0.02, path_tol=None, noise=None):
    """
    Whether `cand`'s artifacts are within tolerance of `base`'s (both from `artifacts`, on
    the same states): no new non-finite or escaped worlds; the mean and 99th-percentile
    penetration within `depth_ratio` and `depth_slack` (m); the creep at rest within
    `creep_ratio` and `creep_slack` (m/s) and the speed at rest within `speed_ratio` and
    `speed_slack`; and, with `path_tol` (m), every world's free bodies ending within it of
    where the baseline's did -- only for scenes that are not chaotic. `noise` (from
    `noise`) widens each continuous limit by twice the metric's run-to-run spread.
    Returns `(ok, why)`.
    """
    nz = {k: 2.0 * v for k, v in (noise or {}).items()}
    why = []
    if cand.get("overflow"):
        why.append(f"overflowed: {', '.join(cand['overflow'])}")
    if cand["nan"] > base["nan"] or cand.get("rest_nan", 0) > base.get("rest_nan", 0):
        why.append(f"{cand['nan']}+{cand.get('rest_nan', 0)} non-finite worlds")
    if cand["escaped"] > base["escaped"]:
        why.append(f"{cand['escaped']} escaped worlds (baseline {base['escaped']})")
    for k in ("deep_mean", "deep_p99"):
        if cand[k] > max(depth_ratio * base[k], base[k] + depth_slack) + nz.get(k, 0.0):
            why.append(f"penetration {k} {cand[k] * 1e3:.3f} mm (baseline {base[k] * 1e3:.3f})")
    if "creep" in base and cand.get("creep", math.inf) > max(creep_ratio * base["creep"],
                                                             base["creep"] + creep_slack) + nz.get("creep", 0.0):
        why.append(f"creep {cand.get('creep', math.inf) * 1e3:.3f} mm/s (baseline {base['creep'] * 1e3:.3f})")
    if "rest_speed" in base and cand.get("rest_speed", math.inf) > max(speed_ratio * base["rest_speed"],
                                                                       base["rest_speed"] + speed_slack) \
            + nz.get("rest_speed", 0.0):
        why.append(f"speed at rest {cand.get('rest_speed', math.inf):.4f} (baseline {base['rest_speed']:.4f})")
    if path_tol is not None and base["final"].size and cand["final"].shape == base["final"].shape:
        with np.errstate(invalid="ignore"):
            dev = float(np.nanmax(np.linalg.norm(cand["final"] - base["final"], axis=2)))
        if not dev <= path_tol:
            why.append(f"paths differ by {dev * 1e3:.2f} mm")
    return not why, "; ".join(why)


def public(art):
    """`artifacts`' result without the arrays, for a JSON record."""
    return {k: v for k, v in (art or {}).items() if k != "final"} if art is not None else None


#######################################################################################
# Settings
#######################################################################################


def baseline_section(src):
    """
    The settings a scene runs with before tuning, at `B`: its own per-world MJWarp
    capacities (the viewer's, `SceneSource.caps`), its solver defaults, and the collision
    defaults (`capacity_tile`, `capacity1`, `capacity2` at 1.0, `tile_size` 8) -- what
    `SceneSource.make_sim(B=B)` builds with no tuning file.
    """
    caps = {k: v for k, v in src.caps.items() if v is not None}
    sec = dict(collision={}, runtime=dict(caps), sim={}, solver_kwargs={})
    return sec


def start_section(rep_small, B_small, B, safety=2.0):
    """
    A first guess at `B` from a calibration at `B_small`: the same fractions (a smaller
    batch's mean is noisier, so its peak fraction is the larger) with `safety` on the
    absolute capacities that scale with `B`.
    """
    sec = lcalibrate.settings(rep_small, safety)
    mw = sec["runtime"]
    mw["naconmax"] = max(B, math.ceil(mw["naconmax"] * B / B_small))
    mw["nconmax"] = max(1, math.ceil(mw["naconmax"] / B))
    if "nccdmax" in mw:
        mw["nccdmax"] = min(mw["nconmax"], mw["nccdmax"])
    return sec


def _keys(obj):
    """`obj` with every dict key JSON can hold: per-pair tile sizes are keyed by tuples."""
    if isinstance(obj, dict):
        return {("default" if k is None else f"{k[0]},{k[1]}" if isinstance(k, tuple) else k): _keys(v)
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_keys(v) for v in obj]
    return obj


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_keys(obj), indent=1, default=ltuning._json_default) + "\n")
    return path


def log(*a):
    print(*a, flush=True)
