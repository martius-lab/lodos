"""
One measurement of Newton's Kamino on a scene, in a process of its own in viewer/newton's environment
(`run.py` drives it):

    viewer/newton/.venv/bin/python bench/lodos_kamino/kamino_point.py measure PKG OPTIONS STARTS CONTROLS OUT
    viewer/newton/.venv/bin/python bench/lodos_kamino/kamino_point.py ab PKG OPTIONS STARTS CONTROLS OUT \
        --sizes SIZES
    viewer/newton/.venv/bin/python bench/lodos_kamino/kamino_point.py point PKG OPTIONS STARTS CONTROLS OUT \
        --sizes SIZES --B B --variant V

The scene comes as `viewer.newton_link.scene_package` writes it, its options (the viewer's Kamino settings, its
iterations `common.kamino_iterations`) as JSON, its starts and recorded controls as `.npz` (CONTROLS "-" for none).
`measure`: the starts as the worlds of one batch for the scene's whole run, each world's contacts counted in every
step -- the worlds are independent, so a world's demand is its start's alone -- and a world's capacity, 1.5 times
the most; and the states half-way through, which `ab` times from. `ab`: Kamino's execution options
(`common.KAMINO_VARIANTS`) against the default. `point`: B worlds from their own starts for the whole run, timed,
the demand tracked in the step (`newton_scene.Sim.track`), the lost worlds counted.

`Sim.set_state` resets Kamino and drops the captured step, so the first step after a start captures it again:
that step runs before the clock starts, and a run is timed over its other steps.
"""

from __future__ import annotations

import argparse
import gc
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "viewer" / "newton"))
sys.path.insert(0, str(ROOT))

import newton_scene as NS  # noqa: E402

from bench.lodos_kamino import common as C  # noqa: E402
from bench.lodos_mjwarp.newton_point import Ticks, load  # noqa: E402

wp.config.quiet = True


def free_joints(m):
    return [int(m.jnt_qposadr[j]) for j in range(m.njnt) if int(m.jnt_type[j]) == 0]  # mjJNT_FREE


def build(pkg, opts, B, contacts_per_world, variant="default"):
    sim = NS.Sim(pkg, "kamino", B=B, caps=NS.Caps(contacts_per_world=int(contacts_per_world)),
                 options={**opts, **C.KAMINO_VARIANTS[variant]})
    sim.track(per_world=True)
    return sim


def status(sim):
    """Kamino's last solve over the worlds: the mean PADMM iterations and the share of worlds that converged."""
    st = sim.solver.status.numpy()
    return dict(mean_iterations=float(np.mean(st["iterations"])), converged=float(np.mean(st["converged"])))


def measure(pkg, opts, q, v, rows, period):
    """The starts as one batch for the whole run: each world's most contacts, its capacity, the half-way states."""
    B, cap = len(q), C.KAMINO_CONTACTS0
    t0 = time.perf_counter()
    for attempt in range(6):
        sim = build(pkg, opts, B, cap)
        dt, n = sim.dt, C.steps(pkg["name"], sim.dt)
        ticks = Ticks(sim, rows, period)
        sim.set_state(q, v)
        sim.reset_peaks()
        mid = None
        for i in range(n):
            ticks.step(i, dt)
            sim.step(1)
            if i == n // 2:
                mid = sim.get_state()
            if (i + 1) % max(1, n // 10) == 0:
                print(f"step {i + 1}: {time.perf_counter() - t0:.0f} s, {sim.peaks()}", flush=True)
        p = sim.peaks()
        if p["world_contacts"] <= cap:
            break
        print(f"a world had {p['world_contacts']} contacts, over {cap}: doubled", flush=True)
        cap *= 2
        del sim
        gc.collect()
    else:
        raise RuntimeError(f"{pkg['name']}: the contacts kept overflowing")
    return p, mid, dict(contacts_per_world=max(1, math.ceil((1.0 + C.OVERHEAD) * p["world_contacts"]))), dict(
        steps=n, dt=dt, seconds=time.perf_counter() - t0, attempts=attempt + 1, measured_contacts_per_world=cap,
        status=status(sim))


def too_large(e):
    """
    Whether an error says the batch does not fit: out of memory, or Kamino's dense Delassus matrices -- every world's,
    (its constraint rows)^2 each, in one array -- past 2^31 entries, which Warp's 32-bit array sizes refuse.
    """
    return any(s in str(e) for s in ("allocate", "out of memory", "cudaErrorMemory", "Array shapes must"))


def ab(pkg, opts, sizes, mid, B, reps=3, nsteps=100, warmup=10):
    """
    Each execution option's world-steps/s at `B` worlds from the half-way states, its median of `reps`; at half as
    many worlds while the default does not fit (`too_large`).
    """
    while True:
        out = variants(pkg, opts, sizes, mid, B, reps, nsteps, warmup)
        if not too_large(out["default"].get("error", "")) or B <= C.B0:
            break
        B //= 2
    base = out["default"].get("world_steps_per_s")
    if not base:
        raise RuntimeError(f"the default failed: {out['default']['error']}")
    best = max((k for k in out if "world_steps_per_s" in out[k]), key=lambda k: out[k]["world_steps_per_s"])
    chosen = best if out[best]["world_steps_per_s"] > (1.0 + C.MIN_GAIN) * base else "default"
    return dict(B=B, variants=out, chosen=chosen)


def variants(pkg, opts, sizes, mid, B, reps, nsteps, warmup):
    k = np.arange(B) % len(mid["qpos"])
    q, v = mid["qpos"][k], mid["qvel"][k]
    out = {}
    for name in C.KAMINO_VARIANTS:
        sim = None
        gc.collect()  # the variant before, gone before this one is built
        try:
            sim = build(pkg, opts, B, sizes["contacts_per_world"], name)
            took = []
            for r in range(reps + 1):
                sim.set_state(q, v)
                sim.step(1)  # captures the step again (`set_state`)
                wp.synchronize()
                t0 = time.perf_counter()
                sim.step(warmup if r == 0 else nsteps)
                wp.synchronize()
                if r:
                    took.append(time.perf_counter() - t0)
            out[name] = dict(world_steps_per_s=B * nsteps / float(np.median(took)), status=status(sim))
        except Exception as e:  # noqa: BLE001 -- recorded: an option a model does not build with
            out[name] = dict(error=f"{type(e).__name__}: {e}"[:300])
        sim = None
        gc.collect()
        wp.synchronize()
    return out


def point(pkg, opts, sizes, variant, B, q, v, rows, period):
    """B worlds from their own starts for the whole run, timed; its demand, overflows and lost worlds."""
    q, v = q[:B], v[:B]
    t_build = time.perf_counter()
    sim = build(pkg, opts, B, sizes["contacts_per_world"], variant)
    ticks = Ticks(sim, rows, period)
    sim.set_state(q, v)
    sim.step(3)  # loading and capturing everything
    wp.synchronize()
    build_s = time.perf_counter() - t_build
    dt, n = sim.dt, C.steps(pkg["name"], sim.dt)
    sim.set_state(q, v)
    sim.reset_peaks()
    ticks.k = None
    ticks.step(0, dt)
    sim.step(1)  # the run's first step, which captures the step again (`set_state`): not timed
    wp.synchronize()
    t0 = time.perf_counter()
    for i in range(1, n):
        ticks.step(i, dt)
        sim.step(1)
        if (i + 1) % max(1, n // 10) == 0:  # when each tenth was queued, a stall's trace in the log
            print(f"step {i + 1}: {time.perf_counter() - t0:.1f} s", flush=True)
    wp.synchronize()
    wall = time.perf_counter() - t0
    peaks = sim.peaks()
    over = ["contacts"] if peaks["world_contacts"] > int(sim.caps.contacts_per_world) else []
    qe, ve = sim.get_state()
    nan = ~(np.isfinite(qe).all(1) & np.isfinite(ve).all(1))
    esc = C.escaped(np.asarray(qe, np.float64), q, free_joints(sim.mjm)) & ~nan
    k = n - 1  # the timed steps
    return dict(B=B, steps=n, timed_steps=k, dt=dt, seconds=wall, world_steps_per_s=B * k / wall,
                sim_seconds_per_s=B * k * dt / wall, build_seconds=build_s, overflowed=bool(over), overflow=over,
                peaks=peaks, nan_worlds=int(nan.sum()), escaped_worlds=int(esc.sum()), variant=variant, sizes=sizes,
                status=status(sim), final_qpos0=np.asarray(qe[0], np.float64).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=("measure", "ab", "point"))
    for k in ("pkg", "options", "starts", "controls", "out"):
        ap.add_argument(k)
    ap.add_argument("--sizes")
    ap.add_argument("--B", type=int)
    ap.add_argument("--variant", default="default")
    a = ap.parse_args()
    t0 = time.perf_counter()
    res = dict(task=a.task, sim="kamino")
    try:
        pkg, opts, q, v, (rows, period) = load(a)
        res["scene"] = pkg["name"]
        if a.task == "measure":
            peaks, mid, sizes, rec = measure(pkg, opts, q, v, rows, period)
            np.savez(a.out.replace(".json", ".mid.npz"), qpos=mid[0], qvel=mid[1])
            res.update(peaks=peaks, sizes=sizes, **rec)
        elif a.task == "ab":
            mid = dict(np.load(a.sizes.replace(".json", ".mid.npz")))
            res.update(ab(pkg, opts, C.read_json(a.sizes)["sizes"], mid, a.B or C.ab_batch(pkg["name"])))
        else:
            res.update(point(pkg, opts, C.read_json(a.sizes)["sizes"], a.variant, a.B, q, v, rows, period))
        res["ok"] = True
    except Exception as e:  # noqa: BLE001 -- recorded, the driver decides
        traceback.print_exc()
        res.update(ok=False, error=f"{type(e).__name__}: {e}"[:500])
    res["process_seconds"] = time.perf_counter() - t0
    C.write_json(a.out, res)


if __name__ == "__main__":
    main()
