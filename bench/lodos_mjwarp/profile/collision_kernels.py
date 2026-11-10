"""
The collision stages' kernels, one by one: the benchmark's graph runs the scene's whole duration, and at `--points`
steps spread over it the collision stages run once more uncaptured under Warp's CUDA activity timing
(`wp.timing_begin`), on the state the run reached. A stage recomputes its outputs from the poses, and the next step
recomputes them again, so the run is not changed.

    .venv/bin/python bench/lodos_mjwarp/profile/collision_kernels.py lodos SCENE GROUPS OUT --sizes S --B B --starts NPZ --variant V [--points 8]
    viewer/newton/.venv/bin/python bench/lodos_mjwarp/profile/collision_kernels.py newton SCENE OUT --sizes S --B B [--points 8]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

_HASH = re.compile(r"_[0-9a-f]{6,}$")


def timed(fn, agg):
    import warp as wp

    wp.synchronize()
    wp.timing_begin(cuda_filter=wp.TIMING_ALL)
    fn()
    res = wp.timing_end()
    total = 0.0
    for r in res:
        name = _HASH.sub("", r.name)
        a = agg.setdefault(name, [0.0, 0])
        a[0] += r.elapsed
        a[1] += 1
        total += r.elapsed
    return total


def lodos(a):
    import numpy as np
    import warp as wp

    import lodos_stages as P

    source = P.LS.builtin(a.scene)
    sizes = P.C.read_json(a.sizes)["sizes"]
    z = np.load(a.starts)
    q, v = z["qpos"][:a.B], z["qvel"][:a.B]
    sim = P.LP.build(source, a.groups, a.B, sizes=sizes, variant=a.variant, qpos=q, qvel=v)
    tr = P.lcalibrate.Tracker(sim)
    replay = tr.capture()
    dt = float(sim.mjm.opt.timestep)
    n = P.C.steps(source.name, dt)
    ticks = P.LP.Ticks(source, sim, n + 1)
    P.LSET.start(sim, q, v, np.zeros(a.B))
    ticks.k = None
    ticks.at(0)
    stages = {}
    if sim.prim.active:
        stages["prim_collision"] = lambda: (sim.prim.poses(sim.drbd.xpos, sim.drbd.xmat), sim.prim.collide())
    if sim.mcol is not None:
        stages["lodos_collision"] = lambda: P.sdf_collide.collide(sim.mcol, sim.dcol)
        if sim.solver.rm is not None:
            stages["reduce"] = lambda: P.cd_reduce.reduce(sim.solver.rm, sim.solver.rd, sim.solver.dcol)
    return sim, replay, ticks, n, dt, stages


def newton(a):
    import argparse as _ap

    import warp as wp

    import newton_stages as P

    s = P.C.OUT / "scenes"
    ctl = s / f"{a.scene}.controls.npz"
    args = _ap.Namespace(pkg=s / f"{a.scene}.pkg", options=s / f"{a.scene}.newton.json",
                         starts=s / f"{a.scene}.starts.npz", controls=str(ctl) if ctl.exists() else "-")
    pkg, opts, q, v, (rows, period) = P.NP.load(args)
    q, v = q[:a.B], v[:a.B]
    sizes = P.C.read_json(a.sizes)["sizes"]
    sim = P.NS.Sim(pkg, "mjwarp", B=a.B, caps=P.NS.Caps(**sizes), options=opts)
    sim.track()
    ticks = P.NP.Ticks(sim, rows, period)
    P.NP.start(sim, q, v)
    sim.step(3)
    wp.synchronize()
    dt, n = sim.dt, P.C.steps(pkg["name"], sim.dt)
    P.NP.start(sim, q, v)
    ticks.k = None
    ticks.at(0)
    stages = dict(
        collide=lambda: sim.pipeline.collide(sim.state_0, sim.contacts),
        convert_contacts=lambda: sim.solver._convert_contacts_to_mjwarp(sim.model, sim.state_0, sim.contacts))
    return sim, (lambda: sim.step(1)), ticks, n, dt, stages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("lodos", "newton"))
    ap.add_argument("scene")
    if "lodos" in sys.argv[1:2]:
        ap.add_argument("groups")
    ap.add_argument("out")
    ap.add_argument("--sizes", required=True)
    ap.add_argument("--B", type=int, required=True)
    ap.add_argument("--starts")
    ap.add_argument("--variant", default="default")
    ap.add_argument("--points", type=int, default=8)
    a = ap.parse_args()
    import warp as wp

    sim, step, ticks, n, dt, stages = (lodos if a.mode == "lodos" else newton)(a)
    points = {int(n * (k + 0.5) / a.points) for k in range(a.points)}
    agg = {k: {} for k in stages}
    totals = {k: 0.0 for k in stages}
    wp.synchronize()
    for i in range(n):
        ticks.step(i, dt)
        step()
        if i in points:
            for k, fn in stages.items():
                totals[k] += timed(fn, agg[k])
    out = dict(scene=a.scene, mode=a.mode, B=a.B, points=len(points), stages={})
    for k in stages:
        rows = sorted(((name, ms / len(points), cnt / len(points)) for name, (ms, cnt) in agg[k].items()),
                      key=lambda r: -r[1])
        out["stages"][k] = dict(ms=totals[k] / len(points), kernels=[dict(name=r[0], ms=r[1], launches=r[2])
                                                                     for r in rows])
    Path(a.out).write_text(json.dumps(out, indent=1))
    for k, s in out["stages"].items():
        print(f"{k}: {s['ms']:.3f} ms a call (uncaptured)")
        for r in s["kernels"][:12]:
            print(f"    {r['ms']:8.3f} ms  x{r['launches']:.0f}  {r['name']}")


if __name__ == "__main__":
    main()
