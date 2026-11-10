"""
Lodos's hull pipeline per world (`lodos.col.prim`): at points spread over the benchmark's run of a scene, the
broadphase's candidate pairs (`d.ncollision`) and the exact hull narrowphase's path counters
(`HybridNarrowphase.stats`: the pairs, those its cache settled, those it certified separated or in contact, the cold
GJK runs, EPA) and its contacts, as the step of the benchmark's graph left them.

    .venv/bin/python bench/lodos_mjwarp/profile/lodos_hybrid.py SCENE --sizes CAPS.json --B B --starts STARTS.npz \
        [--variant V] [--points 8] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--sizes", required=True)
    ap.add_argument("--B", type=int, required=True)
    ap.add_argument("--starts", required=True)
    ap.add_argument("--variant", default="default")
    ap.add_argument("--points", type=int, default=8)
    ap.add_argument("--json")
    a = ap.parse_args()
    import warp as wp

    import collision_kernels as kern

    a.groups = "contact"
    sim, step, ticks, n, dt, _ = kern.lodos(a)
    hy = sim.prim.hy
    if hy is None:
        raise SystemExit(f"{a.scene} has no hull pairs")
    points = {int(n * (k + 0.5) / a.points) for k in range(a.points)}
    acc = {"broadphase_pairs": 0.0}
    for i in range(n):
        ticks.step(i, dt)
        step()
        if i in points:
            wp.synchronize()
            acc["broadphase_pairs"] += float(sim.d.ncollision.numpy()[0]) / a.B
            for k, v in hy.stats().items():
                acc[k] = acc.get(k, 0.0) + v / a.B
            acc["contacts"] = acc.get("contacts", 0.0) + float(hy.shared["con_count"].numpy()[0]) / a.B
    k = len(points)
    out = dict(scene=a.scene, B=a.B, points=k, per_world={name: v / k for name, v in acc.items()})
    print(f"{a.scene} B={a.B}, per world, mean over {k} points:")
    for name, v in out["per_world"].items():
        if v:
            print(f"   {name:22s} {v:8.2f}")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
