"""
Which shape pairs feed Newton's collision, per world: at points spread over the benchmark's run of a scene, Newton's
collision pipeline runs once more, and its mesh-triangle pairs (mesh shape, other shape, triangle) and its exported
contacts are counted per pair of shapes, named `label@body`. Newton's environment:

    viewer/newton/.venv/bin/python bench/lodos_mjwarp/profile/newton_pairs.py SCENE --sizes CAPS.json --B B \
        [--points 8] [--weld-filter] [--json OUT]
"""

from __future__ import annotations

import argparse
import collections
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
    ap.add_argument("--points", type=int, default=8)
    ap.add_argument("--weld-filter", action="store_true", help="leave out the pairs of bodies fixed to the world")
    ap.add_argument("--json")
    a = ap.parse_args()
    import warp as wp

    import collision_kernels as kern

    if a.weld_filter:
        import newton_stages

        newton_stages.weld_filter()
    a.out, a.starts, a.variant = None, None, None
    sim, step, ticks, n, dt, _ = kern.newton(a)
    labels = list(sim.model.shape_label)
    sbody = sim.model.shape_body.numpy()
    blabels = list(sim.model.body_label)

    def name(i):
        s = labels[i].rsplit("/", 1)[-1] if labels[i] else f"shape{i}"
        b = int(sbody[i])
        return f"{s}@{blabels[b].rsplit('/', 1)[-1] if b >= 0 else 'world'}"

    points = {int(n * (k + 0.5) / a.points) for k in range(a.points)}
    tri, con = collections.Counter(), collections.Counter()
    nph = sim.pipeline.narrow_phase
    for i in range(n):
        ticks.step(i, dt)
        step()
        if i in points:
            wp.synchronize()
            sim.pipeline.collide(sim.state_0, sim.contacts)
            wp.synchronize()
            if getattr(nph, "triangle_pairs_count", None) is not None:
                k = int(nph.triangle_pairs_count.numpy()[0])
                t = nph.triangle_pairs.numpy()[:k]
                for (sa, sb), c in collections.Counter(map(tuple, t[:, :2].tolist())).items():
                    tri[(name(sa), name(sb))] += c
            k = int(sim.contacts.rigid_contact_count.numpy()[0])
            s0 = sim.contacts.rigid_contact_shape0.numpy()[:k]
            s1 = sim.contacts.rigid_contact_shape1.numpy()[:k]
            for (sa, sb), c in collections.Counter(zip(s0.tolist(), s1.tolist())).items():
                con[(name(sa), name(sb))] += c
    norm = len(points) * a.B
    out = dict(scene=a.scene, B=a.B, points=len(points), weld_filter=bool(a.weld_filter),
               triangle_pairs={f"{x} x {y}": c / norm for (x, y), c in tri.most_common()},
               contacts={f"{x} x {y}": c / norm for (x, y), c in con.most_common()})
    print(f"{a.scene} B={a.B}, {len(points)} points; per world on average:")
    for what in ("triangle_pairs", "contacts"):
        print(f"  {what}:")
        for k, v in list(out[what].items())[:12]:
            print(f"    {v:9.1f}  {k}")
        print(f"    {sum(out[what].values()):9.1f}  total")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
