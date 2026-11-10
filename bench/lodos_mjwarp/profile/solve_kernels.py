"""
The constraint solve's kernels, one by one, beside the rest of the forward pass: the benchmark's graph runs the
scene's whole duration, and at `--points` steps spread over it one forward pass runs uncaptured under Warp's CUDA
activity timing, which a wrapper around the solve stops and restarts so its kernels are kept apart.

    .venv/bin/python bench/lodos_mjwarp/profile/solve_kernels.py lodos SCENE GROUPS OUT --sizes S --B B --starts NPZ --variant V [--points 8]
    viewer/newton/.venv/bin/python bench/lodos_mjwarp/profile/solve_kernels.py newton SCENE OUT --sizes S --B B [--points 8]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import collision_kernels as kern  # noqa: E402


class Split:
    """Timing results gathered into "solve" and "rest" by stopping the timer around the wrapped solve."""

    def __init__(self):
        self.agg = {"solve": {}, "rest": {}}
        self.ms = {"solve": 0.0, "rest": 0.0}

    def _take(self, part):
        import warp as wp

        for r in wp.timing_end():
            name = kern._HASH.sub("", r.name)
            a = self.agg[part].setdefault(name, [0.0, 0])
            a[0] += r.elapsed
            a[1] += 1
            self.ms[part] += r.elapsed

    def wrap(self, obj, attr):
        import warp as wp

        orig = getattr(obj, attr)
        me = self

        def wrapper(*a, **k):
            me._take("rest")
            wp.timing_begin(cuda_filter=wp.TIMING_ALL)
            out = orig(*a, **k)
            me._take("solve")
            wp.timing_begin(cuda_filter=wp.TIMING_ALL)
            return out

        setattr(obj, attr, wrapper)

    def forward(self, fn):
        import warp as wp

        wp.synchronize()
        wp.timing_begin(cuda_filter=wp.TIMING_ALL)
        fn()
        self._take("rest")


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
    ap.add_argument("--weld-filter", action="store_true")
    a = ap.parse_args()
    import warp as wp

    if a.weld_filter:
        import newton_stages

        newton_stages.weld_filter()
    sp = Split()
    if a.mode == "lodos":
        sim, step, ticks, n, dt, _ = kern.lodos(a)
        from lodos import step as lstep
        from lodos.cd.solver import nlsp

        sp.wrap(nlsp, "solve")
        if sim.solver.fused is not None:
            sp.wrap(sim.solver.fused, "solve")
        fwd = lambda: lstep.forward(sim)  # noqa: E731
        d = sim.d
    else:
        sim, step, ticks, n, dt, _ = kern.newton(a)
        import mujoco_warp._src.forward as mjf
        import mujoco_warp._src.solver as mjsol

        sp.wrap(mjsol, "solve")
        d = sim.solver.mjw_data
        m = sim.solver.mjw_model
        fwd = lambda: mjf.forward(m, d)  # noqa: E731
    points = {int(n * (k + 0.5) / a.points) for k in range(a.points)}
    iters, rows = [], []
    wp.synchronize()
    for i in range(n):
        ticks.step(i, dt)
        step()
        if i in points:
            sp.forward(fwd)
            iters.append(float(d.solver_niter.numpy().mean()))
            rows.append(float(d.nefc.numpy().mean()))
    k = len(points)
    out = dict(scene=a.scene, mode=a.mode, B=a.B, points=k, iters_mean=sum(iters) / k, rows_mean=sum(rows) / k,
               parts={})
    for part in ("solve", "rest"):
        lst = sorted(((nm, ms / k, c / k) for nm, (ms, c) in sp.agg[part].items()), key=lambda r: -r[1])
        out["parts"][part] = dict(ms=sp.ms[part] / k, kernels=[dict(name=r[0], ms=r[1], launches=r[2]) for r in lst])
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"iterations {out['iters_mean']:.2f}, rows {out['rows_mean']:.1f} per world")
    for part, s in out["parts"].items():
        nl = sum(r["launches"] for r in s["kernels"])
        print(f"{part}: {s['ms']:.3f} ms a forward pass (uncaptured), {nl:.0f} launches")
        for r in s["kernels"][:14]:
            print(f"    {r['ms']:8.3f} ms  x{r['launches']:.1f}  {r['name']}")


if __name__ == "__main__":
    main()
