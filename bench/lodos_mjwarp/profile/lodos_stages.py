"""
Where a Lodos (mj mode) step's time goes, over the benchmark's own run of a scene.

    .venv/bin/python bench/lodos_mjwarp/profile/lodos_stages.py SCENE GROUPS OUT --sizes CAPS.json --B B --starts STARTS.npz [--variant V]

Built exactly as `bench.lodos_mjwarp.lodos_point.point` builds it (setting, solref, capacities, variant, starts,
recorded controls). Two runs of the scene's whole duration from the same starts:

1. plain: the benchmark's graph (`Tracker.capture`), timed by wall clock -- the throughput to check against
   `report/best.json`;
2. traced: the same step captured with timing events at its stage boundaries (`tracer.Tracer`), every step
   launched, every `--every`-th read: each stage's GPU time, and the workload counters at that step.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import warp as wp  # noqa: E402

from bench import lodos_setting as LSET  # noqa: E402
from bench.lodos_mjwarp import common as C  # noqa: E402
from bench.lodos_mjwarp import lodos_point as LP  # noqa: E402
from bench.lodos_mjwarp import run as R  # noqa: E402
from lodos import step as lstep  # noqa: E402
from lodos.cd import reduce as cd_reduce  # noqa: E402
from lodos.cd.solver import constraint as lconstraint  # noqa: E402
from lodos.cd.solver import nlsp  # noqa: E402
from lodos.col.prim import driver as prim_driver  # noqa: E402
from lodos.col.sdf import collide as sdf_collide  # noqa: E402
from lodos.rbd import forward as rbd_forward  # noqa: E402
from lodos.rbd import integrate as rbd_integrate  # noqa: E402
from lodos.utils import calibrate as lcalibrate  # noqa: E402
from tracer import Accumulator, Tracer  # noqa: E402
from viewer.scenes import catalog as LS  # noqa: E402


def install(T, sim, tr):
    T.wrap(rbd_forward, "fwd_rbd", "rbd")
    T.wrap(sim.prim, "poses", "prim_poses")
    T.wrap(sim.prim, "collide", "prim_collision")
    T.wrap(prim_driver.prim_collision_driver, "nxn_broadphase", "broadphase")
    T.wrap(prim_driver.prim_collision_driver, "sap_broadphase", "broadphase")
    T.wrap(prim_driver, "convex_narrowphase", "convex_narrow")
    T.wrap(prim_driver, "primitive_narrowphase", "primitive_narrow")
    if sim.prim.hy is not None:
        T.wrap(sim.prim.hy, "launch", "hybrid")
    T.wrap(sdf_collide, "collide", "lodos_collision")
    T.wrap(sdf_collide, "broadphase_bbox", "bbox")
    T.wrap(sdf_collide, "broadphase_sdf", "sdf_prefilter")
    T.wrap(sdf_collide, "narrowphase_sdf", "narrow")
    T.wrap(sim.solver, "contact_dynamics", "contact_dynamics")
    T.wrap(cd_reduce, "reduce", "reduce")
    if sim.solver.groups is not None:
        T.wrap(sim.solver.groups, "before_export", "groups_before")
        T.wrap(sim.solver.groups, "export", "groups_export")
    T.wrap(lconstraint, "make_constraint", "make_constraint")
    if sim.solver.fused is not None:
        T.wrap(sim.solver.fused, "solve", "solve")
    if sim.solver.rowspace is not None:
        T.wrap(sim.solver.rowspace, "prepare", "rowspace_prepare")
    T.wrap(nlsp, "solve", "solve")
    T.wrap(rbd_integrate, "integrate", "integrate")
    T.wrap(tr, "launch", "track")


def counters(sim, B):
    d = sim.d
    out = dict(contacts=float(d.nacon.numpy()[0]) / B)
    nefc = d.nefc.numpy()
    out["rows_mean"], out["rows_max"] = float(nefc.mean()), float(nefc.max())
    it = d.solver_niter.numpy()
    out["iters_mean"], out["iters_max"] = float(it.mean()), float(it.max())
    if sim.dcol is not None:
        ct = sim.dcol.ct.numpy()
        out.update(lodos_stage1=float(ct[0]) / B, lodos_voxel=float(ct[1]) / B, lodos_contacts=float(ct[2]) / B,
                   lodos_tiles=float(ct[3]) / B)
    return out


def run_once(source, groups, sizes, variant, B, q, v, every):
    sim = LP.build(source, groups, B, sizes=sizes, variant=variant, qpos=q, qvel=v)
    tr = lcalibrate.Tracker(sim)
    replay = tr.capture()
    for _ in range(2):
        replay()
    wp.synchronize()
    dt = float(sim.mjm.opt.timestep)
    n = C.steps(source.name, dt)
    ticks = LP.Ticks(source, sim, n + 1)

    # 1. plain, as the benchmark times it
    LSET.start(sim, q, v, np.zeros(B))
    tr.reset()
    ticks.k = None
    ticks.at(0)
    wp.synchronize()
    t0 = time.perf_counter()
    for i in range(n):
        ticks.step(i, dt)
        replay()
    wp.synchronize()
    wall = time.perf_counter() - t0
    rep = tr.report()
    plain = dict(seconds=wall, ms_per_step=1e3 * wall / n, world_steps_per_s=B * n / wall,
                 overflowed=bool(rep["overflowed"]), overflow=lcalibrate.which_overflowed(rep))

    # 2. traced: the same step, timing events at the stage boundaries
    T = Tracer()
    install(T, sim, tr)
    try:
        with wp.ScopedCapture() as cap:
            with T.scope("step"):
                lstep.fwd_step(sim)
                tr.launch()
    finally:
        T.unwrap()
    graph = cap.graph
    LSET.start(sim, q, v, np.zeros(B))
    tr.reset()
    ticks.k = None
    ticks.at(0)
    acc = Accumulator()
    wp.synchronize()
    t0 = time.perf_counter()
    for i in range(n):
        ticks.step(i, dt)
        wp.capture_launch(graph)
        if i % every == every // 2:
            wp.synchronize()
            acc.add(T.read(), counters(sim, B))
    wp.synchronize()
    traced_wall = time.perf_counter() - t0
    out = dict(plain=plain, traced=acc.result(), traced_seconds=traced_wall, steps=n, dt=dt, B=B,
               variant=variant, options={k: v for k, v in sim.options.items() if isinstance(v, (int, float, str, bool))
                                         or v is None})
    del graph, replay, tr, sim
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("groups", choices=C.GROUPS)
    ap.add_argument("out")
    ap.add_argument("--sizes", required=True)
    ap.add_argument("--B", type=int, required=True)
    ap.add_argument("--starts", required=True)
    ap.add_argument("--variant", default="default")
    ap.add_argument("--every", type=int, default=20)
    a = ap.parse_args()
    source = LS.builtin(a.scene)
    sizes = C.read_json(a.sizes)["sizes"]
    z = np.load(a.starts)
    q, v = z["qpos"][:a.B], z["qvel"][:a.B]
    cfg = f"lodos_{a.groups}"
    for attempt in range(4):
        out = run_once(source, a.groups, sizes, a.variant, a.B, q, v, a.every)
        if not out["plain"]["overflowed"]:
            break
        print(f"overflowed {out['plain']['overflow']}: capacities grown, redone", flush=True)
        sizes = R.grow(cfg, sizes, out["plain"]["overflow"], a.B)
    out.update(scene=a.scene, sim=cfg, sizes=sizes, attempts=attempt + 1)
    Path(a.out).write_text(json.dumps(out, indent=1))
    p = out["plain"]
    print(f"{a.scene} {cfg} B={a.B}: {p['world_steps_per_s']:,.0f} world-steps/s, {p['ms_per_step']:.3f} ms a step;"
          f" traced step {out['traced']['ms']['step']:.3f} ms", flush=True)


if __name__ == "__main__":
    main()
