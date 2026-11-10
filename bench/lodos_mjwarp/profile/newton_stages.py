"""
Where a MuJoCo Warp step (Newton's SolverMuJoCo on Newton's collision pipeline) spends its time, over the
benchmark's own run of a scene. Newton's environment:

    viewer/newton/.venv/bin/python bench/lodos_mjwarp/profile/newton_stages.py SCENE OUT --sizes CAPS.json --B B [--every N]

Built exactly as `bench.lodos_mjwarp.newton_point.point` builds it (package, options, capacities, starts, recorded
controls, the tracking in the step). Two runs of the whole duration from the same starts: plain (the benchmark's
graph, wall clock) and traced (the same step with timing events at its stage boundaries, read every N-th step).
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

from bench.lodos_mjwarp import common as C  # noqa: E402
from bench.lodos_mjwarp import newton_point as NP  # noqa: E402
from bench.lodos_mjwarp import run as R  # noqa: E402
import mujoco_warp._src.constraint as mjc  # noqa: E402
import mujoco_warp._src.forward as mjf  # noqa: E402
import mujoco_warp._src.smooth as mjs  # noqa: E402
import mujoco_warp._src.solver as mjsol  # noqa: E402
from tracer import Accumulator, Tracer  # noqa: E402

NS = NP.NS


def weld_filter():
    """
    From now on, every world Newton builds leaves out the shape pairs between two bodies fixed to the world (a
    fixed joint to it), as MuJoCo leaves out the pairs of bodies welded to the same body: the threading's bolt and
    board, which Newton's importer makes bodies of their own.
    """
    import newton

    orig = NS.world_builder

    def wb(pkg, solver, options=None):
        b, mjm, targets = orig(pkg, solver, options)
        fixed = {int(c) for c, p, t in zip(b.joint_child, b.joint_parent, b.joint_type)
                 if int(p) == -1 and int(t) == int(newton.JointType.FIXED)}
        shapes = [s for s, body in enumerate(b.shape_body) if int(body) in fixed or int(body) == -1]
        n = 0
        for i, x in enumerate(shapes):
            for y in shapes[i + 1:]:
                if int(b.shape_body[x]) != int(b.shape_body[y]):
                    b.add_shape_collision_filter_pair(x, y)
                    n += 1
        print(f"weld filter: {len(fixed)} fixed bodies, {n} shape pairs left out", flush=True)
        return b, mjm, targets

    NS.world_builder = wb


def install(T, sim):
    T.wrap(sim.state_0, "clear_forces", "clear_forces")
    T.wrap(sim.pipeline, "collide", "collide")
    T.wrap(sim.pipeline.broad_phase, "launch", "broadphase")
    T.wrap(sim.pipeline.narrow_phase, "launch_custom_write", "narrowphase")
    T.wrap(sim.solver, "step", "solver_step")
    T.wrap(sim.solver, "_apply_mjc_control", "control")
    T.wrap(sim.solver, "_update_mjc_data", "state_in")
    T.wrap(sim.solver, "_convert_contacts_to_mjwarp", "convert_contacts")
    T.wrap(sim.solver, "_mujoco_warp_step", "mjw_step")
    T.wrap(sim.solver, "_update_newton_state", "state_out")
    T.wrap(mjf, "forward", "forward")
    T.wrap(mjf, "fwd_position", "position")
    T.wrap(mjf, "fwd_kinematics", "kinematics")
    T.wrap(mjs, "crb", "crb")
    T.wrap(mjc, "make_constraint", "make_constraint")
    T.wrap(mjs, "transmission", "transmission")
    T.wrap(mjf, "fwd_velocity", "velocity")
    T.wrap(mjf, "fwd_actuation", "actuation")
    T.wrap(mjf, "fwd_acceleration", "acceleration")
    T.wrap(mjsol, "solve", "solve")
    T.wrap(mjf, "implicit", "integrate")
    T.wrap(sim.state_0, "assign", "assign")


def counters(sim, B):
    d = sim.solver.mjw_data
    out = dict(contacts_newton=float(sim.contacts.rigid_contact_count.numpy()[0]) / B,
               contacts=float(d.nacon.numpy()[0]) / B)
    nefc = d.nefc.numpy()
    out["rows_mean"], out["rows_max"] = float(nefc.mean()), float(nefc.max())
    it = d.solver_niter.numpy()
    out["iters_mean"], out["iters_max"] = float(it.mean()), float(it.max())
    out["shape_pairs"] = float(sim.pipeline.broad_phase_pair_count.numpy()[0]) / B
    tri = getattr(sim.pipeline.narrow_phase, "triangle_pairs_count", None)
    if tri is not None:
        out["triangle_pairs"] = float(tri.numpy()[0]) / B
    red = getattr(sim.pipeline.narrow_phase, "global_contact_reducer", None)
    if red is not None:
        out["reducer_input"] = float(red.contact_count.numpy()[0]) / B
    return out


def run_once(pkg, opts, sizes, B, q, v, rows, period, every):
    sim = NS.Sim(pkg, "mjwarp", B=B, caps=NS.Caps(**sizes), options=opts)
    sim.track()
    ticks = NP.Ticks(sim, rows, period)
    NP.start(sim, q, v)
    sim.step(3)  # loading and capturing everything
    wp.synchronize()
    dt, n = sim.dt, C.steps(pkg["name"], sim.dt)

    # 1. plain, as the benchmark times it
    NP.start(sim, q, v)
    sim.reset_peaks()
    ticks.k = None
    ticks.at(0)
    wp.synchronize()
    t0 = time.perf_counter()
    for i in range(n):
        ticks.step(i, dt)
        sim.step(1)
    wp.synchronize()
    wall = time.perf_counter() - t0
    peaks = sim.peaks()
    over = NP.overflows(sim, peaks)
    plain = dict(seconds=wall, ms_per_step=1e3 * wall / n, world_steps_per_s=B * n / wall, overflowed=bool(over),
                 overflow=over, peaks=peaks)

    # 2. traced
    T = Tracer()
    install(T, sim)
    try:
        with wp.ScopedCapture() as cap:
            with T.scope("step"):
                sim._step()
    finally:
        T.unwrap()
    graph = cap.graph
    NP.start(sim, q, v)
    sim.reset_peaks()
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
    m = sim.solver.mjw_model
    info = dict(graph_conditional=bool(getattr(m.opt, "graph_conditional", False)),
                broad_phase=type(sim.pipeline.broad_phase).__name__,
                narrow_phase=type(sim.pipeline.narrow_phase).__name__)
    out = dict(plain=plain, traced=acc.result(), traced_seconds=traced_wall, steps=n, dt=dt, B=B, info=info)
    del graph, sim
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("out")
    ap.add_argument("--sizes", required=True)
    ap.add_argument("--B", type=int, required=True)
    ap.add_argument("--every", type=int, default=20)
    ap.add_argument("--weld-filter", action="store_true", help="leave out the pairs of bodies fixed to the world")
    a = ap.parse_args()
    if a.weld_filter:
        weld_filter()
    s = C.OUT / "scenes"
    ctl = s / f"{a.scene}.controls.npz"
    args = argparse.Namespace(pkg=s / f"{a.scene}.pkg", options=s / f"{a.scene}.newton.json",
                              starts=s / f"{a.scene}.starts.npz", controls=str(ctl) if ctl.exists() else "-")
    pkg, opts, q, v, (rows, period) = NP.load(args)
    q, v = q[:a.B], v[:a.B]
    sizes = C.read_json(a.sizes)["sizes"]
    for attempt in range(4):
        out = run_once(pkg, opts, sizes, a.B, q, v, rows, period, a.every)
        if not out["plain"]["overflowed"]:
            break
        print(f"overflowed {out['plain']['overflow']}: capacities grown, redone", flush=True)
        sizes = R.grow("mjwarp", sizes, out["plain"]["overflow"], a.B)
    out.update(scene=a.scene, sim="mjwarp", sizes=sizes, attempts=attempt + 1, weld_filter=bool(a.weld_filter))
    Path(a.out).write_text(json.dumps(out, indent=1))
    p = out["plain"]
    print(f"{a.scene} mjwarp B={a.B}: {p['world_steps_per_s']:,.0f} world-steps/s, {p['ms_per_step']:.3f} ms a step;"
          f" traced step {out['traced']['ms']['step']:.3f} ms", flush=True)


if __name__ == "__main__":
    main()
