"""
Where a step's time goes, stage by stage, for a scene at a batch size.

    python tools/performance_tuning/profile_scene.py arm_5_sdf --B 4096 \\
        [--tuning tools/performance_tuning/tuned/arm_5_sdf.tuning.json] [--states S.npz] [--pre 50] [--json out.json]

The batch starts from the state set (a weighted draw; the scene's own start in every world
without one) and runs `--pre` steps; each stage is then captured on its own and replayed on
that frozen state, so a stage that iterates sees the same problem every replay:

    rbd                 lodos.rbd: kinematics .. M and its factor
    prim_collision      the prim pipeline: geom poses, broadphase, narrowphases and the hybrid
    lodos_bbox          Lodos's box filter, tile filter and stage-1 compaction
    lodos_sdf           the voxel-SDF prefilter and stage-2 compaction
    lodos_narrow        the spline SDF, stage-3 compaction and the contacts
    contact_dynamics    Lodos's contacts exported to MJWarp, M, the rows and the solve
    integrate           semi-implicit Euler
    whole_step          the step as one graph -- the stages' sum minus launch gaps

The funnel is the Lodos pipeline's survivors at each stage over the batch, beside what
each stage is allocated: how much of each arena the frozen state uses.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

from lodos.utils import states as lstates  # noqa: E402
from lodos import step as lstep  # noqa: E402
from lodos.col.sdf import collide as sdf_collide  # noqa: E402
from lodos.rbd import forward as rbd_forward  # noqa: E402
from lodos.rbd import integrate as rbd_integrate  # noqa: E402


def timed(fn, nrep=30, reps=3):
    """Milliseconds per call of `fn`, captured as a graph and replayed: the best of `reps` runs of `nrep`."""
    fn()
    wp.synchronize()
    with wp.ScopedCapture() as cap:
        fn()
    g = cap.graph
    for _ in range(3):
        wp.capture_launch(g)
    wp.synchronize()
    best = 1e9
    for _ in range(reps):
        t = time.perf_counter()
        for _ in range(nrep):
            wp.capture_launch(g)
        wp.synchronize()
        best = min(best, (time.perf_counter() - t) / nrep)
    return best * 1e3


def stages_of(sim):
    """`{name: callable}`: the stages of one step of `sim`, each capturable on its own."""
    s = sim
    st = {k: getattr(s.drbd, k).numpy().copy() for k in ("qpos", "qvel")}
    wst = s.d.qacc_warmstart.numpy().copy()

    def restore():
        for k, v in st.items():
            getattr(s.drbd, k).assign(v)
        s.d.qacc_warmstart.assign(wst)

    out = dict(rbd=lambda: rbd_forward.fwd_rbd(s.mrbd, s.drbd))
    if s.prim.active:
        out["prim_collision"] = lambda: (s.prim.poses(s.drbd.xpos, s.drbd.xmat), s.prim.collide())
    if s.mcol is not None:
        out["lodos_bbox"] = lambda: sdf_collide.broadphase_bbox(s.mcol, s.dcol)
        out["lodos_sdf"] = lambda: sdf_collide.broadphase_sdf(s.mcol, s.dcol)
        out["lodos_narrow"] = lambda: sdf_collide.narrowphase_sdf(s.mcol, s.dcol)
    # The solve appends Lodos's contacts to MJWarp's (`export_lodos`): each replay starts from
    # MJWarp's own count at this state, or every replay would append them again.
    rbd_forward.fwd_rbd(s.mrbd, s.drbd)
    if s.prim.active:
        s.prim.poses(s.drbd.xpos, s.drbd.xmat)
    s.prim.collide()
    nacon = wp.clone(s.d.nacon)
    out["contact_dynamics"] = lambda: (restore(), wp.copy(s.d.nacon, nacon), s.solver.contact_dynamics())
    out["integrate"] = lambda: (restore(), rbd_integrate.integrate(s.mrbd, s.drbd))
    out["whole_step"] = lambda: (restore(), lstep.fwd_step(s))
    return out


def funnel(sim):
    """The Lodos stages' survivors now and their allocations; MJWarp's contacts and rows."""
    out = {}
    if sim.dcol is not None:
        ct = sim.dcol.ct.numpy().tolist()
        d = sim.dcol
        out["lodos"] = dict(
            candidates=dict(n=int(ct[0]), allocated=int(d.nCa1_max)),
            tiles=dict(n=int(ct[3]), allocated=int(d.nTa_max), of=int(sim.mcol.n_tile_slots)),
            voxel=dict(n=int(ct[1]), allocated=int(d.nCa2_max)),
            contacts=dict(n=int(ct[2]), allocated=int(d.nCa3_max)),
        )
    d = sim.d
    out["runtime"] = dict(contacts=dict(n=int(d.nacon.numpy()[0]), allocated=int(d.naconmax)))
    out["runtime"]["rows_per_world"] = dict(n=int(d.nefc.numpy().max()), allocated=int(d.njmax))
    return out


def profile(src, B, tuning=None, sset=None, pre=50, seed=0):
    sim = src.make_sim(B=B, tuning=tuning)
    if sset is not None:
        idx = lstates.batches(len(sset), B, "independent", weights=sset.weights, rng=np.random.default_rng(seed),
                              cover=False)[0]
        lstates.install(sim, sset, idx)
    replay = lstep.make_step_graph(sim)
    for _ in range(pre):
        replay()
    wp.synchronize()
    times = {k: timed(f) for k, f in stages_of(sim).items()}
    lstep.forward(sim)
    wp.synchronize()
    fun = funnel(sim)
    parts = sum(v for k, v in times.items() if k != "whole_step")
    out = dict(scene=C.scene_name(src.name), B=B, pre=pre, ms=times, stages_sum_ms=parts,
               world_steps_per_s=B / (times["whole_step"] * 1e-3), funnel=fun, mempool_mb=C.mempool_mb())
    C.free(sim)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene")
    ap.add_argument("--B", type=int, default=1024)
    ap.add_argument("--tuning")
    ap.add_argument("--states")
    ap.add_argument("--pre", type=int, default=50)
    ap.add_argument("--json")
    a = ap.parse_args()
    src = C.scene(a.scene)
    sset = lstates.load(a.states, src.mjm) if a.states else None
    out = profile(src, a.B, a.tuning, sset, a.pre)
    w = max(len(k) for k in out["ms"])
    total = out["ms"]["whole_step"]
    for k, v in out["ms"].items():
        share = "" if k == "whole_step" else f"{100 * v / out['stages_sum_ms']:5.1f}%"
        print(f"  {k:<{w}}  {v:8.3f} ms  {share}")
    print(f"  {out['world_steps_per_s']:.4g} world-steps/s at B={a.B} ({total:.3f} ms a step)")
    for group, d in out["funnel"].items():
        for k, v in (d.items() if isinstance(d, dict) and "n" not in d else [(group, d)]):
            print(f"  {group}.{k}: {v['n']} of {v['allocated']} allocated")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
