"""
Makes a state set (`lodos.utils.states`) for a scene: the starting configurations every other
tool calibrates and times from.

    python tools/performance_tuning/explore.py arm_5_sdf --out results/tuning/arm_5_sdf.states.npz \\
        [--strategy scatter actuation perturb initial] [--starts 64] [--horizon 2000] [--stride 50] \\
        [--seed 0] [--max-depth 0.01]

Strategies, each a set of starts rolled out `--horizon` steps and sampled
every `--stride` steps -- the impacts, the motion and the settled state, which a still
frame misses:

    initial     the scene's own start, in every world (deterministic scenes: one rollout)
    scatter     `--starts` starts drawn the way the scene draws its own (a drop's stacks, the
                hands' orientations; `SceneSource.sampler`), or, for an MJCF file, free bodies
                placed at random without overlap above where they start
    actuation   the scene's start under random control (`lodos.utils.states.Controls.random`)
    perturb     the scene's start and keyframes, jittered: free bodies by `--jitter` m and
                rad, joints by `--jitter` of their range
    keyframes   the MJCF's keyframes, as they are (no rollout)

Filters drop states that went non-finite and states whose deepest penetration exceeds
`--max-depth`. The report (`<out>.json`) has each strategy's count, the load histogram
(contacts per world) and the running peak of the load against the number of states, in a
random order: if it is still rising at the end, the set has not found the heaviest states
yet -- explore more.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

from lodos.utils import states as lstates  # noqa: E402
from lodos import step as lstep  # noqa: E402


def _rollout_samples(src, q0, qv0, horizon, stride, controls, label, ctrl=None):
    """States every `stride` steps of a rollout of `len(q0)` worlds from `q0`."""
    B = len(q0)
    sim = src.make_sim(B=B, qpos=q0)
    start = lstates.from_qpos(src.mjm, q0, qvel=qv0,
                              ctrl=ctrl if ctrl is not None else (src.ctrl if src.ctrl is not None else None),
                              labels=[f"{label} {i}" for i in range(B)])
    replay = lstep.make_step_graph(sim)
    out = lstates.sample_rollout(sim, start, np.arange(B), horizon, stride, controls, replay, label)
    C.free(sim)
    return out


def scatter_starts(src, n, seed):
    """`(n, nq)` starts: the scene's own sampler, or free bodies placed at random without overlap."""
    if src.sampler is not None:
        return np.asarray(src.sampler(n, seed), dtype=np.float64).reshape(n, -1)
    import mujoco

    m = src.mjm
    rng = np.random.default_rng(seed)
    d = mujoco.MjData(m)
    free = [int(m.jnt_qposadr[j]) for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    if not free:
        raise ValueError("the scene has no free bodies to scatter and no sampler of its own")
    q0 = src.qpos0
    ext = max(0.1, float(np.ptp(np.stack([q0[a:a + 3] for a in free]), axis=0).max()))
    out = []
    tries = 0
    while len(out) < n and tries < 200 * n:
        tries += 1
        q = q0.copy()
        for a in free:
            q[a:a + 2] += rng.uniform(-0.5 * ext, 0.5 * ext, 2)
            q[a + 2] += rng.uniform(0.0, ext)
            quat = rng.normal(size=4)
            q[a + 3:a + 7] = quat / np.linalg.norm(quat)
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        if d.ncon == 0 or min(d.contact[i].dist for i in range(d.ncon)) > 0.0:
            out.append(q)
    if len(out) < n:
        raise RuntimeError(f"placed only {len(out)} of {n} starts without overlap")
    return np.stack(out)


def perturb_starts(src, n, jitter, seed):
    """`n` starts jittered around the scene's start and keyframes."""
    import mujoco

    m = src.mjm
    rng = np.random.default_rng(seed)
    bases = [src.qpos0] + [np.asarray(k[1], dtype=np.float64) for k in src.keyframes]
    out = []
    for i in range(n):
        q = bases[i % len(bases)].copy()
        for j in range(m.njnt):
            a = int(m.jnt_qposadr[j])
            t = m.jnt_type[j]
            if t == mujoco.mjtJoint.mjJNT_FREE:
                q[a:a + 3] += rng.normal(scale=jitter, size=3)
                w = rng.normal(scale=jitter, size=3)
                dq = np.zeros(4)
                mujoco.mju_axisAngle2Quat(dq, w / (np.linalg.norm(w) + 1e-12), np.linalg.norm(w))
                res = np.zeros(4)
                mujoco.mju_mulQuat(res, dq, q[a + 3:a + 7])
                q[a + 3:a + 7] = res / np.linalg.norm(res)
            elif t in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                lo, hi = m.jnt_range[j] if m.jnt_limited[j] else (-np.pi, np.pi)
                q[a] = np.clip(q[a] + rng.normal(scale=jitter * (hi - lo)), lo, hi)
        out.append(q)
    return np.stack(out)


def loads(src, sset, B=256):
    """
    Contacts per world of every state of `sset`, from one forward pass per batch (no step).
    The batch halves until the scene's own capacities fit in memory.
    """
    out = np.zeros(len(sset), dtype=np.int64)
    deep = np.zeros(len(sset))
    while True:
        try:
            sim = src.make_sim(B=B)
            break
        except RuntimeError:
            C.free()
            if B <= 8:
                raise
            B //= 2
    for start in range(0, len(sset), B):
        idx = np.arange(start, min(start + B, len(sset)))
        full = np.concatenate([idx, np.full(B - len(idx), idx[0])])
        lstates.install(sim, sset, full)
        lstep.forward(sim)
        n = np.zeros(B, dtype=np.int64)
        dmin = np.zeros(B)
        if sim.dcol is not None:
            k = int(sim.dcol.ct.numpy()[2])
            b = sim.dcol.b_nCa3.numpy()[:k]
            np.add.at(n, b, 1)
            np.minimum.at(dmin, b, sim.dcol.sdf_nCa3.numpy()[:k])
        k = min(int(sim.d.nacon.numpy()[0]), int(sim.d.naconmax))
        if k:
            w = sim.d.contact.worldid.numpy()[:k]
            np.minimum.at(dmin, w, sim.d.contact.dist.numpy()[:k])
            if sim.dcol is None or sim.solver is None:  # else the solve has Lodos's contacts among MJWarp's
                np.add.at(n, w, 1)
            else:
                n[:] = 0
                np.add.at(n, w, 1)
        out[idx] = n[:len(idx)]
        deep[idx] = -dmin[:len(idx)]
    C.free(sim)
    return out, deep


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene")
    ap.add_argument("--out", required=True)
    ap.add_argument("--strategy", nargs="+", default=["scatter"],
                    choices=["initial", "scatter", "actuation", "perturb", "keyframes"])
    ap.add_argument("--starts", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=2000)
    ap.add_argument("--stride", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jitter", type=float, default=0.01)
    ap.add_argument("--sigma", type=float, default=0.5, help="random control: walk scale, a fraction of ctrlrange")
    ap.add_argument("--max-depth", type=float, default=0.01, help="drop states penetrating deeper (m)")
    a = ap.parse_args()

    src = C.scene(a.scene)
    t0 = time.time()
    parts, counts = [], {}
    for strat in a.strategy:
        if strat == "keyframes":
            if src.mjm.nkey:
                s = lstates.from_keyframes(src.mjm)
            else:
                s = lstates.from_qpos(src.mjm, np.stack([k[1] for k in src.keyframes]) if src.keyframes else src.qpos0[None],
                                      ctrl=src.ctrl, labels=["keyframe"])
        elif strat == "initial":
            q0 = src.qpos0[None]
            s = _rollout_samples(src, q0, src.qvel0[None], a.horizon, a.stride, None, "initial")
        elif strat == "scatter":
            q0 = scatter_starts(src, a.starts, a.seed)
            s = _rollout_samples(src, q0, np.tile(src.qvel0, (len(q0), 1)), a.horizon, a.stride, None,
                                 "scatter")
        elif strat == "actuation":
            if not src.mjm.nu:
                C.log(f"{a.scene}: no actuators, skipping actuation")
                continue
            q0 = np.tile(src.qpos0, (a.starts, 1))
            s = _rollout_samples(src, q0, np.tile(src.qvel0, (a.starts, 1)), a.horizon, a.stride,
                                 lstates.Controls.random(a.sigma, seed=a.seed), "actuation")
        else:  # perturb
            q0 = perturb_starts(src, a.starts, a.jitter, a.seed)
            s = _rollout_samples(src, q0, np.tile(src.qvel0, (len(q0), 1)), a.horizon, a.stride, None,
                                 "perturb")
        counts[strat] = len(s)
        parts.append(s)
        C.log(f"{a.scene}: {strat}: {len(s)} states ({time.time() - t0:.0f} s)")
    sset = lstates.concat(*parts)

    # filters
    finite = np.isfinite(sset.qpos).all(1) & np.isfinite(sset.qvel).all(1)
    sset = sset.subset(np.nonzero(finite)[0])
    load, deep = loads(src, sset)
    keep = deep <= a.max_depth
    sset, load, deep = sset.subset(np.nonzero(keep)[0]), load[keep], deep[keep]
    sset.source = dict(kind="explore", scene=C.scene_name(a.scene), strategies=a.strategy, starts=a.starts,
                       horizon=a.horizon, stride=a.stride, seed=a.seed, jitter=a.jitter,
                       sigma=a.sigma, max_depth=a.max_depth, dropped_nonfinite=int((~finite).sum()),
                       dropped_deep=int((~keep).sum()))
    path = sset.save(a.out)

    rng = np.random.default_rng(a.seed)
    order = rng.permutation(len(load))
    running = np.maximum.accumulate(load[order]) if len(load) else np.zeros(0)
    hist, edges = np.histogram(load, bins=min(20, max(1, len(np.unique(load)))))
    last_rise = int(np.nonzero(np.diff(running))[0].max() + 1) if len(running) > 1 and np.diff(running).any() else 0
    report = dict(scene=C.scene_name(a.scene), states=len(sset), counts=counts, digest=sset.digest(),
                  load=dict(mean=float(load.mean()) if len(load) else 0.0, max=int(load.max()) if len(load) else 0,
                            p99=float(np.percentile(load, 99)) if len(load) else 0.0,
                            hist=hist.tolist(), edges=edges.tolist()),
                  deepest=float(deep.max()) if len(deep) else 0.0,
                  running_peak=running[np.unique(np.geomspace(1, max(1, len(running)), 40).astype(int)) - 1].tolist()
                  if len(running) else [],
                  peak_found_at=last_rise, still_rising=bool(last_rise > 0.8 * len(running)),
                  seconds=time.time() - t0)
    C.write_json(str(path) + ".json", report)
    C.log(f"{a.scene}: {len(sset)} states -> {path}; load mean {report['load']['mean']:.1f} max {report['load']['max']}"
          f" (peak reached at state {last_rise} of {len(running)}{', still rising' if report['still_rising'] else ''})")


if __name__ == "__main__":
    main()
