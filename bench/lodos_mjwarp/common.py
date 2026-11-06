"""
The Lodos-mj against MuJoCo Warp throughput comparison: its scenes, settings, starts and files (README.md).

Lodos runs its mj mode at the viewer's defaults for each scene, with the implicitfast integrator (no implicit
constraints) and its contacts reduced where it has surfaces, under each kind of contact group: per contact, cone and
ray. MuJoCo Warp is Newton's SolverMuJoCo on Newton's collision pipeline (its mesh contacts reduced, its default),
at Lodos's solver settings. Every contact of both takes `SOLREF`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("LODOS_MJWARP_OUT") or ROOT / "results" / "lodos_mjwarp")  # the env var: a smoke test's
NEWTON_PY = ROOT / "viewer" / "newton" / ".venv" / "bin" / "python"

# Each scene and the simulated seconds a run times: the piles 3 s, the grasp and the threading 5 s.
SCENES = {
    "cube_walled_5_polytope": 3.0, "cube_walled_5_sdf": 3.0, "arm_walled_5_polytope": 3.0, "arm_walled_5_sdf": 3.0,
    "shadow_grasp_polytope": 5.0, "shadow_grasp_sdf": 5.0, "shadow_walled_5_polytope": 3.0,
    "shadow_walled_5_sdf": 3.0, "nist_threading_m8": 5.0, "cube_walled_20_polytope": 3.0, "cube_walled_20_sdf": 3.0,
    "arm_walled_20_polytope": 3.0, "arm_walled_20_sdf": 3.0, "shadow_walled_20_polytope": 3.0,
    "shadow_walled_20_sdf": 3.0,
}
# Every contact's solref in both sims: 5 ms, critically damped; the threading's 2.5 ms, its own (its pad pairs'
# 1.75 ms and its pads' 1 ms are raised to 2.5 ms by MuJoCo's refsafe at its 1.25 ms step)
SOLREF = {name: (0.0025, 1.0) if name == "nist_threading_m8" else (0.005, 1.0) for name in SCENES}
GROUPS = ("contact", "cone", "ray")
N_TUNE = 16  # the starts a configuration's capacities are measured over, each world alone
# MuJoCo Warp's capacities on these scenes measured with the starts as the worlds of one batch, each world's own
# peaks (`newton_point.measure_batched`): the starts one by one stall on the twenty hands' SDF pile
BATCHED_TUNE = ("shadow_walled_20_sdf",)
# MuJoCo Warp also under Newton's sweep-and-prune broad phase on these (`mjwarp_sap`, run 2026-11-03): under its
# explicit pairs, whose work buffers Newton sizes for every pair of shapes in every world, MuJoCo Warp ran out of
# memory there still gaining
SAP = ("arm_walled_20_polytope", "shadow_walled_20_polytope")
OVERHEAD = 0.5  # each capacity its largest demand per world times 1 + OVERHEAD, times the worlds
B0 = 64  # the ladder's first batch size, doubled to the first drop in throughput or out of memory
FIRST = 1000  # the ladder's worlds are starts FIRST, FIRST + 1, ...; the capacities' 0 .. N_TUNE - 1
JITTER = 5e-5  # m: a world's start of the grasp or the threading, its free bodies moved in x and y by up to this
AB_B = {"_20_": 256}  # the batch size the execution options are compared at: 256 for the twenty-body piles,
AB_B_DEFAULT = 1024  # else 1024
VARIANTS = {"default": {}, "no_shared_hessian": {"shared_hessian": 0}, "fused_newton": {"fused_newton": 1}}
MIN_GAIN = 0.03  # an execution option is kept only where it is 3 % faster


def doubled_triangle_pairs(per_world, B):
    """
    Newton's mesh triangle pairs a world, its batch's doubled: the batch's are at least a million
    (`newton_scene.Caps.triangle_pairs`), and they size the table that reduces the mesh contacts too.
    """
    return 2 * max(int(per_world), -(-1_000_000 // int(B)))


def configs(source):
    """
    The configurations a scene is run under: Lodos per kind of group (per contact alone without surfaces), MuJoCo
    Warp, and on the scenes in `SAP` MuJoCo Warp under Newton's sweep-and-prune broad phase.
    """
    groups = GROUPS if source.surfaces is not None else ("contact",)
    return [f"lodos_{g}" for g in groups] + ["mjwarp"] + (["mjwarp_sap"] if source.name in SAP else [])


def newton_args(cfg):
    """`newton_point.py`'s arguments for a configuration of MuJoCo Warp's."""
    return ["--broad-phase", "sap"] if cfg == "mjwarp_sap" else []


def steps(name, dt):
    return int(round(SCENES[name] / dt))


def ab_batch(name):
    return next((b for k, b in AB_B.items() if k in name), AB_B_DEFAULT)


def starts(source, n, first):
    """
    `(qpos, qvel)`, `(n, nq)` and `(n, nv)`: worlds `first` .. `first + n - 1`, each its own start. A pile's from its
    sampler (a stack of its own, by seed); the grasp's and the threading's their recorded start, each free body moved
    in x and y by up to `JITTER`, by seed.
    """
    v = np.tile(np.asarray(source.qvel0 if source.qvel0 is not None else np.zeros(source.mjm.nv), np.float64), (n, 1))
    if source.sampler is not None:
        return np.asarray(source.sampler(n, first), np.float64), v
    import mujoco

    m = source.mjm
    free = [int(m.jnt_qposadr[j]) for j in range(m.njnt) if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)]
    q = np.tile(np.asarray(source.qpos0, np.float64), (n, 1))
    for i in range(n):
        rng = np.random.default_rng(first + i)
        for a in free:
            q[i, a:a + 2] += rng.uniform(-JITTER, JITTER, 2)
    return q, v


def controls(c, n):
    """A scene's recorded controls as `(rows (T, nu), period)`, its first `n` ticks, or None for a scene without."""
    if c is None:
        return None
    rows = np.asarray(c.ctrl, np.float64)
    k = np.minimum(np.arange(n), len(rows) - 1)  # the last row held past the recording
    return rows[k], float(c.period)


def tick(i, dt, period):
    """The tick of recorded controls step `i` (from time 0) takes, as `RecordedControls.tick` finds it."""
    return int(np.floor(i * dt / period + 0.01))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)))
    return path


def read_json(path):
    return json.loads(Path(path).read_text())


def escaped(qpos, qpos0, free):
    """`(B,)`: the worlds a free body of which is 100 m out, or 1 m below the lowest it started at."""
    if not free:
        return np.zeros(len(qpos), bool)
    p = np.stack([qpos[:, a:a + 3] for a in free], 1)
    p0 = np.stack([qpos0[:, a:a + 3] for a in free], 1)
    with np.errstate(invalid="ignore"):
        return (np.linalg.norm(p, axis=-1).max(-1) > 100.0) | (p[..., 2].min(-1) < p0[..., 2].min() - 1.0)
