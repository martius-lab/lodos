"""
The Lodos-exact against Kamino throughput comparison: its scenes, settings, starts and files (README.md).

Lodos runs its exact mode at the viewer's defaults for each scene but for the proximal loop's weight and rounds
(`EXACT`) and the Baumgarte gain (`BAUMGARTE`), with the implicitfast integrator (no implicit constraints) and its
contacts reduced where it has surfaces, under each kind of contact group: per contact, cone and ray. The method --
capacities, starts, execution options, ladders -- is the MuJoCo Warp comparison's (`bench.lodos_mjwarp`), and so
are the constants imported from it.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from bench.lodos_mjwarp.common import (  # noqa: F401 -- the MuJoCo Warp comparison's, unchanged
    B0, FIRST, GROUPS, JITTER, MIN_GAIN, N_TUNE, NEWTON_PY, OVERHEAD, VARIANTS, ab_batch, controls, read_json, starts,
    tick, write_json)

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("LODOS_KAMINO_OUT") or ROOT / "results" / "lodos_kamino")  # the env var: a smoke test's

# Each scene and the simulated seconds a run times: the piles 3 s, the grasp and the threading 5 s, the arch and
# the house of cards 2 s.
SCENES = {
    "cube_walled_5_polytope": 3.0, "cube_walled_5_sdf": 3.0, "arm_walled_5_polytope": 3.0, "arm_walled_5_sdf": 3.0,
    "shadow_grasp_polytope": 5.0, "shadow_grasp_sdf": 5.0, "shadow_walled_5_polytope": 3.0,
    "shadow_walled_5_sdf": 3.0, "nist_threading_m8": 5.0, "masonry_arch_polytope": 2.0, "masonry_arch_sdf": 2.0,
    "house_of_cards_polytope": 2.0, "house_of_cards_sdf": 2.0, "cube_walled_20_polytope": 3.0,
    "cube_walled_20_sdf": 3.0, "arm_walled_20_polytope": 3.0, "arm_walled_20_sdf": 3.0,
    "shadow_walled_20_polytope": 3.0, "shadow_walled_20_sdf": 3.0,
}
# Lodos's exact mode over the viewer's defaults: the physics options (`bench.lodos_setting`) and the Baumgarte
# bias, which the solver takes (`lodos.cd.driver.resolve_options`); the slop, the de Saxce term and the rest stay
# the viewer's
EXACT = dict(prox_rho=0.05, prox_iters_per_qp=2)
BAUMGARTE = dict(baumgarte=True, baumgarte_alpha=0.05)
# Kamino at the viewer's settings (`viewer.newton_link.default_options`: Newton 1.7's PADMM defaults, the scene's
# step, the viewer's contact gap), on the polytope scenes alone (it does not run Lodos's surfaces as Newton's mesh
# SDFs), its PADMM iterations capped at 200 where the scene has hands -- fewer leave artifacts -- and 20 elsewhere
KAMINO_ITERATIONS = (200, 20)
# Kamino's execution options, each solving the same problem to the same iterates up to rounding: its Jacobians
# stored sparse, and its dense Delassus matrix factored after a reverse Cuthill-McKee reordering (`LLTBRCM`, not
# its default `LLTB`). Its sparse dynamics are not among them: they replace the direct solve by an iterative one.
KAMINO_VARIANTS = {"default": {}, "sparse_jacobian": {"sparse_jacobian": 1}, "rcm": {"linear_solver": "LLTBRCM"},
                   "sparse_jacobian_rcm": {"sparse_jacobian": 1, "linear_solver": "LLTBRCM"}}
KAMINO_CONTACTS0 = 1024  # contacts a world Kamino's measurement starts at, doubled while a world overflows them
# The polytope scenes Kamino is not run on, and why: the twenty hands' joints and limits
# alone are 3,520 rows of its dense Delassus matrix a world, which it factors every step -- 16 worlds took 7.2 s a
# step at 200 iterations, so the capacity run (16 starts, 1,500 steps) would take 3 h and a ladder point many hours,
# and at a world's contacts (634 by step 450 of the run) the 2^31 entries of all worlds' matrices cap a batch near 50
KAMINO_EXCLUDED = {"shadow_walled_20_polytope": "a step of 16 worlds takes 7.2 s (dense Delassus matrices of 3,520 "
                                                "joint and limit rows a world before its contacts, factored every "
                                                "step); a batch is capped near 50 worlds by 2^31 entries"}


def configs(source):
    """
    The configurations a scene is run under: Lodos per kind of group (per contact alone without surfaces); Kamino
    on the scenes without surfaces, but those of `KAMINO_EXCLUDED`.
    """
    groups = GROUPS if source.surfaces is not None else ("contact",)
    kamino = source.surfaces is None and source.name not in KAMINO_EXCLUDED
    return [f"lodos_{g}" for g in groups] + (["kamino"] if kamino else [])


def kamino_iterations(name):
    return KAMINO_ITERATIONS[0] if "shadow" in name else KAMINO_ITERATIONS[1]


def steps(name, dt):
    return int(round(SCENES[name] / dt))


def escaped(qpos, qpos0, free):
    """
    `(B,)`: the worlds a free body of which is 100 m farther out than the farthest started, or 1 m below the floor
    (z = 0) and below the lowest it started at: a body that left the scene, not one that fell onto the floor or
    was thrown off it (the arch starts 9.5 to 62 m up, the house of cards 1.2 to 11 m, and a collapse is a result,
    not a lost world).
    """
    if not free:
        return np.zeros(len(qpos), bool)
    p = np.stack([qpos[:, a:a + 3] for a in free], 1)
    p0 = np.stack([qpos0[:, a:a + 3] for a in free], 1)
    with np.errstate(invalid="ignore"):
        out = np.linalg.norm(p, axis=-1).max(-1) > np.linalg.norm(p0, axis=-1).max() + 100.0
        return out | (p[..., 2].min(-1) < min(p0[..., 2].min(), 0.0) - 1.0)
