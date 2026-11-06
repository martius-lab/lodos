"""
Both collision pipelines in one scene: armadillos on Lodos surfaces, supplemented by
their convex pieces, stacked with cubes inside the drop benchmark's walled box.

Each armadillo body carries

- the baked surface, keyed to a mesh of the same surface that MuJoCo takes the
  body's inertia from (density 100, as in `viewer.scenes.armadillo`). Surfaces collide
  with each other and with the floor through Lodos's pipeline, every ordered pair
  declared;
- the 34-piece convex decomposition as massless MJCF meshes (density 0). The
  pieces collide through MJWarp's pipeline with the other bodies' pieces, the cubes,
  the walls and the floor -- the edge-edge contacts a vertex-SDF surface cannot see.

The cubes (MJCF boxes, density 100) and the walls collide through MJWarp's pipeline
only, so a cube meets an armadillo through its pieces. The floor is one MJCF plane,
keyed to a `PlaneSurface` for the surfaces and colliding with the pieces and the
cubes as itself.

Bodies alternate armadillo, cube, armadillo, ... round-robin over the columns, each
stack provably intersection-free by the drop benchmark's construction (per body,
its own vertex set).
"""

from __future__ import annotations

import mujoco
import numpy as np

from lodos import sim as lsim
from viewer.scenes.assets import load_surface, surface_obj
from lodos.col.sdf import types as sdf_types
from viewer.scenes import drop

SCENES = {
    "mix6": dict(n_bodies=6, columns=1),
    "mix20": dict(n_bodies=20, columns=4),
}

# Per-world MJWarp capacities. They also hold the Lodos contacts.
CAPS = {
    "mix6": dict(njmax=2048, nconmax=2048),
    "mix20": dict(njmax=6144, nconmax=8192),
}


def kinds(n_bodies):
    return ["armadillo" if i % 2 == 0 else "cube" for i in range(n_bodies)]


def scene_xml(n_bodies, columns):
    """`(xml, verts)`: the MJCF, and each body's vertices for stacking."""
    pieces, union = drop.load_armadillo_pieces()
    obj = surface_obj("armadillo")
    cube = drop.CUBE_HALF * np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    mu = drop.MU
    piece_geoms = "".join(
        f'<geom type="mesh" mesh="cp{k}" friction="{mu}" density="0" group="1"/>' for k in range(len(pieces))
    )
    bodies, verts = [], []
    for i, kind in enumerate(kinds(n_bodies)):
        if kind == "armadillo":
            geoms = (f'<geom name="surf{i}" type="mesh" mesh="arm" density="{drop.DENSITY}" '
                     f'contype="0" conaffinity="0" group="2"/>' + piece_geoms)
            verts.append(union)
        else:
            geoms = (f'<geom type="box" size="{drop.CUBE_HALF} {drop.CUBE_HALF} {drop.CUBE_HALF}" '
                     f'friction="{mu}" density="{drop.DENSITY}"/>')
            verts.append(cube)
        bodies.append(f'    <body name="b{i}" pos="0 0 {2.0 + i}"><freejoint/>{geoms}</body>')
    walls, _ = drop.wall_xml(columns)
    nl = "\n    "
    assets = [f'<mesh name="arm" file="{obj}"/>'] + drop.piece_assets(pieces)
    xml = f"""
<mujoco>
  {drop.OPTION}
  <asset>
    {nl.join(assets)}
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="20 20 0.1" friction="{mu}"/>
    {walls}
{chr(10).join(bodies)}
  </worldbody>
</mujoco>
"""
    return xml, verts


def initial_qpos(verts, rng, columns=1):
    """`drop.initial_qpos` with each body's own vertex set (same rng consumption order)."""
    n_bodies = len(verts)
    q = np.zeros(7 * n_bodies)
    side = int(np.ceil(np.sqrt(columns)))
    sp = drop.SPACING
    centres = [((c % side - (side - 1) / 2) * sp, (c // side - (side - 1) / 2) * sp) for c in range(columns)]
    top = [0.0] * columns
    for i in range(n_bodies):
        c = i % columns
        quat = rng.normal(size=4)
        quat /= np.linalg.norm(quat)
        vz = (verts[i] @ drop.quat_to_mat(quat).T)[:, 2]
        z = top[c] + drop.GAP - vz.min()
        q[7 * i : 7 * i + 2] = rng.uniform(-drop.JITTER, drop.JITTER, 2) + np.asarray(centres[c])
        q[7 * i + 2] = z
        q[7 * i + 3 : 7 * i + 7] = quat
        top[c] = z + vz.max()
    return q


def scene_and_q0(scene, nworld, seed=0):
    """`(xml, q0)`: the MJCF and `(nworld, nq)` initial positions."""
    cfg = SCENES[scene]
    xml, verts = scene_xml(cfg["n_bodies"], cfg["columns"])
    rng = np.random.default_rng(seed)
    return xml, np.stack([initial_qpos(verts, rng, cfg["columns"]) for _ in range(nworld)])


def surfaces(mjm, n_bodies, mu=drop.MU, stiffness=5.0e3, dissipation=0.15, spline="cubic", col_kwargs=None):
    """The Lodos half: one surface per armadillo, every ordered armadillo pair, each against the floor."""
    surf = load_surface("armadillo", spline_type=spline)
    arms = [i for i, k in enumerate(kinds(n_bodies)) if k == "armadillo"]
    geom = lambda name: mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, name)
    n = len(arms)
    cprs = [[a, b] for a in range(n) for b in range(n) if a != b] + [[a, n] for a in range(n)]
    params = lsim.default_pair_params(mu, len(cprs))
    params[:, 1] = stiffness
    params[:, 3] = dissipation
    return lsim.Surfaces(
        surfaces=[surf] * n + [sdf_types.PlaneSurface()],
        geoms=[geom(f"surf{i}") for i in arms] + [geom("floor")],
        pairs=cprs,
        params=params,
        mu=mu,
        col_kwargs=dict(col_kwargs or {}),
    )


def build(scene="mix6", batch=64, seed=0, solver_kwargs=None, col_kwargs=None, **sim_kwargs):
    """The batched scene as a `lodos.sim.Sim`."""
    xml, q0 = scene_and_q0(scene, batch, seed)
    mjm = mujoco.MjModel.from_xml_string(xml)
    surf = surfaces(mjm, SCENES[scene]["n_bodies"], col_kwargs=col_kwargs)
    kw = dict(CAPS[scene])
    kw.update(sim_kwargs)
    return lsim.make_sim(mjm, batch, surfaces=surf, qpos=q0, solver_kwargs=solver_kwargs, **kw)
