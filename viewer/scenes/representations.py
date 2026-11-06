"""
One drop -- of armadillos or of cubes -- in either collision representation, with or
without walls.

The same bodies, the same inertia and the same randomized stacks, colliding either

- **as Lodos surfaces** (`rep="surfaces"`, the viewer's "sdf"): the armadillo's surface,
  or the cube's (`CUBE_SURFACE`), every ordered body pair declared plus each body
  against the floor and the walls, as in `viewer.scenes.armadillo`; or
- **as polytopes** (`rep="polytopes"`, the viewer's "polytope"): the armadillo's 34-piece
  convex decomposition as MJCF meshes, or the cube as an MJCF box, through MJWarp's
  broadphase and the exact hull narrowphase (GJK/EPA, the quadrangle manifold, the
  floor kernel).

Either way each body's mass and inertia come from one geom -- the armadillo surface's
own mesh, or the box, density 100 -- which as a surface never collides through MJWarp.
An armadillo's pieces are massless, so the two scenes have identical rigid-body dynamics
and differ only in how they collide.

**Walls** (`walls=True`) are `drop.wall_xml`'s open-top box. As polytopes they are its
MJCF boxes. A Lodos surface does not collide with MJCF geoms, so as surfaces each wall's
inner face is a half-space instead, on a static body of its own (`wall_planes_xml`).

Layout: `n_bodies` dealt round-robin into `columns` stacks on a square grid
`drop.SPACING` apart, each stack provably intersection-free for both
representations (an armadillo is stacked by the union of the surface's and the pieces'
vertices). Options are the drop benchmark's: elliptic cone, Newton, 100 iterations, 50
line-search iterations; dt 0.002, Euler.
"""

from __future__ import annotations

import mujoco
import numpy as np

from lodos import sim as lsim
from viewer.scenes.assets import load_surface, surface_obj
from lodos.col.sdf import types as sdf_types
from viewer.scenes import drop

REPS = ("surfaces", "polytopes")
KINDS = ("armadillo", "cube")
# The armadillo drop's spring-damper row (`viewer.scenes.armadillo`), for both kinds of contact.
MU, STIFFNESS, DISSIPATION = 0.8, 5.0e3, 0.15
# The cube's surface: side `2 drop.CUBE_HALF`, vertices spread over its faces.
CUBE_SURFACE = "cube_0.8"
N_WALLS = 4


def wall_planes_xml(columns):
    """`drop.wall_xml`'s walls for Lodos surfaces: `plane_walls_xml` at its size."""
    return plane_walls_xml(drop.wall_xml(columns)[1])


def plane_walls_xml(half, h=drop.WALL_H, mu=MU):
    """
    `drop.box_walls_xml(half, h)`'s walls for Lodos surfaces. Each inner face is a
    half-space on a static body of its own, turned so that its `+z` points into the box: a
    `PlaneSurface` is the half-space below its body frame's xy-plane. The plane geoms,
    `wall0`..`wall3`, draw the faces and are what the surfaces are keyed to.
    """
    s = np.sqrt(0.5)
    out = []
    # (position, wxyz quaternion turning +z onto the inward normal)
    for i, (pos, q) in enumerate(((( half, 0.0, h), (s, 0.0, -s, 0.0)),    # +x wall, facing -x
                                  ((-half, 0.0, h), (s, 0.0, s, 0.0)),     # -x wall, facing +x
                                  ((0.0,  half, h), (s, s, 0.0, 0.0)),     # +y wall, facing -y
                                  ((0.0, -half, h), (s, -s, 0.0, 0.0)))):  # -y wall, facing +y
        # The plane's local x is vertical on the x walls and along x on the y walls.
        hx, hy = (h, half) if i < 2 else (half, h)
        out.append(f'<body name="wall{i}" pos="{pos[0]} {pos[1]} {pos[2]}" quat="{q[0]} {q[1]} {q[2]} {q[3]}">'
                   f'<geom name="wall{i}" type="plane" size="{hx} {hy} 0.1" friction="{mu}" '
                   f'rgba="{drop.WALL_RGBA}"/></body>')
    return "\n    ".join(out)


def scene_xml(n_bodies, rep, kind="armadillo", walls=False, columns=1):
    """`(xml, verts)`: the MJCF, and the `(nv, 3)` body vertices the stacks are built from."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    assets, extra = [], ""
    if kind == "armadillo":
        pieces, union = drop.load_armadillo_pieces()
        verts = np.vstack([np.asarray(load_surface("armadillo").v), union])
        body_geom = f'<geom name="surf{{i}}" type="mesh" mesh="arm" density="{drop.DENSITY}" contype="0" conaffinity="0"/>'
        assets.append(f'<mesh name="arm" file="{surface_obj("armadillo")}"/>')
        if rep == "polytopes":
            extra = "".join(f'<geom type="mesh" mesh="cp{k}" friction="{MU}" density="0"/>' for k in range(len(pieces)))
            assets += drop.piece_assets(pieces)
    else:
        h = drop.CUBE_HALF
        verts = h * np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        collide = f'friction="{MU}"' if rep == "polytopes" else 'contype="0" conaffinity="0"'
        body_geom = f'<geom name="surf{{i}}" type="box" size="{h} {h} {h}" density="{drop.DENSITY}" {collide}/>'
    bodies = "\n".join(
        f'    <body name="b{i}" pos="0 0 {2.0 + i}"><freejoint/>{body_geom.format(i=i)}{extra}</body>'
        for i in range(n_bodies)
    )
    wall = ""
    if walls:
        wall = drop.wall_xml(columns)[0] if rep == "polytopes" else wall_planes_xml(columns)
    nl = "\n    "
    xml = f"""
<mujoco>
  {drop.OPTION}
  <asset>
    {nl.join(assets) if assets else "<!-- no assets -->"}
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="20 20 0.1" friction="{MU}"/>
    {wall}
{bodies}
  </worldbody>
</mujoco>
"""
    return xml, verts


def scene_and_q0(n_bodies, columns, rep, nworld, seed=0, kind="armadillo", walls=False):
    xml, verts = scene_xml(n_bodies, rep, kind, walls, columns)
    return xml, drop.initial_qpos_batch(n_bodies, verts, seed, nworld, columns)


def pair_row():
    row = lsim.default_pair_params(MU, 1)[0]
    row[1] = STIFFNESS
    row[3] = DISSIPATION
    return row


def surfaces(mjm, n_bodies, col_kwargs=None, kind="armadillo", walls=False):
    """
    The Lodos half of the `"surfaces"` representation: every ordered body pair, and each
    body against the floor and, with `walls`, against each wall.
    """
    geom = lambda name: mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, name)  # noqa: E731
    planes = ["floor"] + ([f"wall{k}" for k in range(N_WALLS)] if walls else [])
    cprs = [[a, b] for a in range(n_bodies) for b in range(n_bodies) if a != b]
    cprs += [[a, n_bodies + p] for a in range(n_bodies) for p in range(len(planes))]
    params = np.tile(pair_row(), (len(cprs), 1))
    surf = load_surface("armadillo" if kind == "armadillo" else CUBE_SURFACE)
    return lsim.Surfaces(
        surfaces=[surf] * n_bodies + [sdf_types.PlaneSurface() for _ in planes],
        geoms=[geom(f"surf{i}") for i in range(n_bodies)] + [geom(p) for p in planes],
        pairs=cprs, params=params, mu=MU, col_kwargs=dict(col_kwargs or {}),
    )


def default_columns(n_bodies):
    return 1 if n_bodies <= 5 else 4


def build(rep, n_bodies, B, columns=None, seed=0, solver_kwargs=None, col_kwargs=None,
          kind="armadillo", walls=False, **sim_kwargs):
    """The scene as a `lodos.sim.Sim`. `columns` defaults to 1 for five bodies, else 4."""
    if rep not in REPS:
        raise ValueError(f"rep must be one of {REPS}")
    columns = columns if columns is not None else default_columns(n_bodies)
    xml, q0 = scene_and_q0(n_bodies, columns, rep, B, seed, kind, walls)
    mjm = mujoco.MjModel.from_xml_string(xml)
    surf = surfaces(mjm, n_bodies, col_kwargs, kind, walls) if rep == "surfaces" else None
    return lsim.make_sim(mjm, B, surfaces=surf, qpos=q0, solver_kwargs=solver_kwargs, **sim_kwargs)
