"""
The drop benchmark: bodies dropped in randomized, provably intersection-free stacks
onto a plane, inside an open-top box of four static walls.

Bodies are cubes (one MJCF box each) or armadillos (the 34-piece convex
decomposition, 34 MJCF meshes each), so the whole scene collides through MJWarp's
pipeline. Scene constants follow Lodos-v1's armadillo drop (`papers/lodos_v1.pdf`;
mu = 0.8, 0.15 m vertical clearance, +-0.12 m horizontal jitter, 4 columns 3 m apart
for the 20-body variants, density 100); the integrator and timestep are MuJoCo's
defaults (Euler, dt = 0.002), with Newton, an elliptic cone, 100 iterations and 50
line-search iterations for MJWarp's solvers.

The per-world initial stacks come from one seeded stream, so a larger batch extends
the same stream: its first 64 worlds are the batch-64 worlds.
"""

from __future__ import annotations

import numpy as np

from viewer.scenes.assets import ARMADILLO_CP_DIR

MU = 0.8
GAP = 0.15  # m, vertical clearance between stacked bodies
JITTER = 0.12  # m, horizontal offset range per body
SPACING = 3.0  # m, column grid pitch for multi-column scenes
DENSITY = 100.0
CUBE_HALF = 0.4  # m, cube half extent (~armadillo scale)
DT = 0.002  # MuJoCo's default timestep
WALL_RGBA = "0.7 0.85 1.0 0.12"  # transparent walls in renders
WALL_H, WALL_T = 9.0, 0.05  # m, the walls' half-height and half-thickness

SCENES = {
    "cube5": dict(kind="cube", n_bodies=5, columns=1),
    "cube20": dict(kind="cube", n_bodies=20, columns=4),
    "arm5": dict(kind="armadillo", n_bodies=5, columns=1),
    "arm20": dict(kind="armadillo", n_bodies=20, columns=4),
}

# Per-world capacities, calibrated with probes (armadillos are contact-rich).
CAPS = {
    "cube5": dict(njmax=192, nconmax=256),
    "cube20": dict(njmax=768, nconmax=1024),
    "arm5": dict(njmax=1024, nconmax=2048),
    "arm20": dict(njmax=4096, nconmax=8192),
}

OPTION = '<option cone="elliptic" solver="Newton" iterations="100" ls_iterations="50"/>'


def read_obj(path):
    """((nverts, 3) vertices, (3*ntris,) int32 CCW indices) of an OBJ; polygons fanned."""
    verts, tris = [], []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v":
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f":
                idx = [int(p.split("/")[0]) - 1 for p in parts[1:]]
                for k in range(1, len(idx) - 1):
                    tris += [idx[0], idx[k], idx[k + 1]]
    return np.array(verts), np.array(tris, dtype=np.int32)


# Where the decomposition sits on the baked surface: a quarter turn about x (it is
# y-up, the surface z-up; swapping y and z instead would mirror it), a uniform scale
# (~1 m to 1.64 m) and an offset, fitted to the surface by least squares on closest
# points. A convex decomposition bridges the surface's concavities, so its vertices
# sit a median 1.0 cm off the surface and 2.2 cm at the 90th percentile.
PIECES_ROTATION = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
PIECES_SCALE = 1.64159
PIECES_OFFSET = np.array([0.02473, 0.00015, 0.11494])


def load_armadillo_pieces():
    """
    The 34-piece convex decomposition, placed on the baked armadillo surface (see
    `PIECES_ROTATION`), so that pieces and surface are one body's two descriptions.

    Returns `(pieces, union)`: a list of `((nv_i, 3) verts, (3*nt_i,) tris)` and the
    `(sum nv_i, 3)` stack of every piece vertex, for the stacking extents.
    """
    raw = [read_obj(p) for p in sorted(ARMADILLO_CP_DIR.glob("cp*.obj"))]
    pieces = [(PIECES_SCALE * v @ PIECES_ROTATION.T + PIECES_OFFSET, t) for v, t in raw]
    union = np.vstack([v for v, _ in pieces])
    return pieces, union


def quat_to_mat(q):
    """Rotation matrix from a MuJoCo (w, x, y, z) quaternion."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def initial_qpos(n_bodies, verts, rng, columns=1, gap=GAP, jitter=JITTER, spacing=SPACING):
    """
    One provably intersection-free randomized stack, `(7 * n_bodies,)` free-joint qpos.

    Each body draws an orientation and is raised until its lowest rotated vertex
    clears the previous body's highest by `gap`; bodies go round-robin into `columns`
    stacks on a square grid `spacing` apart, each offset by up to `jitter`.
    """
    q = np.zeros(7 * n_bodies)
    side = int(np.ceil(np.sqrt(columns)))
    centres = [((c % side - (side - 1) / 2) * spacing, (c // side - (side - 1) / 2) * spacing) for c in range(columns)]
    top = [0.0] * columns

    for i in range(n_bodies):
        c = i % columns
        quat = rng.normal(size=4)
        quat /= np.linalg.norm(quat)
        vz = (verts @ quat_to_mat(quat).T)[:, 2]

        z = top[c] + gap - vz.min()
        q[7 * i : 7 * i + 2] = rng.uniform(-jitter, jitter, 2) + np.asarray(centres[c])
        q[7 * i + 2] = z
        q[7 * i + 3 : 7 * i + 7] = quat
        top[c] = z + vz.max()
    return q


def initial_qpos_batch(n_bodies, verts, seed, batch, columns=1, gap=GAP, jitter=JITTER):
    """`(batch, 7*n_bodies)`: independently randomized stacks from one seeded stream."""
    rng = np.random.default_rng(seed)
    return np.stack([initial_qpos(n_bodies, verts, rng, columns, gap, jitter) for _ in range(batch)])


def wall_xml(columns):
    """
    Four transparent static walls forming an open-top box around the columns, and
    the box's interior half-width.
    """
    side = int(np.ceil(np.sqrt(columns)))
    half = (side - 1) / 2 * SPACING + 1.5
    return box_walls_xml(half), half


def box_walls_xml(half, h=WALL_H, t=WALL_T, mu=MU):
    """
    Four transparent static walls of half-height `h` and half-thickness `t`, their inner
    faces `half` from the origin: an open-top box.
    """
    walls = []
    for i, (dx, dy, hx, hy) in enumerate([
        (half + t, 0, t, half + 2 * t),
        (-half - t, 0, t, half + 2 * t),
        (0, half + t, half + 2 * t, t),
        (0, -half - t, half + 2 * t, t),
    ]):
        walls.append(
            f'<geom name="wall{i}" type="box" pos="{dx} {dy} {h}" size="{hx} {hy} {h}" '
            f'friction="{mu}" rgba="{WALL_RGBA}"/>'
        )
    return "\n    ".join(walls)


def piece_assets(pieces):
    """`<mesh>` assets for the convex pieces, named cp0..cpN."""
    return [f'<mesh name="cp{k}" vertex="{" ".join(f"{x:.5f}" for x in v.reshape(-1))}"/>' for k, (v, _) in enumerate(pieces)]


def scene_xml(kind, n_bodies, columns):
    """`(xml, verts)`: the MJCF, and the `(nv, 3)` body vertices the stacks are built from."""
    assets, geoms = [], ""
    if kind == "cube":
        verts = CUBE_HALF * np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        geoms = f'<geom type="box" size="{CUBE_HALF} {CUBE_HALF} {CUBE_HALF}" friction="{MU}" density="{DENSITY}"/>'
    else:
        pieces, verts = load_armadillo_pieces()
        assets = piece_assets(pieces)
        geoms = "".join(f'<geom type="mesh" mesh="cp{k}" friction="{MU}" density="{DENSITY}"/>' for k in range(len(pieces)))

    bodies = "\n".join(f'    <body name="b{i}" pos="0 0 {2.0 + i}"><freejoint/>{geoms}</body>' for i in range(n_bodies))
    walls, _ = wall_xml(columns)
    nl = "\n    "
    return f"""
<mujoco>
  {OPTION}
  <asset>
    {nl.join(assets) if assets else "<!-- no assets -->"}
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="20 20 0.1" friction="{MU}"/>
    {walls}
{bodies}
  </worldbody>
</mujoco>
""", verts


def scene_and_q0(scene, nworld, seed=0):
    """`(xml, q0)`: the scene's MJCF and `(nworld, nq)` initial positions."""
    cfg = SCENES[scene]
    xml, verts = scene_xml(cfg["kind"], cfg["n_bodies"], cfg["columns"])
    return xml, initial_qpos_batch(cfg["n_bodies"], verts, seed, nworld, cfg["columns"])
