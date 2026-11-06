"""
The exactness scenes of Song, Fan, Ascher and Pai's FBF paper ("A Splitting Architecture
for Exact Reduced Coulomb Friction", SCA 2026), in either collision representation:

    incline    a 0.2 m box on a slope of atan(1/2) at mu 0.45, under the slip threshold
               0.5, so it slides, at g (sin - mu cos) = 0.44 m/s^2
    backspin   a ball (r 0.25) launched at 4 m/s with -200 rad/s of backspin, mu 0.5: it
               rolls back at (5 v0 + 2 r w0) / 7 = -11.43 m/s and stays on the ground,
               where a convex relaxation throws it into the air
    painleve   a 0.3 x 0.6 x 1.2 box sliding at 4 m/s, mu 0.5, at the threshold between
               sliding and tipping over: exact friction slides it
    cards      a house of 40 cards in five levels, mu 0.8, which should stand
    arch       a masonry arch of 25 stones, the end ones welded, mu 0.8, which should
               stand

`rep="surfaces"` (the viewer's "sdf") collides every body as a Lodos surface, against the
others and the ground -- a half-space, the incline's slope one on a static body turned by
its angle. `rep="polytopes"` (the viewer's "polytope") collides MJCF primitives -- boxes,
the sphere -- or the arch's convex stones, through MJWarp's pipeline. The bodies, their
inertia and their start are the same either way.

Parameters are FBF's: dt 1/60, the cards and the arch 1/240 (FBF's 1/60 frames in four
substeps). The incline tilts its slope rather than gravity, which is the same physics
and looks like an incline. The Lodos pairs take `lodos.sim.default_pair_params` with
each scene's friction: the spring-damper is not tuned for these scenes yet.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from lodos import sim as lsim
from viewer.scenes.assets import ASSET_DIR, load_surface
from lodos.col.sdf import types as sdf_types

SCENES = ("incline", "backspin", "painleve", "cards", "arch")
REPS = ("surfaces", "polytopes")
G = 9.81

THETA = np.arctan(0.5)  # the incline's slope
INCLINE_HALF, INCLINE_MU = 0.1, 0.45
BALL_R, BALL_V0, BALL_W0, BALL_MU = 0.25, 4.0, -200.0, 0.5
BOX_HALF, BOX_V0, BOX_MU = (0.15, 0.3, 0.6), 4.0, 0.5
CARD_HALF, CARD_MU, CARD_DENSITY, LEVELS = (0.02, 0.625, 1.25), 0.8, 200.0, 5  # thin along x
TENT_HALF_GAP = 0.55
TENT_W = 4 * TENT_HALF_GAP
TENT_H = (np.tan(np.radians(65.0)) * TENT_W + np.tan(np.radians(3.0)) * TENT_W) / 2.0
ARCH_DIR = ASSET_DIR / "arch" / "num_stones=25"
ARCH_STONES, ARCH_MU, ARCH_DENSITY = 25, 0.8, 2000.0

DT = {"incline": 1 / 60, "backspin": 1 / 60, "painleve": 1 / 60, "cards": 1 / 240, "arch": 1 / 240}
MU = {"incline": INCLINE_MU, "backspin": BALL_MU, "painleve": BOX_MU, "cards": CARD_MU, "arch": ARCH_MU}
# MJWarp's capacities per world, for every scene: MJWarp's defaults overflow on the arch's
# stones, and a Lodos contact is an MJWarp contact (the cards' edges).
CAPS = {"surfaces": dict(nconmax=4096, njmax=16384), "polytopes": dict(nconmax=1024, njmax=4096)}

# Each body's Lodos surface (`tools/mesh_processing/scene_surfaces.py` makes them).
SURFACES = {
    "incline": "cube_0.2",
    "backspin": "sphere_0.25",
    "painleve": "box_0.3x0.6x1.2",
    "cards": "card_0.04x1.25x2.5",
}


def stone_surface(i):
    return f"arch25_stone{i:02d}"


def _quat_y(rad):
    """The wxyz quaternion of a turn by `rad` about y."""
    return np.array([np.cos(rad / 2), 0.0, np.sin(rad / 2), 0.0])


def _f(a):
    return " ".join(f"{x:.9g}" for x in np.ravel(a))


def load_stone(i):
    """
    `(v, f, center)`: stone `i` (1-based) of the arch, turned z-up from the dataset's y-up,
    its vertices about `center`, its centroid -- the stone's body frame.
    """
    v, f = [], []
    for line in open(ARCH_DIR / f"stone-{i:02d}.obj"):
        parts = line.split()
        if parts and parts[0] == "v":
            x, y, z = (float(p) for p in parts[1:4])
            v.append([x, z, y])
        elif parts and parts[0] == "f":
            f.append([int(p.split("/")[0]) - 1 for p in parts[1:4]])
    v = np.array(v)
    f = np.array(f)
    # Turning y-up into z-up swaps two axes, a reflection, so the winding flips with it.
    f = f[:, ::-1]
    center = v.mean(axis=0)
    return v - center, f, center


def _bodies(name):
    """
    `[(geom attributes, pos, wxyz quat, free)]`: each body's collision geom, without its
    name, friction or contact switches, and its start; plus the ground's pose.
    """
    if name == "incline":
        q = _quat_y(THETA)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q)
        pos = R.reshape(3, 3) @ np.array([-1.0, 0.0, INCLINE_HALF - 1e-4])
        h = INCLINE_HALF
        return [(f'type="box" size="{h} {h} {h}" mass="1"', pos, q, True)]
    if name == "backspin":
        return [(f'type="sphere" size="{BALL_R}" mass="1"', (0, 0, BALL_R - 1e-4), (1, 0, 0, 0), True)]
    if name == "painleve":
        return [(f'type="box" size="{_f(BOX_HALF)}" mass="1"', (0, 0, BOX_HALF[2] - 1e-4), (1, 0, 0, 0), True)]
    if name == "cards":
        out = []
        size = f'type="box" size="{_f(CARD_HALF)}" density="{CARD_DENSITY}"'
        for i in range(LEVELS):
            for j in range(LEVELS - i):
                x = (j - (LEVELS - i) / 2.0 + 0.5) * TENT_W
                if i != 0:
                    # The plate is thin along x, as every card here is, so that one surface
                    # serves all 40, and turned 89 degrees about y: the place of a plate thin
                    # along z turned -1 degree.
                    out.append((size, (x, 0, i * TENT_H), _quat_y(np.radians(89.0)), True))
                for sign in (1.0, -1.0):
                    out.append((size, (x - sign * TENT_HALF_GAP, 0, (i + 0.5) * TENT_H),
                                _quat_y(np.radians(sign * 25.0)), True))
        return out
    out = []
    for i in range(1, ARCH_STONES + 1):
        _, _, center = load_stone(i)
        out.append((f'type="mesh" mesh="stone{i:02d}" density="{ARCH_DENSITY}"', center, (1, 0, 0, 0),
                    i not in (1, ARCH_STONES)))
    return out


def scene_xml(name, rep):
    """`(xml, qvel0)`: the scene's MJCF, which starts at its `qpos0`, and its initial velocity."""
    if name not in SCENES:
        raise ValueError(f"no scene {name!r}; there are {SCENES}")
    mu = MU[name]
    assets = []
    if name == "arch":
        for i in range(1, ARCH_STONES + 1):
            v, _, _ = load_stone(i)
            assets.append(f'<mesh name="stone{i:02d}" vertex="{_f(v)}"/>')
    collide = f'friction="{mu}"' if rep == "polytopes" else 'contype="0" conaffinity="0"'
    bodies = []
    for k, (geom, pos, q, free) in enumerate(_bodies(name)):
        joint = "<freejoint/>" if free else ""
        bodies.append(f'<body name="b{k}" pos="{_f(pos)}" quat="{_f(q)}">{joint}'
                      f'<geom name="g{k}" {geom} {collide}/></body>')
    ground = f'<geom name="ground" type="plane" size="200 200 0.1" friction="{mu}"/>'
    if name == "incline":
        ground = (f'<body name="slope" quat="{_f(_quat_y(THETA))}">'
                  f'<geom name="ground" type="plane" size="20 20 0.1" friction="{mu}"/></body>')
    nl = "\n    "
    xml = f"""
<mujoco model="{name}">
  <option cone="elliptic" solver="Newton" iterations="100" ls_iterations="50" timestep="{DT[name]}"/>
  <asset>
    {nl.join(assets) if assets else "<!-- no assets -->"}
  </asset>
  <worldbody>
    {ground}
    {nl.join(bodies)}
  </worldbody>
</mujoco>
"""
    nbody = len(bodies)
    qvel0 = np.zeros(6 * sum(1 for b in _bodies(name) if b[3]))
    if name == "backspin":
        qvel0[[0, 4]] = BALL_V0, BALL_W0
    elif name == "painleve":
        qvel0[1] = BOX_V0
    assert nbody == len(_bodies(name))
    return xml, qvel0


def surfaces(mjm, name, col_kwargs=None):
    """
    The Lodos half of the `"surfaces"` representation: every ordered pair of bodies of which
    one moves, and every moving body against the ground.
    """
    bodies = _bodies(name)
    n = len(bodies)
    free = [b[3] for b in bodies]
    surf = ([load_surface(stone_surface(i)) for i in range(1, n + 1)] if name == "arch"
            else [load_surface(SURFACES[name])] * n)
    cprs = [[a, b] for a in range(n) for b in range(n) if a != b and (free[a] or free[b])]
    cprs += [[a, n] for a in range(n) if free[a]]
    geom = lambda g: mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, g)  # noqa: E731
    return lsim.Surfaces(
        surfaces=surf + [sdf_types.PlaneSurface()],
        geoms=[geom(f"g{k}") for k in range(n)] + [geom("ground")],
        pairs=cprs, params=lsim.default_pair_params(MU[name], len(cprs)), mu=MU[name],
        col_kwargs=dict(col_kwargs or {}),
    )


def build(name, rep, B, solver_kwargs=None, col_kwargs=None, **sim_kwargs):
    """
    The scene as a `lodos.sim.Sim` of `B` identical worlds, at its start. MJWarp's
    capacities default to `CAPS[rep]`; `sim_kwargs` override them.
    """
    if rep not in REPS:
        raise ValueError(f"rep must be one of {REPS}")
    xml, qvel0 = scene_xml(name, rep)
    mjm = mujoco.MjModel.from_xml_string(xml)
    surf = surfaces(mjm, name, col_kwargs) if rep == "surfaces" else None
    s = lsim.make_sim(mjm, B, surfaces=surf, solver_kwargs=solver_kwargs, **{**CAPS[rep], **sim_kwargs})
    s.drbd.qvel.assign(np.tile(qvel0.astype(np.float32), (B, 1)))
    return s


def stone_paths():
    """The arch's stone OBJs, for `tools/mesh_processing/scene_surfaces.py`."""
    return [Path(ARCH_DIR / f"stone-{i:02d}.obj") for i in range(1, ARCH_STONES + 1)]
