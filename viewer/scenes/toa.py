"""
Thin features under time of arrival (`toa`, `lodos.col.sdf.types.make_collision_context`): the viewer's `toa_*`
scenes. Each runs its Lodos pairs on `toa` by default -- the Collision pairs section turns it off, and every option
of it, for a comparison from the same state -- with several bodies in one world, a 2 ms step and, but for the
bullet, no gravity:

    wall     five 20 cm cubes shot at a fixed 4 cm card at 10, 20, 40, 80 and 160 m/s, half to eight of the
             card's thicknesses a step. With toa every one stops at the card's face, at rest; without, the slow
             ones sink in and are pushed back out, bouncing, the faster ones stick in it, since a step carries
             their vertices past its midsurface, and the fastest passes through
    flyby    the same cubes flying past the card's edge with 5 and 20 mm to spare, at 10, 40 and 80 m/s, and one
             clipping the edge by 10 mm: the four pass untouched and the fifth is struck. With the trace off
             (`toa trace`), a speculative contact linearized at its start stops or spins the passing ones as if
             they had hit (a ghost collision)
    ring     2 cm cubes sliding around the inside of four thin rings (radius 10 cm, a 1 cm wall), at 5, 10, 20
             and 40 m/s, nearly frictionless, with `toa_drift`. The wall curves into a sliding cube's straight
             path by v^2 dt^2 / (2 rho) a step, which the drift feeds forward; without it the depth settles at
             that over the Baumgarte gain, and the fast cubes stick in the wall or leave the ring. Each cube
             slows down as it goes, to about 2 m/s in a second: a step moves it along a chord of its circle,
             and the wall takes the part of its velocity that runs into it
    hammer   2 cm cubes on 0.3 m hinged arms swung at 30, 60, 90, 120 and 150 rad/s along thin plates (1 cm) on
             the pivot's side, 2 mm off them. Each step the arc bends into the plate by w^2 r dt^2 / 2, which the
             straight lines `toa` takes do not see: past half the plate's thickness (about 90 rad/s) a cube
             crosses its midsurface and is pushed on through. `toa_side` catches some of those, and the
             Collision pairs section reports the bend (`collide.check_toa_bend`): a smaller step is the remedy
    bullet   a light 20 cm cube (0.8 kg) shot at 160 m/s, 32 card thicknesses a step, at a free 1 cm steel card
             (0.6 by 1.2 m, 56 kg) hanging in the air, 0.18 m off its centre, under gravity. With toa it strikes the
             card, which spins away about its long axis (about two turns a second) as both fall to the floor;
             without, the cube clears the card between two steps and passes through it untouched

Every body that collides is a Lodos surface keyed to an MJCF geom in no collision of its own, its own body at
the surface's origin; the fixed ones are bodies without a joint. Each mover's pair is its vertices against its
own fixed surface's field, one-sided: two-sided, the card's vertices against a struck, spinning cube would
report a bend over the card's whole bounding sphere (`check_toa_bend`) that a solid cube has nothing to fear from.
The bullet's two free bodies collide both ways, and each with the floor, a half-space.
"""

from __future__ import annotations

import mujoco
import numpy as np

from lodos import sim as lsim
from lodos.col.sdf import types as sdf_types
from viewer.scenes.assets import load_surface

SCENES = ("wall", "flyby", "ring", "hammer", "bullet")
DT = 0.002

CARD, CARD_HALF = "card_0.04x1.25x2.5", (0.02, 0.625, 1.25)  # thin along x
CUBE, CUBE_HALF = "cube_0.2", 0.1
SMALL, SMALL_HALF = "cube_0.02", 0.01
RING, RING_R, RING_WALL, RING_H = "ring_0.1x0.01x0.06", 0.1, 0.01, 0.06  # inner radius, wall, height
PLATE, PLATE_HALF = "plate_0.01x0.12x0.06", (0.005, 0.06, 0.03)  # thin along x

# The wall: each cube's speed (m/s) and the fraction of a step's travel its start is set back by, which puts
# its arrival at a different point of a step.
WALL = ((10.0, 0.5), (20.0, 0.25), (40.0, 0.5), (80.0, 0.75), (160.0, 0.0))
X0 = -0.5  # where the cubes start, before the set-back (m)
# The fly-by: each cube's speed (m/s) and its near face's clearance past the card's edge (m); < 0 overlaps it.
FLYBY = ((10.0, 0.005), (40.0, 0.005), (80.0, 0.005), (40.0, 0.02), (40.0, -0.01))
CUBE_Z = (-1.0, -0.5, 0.0, 0.5, 1.0)  # the five cubes' heights, along the card
RING_SPEEDS = (5.0, 10.0, 20.0, 40.0)  # m/s
RING_GAP = 0.001  # each cube's start, its outer face this far inside the wall (m)
HAMMER_W = (30.0, 60.0, 90.0, 120.0, 150.0)  # rad/s
ARM, HAMMER_GAP = 0.3, 0.002  # the arm's length to the cube's centre; the cube's inner face to the plate (m)
STACK = 0.1  # the rings' and the hammers' spacing along z (m)
# The bullet: the cube's speed (m/s) and set-back (of a step's travel), how far off the card's centre it strikes
# (along the card's width, m), the card's centre height (m), and the two densities: a light cube and a cardboard-
# light card (kg/m^3), so that the card takes the cube's momentum as spin.
BULLET_V, BULLET_SETBACK, BULLET_Y, BULLET_Z = 160.0, 0.0, 0.18, 1.0
BULLET_DENSITY, BULLET_CARD_DENSITY = 100.0, 7800.0  # a 0.8 kg cube, a 56 kg (steel) card
# Where the cube starts: 3 m off, so that a step leaves its front face 6 cm before the card and the next 26 cm past
# it -- a cube 20 cm long clears a 1 cm card between two steps without touching it, unless toa sees it coming.
BULLET_X0 = -3.04
BULLET_CARD, BULLET_CARD_HALF = "card_0.01x0.6x1.2", (0.005, 0.3, 0.6)  # thin along x
DENSITY = 'density="1000"'  # every cube's, water's: its mass and inertia come from its MJCF box

MU = {"wall": 0.5, "flyby": 0.5, "ring": 0.001, "hammer": 0.5, "bullet": 0.5}
# The pairs' time of arrival: on everywhere, and the ring's drift.
TOA = {"wall": dict(toa=True), "flyby": dict(toa=True), "ring": dict(toa=True, toa_drift=True),
       "hammer": dict(toa=True), "bullet": dict(toa=True)}


def _f(a):
    return " ".join(f"{x:.9g}" for x in np.ravel(a))


def _density(rho):
    """A geom's density attribute (kg/m^3): its body's mass and inertia, from its MJCF shape."""
    return f'density="{rho:g}"'


def _geom(name, size, extra=""):
    """A box (`size` its half extents) or cylinder (radius, half height) keyed to a Lodos surface: no collision."""
    kind = "box" if len(size) == 3 else "cylinder"
    return f'<geom name="{name}" type="{kind}" size="{_f(size)}" contype="0" conaffinity="0" {extra}/>'


def scene_xml(name):
    """
    `(xml, qpos0, qvel0)`: the scene's MJCF and its start. The movers are the cubes: free joints, or under `hammer`
    hinges about z at the origin of each arm, whose cube is the arm's only mass; under `bullet` the card too.
    """
    if name not in SCENES:
        raise ValueError(f"no scene {name!r}; there are {SCENES}")
    bodies, qpos, qvel = [], [], []
    gravity = "0 0 0"
    if name == "bullet":
        gravity = "0 0 -9.81"
        bodies.append('<geom name="ground" type="plane" size="20 20 0.1"/>')
        card = (0.0, 0.0, BULLET_Z)
        bodies.append(f'<body name="card" pos="{_f(card)}"><freejoint/>'
                      f'{_geom("card", BULLET_CARD_HALF, extra=_density(BULLET_CARD_DENSITY))}</body>')
        h = CUBE_HALF
        cube = (BULLET_X0 - BULLET_SETBACK * BULLET_V * DT, BULLET_Y, BULLET_Z)
        bodies.append(f'<body name="cube0" pos="{_f(cube)}"><freejoint/>'
                      f'{_geom("cube0", (h, h, h), extra=_density(BULLET_DENSITY))}</body>')
        qpos += [*card, 1.0, 0.0, 0.0, 0.0, *cube, 1.0, 0.0, 0.0, 0.0]
        qvel += [0.0] * 6 + [BULLET_V, 0.0, 0.0, 0.0, 0.0, 0.0]
    elif name in ("wall", "flyby"):
        bodies.append(f'<body name="card">{_geom("card", CARD_HALF)}</body>')
        h = CUBE_HALF
        rows = WALL if name == "wall" else FLYBY
        for k, ((v, a), z) in enumerate(zip(rows, CUBE_Z)):
            x, y = (X0 - a * v * DT, 0.0) if name == "wall" else (X0, CARD_HALF[1] + h + a)
            bodies.append(f'<body name="cube{k}" pos="{_f((x, y, z))}"><freejoint/>'
                          f'{_geom(f"cube{k}", (h, h, h), extra=DENSITY)}'
                          f'</body>')
            qpos += [x, y, z, 1.0, 0.0, 0.0, 0.0]
            qvel += [v, 0.0, 0.0, 0.0, 0.0, 0.0]
    elif name == "ring":
        h = SMALL_HALF
        rc = RING_R - h - RING_GAP  # the cube's centre: its outer face RING_GAP inside the wall
        for k, v in enumerate(RING_SPEEDS):
            z = k * STACK
            bodies.append(f'<body name="ring{k}" pos="0 0 {z:.9g}">'
                          f'{_geom(f"ring{k}", (RING_R + RING_WALL, RING_H / 2))}</body>')
            bodies.append(f'<body name="cube{k}" pos="{_f((rc, 0.0, z))}"><freejoint/>'
                          f'{_geom(f"cube{k}", (h, h, h), extra=DENSITY)}'
                          f'</body>')
            qpos += [rc, 0.0, z, 1.0, 0.0, 0.0, 0.0]
            qvel += [0.0, v, 0.0, 0.0, 0.0, 0.0]  # along the wall, a face toward it
    else:
        h = SMALL_HALF
        face = ARM - h - HAMMER_GAP  # the plate's face toward the cube (x)
        for k, w in enumerate(HAMMER_W):
            z = k * STACK
            # The plate on the pivot's side of the cube's circle, along its start: the arc bends into it.
            centre = (face - PLATE_HALF[0], PLATE_HALF[1] - 0.03, z)
            bodies.append(f'<body name="plate{k}" pos="{_f(centre)}">{_geom(f"plate{k}", PLATE_HALF)}</body>')
            bodies.append(f'<body name="arm{k}" pos="0 0 {z:.9g}">'
                          f'<joint name="hinge{k}" type="hinge" axis="0 0 1"/>'
                          f'<inertial pos="0 0 0" mass="0.001" diaginertia="1e-6 1e-6 1e-6"/>'
                          f'<body name="tip{k}" pos="{ARM:.9g} 0 0">'
                          f'{_geom(f"cube{k}", (h, h, h), extra=DENSITY)}'
                          f'</body></body>')
            qpos.append(0.0)
            qvel.append(w)
    nl = "\n    "
    xml = f"""
<mujoco model="toa_{name}">
  <option timestep="{DT}" gravity="{gravity}" cone="elliptic" solver="Newton" iterations="100" ls_iterations="50"/>
  <worldbody>
    {nl.join(bodies)}
  </worldbody>
</mujoco>
"""
    return xml, np.array(qpos), np.array(qvel)


def pairs(name):
    """
    `[(a, b)]`: the scene's pairs over its surfaces as `surfaces` orders them -- the movers, then the fixed ones;
    the bullet's cube, card and floor.
    """
    if name == "bullet":
        return [(0, 1), (0, 2), (1, 2)]  # the cube and the card (both ways), each against the floor
    n = 5 if name in ("wall", "flyby") else (len(RING_SPEEDS) if name == "ring" else len(HAMMER_W))
    if name in ("wall", "flyby"):
        return [(k, n) for k in range(n)]  # every cube against the one card
    return [(k, n + k) for k in range(n)]  # each cube against its own ring or plate


def surfaces(mjm, name):
    """The scene's Lodos surfaces: the movers' cubes, then the fixed card, rings or plates; `TOA[name]` on."""
    geom = lambda g: mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, g)  # noqa: E731
    cprs = [list(p) for p in pairs(name)]
    n = len(cprs)
    if name == "bullet":
        surf = [load_surface(CUBE), load_surface(BULLET_CARD), sdf_types.PlaneSurface()]
        geoms = [geom("cube0"), geom("card"), geom("ground")]
        col = dict(TOA[name], cpr_is_symmetric=[True, False, False])  # a plane has no vertices: one-sided
    elif name in ("wall", "flyby"):
        surf = [load_surface(CUBE)] * n + [load_surface(CARD)]
        geoms = [geom(f"cube{k}") for k in range(n)] + [geom("card")]
        col = dict(TOA[name])
    else:
        fixed, key = (RING, "ring") if name == "ring" else (PLATE, "plate")
        surf = [load_surface(SMALL)] * n + [load_surface(fixed)] * n
        geoms = [geom(f"cube{k}") for k in range(n)] + [geom(f"{key}{k}") for k in range(n)]
        col = dict(TOA[name])
    return lsim.Surfaces(surfaces=surf, geoms=geoms, pairs=cprs, params=lsim.default_pair_params(MU[name], n),
                         mu=MU[name], col_kwargs=col)


def assets():
    """The surfaces the scenes load, for `viewer.scenes.catalog`'s availability check."""
    return {"wall": (CUBE, CARD), "flyby": (CUBE, CARD), "ring": (SMALL, RING), "hammer": (SMALL, PLATE),
            "bullet": (CUBE, BULLET_CARD)}
