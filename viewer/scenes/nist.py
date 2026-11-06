"""
Isaac Lab's NIST Task Board 1 parts (`Factory/NIST`, asset release 6.2, the geometry its ten-part
demo `examples/demos/nist.py` collides) on Lodos surfaces. For now the three nut-bolt pairs, M8, M12
and M16, each run as `viewer.scenes.nutbolt` runs Newton's M20 pair: the bolt fixed, the nut started
on its thread, screwing itself down under gravity; and the three together on the board
(`board_xml`, `build_board`), at the demo's layout.

The scene:

    surfaces   0 the bolt, its field only; 1 the nut, its vertices; 2 the floor, a plane at z = 0
    pairs      the nut's vertices on the bolt's field, and on the floor
    bodies     the bolt fixed to the world (a body without a joint), its lowest point on the floor;
               the nut free, 30 g as in Isaac Lab's demo, MuJoCo's exact mesh inertia scaled to it
    friction   0.01 on the thread, the M20 example's; 0.5 on the floor
    dt         1/480, the M20 example's (Isaac Lab's demo substeps at 1/800)
    integrator implicitfast, as Isaac Lab's demo runs MuJoCo Warp
    start      the nut's bottom `START_TURNS` pitches below the bolt's tip, centred on its axis and
               turned to the thread's phase there (`thread_phase`)

What is fixed to the board is a field, and what moves supplies vertices: the bolt contributes its
field and nothing reads its vertices; the nut contributes its vertices, and its own field (0.2 mm)
is for the pairs that will need it.

The assets ship in `viewer/assets/nist` (`$LODOS_NIST_DIR` overrides it), though Isaac Lab's NIST parts state
no licence beyond the older Factory parts' BSD-3-Clause. `tools/mesh_processing/nist` makes them from Isaac
Lab's USD files, `extract.py` then `make_surfaces.py`, which say why each step is what it is. Each nut is remeshed at three edge lengths (`NUTS`): two fifths and a fifth of
its pitch, as the M20 nut's 1 and 0.5 mm stand to its 2.5 mm, and a tenth. The M8 nut has 3235,
13378 and 53120 vertices, the M12 3622, 15191 and 60532, the M16 4881, 19886 and 79611.

The collision meshes, as `extract.py` reads them (z in each part's own frame):

    size  pitch    bolt's thread       tip    thread radii   nut's bottom
    M8    1.25 mm  z 9.8 to 26 mm      26 mm  3.01-3.98 mm   9.34 mm
    M12   1.75     z 12 to 32          32     4.60-5.96      12.99
    M16   2.0      z 10 to 35, no head 35     5.89-7.38      12.66

Below the thread, the M8 and M12 bolts have a flange (r 6.6 and 9.1 mm) that stops the nut. The M16
bolt's collision mesh is its threaded shank alone, scaled 0.95 across its axis by its USD (the
"loose" pair's clearance), and it starts 10 mm up its frame: its body stands 10 mm lower, and the
floor stops the nut. Isaac Lab's demo counts a nut assembled 8.0 (M8), 10.2 (M12) and 13 mm (M16)
below the tip (`SEATED`), which is not a stop: under gravity the nut goes on to the flange.

**The board** collides as one box, 385 x 384.6 x 30 mm (`nistboard.usd`), fixed at the demo's
starting pose (`BOARD_POSE`): turned half about y, so its top face is at z = 20.4 mm. The bolts
stand on it at the demo's offsets (`BOLT_ON_BOARD`), upright, their lower parts inside it; the
board and the bolts are fixed, so nothing collides them with each other. On the board the board's
face stops the M8 and M12 nuts, 1.0 and 0.8 mm below where the demo counts them assembled. The
board is a field, like the bolts (`surfaces/nistboard`, `tools/mesh_processing/nist/make_surfaces.py
--board`), and the nuts' vertices meet it.
"""

from __future__ import annotations

import functools
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import NamedTuple

import mujoco
import numpy as np

from viewer.scenes import assets
from lodos import sim as lsim
from lodos.col import sdf
from lodos.col.sdf import types as sdf_types
from viewer.scenes import nutbolt as NB

ASSET_DIR = Path(os.environ.get("LODOS_NIST_DIR", assets.ASSET_DIR / "nist"))

class Size(NamedTuple):
    """One nut-bolt pair, from its collision meshes (table above); lengths in metres, z in each part's frame."""

    nut: str
    bolt: str
    pitch: float
    base: float  # the bolt's lowest point
    tip: float  # the bolt's tip
    nut_bottom: float
    radius: float  # the thread's mean radius, (3.01 + 3.98) / 2 and so on


SIZES = {
    "m8": Size("nut_m8", "bolt_m8", 1.25e-3, 0.0, 26.0e-3, 9.34e-3, 3.495e-3),
    "m12": Size("nut_m12", "bolt_m12", 1.75e-3, 0.0, 32.0e-3, 12.99e-3, 5.28e-3),
    "m16": Size("nut_m16", "bolt_m16", 2.0e-3, 10.0e-3, 35.0e-3, 12.66e-3, 6.635e-3),
}
NUTS = {"m8": ("nut_m8_s0.5", "nut_m8_s0.25", "nut_m8_s0.125"),  # remeshed at 0.4, 0.2 and 0.1 pitches
        "m12": ("nut_m12_s0.7", "nut_m12_s0.35", "nut_m12_s0.175"),
        "m16": ("nut_m16_s0.8", "nut_m16_s0.4", "nut_m16_s0.2")}

# Isaac Lab's demo (`examples/demos/nist.py`, ASSEMBLIES): the nut's reference point (`held_align`,
# in the nut's frame) is assembled when it reaches `seated` (in the bolt's frame), along z.
HELD_ALIGN = {"m8": 0.0093, "m12": 0.013, "m16": 0.01}
SEATED = {"m8": 0.018, "m12": 0.0218, "m16": 0.022}

# The demo's board, at its starting pose (NistSceneCfg.nistboard), and each bolt's pose on it
# (ASSEMBLIES' board_pose): position, then the quaternion as MuJoCo orders it, wxyz. The demo writes
# them xyzw: the board (0, 1, 0, 0), half a turn about y; each bolt (1, 0, 0, 0), half about x.
BOARD_POSE = ((0.452824, 0.19145, 0.0206), (0.0, 0.0, 1.0, 0.0))
BOLT_ON_BOARD = {"m8": ((0.0473, -0.0407, 0.0172), (0.0, 1.0, 0.0, 0.0)),
                 "m12": ((0.3473, -0.2665, 0.0212), (0.0, 1.0, 0.0, 0.0)),
                 "m16": ((0.04715, -0.3416, 0.0194), (0.0, 1.0, 0.0, 0.0))}
MU_BOARD = 0.75  # the demo's parts and board (CONTACT_MATERIALS)
MU_GROUND = 1.0  # the demo's ground

DT = 1.0 / 480.0
MU = 0.01
MU_FLOOR = 0.5
NUT_MASS = 0.03  # Isaac Lab's demo's plug_mass for the three nuts
START_TURNS = 2.0
FLANK = np.radians(30.0)  # ISO metric: half the 60 degree thread angle
# The M20 nut's binning when its contacts are grouped (`lodos.cd.groups`), which held a seated nut.
GROUP_BINS = NB.GROUP_BINS
# The solver's default 0.2 mm Baumgarte slop lets a self-locked nut (mu 0.75) sink into the thread: errors
# of its solve integrate into penetration that nothing corrects until a contact is 0.2 mm deep, and the M8
# nut went 0.26 mm down in 4 s, tilting to 1.9 degrees. At slop 0 it settles 0.044 mm and holds, as at 40
# proximal iterations and both proximal tolerances at 1e-5; threading at mu 0.01 keeps its pitch either
# way. The solver's options on these scenes, and the viewer's defaults for them too (`viewer.scenes.catalog`).
OPTIONS = dict(baumgarte_slop=0.0)


def solver_options(solver_kwargs=None):
    """The scene's options, `OPTIONS`, under the caller's."""
    return dict(OPTIONS, **(solver_kwargs or {}))


def default_nut(size):
    """The coarsest remesh, two fifths of the pitch: the M20 nut's robust 1 mm stands so to its pitch."""
    return NUTS[size][0]


def assets_present(size="m8", nut=None, asset_dir=None):
    d = Path(asset_dir or ASSET_DIR)
    S = SIZES[size]
    return all((d / p).exists() for p in (f"repair/{S.bolt}.obj", f"repair/{S.nut}.obj",
                                          f"surfaces/{S.bolt}/surface_data.npz",
                                          f"surfaces/{nut or default_nut(size)}/surface_data.npz"))


def _require(size, nut, asset_dir):
    if not assets_present(size, nut, asset_dir):
        raise FileNotFoundError(
            f"no NIST {size} nut-bolt assets in {asset_dir or ASSET_DIR}: make them with tools/mesh_processing/nist "
            f"or point LODOS_NIST_DIR at them"
        )


def start_z(size, start_turns=START_TURNS):
    """The nut body's height with its bottom `start_turns` pitches below the bolt's tip."""
    S = SIZES[size]
    return S.tip - S.base - start_turns * S.pitch - S.nut_bottom


def seated_z(size):
    """The nut body's height at Isaac Lab's demo's assembled pose (`SEATED`, `HELD_ALIGN`)."""
    return SEATED[size] - SIZES[size].base - HELD_ALIGN[size]


def scene_xml(size="m8", nut_z=None, nut_turn=0.0, dt=DT, integrator="implicitfast", asset_dir=None):
    d = Path(asset_dir or ASSET_DIR)
    _require(size, None, d)
    S = SIZES[size]
    z = start_z(size) if nut_z is None else nut_z
    q = (np.cos(nut_turn / 2), 0.0, 0.0, np.sin(nut_turn / 2))
    mjcf_integrator = {"euler": "Euler", "implicitfast": "implicitfast"}[integrator]
    return f"""<mujoco model="nist_nutbolt_{size}">
  <option timestep="{dt}" integrator="{mjcf_integrator}" cone="elliptic"/>
  <asset>
    <mesh name="bolt" file="{d / 'repair' / f'{S.bolt}.obj'}" inertia="exact"/>
    <mesh name="nut" file="{d / 'repair' / f'{S.nut}.obj'}" inertia="exact"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" pos="0 0 0" size="0.1 0.1 0.01"/>
    <body name="bolt" pos="0 0 {-S.base}"><geom name="g_bolt" type="mesh" mesh="bolt"/></body>
    <body name="nut" pos="0 0 {z}" quat="{q[0]} 0 0 {q[3]}"><freejoint/><geom name="g_nut" type="mesh" mesh="nut" mass="{NUT_MASS}"/></body>
  </worldbody>
</mujoco>"""


def load(name, spline=None, asset_dir=None):
    """A bake under `surfaces/`, evaluated tricubically or, `spline="linear"`, trilinearly."""
    d = Path(asset_dir or ASSET_DIR)
    s = sdf.load_surface_data(str(d / "surfaces" / name / "surface_data.npz"), spline_type=spline)
    s.sph_c = np.zeros((0, 3))
    s.sph_r = np.zeros(0)
    return s


def surfaces(mjm, size="m8", nut=None, spline=None, mu=MU, col_kwargs=None, asset_dir=None):
    """The nut's vertices on the bolt's field and on the floor."""
    gid = lambda n: mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, n)
    pairs = [[1, 0], [1, 2]]
    params = lsim.default_pair_params(mu, len(pairs))
    params[1, 0] = MU_FLOOR
    return lsim.Surfaces(
        surfaces=[load(SIZES[size].bolt, spline, asset_dir), load(nut or default_nut(size), spline, asset_dir),
                  sdf_types.PlaneSurface()],
        geoms=[gid("g_bolt"), gid("g_nut"), gid("floor")],
        pairs=pairs, params=params, mu=mu, col_kwargs=dict(col_kwargs or {}),
    )


def initial_qpos(B, size="m8", nut_z=None, nut_turn=0.0, jitter=0.0, seed=0):
    """
    The nut at `nut_z` (by default `start_z`), turned `nut_turn` about the bolt's axis, in every
    world; `jitter` turns each world's nut further by up to that many radians, seeded, as
    `viewer.scenes.nutbolt.initial_qpos` does.
    """
    turn = np.full(B, float(nut_turn))
    if jitter:
        turn += float(jitter) * np.random.default_rng(seed).uniform(-1.0, 1.0, size=B)
    q = np.zeros((B, 7))
    q[:, 2] = start_z(size) if nut_z is None else nut_z
    q[:, 3] = np.cos(turn / 2)
    q[:, 6] = np.sin(turn / 2)
    return q


@functools.lru_cache(maxsize=None)
def thread_phase(size="m8", nut=None, spline=None, start_turns=START_TURNS, asset_dir=None, n=360, tol=5e-6):
    """
    The turn at which the nut's thread meets the bolt's, with the nut at `start_z(size, start_turns)`.

    One collision pass over `n` worlds, the nut turned 2 pi k / n in world k. The turns whose
    deepest contact is shallower than `tol` lie within the thread's clearance; the middle of the
    longest such run (around the circle) is returned, with that run's width in radians and the
    deepest contact at the returned turn. If no turn is clear, the shallowest is returned, width 0.
    """
    from lodos import step as lstep
    from lodos.col.sdf.collide import get_contacts

    z = start_z(size, start_turns)
    turns = 2 * np.pi * np.arange(n) / n
    mjm = mujoco.MjModel.from_xml_string(scene_xml(size, z, asset_dir=asset_dir))
    q = initial_qpos(n, size, z)
    q[:, 3], q[:, 6] = np.cos(turns / 2), np.sin(turns / 2)
    s = lsim.make_sim(mjm, n, surfaces=surfaces(mjm, size, nut, spline, asset_dir=asset_dir), qpos=q)
    lstep.forward(s)
    c = get_contacts(s.mcol, s.dcol)
    deepest = np.zeros(n)
    on_bolt = c["cpr"] == 0
    np.minimum.at(deepest, c["batch"][on_bolt], c["sdf"][on_bolt])
    clear = deepest > -tol
    if not clear.any():
        k = int(np.argmax(deepest))
        return float(turns[k]), 0.0, float(-deepest[k])
    if clear.all():
        raise ValueError(f"{size}: the nut touches the bolt at no turn; it is not on the thread at z = {z}")
    # runs of clear turns, around the circle: start each run after a blocked turn
    k0 = int(np.argmin(clear))
    order = np.roll(np.arange(n), -k0)
    best, start, length = (0, 0), None, 0
    for i, k in enumerate(order):
        if clear[k]:
            start = i if start is None else start
            length += 1
            if length > best[1]:
                best = (start, length)
        else:
            start, length = None, 0
    mid = order[best[0] + best[1] // 2]
    return float(turns[mid]), float(2 * np.pi * best[1] / n), float(-deepest[mid])


def build(B, size="m8", nut=None, spline=None, start_turns=START_TURNS, dt=DT, mu=MU,
          integrator="implicitfast", solver_kwargs=None, col_kwargs=None, asset_dir=None, jitter=0.0, **sim_kwargs):
    """
    `B` worlds of the `size` nut-bolt, as a `lodos.sim.Sim`: the nut started on the thread at its
    phase (`thread_phase`), each world's turned further by up to `jitter` radians.

    solver_kwargs: the solver's options, over the scene's (`solver_options`).
    sim_kwargs: `lodos.sim.make_sim`'s, e.g. `contact_reduction=True` and MJWarp's capacities,
        which every Lodos contact reaches.
    """
    nut = nut or default_nut(size)
    z = start_z(size, start_turns)
    turn = thread_phase(size, nut, spline, start_turns, None if asset_dir is None else str(asset_dir))[0]
    mjm = mujoco.MjModel.from_xml_string(scene_xml(size, z, turn, dt, integrator, asset_dir))
    surf = surfaces(mjm, size, nut, spline, mu, col_kwargs, asset_dir)
    return lsim.make_sim(mjm, B, surfaces=surf, qpos=initial_qpos(B, size, z, turn, jitter),
                         solver_kwargs=solver_options(solver_kwargs), **sim_kwargs)


# The driven nut (`driven_xml`): turned at a set rate about the bolt's axis by a velocity servo, as the
# arm would screw it, free along the axis and held across it by springs; its tilt is fixed.
# The servo is velocity-implicit under implicitfast, which keeps it stable although its time constant,
# the nut's ~1e-6 kg m^2 over kv, is a hundredth of a step. Its torque is not limited: implicitfast
# drops an actuator's derivative where its force is clamped, and with a 50 mN m limit the torque
# flips between the limits every step. So a stall at the flange or the board holds kv times the
# rate, 31 mN m at a turn a second; a thread needing 1 mN m lags the rate by 0.2 rad/s, 3 %.
DRIVE_KV = 0.005  # N m s/rad
LATERAL_K, LATERAL_C = 500.0, 2.0  # N/m, N s/m: the hand's give across the axis


def driven_xml(size="m8", nut_z=None, nut_turn=0.0, dt=DT, integrator="implicitfast", asset_dir=None):
    """
    The lone pair, the nut on joints instead of free: slides across the axis on springs, a free slide
    along it, and a hinge about it turned by the velocity servo `spin` (its control a rate, rad/s;
    negative turns the nut on, clockwise seen from above). qpos is (x, y, z, turn) from the start.
    """
    d = Path(asset_dir or ASSET_DIR)
    _require(size, None, d)
    S = SIZES[size]
    z = start_z(size) if nut_z is None else nut_z
    q = (np.cos(nut_turn / 2), 0.0, 0.0, np.sin(nut_turn / 2))
    mjcf_integrator = {"euler": "Euler", "implicitfast": "implicitfast"}[integrator]
    lat = f'stiffness="{LATERAL_K}" damping="{LATERAL_C}"'
    return f"""<mujoco model="nist_driven_{size}">
  <option timestep="{dt}" integrator="{mjcf_integrator}" cone="elliptic"/>
  <asset>
    <mesh name="bolt" file="{d / 'repair' / f'{S.bolt}.obj'}" inertia="exact"/>
    <mesh name="nut" file="{d / 'repair' / f'{S.nut}.obj'}" inertia="exact"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" pos="0 0 0" size="0.1 0.1 0.01"/>
    <body name="bolt" pos="0 0 {-S.base}"><geom name="g_bolt" type="mesh" mesh="bolt"/></body>
    <body name="nut" pos="0 0 {z}" quat="{q[0]} 0 0 {q[3]}">
      <joint name="nut_x" type="slide" axis="1 0 0" {lat}/>
      <joint name="nut_y" type="slide" axis="0 1 0" {lat}/>
      <joint name="nut_z" type="slide" axis="0 0 1"/>
      <joint name="nut_turn" type="hinge" axis="0 0 1"/>
      <geom name="g_nut" type="mesh" mesh="nut" mass="{NUT_MASS}"/>
    </body>
  </worldbody>
  <actuator>
    <velocity name="spin" joint="nut_turn" kv="{DRIVE_KV}"/>
  </actuator>
</mujoco>"""


def build_driven(B, size="m8", nut=None, rate=-2 * np.pi, spline=None, start_turns=START_TURNS, dt=DT,
                 mu=MU_BOARD, integrator="implicitfast", solver_kwargs=None, col_kwargs=None, asset_dir=None, **sim_kwargs):
    """
    `B` worlds of the driven nut, started on the thread at its phase and turned at `rate` (rad/s; the
    default, one turn a second clockwise, turns it on), at the demo's friction by default.
    """
    from lodos.rbd import io as rbd_io

    nut = nut or default_nut(size)
    z = start_z(size, start_turns)
    turn = thread_phase(size, nut, spline, start_turns, None if asset_dir is None else str(asset_dir))[0]
    mjm = mujoco.MjModel.from_xml_string(driven_xml(size, z, turn, dt, integrator, asset_dir))
    surf = surfaces(mjm, size, nut, spline, mu, col_kwargs, asset_dir)
    s = lsim.make_sim(mjm, B, surfaces=surf, qpos=np.zeros((B, 4)), solver_kwargs=solver_options(solver_kwargs),
                      **sim_kwargs)
    rbd_io.set_state(s.drbd, ctrl=np.full((B, 1), float(rate), dtype=np.float32))
    return s


def _qmul(a, b):
    """Hamilton product, wxyz."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw])


def _rotate(q, v):
    """v rotated by the unit quaternion q, wxyz."""
    w, u = q[0], np.asarray(q[1:])
    t = 2.0 * np.cross(u, v)
    return np.asarray(v) + w * t + np.cross(u, t)


def bolt_pose(size):
    """A bolt's world pose on the fixed board: (position, wxyz quaternion)."""
    (pb, qb), (po, qo) = BOARD_POSE, BOLT_ON_BOARD[size]
    return np.asarray(pb) + _rotate(qb, po), _qmul(qb, qo)


def board_nut_pose(size, nut=None, start_turns=START_TURNS, extra_turn=0.0):
    """A nut started on its bolt's thread on the board, as `start_z` and `thread_phase` place it on the lone bolt."""
    p, q = bolt_pose(size)
    S = SIZES[size]
    turn = thread_phase(size, nut or default_nut(size), None, start_turns)[0] + extra_turn
    z = start_z(size, start_turns) + S.base  # in the bolt's frame
    return p + _rotate(q, (0.0, 0.0, z)), _qmul(q, (np.cos(turn / 2), 0.0, 0.0, np.sin(turn / 2)))


def board_assets_present(nuts=None, asset_dir=None):
    d = Path(asset_dir or ASSET_DIR)
    nuts = nuts or {s: default_nut(s) for s in SIZES}
    return (d / "surfaces/nistboard/surface_data.npz").exists() and all(assets_present(s, nuts[s], d) for s in SIZES)


def board_xml(nuts=None, start_turns=START_TURNS, dt=DT, integrator="implicitfast", asset_dir=None):
    """The board fixed at the demo's pose, the three bolts on it, each nut started on its bolt's thread."""
    d = Path(asset_dir or ASSET_DIR)
    nuts = nuts or {s: default_nut(s) for s in SIZES}
    if not board_assets_present(nuts, d):
        raise FileNotFoundError(f"no NIST board assets in {d}: make them with tools/mesh_processing/nist")
    f = lambda v: " ".join(f"{x:.9g}" for x in v)
    meshes = [f'<mesh name="board" file="{d / "src" / "nistboard_collision.obj"}"/>']
    bodies = [f'<body name="board" pos="{f(BOARD_POSE[0])}" quat="{f(BOARD_POSE[1])}"><geom name="g_board" type="mesh" mesh="board"/></body>']
    for size, S in SIZES.items():
        meshes += [f'<mesh name="{S.bolt}" file="{d / "repair" / f"{S.bolt}.obj"}" inertia="exact"/>',
                   f'<mesh name="{S.nut}" file="{d / "repair" / f"{S.nut}.obj"}" inertia="exact"/>']
        pb, qb = bolt_pose(size)
        pn, qn = board_nut_pose(size, nuts[size], start_turns)
        bodies += [f'<body name="{S.bolt}" pos="{f(pb)}" quat="{f(qb)}"><geom name="g_{S.bolt}" type="mesh" mesh="{S.bolt}"/></body>',
                   f'<body name="{S.nut}" pos="{f(pn)}" quat="{f(qn)}"><freejoint/>'
                   f'<geom name="g_{S.nut}" type="mesh" mesh="{S.nut}" mass="{NUT_MASS}"/></body>']
    mjcf_integrator = {"euler": "Euler", "implicitfast": "implicitfast"}[integrator]
    nl = "\n    "
    return f"""<mujoco model="nist_board">
  <option timestep="{dt}" integrator="{mjcf_integrator}" cone="elliptic"/>
  <asset>
    {nl.join(meshes)}
  </asset>
  <worldbody>
    <geom name="floor" type="plane" pos="0 0 0" size="1 1 0.01"/>
    {nl.join(bodies)}
  </worldbody>
</mujoco>"""


def board_surfaces(mjm, nuts=None, spline=None, mu=MU, col_kwargs=None, asset_dir=None):
    """
    Surfaces 0 the board, 1-3 the bolts, 4-6 the nuts (M8, M12, M16), 7 the floor. Each nut's vertices
    on its own bolt's field (friction `mu`), on the board's (`MU_BOARD`) and on the floor (`MU_GROUND`).
    """
    nuts = nuts or {s: default_nut(s) for s in SIZES}
    gid = lambda n: mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, n)
    sizes = list(SIZES)
    surf = ([load("nistboard", spline, asset_dir)] + [load(SIZES[s].bolt, spline, asset_dir) for s in sizes]
            + [load(nuts[s], spline, asset_dir) for s in sizes] + [sdf_types.PlaneSurface()])
    geoms = ([gid("g_board")] + [gid(f"g_{SIZES[s].bolt}") for s in sizes] + [gid(f"g_{SIZES[s].nut}") for s in sizes]
             + [gid("floor")])
    pairs, mus = [], []
    for k in range(len(sizes)):
        pairs += [[4 + k, 1 + k], [4 + k, 0], [4 + k, 7]]
        mus += [mu, MU_BOARD, MU_GROUND]
    params = lsim.default_pair_params(mu, len(pairs))
    params[:, 0] = mus
    return lsim.Surfaces(surfaces=surf, geoms=geoms, pairs=pairs, params=params, mu=mu, col_kwargs=dict(col_kwargs or {}))


def board_qpos(B, nuts=None, start_turns=START_TURNS, jitter=0.0, seed=0):
    """The three nuts' free joints, each started on its bolt's thread; `jitter` turns each further, per world."""
    nuts = nuts or {s: default_nut(s) for s in SIZES}
    rng = np.random.default_rng(seed)
    q = np.zeros((B, 7 * len(SIZES)))
    for k, size in enumerate(SIZES):
        extra = float(jitter) * rng.uniform(-1.0, 1.0, size=B) if jitter else np.zeros(B)
        for b in range(B):
            p, quat = board_nut_pose(size, nuts[size], start_turns, extra[b])
            q[b, 7 * k: 7 * k + 7] = np.concatenate([p, quat])
    return q


def build_board(B, nuts=None, spline=None, start_turns=START_TURNS, dt=DT, mu=MU, integrator="implicitfast",
                solver_kwargs=None, col_kwargs=None, asset_dir=None, jitter=0.0, **sim_kwargs):
    """`B` worlds of the board with its three nut-bolt pairs, as `build` makes one pair."""
    mjm = mujoco.MjModel.from_xml_string(board_xml(nuts, start_turns, dt, integrator, asset_dir))
    surf = board_surfaces(mjm, nuts, spline, mu, col_kwargs, asset_dir)
    return lsim.make_sim(mjm, B, surfaces=surf, qpos=board_qpos(B, nuts, start_turns, jitter),
                         solver_kwargs=solver_options(solver_kwargs), **sim_kwargs)


def screw_law_acceleration(mjm, size="m8", mu=MU, g=9.81):
    """
    The nut's acceleration down its fixed bolt under Coulomb friction `mu` on one 30 degree flank at
    the thread's mean radius: `viewer.scenes.nutbolt.screw_law_acceleration`'s law, at this size's
    pitch and radius. The coarse collision meshes' flanks are only roughly ISO's 30 degrees.
    """
    S = SIZES[size]
    b = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "nut")
    m = mjm.body_mass[b]
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, mjm.body_iquat[b])
    R = R.reshape(3, 3)
    Ic = R @ np.diag(mjm.body_inertia[b]) @ R.T
    ipos = mjm.body_ipos[b]
    Izz = Ic[2, 2] + m * (ipos[0] ** 2 + ipos[1] ** 2)
    p, r = S.pitch / (2 * np.pi), S.radius
    la = np.arctan(p / r)
    k = r * (np.cos(FLANK) * np.sin(la) - mu * np.cos(la)) / (np.cos(FLANK) * np.cos(la) + mu * np.sin(la))
    return g * (k / p) / (Izz / (m * p * p) + k / p)


# The Franka holding a nut over its bolt on the board (`franka_xml`, `build_franka`): the threading
# demo's scene, with Isaac Lab's NIST demo's arm and gripper (`viewer.scenes.franka`). Its pairs are
# what the threading can touch, each one way: the nut's vertices on the bolt's, the board's and each
# pad's fields, and the bolt's vertices (Isaac Lab's collision mesh's own) on each pad's field, where a
# pad's face meets the thread's crests. The pads and the hand's capsule meet the board through MuJoCo
# Warp, box on box and capsule on box: the board's box is `board_box`, which only the gripper's
# colliders reach. The demo's friction, 0.75, throughout; its substep, 1.25 ms.
FRANKA_DT = 1.0 / 800.0
HOLD_TURNS = START_TURNS - 3  # the held nut's bottom a pitch above its bolt's tip: whole pitches above a phase that fits

# Isaac Lab's NIST demo's contact materials, (stiffness, damping): its board's, bolts' and nuts', and its robot's.
# Newton hands MuJoCo Warp each shape's as a solref, (2 / kd, kd / (2 sqrt(ke))) (its `convert_solref` at unit
# impedance terms), each contact the mean of its two shapes' (MuJoCo's solmix), with MuJoCo's default solimp. The
# Franka's scene carries them, on its Lodos pairs and its MJCF colliders; only the mj mode reads them. MuJoCo
# Warp holds a time constant to two steps at least (refsafe), in the demo as here: 2.5 ms at `FRANKA_DT`.
ISAACLAB_PARTS = (1.6e5, 800.0)
ISAACLAB_ROBOT = (1.0e6, 2000.0)
ISAACLAB_SOLIMP = (0.9, 0.95, 0.001, 0.5, 2.0)


def isaaclab_solref(*materials):
    """The solref Newton gives a contact between shapes of `materials`, `(stiffness, damping)` each; one for a shape's own."""
    return tuple(float(x) for x in np.mean([(2.0 / kd, kd / (2.0 * np.sqrt(ke))) for ke, kd in materials], axis=0))


def franka_assets_present(size="m8", nut=None, asset_dir=None):
    from viewer.scenes import franka as F

    d = Path(asset_dir or ASSET_DIR)
    return (assets_present(size, nut, d) and (d / "surfaces/nistboard/surface_data.npz").exists()
            and (d / "surfaces" / F.PAD / "surface_data.npz").exists() and F.assets_present())


def _obj_bounds(path):
    v = np.array([line.split()[1:4] for line in open(path) if line.startswith("v ")], dtype=float)
    return v.min(0), v.max(0)


# Newton's SDF of a part, as Isaac Lab's demo has Newton bake it (`examples/demos/nist.py`): 256 cells on
# the mesh's longest side, and a band of 5 mm on each side of the surface.
NEWTON_SDF = dict(max_resolution=256, band=0.005)


def _obj(path):
    """A triangle mesh's `(vertices, faces)` from an OBJ of `v` and `f` lines."""
    v, f = [], []
    for line in open(path):
        if line.startswith("v "):
            v.append(line.split()[1:4])
        elif line.startswith("f "):
            idx = [int(x.split("/")[0]) - 1 for x in line.split()[1:]]
            if len(idx) != 3:
                raise ValueError(f"{path}: a face of {len(idx)} vertices; only triangles are read")
            f.append(idx)
    return np.asarray(v, dtype=np.float32), np.asarray(f, dtype=np.int32)


def _mesh_file(mjm, g):
    """The file a mesh geom's mesh was loaded from, or None for any other geom."""
    if int(mjm.geom_type[g]) != mujoco.mjtGeom.mjGEOM_MESH:
        return None
    adr = int(mjm.mesh_pathadr[int(mjm.geom_dataid[g])])
    return None if adr < 0 else Path(bytes(mjm.paths[adr:]).split(b"\x00")[0].decode())


def newton_shapes(mjm, surfaces, asset_dir=None):
    """
    What Newton's sims collide for a NIST scene's Lodos surfaces (`SceneSource.newton_shapes`): the parts
    as Isaac Lab's demo has Newton collide them, not as Lodos's remeshed parts and their bakes. Each nut
    and bolt -- a surface keyed to a part's mesh, `repair/<part>.obj` -- is Isaac Lab's own collision mesh,
    `src/<part>_collision.obj`, with an SDF at the demo's settings (`NEWTON_SDF`); the board, Isaac Lab's
    box, and the Franka's pads are the MJCF's geoms, which Newton collides itself.
    """
    d = Path(asset_dir or ASSET_DIR)
    out = {}
    for s, g in zip(surfaces.surfaces, surfaces.geoms):
        if isinstance(s, sdf_types.PlaneSurface):
            continue
        path = _mesh_file(mjm, int(g))
        if path is not None and path.parent.name == "repair":
            v, f = _obj(d / "src" / f"{path.stem}_collision.obj")
            out[int(g)] = dict(vertices=v, faces=f, sdf=dict(NEWTON_SDF), ke=ISAACLAB_PARTS[0], kd=ISAACLAB_PARTS[1])
        else:
            out[int(g)] = None
    return out


def franka_xml(size="m8", dt=FRANKA_DT, asset_dir=None, franka_dir=None, board=None, bolt=None, fingers=None):
    """
    The board fixed with the `size` bolt on it, its nut free, and the Franka at the origin. `board` and
    `bolt` are world poses, `(position, wxyz quaternion)`: by default the demo's nominal board, and the
    bolt at its offset on it; a recorded episode passes its own, the board having been placed at reset.
    `fingers` is each finger's drive, `(kp, kd)` (`franka.mjcf`).
    """
    from viewer.scenes import franka as F

    d = Path(asset_dir or ASSET_DIR)
    if not franka_assets_present(size, None, d):
        raise FileNotFoundError(f"no NIST Franka assets in {d}: make them with tools/mesh_processing/nist")
    S = SIZES[size]
    f = lambda v: " ".join(f"{x:.9g}" for x in v)
    lo, hi = _obj_bounds(d / "src" / "nistboard_collision.obj")
    board = board or BOARD_POSE
    pb, qb = bolt or bolt_pose(size)
    arm_assets, arm, actuators = F.mjcf(MU_BOARD, franka_dir, solref=isaaclab_solref(ISAACLAB_ROBOT), fingers=fingers)
    nl = "\n    "
    return f"""<mujoco model="nist_franka_{size}">
  <option timestep="{dt}" integrator="implicitfast" cone="elliptic"/>
  <asset>
    <mesh name="board" file="{d / "src" / "nistboard_collision.obj"}"/>
    <mesh name="{S.bolt}" file="{d / "repair" / f"{S.bolt}.obj"}" inertia="exact"/>
    <mesh name="{S.nut}" file="{d / "repair" / f"{S.nut}.obj"}" inertia="exact"/>
    {nl.join(arm_assets)}
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.01" contype="0" conaffinity="0"/>
    <body name="board" pos="{f(board[0])}" quat="{f(board[1])}">
      <geom name="g_board" type="mesh" mesh="board"/>
      <geom name="board_box" type="box" size="{f((hi - lo) / 2)}" pos="{f((hi + lo) / 2)}" contype="0" conaffinity="2"
            friction="{MU_BOARD:g} 0.005 0.0001" solref="{f(isaaclab_solref(ISAACLAB_PARTS))}" group="3"/>
    </body>
    <body name="{S.bolt}" pos="{f(pb)}" quat="{f(qb)}"><geom name="g_{S.bolt}" type="mesh" mesh="{S.bolt}"/></body>
    <body name="{S.nut}"><freejoint/><geom name="g_{S.nut}" type="mesh" mesh="{S.nut}" mass="{NUT_MASS}"/></body>
    {arm}
  </worldbody>
  <actuator>
    {nl.join(actuators)}
  </actuator>
</mujoco>"""


# Isaac Lab's demo as it runs on Newton, from the MJCF its Newton hands MuJoCo Warp
# (`SolverMuJoCo(save_to_mjcf=)`): the contact gaps Newton gives the parts (the demo's `contact_gap`) and the
# board (Newton's default), and what the demo's startup event adds to the held part's inertia's diagonal.
NEWTON_PART_GAP, NEWTON_BOARD_GAP = 0.005, 0.01
NEWTON_NUT_INERTIA = 1e-5


def newton_xml(xml, solver="mjwarp"):
    """
    A Franka scene's MJCF (`franka_xml`) as Newton's sims take it (`SceneSource.newton_xml`): Isaac Lab's
    demo as it runs on Newton, where Lodos's scene departs from it.
    - The gripper's colliders -- the hand's capsule, the pads' boxes -- meet the parts and the board, and
      not each other, where Lodos meets the parts with the pads' fields.
    - The parts and the board have Isaac Lab's material, solref 0.0025 s and friction 0.75, and Newton's
      gaps: `NEWTON_PART_GAP` on the nut's and the bolt's geoms, which their Newton meshes take
      (`newton_shapes`), and `NEWTON_BOARD_GAP` on the board's box, its mesh colliding with nothing; the
      gripper's colliders have none.
    - The joints keep their ranges (`franka.ARM_RANGE`, the fingers 0 to 4 cm).
    - The fingers are the demo's: finger 1 on its drive (kp 350, kd 175, 70 N), finger 2 passive (1 N) and
      held to it by an equality, the demo's mimic; their targets are the demo's binary gripper command
      (`newton_controls`). Kamino takes no equality (`solver="kamino"`): there each finger takes half the
      drive, as Lodos's own Franka scenes have it (`franka.mjcf`).
    - The nut's inertia has `NEWTON_NUT_INERTIA` more on its diagonal, the demo's startup event.
    """
    from viewer.scenes import franka as F

    if solver not in ("mjwarp", "kamino"):
        raise ValueError(f"solver {solver!r}: mjwarp or kamino")
    half = solver == "kamino"
    root = ET.fromstring(xml)
    solref = " ".join(f"{x:.9g}" for x in isaaclab_solref(ISAACLAB_PARTS))
    for g in root.iter("geom"):
        name = g.get("name") or ""
        if name in ("hand_capsule", "left_pad_box", "right_pad_box"):
            g.attrib.update(contype="1", conaffinity="1")
        elif name == "g_board":
            g.attrib.update(contype="0", conaffinity="0")
        elif name == "board_box":
            g.attrib.update(contype="1", conaffinity="1", gap=f"{NEWTON_BOARD_GAP:g}")
        elif name.startswith(("g_nut", "g_bolt")):
            g.attrib.update(solref=solref, friction=f"{MU_BOARD:g} 0.005 0.0001", gap=f"{NEWTON_PART_GAP:g}")
    # The MJCF reads a hinge's range in degrees (MuJoCo's default `compiler angle`), a slide's in metres.
    ranges = dict(zip(F.ARM, np.degrees(F.ARM_RANGE)), **{j: (GRIPPER_CLOSED, GRIPPER_OPEN) for j in F.FINGERS})
    effort = dict(zip(F.FINGERS, (F.FINGER_EFFORT / 2,) * 2 if half else (F.FINGER_EFFORT, 1.0)))
    for j in root.iter("joint"):
        name = j.get("name")
        if name in ranges:
            j.attrib.update(limited="true", range=" ".join(f"{x:.9g}" for x in ranges[name]))
        if name in effort:
            j.set("actuatorfrcrange", f"{-effort[name]:g} {effort[name]:g}")
    kp, kd = F.FINGER_KP, F.FINGER_KD
    gains = dict(zip(F.FINGERS, ((kp / 2, kd / 2),) * 2 if half else ((kp, kd), (0.0, 0.0))))
    for a in root.find("actuator"):
        if a.get("joint") in gains:
            p, v = gains[a.get("joint")]
            a.attrib.update(gainprm=f"{p:g}", biasprm=f"0 {-p:g} {-v:g}")
    if not half:
        eq = root.find("equality")
        if eq is None:
            eq = ET.SubElement(root, "equality")
        ET.SubElement(eq, "joint", {"joint1": F.FINGERS[0], "joint2": F.FINGERS[1]})
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    ET.SubElement(contact, "exclude", {"body1": "left_finger", "body2": "right_finger"})
    # The nut's inertia as MuJoCo compiles it, and the demo's addition, as an explicit `<inertial>`.
    mjm = mujoco.MjModel.from_xml_string(xml)
    f = lambda v: " ".join(f"{x:.9g}" for x in v)  # noqa: E731
    for body in root.iter("body"):
        if body.find("freejoint") is not None and (body.get("name") or "").startswith("nut"):
            k = mjm.body(body.get("name")).id
            body.insert(0, ET.Element("inertial", {
                "pos": f(mjm.body_ipos[k]), "quat": f(mjm.body_iquat[k]), "mass": f"{float(mjm.body_mass[k]):.9g}",
                "diaginertia": f(np.asarray(mjm.body_inertia[k]) + NEWTON_NUT_INERTIA)}))
    return ET.tostring(root, encoding="unicode")


def newton_controls(path=None):
    """
    The recorded threading's targets for Newton's sims (`newton_xml`, `THREADING`): `(ctrl, period)`, the
    arm's as Lodos's, each finger's the demo's binary gripper command, as the demo plays it on Newton.
    """
    with np.load(path or THREADING) as z:
        return np.asarray(z["newton_ctrl"], dtype=np.float64), float(z["period"])


def franka_surfaces(mjm, size="m8", nut=None, spline=None, col_kwargs=None, asset_dir=None):
    """
    Surfaces 0 the board, 1 the bolt, 2 the nut, 3 and 4 the left and right finger pads: the nut's
    vertices on the bolt's, the board's and each pad's fields, the bolt's on each pad's. Their solref
    and solimp, which only the mj mode reads, are Isaac Lab's demo's (`isaaclab_solref`).
    """
    from viewer.scenes import franka as F

    S = SIZES[size]
    gid = lambda n: mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, n)
    surf = [load(n, spline, asset_dir) for n in ("nistboard", S.bolt, nut or default_nut(size), F.PAD, F.PAD)]
    geoms = [gid(n) for n in ("g_board", f"g_{S.bolt}", f"g_{S.nut}", "g_left_pad", "g_right_pad")]
    pairs = [[2, 1], [2, 0], [2, 3], [2, 4], [1, 3], [1, 4]]
    params = lsim.default_pair_params(MU_BOARD, len(pairs))
    for k, (_, s) in enumerate(pairs):  # a part's vertices on a part's field, or on a pad's
        params[k, 5:7] = isaaclab_solref(ISAACLAB_PARTS, ISAACLAB_ROBOT if s in (3, 4) else ISAACLAB_PARTS)
        params[k, 7:12] = ISAACLAB_SOLIMP
    return lsim.Surfaces(surfaces=surf, geoms=geoms, pairs=pairs, params=params, mu=MU_BOARD,
                         col_kwargs=dict(col_kwargs or {}))


def franka_hold(mjm, size="m8", nut=None, spline=None, asset_dir=None):
    """
    The Franka holding the nut a pitch above its bolt's tip, ready to put it on: `(qpos, ctrl)`. The
    nut sits whole pitches above a phase that fits the thread (`thread_phase`). The hand points down,
    its fingers across a pair of the nut's flats, the pair nearest the demo's initial hand, and its
    grasp point at the nut's centre (`franka.ik`, from the demo's initial arm, within the joints'
    ranges); the pads touch the flats. The arm's targets are where it is and the
    gripper's are closed, so the drives hold the arm and squeeze the nut with kp q a finger.
    """
    from viewer.scenes import franka as F

    nut = nut or default_nut(size)
    S = SIZES[size]
    pb, qb = bolt_pose(size)
    turn = thread_phase(size, nut, spline, START_TURNS, None if asset_dir is None else str(asset_dir))[0]
    pn = pb + _rotate(qb, (0.0, 0.0, start_z(size, HOLD_TURNS) + S.base))
    qn = _qmul(qb, (np.cos(turn / 2), 0.0, 0.0, np.sin(turn / 2)))
    Rn = np.zeros(9)
    mujoco.mju_quat2Mat(Rn, qn)
    Rn = Rn.reshape(3, 3)
    v = np.asarray(load(nut, spline, asset_dir).v)
    lo, hi = v.min(0), v.max(0)
    q = np.array(mjm.qpos0, dtype=np.float64)
    qa, _ = F.joint_ids(mjm)
    q[qa[:7]] = F.HOME
    home = F.grasp_pose(mjm, q)[1][:, 1]  # the demo's initial hand's finger axis
    z = np.array([0.0, 0.0, -1.0])
    grips = []
    for k in range(6):  # the nut's x is across a pair of its flats; the other pairs are 60 degrees on
        c, s_ = np.cos(k * np.pi / 3), np.sin(k * np.pi / 3)
        y = Rn @ np.array([c, s_, 0.0])
        y = (y - y[2] * np.array([0.0, 0.0, 1.0])) / np.linalg.norm(y[:2])
        grips.append((-(y @ home), y))
    for _, y in sorted(grips, key=lambda g: g[0]):
        try:
            q = F.ik(mjm, q, pn + Rn @ ((lo + hi) / 2), np.stack([np.cross(y, z), y, z], axis=1))
            break
        except ValueError:
            continue
    else:
        raise ValueError(f"the arm reaches the {size} nut across none of its flats")
    q[qa[7:]] = (hi[0] - lo[0]) / 2 + (F.PAD_HALF[1] - F.PAD_POS[1])  # each pad's inner face on a flat
    a = mjm.jnt_qposadr[mjm.body_jntadr[mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, S.nut)]]
    q[a:a + 7] = np.concatenate([pn, qn])
    return q, np.concatenate([q[qa[:7]], [0.0, 0.0]])


def build_franka(B, size="m8", nut=None, spline=None, dt=FRANKA_DT, solver_kwargs=None, col_kwargs=None,
                 asset_dir=None, franka_dir=None, **sim_kwargs):
    """`B` worlds of the Franka holding the `size` nut over its bolt (`franka_hold`), its drives holding it there."""
    from lodos.rbd import io as rbd_io

    mjm = mujoco.MjModel.from_xml_string(franka_xml(size, dt, asset_dir, franka_dir))
    surf = franka_surfaces(mjm, size, nut, spline, col_kwargs, asset_dir)
    q, ctrl = franka_hold(mjm, size, nut, spline, asset_dir)
    s = lsim.make_sim(mjm, B, surfaces=surf, qpos=np.tile(q, (B, 1)), solver_kwargs=solver_options(solver_kwargs),
                      **sim_kwargs)
    rbd_io.set_state(s.drbd, ctrl=np.tile(ctrl, (B, 1)).astype(np.float32))
    return s


# Isaac Lab's NIST demo's action terms (`examples/demos/nist.py`, `Actions`): the arm's targets are the
# joints' current positions plus the action times `ACTION_SCALE`, refreshed at every physics tick; the
# gripper's are 0, closed, where its action is negative, else 0.04 m, open, on both fingers.
ACTION_SCALE = (0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.2)
ACTION_TICK = 0.01  # the demo's physics tick, which refreshes the targets
ACTION_TICKS = 4  # its ticks a decision
GRIPPER_CLOSED, GRIPPER_OPEN = 0.0, 0.04  # m, each finger's target under the demo's binary gripper command


class ActionReplay:
    """
    Recorded actions, `(n, 8)`, applied as Isaac Lab's demo applies them, one a decision from time 0:
    a `SceneSource.controls` for the viewer. After the last, the arm holds where it is and the gripper
    keeps its last command. `period` is the tick, and the decision is counted in rounded ticks, so that
    time accumulated in float32 cannot carry a tick into the next decision early or late.
    """

    period = ACTION_TICK

    def __init__(self, mjm, actions):
        from viewer.scenes import franka as F

        self.actions = np.asarray(actions, dtype=np.float64)
        self.qa = F.joint_ids(mjm)[0]
        self.open = F.FINGER_OPEN

    def __call__(self, t, qpos):
        k = int(round(t / ACTION_TICK)) // ACTION_TICKS
        a = self.actions[min(k, len(self.actions) - 1)]
        arm = a[:7] * np.asarray(ACTION_SCALE) if k < len(self.actions) else np.zeros(7)
        grip = 0.0 if a[7] < 0 else self.open
        return np.concatenate([np.asarray(qpos)[self.qa[:7]] + arm, [grip, grip]])


# Isaac Lab's NIST policy threading the M8 nut onto its bolt, replayed (`threading`): an episode of the
# demo (Isaac Lab d67292f, seed 1701, one world, its policy `demo-alphaNIST.pt`), decisions 280 to 435:
# 6.2 s from the nut in the hand above the bolt to seated. `THREADING` holds the
# board's and the bolt's poses, the state at decision 280 and the drives' targets at each 10 ms tick
# (made from the episode): the arm's are Isaac Lab's recorded joint
# positions, interpolated within a decision, plus the action, as its own targets were; the fingers' are
# their recorded positions, which drives stiffer than the demo's (`THREADING_FINGERS`) follow within
# 0.5 mm, since our two half drives cannot hold the gripper half open as Isaac Lab's mimic does.
THREADING = Path(__file__).resolve().parent / "data" / "nist_threading_m8.npz"
# The same replay from an easier start: at decision 280 the nut is one pitch onto the thread; here it
# starts at rest 3.4 pitches onto it, on the axis, where it had screwed itself down with the drives held at
# the recording's first targets; the robot starts as recorded,
# and the recording plays from its start.
THREADING_EASY = Path(__file__).resolve().parent / "data" / "nist_threading_m8_easy.npz"
THREADING_FINGERS = (2000.0, 30.0)  # each finger's drive, kp and kd
# The threading replays' settings, per mode (`SceneSource.mode_defaults`): the exact mode at 2 outer
# proximal iterations, the mj mode at MuJoCo's tolerance 1e-5.
THREADING_OPTIONS = {"exact": dict(prox_iters_per_qp=2), "mj": dict(tolerance=1e-5)}


def threading(path=THREADING, start=None, asset_dir=None, franka_dir=None):
    """
    The recorded threading (`THREADING`): `(mjm, xml, qpos, qvel, ctrl, period)`, the Franka scene with
    the recording's board and bolt and its fingers on `THREADING_FINGERS`, its state at the window's
    start, and the drives' targets, `(T, nu)`, a row each `period` s. `start` is a file of another state
    to start from, by the same names (`THREADING_EASY`).
    """
    from viewer.scenes import franka as F

    z = dict(np.load(path))
    if start is not None:
        z.update({k: v for k, v in np.load(start).items() if k in ("joint_pos", "joint_vel", "nut_qpos", "nut_qvel")})
    pose = lambda p: (np.asarray(p[:3], np.float64), np.asarray(p[3:], np.float64))
    xml = franka_xml("m8", asset_dir=asset_dir, franka_dir=franka_dir, board=pose(z["board_pose"]),
                     bolt=pose(z["bolt_pose"]), fingers=THREADING_FINGERS)
    mjm = mujoco.MjModel.from_xml_string(xml)
    qa, da = F.joint_ids(mjm)
    j = mjm.body_jntadr[mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, SIZES["m8"].nut)]
    q, v = np.array(mjm.qpos0, dtype=np.float64), np.zeros(mjm.nv)
    q[qa], v[da] = z["joint_pos"], z["joint_vel"]
    q[mjm.jnt_qposadr[j]:mjm.jnt_qposadr[j] + 7] = z["nut_qpos"]
    v[mjm.jnt_dofadr[j]:mjm.jnt_dofadr[j] + 6] = z["nut_qvel"]
    return mjm, xml, q, v, np.asarray(z["ctrl"], dtype=np.float64), float(z["period"])
