"""
Newton's nut-bolt example (`newton/examples/contacts/example_nut_bolt_hydro.py`) on Lodos
surfaces: IsaacGymEnvs' M20 loose nut and bolt, the nut dropped onto the thread 41 mm above
its seat and turned pi/8, as Newton starts it, to screw itself down under gravity.

The scene, for every solver:

    surfaces   0 the bolt, 1 the nut, 2 the floor (a plane at z = 0)
    pairs      the nut's vertices on the bolt's SDF; the bolt's, and the nut's, on the floor
    friction   0.01 on the thread, as Newton's; 0.5 on the floor, which holds the bolt
    bodies     both free, density 8000; MuJoCo takes inertia from the *exact* mesh volume,
               not the convex hull its default uses (the nut: 63.6 g, not 107.6 g)
    dt         1/480, Newton's hydro example's substep

The assets ship in `viewer/assets/nutbolt` (`ASSET_DIR`; `$LODOS_NUTBOLT_DIR` overrides it):
`src/` IsaacGymEnvs' meshes as downloaded (BSD-3-Clause, NVIDIA; see its LICENSE),
`repair/` the watertight ones MuJoCo takes inertia from, `surfaces/` the bolt's and the
0.5 and 1 mm nuts' surfaces. `tools/mesh_processing/nutbolt` makes them, and the nut at the other
edge lengths of `NUTS` (0.2, 0.3, 0.75 mm), which do not ship; the bolt's vertices are 1 mm
apart, since they only meet the floor.

**What to expect.** With Coulomb friction mu on a 30 degree flank, the screw law gives the
nut a constant acceleration down the thread, 6.64 mm/s^2 at mu = 0.01 and 9.114 frictionless
(`screw_law_acceleration`).
"""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np

from viewer.scenes import assets
from lodos import sim as lsim
from lodos.col import sdf
from lodos.col.sdf import types as sdf_types

ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = Path(os.environ.get("LODOS_NUTBOLT_DIR", assets.ASSET_DIR / "nutbolt"))

BOLT_MESH = "repair/factory_bolt_m20_loose_wt.obj"
NUT_MESH = "repair/factory_nut_m20_loose_subdiv_3x_wt.obj"
NUTS = ("nut_s0.2", "nut_s0.3", "nut_s0.5", "nut_s0.75", "nut_s1.0")  # the nut remeshed at 0.2 ... 1 mm
NUT = "nut_s0.5"

DT = 1.0 / 480.0
MU = 0.01
MU_FLOOR = 0.5
DENSITY = 8000.0
NUT_Z = 0.041  # above the files' assembled pose, as Newton places it
NUT_TURN = np.pi / 8

# The scene's binning when its contacts are grouped (`lodos.cd.groups`): 6 normal bins x 4
# moment sectors held a seated nut.
GROUP_BINS = dict(normal_bins=6, moment_bins=4)

PITCH = 2.5e-3  # the thread's lead, one start
FLANK = np.radians(30.0)  # ISO metric: half the 60 degree thread angle
FLANK_RADIUS = 9.34e-3  # the middle of the flank contact band, 8.87-9.81 mm


def assets_present(asset_dir=None):
    d = Path(asset_dir or ASSET_DIR)
    return (d / BOLT_MESH).exists() and (d / "surfaces" / "bolt" / "surface_data.npz").exists()


def _require(asset_dir):
    if not assets_present(asset_dir):
        raise FileNotFoundError(
            f"no nut-bolt assets in {asset_dir}: make them with tools/mesh_processing/nutbolt, "
            f"or point LODOS_NUTBOLT_DIR at them"
        )


def scene_xml(nut_z=NUT_Z, nut_turn=NUT_TURN, dt=DT, asset_dir=None):
    d = Path(asset_dir or ASSET_DIR)
    _require(d)
    q = (np.cos(nut_turn / 2), 0.0, 0.0, np.sin(nut_turn / 2))
    return f"""<mujoco model="nutbolt">
  <option timestep="{dt}" integrator="Euler" cone="elliptic"><flag eulerdamp="disable"/></option>
  <asset>
    <mesh name="bolt" file="{d / BOLT_MESH}" inertia="exact"/>
    <mesh name="nut" file="{d / NUT_MESH}" inertia="exact"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" pos="0 0 0" size="0.5 0.5 0.01"/>
    <body name="bolt" pos="0 0 0"><freejoint/><geom name="g_bolt" type="mesh" mesh="bolt" density="{DENSITY}"/></body>
    <body name="nut" pos="0 0 {nut_z}" quat="{q[0]} 0 0 {q[3]}"><freejoint/><geom name="g_nut" type="mesh" mesh="nut" density="{DENSITY}"/></body>
  </worldbody>
</mujoco>"""


def load(name, spline=None, asset_dir=None):
    """A bake under `surfaces/`, evaluated tricubically or, `spline="linear"`, trilinearly."""
    d = Path(asset_dir or ASSET_DIR)
    s = sdf.load_surface_data(str(d / "surfaces" / name / "surface_data.npz"), spline_type=spline)
    s.sph_c = np.zeros((0, 3))
    s.sph_r = np.zeros(0)
    return s


def surfaces(mjm, nut=NUT, spline=None, two_way=False, mu=MU, col_kwargs=None, asset_dir=None):
    """
    The Lodos half: the nut's vertices on the bolt's SDF, both bodies on the floor, and with
    `two_way` the bolt's vertices on the nut's SDF too.
    """
    gid = lambda n: mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, n)
    pairs = [[1, 0], [0, 2], [1, 2]] + ([[0, 1]] if two_way else [])
    params = lsim.default_pair_params(mu, len(pairs))
    params[1, 0] = MU_FLOOR
    params[2, 0] = MU_FLOOR
    return lsim.Surfaces(
        surfaces=[load("bolt", spline, asset_dir), load(nut, spline, asset_dir), sdf_types.PlaneSurface()],
        geoms=[gid("g_bolt"), gid("g_nut"), gid("floor")],
        pairs=pairs, params=params, mu=mu, col_kwargs=dict(col_kwargs or {}),
    )


def initial_qpos(B, nut_z=NUT_Z, nut_turn=NUT_TURN, jitter=0.0, seed=0):
    """
    Newton's start in every world. `jitter`: each world's nut turned further by an angle
    drawn uniformly from [-jitter, jitter] radians (seeded; the first worlds of a batch
    draw what a smaller batch draws). Where the nut wedges on its seat -- and so how many
    contacts and variables it keeps there, and what its steps cost -- is chaotic in its
    arrival, so identical worlds replicate one outcome; 0.05 rad shifts the nut's thread
    by 0.02 mm along the bolt's, within the flank clearance.
    """
    turn = np.full(B, float(nut_turn))
    if jitter:
        turn += float(jitter) * np.random.default_rng(seed).uniform(-1.0, 1.0, size=B)
    q = np.zeros((B, 14))
    q[:, 3] = 1.0
    q[:, 9] = nut_z
    q[:, 10] = np.cos(turn / 2)
    q[:, 13] = np.sin(turn / 2)
    return q


def build(B, nut=NUT, spline=None, two_way=False, nut_z=NUT_Z, nut_turn=NUT_TURN, dt=DT, mu=MU,
          solver_kwargs=None, col_kwargs=None, asset_dir=None, jitter=0.0, **sim_kwargs):
    """
    `B` worlds of the scene, as a `lodos.sim.Sim`.

    jitter: the spread of the nut's start turn across worlds (`initial_qpos`); 0 starts
        every world alike.
    solver_kwargs: the solver's options.
    sim_kwargs: `lodos.sim.make_sim`'s, e.g. `contact_reduction=True` and MJWarp's capacities,
        which must hold Lodos's contacts, ~2300 per world mid-thread unreduced.
    """
    mjm = mujoco.MjModel.from_xml_string(scene_xml(nut_z, nut_turn, dt, asset_dir))
    surf = surfaces(mjm, nut, spline, two_way, mu, col_kwargs, asset_dir)
    return lsim.make_sim(mjm, B, surfaces=surf, qpos=initial_qpos(B, nut_z, nut_turn, jitter),
                         solver_kwargs=dict(solver_kwargs or {}), **sim_kwargs)


def screw_law_acceleration(mjm, mu=MU, r=FLANK_RADIUS, g=9.81):
    """
    The nut's acceleration down a fixed bolt under Coulomb friction `mu` on one flank, m/s^2.

    Along the helix z = -p theta, p = PITCH / 2 pi, with lead angle tan(la) = p / r: the flank
    normal's axial and tangential parts carry the weight and the spin, friction opposes the
    slip, and eliminating the normal force gives a = g (k/p) / (I/(m p^2) + k/p) with
    k = r (cos(al) sin(la) - mu cos(la)) / (cos(al) cos(la) + mu sin(la)). At mu = 0, k = p and
    a = g m / (m + I (2 pi / P)^2). `I` is the nut's inertia about the bolt axis, from `mjm`.
    """
    b = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "nut")
    m = mjm.body_mass[b]
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, mjm.body_iquat[b])
    R = R.reshape(3, 3)
    Ic = R @ np.diag(mjm.body_inertia[b]) @ R.T
    ipos = mjm.body_ipos[b]
    Izz = Ic[2, 2] + m * (ipos[0] ** 2 + ipos[1] ** 2)
    p = PITCH / (2 * np.pi)
    la = np.arctan(p / r)
    k = r * (np.cos(FLANK) * np.sin(la) - mu * np.cos(la)) / (np.cos(FLANK) * np.cos(la) + mu * np.sin(la))
    return g * (k / p) / (Izz / (m * p * p) + k / p)
