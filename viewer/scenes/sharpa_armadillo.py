"""
The SHARPA hand holding a 70 mm armadillo, palm up: the viewer's `sharpa_armadillo_*` scenes,
and the worked example of `docs/creating_new_sim_from_mjcf.pdf`.

Ported from Lodos-v1's in-hand reorientation task (`papers/lodos_v1.pdf`, Sections
V-B.2 and V-C.2): its MJCF, the original, collides both halves as convex decompositions
under MuJoCo, and Lodos-v1 ran the same scene on Lodos surfaces. Here both are one scene
in the two representations every built-in scene has:

- `rep="polytopes"` (the viewer's "polytope"): the hand's 87 CoACD hulls and the
  armadillo's 34 pieces as MJCF meshes, through MJWarp's pipeline and its exact hull
  narrowphase.
- `rep="surfaces"` (the viewer's "sdf"): the hand's 23 link surfaces and the armadillo's
  surface (`viewer/assets/surfaces/armadillo`, the canonical 1.64 m bake scaled to 70 mm,
  `lodos.col.sdf.scale_surface`), as Lodos-v1 collided them: 23 hand-armadillo
  pairs with the armadillo on the SDF side, the hand's own pairs as MuJoCo would collide
  them (`sharpa.collides`), the more densely sampled link supplying the vertices, and the
  armadillo against the floor.

The hand is the menagerie's SHARPA Wave right hand with its own servo gains (the dropped
hands of `viewer.scenes.sharpa` soften theirs; a grasp needs them as shipped), hung palm up
from a wrist of three world-aligned hinges, each with a position servo. The armadillo
starts where the original's home keyframe holds it, and the scene's control is that
keyframe's: fingers closed on the object, wrist level (`KEY_CTRL`; the viewer applies it
through `SceneSource.ctrl`). The original's goal marker (a mocap body) is dropped.

What the port changes, and why:
- The step is Lodos's, 2 ms explicit Euler on the elliptic cone, not the original MJCF's
  10 ms implicit step on the pyramidal cone (an MPC model's): Lodos-v1 measured the grasp
  going NaN at 10 ms and the object sinking 22 mm in 2 s at 5 ms, at the spring-damper's
  default stiffness on a 49 g object.
- Friction is `MU` = 0.6 on every pair of both representations, Lodos-v1's, where
  the original MJCF left MuJoCo's default 1 on its hulls.
- Joint limits and friction loss are removed unless `jlimit`, as in the other SHARPA scenes.

The armadillo's frame is the original's: the drop scenes' convention (`drop.PIECES_*`,
which places the 34 pieces on the 1.64 m surface) scaled by `ARM_SCALE` and moved by
`ARM_SHIFT`, which reproduces the original's piece transform exactly, so its keyframe
pose holds the object where it held it. Both representations take the armadillo's mass
and inertia from the 34 pieces at the original's density (`ARM_DENSITY`), 49 g, as an
explicit `<inertial>`.
"""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from lodos import sim as lsim
from lodos.col.sdf import scale_surface
from viewer.scenes.assets import load_surface, surface_obj
from lodos.col.sdf import types as sdf_types
from viewer.scenes import drop
from viewer.scenes import sharpa as S

REPS = S.REPS
MU = 0.6
# The original's piece transform: mesh scale and geom position (its quarter turn about x is
# `drop.PIECES_ROTATION`'s).
_ORIG_PIECE_SCALE = 0.0705458
_ORIG_PIECE_POS = np.array([0.00110425, 0.000111763, 0.00512494])
ARM_SCALE = _ORIG_PIECE_SCALE / drop.PIECES_SCALE
ARM_SHIFT = _ORIG_PIECE_POS - ARM_SCALE * drop.PIECES_OFFSET
ARM_DENSITY = 1900.0
ARM_POS = (-0.095, 0.010, 0.065)
FLOOR_Z = -0.30
PALM_QUAT = "0.707106781 0 -0.707106781 0"
WRIST = dict(range=0.6, damping=3.0, armature=0.02, kp=60.0, kv=3.0)
# The original's home keyframe: the wrist (3), the fingers (22), the armadillo (7); and
# its control: the fingers' targets (22), the wrist's (3).
KEY_QPOS = np.array([
    -0.00052085792, -0.0140711415, 2.35327073e-08, 1.57432433, 0.173502499, 1.09271047, 0.226651552,
    1.74588342, 0.779616418, 0.0350511493, 0.962124197, 0.768820018, 0.758195653, 0.0353584908,
    0.957776598, 0.767380363, 0.779902708, 0.0323883221, 0.964849559, 0.77021838, 0.106867171,
    0.778506295, 0.0392333024, 0.956427752, 0.765997256, -0.084111054, -0.0122296219, 0.0526014456,
    0.93145083, -0.0186025936, -0.0591484563, -0.358545331])
KEY_CTRL = np.array([
    1.57874615, 0.187044076, 1.09552892, 0.230556078, 1.68343278, 0.782242457, 0.0349841442,
    0.961596879, 0.768392938, 0.764960045, 0.0353961443, 0.957994654, 0.767266427, 0.782299647,
    0.0316072207, 0.962946402, 0.76918239, 0.115718102, 0.781613986, 0.0394049438, 0.958035842,
    0.766768265, 0.0, 0.0, 0.0])


def armadillo_pieces():
    """The 34 pieces in the armadillo's frame: `[(nv_i, 3) verts]`."""
    pieces, _ = drop.load_armadillo_pieces()
    return [ARM_SCALE * v + ARM_SHIFT for v, _ in pieces]


def armadillo_surface():
    """The armadillo's surface in its body frame: the 1.64 m bake scaled to 70 mm."""
    return scale_surface(load_surface("armadillo"), ARM_SCALE, ARM_SHIFT)


def _vertex_mesh(name, v):
    return ET.Element("mesh", {"name": name, "vertex": " ".join(f"{x:.8f}" for x in np.asarray(v).reshape(-1))})


def armadillo_inertial():
    """`(mass, pos, quat, diaginertia)` of the 34 pieces at `ARM_DENSITY`, as MuJoCo compiles them."""
    meshes = "".join(ET.tostring(_vertex_mesh(f"p{k}", v), encoding="unicode") for k, v in enumerate(armadillo_pieces()))
    geoms = "".join(f'<geom type="mesh" mesh="p{k}" density="{ARM_DENSITY}"/>' for k in range(34))
    m = mujoco.MjModel.from_xml_string(
        f'<mujoco><asset>{meshes}</asset><worldbody><body name="a"><freejoint/>{geoms}</body></worldbody></mujoco>')
    return float(m.body_mass[1]), m.body_ipos[1].copy(), m.body_iquat[1].copy(), m.body_inertia[1].copy()


def scene_xml(rep, jlimit=False, asset_dir=None):
    """The MJCF of the hand on its wrist and the armadillo, in `rep`; see the module docstring."""
    if rep not in REPS:
        raise ValueError(f"rep must be one of {REPS}")
    d = Path(asset_dir or S.ASSET_DIR)
    S._require(d)
    root = ET.parse(d / "menagerie" / "right_hand.xml").getroot()
    root.set("model", "sharpa_armadillo")
    root.find("compiler").set("meshdir", str(d / "menagerie" / "assets" / "right"))
    opt = root.find("option")
    opt.attrib.clear()
    opt.attrib.update(S.OPTION)
    if not jlimit:
        for joint in root.iter("joint"):
            joint.set("limited", "false")
            joint.set("frictionloss", "0")
    for tag in ("keyframe", "sensor"):
        for el in root.findall(tag):
            root.remove(el)

    asset, world = root.find("asset"), root.find("worldbody")
    palm = world.find("body")
    for body in palm.iter("body"):
        name = body.get("name")
        for g in [g for g in body.findall("geom") if S._is_collision(g)]:
            body.remove(g)
        if rep == "polytopes":
            cls = "elastomer_collision" if name in S.SOFT else "collision"
            for i, path in enumerate(S._hulls(d, name)):
                ET.SubElement(asset, "mesh", {"name": f"{name}_cp{i}", "file": str(path)})
                ET.SubElement(body, "geom", {"name": f"{name}_cp{i}", "class": cls, "mesh": f"{name}_cp{i}",
                                             "friction": f"{MU}"})
        else:
            ET.SubElement(body, "geom", {"name": f"{name}_sdf", "type": "sphere", "size": "0.001",
                                         "contype": "0", "conaffinity": "0", "group": "5", "mass": "0"})
    used = {g.get("mesh") for g in root.iter("geom")} - {None}
    for mesh in [m for m in asset.findall("mesh") if m.get("name") not in used]:
        asset.remove(mesh)

    # The hand hangs palm up from three world-aligned hinges, each with a position servo.
    world.remove(palm)
    palm.set("quat", PALM_QUAT)
    palm.attrib.pop("pos", None)
    wrist = ET.Element("body", {"name": "wrist"})
    lim = "true" if jlimit else "false"
    for axis, vec in (("x", "1 0 0"), ("y", "0 1 0"), ("z", "0 0 1")):
        ET.SubElement(wrist, "joint", {"name": f"wrist_{axis}", "type": "hinge", "axis": vec, "limited": lim,
                                       "range": f"{-WRIST['range']} {WRIST['range']}",
                                       "damping": f"{WRIST['damping']}", "armature": f"{WRIST['armature']}"})
    wrist.append(palm)
    world.insert(0, wrist)

    # The armadillo: its surface's mesh (drawn, and keyed to the surface) and, as polytopes,
    # its 34 pieces; its inertia from the pieces in both.
    mass, ipos, iquat, inertia = armadillo_inertial()
    asset.append(ET.Element("mesh", {"name": "armadillo", "file": str(surface_obj("armadillo")),
                                     "scale": f"{ARM_SCALE:.9f} {ARM_SCALE:.9f} {ARM_SCALE:.9f}"}))
    arm = ET.Element("body", {"name": "armadillo", "pos": " ".join(f"{x}" for x in ARM_POS)})
    ET.SubElement(arm, "freejoint", {"name": "armadillo_root"})
    ET.SubElement(arm, "inertial", {"pos": " ".join(f"{x:.9g}" for x in ipos), "quat": " ".join(f"{x:.9g}" for x in iquat),
                                    "mass": f"{mass:.9g}", "diaginertia": " ".join(f"{x:.9g}" for x in inertia)})
    ET.SubElement(arm, "geom", {"name": "armadillo_surf", "type": "mesh", "mesh": "armadillo",
                                "pos": " ".join(f"{x:.9g}" for x in ARM_SHIFT), "contype": "0", "conaffinity": "0",
                                "group": "1" if rep == "polytopes" else "0", "mass": "0", "rgba": "0 0.7 1 1"})
    if rep == "polytopes":
        # In MuJoCo's collision group, as the hand's hulls are: MuJoCo's viewer shows the
        # armadillo's mesh instead, and the live viewer draws the pieces in the body's colour.
        for k, v in enumerate(armadillo_pieces()):
            asset.append(_vertex_mesh(f"arm_cp{k}", v))
            ET.SubElement(arm, "geom", {"name": f"armadillo_cp{k}", "type": "mesh", "mesh": f"arm_cp{k}", "mass": "0",
                                        "conaffinity": "2", "friction": f"{MU}", "group": "3",
                                        "rgba": "0.2 0.6 0.2 0.3"})
    world.append(arm)
    # The floor collides with the armadillo (contype 2 against its conaffinity 2), never
    # with the hand, as in the original.
    world.insert(1, ET.fromstring(f'<geom name="floor" type="plane" pos="0 0 {FLOOR_Z}" size="2 2 0.05" '
                                  f'contype="2" conaffinity="2" friction="{MU}"/>'))

    acts = root.find("actuator")
    for axis in ("x", "y", "z"):
        ET.SubElement(acts, "position", {"name": f"wrist_{axis}_act", "joint": f"wrist_{axis}",
                                         "kp": f"{WRIST['kp']}", "kv": f"{WRIST['kv']}",
                                         "ctrlrange": f"{-WRIST['range']} {WRIST['range']}"})
    key = ET.SubElement(root, "keyframe")
    ET.SubElement(key, "key", {"name": "home", "qpos": " ".join(f"{x:.9g}" for x in KEY_QPOS),
                               "ctrl": " ".join(f"{x:.9g}" for x in KEY_CTRL)})
    return ET.tostring(root, encoding="unicode")


def surfaces(mjm, asset_dir=None, col_kwargs=None):
    """The Lodos half of the `"surfaces"` representation; see the module docstring."""
    d = Path(asset_dir or S.ASSET_DIR)
    hand = S._surfaces(d)
    names = list(hand)
    dens = {n: S._density(s) for n, s in hand.items()}
    body = lambda n: mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
    geom = lambda n: mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, n)  # noqa: E731

    surfs = [hand[n] for n in names] + [armadillo_surface(), sdf_types.PlaneSurface()]
    geoms = [geom(f"{n}_sdf") for n in names] + [geom("armadillo_surf"), geom("floor")]
    arm, floor = len(names), len(names) + 1
    cprs = []
    for i, a in enumerate(names):
        for j in range(i + 1, len(names)):
            b = names[j]
            if S.collides(mjm, body(a), body(b)):
                cprs.append([i, j] if dens[a] >= dens[b] else [j, i])
    cprs += [[i, arm] for i in range(len(names))]
    cprs.append([arm, floor])
    return lsim.Surfaces(surfaces=surfs, geoms=geoms, pairs=cprs, params=lsim.default_pair_params(MU, len(cprs)),
                         mu=MU, col_kwargs=dict(col_kwargs or {}))


def build(rep, B, jlimit=False, solver_kwargs=None, col_kwargs=None, asset_dir=None, **sim_kwargs):
    """The scene as a `lodos.sim.Sim` of `B` worlds, each at the home keyframe with its control."""
    from lodos.rbd import io as rbd_io

    mjm = mujoco.MjModel.from_xml_string(scene_xml(rep, jlimit, asset_dir))
    surf = surfaces(mjm, asset_dir, col_kwargs) if rep == "surfaces" else None
    s = lsim.make_sim(mjm, B, surfaces=surf, qpos=np.tile(KEY_QPOS, (B, 1)), solver_kwargs=solver_kwargs,
                      **sim_kwargs)
    rbd_io.set_state(s.drbd, ctrl=np.tile(KEY_CTRL, (B, 1)))
    return s
