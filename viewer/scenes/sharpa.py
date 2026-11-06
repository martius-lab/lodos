"""
SHARPA hands dropped on a floor or into a walled box: the viewer's `sharpa_*` scenes.

The hand is the SHARPA Wave right hand -- MuJoCo Menagerie's model (Apache-2.0), with the
link surfaces and convex decomposition of Lodos-v1's experiments (`papers/lodos_v1.pdf`:
3,479 vertices on 23 links, and 87 hulls for MJWarp) -- its palm freed by a free joint and its
fingers held open by its own position servos at `ctrl = 0`.
`n_hands` copies are stacked one above another at random orientations, intersection-free
(`viewer.scenes.drop.initial_qpos`), and dropped: five in one stack, twenty in four (`columns`),
`HAND_SPACING` apart, where no two hands of different stacks can meet.

The servos are softened to `KP_SCALE` of the menagerie's stiffness, so that a drop bends
the fingers visibly. As shipped (kp 0.9-13 N m/rad), they hold a five-hand pile within 0.05
rad of open. At a tenth, its fingers rest bent by up to 0.56 rad and spring back open after
an impact, under MuJoCo's own step. Their damping ratio, the 0.9 the menagerie gives
as a positive velocity bias, is kept: MuJoCo turns it into kv from the scaled kp. Without
joint limits (`jlimit=False`) a soft finger also bends
backwards past its range: up to 0.74 rad at a tenth in that pile, 2.3 rad at a twentieth.

Either collision representation, from the same bodies and the same stacks:

- `rep="surfaces"` (the viewer's "sdf"): each of a hand's 23 links is a Lodos surface,
  already in its link's frame (`final_sdata`). Within a hand the pairs are the ones MuJoCo
  would collide -- no parent and child, none the model's `<exclude>`s name -- and between
  hands every pair of links; each link meets the floor and the walls. A pair is
  one-directional, the link whose surface is sampled more densely supplying the vertices:
  the rule of Lodos-v1's SHARPA scenes, where the fingertips sample their surface twice
  as finely as the palm.
- `rep="polytopes"` (the viewer's "polytope"): the 87-hull CoACD decomposition
  (`right_cp`) as MJCF meshes in place of the menagerie's collision geoms, through
  MJWarp's pipeline, which applies the same filter.

Joint constraints (`jlimit=True`, every one of the viewer's SHARPA scenes): the hand's joint
limits and friction loss stay in the model, and the solve has them as MuJoCo's soft rows
(`lodos.sim.make_sim`). By default both are removed.

The walls are a box sized to the hands' stacks (`wall_half`), half-width `WALL_HALF` around
one. Friction is MuJoCo's default, 1, which the menagerie's collision geoms keep; the Lodos
pairs take the same. The assets ship in `viewer/assets/sharpa` (`ASSET_DIR`; `$LODOS_SHARPA_DIR`
overrides it): `menagerie/` (MuJoCo Menagerie's Sharpa Wave, Apache-2.0: `right_hand.xml`, its
licence and `assets/right/*.STL`), and Lodos-v1's hulls and surfaces of its links,
`right_cp/<link>/hull_*.obj` and `final_sdata/<link>.npz`.
"""

from __future__ import annotations

import copy
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from viewer.scenes import assets
from lodos import sim as lsim
from lodos.col.sdf import load_surface_file
from lodos.col.sdf import types as sdf_types
from viewer.scenes import drop
from viewer.scenes import representations as R

ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = Path(os.environ.get("LODOS_SHARPA_DIR", assets.ASSET_DIR / "sharpa"))
REPS = ("surfaces", "polytopes")
SOFT = tuple(f"right_{f}_DP" for f in ("thumb", "index", "middle", "ring", "pinky"))  # elastomer tips
MU = 1.0
DT = 0.002
GAP, JITTER = 0.05, 0.03  # m: a hand is ~20 cm, against the armadillo's ~1.4 m
WALL_HALF, WALL_H = 0.25, 0.4
# m, between stacks: a hand reaches 0.21 m from its palm's origin (the Shadow hand 0.20), so two stacks
# HAND_SPACING apart, each hand jittered by JITTER, cannot meet.
HAND_SPACING = 0.5
KP_SCALE = 0.1  # the finger servos' stiffness, as a fraction of the menagerie's
OPTION = {"timestep": f"{DT}", "integrator": "Euler", "cone": "elliptic", "impratio": "10", "solver": "Newton",
          "iterations": "100", "ls_iterations": "50"}


def assets_present(asset_dir=None):
    d = Path(asset_dir or ASSET_DIR)
    return all((d / p).exists() for p in ("menagerie/right_hand.xml", "right_cp", "final_sdata"))


def _require(d):
    if not assets_present(d):
        raise FileNotFoundError(f"no SHARPA assets in {d}: they ship in viewer/assets/sharpa; "
                                f"point LODOS_SHARPA_DIR at them if they are elsewhere")


def columns(n_hands):
    """The stacks `n_hands` hands are dropped in: one for five, four for twenty, as the armadillos'."""
    return R.default_columns(n_hands)


def wall_half(n_hands):
    """The walled box's inner half-width around `n_hands` hands' stacks."""
    side = int(np.ceil(np.sqrt(columns(n_hands))))
    return WALL_HALF + (side - 1) / 2 * HAND_SPACING


def _is_collision(geom):
    return geom.get("class") in ("collision", "elastomer_collision") or geom.get("group") == "3"


def _hulls(d, link):
    return sorted((d / "right_cp" / link).glob("hull_*.obj"))


def links(asset_dir=None):
    """The hand's 23 links, in the menagerie's order (the palm first)."""
    d = Path(asset_dir or ASSET_DIR)
    _require(d)
    palm = ET.parse(d / "menagerie" / "right_hand.xml").getroot().find("worldbody").find("body")
    return [b.get("name") for b in palm.iter("body")]


def scene_xml(n_hands, rep, walls=False, jlimit=False, asset_dir=None, kp_scale=KP_SCALE):
    """The MJCF of `n_hands` hands, each body, joint, geom, site and actuator named `h<k>_...`."""
    if rep not in REPS:
        raise ValueError(f"rep must be one of {REPS}")
    d = Path(asset_dir or ASSET_DIR)
    _require(d)
    root = ET.parse(d / "menagerie" / "right_hand.xml").getroot()
    root.set("model", f"sharpa_{n_hands}")
    root.find("compiler").set("meshdir", str(d / "menagerie" / "assets" / "right"))
    opt = root.find("option")
    opt.attrib.clear()
    opt.attrib.update(OPTION)
    if not jlimit:
        # Removed from the joints and their classes alike: `iter` reaches the `<joint>`s
        # inside `<default>` too.
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
        for g in [g for g in body.findall("geom") if _is_collision(g)]:
            body.remove(g)
        if rep == "polytopes":
            cls = "elastomer_collision" if name in SOFT else "collision"
            for i, path in enumerate(_hulls(d, name)):
                ET.SubElement(asset, "mesh", {"name": f"{name}_cp{i}", "file": str(path)})
                ET.SubElement(body, "geom", {"name": f"{name}_cp{i}", "class": cls, "mesh": f"{name}_cp{i}"})
        else:
            # What the link's surface is keyed to: its inertia is the link's own `<inertial>`.
            ET.SubElement(body, "geom", {"name": f"{name}_sdf", "type": "sphere", "size": "0.001",
                                         "contype": "0", "conaffinity": "0", "group": "5", "mass": "0"})
    used = {g.get("mesh") for g in root.iter("geom")} - {None}
    for mesh in [m for m in asset.findall("mesh") if m.get("name") not in used]:
        asset.remove(mesh)

    world.remove(palm)
    world.insert(0, ET.fromstring(f'<geom name="floor" type="plane" size="5 5 0.1" friction="{MU}"/>'))
    if walls:
        half = wall_half(n_hands)
        xml = drop.box_walls_xml(half, WALL_H, drop.WALL_T, MU) if rep == "polytopes" else \
            R.plane_walls_xml(half, WALL_H, MU)
        for el in ET.fromstring(f"<w>{xml}</w>"):
            world.insert(1, el)
    acts = root.find("actuator")
    excludes = root.find("contact")
    act_proto = list(acts) if acts is not None else []
    excl_proto = list(excludes) if excludes is not None else []
    for el in act_proto:
        acts.remove(el)
        # kp is the gain and, negated, the bias on the joint's position; the bias on its
        # velocity, positive here, is the damping ratio that MuJoCo makes kv of.
        gain, bias = el.get("gainprm").split(), el.get("biasprm").split()
        gain[0], bias[1] = (f"{float(x) * kp_scale:.6g}" for x in (gain[0], bias[1]))
        el.set("gainprm", " ".join(gain))
        el.set("biasprm", " ".join(bias))
    for el in excl_proto:
        excludes.remove(el)
    for k in range(n_hands):
        hand = copy.deepcopy(palm)
        for el in hand.iter():
            if "name" in el.attrib:
                el.set("name", f"h{k}_{el.get('name')}")
        hand.insert(0, ET.Element("freejoint", {"name": f"h{k}_root"}))
        world.append(hand)
        for a in act_proto:
            b = copy.deepcopy(a)
            b.set("name", f"h{k}_{a.get('name')}")
            b.set("joint", f"h{k}_{a.get('joint')}")
            acts.append(b)
        for e in excl_proto:
            b = copy.deepcopy(e)
            b.set("body1", f"h{k}_{e.get('body1')}")
            b.set("body2", f"h{k}_{e.get('body2')}")
            excludes.append(b)
    return ET.tostring(root, encoding="unicode")


def _surfaces(d):
    return {link: load_surface_file(d / "final_sdata" / f"{link}.npz") for link in links(d)}


def _density(s):
    """Vertices per unit area of a surface: the denser of a pair supplies its vertices."""
    v, f = np.asarray(s.v), np.asarray(s.f)
    tri = v[f]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=-1).sum()
    return len(v) / float(area)


def hand_points(asset_dir=None):
    """`(n, 3)`: a hand's surface and hull vertices in its palm's frame at `qpos0`, for the stacks."""
    d = Path(asset_dir or ASSET_DIR)
    mjm = mujoco.MjModel.from_xml_string(scene_xml(1, "polytopes", asset_dir=d))
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    pts = []
    for link, s in _surfaces(d).items():
        b = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, f"h0_{link}")
        R_, p = mjd.xmat[b].reshape(3, 3), mjd.xpos[b]
        local = [np.asarray(s.v, float)]
        for path in _hulls(d, link):
            local.append(np.array([[float(x) for x in ln.split()[1:4]] for ln in open(path) if ln.startswith("v ")]))
        pts.append(np.concatenate(local) @ R_.T + p)
    return np.concatenate(pts)


def initial_qpos(mjm, n_hands, seed, asset_dir=None):
    """`(nq,)`: the hands stacked at random orientations (`columns`), their fingers at their `qpos0`."""
    rng = np.random.default_rng(seed)
    free = drop.initial_qpos(n_hands, hand_points(asset_dir), rng, columns(n_hands), GAP, JITTER, HAND_SPACING)
    q = np.array(mjm.qpos0, dtype=np.float64)
    for k in range(n_hands):
        j = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, f"h{k}_root")
        a = int(mjm.jnt_qposadr[j])
        q[a:a + 7] = free[7 * k:7 * k + 7]
    return q


def collides(mjm, ba, bb):
    """
    Whether MuJoCo would collide two bodies of one hand: not welded together, not parent
    and child (after welding), and not named by an `<exclude>`.
    """
    weld, parent = mjm.body_weldid, mjm.body_parentid
    if weld[ba] == weld[bb]:
        return False
    if weld[parent[weld[ba]]] == weld[bb] or weld[parent[weld[bb]]] == weld[ba]:
        return False
    excluded = {frozenset(((int(sig) >> 16), int(sig) & 0xFFFF)) for sig in mjm.exclude_signature}
    return frozenset((ba, bb)) not in excluded


def surfaces(mjm, n_hands, walls=False, asset_dir=None, col_kwargs=None):
    """The Lodos half of the `"surfaces"` representation; see the module docstring."""
    d = Path(asset_dir or ASSET_DIR)
    surf = _surfaces(d)
    names = list(surf)
    dens = {n: _density(s) for n, s in surf.items()}
    body = lambda n: mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
    geom = lambda n: mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, n)  # noqa: E731

    index = {}  # (hand, link) -> surface index
    surfs, geoms = [], []
    for k in range(n_hands):
        for n in names:
            index[k, n] = len(surfs)
            surfs.append(surf[n])
            geoms.append(geom(f"h{k}_{n}_sdf"))
    planes = ["floor"] + ([f"wall{i}" for i in range(4)] if walls else [])
    for p in planes:
        surfs.append(sdf_types.PlaneSurface())
        geoms.append(geom(p))

    def order(a, b):
        return [index[a], index[b]] if dens[a[1]] >= dens[b[1]] else [index[b], index[a]]

    cprs = []
    for k in range(n_hands):
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if collides(mjm, body(f"h{k}_{a}"), body(f"h{k}_{b}")):
                    cprs.append(order((k, a), (k, b)))
    for k in range(n_hands):
        for m in range(k + 1, n_hands):
            cprs += [order((k, a), (m, b)) for a in names for b in names]
    first_plane = len(surfs) - len(planes)
    cprs += [[index[k, n], first_plane + p] for k in range(n_hands) for n in names for p in range(len(planes))]
    return lsim.Surfaces(surfaces=surfs, geoms=geoms, pairs=cprs, params=lsim.default_pair_params(MU, len(cprs)),
                         mu=MU, col_kwargs=dict(col_kwargs or {}))


def build(rep, n_hands, B, walls=False, jlimit=False, seed=0, solver_kwargs=None, col_kwargs=None,
          asset_dir=None, kp_scale=KP_SCALE, **sim_kwargs):
    """The scene as a `lodos.sim.Sim` of `B` worlds, every one from the same stack."""
    mjm = mujoco.MjModel.from_xml_string(scene_xml(n_hands, rep, walls, jlimit, asset_dir, kp_scale))
    q0 = initial_qpos(mjm, n_hands, seed, asset_dir)
    surf = surfaces(mjm, n_hands, walls, asset_dir, col_kwargs) if rep == "surfaces" else None
    return lsim.make_sim(mjm, B, surfaces=surf, qpos=np.tile(q0, (B, 1)), solver_kwargs=solver_kwargs,
                         **sim_kwargs)
