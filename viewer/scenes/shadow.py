"""
Shadow hands dropped on a floor or into a walled box: the viewer's `shadow_*` scenes.

The hand is the Kong scene's (`viewer.scenes.shadow_kong`): MuJoCo Menagerie's Shadow Hand E3M5 as
Dexonomy ships it, without its tendons, forearm and wrist, with the 21 link surfaces baked for it. Its
palm is freed by a free joint and its fingers held open by their own servos at `ctrl = 0`, the Kong
scene's, with the joint limits and friction loss it keeps. `n_hands` copies are stacked at random
orientations, intersection-free, and dropped as the SHARPA hands are (`viewer.scenes.sharpa`): five in one
stack, twenty in four, in its walls and under its options.

Either collision representation, from the same bodies and the same stacks:

- `rep="surfaces"` (the viewer's "sdf"): each link a Lodos surface. Within a hand the pairs are the ones
  MuJoCo would collide (`sharpa.collides`), between hands every pair of links, and each link meets the
  floor and the walls; a pair's vertices come from its denser link (`shadow_kong._density`). The two
  thumb links without a surface, `rh_thbase` and `rh_thhub`, collide with nothing, as in the Kong scene.
- `rep="polytopes"` (the viewer's "polytope"): the convex decomposition of the same links
  (`shadow_kong.hulls`, 70 hulls) as MJCF meshes, through MJWarp's pipeline, which applies the same filter.

Friction is `MU` = 0.6, the hand's in the Kong scene, on every pair, the floor and the walls.
"""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np

from lodos import sim as lsim
from lodos.col.sdf import load_surface_file
from lodos.col.sdf import types as sdf_types
from viewer.scenes import drop
from viewer.scenes import representations as R
from viewer.scenes import shadow_kong as SK
from viewer.scenes import sharpa as S

ASSET_DIR = SK.ASSET_DIR
REPS = SK.REPS
MU = SK.MU
PALM = SK.HAND + "rh_palm"


def assets_present(asset_dir=None):
    return SK.assets_present(asset_dir, "polytopes")


def scene_xml(n_hands, rep, walls=False, asset_dir=None):
    """The MJCF of `n_hands` hands, each body, joint, geom, site and actuator named `h<k>_...`."""
    if rep not in REPS:
        raise ValueError(f"rep must be one of {REPS}")
    d = Path(asset_dir or ASSET_DIR)
    SK._require(d, "polytopes")
    root = ET.parse(d / "scene.xml").getroot()
    root.set("model", f"shadow_{n_hands}")
    opt = root.find("option")
    for el in list(opt):  # Dexonomy's scene turns gravity off
        opt.remove(el)
    opt.attrib.clear()
    opt.attrib.update(S.OPTION)
    for el in root.findall("keyframe"):
        root.remove(el)
    size = root.find("size")
    if size is not None:
        size.attrib.pop("nkey", None)

    asset, world = root.find("asset"), root.find("worldbody")
    palm = next(b for b in world.iter("body") if b.get("name") == PALM)
    for k in ("pos", "quat"):  # the free joint places it
        palm.attrib.pop(k, None)
    key = {"type": "sphere", "size": "0.001", "contype": "0", "conaffinity": "0", "group": "5", "mass": "0"}
    bodies = {b.get("name"): b for b in palm.iter("body")}
    for link in SK.links(d):
        if rep == "polytopes":
            SK.add_hulls(asset, bodies[SK.HAND + link], link, SK.hulls(link, d))
        else:  # what the link's surface is keyed to: its inertia is the link's own `<inertial>`
            ET.SubElement(bodies[SK.HAND + link], "geom", dict(key, name=f"{link}_sdf"))
    for el in [el for el in world if el.tag == "body"]:  # the wrist, the toy; the hand is copied below
        world.remove(el)
    world.insert(0, ET.fromstring(f'<geom name="floor" type="plane" size="5 5 0.1" friction="{MU}"/>'))
    if walls:
        half = S.wall_half(n_hands)
        xml = drop.box_walls_xml(half, S.WALL_H, drop.WALL_T, MU) if rep == "polytopes" else \
            R.plane_walls_xml(half, S.WALL_H, MU)
        for el in ET.fromstring(f"<w>{xml}</w>"):
            world.insert(1, el)
    acts, excludes = root.find("actuator"), root.find("contact")
    joints = {j.get("name") for j in palm.iter("joint")}
    act_proto = [a for a in acts if a.get("joint") in joints]  # the fingers' servos, not the wrist's
    excl_proto = list(excludes) if excludes is not None else []
    for el in list(acts):
        acts.remove(el)
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


def _obj_vertices(path):
    return np.array([[float(x) for x in ln.split()[1:4]] for ln in open(path) if ln.startswith("v ")])


@lru_cache(maxsize=4)
def hand_points(asset_dir=None):
    """`(n, 3)`: a hand's surface and hull vertices in its palm's frame at `qpos0`, for the stacks."""
    d = Path(asset_dir or ASSET_DIR)
    mjm = mujoco.MjModel.from_xml_string(scene_xml(1, "polytopes", asset_dir=d))
    mjd = mujoco.MjData(mjm)
    mujoco.mj_kinematics(mjm, mjd)
    pts = []
    for link in SK.links(d):
        b = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, f"h0_{SK.HAND}{link}")
        local = [np.asarray(load_surface_file(d / "hand" / f"{link}.npz").v, float)]
        local += [_obj_vertices(p) for p in SK.hulls(link, d)]
        pts.append(np.concatenate(local) @ mjd.xmat[b].reshape(3, 3).T + mjd.xpos[b])
    return np.concatenate(pts)


def initial_qpos(mjm, n_hands, seed, asset_dir=None):
    """`(nq,)`: the hands stacked at random orientations (`sharpa.columns`), their fingers at their `qpos0`."""
    rng = np.random.default_rng(seed)
    free = drop.initial_qpos(n_hands, hand_points(asset_dir), rng, S.columns(n_hands), S.GAP, S.JITTER,
                             S.HAND_SPACING)
    q = np.array(mjm.qpos0, dtype=np.float64)
    for k in range(n_hands):
        a = int(mjm.jnt_qposadr[mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, f"h{k}_root")])
        q[a:a + 7] = free[7 * k:7 * k + 7]
    return q


def surfaces(mjm, n_hands, walls=False, asset_dir=None, col_kwargs=None):
    """The Lodos half of the `"surfaces"` representation; see the module docstring."""
    d = Path(asset_dir or ASSET_DIR)
    names = SK.links(d)
    surf = {n: load_surface_file(d / "hand" / f"{n}.npz") for n in names}
    dens = {n: SK._density(s) for n, s in surf.items()}
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
                if S.collides(mjm, body(f"h{k}_{SK.HAND}{a}"), body(f"h{k}_{SK.HAND}{b}")):
                    cprs.append(order((k, a), (k, b)))
    for k in range(n_hands):
        for m in range(k + 1, n_hands):
            cprs += [order((k, a), (m, b)) for a in names for b in names]
    first_plane = len(surfs) - len(planes)
    cprs += [[index[k, n], first_plane + p] for k in range(n_hands) for n in names for p in range(len(planes))]
    return lsim.Surfaces(surfaces=surfs, geoms=geoms, pairs=cprs, params=lsim.default_pair_params(MU, len(cprs)),
                         mu=MU, col_kwargs=dict(col_kwargs or {}))
