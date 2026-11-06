"""
The Shadow hand lifting a Kong toy in Dexonomy's four-finger prismatic grasp: the viewer's
`shadow_grasp_sdf`, ported from Lodos-v1's run of the grasp (`papers/lodos_v1.pdf`, Section V-A.4),
which held the toy through the lift and 3.3 s after it, and `shadow_grasp_polytope`, the same with the hand
and the toy as convex hulls (`rep="polytopes"`).

The grasp is Dexonomy's grasp 4_3 of its `6_Prismatic_4_Finger` template, made for the Shadow hand
without its tendons (`shadow_notendon`), on DGN_5k's Kong Puppy Teething toy (81 g, 48 x 47 x 76 mm).
The toy floats weightless while the hand approaches, closes and squeezes, and is then lifted 100 mm.
The hand's root moves through the dynamics: six joints stand in for its free joint -- three slides,
then hinges about z, x and y -- each on a position servo, and the fingers are on their own servos at a
twentieth of the original's gain (kp 0.25). The run had the six on one body; here the slides move a
body of their own (`WRIST`) and the hinges the hand's root on it, which composes the same way, so that
Kamino, which takes no joint of six axes, can run the scene. MuJoCo takes no moving body without mass,
so that body weighs a gram (`WRIST_MASS`): the slides carry 0.15% more than the run's, the hinges the
same.

Everything is played open loop from the recorded schedule (`lift`, made with the run's
own scheduling code): a `RecordedControls` row each 4 ms of the wrist's six
targets, the fingers' 22 and the toy's weight. Gravity is off in the scene, as in Dexonomy's, and the
weight is an applied force, `-m g` on the toy from 0.8 s on, 10 mm into the lift, where the trajectory
gives its direction (`ext_fdir`). Here it is a force actuator (`WEIGHT`): a site transmission at the
toy's centre of mass, in a world frame, whose control is the force, so that the weight is one more
column of the recording. The recording ends with the lift, at 1.7 s; its last row then holds -- the
wrist where the lift left it, the weight on -- which is how the run held the grasp.

As the run had it:
- every contact is a Lodos one, between the 21 link surfaces and the toy's (`ASSET_DIR`); `scene.xml`
  is the run's MJCF without its meshes, so none of its collision geometry is left (the two thumb links
  without a surface, `rh_thbase` and `rh_thhub`, collide with nothing);
- friction `MU` = 0.6 on every pair, and under the mj mode MuJoCo's default solref and solimp
  (`SOLREF`, `SOLIMP`), which the MJCF's geoms had;
- the hand against itself on the pairs MuJoCo would collide (`sharpa.collides`), the link with more
  vertices per unit of its bounding box's area supplying the vertices (`_density`, the run's measure);
- explicit Euler at 2 ms, two steps a row.

What the port changes: the fingers keep their joint limits and friction loss, as MuJoCo's reference run
had them (Lodos-v1's dynamics had neither, and its run went without; here they are rows of MJWarp's
solve, `lodos.sim.make_sim`); the toy's inner spheres are not used; and its pairs with the hand are
single-sided, each link's vertices against the toy's field, where the run's were two-sided.

As polytopes, the links and the toy are convex decompositions of them (`hulls`: 70 for the hand, 13 for
the toy), MJCF meshes colliding through MJWarp's pipeline: the toy with the hand, the hand with itself
where MuJoCo's filter has it, which is where the surfaces' pairs are; friction `MU` on every one.

The assets are in `viewer/assets/shadow_kong` (`ASSET_DIR`; `$LODOS_SHADOW_KONG_DIR` overrides it), taken
from the run's files.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from viewer.scenes import assets
from lodos import sim as lsim
from lodos.col.sdf import load_surface_file
from viewer.scenes import sharpa as S

ASSET_DIR = Path(os.environ.get("LODOS_SHADOW_KONG_DIR", assets.ASSET_DIR / "shadow_kong"))
# The recorded lift (`lift`): the run's start, and its controls a row each tick.
LIFT = Path(__file__).resolve().parent / "data" / "shadow_kong_lift.npz"
MU = 0.6
DT = 0.002
SOLREF = (0.02, 1.0)
SOLIMP = (0.9, 0.95, 0.001, 0.5, 2.0)
HAND = "hand-"  # the hand's bodies' prefix, before the link's name
TOY = "obj-mujoco_Kong_Puppy_Teething_Rubber_Small_Pinkobject"  # the toy's body, below its free joint's
WEIGHT = "kong_weight"  # the toy's weight: a force along the world's z, in newtons
WRIST = "wrist"  # the body the wrist's slides move, the hand's root (`hand-world`) its child
# Its mass and inertia: a gram, and a centimetre's inertia, as Kamino gives a massless moving body;
# MuJoCo takes none without mass, and nothing is welded to it.
WRIST_MASS, WRIST_INERTIA = 1e-3, 1e-8


REPS = ("surfaces", "polytopes")


def assets_present(asset_dir=None, rep="surfaces"):
    d = Path(asset_dir or ASSET_DIR)
    hulls = ("hand_cp", "kong_cp") if rep == "polytopes" else ()
    return all((d / p).exists() for p in ("scene.xml", "hand", "kong/kong.npz", *hulls))


def _require(d, rep="surfaces"):
    if not assets_present(d, rep):
        raise FileNotFoundError(f"no Shadow hand and Kong assets in {d}: they ship in viewer/assets/shadow_kong; "
                                f"point LODOS_SHADOW_KONG_DIR at them if they are elsewhere")


def links(asset_dir=None):
    """The 21 links with a surface, in the run's order: by name."""
    d = Path(asset_dir or ASSET_DIR)
    _require(d)
    return sorted(p.stem for p in (d / "hand").glob("*.npz"))


def hulls(name, asset_dir=None):
    """A link's convex hulls (`hand_cp/<link>`), or the toy's (`name="kong"`, `kong_cp`)."""
    d = Path(asset_dir or ASSET_DIR)
    return sorted((d / "kong_cp" if name == "kong" else d / "hand_cp" / name).glob("hull_*.obj"))


def add_hulls(asset, body, name, paths):
    """`paths` as `body`'s colliding meshes, `<name>_cp<i>`, at the scene's friction."""
    for i, path in enumerate(paths):
        ET.SubElement(asset, "mesh", {"name": f"{name}_cp{i}", "file": str(path)})
        ET.SubElement(body, "geom", {"name": f"{name}_cp{i}", "type": "mesh", "mesh": f"{name}_cp{i}",
                                     "friction": f"{MU}", "contype": "1", "conaffinity": "1", "group": "3"})


def scene_xml(asset_dir=None, rep="surfaces"):
    """The MJCF of the hand on its wrist and the toy, as `rep`; see the module docstring."""
    if rep not in REPS:
        raise ValueError(f"rep must be one of {REPS}")
    d = Path(asset_dir or ASSET_DIR)
    _require(d, rep)
    root = ET.parse(d / "scene.xml").getroot()
    root.set("model", "shadow_kong")
    opt = root.find("option")
    opt.set("timestep", f"{DT}")
    opt.set("integrator", "Euler")
    opt.attrib.pop("noslip_iterations", None)
    for el in root.findall("keyframe"):  # Dexonomy's; the start is the run's (`lift`)
        root.remove(el)
    size = root.find("size")
    if size is not None:
        size.attrib.pop("nkey", None)

    world = root.find("worldbody")
    bodies = {b.get("name"): b for b in world.iter("body")}
    # The wrist's slides on a body of their own, without mass, the hinges staying on the hand's root: the
    # same composition as the run's six joints on one body (translate, then turn), so the same kinematics
    # and mass matrix, in a form Kamino takes, which has no joint of six axes.
    hand = bodies[HAND + "world"]
    wrist = ET.Element("body", {"name": WRIST, **{k: hand.attrib.pop(k) for k in ("pos", "quat") if k in hand.attrib}})
    ET.SubElement(wrist, "inertial", {"pos": "0 0 0", "mass": f"{WRIST_MASS}",
                                      "diaginertia": " ".join([f"{WRIST_INERTIA}"] * 3)})
    for joint in [j for j in hand.findall("joint") if j.get("type") == "slide"]:
        hand.remove(joint)
        wrist.append(joint)
    world.insert(list(world).index(hand), wrist)
    world.remove(hand)
    wrist.append(hand)
    key = {"type": "sphere", "size": "0.001", "contype": "0", "conaffinity": "0", "group": "5", "mass": "0"}
    asset, toy = root.find("asset"), bodies[TOY]
    for link in links(d):  # what each surface is keyed to: its inertia is the link's own `<inertial>`
        if rep == "polytopes":
            add_hulls(asset, bodies[HAND + link], link, hulls(link, d))
        else:
            ET.SubElement(bodies[HAND + link], "geom", dict(key, name=f"{link}_sdf"))
    if rep == "polytopes":
        add_hulls(asset, toy, "kong", hulls("kong", d))
    else:
        ET.SubElement(toy, "geom", dict(key, name="kong_sdf"))
    ET.SubElement(toy, "site", {"name": "kong_com", "pos": toy.find("inertial").get("pos")})
    world.insert(0, ET.Element("site", {"name": "world_frame"}))
    ET.SubElement(root.find("actuator"), "motor", {"name": WEIGHT, "site": "kong_com", "refsite": "world_frame",
                                                   "gear": "0 0 1 0 0 0"})
    return ET.tostring(root, encoding="unicode")


def _density(s):
    """Vertices per unit area of a surface's bounding box: the denser of a pair supplies its vertices."""
    ext = np.maximum(np.ptp(np.asarray(s.v), axis=0), 1e-9)
    return len(s.v) / (2.0 * (ext[0] * ext[1] + ext[1] * ext[2] + ext[2] * ext[0]))


def surfaces(mjm, asset_dir=None, col_kwargs=None):
    """The Lodos half: the 21 link surfaces and the toy's; see the module docstring."""
    d = Path(asset_dir or ASSET_DIR)
    names = links(d)
    hand = [load_surface_file(d / "hand" / f"{n}.npz") for n in names]
    toy = load_surface_file(d / "kong" / "kong.npz")
    geom = lambda n: mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, n)  # noqa: E731
    body = lambda n: mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, HAND + n)  # noqa: E731

    k = len(names)
    cprs = [[i, k] for i in range(k)]  # each link's vertices against the toy's field
    dens = [_density(s) for s in hand]
    for i in range(k):
        for j in range(i + 1, k):
            if S.collides(mjm, body(names[i]), body(names[j])):
                cprs.append([j, i] if dens[j] > dens[i] else [i, j])
    params = lsim.default_pair_params(MU, len(cprs))
    params[:, 5:7] = SOLREF
    params[:, 7:12] = SOLIMP
    return lsim.Surfaces(surfaces=hand + [toy], geoms=[geom(f"{n}_sdf") for n in names] + [geom("kong_sdf")],
                         pairs=cprs, params=params, mu=MU, col_kwargs=dict(col_kwargs or {}))


def lift(mjm, path=LIFT):
    """
    The recorded lift: `(qpos, qvel, ctrl, period)`, the state the run started from and the controls,
    `(T, nu)`, a row each `period` s. Refused unless its actuators are `mjm`'s, in its order.
    """
    with np.load(path) as z:
        names = [str(n) for n in z["actuators"]]
        have = [mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_ACTUATOR, u) for u in range(mjm.nu)]
        if names != have:
            raise ValueError(f"{path} records controls for the actuators {names}, the model has {have}")
        q, v = np.asarray(z["qpos"], dtype=np.float64), np.asarray(z["qvel"], dtype=np.float64)
        if q.shape != (mjm.nq,) or v.shape != (mjm.nv,):
            raise ValueError(f"{path} records a state of {q.shape} and {v.shape}, the model's is ({mjm.nq},) and ({mjm.nv},)")
        return q, v, np.asarray(z["ctrl"], dtype=np.float64), float(z["period"])
