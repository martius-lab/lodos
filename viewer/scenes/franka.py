"""
The Franka Panda of Isaac Lab's NIST demo (`examples/demos/nist.py`, Isaac Lab d67292f), for
`viewer.scenes.nist`.

Isaac Lab's `franka_panda.usda` was generated from Menagerie's `mjx_panda` (Apache-2.0), and the MJCF
that Newton hands MuJoCo Warp for the demo (`SolverMuJoCo(save_to_mjcf=)`) has Menagerie's kinematics
and inertias. So the arm is Menagerie's `mjx_panda.xml`, with the demo's settings in place of its own:

    drives      implicit PD on every joint: kp 600 (joints 1-4), 250, 150, 50; kd 50 (1-4), 30, 25, 15.
                Newton writes each as two MuJoCo actuators, a position part and a velocity part whose
                target is 0; here they are one, gain kp and bias (0, -kp, -kd). The force on each joint
                is limited to 87 N m (1-4) and 12 N m (5-7) by `actuatorfrcrange`, as Newton exports it
    armature    0.6057 (joints 1-2), 0.4625 (3-4), 0.2055 (5-7), 0.1 on each finger; no joint damping
                (Menagerie's 40 and 2 are its own, and Newton's MJCF has none)
    gravcomp    1 on every body
    fingers     Isaac Lab drives finger 1 (kp 350, kd 175, 70 N) and ties finger 2 to it by a mimic
                joint, an equality constraint, which `lodos.rbd` refuses. Here both are driven to the
                same target at half the drive (kp 175, kd 87.5, 35 N). Closing together, the pair is one
                mode of twice a finger's mass, whose frequency and damping are the mimic's, and the grip
                on a centred part is the mimic's too: kp q per finger. They differ only under unequal
                loads (a finger blocked, or a part held off-centre by friction), which the mimic keeps
                symmetric
    limits      none: the threading keeps well inside them
    colliders   Isaac Lab's `gripper_only` set: the hand's capsule and the two finger pads, boxes of
                17.5 x 15.2 x 16.5 mm. They are MJCF geoms in `COLLIDER` (contype 2, conaffinity 0), so
                they meet what a scene gives conaffinity 2 and not each other, as Newton runs the robot
                without self-collision. Neither the arm's links nor the fingers' collide. Each finger
                also carries its pad's Lodos surface (`PAD`, in the finger's frame), keyed to a box geom
                of its own (`g_left_pad`, `g_right_pad`) that MuJoCo Warp never collides
    visuals     Menagerie's meshes, which collide with nothing

The meshes ship in `viewer/assets/nist/franka` (33 MB), a copy of Menagerie's `franka_emika_panda`
(`$LODOS_FRANKA_DIR` overrides it).
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from viewer.scenes import assets

ASSET_DIR = Path(os.environ.get("LODOS_FRANKA_DIR", assets.ASSET_DIR / "nist" / "franka"))

ARM = tuple(f"joint{i}" for i in range(1, 8))
FINGERS = ("finger_joint1", "finger_joint2")
ARM_KP = (600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0)
ARM_KD = (50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0)
ARM_ARMATURE = (0.6057, 0.6057, 0.4625, 0.4625, 0.2055, 0.2055, 0.2055)
ARM_EFFORT = (87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0)
# The joints' ranges (Menagerie's and Newton's), which the model leaves out but `ik` keeps to.
ARM_RANGE = ((-2.8973, 2.8973), (-1.7628, 1.7628), (-2.8973, 2.8973), (-3.0718, -0.0698), (-2.8973, 2.8973),
             (-0.0175, 3.7525), (-2.8973, 2.8973))
# Isaac Lab's drive on finger 1, which its mimic joint shares with finger 2: each finger takes half.
FINGER_KP, FINGER_KD, FINGER_EFFORT, FINGER_ARMATURE = 350.0, 175.0, 70.0, 0.1
FINGER_OPEN = 0.04  # the gripper's open target; closed is 0
HOME = (0.00871, -0.10368, -0.00794, -1.49139, -0.00083, 1.38774, 0.0)  # the demo's initial arm
GRASP = (0.0, 0.0, 0.107)  # the demo's grasp point, in the hand's frame
PAD_HALF = (0.00875, 0.0076, 0.00825)  # a finger pad's half extents, centred at PAD_POS in its finger's frame
PAD_POS = (0.0, 0.00758, 0.04525)
CAPSULE = dict(size="0.04 0.06", pos="0 0 0.03", quat="1 1 0 0")  # the hand's, in its frame
PAD = "franka_pad"  # the pads' Lodos surface, `tools/mesh_processing/nist/make_surfaces.py --pad`
COLLIDER = dict(contype="2", conaffinity="0")


def assets_present(asset_dir=None):
    return (Path(asset_dir or ASSET_DIR) / "mjx_panda.xml").exists()


def _f(v):
    return " ".join(f"{x:.9g}" for x in v)


def mjcf(mu=0.75, asset_dir=None, solref=None, fingers=None):
    """
    The arm as MJCF for a scene to embed: `(assets, body, actuators)`, the `<asset>` children (meshes
    and materials), the base's `<body>` with its tree, and the `<actuator>` children, in the order of
    `ARM` then `FINGERS`: `ctrl` is the seven joint targets, then each finger's. `mu` is the colliders'
    friction, and `solref` their MuJoCo solref when given (only the mj mode reads it). `fingers`
    is each finger's drive, `(kp, kd)`, by default half the demo's (its mimic as two half drives). The
    base sits at the scene's origin, fixed, as Isaac Lab fixes it.
    """
    d = Path(asset_dir or ASSET_DIR)
    if not assets_present(d):
        raise FileNotFoundError(f"no Franka assets in {d}: copy Menagerie's franka_emika_panda there")
    root = ET.parse(d / "mjx_panda.xml").getroot()
    files = {(m.get("name") or Path(m.get("file")).stem): d / "assets" / m.get("file") for m in root.find("asset").iter("mesh")}
    materials = [f'<material name="{m.get("name")}" rgba="{m.get("rgba")}" specular="0.5" shininess="0.25"/>'
                 for m in root.find("asset").iter("material")]
    joints = {j: (a, e, "hinge", "0 0 1") for j, a, e in zip(ARM, ARM_ARMATURE, ARM_EFFORT)}
    joints.update({j: (FINGER_ARMATURE, FINGER_EFFORT / 2, "slide", "0 1 0") for j in FINGERS})
    collider = dict(COLLIDER, friction=f"{mu:g} 0.005 0.0001", **({} if solref is None else dict(solref=_f(solref))))
    used = []

    def convert(src):
        b = ET.Element("body", {k: src.get(k) for k in ("name", "pos", "quat") if src.get(k) is not None})
        b.set("gravcomp", "1")
        for el in src:
            if el.tag == "inertial":
                b.append(el)
            elif el.tag == "joint":
                armature, effort, kind, axis = joints[el.get("name")]
                ET.SubElement(b, "joint", dict(name=el.get("name"), type=kind, axis=axis, armature=f"{armature:g}",
                                               damping="0", actuatorfrclimited="true",
                                               actuatorfrcrange=f"{-effort:g} {effort:g}"))
            elif el.tag == "geom" and el.get("class") == "visual":
                used.append(el.get("mesh"))
                ET.SubElement(b, "geom", dict(type="mesh", mesh=el.get("mesh"), material=el.get("material"),
                                              contype="0", conaffinity="0", group="2"))
            elif el.tag == "body":
                b.append(convert(el))
        name = src.get("name")
        if name == "hand":
            ET.SubElement(b, "geom", dict(name="hand_capsule", type="capsule", group="3", rgba="1 0.45 0 0.3", **CAPSULE,
                                          **collider))
        elif name in ("left_finger", "right_finger"):
            side = name.split("_")[0]
            box = dict(type="box", size=_f(PAD_HALF), pos=_f(PAD_POS), group="3")
            ET.SubElement(b, "geom", dict(name=f"g_{side}_pad", contype="0", conaffinity="0", rgba="0.1 0.45 1 0.6", **box))
            ET.SubElement(b, "geom", dict(name=f"{side}_pad_box", rgba="0.1 0.45 1 0.3", **box, **collider))
        return b

    body = convert(root.find("worldbody").find("body"))
    meshes = [f'<mesh name="{m}" file="{files[m]}"/>' for m in dict.fromkeys(used)]
    kp, kd = (FINGER_KP / 2, FINGER_KD / 2) if fingers is None else fingers
    actuators = [f'<general name="a_{j}" joint="{j}" biastype="affine" gainprm="{p:g}" biasprm="0 {-p:g} {-v:g}"/>'
                 for j, p, v in zip(ARM + FINGERS, ARM_KP + (kp, kp), ARM_KD + (kd, kd))]
    return materials + meshes, ET.tostring(body, encoding="unicode"), actuators


def joint_ids(mjm):
    """`(qpos addresses, dof addresses)` of `ARM` then `FINGERS`."""
    ids = [mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, j) for j in ARM + FINGERS]
    return np.array([mjm.jnt_qposadr[i] for i in ids]), np.array([mjm.jnt_dofadr[i] for i in ids])


def grasp_pose(mjm, qpos):
    """The grasp point's world position and the hand's world rotation matrix at `qpos`, by MuJoCo."""
    d = mujoco.MjData(mjm)
    d.qpos[:] = qpos
    mujoco.mj_kinematics(mjm, d)
    h = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "hand")
    R = d.xmat[h].reshape(3, 3).copy()
    return d.xpos[h] + R @ np.asarray(GRASP), R


def ik(mjm, qpos, target_pos, target_R, iters=100, tol=1e-12):
    """
    `qpos` with the arm's joints moved, by damped least squares within `ARM_RANGE`, until the grasp
    point is at `target_pos` and the hand's frame at `target_R`; raises if it does not get there.
    """
    d = mujoco.MjData(mjm)
    q = np.array(qpos, dtype=np.float64)
    qa, da = joint_ids(mjm)
    qa, da = qa[:7], da[:7]
    h = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "hand")
    lo, hi = np.array(ARM_RANGE).T
    jp, jr = np.zeros((3, mjm.nv)), np.zeros((3, mjm.nv))
    for _ in range(iters):
        d.qpos[:] = q
        mujoco.mj_kinematics(mjm, d)
        mujoco.mj_comPos(mjm, d)
        R = d.xmat[h].reshape(3, 3)
        p = d.xpos[h] + R @ np.asarray(GRASP)
        e = np.concatenate([np.asarray(target_pos) - p, 0.5 * sum(np.cross(R[:, i], target_R[:, i]) for i in range(3))])
        if e @ e < tol:
            return q
        mujoco.mj_jac(mjm, d, jp, jr, p, h)
        J = np.vstack([jp[:, da], jr[:, da]])
        q[qa] = np.clip(q[qa] + J.T @ np.linalg.solve(J @ J.T + 1e-8 * np.eye(6), e), lo, hi)
    raise ValueError(f"the arm does not reach {np.asarray(target_pos)} (residual {np.sqrt(e @ e):.2e})")
