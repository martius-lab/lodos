"""
Editing a state joint by joint: what the panel's fields show, and what a gizmo does.

Positions are edited through each joint's own coordinates -- a free joint's position
and quaternion, a ball joint's quaternion, a hinge's angle, a slide's displacement --
and velocities through its dofs, as `qvel` holds them. MuJoCo's conventions throughout:
a free joint's linear velocity is in the world frame and its angular velocity, like a
ball joint's, in the body's frame; `world_angular`/`local_angular` convert, by the body
orientation, for a panel that offers both.

## Gizmos

A joint's gizmo is a frame placed where the joint's state puts it:

    free    the body's pose -- translate and rotate
    ball    the body's orientation at the joint anchor -- rotate
    hinge   the anchor, z along the axis, turning with the body -- rotate about z
    slide   the anchor moved by the displacement, z along the axis -- translate along z

`gizmo_frame` gives it and `apply_gizmo` turns a moved frame back into the joint's
coordinates. Because the frame is the one the *state* implies -- rebuilt each frame
from the edited state, a hinge's frame turning with its body -- the difference between
the frame passed to the gizmo and the frame it hands back is exactly the motion the
user made this frame, whether the gizmo reports its result relative to the drag's start
or to the frame it was given.
"""

from __future__ import annotations

import dataclasses

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

FREE, BALL, SLIDE, HINGE = (int(mujoco.mjtJoint.mjJNT_FREE), int(mujoco.mjtJoint.mjJNT_BALL),
                            int(mujoco.mjtJoint.mjJNT_SLIDE), int(mujoco.mjtJoint.mjJNT_HINGE))
TYPE_NAMES = {FREE: "free", BALL: "ball", SLIDE: "slide", HINGE: "hinge"}
NQ = {FREE: 7, BALL: 4, SLIDE: 1, HINGE: 1}
NV = {FREE: 6, BALL: 3, SLIDE: 1, HINGE: 1}


@dataclasses.dataclass
class Joint:
    id: int
    name: str
    type: int
    body: int
    qposadr: int
    dofadr: int
    axis: np.ndarray  # (3,) in the body's frame, for hinge and slide
    qpos0: np.ndarray  # (nq,) the joint's reference coordinates

    @property
    def nq(self):
        return NQ[self.type]

    @property
    def nv(self):
        return NV[self.type]

    @property
    def type_name(self):
        return TYPE_NAMES[self.type]


def joints(mjm):
    """Every joint of the model."""
    out = []
    for j in range(int(mjm.njnt)):
        out.append(Joint(
            id=j,
            name=mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint {j}",
            type=int(mjm.jnt_type[j]),
            body=int(mjm.jnt_bodyid[j]),
            qposadr=int(mjm.jnt_qposadr[j]),
            dofadr=int(mjm.jnt_dofadr[j]),
            axis=np.asarray(mjm.jnt_axis[j], np.float64),
            qpos0=np.asarray(mjm.qpos0[int(mjm.jnt_qposadr[j]):int(mjm.jnt_qposadr[j]) + NQ[int(mjm.jnt_type[j])]],
                             np.float64),
        ))
    return out


def body_joints(all_joints, body):
    return [j for j in all_joints if j.body == body]


#######################################################################################
# Rotations, in MuJoCo's (w, x, y, z) order
#######################################################################################


def quat_to_mat(q):
    w, x, y, z = q
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def mat_to_quat(R):
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    q = np.array([w, x, y, z])
    return q if q[0] >= 0 else -q


def quat_to_euler(q):
    """Extrinsic x-y-z angles, degrees: roll about world x, then pitch about y, then yaw about z."""
    w, x, y, z = q
    return Rotation.from_quat([x, y, z, w]).as_euler("xyz", degrees=True)


def euler_to_quat(e):
    x, y, z, w = Rotation.from_euler("xyz", e, degrees=True).as_quat()
    q = np.array([w, x, y, z])
    return q if q[0] >= 0 else -q


def normalize(q):
    q = np.asarray(q, np.float64)
    n = np.linalg.norm(q)
    return q / n if n > 0 else np.array([1.0, 0.0, 0.0, 0.0])


def align_z(axis):
    """A rotation taking z onto `axis` (any one: about z it is free)."""
    a = np.asarray(axis, np.float64)
    a = a / max(np.linalg.norm(a), 1e-12)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, a)
    s, c = np.linalg.norm(v), float(z @ a)
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    return Rotation.from_rotvec(v / s * np.arctan2(s, c)).as_matrix()


#######################################################################################
# Gizmos
#######################################################################################


def _body_pose(frame, b):
    x = frame.body_xform[b].astype(np.float64)  # [pos, quat (x, y, z, w)]
    return x[:3], Rotation.from_quat(x[3:]).as_matrix()


def gizmo_frame(frame, joint):
    """`(4, 4)` world frame of the joint's gizmo in `frame`; see the module docstring."""
    p, R = _body_pose(frame, joint.body)
    M = np.eye(4)
    if joint.type == FREE:
        M[:3, :3], M[:3, 3] = R, p
    elif joint.type == BALL:
        M[:3, :3], M[:3, 3] = R, frame.jnt_anchor[joint.id]
    else:
        M[:3, :3] = R @ align_z(joint.axis)
        M[:3, 3] = frame.jnt_anchor[joint.id]
        if joint.type == SLIDE:
            # MuJoCo places a slide's anchor before applying its displacement, so the
            # anchor stays put as the joint moves; the gizmo goes where the joint is.
            M[:3, 3] = M[:3, 3] + frame.jnt_axis[joint.id] * (frame.qpos[joint.qposadr] - joint.qpos0[0])
    return M


def gizmo_ops(joint):
    """`(translate, rotate)`: the axes, of the gizmo's frame, its handles move along."""
    if joint.type == FREE:
        return "xyz", "xyz"
    if joint.type == BALL:
        return "", "xyz"
    if joint.type == HINGE:
        return "", "z"
    return "z", ""


def apply_gizmo(frame, joint, M_old, M_new, qpos):
    """
    The `qpos` that moves `joint` as the gizmo moved from `M_old` (what `gizmo_frame`
    gave) to `M_new`. Other joints' coordinates are left alone.
    """
    q = np.array(qpos, np.float64)
    a = joint.qposadr
    D = M_new @ np.linalg.inv(M_old)
    if joint.type == FREE:
        q[a:a + 3] = M_new[:3, 3]
        q[a + 3:a + 7] = mat_to_quat(_orthonormal(M_new[:3, :3]))
    elif joint.type == BALL:
        _, R = _body_pose(frame, joint.body)
        Rq = quat_to_mat(normalize(q[a:a + 4]))
        # The body's orientation is R = R_pre Rq: everything above the joint, then the
        # joint. The gizmo turned R by D in the world; the joint absorbs all of it.
        R_pre = R @ Rq.T
        q[a:a + 4] = mat_to_quat(R_pre.T @ _orthonormal(D[:3, :3]) @ R)
    elif joint.type == HINGE:
        axis = M_old[:3, 2]
        q[a] += float(Rotation.from_matrix(_orthonormal(D[:3, :3])).as_rotvec() @ axis)
    else:  # slide
        axis = M_old[:3, 2]
        q[a] += float((M_new[:3, 3] - M_old[:3, 3]) @ axis)
    return q


def _orthonormal(R):
    u, _, vt = np.linalg.svd(R)
    R = u @ vt
    if np.linalg.det(R) < 0:
        u[:, -1] *= -1
        R = u @ vt
    return R


#######################################################################################
# Angular velocities, body frame <-> world frame
#######################################################################################


def world_angular(frame, joint, w_local):
    """A free or ball joint's angular velocity from `qvel`'s body frame to the world's."""
    _, R = _body_pose(frame, joint.body)
    return R @ np.asarray(w_local, np.float64)


def local_angular(frame, joint, w_world):
    _, R = _body_pose(frame, joint.body)
    return R.T @ np.asarray(w_world, np.float64)
