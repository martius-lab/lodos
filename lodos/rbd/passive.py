"""
Passive forces -- joint springs, dof damping, and gravity compensation.

There are no fluid forces; `types._validate` rejects a model that asks for them.
"""

import warp as wp

from lodos.rbd.types import BALL, FREE, RbdData, RbdModel
from lodos.utils import math as lmath


@wp.kernel
def _spring_damper(m: RbdModel, d: RbdData):  # type: ignore
    """
    `-stiffness * (q - q_spring) - damping * qvel`, per dof.

    The spring displacement of a rotational joint is a rotation *vector* rather
    than a coordinate difference, so a free or ball joint's angular dofs take
    `quat_sub`, the short way round from the rest orientation to the current one.
    All three of a joint's angular threads recompute that same quaternion
    difference; it is a handful of flops against the cost of coordinating them.

    Launch: dim=(B, nv), a thread per world and dof, by `passive`.
    Reads: d.qpos (B, nq) float32, d.qvel (B, nv) float32; the joints' and dofs' springs and dampers.
    Writes: d.qfrc_passive (B, nv) float32.
    """
    b, v = wp.tid()  # int, int: the world, the dof

    j = m.dof_jntid[v]  # int: the dof's joint
    k = v - m.jnt_dofadr[j]  # int: the dof's index in its joint
    qadr = m.jnt_qposadr[j]  # int: the joint's first qpos
    stiffness = m.jnt_stiffness[j]  # float

    qfrc = float(0.0)  # float: the dof's passive force
    if stiffness != 0.0 and m.disable_spring == 0:
        jtype = m.jnt_type[j]  # int: FREE, BALL, SLIDE or HINGE
        if jtype == FREE and k < 3:
            qfrc = -stiffness * (d.qpos[b, qadr + k] - m.qpos_spring[qadr + k])
        elif jtype == FREE or jtype == BALL:
            o = qadr  # int: the joint's quaternion's first qpos
            c = k  # int: the dof's angular component, 0 to 2
            if jtype == FREE:
                o = qadr + 3
                c = k - 3
            q = wp.vec4(d.qpos[b, o], d.qpos[b, o + 1], d.qpos[b, o + 2], d.qpos[b, o + 3])  # wp.vec4: the orientation
            qs = wp.vec4(  # wp.vec4: the rest orientation
                m.qpos_spring[o], m.qpos_spring[o + 1],
                m.qpos_spring[o + 2], m.qpos_spring[o + 3],
            )
            qfrc = -stiffness * lmath.quat_sub(q, qs)[c]
        else:
            qfrc = -stiffness * (d.qpos[b, qadr] - m.qpos_spring[qadr])

    if m.disable_damper == 0:
        qfrc -= m.dof_damping[v] * d.qvel[b, v]

    d.qfrc_passive[b, v] = qfrc


@wp.kernel
def _gravcomp(m: RbdModel, d: RbdData):  # type: ignore
    """
    The generalized force that cancels a chosen fraction of each body's weight.

    `qfrc[v] = sum_i J_v(xipos_i, i)^T f_i`, and `J_v` is nonzero only for bodies
    in the subtree of `v`'s body -- which is what `body_subtree_id` lists, so the
    sum walks exactly the terms that can contribute.

    Launch: dim=(B, nv), a thread per world and dof, by `passive`.
    Reads: d.cdof (B, nv) wp.spatial_vector, d.subtree_com and d.xipos (B, nbody) wp.vec3; the bodies' masses
    and gravcomp fractions. Writes: d.qfrc_gravcomp (B, nv) float32.
    """
    b, v = wp.tid()  # int, int: the world, the dof

    body = m.dof_bodyid[v]  # int: the dof's body
    cdof = d.cdof[b, v]  # wp.spatial_vector: the dof's motion axis, [angular, linear]
    ang = wp.vec3(cdof[0], cdof[1], cdof[2])  # wp.vec3
    lin = wp.vec3(cdof[3], cdof[4], cdof[5])  # wp.vec3
    root_com = d.subtree_com[b, m.body_rootid[body]]  # wp.vec3: the kinematic tree's centre of mass

    qfrc = float(0.0)  # float: the dof's gravity-compensating force
    for s in range(m.body_subtree_beg[body], m.body_subtree_beg[body + 1]):  # s: int, a subtree slot
        i = m.body_subtree_id[s]  # int: a body in the dof's body's subtree
        force = -m.gravity * (m.body_mass[i] * m.body_gravcomp[i])  # wp.vec3: the force cancelling its weight
        qfrc += wp.dot(lin + wp.cross(ang, d.xipos[b, i] - root_com), force)

    d.qfrc_gravcomp[b, v] = qfrc


@wp.kernel
def _add_gravcomp(m: RbdModel, d: RbdData):  # type: ignore
    """
    Adds gravity compensation to the passive force, unless an actuator does it.

    Launch: dim=(B, nv), a thread per world and dof, by `passive`.
    Reads: d.qfrc_gravcomp (B, nv) float32. Writes: d.qfrc_passive (B, nv) float32, added to.
    """
    b, v = wp.tid()  # int, int: the world, the dof

    if m.jnt_actgravcomp[m.dof_jntid[v]] == 0:
        d.qfrc_passive[b, v] += d.qfrc_gravcomp[b, v]


def passive(m: RbdModel, d: RbdData):  # type: ignore
    """
    All passive forces.

    `mjDSBL_SPRING` and `mjDSBL_DAMPER` are honoured inside the kernel rather than
    by skipping it: they are independent in MuJoCo 3.6, so one being set does not
    mean there is nothing to compute.

    Called by `lodos.rbd.forward.fwd_velocity`, every step.
    In: m, d, B worlds' model and state. Out: d.qfrc_passive (B, nv) float32 and d.qfrc_gravcomp (B, nv) float32.
    """
    wp.launch(_spring_damper, dim=(m.B, m.nv), inputs=(m, d))

    if not m.ngravcomp or m.disable_gravity:
        d.qfrc_gravcomp.zero_()
        return

    wp.launch(_gravcomp, dim=(m.B, m.nv), inputs=(m, d))
    wp.launch(_add_gravcomp, dim=(m.B, m.nv), inputs=(m, d))
