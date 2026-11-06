"""
Position-dependent rigid-body quantities, in Warp.

A forward tree pass is one kernel launch per tree level, reading parents the
previous launch wrote, and a reverse pass is one launch per level in the other
order, reading *children* -- see `types.RbdModel` for why the reverse direction
pulls rather than pushes.

Every launch is `(B, nbody)` regardless of how wide the level actually is, and
threads past the level's width exit after two integer loads. That trades a little
occupancy on deep trees for not needing any per-level bound on the host, which
would otherwise have to travel outside the `RbdModel` struct -- Warp structs hold
only scalars and device arrays.
"""

import warp as wp

from lodos.rbd.types import (BALL, FREE, HINGE, MINVAL, SLIDE,
                                 TRN_JOINTINPARENT, TRN_SITE, RbdData, RbdModel,
                                 vec10)
from lodos.utils import math as lmath


#######################################################################################
# Kinematics
#######################################################################################


@wp.kernel
def _kinematics_level(m: RbdModel, d: RbdData, level: int):  # type: ignore
    """
    World pose of every body at one tree level, and of the joints it hangs on.

    Each joint is applied in order to a running `(pos, quat)`: the anchor and axis
    are recorded against the pose *before* that joint moves the body, and the pose
    after it is what the next joint -- and the body's children -- see. `qpos` is
    normalized in place for the joints that carry a quaternion.

    Launch: dim=(B, nbody), a thread per world and body of the level (the threads past its width exit), once per
    tree level, by `kinematics`.
    In: level int, the tree level. Reads: d.qpos (B, nq) float32; the parent's d.xpos and d.xquat.
    Writes: d.xpos and d.xipos (B, nbody) wp.vec3, d.xquat (B, nbody) wp.vec4, d.xmat and d.ximat (B, nbody)
    wp.mat33, d.xanchor and d.xaxis (B, njnt) wp.vec3; the joints' quaternions in d.qpos, normalized.
    """
    b, k = wp.tid()  # int, int: the world, the body's slot in the level

    beg = m.body_level_beg[level]  # int: the level's first slot in body_level_id
    if beg + k >= m.body_level_beg[level + 1]:
        return
    i = m.body_level_id[beg + k]  # int: the body

    # The world body has no parent to compose against; every other body starts
    # from its parent's frame.
    pos = m.body_pos[i]  # wp.vec3: the running frame's position
    quat = m.body_quat[i]  # wp.vec4: the running frame's orientation
    if i > 0:
        p = m.body_parentid[i]  # int: the parent
        pos = d.xpos[b, p] + lmath.rotate(pos, d.xquat[b, p])
        quat = lmath.quat_mul(d.xquat[b, p], quat)

    jadr = m.body_jntadr[i]  # int: the body's first joint
    for n in range(m.body_jntnum[i]):  # n: int, a joint of the body, in order
        j = jadr + n  # int: the joint
        jtype = m.jnt_type[j]  # int: FREE, BALL, SLIDE or HINGE
        qadr = m.jnt_qposadr[j]  # int: the joint's first qpos

        anchor = wp.vec3(0.0, 0.0, 1.0)  # wp.vec3: the joint's anchor, world frame
        axis = wp.vec3(0.0, 0.0, 1.0)  # wp.vec3: the joint's axis, world frame
        if jtype == FREE:
            anchor = wp.vec3(d.qpos[b, qadr], d.qpos[b, qadr + 1], d.qpos[b, qadr + 2])
        else:
            anchor = lmath.rotate(m.jnt_pos[j], quat) + pos
            axis = lmath.rotate(m.jnt_axis[j], quat)

        d.xanchor[b, j] = anchor
        d.xaxis[b, j] = axis

        if jtype == FREE:
            pos = anchor
            quat = lmath.normalize4(
                wp.vec4(
                    d.qpos[b, qadr + 3],
                    d.qpos[b, qadr + 4],
                    d.qpos[b, qadr + 5],
                    d.qpos[b, qadr + 6],
                )
            )
            for c in range(4):  # c: int, a quaternion component
                d.qpos[b, qadr + 3 + c] = quat[c]
        elif jtype == BALL:
            qloc = lmath.normalize4(  # wp.vec4: the ball joint's rotation, normalized
                wp.vec4(
                    d.qpos[b, qadr],
                    d.qpos[b, qadr + 1],
                    d.qpos[b, qadr + 2],
                    d.qpos[b, qadr + 3],
                )
            )
            for c in range(4):
                d.qpos[b, qadr + c] = qloc[c]
            quat = lmath.quat_mul(quat, qloc)
            pos = anchor - lmath.rotate(m.jnt_pos[j], quat)
        elif jtype == HINGE:
            angle = d.qpos[b, qadr] - m.qpos0[qadr]  # float: the hinge angle from qpos0 (rad)
            quat = lmath.quat_mul(quat, lmath.axis_angle_to_quat(m.jnt_axis[j], angle))
            pos = anchor - lmath.rotate(m.jnt_pos[j], quat)
        else:  # SLIDE
            pos += axis * (d.qpos[b, qadr] - m.qpos0[qadr])

    d.xpos[b, i] = pos
    d.xquat[b, i] = quat
    d.xmat[b, i] = lmath.quat_to_mat(quat)
    d.xipos[b, i] = pos + lmath.rotate(m.body_ipos[i], quat)
    d.ximat[b, i] = lmath.quat_to_mat(lmath.quat_mul(quat, m.body_iquat[i]))


@wp.kernel
def _site_poses(m: RbdModel, d: RbdData):  # type: ignore
    """
    Site frames in world coordinates. Only actuator transmission reads these.

    Launch: dim=(B, nsite), a thread per world and site, by `kinematics`.
    Reads: d.xpos (B, nbody) wp.vec3, d.xquat (B, nbody) wp.vec4. Writes: d.site_xpos (B, nsite) wp.vec3,
    d.site_xmat (B, nsite) wp.mat33.
    """
    b, s = wp.tid()  # int, int: the world, the site

    i = m.site_bodyid[s]  # int: the site's body
    d.site_xpos[b, s] = d.xpos[b, i] + lmath.rotate(m.site_pos[s], d.xquat[b, i])
    d.site_xmat[b, s] = lmath.quat_to_mat(
        lmath.quat_mul(d.xquat[b, i], m.site_quat[s])
    )


def kinematics(m: RbdModel, d: RbdData):  # type: ignore
    """
    Generalized coordinates to world-frame body, joint and site poses.

    Called by `lodos.rbd.forward.fwd_position`, `lodos.sim.make_sim`, `lodos.utils.states.install` and 1 other.
    In: m, d, B worlds' model and state; reads d.qpos (B, nq) float32 and normalizes its quaternions in place.
    Out: d.xpos and d.xipos (B, nbody) wp.vec3, d.xquat (B, nbody) wp.vec4, d.xmat and d.ximat (B, nbody) wp.mat33,
    d.xanchor and d.xaxis (B, njnt) wp.vec3, d.site_xpos (B, nsite) wp.vec3, d.site_xmat (B, nsite) wp.mat33.
    """
    for level in range(m.nlevel):  # level: int, a tree level, the world's first
        wp.launch(_kinematics_level, dim=(m.B, m.nbody), inputs=(m, d, level))

    if m.nsite:
        wp.launch(_site_poses, dim=(m.B, m.nsite), inputs=(m, d))


#######################################################################################
# Center-of-mass frames
#######################################################################################


@wp.kernel
def _subtree_init(m: RbdModel, d: RbdData):  # type: ignore
    """
    Seeds the subtree sums with each body's own mass moment.

    Launch: dim=(B, nbody), a thread per world and body, by `com_pos`.
    Reads: d.xipos (B, nbody) wp.vec3. Writes: d.subtree_com (B, nbody) wp.vec3, the mass moment for now, and
    d.subtree_mass (B, nbody) float32.
    """
    b, i = wp.tid()  # int, int: the world, the body

    d.subtree_com[b, i] = d.xipos[b, i] * m.body_mass[i]
    d.subtree_mass[b, i] = m.body_mass[i]


@wp.kernel
def _subtree_acc(m: RbdModel, d: RbdData, level: int):  # type: ignore
    """
    Adds each body's children's subtree sums into its own.

    Launch: dim=(B, nbody), a thread per world and body of the level (the threads past its width exit), once per
    tree level, deepest first, by `com_pos`.
    In: level int, the tree level. Reads: the children's d.subtree_com (B, nbody) wp.vec3 and d.subtree_mass
    (B, nbody) float32. Writes: the body's.
    """
    b, k = wp.tid()  # int, int: the world, the body's slot in the level

    beg = m.body_level_beg[level]  # int: the level's first slot in body_level_id
    if beg + k >= m.body_level_beg[level + 1]:
        return
    i = m.body_level_id[beg + k]  # int: the body

    pos = d.subtree_com[b, i]  # wp.vec3: the subtree's mass moment
    mass = d.subtree_mass[b, i]  # float: the subtree's mass
    for c in range(m.body_child_beg[i], m.body_child_beg[i + 1]):  # c: int, a slot in body_child_id
        child = m.body_child_id[c]  # int: a child of the body
        pos += d.subtree_com[b, child]
        mass += d.subtree_mass[b, child]

    d.subtree_com[b, i] = pos
    d.subtree_mass[b, i] = mass


@wp.kernel
def _subtree_div(d: RbdData):  # type: ignore
    """
    Turns the accumulated mass moment into a center of mass.

    Launch: dim=(B, nbody), a thread per world and body, by `com_pos`.
    Reads: d.subtree_mass (B, nbody) float32, d.xipos (B, nbody) wp.vec3. Writes: d.subtree_com (B, nbody) wp.vec3.
    """
    b, i = wp.tid()  # int, int: the world, the body

    mass = d.subtree_mass[b, i]  # float: the subtree's mass
    if mass < MINVAL:
        d.subtree_com[b, i] = d.xipos[b, i]
    else:
        d.subtree_com[b, i] = d.subtree_com[b, i] / mass


@wp.kernel
def _cinert(m: RbdModel, d: RbdData):  # type: ignore
    """
    Body inertia about the root's center of mass, in world axes.

    `R diag(I) R^T` rotates the inertia into world axes and `mass * skew(off)
    skew(off)^T` moves it off the body's own center of mass -- the parallel-axis
    term. Packed as the upper triangle [xx, yy, zz, xy, xz, yz], then `mass * off`, then `mass`.

    Launch: dim=(B, nbody), a thread per world and body, by `com_pos`.
    Reads: d.xipos and d.subtree_com (B, nbody) wp.vec3, d.ximat (B, nbody) wp.mat33; the bodies' masses and
    inertias. Writes: d.cinert (B, nbody) vec10.
    """
    b, i = wp.tid()  # int, int: the world, the body

    off = d.xipos[b, i] - d.subtree_com[b, m.body_rootid[i]]  # wp.vec3: the body's COM from its root's
    mass = m.body_mass[i]  # float

    r = d.ximat[b, i]  # wp.mat33: the inertial frame
    diag = m.body_inertia[i]  # wp.vec3: the principal inertia
    rd = wp.mat33(  # wp.mat33: r diag(I)
        r[0, 0] * diag[0], r[0, 1] * diag[1], r[0, 2] * diag[2],
        r[1, 0] * diag[0], r[1, 1] * diag[1], r[1, 2] * diag[2],
        r[2, 0] * diag[0], r[2, 1] * diag[1], r[2, 2] * diag[2],
    )
    inr = rd * wp.transpose(r)  # wp.mat33: the inertia in world axes, then about the root's COM

    h = wp.mat33(  # wp.mat33: skew(off)
        0.0, -off[2], off[1],
        off[2], 0.0, -off[0],
        -off[1], off[0], 0.0,
    )
    inr += (h * wp.transpose(h)) * mass

    d.cinert[b, i] = vec10(
        inr[0, 0], inr[1, 1], inr[2, 2],
        inr[0, 1], inr[0, 2], inr[1, 2],
        off[0] * mass, off[1] * mass, off[2] * mass,
        mass,
    )


@wp.kernel
def _cdof(m: RbdModel, d: RbdData):  # type: ignore
    """
    Motion axis of every dof, about the root center of mass, in world axes.

    A rotational dof about world-frame axis `a` through `xanchor` moves the root
    COM at `cross(a, root_com - xanchor)`, which is the linear half; a sliding dof
    has no angular part at all. The three rotational dofs of a free or ball joint
    turn about the body's own axes, which are the columns of `xmat`.

    One thread per *dof* rather than per joint. Both write the same 36 bytes per
    free joint, but per joint they land 144 bytes apart and per dof they are
    consecutive, so the writes coalesce.

    Launch: dim=(B, nv), a thread per world and dof, by `com_pos`.
    Reads: d.subtree_com (B, nbody) wp.vec3, d.xanchor and d.xaxis (B, njnt) wp.vec3, d.xmat (B, nbody) wp.mat33.
    Writes: d.cdof (B, nv) wp.spatial_vector, [angular, linear].
    """
    b, i = wp.tid()  # int, int: the world, the dof

    j = m.dof_jntid[i]  # int: the dof's joint
    k = i - m.jnt_dofadr[j]  # int: the dof's index in its joint
    jtype = m.jnt_type[j]  # int: FREE, BALL, SLIDE or HINGE
    body = m.jnt_bodyid[j]  # int: the joint's body
    off = d.subtree_com[b, m.body_rootid[body]] - d.xanchor[b, j]  # wp.vec3: the root's COM from the anchor

    if jtype == FREE and k < 3:
        # World-aligned translation, along world axis k.
        lin = wp.vec3(0.0, 0.0, 0.0)  # wp.vec3: the translation's axis
        lin[k] = 1.0
        d.cdof[b, i] = wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), lin)
        return

    axis = d.xaxis[b, j]  # wp.vec3: the dof's axis, world frame
    if jtype == FREE or jtype == BALL:
        c = k  # int: the dof's rotational component, 0 to 2
        if jtype == FREE:
            c = k - 3
        r = d.xmat[b, body]  # wp.mat33: the body's frame, its columns the axes
        axis = wp.vec3(r[0, c], r[1, c], r[2, c])
    elif jtype == SLIDE:
        d.cdof[b, i] = wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), axis)
        return

    d.cdof[b, i] = wp.spatial_vector(axis, wp.cross(axis, off))


def com_pos(m: RbdModel, d: RbdData):  # type: ignore
    """
    Subtree centers of mass, and the inertias and motion dofs about them.

    Called by `lodos.rbd.forward.fwd_position`.
    In: m, d, B worlds' model and state; reads the poses `kinematics` wrote. Out: d.subtree_com (B, nbody) wp.vec3,
    d.subtree_mass (B, nbody) float32, d.cinert (B, nbody) vec10, d.cdof (B, nv) wp.spatial_vector.
    """
    wp.launch(_subtree_init, dim=(m.B, m.nbody), inputs=(m, d))
    for level in range(m.nlevel - 1, -1, -1):  # level: int, a tree level, the deepest first
        wp.launch(_subtree_acc, dim=(m.B, m.nbody), inputs=(m, d, level))
    wp.launch(_subtree_div, dim=(m.B, m.nbody), inputs=(d,))

    wp.launch(_cinert, dim=(m.B, m.nbody), inputs=(m, d))
    if m.nv:
        wp.launch(_cdof, dim=(m.B, m.nv), inputs=(m, d))


#######################################################################################
# Velocity
#######################################################################################


@wp.kernel
def _com_vel_level(m: RbdModel, d: RbdData, level: int):  # type: ignore
    """
    Body velocities about the subtree COM, and the time derivative of each dof's
    motion axis.

    Both come out of the same walk: a body's velocity is its parent's plus each of
    its own dofs' contributions, applied in dof order, and `cdof_dot` is the
    motion cross of the velocity *seen so far* with the axis about to be added.
    The order matters -- a free joint's three rotational axes are dragged by the
    translation that precedes them, and by nothing else, which is why their
    `cdof_dot` is taken after the translational sum and the translational axes'
    is exactly zero.

    Launch: dim=(B, nbody), a thread per world and body of the level (the threads past its width exit), once per
    tree level, by `com_vel`.
    In: level int, the tree level. Reads: d.cdof (B, nv) wp.spatial_vector, d.qvel (B, nv) float32; the parent's
    d.cvel. Writes: d.cvel (B, nbody) and d.cdof_dot (B, nv) wp.spatial_vector.
    """
    b, k = wp.tid()  # int, int: the world, the body's slot in the level

    beg = m.body_level_beg[level]  # int: the level's first slot in body_level_id
    if beg + k >= m.body_level_beg[level + 1]:
        return
    i = m.body_level_id[beg + k]  # int: the body

    cvel = wp.spatial_vector()  # wp.spatial_vector: the running velocity, [angular, linear]
    if i > 0:
        cvel = d.cvel[b, m.body_parentid[i]]

    jadr = m.body_jntadr[i]  # int: the body's first joint
    for n in range(m.body_jntnum[i]):  # n: int, a joint of the body, in order
        j = jadr + n  # int: the joint
        adr = m.jnt_dofadr[j]  # int: the joint's first dof
        jtype = m.jnt_type[j]  # int: FREE, BALL, SLIDE or HINGE

        if jtype == FREE:
            for c in range(3):  # c: int, a dof of the joint
                cvel += d.cdof[b, adr + c] * d.qvel[b, adr + c]
                d.cdof_dot[b, adr + c] = wp.spatial_vector()
            for c in range(3, 6):
                d.cdof_dot[b, adr + c] = lmath.motion_cross(cvel, d.cdof[b, adr + c])
            for c in range(3, 6):
                cvel += d.cdof[b, adr + c] * d.qvel[b, adr + c]
        elif jtype == BALL:
            for c in range(3):
                d.cdof_dot[b, adr + c] = lmath.motion_cross(cvel, d.cdof[b, adr + c])
            for c in range(3):
                cvel += d.cdof[b, adr + c] * d.qvel[b, adr + c]
        else:
            d.cdof_dot[b, adr] = lmath.motion_cross(cvel, d.cdof[b, adr])
            cvel += d.cdof[b, adr] * d.qvel[b, adr]

    d.cvel[b, i] = cvel


def com_vel(m: RbdModel, d: RbdData):  # type: ignore
    """
    Body velocities and dof-axis derivatives, about the subtree COM.

    Called by `lodos.rbd.forward.fwd_velocity`.
    In: m, d, B worlds' model and state; reads d.cdof (B, nv) wp.spatial_vector and d.qvel (B, nv) float32.
    Out: d.cvel (B, nbody) and d.cdof_dot (B, nv) wp.spatial_vector.
    """
    for level in range(m.nlevel):  # level: int, a tree level, the world's first
        wp.launch(_com_vel_level, dim=(m.B, m.nbody), inputs=(m, d, level))


#######################################################################################
# Bias forces
#######################################################################################


@wp.kernel
def _cacc_level(m: RbdModel, d: RbdData, level: int):  # type: ignore
    """
    Bias acceleration: what a body accelerates at with `qacc = 0`.

    The world body carries `-gravity`, so that the recursion produces weight as a
    bias force and the constrained solve never has to add it separately. Each
    body then adds `cdof_dot . qvel` over its own dofs -- the part of the
    acceleration that comes from the axes themselves moving.

    Launch: dim=(B, nbody), a thread per world and body of the level (the threads past its width exit), once per
    tree level, by `rne`.
    In: level int, the tree level. Reads: d.cdof_dot (B, nv) wp.spatial_vector, d.qvel (B, nv) float32; the parent's
    d.cacc. Writes: d.cacc (B, nbody) wp.spatial_vector.
    """
    b, k = wp.tid()  # int, int: the world, the body's slot in the level

    beg = m.body_level_beg[level]  # int: the level's first slot in body_level_id
    if beg + k >= m.body_level_beg[level + 1]:
        return
    i = m.body_level_id[beg + k]  # int: the body

    if i == 0:
        cacc = wp.spatial_vector()  # wp.spatial_vector: the body's bias acceleration, [angular, linear]
        if m.disable_gravity == 0:
            cacc = wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), -m.gravity)
        d.cacc[b, 0] = cacc
        return

    cacc = d.cacc[b, m.body_parentid[i]]
    for v in range(m.body_dofadr[i], m.body_dofadr[i] + m.body_dofnum[i]):  # v: int, a dof of the body
        cacc += d.cdof_dot[b, v] * d.qvel[b, v]

    d.cacc[b, i] = cacc


@wp.kernel
def _cfrc_local(d: RbdData):  # type: ignore
    """
    Each body's own share of the bias force, before the subtree sum.

    Launch: dim=(B, nbody), a thread per world and body, by `rne`.
    Reads: d.cinert (B, nbody) vec10, d.cacc and d.cvel (B, nbody) wp.spatial_vector. Writes: d.cfrc (B, nbody)
    wp.spatial_vector.
    """
    b, i = wp.tid()  # int, int: the world, the body

    inr = d.cinert[b, i]  # vec10: the body's inertia
    d.cfrc[b, i] = lmath.inert_mul(inr, d.cacc[b, i]) + lmath.motion_cross_force(
        d.cvel[b, i], lmath.inert_mul(inr, d.cvel[b, i])
    )


@wp.kernel
def _cfrc_acc(m: RbdModel, d: RbdData, level: int):  # type: ignore
    """
    Accumulates each body's children's bias forces into its own.

    Launch: dim=(B, nbody), a thread per world and body of the level (the threads past its width exit), once per
    tree level, deepest first, by `rne`.
    In: level int, the tree level. Reads: the children's d.cfrc (B, nbody) wp.spatial_vector. Writes: the body's.
    """
    b, k = wp.tid()  # int, int: the world, the body's slot in the level

    beg = m.body_level_beg[level]  # int: the level's first slot in body_level_id
    if beg + k >= m.body_level_beg[level + 1]:
        return
    i = m.body_level_id[beg + k]  # int: the body

    frc = d.cfrc[b, i]  # wp.spatial_vector: the subtree's bias force
    for c in range(m.body_child_beg[i], m.body_child_beg[i + 1]):  # c: int, a slot in body_child_id
        frc += d.cfrc[b, m.body_child_id[c]]

    d.cfrc[b, i] = frc


@wp.kernel
def _qfrc_bias(m: RbdModel, d: RbdData):  # type: ignore
    """
    Projects the accumulated spatial forces onto the dof axes.

    Launch: dim=(B, nv), a thread per world and dof, by `rne`.
    Reads: d.cdof (B, nv) and d.cfrc (B, nbody) wp.spatial_vector. Writes: d.qfrc_bias (B, nv) float32.
    """
    b, v = wp.tid()  # int, int: the world, the dof

    d.qfrc_bias[b, v] = wp.dot(d.cdof[b, v], d.cfrc[b, m.dof_bodyid[v]])


def rne(m: RbdModel, d: RbdData):  # type: ignore
    """
    Recursive Newton-Euler: the bias force at `qacc = 0`.

    Called by `lodos.rbd.forward.fwd_velocity`.
    In: m, d, B worlds' model and state; reads d.cdof and d.cdof_dot (B, nv) and d.cvel (B, nbody) wp.spatial_vector,
    d.cinert (B, nbody) vec10, d.qvel (B, nv) float32. Out: d.qfrc_bias (B, nv) float32; d.cacc and d.cfrc
    (B, nbody) wp.spatial_vector.
    """
    for level in range(m.nlevel):  # level: int, a tree level, the world's first
        wp.launch(_cacc_level, dim=(m.B, m.nbody), inputs=(m, d, level))

    wp.launch(_cfrc_local, dim=(m.B, m.nbody), inputs=(d,))
    for level in range(m.nlevel - 1, -1, -1):
        wp.launch(_cfrc_acc, dim=(m.B, m.nbody), inputs=(m, d, level))

    wp.launch(_qfrc_bias, dim=(m.B, m.nv), inputs=(m, d))


#######################################################################################
# Mass matrix
#######################################################################################
#
# M is block diagonal across kinematic trees -- `support.make_m` masks it to the
# dof-ancestor pattern -- so it is kept as one dense `n_t x n_t` block per tree, the
# decomposition `mujoco_warp` derives as `qM_tiles`, and its factor is block diagonal
# too. A branched tree's block is dense even where that pattern is not.


@wp.kernel
def _crb_acc(m: RbdModel, d: RbdData, level: int):  # type: ignore
    """
    Composite rigid-body inertia: each body accumulates its whole subtree.

    Launch: dim=(B, nbody), a thread per world and body of the level (the threads past its width exit), once per
    tree level, deepest first, by `build_m`.
    In: level int, the tree level. Reads: d.cinert (B, nbody) vec10, the children's d.crb. Writes: d.crb (B, nbody)
    vec10.
    """
    b, k = wp.tid()  # int, int: the world, the body's slot in the level

    beg = m.body_level_beg[level]  # int: the level's first slot in body_level_id
    if beg + k >= m.body_level_beg[level + 1]:
        return
    i = m.body_level_id[beg + k]  # int: the body

    # The world's own entry is zeroed rather than summed. Nothing reads it --
    # `dof_bodyid` is never 0.
    if i == 0:
        d.crb[b, 0] = vec10()
        return

    inr = d.cinert[b, i]  # vec10: the subtree's inertia
    for c in range(m.body_child_beg[i], m.body_child_beg[i + 1]):  # c: int, a slot in body_child_id
        inr += d.crb[b, m.body_child_id[c]]

    d.crb[b, i] = inr


@wp.kernel
def _build_M(m: RbdModel, d: RbdData):  # type: ignore
    """
    One tree's mass-matrix block: `M[i, j] = (crb[body(i)] cdof[i]) . cdof[j]` for
    every `j` on the dof path from `i` to the tree root, plus armature.

    Walking `dof_parentid` up from `i` visits exactly the pattern
    `support.make_m` keeps, and the walk cannot leave the tree, so writing both
    `(r, c)` and `(c, r)` fills the block symmetrically. Everything else in the
    block is structurally zero.

    Launch: dim=(ntree, B), a thread per tree and world, by `build_m`; a large tree's thread returns at once. The batch
    is last, matching the layout of `M_blk`: consecutive threads differ in `b` and write adjacent floats, rather than
    addresses a whole block apart.
    Reads: d.crb (B, nbody) vec10, d.cdof (B, nv) wp.spatial_vector; the dofs' armature. Writes: d.M_blk (nblk, B)
    float32, the small trees' blocks.
    """
    t, b = wp.tid()  # int, int: the tree, the world
    if m.tree_big[t] >= 0:
        return  # `_build_M_rows`

    beg = m.tree_beg[t]  # int: the tree's first dof
    end = m.tree_beg[t + 1]  # int: one past its last
    n = end - beg  # int: the tree's dof count
    o = m.blk_beg[t]  # int: the tree's block offset in M_blk

    for e in range(n * n):  # e: int, an entry of the block
        d.M_blk[o + e, b] = 0.0

    for i in range(beg, end):  # i: int, a dof of the tree
        f_i = lmath.inert_mul(d.crb[b, m.dof_bodyid[i]], d.cdof[b, i])  # wp.spatial_vector: crb[body(i)] cdof[i]
        r = i - beg  # int: i's row in the block

        j = i  # int: a dof on the path from i to the root
        while j > -1:
            v = wp.dot(f_i, d.cdof[b, j])  # float: M[i, j]
            c = j - beg  # int: j's column in the block
            d.M_blk[o + r * n + c, b] = v
            d.M_blk[o + c * n + r, b] = v
            j = m.dof_parentid[j]

        d.M_blk[o + r * n + r, b] += m.dof_armature[i]


@wp.kernel
def _factor_M(
    m: RbdModel,  # type: ignore
    M_blk: wp.array(dtype=float, ndim=2),  # (nblk, B) # type: ignore
    L_blk: wp.array(dtype=float, ndim=2),  # (nblk, B) # type: ignore
):
    """
    Cholesky factorization of one tree's block, `M = L L^T`: `M_blk`, or under
    `implicitfast` the implicit matrix (`lodos.rbd.implicit`).

    Scalar Cholesky-Banachiewicz in a single thread. For the free-floating case
    that is 6x6 -- 36 fused multiply-adds, entirely in cache -- and there is one
    thread per (world, tree), so the launch is `B * ntree` wide. A wide tree is
    better served by `wp.tile_cholesky` over a block per tree, which is how
    `mujoco_warp` does it; a tree of at least `tile_min` dofs is factored that
    way, in `factor_blocks`.

    Launch: dim=(ntree, B), a thread per tree and world, the batch fastest as in `_build_M`, by `factor_blocks`; a
    large tree's thread returns at once.
    In: M_blk (nblk, B) float32, the blocks. Out: L_blk (nblk, B) float32, the small trees' lower factors.
    """
    t, b = wp.tid()  # int, int: the tree, the world
    if m.tree_big[t] >= 0:
        return  # the tile factor in `factor_blocks`

    n = m.tree_beg[t + 1] - m.tree_beg[t]  # int: the tree's dof count
    o = m.blk_beg[t]  # int: the tree's block offset

    for r in range(n):  # r: int, a row
        for c in range(r + 1):  # c: int, a column, up to the diagonal
            s = M_blk[o + r * n + c, b]  # float: M[r, c] less the earlier columns' products
            for k in range(c):  # k: int, an earlier column
                s -= L_blk[o + r * n + k, b] * L_blk[o + c * n + k, b]

            if r == c:
                L_blk[o + r * n + r, b] = wp.sqrt(s)
            else:
                L_blk[o + r * n + c, b] = s / L_blk[o + c * n + c, b]


#######################################################################################
# Large trees: one dof per thread to build, one tile per (world, tree) to factor
#######################################################################################


@wp.kernel
def _zero_big(m: RbdModel, d: RbdData):  # type: ignore
    """
    Zeroes the large trees' blocks of `M_blk`, which `_build_M_rows` then fills on the dof-ancestor pattern.

    Launch: dim=(nbig, big_np * big_np, B), a thread per large tree, entry and world, the batch fastest as `M_blk` is
    laid out, by `build_m`.
    Writes: d.M_blk (nblk, B) float32, the large trees' blocks.
    """
    k, e, b = wp.tid()  # int, int, int: the large tree, the block entry, the world
    t = m.big_tree[k]  # int: the tree
    n = m.tree_beg[t + 1] - m.tree_beg[t]  # int: the tree's dof count
    if e < n * n:
        d.M_blk[m.blk_beg[t] + e, b] = 0.0


@wp.kernel
def _build_M_rows(m: RbdModel, d: RbdData):  # type: ignore
    """
    `_build_M`'s inner loop for one dof of a large tree: its row and column of the
    block, out to the root. Every entry is computed exactly as there.

    Race-free: entry `(i, j)` and its mirror belong to the thread of whichever of
    `i`, `j` is the descendant, since only a descendant walks up to the other.

    Launch: dim=(nbig_dof, B), a thread per large-tree dof and world, the batch fastest as `M_blk` is laid out, by
    `build_m`.
    Reads: d.crb (B, nbody) vec10, d.cdof (B, nv) wp.spatial_vector; the dofs' armature. Writes: d.M_blk (nblk, B)
    float32, the dof's row and column of its tree's block.
    """
    k, b = wp.tid()  # int, int: the dof's index in big_dof, the world
    i = m.big_dof[k]  # int: the dof
    t = m.big_dof_tree[k]  # int: its tree
    beg = m.tree_beg[t]  # int: the tree's first dof
    n = m.tree_beg[t + 1] - beg  # int: the tree's dof count
    o = m.blk_beg[t]  # int: the tree's block offset in M_blk

    f_i = lmath.inert_mul(d.crb[b, m.dof_bodyid[i]], d.cdof[b, i])  # wp.spatial_vector: crb[body(i)] cdof[i]
    r = i - beg  # int: i's row in the block
    j = i  # int: a dof on the path from i to the root
    while j > -1:
        v = wp.dot(f_i, d.cdof[b, j])  # float: M[i, j]
        c = j - beg  # int: j's column in the block
        d.M_blk[o + r * n + c, b] = v
        d.M_blk[o + c * n + r, b] = v
        j = m.dof_parentid[j]

    d.M_blk[o + r * n + r, b] += m.dof_armature[i]


@wp.kernel
def _gather_big(
    m: RbdModel,  # type: ignore
    M_blk: wp.array(dtype=float, ndim=2),  # (nblk, B) # type: ignore
    M_big: wp.array(dtype=float, ndim=3),  # (B * nbig, big_np, big_np) # type: ignore
):
    """
    A large tree's block, batch first and padded with an identity.

    Launch: dim=(B * nbig, big_np, big_np), a thread per (world, large tree) and entry, by `factor_blocks`. The batch
    is slowest so that the writes to `M_big` coalesce; `_scatter_big` is the other way round, for the same reason:
    writes are what cost.
    In: M_blk (nblk, B) float32, the blocks. Out: M_big (B * nbig, big_np, big_np) float32.
    """
    q, r, c = wp.tid()  # int, int, int: the (world, large tree) b * nbig + k, the row, the column
    b = q / m.nbig  # int: the world
    t = m.big_tree[q - b * m.nbig]  # int: the tree
    n = m.tree_beg[t + 1] - m.tree_beg[t]  # int: the tree's dof count
    v = float(0.0)  # float: the entry, the identity's in the padding
    if r < n and c < n:
        v = M_blk[m.blk_beg[t] + r * n + c, b]
    elif r == c:
        v = 1.0
    M_big[q, r, c] = v


@wp.kernel
def _scatter_big(
    m: RbdModel,  # type: ignore
    L_big: wp.array(dtype=float, ndim=3),  # (B * nbig, big_np, big_np) # type: ignore
    L_blk: wp.array(dtype=float, ndim=2),  # (nblk, B) # type: ignore
):
    """
    The tile factor back into `L_blk`, its strict upper triangle zero as `_factor_M` leaves it.

    Launch: dim=(nbig, big_np, big_np, B), a thread per large tree, entry and world, by `factor_blocks`. The batch is
    fastest so that the writes to `L_blk` coalesce.
    In: L_big (B * nbig, big_np, big_np) float32, the tile factors. Out: L_blk (nblk, B) float32, the large trees'
    blocks.
    """
    k, r, c, b = wp.tid()  # int, int, int, int: the large tree, the row, the column, the world
    q = b * m.nbig + k  # int: the (world, large tree)
    t = m.big_tree[k]  # int: the tree
    n = m.tree_beg[t + 1] - m.tree_beg[t]  # int: the tree's dof count
    if r < n and c < n:
        L_blk[m.blk_beg[t] + r * n + c, b] = wp.where(c <= r, L_big[q, r, c], 0.0)


_TILE_KERNELS = {}  # {npad: (factor, solve)}: the tile kernels built so far, per padded size (`_tile_kernels`)


def _tile_kernels(npad: int):
    """
    The tile factor and solve for one padded size, built once per size.

    Called by `factor_blocks` and `solve_m`.
    In: npad int, the padded size, `big_np`. Out: (factor, solve), the two kernels, cached in `_TILE_KERNELS`.
    """
    if npad not in _TILE_KERNELS:

        @wp.kernel(module="unique", enable_backward=False)
        def factor(M_big: wp.array(dtype=float, ndim=3), L_big: wp.array(dtype=float, ndim=3)):  # type: ignore
            """
            `L_big[q]`, the lower Cholesky factor of `M_big[q]`, by a tile.

            Launch: `wp.launch_tiled`, dim=(B * nbig,), a block of `TILE_BLOCK` threads per (world, large tree), by
            `factor_blocks`.
            In: M_big (B * nbig, big_np, big_np) float32. Out: L_big (B * nbig, big_np, big_np) float32.
            """
            q = wp.tid()  # int: the (world, large tree), b * nbig + k
            wp.tile_store(L_big[q], wp.tile_cholesky(wp.tile_load(M_big[q], shape=(npad, npad))))

        @wp.kernel(module="unique", enable_backward=False)
        def solve(L_big: wp.array(dtype=float, ndim=3), y_big: wp.array(dtype=float, ndim=2)):  # type: ignore
            """
            `y_big[q] = (L L^T)^-1 y_big[q]` in place, `L` the factor `factor` left in `L_big[q]`, by tiles.

            Launch: `wp.launch_tiled`, dim=(B * nbig,), a block of `TILE_BLOCK` threads per (world, large tree), by
            `solve_m`.
            In: L_big (B * nbig, big_np, big_np) float32. In and out: y_big (B * nbig, big_np) float32.
            """
            q = wp.tid()  # int: the (world, large tree), b * nbig + k
            L = wp.tile_load(L_big[q], shape=(npad, npad))  # tile (npad, npad) float32: the factor
            y = wp.tile_load(y_big[q], shape=(npad,))  # tile (npad,) float32: the right-hand side
            wp.tile_store(y_big[q], wp.tile_upper_solve(wp.tile_transpose(L), wp.tile_lower_solve(L, y)))

        _TILE_KERNELS[npad] = (factor, solve)
    return _TILE_KERNELS[npad]


@wp.kernel
def _gather_rhs_big(m: RbdModel, d: RbdData):  # type: ignore
    """
    A large tree's slice of `qfrc_smooth`, batch first and zero-padded: the tile solve's right-hand side.

    Launch: dim=(B * nbig, big_np), a thread per (world, large tree) and row, by `solve_m`.
    Reads: d.qfrc_smooth (B, nv) float32. Writes: d.y_big (B * nbig, big_np) float32.
    """
    q, r = wp.tid()  # int, int: the (world, large tree) b * nbig + k, the row
    b = q / m.nbig  # int: the world
    t = m.big_tree[q - b * m.nbig]  # int: the tree
    beg = m.tree_beg[t]  # int: the tree's first dof
    v = float(0.0)  # float: the entry, zero in the padding
    if r < m.tree_beg[t + 1] - beg:
        v = d.qfrc_smooth[b, beg + r]
    d.y_big[q, r] = v


@wp.kernel
def _scatter_qacc_big(m: RbdModel, d: RbdData):  # type: ignore
    """
    The tile solve's result back into `qacc_smooth`, the padding dropped.

    Launch: dim=(B * nbig, big_np), a thread per (world, large tree) and row, by `solve_m`.
    Reads: d.y_big (B * nbig, big_np) float32. Writes: d.qacc_smooth (B, nv) float32, the large trees' dofs.
    """
    q, r = wp.tid()  # int, int: the (world, large tree) b * nbig + k, the row
    b = q / m.nbig  # int: the world
    t = m.big_tree[q - b * m.nbig]  # int: the tree
    beg = m.tree_beg[t]  # int: the tree's first dof
    if r < m.tree_beg[t + 1] - beg:
        d.qacc_smooth[b, beg + r] = d.y_big[q, r]


# Threads per tile: one warp, faster than 64 or 128 on a tree padded to 32 dofs.
TILE_BLOCK = 32


@wp.func
def solve_M_tree(
    L: wp.array(dtype=float, ndim=2),  # (nblk, B) # type: ignore
    rhs: wp.array(dtype=float, ndim=2),  # (B, nv) # type: ignore
    out: wp.array(dtype=float, ndim=2),  # (B, nv) # type: ignore
    b: int,
    beg: int,
    n: int,
    o: int,
):
    """
    `out = M^-1 rhs` over one tree's dof range: forward with L, backward with L^T.

    `out` doubles as scratch: the forward pass leaves `L^-1 rhs` in it
    and the backward pass overwrites entry by entry from the bottom up, so `rhs`
    and `out` may be the same array.

    Called by `_solve_qacc_smooth` and `lodos.rbd.implicit._implicit_qacc`.
    In: L (nblk, B) float32, the lower factors; rhs (B, nv) float32; b int, the world; beg int, the tree's first dof;
    n int, its dof count; o int, its block's offset in L. Out: out (B, nv) float32, over the tree's dofs.
    """
    for i in range(n):  # i: int, a row of the tree's block
        s = rhs[b, beg + i]  # float: the row's running sum
        for j in range(i):  # j: int, an earlier row
            s -= L[o + i * n + j, b] * out[b, beg + j]
        out[b, beg + i] = s / L[o + i * n + i, b]

    for i in range(n - 1, -1, -1):
        s = out[b, beg + i]
        for j in range(i + 1, n):
            s -= L[o + j * n + i, b] * out[b, beg + j]
        out[b, beg + i] = s / L[o + i * n + i, b]


@wp.func
def mul_M_tree(
    M: wp.array(dtype=float, ndim=2),  # (nblk, B) # type: ignore
    vec: wp.array(dtype=float, ndim=2),  # (B, nv) # type: ignore
    out: wp.array(dtype=float, ndim=2),  # (B, nv) # type: ignore
    b: int,
    beg: int,
    n: int,
    o: int,
):
    """
    `out = M vec` over one tree's dof range.

    The counterpart of `solve_M_tree`, and exact for the same reason: M is block
    diagonal across trees, so a vector restricted to one tree stays there. `_build_M`
    fills both triangles of each block, so this is a plain dense product.

    `out` may not alias `vec` -- the product reads the whole row while writing entry
    by entry.

    Called by `lodos.rbd.implicit._implicit_qacc`.
    In: M (nblk, B) float32, the blocks; vec (B, nv) float32; b, beg, n, o as `solve_M_tree` takes them.
    Out: out (B, nv) float32, over the tree's dofs.
    """
    for i in range(n):  # i: int, a row of the tree's block
        s = float(0.0)  # float: the row's product with vec
        for j in range(n):  # j: int, a column
            s += M[o + i * n + j, b] * vec[b, beg + j]
        out[b, beg + i] = s


@wp.kernel
def _solve_qacc_smooth(m: RbdModel, d: RbdData):  # type: ignore
    """
    The unconstrained acceleration, `qacc_smooth = M^-1 qfrc_smooth`.

    Launch: dim=(B, ntree), a thread per world and tree, by `solve_m`; a large tree's thread returns at once. The
    batch is slowest, the opposite of `_build_M`: the batch-first `qfrc_smooth` and `qacc_smooth`, nv-strided rows
    for each thread, cost more than its contiguous run of `L_blk`.
    Reads: d.L_blk (nblk, B) float32, d.qfrc_smooth (B, nv) float32. Writes: d.qacc_smooth (B, nv) float32, the small
    trees' dofs.
    """
    b, t = wp.tid()  # int, int: the world, the tree
    if m.tree_big[t] >= 0:
        return  # the tile solve in `solve_m`

    beg = m.tree_beg[t]  # int: the tree's first dof
    solve_M_tree(
        d.L_blk, d.qfrc_smooth, d.qacc_smooth, b, beg,
        m.tree_beg[t + 1] - beg, m.blk_beg[t],
    )


def build_m(m: RbdModel, d: RbdData):  # type: ignore
    """
    Builds the mass matrix, per kinematic tree, into `M_blk`.

    Called by `factor_m` and `lodos.rbd.forward.fwd_position`.
    In: m, d, B worlds' model and state; reads d.cinert (B, nbody) vec10 and d.cdof (B, nv) wp.spatial_vector.
    Out: d.crb (B, nbody) vec10, d.M_blk (nblk, B) float32.
    """
    for level in range(m.nlevel - 1, -1, -1):  # level: int, a tree level, the deepest first
        wp.launch(_crb_acc, dim=(m.B, m.nbody), inputs=(m, d, level))
    wp.launch(_build_M, dim=(m.ntree, m.B), inputs=(m, d))
    if m.nbig:
        npad = int(m.big_np)  # int: the tile size
        wp.launch(_zero_big, dim=(m.nbig, npad * npad, m.B), inputs=(m, d))
        wp.launch(_build_M_rows, dim=(m.nbig_dof, m.B), inputs=(m, d))


def factor_blocks(m: RbdModel, d: RbdData, M_blk, L_blk):  # type: ignore
    """
    Factors the per-tree blocks `M_blk` into `L_blk`: the mass matrix, or under
    `implicitfast` the implicit one. The large trees go through `M_big` and `L_big`,
    which hold the last factorization's.

    Called by `factor_m` and `lodos.rbd.implicit.prepare`.
    In: m, d, B worlds' model and state; M_blk (nblk, B) float32, the blocks. Out: L_blk (nblk, B) float32, their lower
    factors; d.M_big and d.L_big (B * nbig, big_np, big_np) float32.
    """
    wp.launch(_factor_M, dim=(m.ntree, m.B), inputs=(m, M_blk, L_blk))
    if m.nbig:
        npad = int(m.big_np)  # int: the tile size
        wp.launch(_gather_big, dim=(m.B * m.nbig, npad, npad), inputs=(m, M_blk, d.M_big))
        factor, _ = _tile_kernels(npad)  # wp.Kernel: the tile factor; the solve unused
        wp.launch_tiled(factor, dim=[m.B * m.nbig], inputs=(d.M_big,), outputs=(d.L_big,), block_dim=TILE_BLOCK)
        wp.launch(_scatter_big, dim=(m.nbig, npad, npad, m.B), inputs=(m, d.L_big, L_blk))


def factor_m(m: RbdModel, d: RbdData):  # type: ignore
    """
    Builds and factors the mass matrix, per kinematic tree.

    Called by `lodos.rbd.forward.fwd_position`.
    In: m, d, B worlds' model and state; reads what `build_m` reads. Out: d.crb (B, nbody) vec10, d.M_blk and d.L_blk
    (nblk, B) float32, d.M_big and d.L_big (B * nbig, big_np, big_np) float32.
    """
    build_m(m, d)
    factor_blocks(m, d, d.M_blk, d.L_blk)


def solve_m(m: RbdModel, d: RbdData):  # type: ignore
    """
    `qacc_smooth = M^-1 qfrc_smooth`, from the factor `factor_m` left.

    Called by `lodos.rbd.forward.fwd_acceleration`.
    In: m, d, B worlds' model and state; reads d.qfrc_smooth (B, nv) float32, d.L_blk (nblk, B) float32 and d.L_big
    (B * nbig, big_np, big_np) float32. Out: d.qacc_smooth (B, nv) float32; d.y_big (B * nbig, big_np) float32.
    """
    wp.launch(_solve_qacc_smooth, dim=(m.B, m.ntree), inputs=(m, d))
    if m.nbig:
        npad = int(m.big_np)  # int: the tile size
        wp.launch(_gather_rhs_big, dim=(m.B * m.nbig, npad), inputs=(m, d))
        _, solve = _tile_kernels(npad)  # wp.Kernel: the tile solve; the factor unused
        wp.launch_tiled(solve, dim=[m.B * m.nbig], inputs=(d.L_big, d.y_big), block_dim=TILE_BLOCK)
        wp.launch(_scatter_qacc_big, dim=(m.B * m.nbig, npad), inputs=(m, d))


#######################################################################################
# Actuator transmission
#######################################################################################


@wp.func
def _dof_width(jtype: int):
    """
    Dofs a joint of this type owns.

    Called by `_moment`.
    In: jtype int, FREE, BALL, SLIDE or HINGE. Out: int, 6, 3 or 1.
    """
    if jtype == FREE:
        return 6
    if jtype == BALL:
        return 3
    return 1


@wp.func
def _moves(m: RbdModel, dof_body: int, body: int):  # type: ignore
    """
    Whether a dof belonging to `dof_body` moves `body` -- i.e. whether `dof_body`
    is `body` or one of its ancestors.

    The mask of the dofs that move a body, evaluated one entry at a time by walking
    the parent chain.

    Called by `_moment`.
    In: dof_body int, the dof's body; body int, the body moved. Out: bool.
    """
    i = body  # int: a body on the chain from `body` up to the world
    while i > 0:
        if i == dof_body:
            return True
        i = m.body_parentid[i]
    return dof_body == 0


@wp.func
def _site_quat(m: RbdModel, d: RbdData, b: int, s: int):  # type: ignore
    """
    The site orientation `transmission` compares against a reference site.

    Note the multiplication order: `quat_mul(site_quat, xquat)`, the reverse of
    `local_to_global`'s composition, and `site_xmat` is built the other way round.
    MuJoCo Warp's transmission (`smooth.py`) forms it in the same order -- this is
    the quantity `quat_sub` measures a spring-like length against, so the order is
    part of the model.

    Called by `_length`.
    In: b int, the world; s int, the site. Reads: d.xquat (B, nbody) wp.vec4. Out: wp.vec4, (w, x, y, z).
    """
    return lmath.quat_mul(m.site_quat[s], d.xquat[b, m.site_bodyid[s]])


@wp.kernel
def _moment(m: RbdModel, d: RbdData):  # type: ignore
    """
    `d(actuator length)/d(qvel)`, one entry per (actuator, dof).

    A joint transmission touches only its own joint's dofs, so all but a handful
    of entries are structurally zero and the thread writes zero without reading
    anything. A site transmission is dense: it is a Jacobian row, masked to the
    dofs that move the site but not its reference site.

    Launch: dim=(B, nu, nv), a thread per world, actuator and dof, by `transmission`.
    Reads: d.qpos (B, nq) float32, d.site_xpos (B, nsite) wp.vec3, d.site_xmat (B, nsite) wp.mat33, d.cdof (B, nv)
    wp.spatial_vector, d.subtree_com (B, nbody) wp.vec3; the gears and m.site_dof_mask (nu, nv) float32.
    Writes: d.actuator_moment (B, nu, nv) float32.
    """
    b, u, v = wp.tid()  # int, int, int: the world, the actuator, the dof

    trn = m.actuator_trntype[u]  # int: TRN_JOINT, TRN_JOINTINPARENT or TRN_SITE

    if trn != TRN_SITE:
        j = m.actuator_trnid[u, 0]  # int: the transmission's joint
        jtype = m.jnt_type[j]  # int: FREE, BALL, SLIDE or HINGE
        k = v - m.jnt_dofadr[j]  # int: the dof's index in the joint, out of range if not the joint's
        if k < 0 or k >= _dof_width(jtype):
            d.actuator_moment[b, u, v] = 0.0
            return

        qadr = m.jnt_qposadr[j]  # int: the joint's first qpos
        mom = m.actuator_gear[u, k]  # float: the moment

        if jtype == FREE and k >= 3 and trn == TRN_JOINTINPARENT:
            q = wp.vec4(  # wp.vec4: the joint's orientation
                d.qpos[b, qadr + 3], d.qpos[b, qadr + 4],
                d.qpos[b, qadr + 5], d.qpos[b, qadr + 6],
            )
            # wp.vec3: the gear's rotational part
            g = wp.vec3(m.actuator_gear[u, 3], m.actuator_gear[u, 4], m.actuator_gear[u, 5])
            mom = lmath.rotate(g, lmath.quat_inv(q))[k - 3]
        elif jtype == BALL and trn == TRN_JOINTINPARENT:
            q = wp.vec4(
                d.qpos[b, qadr], d.qpos[b, qadr + 1],
                d.qpos[b, qadr + 2], d.qpos[b, qadr + 3],
            )
            g = wp.vec3(m.actuator_gear[u, 0], m.actuator_gear[u, 1], m.actuator_gear[u, 2])
            mom = lmath.rotate(g, lmath.quat_inv(q))[k]

        d.actuator_moment[b, u, v] = mom
        return

    # Site transmission: a masked Jacobian row contracted with the gear wrench.
    site = m.actuator_trnid[u, 0]  # int: the site
    ref = m.actuator_trnid[u, 1]  # int: the reference site, -1 if none

    frame = d.site_xmat[b, site]  # wp.mat33: the frame the gear is in, the reference site's if there is one
    if ref != -1:
        frame = d.site_xmat[b, ref]
    gear_lin = wp.vec3(m.actuator_gear[u, 0], m.actuator_gear[u, 1], m.actuator_gear[u, 2])  # wp.vec3
    gear_ang = wp.vec3(m.actuator_gear[u, 3], m.actuator_gear[u, 4], m.actuator_gear[u, 5])  # wp.vec3
    w_lin = frame * gear_lin  # wp.vec3: the wrench's force, world axes
    w_ang = frame * gear_ang  # wp.vec3: the wrench's torque, world axes

    cdof = d.cdof[b, v]  # wp.spatial_vector: the dof's motion axis, [angular, linear]
    ang = wp.vec3(cdof[0], cdof[1], cdof[2])  # wp.vec3
    lin = wp.vec3(cdof[3], cdof[4], cdof[5])  # wp.vec3
    dof_body = m.dof_bodyid[v]  # int: the dof's body

    mom = float(0.0)
    if _moves(m, dof_body, m.site_bodyid[site]):
        # wp.vec3: the site from its tree's COM
        off = d.site_xpos[b, site] - d.subtree_com[b, m.body_rootid[m.site_bodyid[site]]]
        mom += wp.dot(lin + wp.cross(ang, off), w_lin) + wp.dot(ang, w_ang)

    if ref != -1 and _moves(m, dof_body, m.site_bodyid[ref]):
        off = d.site_xpos[b, ref] - d.subtree_com[b, m.body_rootid[m.site_bodyid[ref]]]
        mom -= wp.dot(lin + wp.cross(ang, off), w_lin) + wp.dot(ang, w_ang)

    d.actuator_moment[b, u, v] = mom * m.site_dof_mask[u, v]


@wp.kernel
def _length(m: RbdModel, d: RbdData):  # type: ignore
    """
    Actuator length. Zero for a free joint and for a site transmission with no
    reference site -- neither has a scalar coordinate to measure.

    Launch: dim=(B, nu), a thread per world and actuator, by `transmission`.
    Reads: d.qpos (B, nq) float32, d.site_xpos (B, nsite) wp.vec3, d.site_xmat (B, nsite) wp.mat33, d.xquat (B, nbody)
    wp.vec4. Writes: d.actuator_length (B, nu) float32.
    """
    b, u = wp.tid()  # int, int: the world, the actuator

    trn = m.actuator_trntype[u]  # int: TRN_JOINT, TRN_JOINTINPARENT or TRN_SITE
    length = float(0.0)  # float: the actuator's length

    if trn != TRN_SITE:
        j = m.actuator_trnid[u, 0]  # int: the transmission's joint
        jtype = m.jnt_type[j]  # int: FREE, BALL, SLIDE or HINGE
        qadr = m.jnt_qposadr[j]  # int: the joint's first qpos

        if jtype == BALL:
            q = wp.vec4(  # wp.vec4: the joint's orientation
                d.qpos[b, qadr], d.qpos[b, qadr + 1],
                d.qpos[b, qadr + 2], d.qpos[b, qadr + 3],
            )
            axis, angle = lmath.quat_to_axis_angle(q)  # wp.vec3, float: its rotation's axis and angle (rad)
            # wp.vec3: the gear's rotational part
            g = wp.vec3(m.actuator_gear[u, 0], m.actuator_gear[u, 1], m.actuator_gear[u, 2])
            if trn == TRN_JOINTINPARENT:
                g = lmath.rotate(g, lmath.quat_inv(q))
            length = wp.dot(axis * angle, g)
        elif jtype != FREE:
            length = d.qpos[b, qadr] * m.actuator_gear[u, 0]
    else:
        ref = m.actuator_trnid[u, 1]  # int: the reference site, -1 if none
        if ref != -1:
            site = m.actuator_trnid[u, 0]  # int: the site
            vp = wp.transpose(d.site_xmat[b, ref]) * (  # wp.vec3: the site's offset, in the reference site's frame
                d.site_xpos[b, site] - d.site_xpos[b, ref]
            )
            vr = lmath.quat_sub(_site_quat(m, d, b, site), _site_quat(m, d, b, ref))  # wp.vec3: the relative rotation
            for c in range(3):  # c: int, a component
                length += vp[c] * m.actuator_gear[u, c]
                length += vr[c] * m.actuator_gear[u, c + 3]

    d.actuator_length[b, u] = length


@wp.kernel
def _actuator_velocity(d: RbdData):  # type: ignore
    """
    `actuator_moment @ qvel`, the rate each actuator's length changes at.

    Launch: dim=(B, nu), a thread per world and actuator, by `actuator_velocity`.
    Reads: d.actuator_moment (B, nu, nv) float32, d.qvel (B, nv) float32. Writes: d.actuator_velocity (B, nu) float32.
    """
    b, u = wp.tid()  # int, int: the world, the actuator

    vel = float(0.0)  # float: the actuator's velocity
    for v in range(d.qvel.shape[1]):  # v: int, a dof
        vel += d.actuator_moment[b, u, v] * d.qvel[b, v]

    d.actuator_velocity[b, u] = vel


def transmission(m: RbdModel, d: RbdData):  # type: ignore
    """
    Actuator lengths and moment arms. A no-op for a model with no actuators.

    Called by `lodos.rbd.forward.fwd_position`.
    In: m, d, B worlds' model and state; reads d.qpos (B, nq) float32, the site poses, d.xquat, d.cdof and
    d.subtree_com. Out: d.actuator_length (B, nu) and d.actuator_moment (B, nu, nv) float32.
    """
    if not m.nu:
        return

    wp.launch(_length, dim=(m.B, m.nu), inputs=(m, d))
    wp.launch(_moment, dim=(m.B, m.nu, m.nv), inputs=(m, d))


def actuator_velocity(m: RbdModel, d: RbdData):  # type: ignore
    """
    Actuator velocities, once `qvel` and the moment arms are known.

    Called by `lodos.rbd.forward.fwd_velocity`.
    In: m, d, B worlds' model and state; reads d.actuator_moment (B, nu, nv) and d.qvel (B, nv) float32.
    Out: d.actuator_velocity (B, nu) float32.
    """
    if m.nu:
        wp.launch(_actuator_velocity, dim=(m.B, m.nu), inputs=(d,))
