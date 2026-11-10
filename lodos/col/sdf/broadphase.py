import warp as wp

from lodos.col.sdf.compact import resolve_nCa1_tiled, resolve_nCa1_partition, resolve_nTb
from lodos.col.sdf.types import (
    CollisionModel,
    CollisionData,
    idm_nB_nx_ny_nz_to_nVox_vox,
)

#######################################################################################
# Pose Preparation
#######################################################################################


@wp.kernel
def compute_T_w(mcol: CollisionModel, dcol: CollisionData):
    """
    World pose of each collision geometry, gathered from the body poses.

    It is a gather and a component reorder, so it belongs on the device:
    `xpos`/`xquat` come straight out of `Data` and nothing has to round-trip
    through the host.

    MuJoCo stores quaternions as (w, x, y, z) and `wp.transformf` reads them as
    (x, y, z, w). Getting that wrong does not fail loudly -- every
    `wp.transform_point` downstream simply rotates by a scrambled quaternion -- so
    the reorder lives here, at the one boundary where the convention changes.

    Launch: dim=(B, nB), a thread per world and geometry, by `collide.broadphase_bbox`.
    Reads: dcol.xpos (B, nbody) wp.vec3, dcol.xquat (B, nbody) wp.vec4 in (w, x, y, z); mcol.g_bid (nB,) int32.
    Writes: dcol.T_w (B, nB) wp.transformf.
    """
    b, g = wp.tid()  # int, int: the world, the geometry

    bid = mcol.g_bid[g]  # int: the geometry's body
    q = dcol.xquat[b, bid]  # wp.vec4: the body's orientation, (w, x, y, z)

    dcol.T_w[b, g] = wp.transformf(
        dcol.xpos[b, bid], wp.quat(q[1], q[2], q[3], q[0])
    )


@wp.kernel
def compute_T_rel(mcol: CollisionModel, dcol: CollisionData):
    """
    Relative pose taking the vertex geom's body frame to the SDF geom's body frame.

    Contacts are then evaluated straight from the static `mcol.v_b` (a few kB,
    L1-resident) instead of from a world-space vertex buffer of size (B, nV, 3),
    which removes both the per-contact pose inversion and tens of MB of gather
    traffic per step.

    Launched per *pair*, not per segment: the composition runs once a pair and
    is then materialized for each of the pair's segments. The two stages that
    consume it -- each running over millions of contact candidates -- therefore
    index a transform directly and need to know nothing about segments.

    A symmetric pair is the one case whose segments do not all share a frame: its
    two directions swap the vertex and SDF roles, so direction 1 takes the
    *inverse* of direction 0's transform. Inverted rather than recomposed from
    the world poses, which is both cheaper and exact -- the two are only equal up
    to rounding.

    Launch: dim=(B, nCPR), a thread per world and pair, by `collide.broadphase_bbox`.
    Reads: dcol.T_w (B, nB) wp.transformf; mcol.cprs (nCPR,) wp.vec2i, cpr_seg_beg (nCPR+1,) int32 and seg_dir
    (nSEG,) int32. Writes: dcol.T_rel (B, nSEG) wp.transformf.
    """
    b, j = wp.tid()  # int, int: the world, the pair

    cpr = mcol.cprs[j]  # wp.vec2i: the pair's two geometries
    T_1 = dcol.T_w[b, cpr[0]]  # wp.transformf: the first geometry's world pose
    T_2 = dcol.T_w[b, cpr[1]]  # wp.transformf: the second's

    T = wp.transform_inverse(T_2) * T_1  # wp.transformf: the first geometry's body frame to the second's
    T_inv = wp.transform_inverse(T)  # wp.transformf: the second's to the first's

    for s in range(mcol.cpr_seg_beg[j], mcol.cpr_seg_beg[j + 1]):  # s: int, a segment of the pair
        if mcol.seg_dir[s] == 0:
            dcol.T_rel[b, s] = T
        else:
            dcol.T_rel[b, s] = T_inv


#######################################################################################
# Bounding-Box Based Filtering
#######################################################################################


@wp.func
def obb_axis_radius(A: wp.mat33, h: wp.vec3, L: wp.vec3):
    """
    Projection radius of an OBB onto axis L: sum_i |L . A_i| * h_i.

    Called by `obb_separating`.
    In: A wp.mat33, the box's axes as rows; h wp.vec3, its half-extents (m); L wp.vec3, the axis. Out: float.
    """
    return (
        wp.abs(L[0] * A[0, 0] + L[1] * A[0, 1] + L[2] * A[0, 2]) * h[0]
        + wp.abs(L[0] * A[1, 0] + L[1] * A[1, 1] + L[2] * A[1, 2]) * h[1]
        + wp.abs(L[0] * A[2, 0] + L[1] * A[2, 1] + L[2] * A[2, 2]) * h[2]
    )


@wp.func
def obb_separating(
    Aa: wp.mat33, ha: wp.vec3, Ab: wp.mat33, hb: wp.vec3, t: wp.vec3, L: wp.vec3
):
    """
    True iff L is a separating axis. Degenerate axes (|L|^2 ~ 0) never separate.

    Called by `obb_filter`.
    In: Aa, Ab wp.mat33, the two boxes' world axes as rows; ha, hb wp.vec3, their half-extents (m); t wp.vec3,
    the centre of b less the centre of a; L wp.vec3, the axis. Out: bool.
    """
    if wp.dot(L, L) <= 1.0e-12:
        return False

    ra = obb_axis_radius(Aa, ha, L)  # float: box a's projection radius
    rb = obb_axis_radius(Ab, hb, L)  # float: box b's

    return wp.abs(wp.dot(L, t)) > ra + rb


@wp.kernel
def obb_filter(mcol: CollisionModel, dcol: CollisionData):
    """
    15-axis SAT between the two OBBs of each collision pair, kept entirely in
    registers. The world-space boxes are derived from the static body-frame OBBs
    and the body poses, so no (B, nB, 8, 3) corner array is ever materialized.

    A half-space SDF geometry takes a separate path: it has no finite box, so the
    test becomes "does the vertex geometry's box reach below the plane", which is
    both exact and much cheaper than the SAT.

    Launch: dim=(B, nCPR), a thread per world and pair, by `collide.broadphase_bbox`.
    Reads: dcol.T_w (B, nB) wp.transformf; mcol.cprs (nCPR,) wp.vec2i, g_plane (nB,) int32, g_obb_c and g_obb_h
    (nB,) wp.vec3, g_obb_A (nB,) wp.mat33. Writes: dcol.cpr_active (B, nCPR) bool.
    """
    b, j = wp.tid()  # int, int: the world, the pair

    cpr = mcol.cprs[j]  # wp.vec2i
    ia, ib = cpr[0], cpr[1]  # int, int: the vertex geometry, the SDF geometry

    if mcol.g_plane[ib] == 1:
        # Work in the plane's own frame, where the half-space is {z < 0}. The box's
        # lowest point is its centre less the sum of its half-extents projected onto
        # the plane normal.
        # Composed here rather than read out of `T_rel`, which keeps this kernel
        # independent of `compute_T_rel` -- at the cost of one transform compose
        # in a (B, nCPR) launch, taken only by pairs against a half-space.
        T_rel = wp.transform_inverse(dcol.T_w[b, ib]) * dcol.T_w[b, ia]  # wp.transformf: a's frame to the plane's
        c = wp.transform_point(T_rel, mcol.g_obb_c[ia])  # wp.vec3: a's box centre, in the plane's frame
        q_rel = wp.transform_get_rotation(T_rel)  # wp.quat
        A, h = mcol.g_obb_A[ia], mcol.g_obb_h[ia]  # wp.mat33, wp.vec3: a's box axes (rows), half-extents (m)

        reach = float(0.0)  # float: the box's half-extent along the plane normal (m)
        for i in range(3):  # i: int, a box axis
            axis = wp.quat_rotate(q_rel, wp.vec3(A[i, 0], A[i, 1], A[i, 2]))  # wp.vec3: axis i in the plane's frame
            reach += h[i] * wp.abs(axis[2])

        dcol.cpr_active[b, j] = c[2] - reach < 0.0
        return

    T_a, T_b = dcol.T_w[b, ia], dcol.T_w[b, ib]  # wp.transformf, wp.transformf: the two world poses
    q_a, q_b = wp.transform_get_rotation(T_a), wp.transform_get_rotation(T_b)  # wp.quat, wp.quat

    ca = wp.transform_point(T_a, mcol.g_obb_c[ia])  # wp.vec3: a's box centre, in world
    cb = wp.transform_point(T_b, mcol.g_obb_c[ib])  # wp.vec3: b's
    ha, hb = mcol.g_obb_h[ia], mcol.g_obb_h[ib]  # wp.vec3, wp.vec3: the half-extents (m)

    # Bounding spheres first, which is what MuJoCo Warp's default prefilter mask
    # does before its own box test. The sphere about an OBB's centre with the
    # half-diagonal as radius contains it, so a rejection here is exact, and it
    # costs one subtraction and two lengths against the fifteen axes below. In a
    # scene of many bodies almost every pair is far apart, so almost every pair
    # stops here.
    d2 = wp.length_sq(cb - ca)  # float: the squared centre distance (m^2)
    rr = wp.length(ha) + wp.length(hb)  # float: the two bounding spheres' radii, summed (m)
    if d2 > rr * rr:
        dcol.cpr_active[b, j] = False
        return

    Ab_a, Ab_b = mcol.g_obb_A[ia], mcol.g_obb_A[ib]  # wp.mat33, wp.mat33: the body-frame axes, as rows

    # Rotate the body-frame axes (stored as rows) into world.
    a0 = wp.quat_rotate(q_a, wp.vec3(Ab_a[0, 0], Ab_a[0, 1], Ab_a[0, 2]))  # wp.vec3
    a1 = wp.quat_rotate(q_a, wp.vec3(Ab_a[1, 0], Ab_a[1, 1], Ab_a[1, 2]))  # wp.vec3
    a2 = wp.quat_rotate(q_a, wp.vec3(Ab_a[2, 0], Ab_a[2, 1], Ab_a[2, 2]))  # wp.vec3
    b0 = wp.quat_rotate(q_b, wp.vec3(Ab_b[0, 0], Ab_b[0, 1], Ab_b[0, 2]))  # wp.vec3
    b1 = wp.quat_rotate(q_b, wp.vec3(Ab_b[1, 0], Ab_b[1, 1], Ab_b[1, 2]))  # wp.vec3
    b2 = wp.quat_rotate(q_b, wp.vec3(Ab_b[2, 0], Ab_b[2, 1], Ab_b[2, 2]))  # wp.vec3

    Aa = wp.mat33(a0[0], a0[1], a0[2], a1[0], a1[1], a1[2], a2[0], a2[1], a2[2])  # wp.mat33: a's world axes, as rows
    Ab = wp.mat33(b0[0], b0[1], b0[2], b1[0], b1[1], b1[2], b2[0], b2[1], b2[2])  # wp.mat33: b's

    t = cb - ca  # wp.vec3: b's centre less a's (m)

    sep = False  # bool: whether some axis separates the boxes
    # 3 face normals of A, 3 of B, then the 9 edge-edge cross products.
    for i in range(3):
        La = wp.vec3(Aa[i, 0], Aa[i, 1], Aa[i, 2])  # wp.vec3: a face normal of a
        if obb_separating(Aa, ha, Ab, hb, t, La):
            sep = True
        Lb = wp.vec3(Ab[i, 0], Ab[i, 1], Ab[i, 2])  # wp.vec3: a face normal of b
        if obb_separating(Aa, ha, Ab, hb, t, Lb):
            sep = True

    for i in range(3):
        La = wp.vec3(Aa[i, 0], Aa[i, 1], Aa[i, 2])
        for k in range(3):  # k: int, an axis of b
            Lb = wp.vec3(Ab[k, 0], Ab[k, 1], Ab[k, 2])
            if obb_separating(Aa, ha, Ab, hb, t, wp.cross(La, Lb)):
                sep = True

    dcol.cpr_active[b, j] = not sep


#######################################################################################
# Time of arrival (`toa` pairs)
#######################################################################################
#
# A `toa` pair's tests keep, besides what is within reach now, whatever its relative
# motion could bring within reach over the horizon `mcol.toa_h`. Every test is a distance
# bound on a 1-Lipschitz field, so a margin equal to the farthest any point of the tested
# sphere can travel (a path length, which rotation does not lengthen) keeps it
# conservative. The kernels below are the `toa` variants of the ones above and below;
# a context without a `toa` pair never launches them, and in them a pair or segment that
# is not `toa` gets a margin of exactly zero, so it is decided as before.


@wp.func
def body_acc(b: int, i: int, mcol: CollisionModel, dcol: CollisionData):  # type: ignore
    """
    Body i's acceleration under the free (contact-free) motion, `J_i qacc_smooth`, as a spatial
    vector [angular, linear] about its tree root's subtree COM, as `cvel` is: every `cdof` of a
    tree is expressed there, so a body's motion is the sum over the dofs from it to its root.

    Called by `compute_twist_rel`.
    In: b int, the world; i int, the body; mcol, read for body_parentid, body_dofadr and body_dofnum (nbody,);
    dcol, read for cdof (B, nv) wp.spatial_vector and qacc_smooth (B, nv) float32. Out: wp.spatial_vector.
    """
    acc = wp.spatial_vector()  # wp.spatial_vector: the sum so far
    k = int(i)  # int: the body whose dofs are added, walking to the root
    while k > 0:
        adr = mcol.body_dofadr[k]  # int: its first dof (-1 without)
        for d in range(adr, adr + mcol.body_dofnum[k]):  # d: int, a dof of the body
            acc += dcol.cdof[b, d] * dcol.qacc_smooth[b, d]
        k = mcol.body_parentid[k]
    return acc


@wp.func
def twist_in_b(va: wp.spatial_vector, vb: wp.spatial_vector, pa: wp.vec3, pb: wp.vec3, T_b: wp.transformf):  # type: ignore
    """
    Body A's motion relative to body B, in B's frame: see `compute_twist_rel`.

    Called by `compute_twist_rel`.
    In: va, vb wp.spatial_vector, the bodies' [angular, linear] velocities about pa and pb (wp.vec3, world, m);
    T_b wp.transformf, B's pose. Out: wp.spatial_vector, (Omega, a) in B's frame.
    """
    wa, la = wp.spatial_top(va), wp.spatial_bottom(va)  # wp.vec3, wp.vec3
    wb, lb = wp.spatial_top(vb), wp.spatial_bottom(vb)  # wp.vec3, wp.vec3
    W = wa - wb  # wp.vec3: the relative angular velocity, world frame (rad/s)
    a_w = la - lb - wp.cross(wa, pa) + wp.cross(wb, pb)  # wp.vec3: the relative velocity at the world origin
    q_inv = wp.quat_inverse(wp.transform_get_rotation(T_b))  # wp.quat: world to B's frame
    omega = wp.quat_rotate(q_inv, W)  # wp.vec3: Omega, B's frame
    a = wp.quat_rotate(q_inv, a_w + wp.cross(W, wp.transform_get_translation(T_b)))  # wp.vec3: at B's origin
    return wp.spatial_vector(omega, a)


@wp.kernel
def compute_twist_rel(mcol: CollisionModel, dcol: CollisionData):
    """
    A `toa` segment's relative motion in its SDF geometry's body frame, now and at the end of
    the horizon under the free motion.

    The vertex geometry A's material point at world x moves at `v_A + w_A x (x - p_A)`, with
    (w_A, v_A) its body's `cvel`, taken about its tree root's subtree COM p_A; the SDF
    geometry B's at `v_B + w_B x (x - p_B)`. Their difference is `a_w + W x x`, with
    `W = w_A - w_B` and `a_w = v_A - v_B - w_A x p_A + w_B x p_B`. In B's body frame
    (rotation R, origin x_B), at `y = R^T (x - x_B)`, it is `a + Omega x y` with
    `Omega = R^T W` and `a = R^T (a_w + W x x_B)`, stored as the spatial vector (Omega, a).
    The end twist is the same with each body's `cvel + toa_h J qacc_smooth` (`body_acc`), at the
    same poses: the step's own acceleration (gravity, actuation), though not its contact forces.

    Launch: dim=(B, nSEG), a thread per world and segment, by `collide.broadphase_bbox` in a
    context with a `toa` pair; a segment that is not `toa` is left as it is.
    Reads: dcol.cvel (B, nbody) wp.spatial_vector, dcol.subtree_com (B, nbody) wp.vec3, dcol.T_w
    (B, nB) wp.transformf, dcol.cdof and qacc_smooth (`body_acc`); mcol.seg_toa, seg_g_v, seg_g_s (nSEG,),
    g_bid, g_root (nB,) int and toa_h. Writes: dcol.twist_rel and twist_end (B, nSEG) wp.spatial_vector.
    """
    b, s = wp.tid()  # int, int: the world, the segment
    if mcol.seg_toa[s] == 0:
        return
    ga = mcol.seg_g_v[s]  # int: the vertex geometry
    gb = mcol.seg_g_s[s]  # int: the SDF geometry
    ia = mcol.g_bid[ga]  # int: A's body
    ib = mcol.g_bid[gb]  # int: B's body
    va = dcol.cvel[b, ia]  # wp.spatial_vector: A's body velocity, [angular, linear]
    vb = dcol.cvel[b, ib]  # wp.spatial_vector: B's
    pa = dcol.subtree_com[b, mcol.g_root[ga]]  # wp.vec3: the point A's linear velocity is taken at
    pb = dcol.subtree_com[b, mcol.g_root[gb]]  # wp.vec3: B's
    T_b = dcol.T_w[b, gb]  # wp.transformf: B's body pose
    dcol.twist_rel[b, s] = twist_in_b(va, vb, pa, pb, T_b)
    h = mcol.toa_h  # float: the horizon (s)
    dcol.twist_end[b, s] = twist_in_b(va + h * body_acc(b, ia, mcol, dcol), vb + h * body_acc(b, ib, mcol, dcol),
                                      pa, pb, T_b)


@wp.func
def toa_margin(b: int, s: int, y: wp.vec3, R: float, mcol: CollisionModel, dcol: CollisionData):  # type: ignore
    """
    How far segment s's relative motion can carry any point of the sphere (y, R) over the
    horizon: `(|a + Omega x y| + |Omega| R) toa_h` -- a point of the sphere moves at most
    |Omega| R faster than its centre -- with the larger of the speeds now (`twist_rel`) and at
    the horizon's end under the free motion (`twist_end`): under a constant acceleration a
    point's speed is convex in time, so its largest is at one end. Zero for a segment that is
    not `toa`.

    Called by `obb_filter_toa`, `sphere_survives_toa`, the `toa` voxel prefilters and
    `narrowphase`'s `toa` kernels.
    In: b int, the world; s int, the segment; y wp.vec3, the sphere's centre in the SDF geometry's
    body frame (m); R float, its radius (m); mcol, read for seg_toa and toa_h; dcol, read for
    twist_rel and twist_end. Out: float, the distance (m).
    """
    if mcol.seg_toa[s] == 0:
        return float(0.0)
    tw = dcol.twist_rel[b, s]  # wp.spatial_vector: (Omega, a) now
    te = dcol.twist_end[b, s]  # wp.spatial_vector: at the horizon's end
    o0 = wp.spatial_top(tw)  # wp.vec3
    o1 = wp.spatial_top(te)  # wp.vec3
    u0 = wp.length(wp.spatial_bottom(tw) + wp.cross(o0, y)) + wp.length(o0) * R  # float: the speed bound now (m/s)
    u1 = wp.length(wp.spatial_bottom(te) + wp.cross(o1, y)) + wp.length(o1) * R  # float: at the end
    return wp.max(u0, u1) * mcol.toa_h


@wp.func
def obb_separating_m(Aa: wp.mat33, ha: wp.vec3, Ab: wp.mat33, hb: wp.vec3, t: wp.vec3, L: wp.vec3, m: float):  # type: ignore
    """
    `obb_separating` with the boxes apart by more than m along L: `|L.t| > ra + rb + m |L|`
    (L is not normalized, so the margin is scaled with it).

    Called by `obb_filter_toa`.
    In: as `obb_separating`; m float, the margin (m). Out: bool.
    """
    if wp.dot(L, L) <= 1.0e-12:
        return False
    ra = obb_axis_radius(Aa, ha, L)  # float: box a's projection radius
    rb = obb_axis_radius(Ab, hb, L)  # float: box b's
    return wp.abs(wp.dot(L, t)) > ra + rb + m * wp.length(L)


@wp.kernel
def obb_filter_toa(mcol: CollisionModel, dcol: CollisionData):
    """
    `obb_filter` for a context with a `toa` pair: a `toa` pair stays active while its boxes are
    within the distance its motion can close over the horizon, the largest of its segments'
    `toa_margin` over its vertex geometry's box sphere; any other pair has margin 0 and is
    tested exactly as `obb_filter` tests it.

    Launch: dim=(B, nCPR), by `collide.broadphase_bbox`, after `compute_T_rel` and
    `compute_twist_rel`. Reads and writes: as `obb_filter`; also mcol.cpr_toa, cpr_seg_beg,
    seg_g_v, dcol.T_rel and twist_rel.
    """
    b, j = wp.tid()  # int, int: the world, the pair
    cpr = mcol.cprs[j]  # wp.vec2i
    ia, ib = cpr[0], cpr[1]  # int, int: the vertex geometry, the SDF geometry
    m = float(0.0)  # float: the distance the pair's motion can close over the horizon (m)
    if mcol.cpr_toa[j] == 1:
        for s in range(mcol.cpr_seg_beg[j], mcol.cpr_seg_beg[j + 1]):  # s: int, a direction of the pair
            gv = mcol.seg_g_v[s]  # int: its vertex geometry
            y = wp.transform_point(dcol.T_rel[b, s], mcol.g_obb_c[gv])  # wp.vec3: its box centre, SDF frame
            m = wp.max(m, toa_margin(b, s, y, wp.length(mcol.g_obb_h[gv]), mcol, dcol))
    if mcol.g_plane[ib] == 1:
        T_rel = wp.transform_inverse(dcol.T_w[b, ib]) * dcol.T_w[b, ia]  # wp.transformf: a's frame to the plane's
        c = wp.transform_point(T_rel, mcol.g_obb_c[ia])  # wp.vec3: a's box centre, in the plane's frame
        q_rel = wp.transform_get_rotation(T_rel)  # wp.quat
        A, h = mcol.g_obb_A[ia], mcol.g_obb_h[ia]  # wp.mat33, wp.vec3: a's box axes (rows), half-extents (m)
        reach = float(0.0)  # float: the box's half-extent along the plane normal (m)
        for i in range(3):  # i: int, a box axis
            axis = wp.quat_rotate(q_rel, wp.vec3(A[i, 0], A[i, 1], A[i, 2]))  # wp.vec3: axis i in the plane's frame
            reach += h[i] * wp.abs(axis[2])
        dcol.cpr_active[b, j] = c[2] - reach < m
        return
    T_a, T_b = dcol.T_w[b, ia], dcol.T_w[b, ib]  # wp.transformf, wp.transformf: the two world poses
    q_a, q_b = wp.transform_get_rotation(T_a), wp.transform_get_rotation(T_b)  # wp.quat, wp.quat
    ca = wp.transform_point(T_a, mcol.g_obb_c[ia])  # wp.vec3: a's box centre, in world
    cb = wp.transform_point(T_b, mcol.g_obb_c[ib])  # wp.vec3: b's
    ha, hb = mcol.g_obb_h[ia], mcol.g_obb_h[ib]  # wp.vec3, wp.vec3: the half-extents (m)
    d2 = wp.length_sq(cb - ca)  # float: the squared centre distance (m^2)
    rr = wp.length(ha) + wp.length(hb) + m  # float: the two bounding spheres' radii and the margin, summed (m)
    if d2 > rr * rr:
        dcol.cpr_active[b, j] = False
        return
    Ab_a, Ab_b = mcol.g_obb_A[ia], mcol.g_obb_A[ib]  # wp.mat33, wp.mat33: the body-frame axes, as rows
    a0 = wp.quat_rotate(q_a, wp.vec3(Ab_a[0, 0], Ab_a[0, 1], Ab_a[0, 2]))  # wp.vec3
    a1 = wp.quat_rotate(q_a, wp.vec3(Ab_a[1, 0], Ab_a[1, 1], Ab_a[1, 2]))  # wp.vec3
    a2 = wp.quat_rotate(q_a, wp.vec3(Ab_a[2, 0], Ab_a[2, 1], Ab_a[2, 2]))  # wp.vec3
    b0 = wp.quat_rotate(q_b, wp.vec3(Ab_b[0, 0], Ab_b[0, 1], Ab_b[0, 2]))  # wp.vec3
    b1 = wp.quat_rotate(q_b, wp.vec3(Ab_b[1, 0], Ab_b[1, 1], Ab_b[1, 2]))  # wp.vec3
    b2 = wp.quat_rotate(q_b, wp.vec3(Ab_b[2, 0], Ab_b[2, 1], Ab_b[2, 2]))  # wp.vec3
    Aa = wp.mat33(a0[0], a0[1], a0[2], a1[0], a1[1], a1[2], a2[0], a2[1], a2[2])  # wp.mat33: a's world axes, as rows
    Ab = wp.mat33(b0[0], b0[1], b0[2], b1[0], b1[1], b1[2], b2[0], b2[1], b2[2])  # wp.mat33: b's
    t = cb - ca  # wp.vec3: b's centre less a's (m)
    sep = False  # bool: whether some axis separates the boxes by more than m
    for i in range(3):
        La = wp.vec3(Aa[i, 0], Aa[i, 1], Aa[i, 2])  # wp.vec3: a face normal of a
        if obb_separating_m(Aa, ha, Ab, hb, t, La, m):
            sep = True
        Lb = wp.vec3(Ab[i, 0], Ab[i, 1], Ab[i, 2])  # wp.vec3: a face normal of b
        if obb_separating_m(Aa, ha, Ab, hb, t, Lb, m):
            sep = True
    for i in range(3):
        La = wp.vec3(Aa[i, 0], Aa[i, 1], Aa[i, 2])
        for k in range(3):  # k: int, an axis of b
            Lb = wp.vec3(Ab[k, 0], Ab[k, 1], Ab[k, 2])
            if obb_separating_m(Aa, ha, Ab, hb, t, wp.cross(La, Lb), m):
                sep = True
    dcol.cpr_active[b, j] = not sep


#######################################################################################
# SDF-Based Filtering
#######################################################################################


@wp.func
def eval_sdf_vox(p_b: wp.vec3, B_id: int, mcol: CollisionModel):  # type: ignore
    """
    The voxel-centre distance sample nearest `p_b`, for the prefilter.

    **No `narrowphase.grid_excess`.** Outside the grid this extends the
    boundary voxel as a constant, which *under*-estimates the true
    distance -- and a filter that under-estimates only ever admits too much,
    which costs work. Adding the distance to the box would give the sum
    `sdf(clamp(p)) + ||p - clamp(p)||`, an *upper* bound by the triangle
    inequality, and a filter that over-estimates a distance drops real contacts.
    The narrowphase wants the accurate value and takes the sum; this wants the
    conservative one and does not.

    The margin this feeds is worked out in `sphere_survives`, and getting it
    wrong is a correctness question rather than a performance one.

    Called by `sphere_survives` and `_tiled_sdf_filter_one`.
    In: p_b wp.vec3, a point in the SDF geometry's body frame (m); B_id int, the geometry; mcol, read for
    g_Vox_pmin and g_Vox_h (nB,) wp.vec3, g_Vox_N (nB,) wp.vec3i, g_nVox_cumsum (nB+1,) int32 and vox_c_sdf_vox
    (nVox,) float32. Out: float, the distance (m).
    """
    p_min = mcol.g_Vox_pmin[B_id]  # wp.vec3: the grid's lower corner
    N = mcol.g_Vox_N[B_id]  # wp.vec3i: voxels per axis
    h = mcol.g_Vox_h[B_id]  # wp.vec3: voxel edge lengths (m)

    p_b = p_b - p_min
    x, y, z = p_b[0] / h[0], p_b[1] / h[1], p_b[2] / h[2]  # float, float, float: the point in voxels

    ix, iy, iz = (  # int, int, int: the voxel holding it, clamped into the grid
        wp.clamp(wp.int32(wp.floor(x)), 0, N[0] - 1),
        wp.clamp(wp.int32(wp.floor(y)), 0, N[1] - 1),
        wp.clamp(wp.int32(wp.floor(z)), 0, N[2] - 1),
    )

    id = idm_nB_nx_ny_nz_to_nVox_vox(B_id, ix, iy, iz, mcol)  # int: the voxel's entry in vox_c_sdf_vox

    return mcol.vox_c_sdf_vox[id]


#######################################################################################
# The two-level broadphase
#######################################################################################


@wp.func
def sphere_survives(  # type: ignore
    b: int,
    s: int,
    c_b: wp.vec3,
    r: float,
    mcol: CollisionModel,
    dcol: CollisionData,
):
    """Can a sphere bounding some of segment `s`'s primitives hold a survivor?

    **Why it is conservative.** `eval_sdf_vox` samples voxel centres, so it is
    within the circumsphere radius `r` of the true distance, and the primitive
    filter `sdf_vox(v) < r` therefore implies `sdf(v) < 2r`. A signed distance
    field is 1-Lipschitz, so for a primitive within `R` of the sphere's centre
    `c`, `sdf(v) >= sdf(c) - R >= sdf_vox(c) - r - R`. None of them can pass
    unless `sdf_vox(c) < R + 3r`. Getting that margin wrong is not a performance
    bug -- it silently drops contacts.

    The derivation never mentions how many primitives the sphere covers, which
    is what lets both levels of the broadphase share it: a super-tile is bounded
    from its primitives exactly as a tile is, so it *is* the tile of size
    `seg_K * super_K` and this is the same test on a bigger group.

    Called by `tile_survives` and `gen_weights`.
    In: b int, the world; s int, the segment; c_b wp.vec3, the sphere's centre in the vertex geometry's body frame
    (m); r float, its radius (m); mcol, read for seg_g_s (nSEG,) int32, g_plane (nB,) int32, g_Vox_r (nB,) float32
    and the vox table; dcol, read for T_rel (B, nSEG) wp.transformf. Out: int, 1 if the sphere can hold a
    survivor, else 0.
    """
    c = wp.transform_point(dcol.T_rel[b, s], c_b)  # wp.vec3: the centre, in the SDF geometry's body frame
    id_nB_2 = mcol.seg_g_s[s]  # int: the SDF geometry

    if mcol.g_plane[id_nB_2] == 1:
        # A half-space is exact: the SDF is the height in its own frame, so the
        # sphere reaches it iff its lowest point does.
        if c[2] - r < 0.0:
            return 1
        return 0

    if eval_sdf_vox(c, id_nB_2, mcol) < r + 3.0 * mcol.g_Vox_r[id_nB_2]:
        return 1
    return 0


@wp.func
def tile_survives(b: int, s: int, j: int, mcol: CollisionModel, dcol: CollisionData):  # type: ignore
    """
    Can dense tile `j` of segment `s` hold a candidate the primitive filter keeps?

    Called by `tile_filter` and `tile_filter_seg`.
    In: b int, the world; s int, the segment; j int, the dense tile; mcol, read for tile_c (nT_world,) wp.vec3 and
    tile_r (nT_world,) float32; dcol, as `sphere_survives`. Out: int, 1 or 0.
    """
    return sphere_survives(b, s, mcol.tile_c[j], mcol.tile_r[j], mcol, dcol)


@wp.kernel
def gen_weights(mcol: CollisionModel, dcol: CollisionData):
    """Per (batch, generator): how many tiles it contributes to the filter's launch.

    This is the level above the tiles, and the only place a super-tile is ever
    tested. Two kinds of generator meet here (`CollisionModel.gen_seg`):

    - a **super-tile** contributes its children only if its own bounding sphere
      survives, which is how a rejection removes them from the tile filter's
      launch, its scan and its scatter at once rather than from stage 1;
    - a **pass-through** contributes all of its segment's tiles unconditionally,
      running no SDF test at all.

    The second kind is the per-pair disable: one thread per (batch, segment)
    writing `seg_nT[s]`, as if there were no level above the tiles. So a segment
    with `super_size = 0` pays nothing for the level, and an 8-vertex cube never
    gathers a super-tile to reach its one tile.

    A rejected generator writes zero and so shares its successor's scan offset,
    which is what keeps `resolve_nTb` correct without a separate compaction of
    the survivors.

    It also clears `gcnt`, the survivor counter the tile filter accumulates into
    and `compact_nTa` scans -- an atomic counter has to start at zero, which is
    also why `zero_tcnt` clears `tcnt` -- and it is free here: this launch is
    already one thread per generator and already writing to that thread's own
    slot.

    Launch: dim=(B*nG_world,), a thread per world and generator, by `collide.broadphase_bbox`.
    Reads: dcol.cpr_active (B, nCPR) bool; mcol.gen_seg, gen_nT and gen_sup (nG_world,) int32, gen_c (nG_world,)
    wp.vec3, gen_r (nG_world,) float32, seg_cpr (nSEG,) int32. Writes: dcol.gen_w (B*nG_world,) int32;
    dcol.gcnt (B*nG_world,) int32, zeroed.
    """
    i = wp.tid()  # int: the (world, generator) slot, b * nG_world + g

    dcol.gcnt[i] = 0

    b = i / mcol.nG_world  # int: the world
    g = i - b * mcol.nG_world  # int: the generator
    s = mcol.gen_seg[g]  # int: its segment

    w = int(0)  # int: the tiles it contributes
    if dcol.cpr_active[b, mcol.seg_cpr[s]]:
        w = mcol.gen_nT[g]
        if mcol.gen_sup[g] == 1:
            if sphere_survives(b, s, mcol.gen_c[g], mcol.gen_r[g], mcol, dcol) == 0:
                w = 0

    dcol.gen_w[i] = w


@wp.kernel
def tile_filter(mcol: CollisionModel, dcol: CollisionData):
    """
    One gather per tile: can this run of a segment's primitives hold a candidate
    the vertex filter would keep?

    Launched over the tiles a world *has*, densely: thread `(b, j)` owns dense
    tile `j`, whose segment is one indexed load (`tile_seg[j]`) shared by the
    `tile_K` threads that tile becomes downstream. That load is much cheaper
    than an `nSEG x max_nT` rectangle, where `(b, s, t)` would come free from
    the thread id but every segment would be allotted as many slots as the
    longest one has -- where segment lengths spread widely, most of the launch,
    of the scan and of the scatter would exist only to write a zero.

    `j` indexes `tile_c`/`tile_r` directly, since those are stored in exactly
    this order (`seg_nT_beg[s] + t`).

    The test itself is `tile_survives`, shared with the compacted launch below.

    Launch: dim=(B, nT_world), a thread per world and dense tile, by `collide.broadphase_bbox` where
    `seg_compact` is 0.
    Reads: dcol.cpr_active (B, nCPR) bool, dcol.gen_w (B*nG_world,) int32 where `has_super`; mcol.tile_seg and
    tile_gen (nT_world,) int32, seg_cpr (nSEG,) int32 and the tile spheres. Writes: dcol.tflag (nTb_max,) int32,
    here B*nT_world long; dcol.gcnt (B*nG_world,) int32, added to.
    """
    b, j = wp.tid()  # int, int: the world, the dense tile

    # No padding to skip: every `j` is a tile some segment really has.
    s = mcol.tile_seg[j]  # int: the tile's segment
    keep = int(0)  # int: 1 if the tile survives
    if dcol.cpr_active[b, mcol.seg_cpr[s]]:
        keep = 1
        # A rejected super-tile takes its children with it here too, so the two
        # launches agree on the survivors. It cannot shrink *this* launch, which
        # is sized by the grid -- shrinking it is what the compacted path is for.
        # Nothing to read where no segment has super-tiles, which is the state
        # this branch is uniformly false in.
        if mcol.has_super == 1:
            if dcol.gen_w[b * mcol.nG_world + mcol.tile_gen[j]] == 0:
                keep = 0
        if keep == 1:
            keep = tile_survives(b, s, j, mcol, dcol)

    dcol.tflag[b * mcol.nT_world + j] = keep
    if keep == 1:
        wp.atomic_add(dcol.gcnt, b * mcol.nG_world + mcol.tile_gen[j], 1)


@wp.kernel
def tile_filter_seg(mcol: CollisionModel, dcol: CollisionData):
    """
    The same test, over the compacted list of active segments' tiles.

    The box filter rejects most pairs, and with them most of the tile grid --
    but `cpr_active` only makes a thread of the dense launch return early: the
    launch, the scan over `tflag` and the scatter are all sized by the grid
    regardless. Running over the compacted list shrinks all three at once, which
    is why this exists, and why the pair test is the full OBB test (an AABB
    culls worse and measured slower end to end).

    A thread's own segment costs a binary search here where the dense launch
    reads `tile_seg[j]`, so the dense slot is written to `tslot` for the scatter
    rather than resolved twice. The two launches keep identical candidates.

    A survivor is counted against its generator, which is what `compact_nTa`
    scans. That also means a thread past the live count has nothing to do at all:
    the scan does not run over this array, so there is no tail to keep valid --
    as `compact_nCa2` keeps none in stage 1, for the same reason.

    Launch: dim=(nTb_max,), a thread per tile candidate, by `collide.broadphase_bbox` where `seg_compact` is 1.
    Reads: dcol.ct[4], dcol.gen_s (B*nG_world,) int32 through `resolve_nTb`; mcol.seg_nT_beg (nSEG,) int32 and
    the tile spheres. Writes: dcol.tflag and tslot (nTb_max,) int32; dcol.gcnt (B*nG_world,) int32, added to.
    """
    i = wp.tid()  # int: the tile candidate

    if i >= dcol.ct[4]:
        return

    b, s, t, k = resolve_nTb(i, mcol, dcol)  # int, int, int, int: world, segment, tile within it, generator slot
    j = mcol.seg_nT_beg[s] + t  # int: the dense tile

    keep = tile_survives(b, s, j, mcol, dcol)  # int: 1 if the tile survives
    dcol.tflag[i] = keep
    dcol.tslot[i] = b * mcol.nT_world + j
    if keep == 1:
        wp.atomic_add(dcol.gcnt, k, 1)


@wp.func
def sphere_survives_toa(  # type: ignore
    b: int,
    s: int,
    c_b: wp.vec3,
    r: float,
    mcol: CollisionModel,
    dcol: CollisionData,
):
    """
    `sphere_survives` for a context with a `toa` pair: the margin also holds the distance segment
    s's motion can carry the sphere over the horizon (`toa_margin`, zero for a segment that is not
    `toa`). The derivation is `sphere_survives`'s with the prefilter's threshold raised by the
    same bound for each primitive, which is at most the sphere's.

    Called by `tile_survives_toa` and `gen_weights_toa`.
    In: as `sphere_survives`; also reads mcol.seg_toa, toa_h and dcol.twist_rel. Out: int, 1 or 0.
    """
    c = wp.transform_point(dcol.T_rel[b, s], c_b)  # wp.vec3: the centre, in the SDF geometry's body frame
    id_nB_2 = mcol.seg_g_s[s]  # int: the SDF geometry
    m = toa_margin(b, s, c, r, mcol, dcol)  # float: the distance the motion can close (m)
    if mcol.g_plane[id_nB_2] == 1:
        if c[2] - r < m:
            return 1
        return 0
    if eval_sdf_vox(c, id_nB_2, mcol) < r + 3.0 * mcol.g_Vox_r[id_nB_2] + m:
        return 1
    return 0


@wp.func
def tile_survives_toa(b: int, s: int, j: int, mcol: CollisionModel, dcol: CollisionData):  # type: ignore
    """
    `tile_survives` through `sphere_survives_toa`.

    Called by `tile_filter_toa` and `tile_filter_seg_toa`.
    In, Out: as `tile_survives`.
    """
    return sphere_survives_toa(b, s, mcol.tile_c[j], mcol.tile_r[j], mcol, dcol)


@wp.kernel
def gen_weights_toa(mcol: CollisionModel, dcol: CollisionData):
    """
    `gen_weights` for a context with a `toa` pair: a super-tile is tested with `sphere_survives_toa`.

    Launch, Reads, Writes: as `gen_weights`, by `collide.broadphase_bbox`.
    """
    i = wp.tid()  # int: the (world, generator) slot, b * nG_world + g
    dcol.gcnt[i] = 0
    b = i / mcol.nG_world  # int: the world
    g = i - b * mcol.nG_world  # int: the generator
    s = mcol.gen_seg[g]  # int: its segment
    w = int(0)  # int: the tiles it contributes
    if dcol.cpr_active[b, mcol.seg_cpr[s]]:
        w = mcol.gen_nT[g]
        if mcol.gen_sup[g] == 1:
            if sphere_survives_toa(b, s, mcol.gen_c[g], mcol.gen_r[g], mcol, dcol) == 0:
                w = 0
    dcol.gen_w[i] = w


@wp.kernel
def tile_filter_toa(mcol: CollisionModel, dcol: CollisionData):
    """
    `tile_filter` for a context with a `toa` pair: a tile is tested with `tile_survives_toa`.

    Launch, Reads, Writes: as `tile_filter`, by `collide.broadphase_bbox`.
    """
    b, j = wp.tid()  # int, int: the world, the dense tile
    s = mcol.tile_seg[j]  # int: the tile's segment
    keep = int(0)  # int: 1 if the tile survives
    if dcol.cpr_active[b, mcol.seg_cpr[s]]:
        keep = 1
        if mcol.has_super == 1:
            if dcol.gen_w[b * mcol.nG_world + mcol.tile_gen[j]] == 0:
                keep = 0
        if keep == 1:
            keep = tile_survives_toa(b, s, j, mcol, dcol)
    dcol.tflag[b * mcol.nT_world + j] = keep
    if keep == 1:
        wp.atomic_add(dcol.gcnt, b * mcol.nG_world + mcol.tile_gen[j], 1)


@wp.kernel
def tile_filter_seg_toa(mcol: CollisionModel, dcol: CollisionData):
    """
    `tile_filter_seg` for a context with a `toa` pair: a tile is tested with `tile_survives_toa`.

    Launch, Reads, Writes: as `tile_filter_seg`, by `collide.broadphase_bbox`.
    """
    i = wp.tid()  # int: the tile candidate
    if i >= dcol.ct[4]:
        return
    b, s, t, k = resolve_nTb(i, mcol, dcol)  # int, int, int, int: world, segment, tile within it, generator slot
    j = mcol.seg_nT_beg[s] + t  # int: the dense tile
    keep = tile_survives_toa(b, s, j, mcol, dcol)  # int: 1 if the tile survives
    dcol.tflag[i] = keep
    dcol.tslot[i] = b * mcol.nT_world + j
    if keep == 1:
        wp.atomic_add(dcol.gcnt, k, 1)


def _make_tiled_sdf_filter(with_radius: bool, partition: bool = False, toa: bool = False):
    """The voxel prefilter, with or without the per-primitive radius read.

    Two kernels rather than a branch: this is the widest launch in the pipeline
    and bound by its gather, so a context whose geometries carry no spheres
    should not pay a second one to read a zero. The `with_radius=False` kernel
    reads no radius at all, which makes "no spheres changes nothing" hold by
    construction.
    `make_collision_context` picks on `has_spheres`.

    Called by this module at import, for the four kernels below it.
    In: with_radius bool, read `mcol.prim_r`; partition bool, resolve a slot by block partition rather than by
    dividing by `tile_K`; toa bool, add each `toa` segment's motion margin (`toa_margin`) to the threshold.
    Out: the `tiled_sdf_filter` kernel.
    """

    @wp.func
    def _tiled_sdf_filter_one(id_nCa1: int, mcol: CollisionModel,  # type: ignore
                              dcol: CollisionData):
        """
        One primitive slot; `tiled_sdf_filter` runs it in a loop.

        Called by `tiled_sdf_filter`.
        In: id_nCa1 int, the stage-1 slot; mcol, read for idul_nSEG_to_nC (nSEG+1,) int32, idm_nC_to_prim (nC,)
        int32, v_b (nV,) wp.vec3, prim_r (nV,) float32 if `with_radius`, seg_g_s (nSEG,) int32, g_plane (nB,)
        int32, g_Vox_r (nB,) float32 and the vox table; dcol, read for T_rel (B, nSEG) wp.transformf and the
        resolve's tables. Writes: dcol.flag1[id_nCa1]; dcol.tcnt (nTa_max,) int32, added to.
        """
        if wp.static(partition):
            # int, int, int, int: the world, the candidate (an index into nC), its segment, its surviving tile
            id_Batch, id_nC, id_SEG, id_Tile = resolve_nCa1_partition(
                id_nCa1, mcol, dcol
            )
        else:
            id_Batch, id_nC, id_SEG, id_Tile = resolve_nCa1_tiled(id_nCa1, mcol, dcol)

        keep = int(0)  # int: 1 if the primitive survives
        # Padding at the end of a segment's last tile.
        if id_nC < mcol.idul_nSEG_to_nC[id_SEG + 1]:
            id_nV = mcol.idm_nC_to_prim[id_nC]  # int: the primitive, an index into v_b
            # wp.vec3: the primitive in the SDF geometry's body frame (m)
            v_b = wp.transform_point(dcol.T_rel[id_Batch, id_SEG], mcol.v_b[id_nV])
            id_nB_2 = mcol.seg_g_s[id_SEG]  # int: the SDF geometry

            # A sphere reaches the surface from `r` further out than its centre
            # does, so it is kept from `r` further out. A vertex has r = 0 and
            # this is the plain vertex test.
            r = float(0.0)  # float: the primitive's radius (m), 0 for a vertex
            if wp.static(with_radius):
                r = mcol.prim_r[id_nV]
            if wp.static(toa):
                # A `toa` segment keeps it from as far again as its motion can carry it over the
                # horizon; the threshold is all `r` is used for below.
                r += toa_margin(id_Batch, id_SEG, v_b, 0.0, mcol, dcol)

            if mcol.g_plane[id_nB_2] == 1:
                if v_b[2] < r:
                    keep = 1
            elif eval_sdf_vox(v_b, id_nB_2, mcol) < mcol.g_Vox_r[id_nB_2] + r:
                keep = 1

        dcol.flag1[id_nCa1] = keep
        if keep == 1:
            # The tile comes from the resolve, which had to know it anyway.
            wp.atomic_add(dcol.tcnt, id_Tile, 1)

    @wp.kernel
    def tiled_sdf_filter(mcol: CollisionModel, dcol: CollisionData):
        """
        The voxel prefilter over the surviving tiles' primitive slots.

        A thread learns which candidate it is through `resolve_nCa1_tiled`, one
        load shared across a tile rather than a binary search per primitive.

        A tile whose last run overhangs its segment's primitives has padding
        slots. They are masked here rather than compacted away, which is what
        lets the tile count times `tile_K` be the stage-1 count exactly.

        **It also counts its tile's survivors**, with one atomic per survivor
        into `tcnt`. That is what `compact_nCa2` scans, and having it here saves
        a second pass over `flag1` -- a whole read of the live stage-1 array.

        The thread owns one *primitive*, not one tile: a thread per tile would
        have each lane of a warp reading `v_b` 16 apart and lose the coalescing
        on the gather the narrowphase is bound by. One atomic per survivor is
        much cheaper than that.

        Nothing reads `flag1` past `ct[0]`, since the compaction walks tiles,
        so the tail is left alone rather than cleared -- clearing it would be
        writes over the whole capacity that buy nothing.

        **The launch is sized by the hardware, not by the arena.** A thread walks
        its slots `nCa1_grid` apart until it passes the live count, so the work
        tracks `ct[0]` while the allocation stays worst-case. That matters
        because this is by far the widest launch in the pipeline: an idle thread
        is nearly free, but an over-sized arena holds a great many of them, and
        this kernel can then spend longer retiring empty threads than working.
        See `make_collision_context`'s `grid_stride`.

        With `grid_stride` off, `nCa1_grid == nCa1_max`, the loop runs at most
        once and this is byte for byte a one-thread-per-slot kernel.

        Launch: dim=(nCa1_grid,), a thread per run of stage-1 slots `nCa1_grid` apart, up to `ct[0]`, by
        `collide.broadphase_sdf` (through `sdf_filter_kernel`).
        Reads: dcol.ct[0], dcol.nCa1_grid and what `_tiled_sdf_filter_one` reads. Writes: dcol.flag1 (nCa1_max,)
        int32; dcol.tcnt (nTa_max,) int32, added to.
        """
        stride = dcol.nCa1_grid  # int: the launch width, and the step between a thread's slots
        id_nCa1 = wp.tid()  # int: the thread's first stage-1 slot

        while id_nCa1 < dcol.ct[0]:
            _tiled_sdf_filter_one(id_nCa1, mcol, dcol)
            id_nCa1 += stride

    return tiled_sdf_filter


tiled_sdf_filter = _make_tiled_sdf_filter(False)  # the stage-1 filter: no radius read, a slot's tile by division
tiled_sdf_filter_r = _make_tiled_sdf_filter(True)  # with the radius read, tile by division
tiled_sdf_filter_p = _make_tiled_sdf_filter(False, partition=True)  # no radius read, tile by block partition
tiled_sdf_filter_rp = _make_tiled_sdf_filter(True, partition=True)  # with the radius read, tile by block partition
# The same four for a context with a `toa` pair, whose `toa` segments' thresholds take the motion margin.
tiled_sdf_filter_t = _make_tiled_sdf_filter(False, toa=True)
tiled_sdf_filter_rt = _make_tiled_sdf_filter(True, toa=True)
tiled_sdf_filter_pt = _make_tiled_sdf_filter(False, partition=True, toa=True)
tiled_sdf_filter_rpt = _make_tiled_sdf_filter(True, partition=True, toa=True)


def sdf_filter_kernel(mcol: CollisionModel):
    """
    The stage-1 filter this context runs.

    Called by `collide.broadphase_sdf`.
    In: mcol, read for has_spheres, partition and has_toa. Out: one of the eight `tiled_sdf_filter*` kernels; the
    four `toa` ones only for a context with a `toa` pair.
    """
    r, p = int(mcol.has_spheres), int(mcol.partition)  # int, int: read radii, resolve by block partition
    if int(mcol.has_toa):
        if p:
            return tiled_sdf_filter_rpt if r else tiled_sdf_filter_pt
        return tiled_sdf_filter_rt if r else tiled_sdf_filter_t
    if p:
        return tiled_sdf_filter_rp if r else tiled_sdf_filter_p
    return tiled_sdf_filter_r if r else tiled_sdf_filter
