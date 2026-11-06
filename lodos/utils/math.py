"""
Quaternion and spatial-algebra helpers, in Warp.

Two conventions are worth stating because getting either wrong is silent:

  - quaternions are `wp.vec4` in **(w, x, y, z)** order, not `wp.quat`, which is
    (x, y, z, w);
  - spatial vectors are **(angular, linear)**, matching `cdof` and `cvel` in
    `lodos/rbd`.

MuJoCo Warp's helpers that Lodos's copies of its collision and solver use follow, at the end. They take
its quaternions, `wp.quat` holding (w, x, y, z) in its four slots; `quat_inv` and `quat_mul` take both,
as overloads, and are bit for bit MuJoCo Warp's `quat_inv` and `mul_quat` (as are `motion_cross` and
the 6x6 LU, the same functions in both).
"""

from typing import Any

import warp as wp

MINVAL = 1e-15  # MuJoCo's mjMINVAL: the divisor `safe_div` takes in place of a zero

# Lodos's inertia, [Ixx, Iyy, Izz, Ixy, Ixz, Iyz, mass*offset(3), mass], as `lodos.rbd.types.vec10`; defined
# here so that utils imports nothing of the packages that use it
vec10 = wp.types.vector(length=10, dtype=wp.float32)


@wp.func
def normalize4(q: wp.vec4):
    """
    Unit quaternion, mapping the zero vector to itself.

    Called by `quat_integrate` and `lodos.rbd.smooth._kinematics_level`.
    In: q wp.vec4. Out: wp.vec4.
    """
    n = wp.length(q)  # float: |q|
    if n > 0.0:
        return q / n
    return wp.vec4(0.0, 0.0, 0.0, 0.0)


@wp.func
def rotate(v: wp.vec3, q: wp.vec4):
    """
    Rotates `v` by the unit quaternion `q`.

    Called by `lodos.rbd.smooth._kinematics_level`, `lodos.rbd.smooth._site_poses`, `lodos.rbd.smooth._moment` and
    1 other.
    In: v wp.vec3; q wp.vec4, (w, x, y, z). Out: wp.vec3.
    """
    s = q[0]  # float: the scalar part
    u = wp.vec3(q[1], q[2], q[3])  # wp.vec3: the vector part
    return (
        2.0 * wp.dot(u, v) * u
        + (s * s - wp.dot(u, u)) * v
        + 2.0 * s * wp.cross(u, v)
    )


@wp.func
def quat_inv(q: wp.vec4):
    """
    Conjugate of a unit quaternion.

    Called by `quat_sub`, `lodos.rbd.smooth._moment` and `lodos.rbd.smooth._length`.
    In: q wp.vec4, (w, x, y, z). Out: wp.vec4.
    """
    return wp.vec4(q[0], -q[1], -q[2], -q[3])


@wp.func
def quat_inv(quat: wp.quat) -> wp.quat:  # noqa: F811  # the wp.quat overload (MuJoCo Warp's)
    """Adapted from MuJoCo Warp 3.14.0's `mujoco_warp/_src/math.py`."""
    return wp.quat(quat[0], -quat[1], -quat[2], -quat[3])


@wp.func
def quat_mul(u: wp.vec4, v: wp.vec4):
    """
    Quaternion product `u * v`.

    Called by `quat_sub`, `quat_integrate`, `lodos.rbd.smooth._kinematics_level` and 2 others.
    In: u, v wp.vec4, (w, x, y, z). Out: wp.vec4.
    """
    return wp.vec4(
        u[0] * v[0] - u[1] * v[1] - u[2] * v[2] - u[3] * v[3],
        u[0] * v[1] + u[1] * v[0] + u[2] * v[3] - u[3] * v[2],
        u[0] * v[2] - u[1] * v[3] + u[2] * v[0] + u[3] * v[1],
        u[0] * v[3] + u[1] * v[2] - u[2] * v[1] + u[3] * v[0],
    )


@wp.func
def quat_mul(u: wp.quat, v: wp.quat) -> wp.quat:  # noqa: F811  # the wp.quat overload (MuJoCo Warp's mul_quat)
    """Adapted from MuJoCo Warp 3.14.0's `mujoco_warp/_src/math.py`: its `mul_quat`."""
    return wp.quat(
        u[0] * v[0] - u[1] * v[1] - u[2] * v[2] - u[3] * v[3],
        u[0] * v[1] + u[1] * v[0] + u[2] * v[3] - u[3] * v[2],
        u[0] * v[2] - u[1] * v[3] + u[2] * v[0] + u[3] * v[1],
        u[0] * v[3] + u[1] * v[2] - u[2] * v[1] + u[3] * v[0],
    )


@wp.func
def quat_to_mat(q: wp.vec4):
    """
    Rotation matrix of a unit quaternion.

    Called by `lodos.rbd.smooth._kinematics_level` and `lodos.rbd.smooth._site_poses`.
    In: q wp.vec4, (w, x, y, z). Out: wp.mat33.
    """
    w, x, y, z = q[0], q[1], q[2], q[3]  # float, float, float, float

    return wp.mat33(
        w * w + x * x - y * y - z * z,
        2.0 * (x * y - w * z),
        2.0 * (x * z + w * y),
        2.0 * (x * y + w * z),
        w * w - x * x + y * y - z * z,
        2.0 * (y * z - w * x),
        2.0 * (x * z - w * y),
        2.0 * (y * z + w * x),
        w * w - x * x - y * y + z * z,
    )


@wp.func
def axis_angle_to_quat(axis: wp.vec3, angle: float):
    """
    Rotation of `angle` about `axis`, as a quaternion.

    Called by `quat_integrate` and `lodos.rbd.smooth._kinematics_level`.
    In: axis wp.vec3, a unit vector; angle float, rad. Out: wp.vec4, (w, x, y, z).
    """
    s = wp.sin(0.5 * angle)  # float: sin(angle / 2)
    return wp.vec4(wp.cos(0.5 * angle), axis[0] * s, axis[1] * s, axis[2] * s)


@wp.func
def quat_to_axis_angle(q: wp.vec4):
    """
    Axis and angle of a unit quaternion, with the angle wrapped to (-pi, pi].

    The wrap matters: a full turn is subtracted above pi so that `quat_sub`
    returns the *short* way round, which is what makes a joint spring pull toward
    its reference rather than away from it.

    Called by `quat_sub` and `lodos.rbd.smooth._length`.
    In: q wp.vec4, (w, x, y, z). Out: wp.vec3, the unit axis (zero for the identity); float, the angle (rad).
    """
    v = wp.vec3(q[1], q[2], q[3])  # wp.vec3: the vector part
    n = wp.length(v)  # float: its length

    axis = wp.vec3(0.0, 0.0, 0.0)  # wp.vec3: the unit axis, zero where q has no vector part
    if n > 0.0:
        axis = v / n

    angle = 2.0 * wp.atan2(n, q[0])  # float: the angle, rad
    if angle > wp.pi:
        angle -= 2.0 * wp.pi

    return axis, angle


@wp.func
def quat_sub(u: wp.vec4, v: wp.vec4):
    """
    The rotation from `v` to `u`, as a 3D rotation vector.

    Called by `lodos.rbd.passive._spring_damper` and `lodos.rbd.smooth._length`.
    In: u, v wp.vec4, (w, x, y, z). Out: wp.vec3, the axis times the angle (rad), in `v`'s frame.
    """
    axis, angle = quat_to_axis_angle(quat_mul(quat_inv(v), u))  # wp.vec3, float: the rotation's axis and angle
    return axis * angle


@wp.func
def quat_integrate(q: wp.vec4, w: wp.vec3, dt: float):
    """
    Advances `q` by angular velocity `w` over `dt`, renormalizing.

    Called by `lodos.rbd.integrate._integrate_pos`.
    In: q wp.vec4, (w, x, y, z); w wp.vec3, the angular velocity (rad/s) in `q`'s own frame; dt float, s.
    Out: wp.vec4.
    """
    n = wp.length(w)  # float: the angular speed, rad/s

    axis = wp.vec3(0.0, 0.0, 0.0)  # wp.vec3: the unit rotation axis, zero at rest
    if n > 0.0:
        axis = w / n

    return normalize4(quat_mul(q, axis_angle_to_quat(axis, dt * n)))


@wp.func
def inert_mul(inr: vec10, v: wp.spatial_vector):  # type: ignore
    """
    Inertia times spatial motion, giving spatial force.

    `cinert`'s packing: the rotational inertia's upper triangle as
    [xx, yy, zz, xy, xz, yz], then mass*offset, then mass.

    Called by `lodos.rbd.smooth._cfrc_local`, `lodos.rbd.smooth._build_M` and `lodos.rbd.smooth._build_M_rows`.
    In: inr vec10; v wp.spatial_vector, (angular, linear). Out: wp.spatial_vector, (angular, linear).
    """
    w = wp.vec3(v[0], v[1], v[2])  # wp.vec3: the angular part
    u = wp.vec3(v[3], v[4], v[5])  # wp.vec3: the linear part

    inertia = wp.mat33(  # wp.mat33: the rotational inertia
        inr[0], inr[3], inr[4],
        inr[3], inr[1], inr[5],
        inr[4], inr[5], inr[2],
    )
    pos = wp.vec3(inr[6], inr[7], inr[8])  # wp.vec3: mass times the centre of mass's offset

    return wp.spatial_vector(
        inertia * w + wp.cross(pos, u), inr[9] * u - wp.cross(pos, w)
    )


@wp.func
def motion_cross(u: wp.spatial_vector, v: wp.spatial_vector):
    """
    Cross product of two spatial motions.

    Called by `lodos.rbd.smooth._com_vel_level` and `lodos.cd.solver.constraint.jac_dot_dof`.
    In: u, v wp.spatial_vector, (angular, linear). Out: wp.spatial_vector, (angular, linear).
    """
    ua, ul = wp.vec3(u[0], u[1], u[2]), wp.vec3(u[3], u[4], u[5])  # wp.vec3, wp.vec3: u's angular and linear parts
    va, vl = wp.vec3(v[0], v[1], v[2]), wp.vec3(v[3], v[4], v[5])  # wp.vec3, wp.vec3: v's angular and linear parts

    return wp.spatial_vector(
        wp.cross(ua, va), wp.cross(ul, va) + wp.cross(ua, vl)
    )


@wp.func
def motion_cross_force(v: wp.spatial_vector, f: wp.spatial_vector):
    """
    Cross product of a spatial motion and a spatial force.

    Called by `lodos.rbd.smooth._cfrc_local`.
    In: v wp.spatial_vector, a motion; f wp.spatial_vector, a force; both (angular, linear).
    Out: wp.spatial_vector, a force.
    """
    va, vl = wp.vec3(v[0], v[1], v[2]), wp.vec3(v[3], v[4], v[5])  # wp.vec3, wp.vec3: v's angular and linear parts
    fa, fl = wp.vec3(f[0], f[1], f[2]), wp.vec3(f[3], f[4], f[5])  # wp.vec3, wp.vec3: f's torque and force

    return wp.spatial_vector(
        wp.cross(va, fa) + wp.cross(vl, fl), wp.cross(va, fl)
    )


#######################################################################################
# A free body's implicit step (`lodos.rbd.implicit`), from MJWarp's `math.py`
#######################################################################################
#
# Adapted from MuJoCo Warp 3.14.0's `mujoco_warp/_src/math.py`; `vec6`, `vec6i` and `mat66` from its
# `mujoco_warp/_src/types.py`.

vec6 = wp.types.vector(length=6, dtype=float)
vec6i = wp.types.vector(length=6, dtype=int)
mat66 = wp.types.matrix(shape=(6, 6), dtype=float)


@wp.func
def free_bias_vel_blocks(mass: float, R: wp.mat33, Xi: wp.mat33, inertia: wp.vec3, s: wp.vec3, w_local: wp.vec3):
    """
    The 3x3 blocks of `d qfrc_bias / d qvel` for a free body standing alone: `lin`, the
    linear rows over the angular columns (scaled by the mass where used), and `rot`, the
    angular rows over the angular columns. `R` is the body frame, `Xi` the inertial one,
    `s` the centre of mass from the body origin, `w_local` the angular velocity in the
    body frame. MJWarp's `free_bias_vel_blocks`.
    """
    w = R @ w_local
    Iw = Xi @ wp.diag(inertia) @ wp.transpose(Xi)

    ws = wp.cross(w, s)
    Iww = Iw @ w
    K = wp.outer(s, w) - wp.identity(n=3, dtype=float) * wp.dot(w, s) + wp.skew(ws)
    lin = K @ R

    neg_skew_Iww = wp.mat33(0.0, Iww[2], -Iww[1], -Iww[2], 0.0, Iww[0], Iww[1], -Iww[0], 0.0)
    C = -mass * (wp.skew(s) @ K) + wp.skew(w) @ Iw + neg_skew_Iww
    rot = wp.transpose(R) @ C @ R
    return lin, rot


@wp.func
def lu_factor_6x6(A_in: mat66):  # type: ignore
    """LU with partial pivoting; `ok` False where a pivot falls below MuJoCo's `mjMINVAL`. MJWarp's."""
    A = A_in
    pivot = vec6i(0)
    for k in range(6):
        maxval = wp.abs(A[k, k])
        maxrow = k
        for i in range(k + 1, 6):
            val = wp.abs(A[i, k])
            if val > maxval:
                maxval = val
                maxrow = i
        if maxval < 1e-15:
            return A, pivot, False
        pivot[k] = maxrow
        if maxrow != k:
            for j in range(6):
                tmp = A[k, j]
                A[k, j] = A[maxrow, j]
                A[maxrow, j] = tmp
        diaginv = 1.0 / A[k, k]
        for i in range(k + 1, 6):
            A[i, k] *= diaginv
            Aik = A[i, k]
            for j in range(k + 1, 6):
                A[i, j] -= Aik * A[k, j]
    return A, pivot, True


@wp.func
def lu_solve_6x6(A: mat66, pivot: vec6i, b: vec6):  # type: ignore
    """`A x = b` from `lu_factor_6x6`'s factor and pivots. MJWarp's."""
    x = b
    for i in range(6):
        p = pivot[i]
        if p != i:
            tmp = x[i]
            x[i] = x[p]
            x[p] = tmp
        for j in range(i):
            x[i] -= A[i, j] * x[j]
    for i in range(5, -1, -1):
        for j in range(i + 1, 6):
            x[i] -= A[i, j] * x[j]
        x[i] /= A[i, i]
    return x


#######################################################################################
# From MuJoCo Warp's mujoco_warp/_src/math.py (Copyright 2025 The Newton Developers, Apache License 2.0,
# http://www.apache.org/licenses/LICENSE-2.0): the helpers Lodos's copies of its collision and solver use,
# reindented to four spaces
#######################################################################################
#
# Adapted from MuJoCo Warp 3.14.0's `mujoco_warp/_src/math.py`.


@wp.func
def quat_mul_axis(q: wp.quat, axis: wp.vec3f) -> wp.quat:
    """Multiplies a quaternion and an axis."""
    return wp.quat(
        -q[1] * axis[0] - q[2] * axis[1] - q[3] * axis[2],
        q[0] * axis[0] + q[2] * axis[2] - q[3] * axis[1],
        q[0] * axis[1] + q[3] * axis[0] - q[1] * axis[2],
        q[0] * axis[2] + q[1] * axis[1] - q[2] * axis[0],
    )


@wp.func
def quat_to_vel(quat: wp.quat) -> wp.vec3:
    axis = wp.vec3(quat[1], quat[2], quat[3])
    sin_a_2 = wp.norm_l2(axis)

    if sin_a_2 == 0.0:
        return wp.vec3(0.0)

    speed = 2.0 * wp.atan2(sin_a_2, quat[0])
    # when axis-angle is larger than pi, rotation is in the opposite direction
    if speed > wp.pi:
        speed -= 2.0 * wp.pi

    return axis * speed / sin_a_2


@wp.func
def orthogonals(a: wp.vec3):
    y = wp.vec3(0.0, 1.0, 0.0)
    z = wp.vec3(0.0, 0.0, 1.0)
    b = wp.where((-0.5 < a[1]) and (a[1] < 0.5), y, z)
    b = b - a * wp.dot(a, b)
    b = wp.normalize(b)
    if wp.length(a) == 0.0:
        b = wp.vec3(0.0, 0.0, 0.0)
    c = wp.cross(a, b)

    return b, c


@wp.func
def make_frame(a: wp.vec3):
    a = wp.normalize(a)
    b, c = orthogonals(a)

    # fmt: off
    return wp.mat33(
        a.x, a.y, a.z,
        b.x, b.y, b.z,
        c.x, c.y, c.z
    )


@wp.func
def normalize_with_norm(x: Any):
    norm = wp.length(x)
    if norm == 0.0:
        return x, 0.0
    return x / norm, norm


@wp.func
def closest_segment_point(a: wp.vec3, b: wp.vec3, pt: wp.vec3) -> wp.vec3:
    """Returns the closest point on the a-b line segment to a point pt."""
    ab = b - a
    t = wp.dot(pt - a, ab) / (wp.dot(ab, ab) + 1e-6)
    return a + wp.clamp(t, 0.0, 1.0) * ab


@wp.func
def safe_div(x: Any, y: Any) -> Any:
    return x / wp.where(y != 0.0, y, MINVAL)


@wp.func
def upper_tri_index(n: int, i: int, j: int) -> int:
    """Returns index of a_ij = a_ji in upper triangular matrix (excluding diagonal)."""
    return (i * (2 * n - i - 3)) // 2 + j - 1


@wp.func
def upper_trid_index(n: int, i: int, j: int) -> int:
    """Returns index of a_ij = a_ji in upper triangular matrix (including diagonal)."""
    if j < i:
        i, j = j, i
    return (i * (2 * n - i - 1)) // 2 + j
