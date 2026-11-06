"""
The tricubic B-spline of a sampled SDF: C2 field, C1 normals.

`spline_c_3d` fits it to the signed distance at the grid's knots (the bake), through every sample, with natural end
conditions: the phantom coefficient past each end is extrapolated linearly, so the second derivative vanishes at the
end knots. The evaluators read the 4x4x4 coefficients around a query. The uniform cubic B-spline basis is C2, and so is
the tensor-product field inside the grid box; its gradient, the contact normal, is C1. The evaluators clamp a query into
the box, and `lodos.col.sdf.implicit` and the narrowphase add the distance to the box outside it: there the field is
continuous but not smooth across the box's faces. `c0` is the trilinear alternative over the same samples.
"""

import numpy as np
import warp as wp

from scipy.linalg import solve_banded

#######################################################################################
# Main Functionality (Host-Side)
#######################################################################################

# --------------------------------------------------------------------
# 1D Spline Coefficient Computation
# --------------------------------------------------------------------


def spline_c_1d(f, N):
    """
    The tricubic B-spline coefficients interpolating `f`, along one axis.

    The spline has N intervals, N+1 knot points.

    f: array of shape (N+1,)
    N: int

    The two end coefficients are pinned so that the spline interpolates the first
    and last samples -- a B-spline's value at a knot is c[i-1] + 4c[i] + c[i+1]
    in this module's unnormalized basis (the 1/6 folded into the coefficients),
    so an end sample fixes its own coefficient at f/6 -- and the interior ones
    solve the tridiagonal system that leaves. `c[0]` and `c[-1]` are the phantom
    coefficients outside the grid, extrapolated linearly, which is what makes the
    evaluator's 4-tap stencil defined at the boundary.

    Solved in float64 by LAPACK's banded routine. The system is strongly
    diagonally dominant (4 against 2).

    Called by `spline_c_3d`, along each axis through `np.apply_along_axis`.
    Out: (N+3,) float64 numpy array.
    """
    f = np.array(f, dtype=np.float64)  # (N+1,) float64, a copy: the right-hand side
    c = np.zeros(N + 3, dtype=np.float64)  # (N+3,) float64

    c[1] = f[0] / 6.0
    c[-2] = f[-1] / 6.0
    f[1] -= c[1]
    f[-2] -= c[-2]

    # (1, 1) banded storage: superdiagonal, diagonal, subdiagonal, each 1-4-1.
    ab = np.zeros((3, N - 1))  # (3, N-1) float64
    ab[0, 1:] = 1.0
    ab[1, :] = 4.0
    ab[2, :-1] = 1.0
    c[2:-2] = solve_banded((1, 1), ab, f[1:-1])

    c[0] = 2.0 * c[1] - c[2]
    c[-1] = 2.0 * c[-2] - c[-3]

    return c


# --------------------------------------------------------------------
# 3D Spline Coefficient Computation
# --------------------------------------------------------------------


def spline_c_3d(f, Nx, Ny, Nz):
    """
    The same, on a 3D grid: the 1D solve along each axis in turn.

    A tensor-product B-spline separates, so interpolating in x, then y, then z
    gives the coefficients of the 3D spline through every sample -- and each pass
    grows its axis from N+1 to N+3.

    f: array of shape (Nx+1, Ny+1, Nz+1)
    Nx/y/z: int

    Returns (Nx+3, Ny+3, Nz+3).

    Called by `lodos.col.sdf.implicit.create_implicit_surface_from_mesh`.
    Out: (Nx+3, Ny+3, Nz+3) float64 numpy array.
    """
    c = np.apply_along_axis(spline_c_1d, 0, np.asarray(f, dtype=np.float64), Nx)  # (Nx+3, Ny+1, Nz+1) float64
    c = np.apply_along_axis(spline_c_1d, 1, c, Ny)
    c = np.apply_along_axis(spline_c_1d, 2, c, Nz)

    return c


#######################################################################################
# Main Functionality (Warp-Side)
#######################################################################################

# --------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------

# Functions
# -----------------------


@wp.func
def eval_basis_1d_wp(t: float):
    """
    The uniform cubic B-spline basis at `t`, six times the usual normalization: 4 at 0, 1 at +-1, 0 from |t| = 2.

    Called by `eval_cs3`, `eval_cs3_grad` and `lodos.col.sdf.narrowphase.basis_w`.
    In: t float, in cells from the basis function's centre knot. Out: float.
    """
    t = wp.abs(t)  # float: |t|

    if t <= 1:
        return 4.0 - 6.0 * t * t + 3.0 * t * t * t
    elif t <= 2:
        t2 = 2.0 - t  # float
        return t2 * t2 * t2
    else:
        return 0.0


@wp.func
def eval_basis_1d_d_wp(t: float):
    """
    d/dt of `eval_basis_1d_wp`.

        B(t)  = 4 - 6t^2 + 3|t|^3          |t| <= 1
              = (2 - |t|)^3                1 < |t| <= 2
        B'(t) = 3t(3|t| - 4)               |t| <= 1
              = -3 sign(t) (2 - |t|)^2     1 < |t| <= 2

    Both branches give -3 sign(t) at |t| = 1 and the outer one vanishes at
    |t| = 2, so B' is continuous; so is B'' (6 from both sides at |t| = 1, 0 at
    |t| = 2), which is the statement that B is C^2.

    Unlike `eval_basis_1d_wp` this must *not* fold `t` onto its absolute value:
    the inner branch is odd in `t`, and the outer one carries an explicit sign.

    Called by `eval_cs3_grad` and `lodos.col.sdf.narrowphase.basis_dw`.
    In: t float, in cells. Out: float, per cell.
    """
    a = wp.abs(t)  # float: |t|

    if a <= 1:
        return 3.0 * t * (3.0 * a - 4.0)
    elif a <= 2:
        d = 2.0 - a  # float
        if t > 0.0:
            return -3.0 * d * d
        else:
            return 3.0 * d * d
    else:
        return 0.0


@wp.func
def eval_cs3(
    pq: wp.vec3,  # type: ignore
    p_min: wp.vec3,
    N: wp.vec3i,
    h: wp.vec3,
    c: wp.array(dtype=wp.float32, ndim=3),  # type: ignore
):
    """
    The tricubic B-spline with coefficients `c` at `pq` clamped into the grid box: a 4x4x4 sum of coefficients
    times basis weights.

    Called by `lodos.col.sdf.implicit.eval_sdf`, `lodos.col.sdf.implicit.eval_sdf_c2` and `eval_cs3_k`.
    In: pq wp.vec3, the query (m); p_min wp.vec3, N wp.vec3i, h wp.vec3, the grid; c (Nx+3, Ny+3, Nz+3) float32.
    Out: float.
    """
    pq = pq - p_min  # wp.vec3: from the grid's lower corner
    # Clamped into the grid, not merely its cell index: with the index clamped
    # and the basis argument left outside, all four cubic weights fall outside
    # their support and the sum is exactly zero -- a point metres away reads as
    # lying on the surface. `implicit.eval_sdf` adds the distance to the box back
    # on, which is what makes the outside meaningful rather than merely finite.
    x = wp.clamp(pq[0] / h[0], 0.0, wp.float32(N[0]))  # float: in cells, in [0, Nx]
    y = wp.clamp(pq[1] / h[1], 0.0, wp.float32(N[1]))  # float: in [0, Ny]
    z = wp.clamp(pq[2] / h[2], 0.0, wp.float32(N[2]))  # float: in [0, Nz]

    lx, ly, lz = (  # float, float, float: the cell's lower knot
        wp.floor(wp.clamp(x, 0.0, wp.float32(N[0] + 2))),
        wp.floor(wp.clamp(y, 0.0, wp.float32(N[1] + 2))),
        wp.floor(wp.clamp(z, 0.0, wp.float32(N[2] + 2))),
    )

    lxi, lyi, lzi = wp.int32(lx), wp.int32(ly), wp.int32(lz)  # int, int, int: the same, as indices

    ux, uy, uz = (  # int, int, int: the largest coefficient index read, per axis
        wp.clamp(lxi + 3, 0, N[0] + 2),
        wp.clamp(lyi + 3, 0, N[1] + 2),
        wp.clamp(lzi + 3, 0, N[2] + 2),
    )

    fx = wp.vec4f()  # wp.vec4f: the four x weights
    for i in range(4):  # i: int, the stencil's x tap
        t = x - lx - wp.float32(i) + 1.0  # float: the tap's basis argument, in cells
        fx[i] = eval_basis_1d_wp(t)

    fy = wp.vec4f()  # wp.vec4f: the four y weights
    for i in range(4):
        t = y - ly - wp.float32(i) + 1.0
        fy[i] = eval_basis_1d_wp(t)

    fz = wp.vec4f()  # wp.vec4f: the four z weights
    for i in range(4):
        t = z - lz - wp.float32(i) + 1.0
        fz[i] = eval_basis_1d_wp(t)

    sum = wp.float32(0.0)  # float: the value
    for i in range(4):
        for j in range(4):  # j: int, the y tap
            for k in range(4):  # k: int, the z tap
                i_c = wp.min(i + lxi, ux)  # int: the coefficient's x index
                j_c = wp.min(j + lyi, uy)  # int: its y index
                k_c = wp.min(k + lzi, uz)  # int: its z index
                sum += c[i_c, j_c, k_c] * fx[i] * fy[j] * fz[k]

    return sum


@wp.func
def eval_cs3_grad(
    pq: wp.vec3,  # type: ignore
    p_min: wp.vec3,
    N: wp.vec3i,
    h: wp.vec3,
    c: wp.array(dtype=wp.float32, ndim=3),  # type: ignore
):
    """
    The spline's value and its analytic gradient, from a single 4x4x4 sweep.

    The tensor-product weight of a coefficient factorizes as
    `B(t_i^x) B(t_j^y) B(t_k^z)`, so differentiating along one axis swaps that
    axis' factor for `B'` and leaves the other two alone. The gradient is
    therefore the same 64-term sum three more times over the *same* coefficients
    -- no extra gather, only extra arithmetic on values already in registers.

    `eval_cs3` maps `x = (pq - p_min) / h` and the basis arguments are
    `t_i = x - l - i + 1`, so `dt_i/dpq_x = 1 / h_x`: one division per axis, and
    each axis carries its own `h`.

    Returns (value, gradient). The gradient is *not* normalized; for a signed
    distance field it has length ~1 near the surface and drifts away from it,
    and it vanishes on the medial axis, so normalization is the caller's
    business.

    Called by `lodos.col.sdf.implicit.eval_nsdf` and `lodos.col.sdf.implicit.eval_n_c2`.
    In: as `eval_cs3`. Out: float, the value, and wp.vec3, its gradient with respect to `pq`.
    """
    pq = pq - p_min  # wp.vec3: from the grid's lower corner
    # Clamped into the grid, not merely its cell index: with the index clamped
    # and the basis argument left outside, all four cubic weights fall outside
    # their support and the sum is exactly zero -- a point metres away reads as
    # lying on the surface. `implicit.eval_sdf` adds the distance to the box back
    # on, which is what makes the outside meaningful rather than merely finite.
    x = wp.clamp(pq[0] / h[0], 0.0, wp.float32(N[0]))  # float: in cells, in [0, Nx]
    y = wp.clamp(pq[1] / h[1], 0.0, wp.float32(N[1]))  # float: in [0, Ny]
    z = wp.clamp(pq[2] / h[2], 0.0, wp.float32(N[2]))  # float: in [0, Nz]

    lx, ly, lz = (  # float, float, float: the cell's lower knot
        wp.floor(wp.clamp(x, 0.0, wp.float32(N[0] + 2))),
        wp.floor(wp.clamp(y, 0.0, wp.float32(N[1] + 2))),
        wp.floor(wp.clamp(z, 0.0, wp.float32(N[2] + 2))),
    )

    lxi, lyi, lzi = wp.int32(lx), wp.int32(ly), wp.int32(lz)  # int, int, int: the same, as indices

    ux, uy, uz = (  # int, int, int: the largest coefficient index read, per axis
        wp.clamp(lxi + 3, 0, N[0] + 2),
        wp.clamp(lyi + 3, 0, N[1] + 2),
        wp.clamp(lzi + 3, 0, N[2] + 2),
    )

    fx, dfx = wp.vec4f(), wp.vec4f()  # wp.vec4f, wp.vec4f: the four x weights, their derivatives per cell
    for i in range(4):  # i: int, the stencil's x tap
        t = x - lx - wp.float32(i) + 1.0  # float: the tap's basis argument, in cells
        fx[i] = eval_basis_1d_wp(t)
        dfx[i] = eval_basis_1d_d_wp(t)

    fy, dfy = wp.vec4f(), wp.vec4f()  # wp.vec4f, wp.vec4f: the same along y
    for i in range(4):
        t = y - ly - wp.float32(i) + 1.0
        fy[i] = eval_basis_1d_wp(t)
        dfy[i] = eval_basis_1d_d_wp(t)

    fz, dfz = wp.vec4f(), wp.vec4f()  # wp.vec4f, wp.vec4f: the same along z
    for i in range(4):
        t = z - lz - wp.float32(i) + 1.0
        fz[i] = eval_basis_1d_wp(t)
        dfz[i] = eval_basis_1d_d_wp(t)

    s = wp.float32(0.0)  # float: the value
    gx = wp.float32(0.0)  # float: its derivative along x, per cell
    gy = wp.float32(0.0)  # float: along y, per cell
    gz = wp.float32(0.0)  # float: along z, per cell
    for i in range(4):
        i_c = wp.min(i + lxi, ux)  # int: the coefficient's x index
        for j in range(4):  # j: int, the y tap
            j_c = wp.min(j + lyi, uy)  # int: its y index
            for k in range(4):  # k: int, the z tap
                k_c = wp.min(k + lzi, uz)  # int: its z index
                cv = c[i_c, j_c, k_c]  # float: the coefficient
                s += cv * fx[i] * fy[j] * fz[k]
                gx += cv * dfx[i] * fy[j] * fz[k]
                gy += cv * fx[i] * dfy[j] * fz[k]
                gz += cv * fx[i] * fy[j] * dfz[k]

    return s, wp.vec3(gx / h[0], gy / h[1], gz / h[2])


# Kernels
# -----------------------


@wp.kernel
def eval_cs3_k(
    p: wp.array(dtype=wp.vec3),  # type: ignore
    f: wp.array(dtype=float),  # type: ignore
    p_min: wp.vec3,
    N: wp.vec3i,
    h: wp.vec3,
    c: wp.array(dtype=wp.float32, ndim=3),  # type: ignore
):
    """
    p: shape(N, 3)
    f: shape(N,)

    The tricubic B-spline with coefficients `c` at each point, by `eval_cs3`.

    Launch: a thread per point; not launched in the repository.
    In: p (n,) wp.vec3, the points (m); p_min wp.vec3, N wp.vec3i, h wp.vec3, the grid; c (Nx+3, Ny+3, Nz+3)
    float32. Out: f (n,) float32.
    """

    tid = wp.tid()  # int: the point
    pq = p[tid]  # wp.vec3
    f[tid] = eval_cs3(pq, p_min, N, h, c)
