"""
Trilinear interpolation of a sampled SDF: C0 field, normals discontinuous across cell faces. The cheap sibling of
`c2`.

`c2` fits a tricubic B-spline to the signed distance at the grid's knots and
reads 64 coefficients per query; this reads the 8 knot samples of the containing
cell and interpolates them directly. Same lattice, same samples, an eighth of the
gathers.

**Naming.** One representation is spelled two ways: `spline_type="linear"` on a
`SurfaceData`, and `c0` in this module and the packed collision tables (as the cubic
is `c2`, each named for its field's continuity, and the voxel-centre lookup is `vox`).

**Continuity.** The field is C0: its value is continuous across cell faces and
its gradient jumps there -- it is piecewise-bilinear in the two transverse
coordinates. Since the contact normal is `grad(sdf)`, normals step
discontinuously as a point slides along a surface, roughly once per cell
crossed, where the tricubic's are C1 (`c2`: a C2 field). That is the price of
the 8-gather read, and it is the reason this is a choice per surface.
"""

import numpy as np
import warp as wp

#######################################################################################
# Main Functionality (Host-Side)
#######################################################################################


def knots_from_c_3d(c):
    """
    The knot samples a tricubic coefficient array interpolates: the inverse of
    `c2.spline_c_3d`, taking (Nx+3, Ny+3, Nz+3) coefficients to (Nx+1, Ny+1,
    Nz+1) values.

    At a knot the four basis weights are `B(1), B(0), B(-1), B(-2) = 1, 4, 1, 0`
    (in `c2`'s unnormalized convention, where the 1/6 is folded into the
    coefficients), so the value there is `c[l] + 4 c[l+1] + c[l+2]` -- one
    separable [1, 4, 1] pass per axis. This is exact: `c2.spline_c_1d` solves
    precisely so that the spline interpolates its knots, and it holds at the
    boundary knots too.

    That is what lets a surface baked for the tricubic be evaluated trilinearly
    with no new asset: the coefficients are the sample lattice, re-encoded.

    Called by `lodos.col.sdf.types.make_collision_context` and `lodos.col.sdf.implicit_surface`.
    In: c (Nx+3, Ny+3, Nz+3) float, array-like. Out: (Nx+1, Ny+1, Nz+1) float64 numpy array.
    """
    c = np.asarray(c, dtype=np.float64)  # (Nx+3, Ny+3, Nz+3) float64
    assert c.ndim == 3, f"expected a 3-D coefficient array, got shape {c.shape}"

    for ax in range(3):  # ax: int, the axis the pass runs along
        c = np.moveaxis(c, ax, 0)
        assert c.shape[0] >= 3, f"axis {ax} has {c.shape[0]} coefficients, needs >= 3"
        c = c[:-2] + 4.0 * c[1:-1] + c[2:]
        c = np.moveaxis(c, 0, ax)

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
def cell_cs1(pq: wp.vec3, p_min: wp.vec3, N: wp.vec3i, h: wp.vec3):
    """
    The knot cell containing `pq`, and the local coordinates inside it.

    Returns the cell's lower corner as an index into the (N+1)^3 knot lattice --
    capped at N-1 so that `lo + 1` is always a knot -- together with `u` in
    [0, 1]^3.

    The query is clamped to the grid box *before* the cell is picked, so outside
    the grid the field is the constant extension of the nearest boundary face
    rather than an extrapolation. No vertex that reaches a narrowphase is out
    there -- the vox prefilter admits a vertex only within a voxel radius of the
    surface, and the grid is dilated well beyond the mesh -- but an inner sphere,
    admitted within its own radius more, can be (`narrowphase.grid_excess`), and
    the choice matters, because extrapolation is unbounded: a linear continuation of a
    boundary cell can run *negative* far outside the grid and invent a contact
    where there is no geometry at all. The clamped value is the boundary
    distance, which is large and positive. `c2` clamps the same way, so here too
    the two representations differ by no more than interpolation order.

    Called by `eval_cs1`, `eval_cs1_grad`, `lodos.col.sdf.narrowphase.eval_sdf_c0` and 1 other.
    In: pq wp.vec3, the query (m); p_min wp.vec3, N wp.vec3i, h wp.vec3, the grid (`implicit.GridParams`).
    Out: lo wp.vec3i, the cell's lower knot, each component in [0, N-1]; u wp.vec3, in [0, 1]^3.
    """
    d = pq - p_min  # wp.vec3: from the grid's lower corner
    g = wp.vec3(  # wp.vec3: in cells, clamped into [0, N]
        wp.clamp(d[0] / h[0], 0.0, wp.float32(N[0])),
        wp.clamp(d[1] / h[1], 0.0, wp.float32(N[1])),
        wp.clamp(d[2] / h[2], 0.0, wp.float32(N[2])),
    )

    # g >= 0, so truncation is floor.
    lo = wp.vec3i(  # wp.vec3i: the cell's lower knot
        wp.min(wp.int32(g[0]), N[0] - 1),
        wp.min(wp.int32(g[1]), N[1] - 1),
        wp.min(wp.int32(g[2]), N[2] - 1),
    )
    u = wp.vec3(  # wp.vec3: the local coordinates
        g[0] - wp.float32(lo[0]),
        g[1] - wp.float32(lo[1]),
        g[2] - wp.float32(lo[2]),
    )

    return lo, u


@wp.func
def basis_w_cs1(t: float):
    """
    The two corner weights at local coordinate `t`.

    Called by `eval_cs1`, `eval_cs1_grad`, `lodos.col.sdf.narrowphase.eval_sdf_c0` and 1 other.
    In: t float, in [0, 1]. Out: wp.vec2, the lower and the upper corner's weight.
    """
    return wp.vec2(1.0 - t, t)


@wp.func
def basis_dw_cs1(t: float):
    """
    Spline derivative with respect to `t`, which is measured in grid cells; a
    gradient in metres divides by that axis' cell width `h`.

    `t` is unused: the derivative of a linear weight is constant within a
    cell -- which is precisely the statement that the trilinear gradient is
    discontinuous across cell faces. It takes the argument so that this and
    `basis_w_cs1` read the same at the call site, as `basis_dw` and `basis_w` do
    for the tricubic.

    Called by `eval_cs1_grad` and `lodos.col.sdf.narrowphase.eval_nsdf_c0`.
    In: t float, unused. Out: wp.vec2, (-1, 1), per cell.
    """
    return wp.vec2(-1.0, 1.0)


@wp.func
def eval_cs1(
    pq: wp.vec3,  # type: ignore
    p_min: wp.vec3,
    N: wp.vec3i,
    h: wp.vec3,
    c: wp.array(dtype=wp.float32, ndim=3),  # type: ignore
):
    """
    Trilinear interpolation of the (N+1)^3 knot samples `c` at `pq`.

    Called by `lodos.col.sdf.implicit.eval_sdf`, `lodos.col.sdf.implicit.eval_sdf_c0` and `eval_cs1_k`.
    In: pq wp.vec3, the query (m); p_min wp.vec3, N wp.vec3i, h wp.vec3, the grid; c (Nx+1, Ny+1, Nz+1) float32.
    Out: float.
    """
    lo, u = cell_cs1(pq, p_min, N, h)  # wp.vec3i, wp.vec3: the cell's lower knot, the local coordinates

    fx = basis_w_cs1(u[0])  # wp.vec2: the x weights
    fy = basis_w_cs1(u[1])  # wp.vec2: the y weights
    fz = basis_w_cs1(u[2])  # wp.vec2: the z weights

    s = wp.float32(0.0)  # float: the value
    for i in range(2):  # i: int, the corner's x offset
        for j in range(2):  # j: int, its y offset
            for k in range(2):  # k: int, its z offset
                s += c[lo[0] + i, lo[1] + j, lo[2] + k] * fx[i] * fy[j] * fz[k]

    return s


@wp.func
def eval_cs1_grad(
    pq: wp.vec3,  # type: ignore
    p_min: wp.vec3,
    N: wp.vec3i,
    h: wp.vec3,
    c: wp.array(dtype=wp.float32, ndim=3),  # type: ignore
):
    """
    The trilinear value and its analytic gradient, from a single 2x2x2 read.

    The weight of a corner factorizes as `w(u_x) w(u_y) w(u_z)`, so
    differentiating along one axis swaps that axis' factor for its derivative and
    leaves the other two alone -- the same 8-term sum three more times over the
    *same* gathered samples, then `du/dpq = 1/h` per axis.

    Returns (value, gradient), the gradient unnormalized: for a signed distance
    field it has length ~1 near the surface and drifts away from it, and it
    vanishes on the medial axis, so normalization is the caller's business.

    Called by `lodos.col.sdf.implicit.eval_nsdf` and `lodos.col.sdf.implicit.eval_n_c0`.
    In: as `eval_cs1`. Out: float, the value, and wp.vec3, its gradient with respect to `pq`.
    """
    lo, u = cell_cs1(pq, p_min, N, h)  # wp.vec3i, wp.vec3: the cell's lower knot, the local coordinates

    fx, dfx = basis_w_cs1(u[0]), basis_dw_cs1(u[0])  # wp.vec2, wp.vec2: the x weights, their derivatives
    fy, dfy = basis_w_cs1(u[1]), basis_dw_cs1(u[1])  # wp.vec2, wp.vec2: the same along y
    fz, dfz = basis_w_cs1(u[2]), basis_dw_cs1(u[2])  # wp.vec2, wp.vec2: the same along z

    s = wp.float32(0.0)  # float: the value
    gx = wp.float32(0.0)  # float: its derivative along x, per cell
    gy = wp.float32(0.0)  # float: along y, per cell
    gz = wp.float32(0.0)  # float: along z, per cell
    for i in range(2):  # i: int, the corner's x offset
        for j in range(2):  # j: int, its y offset
            for k in range(2):  # k: int, its z offset
                cv = c[lo[0] + i, lo[1] + j, lo[2] + k]  # float: the corner's sample
                s += cv * fx[i] * fy[j] * fz[k]
                gx += cv * dfx[i] * fy[j] * fz[k]
                gy += cv * fx[i] * dfy[j] * fz[k]
                gz += cv * fx[i] * fy[j] * dfz[k]

    return s, wp.vec3(gx / h[0], gy / h[1], gz / h[2])


# Kernels
# -----------------------


@wp.kernel
def eval_cs1_k(
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

    The trilinear interpolation of `c` at each point, by `eval_cs1`.

    Launch: a thread per point; not launched in the repository.
    In: p (n,) wp.vec3, the points (m); p_min wp.vec3, N wp.vec3i, h wp.vec3, the grid; c (Nx+1, Ny+1, Nz+1)
    float32. Out: f (n,) float32.
    """

    tid = wp.tid()  # int: the point
    pq = p[tid]  # wp.vec3
    f[tid] = eval_cs1(pq, p_min, N, h, c)
