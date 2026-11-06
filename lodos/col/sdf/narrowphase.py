"""
Narrowphase SDF evaluation over the stage-2 survivors.

**Two SDF representations, one per context.** A geometry's signed distance is
either a tricubic B-spline over 64 packed coefficients (`c2`) or trilinear
interpolation of the 8 knot samples of the containing cell (`c0`); `mcol.spline`
says which, and `make_collision_context` requires every non-plane geometry of a
context to agree. The normal is `grad(sdf)` either way.

The kernels rely on that homogeneity. A step evaluates the SDF once per
surviving candidate -- millions of them -- so the choice is compiled in rather
than branched on: every kernel below is built twice by a factory, once against
each evaluator, and `collide` picks one at launch. A runtime branch on
`mcol.spline` per candidate would put a divergence-free but register-costly
test in the hottest loop in the pipeline for a value that is uniform across the
whole launch.
"""

import warp as wp

from lodos.col.sdf.spline.c0 import cell_cs1, basis_w_cs1, basis_dw_cs1
from lodos.col.sdf.spline.c2 import eval_basis_1d_wp, eval_basis_1d_d_wp
from lodos.col.sdf.compact import resolve_nCa1
from lodos.col.sdf.types import (
    CONTACT_POINTS,
    CollisionModel,
    CollisionData,
    idm_nB_nx_ny_nz_to_nVox_c0,
    idm_nB_nx_ny_nz_to_nVox_c2,
)

#######################################################################################
# V-SDF Contacts
#######################################################################################


@wp.func
def grid_excess(p_b: wp.vec3, B_id: int, mcol: CollisionModel):  # type: ignore
    """
    How far `p_b` lies outside this geometry's voxel grid, as a vector and a length.

    Zero inside, so it costs the interior nothing and changes no interior value.

    A baked field only exists inside its grid, and *both* representations
    misreport outside it: `spline.c0`'s clamp extends the boundary value as a constant,
    and so does the tricubic's (`spline_cell`).

    That is not reachable by a *vertex*, which the vox prefilter admits only near
    the surface. It is very reachable by an **inner sphere**: a sphere of radius
    `r` contacts when `sdf < r`, and the constant extension reads the boundary
    value, the margin by which the grid clears the surface, so wherever that
    margin is less than `r` a contact is invented at any separation.

    So the field is evaluated at the nearest point of the grid and this distance
    added back: `sdf(clamp(p)) + ||p - clamp(p)||`. Inside, identity -- the term
    is zero and no interior value moves. Outside, it grows with true separation.

    **It is an upper bound on the true distance, not a lower one.** By the
    triangle inequality `d(p, S) <= ||p - clamp(p)|| + sdf(clamp(p))`, with the
    bare box distance `||p - clamp(p)||` the lower bound. For an *evaluation*
    that is the right choice, since it is exact where the nearest surface point
    lies near the clamp point and the error only grows where nothing is in
    contact anyway. A *filter* wanting to reject candidates must not use it --
    over-estimating a distance is how a real contact gets dropped -- and
    `broadphase.eval_sdf_vox` accordingly extends its boundary voxel as a
    constant, which under-estimates and so only ever admits too much.

    Called by `eval_sdf_c2`, `eval_nsdf_c2`, `eval_sdf_c0` and `eval_nsdf_c0`.
    In: p_b wp.vec3, a point in the geometry's body frame (m); B_id int, the geometry; mcol, read for g_Vox_pmin
    and g_Vox_h (nB,) wp.vec3 and g_Vox_N (nB,) wp.vec3i. Out: wp.vec3, float: `p_b` less its nearest point of
    the grid box, and that offset's length (m).
    """
    p_min = mcol.g_Vox_pmin[B_id]  # wp.vec3: the grid's lower corner
    N = mcol.g_Vox_N[B_id]  # wp.vec3i: voxels per axis
    h = mcol.g_Vox_h[B_id]  # wp.vec3: voxel edge lengths (m)
    d = wp.vec3(  # wp.vec3: `p_b` less its clamp into the grid box (m)
        p_b[0] - wp.clamp(p_b[0], p_min[0], p_min[0] + h[0] * wp.float32(N[0])),
        p_b[1] - wp.clamp(p_b[1], p_min[1], p_min[1] + h[1] * wp.float32(N[1])),
        p_b[2] - wp.clamp(p_b[2], p_min[2], p_min[2] + h[2] * wp.float32(N[2])),
    )
    return d, wp.length(d)


@wp.func
def spline_cell(p_b: wp.vec3, B_id: int, mcol: CollisionModel):  # type: ignore
    """
    Tricubic B-spline setup at `p_b`: the base cell, the clamped upper bounds, and
    the query point in grid coordinates together with that cell's lower knot.

    Shared by every evaluation so they cannot drift apart. The basis weights are
    built separately by `basis_w` and `basis_dw`, so the distance-only evaluator
    never constructs derivative weights at all rather than constructing them and
    relying on the compiler to remove them again.

    **The query is clamped into the grid**, not only its cell index. Clamping the
    index alone leaves the basis argument outside its support, where all four
    cubic weights are zero and the sum with them -- a geometry hundreds of
    millimetres away would read as lying precisely on the surface.
    `grid_excess` says how the distance is put back.

    Called by `eval_sdf_c2` and `eval_nsdf_c2`.
    In: p_b wp.vec3, a point in the geometry's body frame (m); B_id int, the geometry; mcol, read for g_Vox_pmin
    and g_Vox_h (nB,) wp.vec3 and g_Vox_N (nB,) wp.vec3i. Out: lo wp.vec3i, the cell's lower knot; up wp.vec3i,
    the last coefficient index per axis; g wp.vec3, the query in grid units; fl wp.vec3, `lo` as floats.
    """
    p_min = mcol.g_Vox_pmin[B_id]  # wp.vec3: the grid's lower corner
    N = mcol.g_Vox_N[B_id]  # wp.vec3i: voxels per axis
    h = mcol.g_Vox_h[B_id]  # wp.vec3: voxel edge lengths (m)

    p_b = p_b - p_min
    g = wp.vec3(  # wp.vec3: the query in grid units, clamped into the grid
        wp.clamp(p_b[0] / h[0], 0.0, wp.float32(N[0])),
        wp.clamp(p_b[1] / h[1], 0.0, wp.float32(N[1])),
        wp.clamp(p_b[2] / h[2], 0.0, wp.float32(N[2])),
    )

    fl = wp.vec3(  # wp.vec3: the cell's lower knot, floored
        wp.floor(wp.clamp(g[0], 0.0, wp.float32(N[0] + 2))),
        wp.floor(wp.clamp(g[1], 0.0, wp.float32(N[1] + 2))),
        wp.floor(wp.clamp(g[2], 0.0, wp.float32(N[2] + 2))),
    )

    lo = wp.vec3i(wp.int32(fl[0]), wp.int32(fl[1]), wp.int32(fl[2]))  # wp.vec3i: `fl` as ints
    up = wp.vec3i(  # wp.vec3i: the last coefficient index per axis, `lo + 3` clamped to `N + 2`
        wp.clamp(lo[0] + 3, 0, N[0] + 2),
        wp.clamp(lo[1] + 3, 0, N[1] + 2),
        wp.clamp(lo[2] + 3, 0, N[2] + 2),
    )

    return lo, up, g, fl


@wp.func
def basis_w(t: float, l: float):
    """
    The four basis weights of the cell whose lower knot is `l`, at coordinate `t`.

    Called by `eval_sdf_c2` and `eval_nsdf_c2`.
    In: t float, a coordinate in grid cells; l float, the cell's lower knot. Out: wp.vec4f.
    """
    f = wp.vec4f()  # wp.vec4f: the weights
    for i in range(4):  # i: int, a weight
        f[i] = eval_basis_1d_wp(t - l - wp.float32(i) + 1.0)
    return f


@wp.func
def basis_dw(t: float, l: float):
    """
    Spline derivative with respect to `t`, which is measured in grid cells. A
    gradient in metres divides these by that axis' cell width `h`.

    Called by `eval_nsdf_c2`.
    In: t float, l float, as `basis_w`. Out: wp.vec4f, per grid cell.
    """
    f = wp.vec4f()  # wp.vec4f: the weights' derivatives
    for i in range(4):  # i: int, a weight
        f[i] = eval_basis_1d_d_wp(t - l - wp.float32(i) + 1.0)
    return f


@wp.func
def eval_sdf_c2(p_b: wp.vec3, B_id: int, mcol: CollisionModel):  # type: ignore
    """
    Tricubic B-spline evaluation of the distance alone.

    Sphere tracing needs the distance and never the normal, so this builds only
    the value weights and skips the three gradient accumulations `eval_nsdf_c2`
    does over the same 64 gathered coefficients.

    Not called in the repository.
    In: p_b wp.vec3, a point in the geometry's body frame (m); B_id int, the geometry; mcol, read for the grid
    parameters, g_nVox_c2_cumsum (nB+1,) int32 and vox_c_sdf_c2 (sum prod(N+3),) float32. Out: float, the
    distance (m).
    """
    lo, up, g, fl = spline_cell(p_b, B_id, mcol)  # wp.vec3i, wp.vec3i, wp.vec3, wp.vec3: see `spline_cell`
    fx = basis_w(g[0], fl[0])  # wp.vec4f: the x weights
    fy = basis_w(g[1], fl[1])  # wp.vec4f: the y weights
    fz = basis_w(g[2], fl[2])  # wp.vec4f: the z weights

    acc = float(0.0)  # float: the spline sum (m)
    for i in range(4):  # i: int, the x offset in the stencil
        i_c = wp.min(i + lo[0], up[0])  # int: the x coefficient index
        for j in range(4):  # j: int, the y offset
            j_c = wp.min(j + lo[1], up[1])  # int: the y coefficient index
            for k in range(4):  # k: int, the z offset
                k_c = wp.min(k + lo[2], up[2])  # int: the z coefficient index
                id = idm_nB_nx_ny_nz_to_nVox_c2(B_id, i_c, j_c, k_c, mcol)  # int: the coefficient's index
                acc += mcol.vox_c_sdf_c2[id] * fx[i] * fy[j] * fz[k]

    _, out = grid_excess(p_b, B_id, mcol)  # wp.vec3 (unused), float: how far `p_b` lies outside the grid (m)
    return acc + out


@wp.func
def eval_nsdf_c2(p_b: wp.vec3, B_id: int, mcol: CollisionModel):  # type: ignore
    """
    Tricubic B-spline evaluation of the signed distance and its gradient at `p_b`.

    The normal is `grad(sdf)`, not a stored field: the direction in which the
    distance grows fastest is perpendicular to the level set, pointing out of the
    solid since `sdf` is positive outside. The tensor-product weight factorizes as
    `fx[i]*fy[j]*fz[k]`, so differentiating along one axis swaps that axis' factor
    for its derivative and leaves the other two alone -- the gradient is the same
    64-term sum three more times over the *same* gathered coefficients. Then
    `dt/dp = 1/h` per axis, each axis carrying its own `h`.

    Called by `compute_nsdf` (in the `compute_nsdf` and `compute_nsdf_r` kernels).
    In: as `eval_sdf_c2`. Out: float, wp.vec3: the distance (m) and the unit outward normal, in the geometry's
    body frame.
    """
    lo, up, g, fl = spline_cell(p_b, B_id, mcol)  # wp.vec3i, wp.vec3i, wp.vec3, wp.vec3: see `spline_cell`
    fx = basis_w(g[0], fl[0])  # wp.vec4f: the x weights
    fy = basis_w(g[1], fl[1])  # wp.vec4f: the y weights
    fz = basis_w(g[2], fl[2])  # wp.vec4f: the z weights
    dfx = basis_dw(g[0], fl[0])  # wp.vec4f: the x weights' derivatives, per grid cell
    dfy = basis_dw(g[1], fl[1])  # wp.vec4f: the y weights' derivatives
    dfz = basis_dw(g[2], fl[2])  # wp.vec4f: the z weights' derivatives

    sdf = float(0.0)  # float: the spline sum (m)
    gx = float(0.0)  # float: its derivative along x, per grid cell
    gy = float(0.0)  # float: along y
    gz = float(0.0)  # float: along z
    for i in range(4):  # i: int, the x offset in the stencil
        i_c = wp.min(i + lo[0], up[0])  # int: the x coefficient index
        for j in range(4):  # j: int, the y offset
            j_c = wp.min(j + lo[1], up[1])  # int: the y coefficient index
            for k in range(4):  # k: int, the z offset
                k_c = wp.min(k + lo[2], up[2])  # int: the z coefficient index
                id = idm_nB_nx_ny_nz_to_nVox_c2(B_id, i_c, j_c, k_c, mcol)  # int: the coefficient's index
                c = mcol.vox_c_sdf_c2[id]  # float: the coefficient
                sdf += c * fx[i] * fy[j] * fz[k]
                gx += c * dfx[i] * fy[j] * fz[k]
                gy += c * fx[i] * dfy[j] * fz[k]
                gz += c * fx[i] * fy[j] * dfz[k]

    h = mcol.g_Vox_h[B_id]  # wp.vec3: voxel edge lengths (m)
    n = wp.vec3(gx / h[0], gy / h[1], gz / h[2])  # wp.vec3: the gradient, then the unit normal
    n = n / wp.max(wp.length(n), 1.0e-6)

    # Outside the grid the field is the boundary value plus the distance to the
    # box, so its gradient is the direction out of the box.
    d, out = grid_excess(p_b, B_id, mcol)  # wp.vec3, float: the offset out of the grid box and its length (m)
    if out > 0.0:
        n = d / out

    return sdf + out, n


@wp.func
def eval_sdf_c0(p_b: wp.vec3, B_id: int, mcol: CollisionModel):  # type: ignore
    """
    Trilinear evaluation of the distance alone, over the packed knot table.

    The cell setup is `spline.c0.cell_cs1` itself rather than a packed copy of it --
    the grid parameters are all it needs and the model has them per geometry --
    so the only thing this adds to the per-surface evaluator is the indexing.

    Not called in the repository.
    In: p_b wp.vec3, a point in the geometry's body frame (m); B_id int, the geometry; mcol, read for the grid
    parameters, g_nVox_c0_cumsum (nB+1,) int32 and vox_c_sdf_c0 (sum prod(N+1),) float32. Out: float, the
    distance (m).
    """
    lo, u = cell_cs1(  # wp.vec3i, wp.vec3: the knot cell's lower corner, the local coordinates in [0, 1]^3
        p_b, mcol.g_Vox_pmin[B_id], mcol.g_Vox_N[B_id], mcol.g_Vox_h[B_id]
    )
    fx = basis_w_cs1(u[0])  # wp.vec2: the x corner weights
    fy = basis_w_cs1(u[1])  # wp.vec2: the y corner weights
    fz = basis_w_cs1(u[2])  # wp.vec2: the z corner weights

    acc = float(0.0)  # float: the interpolated distance (m)
    for i in range(2):  # i: int, the x corner
        for j in range(2):  # j: int, the y corner
            for k in range(2):  # k: int, the z corner
                id = idm_nB_nx_ny_nz_to_nVox_c0(  # int: the knot sample's index
                    B_id, lo[0] + i, lo[1] + j, lo[2] + k, mcol
                )
                acc += mcol.vox_c_sdf_c0[id] * fx[i] * fy[j] * fz[k]

    _, out = grid_excess(p_b, B_id, mcol)  # wp.vec3 (unused), float: how far `p_b` lies outside the grid (m)
    return acc + out


@wp.func
def eval_nsdf_c0(p_b: wp.vec3, B_id: int, mcol: CollisionModel):  # type: ignore
    """
    Trilinear evaluation of the signed distance and its gradient at `p_b`.

    The same structure as `eval_nsdf_c2` over 8 samples instead of 64: the corner
    weight factorizes per axis, so differentiating along one swaps that axis'
    factor and leaves the others alone, and the gradient costs three more sums
    over samples already gathered. Then `du/dp = 1/h` per axis.

    Unlike the tricubic's, this gradient is *discontinuous across cell faces* --
    trilinear interpolation is C0 -- so a point sliding along the surface sees the
    normal step roughly once per cell crossed. See `spline.c0`.

    Called by `compute_nsdf` (in the `compute_nsdf_c0` and `compute_nsdf_c0_r` kernels).
    In: as `eval_sdf_c0`. Out: float, wp.vec3: the distance (m) and the unit outward normal, in the geometry's
    body frame.
    """
    lo, u = cell_cs1(  # wp.vec3i, wp.vec3: the knot cell's lower corner, the local coordinates in [0, 1]^3
        p_b, mcol.g_Vox_pmin[B_id], mcol.g_Vox_N[B_id], mcol.g_Vox_h[B_id]
    )
    fx, dfx = basis_w_cs1(u[0]), basis_dw_cs1(u[0])  # wp.vec2, wp.vec2: the x weights and their derivatives
    fy, dfy = basis_w_cs1(u[1]), basis_dw_cs1(u[1])  # wp.vec2, wp.vec2: the y weights and their derivatives
    fz, dfz = basis_w_cs1(u[2]), basis_dw_cs1(u[2])  # wp.vec2, wp.vec2: the z weights and their derivatives

    sdf = float(0.0)  # float: the interpolated distance (m)
    gx = float(0.0)  # float: its derivative along x, per grid cell
    gy = float(0.0)  # float: along y
    gz = float(0.0)  # float: along z
    for i in range(2):  # i: int, the x corner
        for j in range(2):  # j: int, the y corner
            for k in range(2):  # k: int, the z corner
                id = idm_nB_nx_ny_nz_to_nVox_c0(  # int: the knot sample's index
                    B_id, lo[0] + i, lo[1] + j, lo[2] + k, mcol
                )
                c = mcol.vox_c_sdf_c0[id]  # float: the knot sample (m)
                sdf += c * fx[i] * fy[j] * fz[k]
                gx += c * dfx[i] * fy[j] * fz[k]
                gy += c * fx[i] * dfy[j] * fz[k]
                gz += c * fx[i] * fy[j] * dfz[k]

    h = mcol.g_Vox_h[B_id]  # wp.vec3: voxel edge lengths (m)
    n = wp.vec3(gx / h[0], gy / h[1], gz / h[2])  # wp.vec3: the gradient, then the unit normal
    n = n / wp.max(wp.length(n), 1.0e-6)

    d, out = grid_excess(p_b, B_id, mcol)  # wp.vec3, float: the offset out of the grid box and its length (m)
    if out > 0.0:
        n = d / out

    return sdf + out, n


def _make_nsdf_kernel(eval_nsdf, with_radius: bool = False):
    """The narrowphase, bound to one SDF evaluator and one radius policy.

    `with_radius` is a separate specialization for the same reason the filter has
    one: this launch is the other gather-bound stage, and a context without
    spheres should not read a zero radius per candidate. See
    `broadphase._make_tiled_sdf_filter`.

    Called by this module at import, for the four kernels below it.
    In: eval_nsdf, a `wp.func` (p_b wp.vec3, B_id int, mcol) -> (float, wp.vec3): `eval_nsdf_c2` or
    `eval_nsdf_c0`; with_radius bool, subtract `mcol.prim_r`. Out: the `compute_nsdf` kernel.
    """

    @wp.kernel
    def compute_nsdf(mcol: CollisionModel, dcol: CollisionData):
        """
        The narrowphase: each stage-2 candidate's gap and normal from the context's SDF evaluator, and whether it
        penetrates.

        Launch: dim=(nCa2_max,), a thread per stage-2 slot, by `collide.narrowphase_sdf` (through `nsdf_kernel`).
        Reads: dcol.ct[1] and dyn_scan, dcol.idm_nCa2_to_nCa1 (nCa2_max,) int32, dcol.T_rel (B, nSEG) wp.transformf;
        mcol.idm_nC_to_prim (nC,) int32, v_b (nV,) wp.vec3, prim_r (nV,) float32 if `with_radius`, seg_g_s (nSEG,)
        int32, g_plane (nB,) int32 and the SDF tables. Writes: dcol.sdf_nCa2 (nCa2_max,) float32, n_nCa2
        (nCa2_max,) wp.vec3 in the SDF geometry's body frame, and flag2 (nCa2_max,) int32.
        """
        id_nCa2 = wp.tid()  # int: the stage-2 slot

        # Clear the tail only where the scan that follows will read it: with a
        # scan bounded by `ct[1]` nothing past the count is summed, so the write
        # is pure capacity-proportional waste. See `compact._scan`.
        if id_nCa2 >= dcol.ct[1]:
            if dcol.dyn_scan == 0:
                dcol.flag2[id_nCa2] = 0
            return

        id_nCa1 = dcol.idm_nCa2_to_nCa1[id_nCa2]  # int: its stage-1 slot
        # int, int, int, int: the world, the candidate (an index into nC), the segment, the surviving tile
        id_Batch, id_nC, id_SEG, id_Tile = resolve_nCa1(id_nCa1, mcol, dcol)

        # The primitive, in the target surface's body frame, straight from body
        # coords. For a sphere that is its centre.
        id_nV = mcol.idm_nC_to_prim[id_nC]  # int: the primitive, an index into v_b
        v_b = wp.transform_point(dcol.T_rel[id_Batch, id_SEG], mcol.v_b[id_nV])  # wp.vec3

        id_nB_2 = mcol.seg_g_s[id_SEG]  # int: the SDF geometry

        # Compute NSDF. A half-space is the trivial case: the signed distance is the
        # height above the plane in its own frame, and the normal is constant.
        sdf = float(0.0)  # float: the gap (m)
        n = wp.vec3(0.0, 0.0, 1.0)  # wp.vec3: the outward normal, in the SDF geometry's body frame
        if mcol.g_plane[id_nB_2] == 1:
            sdf = v_b[2]
        else:
            sdf, n = eval_nsdf(v_b, id_nB_2, mcol)

        # A sphere of radius `r` centred at `p` is the r-offset of `p`, so its
        # gap to the surface is exactly `sdf(p) - r` -- one evaluation, no
        # sampling. Everything downstream reads that gap and needs to know
        # nothing else; a vertex is the r = 0 case.
        if wp.static(with_radius):
            sdf -= mcol.prim_r[id_nV]

        # Store in memory
        dcol.sdf_nCa2[id_nCa2] = sdf
        dcol.n_nCa2[id_nCa2] = n

        # Penetrating contacts survive to stage 3.
        if sdf < 0.0:
            dcol.flag2[id_nCa2] = 1
        else:
            dcol.flag2[id_nCa2] = 0

    return compute_nsdf


compute_nsdf = _make_nsdf_kernel(eval_nsdf_c2)  # the narrowphase: tricubic SDF, no radius read
compute_nsdf_c0 = _make_nsdf_kernel(eval_nsdf_c0)  # trilinear SDF, no radius read
compute_nsdf_r = _make_nsdf_kernel(eval_nsdf_c2, with_radius=True)  # tricubic, with the radius read
compute_nsdf_c0_r = _make_nsdf_kernel(eval_nsdf_c0, with_radius=True)  # trilinear, with the radius read


#######################################################################################
# Kernel selection
#######################################################################################
#
# Which specialization a context runs. Here rather than in `collide` so that the
# reason for each specialization sits beside the specialization itself.


def nsdf_kernel(mcol: CollisionModel):
    """
    The narrowphase for this context's SDF representation and primitives.

    Called by `collide.narrowphase_sdf`.
    In: mcol, read for has_spheres and spline. Out: one of the four `compute_nsdf*` kernels.
    """
    if int(mcol.has_spheres):
        return compute_nsdf_c0_r if int(mcol.spline) == 1 else compute_nsdf_r
    return compute_nsdf_c0 if int(mcol.spline) == 1 else compute_nsdf


#######################################################################################
# Contact Output
#######################################################################################


def _make_gather_contacts(contact_point: str):
    """The contact emitter, bound to one placement convention.

    A specialization rather than a branch, so each variant is exactly the kernel
    it would be if the other did not exist -- the same reason the filter and the
    narrowphase have `with_radius` pairs. `wp.static` resolves the choice when the
    kernel is built, and `make_collision_context` makes it once per context, well
    before graph capture.

    Called by this module at import, for the two kernels below it.
    In: contact_point str, one of `CONTACT_POINTS`. Out: the `gather_contacts` kernel.
    """
    assert contact_point in CONTACT_POINTS, (
        f"contact_point must be one of {CONTACT_POINTS}, got {contact_point!r}"
    )
    midpoint = contact_point == "midpoint"  # bool: place a sphere's contact at its witness points' midpoint

    @wp.kernel
    def gather_contacts(mcol: CollisionModel, dcol: CollisionData):
        """
        Builds the final world-space contact list over the surviving contacts only.

        Positions are formed here rather than by transforming every (B, nV) vertex
        beforehand: a step produces a few thousand contacts out of millions of
        candidate vertices, and by this point they are already compacted.

        `eval_nsdf_c2` returns normals in the SDF geometry's body frame, so they are
        rotated to world here by that geometry's world rotation, `n_w = R_w n_b`.

        Normals are left in the sense the contact was actually computed in: a vertex of
        the pair's first geometry carries the outward normal of its second. An
        interface declared both ways is two pairs, and their normals point opposite
        ways -- not canonicalized, since each contact's force must be evaluated
        against the surface it penetrated.

        **Where the contact is placed**, of the three points on the segment
        between a sphere's two witness points, with `phi = sdf + r`:

            p - n r              the primitive's own surface   ("surface")
            p - n (sdf + r) / 2  the midpoint                  ("midpoint")
            p - n (sdf + r)      the field's surface, unused

        Depth and normal are identical wherever it goes; only the lever arm
        differs. `"surface"` is the default and needs no branch: at `r = 0` it is
        the vertex itself, which is where a vertex contact belongs. The
        midpoint does not degenerate -- at `r = 0` it would land half the
        penetration depth off the vertex -- so that variant moves a contact only
        where its primitive has a radius. Each pays only
        for its own convention.

        Launch: dim=(nCa3_max,), a thread per stage-3 slot, by `collide.narrowphase_sdf` (through `contact_kernel`).
        Reads: dcol.ct[2], idm_nCa3_to_nCa2 (nCa3_max,) int32, idm_nCa2_to_nCa1 (nCa2_max,) int32, T_w (B, nB)
        wp.transformf, sdf_nCa2 (nCa2_max,) float32 and n_nCa2 (nCa2_max,) wp.vec3; mcol.idm_nC_to_prim (nC,)
        int32, v_b (nV,) wp.vec3, prim_r (nV,) float32, seg_g_v and seg_g_s (nSEG,) int32. Writes: dcol.p_nCa3
        and n_nCa3 (nCa3_max,) wp.vec3, sdf_nCa3 (nCa3_max,) float32, b_nCa3, seg_nCa3 and v_nCa3 (nCa3_max,) int32.
        """
        id_nCa3 = wp.tid()  # int: the stage-3 slot, a contact
        if id_nCa3 >= dcol.ct[2]:
            return

        id_nCa2 = dcol.idm_nCa3_to_nCa2[id_nCa3]  # int: its stage-2 slot
        id_nCa1 = dcol.idm_nCa2_to_nCa1[id_nCa2]  # int: its stage-1 slot
        # int, int, int, int: the world, the candidate (an index into nC), the segment, the surviving tile
        id_Batch, id_nC, id_SEG, id_Tile = resolve_nCa1(id_nCa1, mcol, dcol)

        id_nV = mcol.idm_nC_to_prim[id_nC]  # int: the primitive, an index into v_b

        # A contact sits on the primitive itself, so it is placed with that
        # geometry's own pose rather than transformed out of the SDF geometry's frame.
        p = wp.transform_point(dcol.T_w[id_Batch, mcol.seg_g_v[id_SEG]], mcol.v_b[id_nV])  # wp.vec3: in world (m)

        # Surface normal: out of the SDF geometry's body frame, into world.
        n = wp.quat_rotate(  # wp.vec3: the world surface normal
            wp.transform_get_rotation(dcol.T_w[id_Batch, mcol.seg_g_s[id_SEG]]),
            dcol.n_nCa2[id_nCa2],
        )

        if wp.static(midpoint):
            r = mcol.prim_r[id_nV]  # float: the primitive's radius (m)
            if r > 0.0:
                p -= n * (0.5 * dcol.sdf_nCa2[id_nCa2] + r)
        else:
            # Exactly `p` for a vertex, since `prim_r` is zero there: no branch,
            # and the vertex's own position to the bit.
            p -= n * mcol.prim_r[id_nV]

        dcol.p_nCa3[id_nCa3] = p
        dcol.n_nCa3[id_nCa3] = n
        dcol.sdf_nCa3[id_nCa3] = dcol.sdf_nCa2[id_nCa2]
        dcol.b_nCa3[id_nCa3] = id_Batch
        dcol.seg_nCa3[id_nCa3] = id_SEG
        dcol.v_nCa3[id_nCa3] = mcol.idm_nC_to_prim[id_nC]

    return gather_contacts


gather_contacts = _make_gather_contacts("surface")  # the contact emitter: on the primitive's own surface
gather_contacts_midpoint = _make_gather_contacts("midpoint")  # a sphere's contact at its witness points' midpoint


def contact_kernel(mcol: CollisionModel):
    """
    The contact emitter this context runs.

    Called by `collide.narrowphase_sdf`.
    In: mcol, read for contact_point. Out: `gather_contacts` or `gather_contacts_midpoint`.
    """
    return (gather_contacts_midpoint if int(mcol.contact_point) == 1
            else gather_contacts)
