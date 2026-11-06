import math
import numpy as np
import warp as wp


from lodos.col.sdf.spline import c0 as spline_c0, c2 as spline_c2

#######################################################################################
# Data Structures
#######################################################################################


@wp.struct
class GridParams:
    """
    Along each axis (x,y,z), the grid has:

    - Nx, Ny, Nz intervals

    - Nx+1, Ny+1, Nz+1 knot points (that sit on interval endpoints)

    - p_min/max sit at the corners of the voxel,
      constituting the 0th/(N)th knot point

    The voxel sizes can be derived from p_max - p_min = N * h

    A surface's voxel grid, in the surface's frame (m). Built by `create_voxel_grid_from_verts` and
    `lodos.col.sdf.implicit_surface`; held in `ImplicitSurface.grid_params`.
    """

    p_min: wp.vec3  # minimum knot point
    p_max: wp.vec3  # maximum knot point
    N: wp.vec3i  # (3,) number of voxels per axis
    h: wp.vec3  # (3,) voxel edge lengths per axis
    r: wp.float32  # voxel circumsphere radius


@wp.struct
class ImplicitSurface:
    """
    - The cubic spline coefficients for the SDF correspond to Nx+3, Ny+3, Nz+3 basis functions
      (that are centered on the N+1 knot points within the grid
      and 2 more auxiliary knots extending out of the grid along each axis)

    - The SDF values for the VSDF are evaluated at the center point of each voxel

    **Two SDF representations over one lattice.** `spline` picks which of the two
    the evaluators below read: the tricubic coefficients (`spline.c2`, the default: C2
    field, C1 normals), or the knot samples those coefficients interpolate (`spline.c0`,
    trilinear: C0 field, normals discontinuous across cell faces). They describe the same
    sampled field to different orders -- see `spline.c0` for what losing the normals'
    continuity costs -- and `spline.c0.knots_from_c_3d` recovers the
    samples exactly from the coefficients, so a surface baked for one can be
    evaluated with the other.

    `lodos.col.sdf.implicit_surface` populates both; `create_implicit_surface_from_mesh`
    only the array the live representation needs, the other a 1x1x1 placeholder no
    evaluator reads.

    One surface's signed distance field on its voxel grid, for the per-surface evaluators below. Built by
    `create_implicit_surface_from_mesh` and `lodos.col.sdf.implicit_surface`.
    """

    # Voxel grid
    grid_params: GridParams  # voxel grid for SDF and VSDF evaluations

    # 0 = tricubic B-spline over `sdf_coeffs`, 1 = trilinear over `sdf_knots`.
    spline: wp.int32

    # Signed-distance cubic spline coefficients.
    sdf_coeffs: wp.array(dtype=float, ndim=3)  # (Nx+3, Ny+3, Nz+3) # type: ignore

    # Signed distance sampled at the grid knots, for the trilinear evaluation.
    sdf_knots: wp.array(dtype=float, ndim=3)  # (Nx+1, Ny+1, Nz+1) # type: ignore

    # VSDF values
    vsdf_values: wp.array(dtype=float, ndim=3)  # (Nx, Ny, Nz) # type: ignore

    # The unit normal sampled at the same voxel centres -- the **vox** normal, as
    # `vsdf_values` is the vox distance. Baked, so that vox is a quantized c2 and
    # the three evaluations differ only in how much of the field they reconstruct.
    #
    # Left empty by anything that does not bake it: only the vox normal path
    # reads it, and nothing here takes that path. Nothing here bakes it either.
    vsnf_values: wp.array(dtype=wp.vec3, ndim=3)  # (Nx, Ny, Nz) # type: ignore


#######################################################################################
# Creation
#######################################################################################


def print_aabb_info(v):
    """
    Prints the bounds, extent and centre of the points' axis-aligned bounding box.

    Not called in the repository.
    In: v (V, 3) float numpy array, the points.
    """
    aabb_bounds = np.stack([v.min(axis=0), v.max(axis=0)], axis=0)  # (2, 3)
    aabb_center = aabb_bounds.sum(axis=0) / 2  # (3,)
    aabb_extent = aabb_bounds[1] - aabb_bounds[0]  # (3,)

    print("AABB Info")
    print("----------------------")
    print("Bounds: {}".format(aabb_bounds))
    print("Extent: {}".format(aabb_extent))
    print("Center: {}".format(aabb_center))


def create_voxel_grid_from_verts(
    v, N=None, h=None, grid_aabb=None, aabb_dilation=2, d_max=0.0, verbose=True
):
    """
    v: np.array of shape(V, 3)
    h: float or (3,)
    N: int or (3,), the number of intervals in the grid (hence N+1 points per grid axis)
    grid_aabb: (2, 3), aabb[0] is p_min, aabb[1] is p_max

    N overrides h if both are provided.
    grid_aabb overrides aabb_dilation if both are provided.

    The voxel grid to bake a surface on, around the points `v`: the box `grid_aabb`, or else `v`'s bounding box
    scaled by `aabb_dilation` about its centre and grown by `d_max` (m) on every side; cut into `N` intervals per
    axis, or into cells of edge `h` (m) rounded up to whole cells, the box grown evenly to fit. Prints the grid
    when `verbose`.

    Called by `tools.mesh_processing.create_surface_data.bake` and `tools.mesh_processing.nutbolt.bake.build`.
    Out: the `GridParams`; the knot coordinates per axis, a list of (Nx+1,), (Ny+1,), (Nz+1,) float32 Warp arrays;
    the voxel-centre coordinates per axis, a list of (Nx,), (Ny,), (Nz,) float32 Warp arrays; and the knots along
    six edges of the box, for drawing, a (2 * (Nx + Ny + Nz + 3), 3) float numpy array. Raises ValueError without
    `N` or `h`, or for an `N` or `h` that is neither one value nor three.
    """

    # Parse Arguments and create voxel grid object
    aabb_bounds = np.stack([v.min(axis=0), v.max(axis=0)], axis=0)  # (2, 3)
    aabb_center = aabb_bounds.sum(axis=0) / 2  # (3,)

    if grid_aabb is None:
        p_min = aabb_center + (aabb_bounds[0] - aabb_center) * aabb_dilation - d_max  # (3,): the lower corner
        p_max = aabb_center + (aabb_bounds[1] - aabb_center) * aabb_dilation + d_max  # (3,): the upper corner

    else:
        p_min, p_max = grid_aabb[0], grid_aabb[1]

    grid_extent = p_max - p_min  # (3,)

    if h is None and N is None:
        raise ValueError("create_voxel_grid_from_verts takes N or h")

    elif N is not None:
        if np.shape(N) == (3,):
            Nx, Ny, Nz = (int(n) for n in N)  # int, int, int: intervals per axis
        elif np.ndim(N) == 0:
            Nx, Ny, Nz = int(N), int(N), int(N)
        else:
            raise ValueError(f"N must be one integer or three, not {N!r}")

        hx, hy, hz = grid_extent[0] / Nx, grid_extent[1] / Ny, grid_extent[2] / Nz  # float, float, float (m)

    else:
        if np.shape(h) == (3,):
            hx, hy, hz = (float(x) for x in h)
        elif np.ndim(h) == 0:
            hx, hy, hz = float(h), float(h), float(h)
        else:
            raise ValueError(f"h must be one length or three, not {h!r}")

        Nx, Ny, Nz = (
            math.ceil(grid_extent[0] / hx),
            math.ceil(grid_extent[1] / hy),
            math.ceil(grid_extent[2] / hz),
        )
        grid_extent_new = np.array([hx * Nx, hy * Ny, hz * Nz])  # (3,)
        grid_extent_delta = grid_extent_new - grid_extent  # (3,)
        p_max = p_max + grid_extent_delta / 2
        p_min = p_min - grid_extent_delta / 2
        grid_extent = grid_extent_new

    voxel_grid_params = GridParams()  # GridParams
    voxel_grid_params.p_min = wp.vec3(p_min)
    voxel_grid_params.p_max = wp.vec3(p_max)
    voxel_grid_params.N = wp.vec3i(Nx, Ny, Nz)
    voxel_grid_params.h = wp.vec3(hx, hy, hz)
    voxel_grid_params.r = wp.float32(max(hx, hy, hz) * 0.5 * np.sqrt(3))

    # Print grid parameters
    if verbose:
        print("Grid Parameters")
        print("----------------------")
        print("AABB Bounds: {}".format(aabb_bounds))
        print("AABB Extent: {}".format(aabb_bounds[1] - aabb_bounds[0]))
        print("Grid Bounds: {}".format(np.stack([p_min, p_max], axis=0)))
        print("Grid Extent: {}".format(grid_extent))
        print("N: {}".format(np.array([Nx, Ny, Nz])))
        print("h: {}".format(np.array([hx, hy, hz])))

    # Create grid knot points for NSDF evals
    x_grid_knot_wp = wp.array1d(np.linspace(p_min[0], p_max[0], Nx + 1), dtype=float)  # (Nx+1,) float32
    y_grid_knot_wp = wp.array1d(np.linspace(p_min[1], p_max[1], Ny + 1), dtype=float)  # (Ny+1,) float32
    z_grid_knot_wp = wp.array1d(np.linspace(p_min[2], p_max[2], Nz + 1), dtype=float)  # (Nz+1,) float32
    grid_knots_wp = [x_grid_knot_wp, y_grid_knot_wp, z_grid_knot_wp]  # list of the three

    # Create grid center points for VSDF evals
    x_grid_center_wp = wp.array1d(  # (Nx,) float32
        np.linspace(p_min[0] + hx / 2, p_max[0] - hx / 2, Nx), dtype=float
    )
    y_grid_center_wp = wp.array1d(  # (Ny,) float32
        np.linspace(p_min[1] + hy / 2, p_max[1] - hy / 2, Ny), dtype=float
    )
    z_grid_center_wp = wp.array1d(  # (Nz,) float32
        np.linspace(p_min[2] + hz / 2, p_max[2] - hz / 2, Nz), dtype=float
    )
    grid_centers_wp = [x_grid_center_wp, y_grid_center_wp, z_grid_center_wp]  # list of the three

    # Create grid visualization points
    x_grid_np1 = np.stack(  # (Nx+1, 3): the knots along the x edge through p_min
        [
            x_grid_knot_wp.numpy(),
            np.repeat(p_min[1], Nx + 1),
            np.repeat(p_min[2], Nx + 1),
        ],
        axis=-1,
    )

    x_grid_np2 = np.stack(  # (Nx+1, 3): the x edge through p_max
        [
            x_grid_knot_wp.numpy(),
            np.repeat(p_max[1], Nx + 1),
            np.repeat(p_max[2], Nx + 1),
        ],
        axis=-1,
    )

    y_grid_np1 = np.stack(  # (Ny+1, 3): the y edge through p_min
        [
            np.repeat(p_min[0], Ny + 1),
            y_grid_knot_wp.numpy(),
            np.repeat(p_min[2], Ny + 1),
        ],
        axis=-1,
    )

    y_grid_np2 = np.stack(  # (Ny+1, 3): the y edge through p_max
        [
            np.repeat(p_max[0], Ny + 1),
            y_grid_knot_wp.numpy(),
            np.repeat(p_max[2], Ny + 1),
        ],
        axis=-1,
    )

    z_grid_np1 = np.stack(  # (Nz+1, 3): the z edge through p_min
        [
            np.repeat(p_min[0], Nz + 1),
            np.repeat(p_min[1], Nz + 1),
            z_grid_knot_wp.numpy(),
        ],
        axis=-1,
    )

    z_grid_np2 = np.stack(  # (Nz+1, 3): the z edge through p_max
        [
            np.repeat(p_max[0], Nz + 1),
            np.repeat(p_max[1], Nz + 1),
            z_grid_knot_wp.numpy(),
        ],
        axis=-1,
    )

    grid_vis_np = np.concatenate(  # (2 * (Nx + Ny + Nz + 3), 3): the six edges
        [x_grid_np1, x_grid_np2, y_grid_np1, y_grid_np2, z_grid_np1, z_grid_np2], axis=0
    )

    return voxel_grid_params, grid_knots_wp, grid_centers_wp, grid_vis_np


@wp.kernel
def compute_grid_knot_sdf(
    wp_mesh: wp.uint64,
    x_grid_knot: wp.array(dtype=float),  # type: ignore
    y_grid_knot: wp.array(dtype=float),  # type: ignore
    z_grid_knot: wp.array(dtype=float),  # type: ignore
    knot_sdf: wp.array(dtype=float, ndim=3),  # type: ignore
):
    """
    knot_sdf: (Nx+1, Ny+1, Nz+1), the signed distance at each spline knot.

    Launch: dim=knot_sdf.shape = (Nx+1, Ny+1, Nz+1), a thread per knot, by `create_implicit_surface_from_mesh`.
    In: wp_mesh, the `wp.Mesh` id; x_grid_knot (Nx+1,), y_grid_knot (Ny+1,), z_grid_knot (Nz+1,) float32, the
    knots' coordinates (m). Out: knot_sdf (Nx+1, Ny+1, Nz+1) float32, the distance to the nearest point of the mesh
    (m), negative inside; a knot with no face within 1e6 m is left as it was.
    """

    i, j, k = wp.tid()  # int, int, int: the knot

    point = wp.vec3(  # wp.vec3: the knot's position
        x_grid_knot[i],
        y_grid_knot[j],
        z_grid_knot[k],
    )

    # Query the nearest location on the mesh using BVH.
    query = wp.mesh_query_point(wp_mesh, point, max_dist=1e6)  # wp.MeshQueryPoint: the nearest face and the sign
    if not query.result:
        return

    # Evaluate the position of the nearest location found.
    nearest_pos = wp.mesh_eval_position(  # wp.vec3: the nearest point on the mesh
        wp_mesh,
        query.face,
        query.u,
        query.v,
    )

    # Compute the SDF value: positive outside, negative inside.
    dist = wp.length(point - nearest_pos)  # float

    sdf = dist  # float
    if query.sign < 0:
        sdf = -dist

    knot_sdf[i, j, k] = sdf


@wp.kernel
def compute_grid_center_sdf(
    wp_mesh: wp.uint64,
    x_grid_center: wp.array(dtype=float),  # type: ignore
    y_grid_center: wp.array(dtype=float),  # type: ignore
    z_grid_center: wp.array(dtype=float),  # type: ignore
    center_sdf: wp.array(dtype=float, ndim=3),  # type: ignore
):
    """
    The signed distance from each voxel centre to the mesh: the vox samples, `vsdf_values`.

    Launch: dim=vsdf_values.shape = (Nx, Ny, Nz), a thread per voxel, by `create_implicit_surface_from_mesh`.
    In: wp_mesh, the `wp.Mesh` id; x_grid_center (Nx,), y_grid_center (Ny,), z_grid_center (Nz,) float32, the
    voxel centres' coordinates (m). Out: center_sdf (Nx, Ny, Nz) float32 (m), negative inside; a voxel with no face
    within 1e6 m is left as it was.
    """
    i, j, k = wp.tid()  # int, int, int: the voxel

    point = wp.vec3(  # wp.vec3: the voxel's centre
        x_grid_center[i],
        y_grid_center[j],
        z_grid_center[k],
    )

    # Query the nearest location on the mesh using BVH.
    query = wp.mesh_query_point(wp_mesh, point, max_dist=1e6)  # wp.MeshQueryPoint: the nearest face and the sign
    if not query.result:
        return

    # Evaluate the position of the nearest location found.
    nearest_pos = wp.mesh_eval_position(  # wp.vec3: the nearest point on the mesh
        wp_mesh,
        query.face,
        query.u,
        query.v,
    )

    # Compute the SDF value.
    dist = wp.length(nearest_pos - point)  # float
    if query.sign >= 0:
        sdf = dist  # float
    else:
        sdf = -dist

    center_sdf[i, j, k] = sdf


def create_implicit_surface_from_mesh(
    v,
    f,
    voxel_grid_params,
    grid_knots_wp,
    grid_centers_wp,
):
    """
    Bakes the triangle mesh (`v`, `f`) on the grid: the signed distance at the knots, fitted with the tricubic
    B-spline (`spline.c2.spline_c_3d`), and at the voxel centres. The sign is `wp.mesh_query_point`'s, so the mesh must
    be closed and outward-facing.

    Called by `tools.mesh_processing.create_surface_data.bake` and `tools.mesh_processing.nutbolt.bake.build`.
    In: v (V, 3) float and f (F, 3) int numpy arrays; voxel_grid_params, grid_knots_wp and grid_centers_wp as
    `create_voxel_grid_from_verts` returns them.
    Out: an `ImplicitSurface` with spline 0, sdf_coeffs (Nx+3, Ny+3, Nz+3) float32, vsdf_values (Nx, Ny, Nz)
    float32 and a (1, 1, 1) placeholder for sdf_knots; vsnf_values is not set.
    """

    # Create warp mesh
    v_wp = wp.array(v, dtype=wp.vec3)  # (V,) wp.vec3
    f_wp = wp.array(f.flatten(), dtype=int)  # (3 F,) int32
    wp_mesh = wp.Mesh(  # wp.Mesh, with its BVH
        points=v_wp,
        indices=f_wp,
    )

    # Compute knot SDF values
    knot_sdf = wp.zeros(  # (Nx+1, Ny+1, Nz+1) float32
        [
            voxel_grid_params.N[0] + 1,
            voxel_grid_params.N[1] + 1,
            voxel_grid_params.N[2] + 1,
        ]
    )
    wp.launch(
        kernel=compute_grid_knot_sdf,
        dim=knot_sdf.shape,
        inputs=(
            wp_mesh.id,
            grid_knots_wp[0],
            grid_knots_wp[1],
            grid_knots_wp[2],
            knot_sdf,
        ),
    )

    # Compute SDF spline coefficients.
    sdf_c = wp.array(  # (Nx+3, Ny+3, Nz+3) float32
        np.asarray(
            spline_c2.spline_c_3d(
                knot_sdf.numpy(),
                voxel_grid_params.N[0],
                voxel_grid_params.N[1],
                voxel_grid_params.N[2],
            )
        ),
        dtype=wp.float32,
    )

    # Compute VSDF values
    vsdf_values = wp.zeros(  # (Nx, Ny, Nz) float32
        [voxel_grid_params.N[0], voxel_grid_params.N[1], voxel_grid_params.N[2]]
    )
    wp.launch(
        kernel=compute_grid_center_sdf,
        dim=vsdf_values.shape,
        inputs=(
            wp_mesh.id,
            grid_centers_wp[0],
            grid_centers_wp[1],
            grid_centers_wp[2],
            vsdf_values,
        ),
    )

    imps = ImplicitSurface()  # ImplicitSurface
    imps.grid_params = voxel_grid_params
    imps.spline = wp.int32(0)
    imps.sdf_coeffs = sdf_c
    imps.sdf_knots = wp.zeros((1, 1, 1), dtype=wp.float32)
    imps.vsdf_values = vsdf_values

    return imps


#######################################################################################
# Evaluation
#######################################################################################


@wp.func
def grid_excess(pq: wp.vec3, p_min: wp.vec3, N: wp.vec3i, h: wp.vec3):  # type: ignore
    """
    How far `pq` lies outside the grid box, as a vector and a length. Zero inside.

    The per-surface twin of `lodos.col.sdf.narrowphase.grid_excess`, whose
    docstring says why a baked field needs it: outside its grid the evaluation is
    meaningless, and both representations report something small there -- `spline.c0` and
    `spline.c2` alike a constant extension of the boundary. Adding the distance
    to the box makes the outside grow with true separation. Note the sum is an
    *upper* bound on the true distance, by the triangle inequality; the bare box
    distance is the lower one.

    Called by `eval_sdf`, `eval_nsdf`, `eval_sdf_vox` and 5 others.
    In: pq wp.vec3, the query (m); p_min wp.vec3, N wp.vec3i, h wp.vec3, the grid (`GridParams`).
    Out: wp.vec3, `pq` minus its clamp into the box, and float, that vector's length (m).
    """
    d = wp.vec3(  # wp.vec3: pq minus its clamp into the box
        pq[0] - wp.clamp(pq[0], p_min[0], p_min[0] + h[0] * wp.float32(N[0])),
        pq[1] - wp.clamp(pq[1], p_min[1], p_min[1] + h[1] * wp.float32(N[1])),
        pq[2] - wp.clamp(pq[2], p_min[2], p_min[2] + h[2] * wp.float32(N[2])),
    )
    return d, wp.length(d)


#######################################################################################
# The three evaluations, named rather than dispatched
#######################################################################################
#
# `eval_sdf` and `eval_nsdf` branch on `imps.spline`, one representation per
# surface. These name one instead, so a caller can take its distances from
# one and its normals from another -- marching on cheap vox distances, say, while
# the axis it builds comes from c2 normals. Nothing here calls these.
#
#   vox  the value stored at the voxel centre. One load, piecewise constant.
#   c0   trilinear over the knots; its gradient is discontinuous across cells.
#   c2   the tricubic B-spline; its gradient is smooth.
#
# All three agree outside the grid, where the field is the boundary value plus
# the distance to the box -- `grid_excess`, without which a geometry hundreds of
# millimetres away reads as lying on the surface.


@wp.func
def eval_sdf_vox(pq: wp.vec3, imps: ImplicitSurface):  # type: ignore
    """
    The vox signed distance at `pq`: the value at the voxel containing it, plus the distance to the grid box.

    Not called in the repository.
    In: pq wp.vec3, the query (m); imps, reads grid_params and vsdf_values (Nx, Ny, Nz) float32. Out: float (m).
    """
    # wp.vec3, float: the excess past the grid box, its length
    _, out = grid_excess(pq, imps.grid_params.p_min, imps.grid_params.N,
                         imps.grid_params.h)
    return eval_vsdf(pq, imps) + out


@wp.func
def eval_sdf_c0(pq: wp.vec3, imps: ImplicitSurface):  # type: ignore
    """
    The c0 signed distance at `pq`: trilinear over the knot samples, plus the distance to the grid box.

    Not called in the repository.
    In: pq wp.vec3, the query (m); imps, reads grid_params and sdf_knots (Nx+1, Ny+1, Nz+1) float32.
    Out: float (m).
    """
    gp = imps.grid_params  # GridParams
    _, out = grid_excess(pq, gp.p_min, gp.N, gp.h)  # wp.vec3, float: the excess past the grid box, its length
    return spline_c0.eval_cs1(pq, gp.p_min, gp.N, gp.h, imps.sdf_knots) + out


@wp.func
def eval_sdf_c2(pq: wp.vec3, imps: ImplicitSurface):  # type: ignore
    """
    The c2 signed distance at `pq`: the tricubic B-spline over the coefficients, plus the distance to the grid box.

    Not called in the repository.
    In: pq wp.vec3, the query (m); imps, reads grid_params and sdf_coeffs (Nx+3, Ny+3, Nz+3) float32.
    Out: float (m).
    """
    gp = imps.grid_params  # GridParams
    _, out = grid_excess(pq, gp.p_min, gp.N, gp.h)  # wp.vec3, float: the excess past the grid box, its length
    return spline_c2.eval_cs3(pq, gp.p_min, gp.N, gp.h, imps.sdf_coeffs) + out


@wp.func
def eval_n_vox(pq: wp.vec3, imps: ImplicitSurface):  # type: ignore
    """
    The vox unit normal at `pq`: the normal baked at the voxel containing it; outside the grid, the direction out
    of the box.

    Not called in the repository.
    In: pq wp.vec3, the query (m); imps, reads grid_params and vsnf_values (Nx, Ny, Nz) wp.vec3. Out: wp.vec3.
    """
    gp = imps.grid_params  # GridParams
    n = eval_vsnf(pq, imps)  # wp.vec3: the unit normal
    d, out = grid_excess(pq, gp.p_min, gp.N, gp.h)  # wp.vec3, float: the excess past the grid box, its length
    if out > 0.0:
        n = d / out
    return n


@wp.func
def eval_n_c0(pq: wp.vec3, imps: ImplicitSurface):  # type: ignore
    """
    The c0 unit normal at `pq`: the trilinear gradient, normalized; outside the grid, the direction out of the box.

    Not called in the repository.
    In: pq wp.vec3, the query (m); imps, reads grid_params and sdf_knots (Nx+1, Ny+1, Nz+1) float32.
    Out: wp.vec3.
    """
    gp = imps.grid_params  # GridParams
    _, n = spline_c0.eval_cs1_grad(pq, gp.p_min, gp.N, gp.h, imps.sdf_knots)  # float, wp.vec3: the value, the gradient
    n = n / wp.max(wp.length(n), 1.0e-6)
    d, out = grid_excess(pq, gp.p_min, gp.N, gp.h)  # wp.vec3, float: the excess past the grid box, its length
    if out > 0.0:
        n = d / out
    return n


@wp.func
def eval_n_c2(pq: wp.vec3, imps: ImplicitSurface):  # type: ignore
    """
    The c2 unit normal at `pq`: the tricubic gradient, normalized; outside the grid, the direction out of the box.

    Not called in the repository.
    In: pq wp.vec3, the query (m); imps, reads grid_params and sdf_coeffs (Nx+3, Ny+3, Nz+3) float32.
    Out: wp.vec3.
    """
    gp = imps.grid_params  # GridParams
    _, n = spline_c2.eval_cs3_grad(pq, gp.p_min, gp.N, gp.h, imps.sdf_coeffs)  # float, wp.vec3: the value, the gradient
    n = n / wp.max(wp.length(n), 1.0e-6)
    d, out = grid_excess(pq, gp.p_min, gp.N, gp.h)  # wp.vec3, float: the excess past the grid box, its length
    if out > 0.0:
        n = d / out
    return n


@wp.func
def eval_sdf(
    pq: wp.vec3,  # type: ignore
    imps: ImplicitSurface,
):
    """
    The signed distance at `pq`, through whichever representation this surface
    carries.

    The branch is on a struct field, so it is decided per query rather than at
    code generation. This is the per-surface path, used for visualization,
    meshing and reference checks, where one branch on a uniform value costs
    nothing worth arranging around. The batched pipeline is the
    opposite case and specializes its kernels instead -- see
    `lodos.col.sdf.narrowphase`.

    Called by `eval_sdf_k`.
    In: pq wp.vec3, the query (m); imps, reads grid_params, spline, and sdf_knots (Nx+1, Ny+1, Nz+1) float32 when
    spline is 1, sdf_coeffs (Nx+3, Ny+3, Nz+3) float32 otherwise. Out: float (m).
    """
    p_min = imps.grid_params.p_min  # wp.vec3
    N = imps.grid_params.N  # wp.vec3i
    h = imps.grid_params.h  # wp.vec3

    sdf = wp.float32(0.0)  # float: the interpolated field
    if imps.spline == 1:
        sdf = spline_c0.eval_cs1(pq, p_min, N, h, imps.sdf_knots)
    else:
        sdf = spline_c2.eval_cs3(pq, p_min, N, h, imps.sdf_coeffs)

    _, out = grid_excess(pq, p_min, N, h)  # wp.vec3, float: the excess past the grid box, its length
    return sdf + out


@wp.func
def eval_nsdf(
    pq: wp.vec3,  # type: ignore
    imps: ImplicitSurface,
):
    """
    The signed distance and the unit normal at `pq`.

    The gradient of a signed distance field points away from the surface and has
    length ~1 there, but neither representation is exactly one, so renormalize.
    The guard bites only where the field cancels -- deep inside, on the
    medial axis -- where no direction is correct anyway.

    Called by `eval_n` and `eval_nsdf_k`.
    In: pq wp.vec3, the query (m); imps, reads as `eval_sdf`. Out: the signed distance, float (m), and the unit
    normal, wp.vec3.
    """
    p_min = imps.grid_params.p_min  # wp.vec3
    N = imps.grid_params.N  # wp.vec3i
    h = imps.grid_params.h  # wp.vec3

    sdf = wp.float32(0.0)  # float: the interpolated field
    n = wp.vec3(0.0, 0.0, 1.0)  # wp.vec3: its gradient, then the unit normal
    if imps.spline == 1:
        sdf, n = spline_c0.eval_cs1_grad(pq, p_min, N, h, imps.sdf_knots)
    else:
        sdf, n = spline_c2.eval_cs3_grad(pq, p_min, N, h, imps.sdf_coeffs)

    n = n / wp.max(wp.length(n), 1.0e-6)

    # Outside the grid the field is the boundary value plus the distance to the
    # box, whose gradient is the direction out of the box.
    d, out = grid_excess(pq, p_min, N, h)  # wp.vec3, float: the excess past the grid box, its length
    if out > 0.0:
        n = d / out

    return sdf + out, n


@wp.func
def eval_n(
    pq: wp.vec3,  # type: ignore
    imps: ImplicitSurface,
):
    """
    The unit normal at `pq`, from `eval_nsdf`.

    Called by `eval_n_k`.
    In: pq wp.vec3, the query (m); imps, reads as `eval_sdf`. Out: wp.vec3.
    """
    sdf, n = eval_nsdf(pq, imps)  # float, wp.vec3: the signed distance (unused), the unit normal

    return n


@wp.func
def eval_vsdf(
    pq: wp.vec3,  # type: ignore
    imps: ImplicitSurface,
):
    """
    The vox signed distance at `pq`: the value stored at the voxel containing it, the voxel index clamped into the
    grid. Without the distance to the box, which `eval_sdf_vox` adds.

    Called by `eval_sdf_vox` and `eval_vsdf_k`.
    In: pq wp.vec3, the query (m); imps, reads grid_params and vsdf_values (Nx, Ny, Nz) float32. Out: float (m).
    """
    p_min = imps.grid_params.p_min  # wp.vec3
    N = imps.grid_params.N  # wp.vec3i
    h = imps.grid_params.h  # wp.vec3

    pq = pq - p_min  # wp.vec3: from the grid's lower corner
    x, y, z = pq[0] / h[0], pq[1] / h[1], pq[2] / h[2]  # float, float, float: in voxels

    idx, idy, idz = (  # int, int, int: the voxel, clamped into the grid
        wp.clamp(wp.int32(wp.floor(x)), 0, N[0] - 1),
        wp.clamp(wp.int32(wp.floor(y)), 0, N[1] - 1),
        wp.clamp(wp.int32(wp.floor(z)), 0, N[2] - 1),
    )

    return imps.vsdf_values[idx, idy, idz]


@wp.func
def eval_vsnf(
    pq: wp.vec3,  # type: ignore
    imps: ImplicitSurface,
):
    """The vox normal: the unit normal stored at the voxel containing `pq`.

    Piecewise constant, so it is discontinuous across cell boundaries and the
    axis built from it is piecewise constant in pose, as one built from GJK/EPA
    witness features is. It is a speed floor to measure against.

    Called by `eval_n_vox`.
    In: pq wp.vec3, the query (m); imps, reads grid_params and vsnf_values (Nx, Ny, Nz) wp.vec3. Out: wp.vec3.
    """
    p_min = imps.grid_params.p_min  # wp.vec3
    N = imps.grid_params.N  # wp.vec3i
    h = imps.grid_params.h  # wp.vec3

    pq = pq - p_min  # wp.vec3: from the grid's lower corner
    idx = wp.clamp(wp.int32(wp.floor(pq[0] / h[0])), 0, N[0] - 1)  # int: the voxel, clamped into the grid
    idy = wp.clamp(wp.int32(wp.floor(pq[1] / h[1])), 0, N[1] - 1)  # int
    idz = wp.clamp(wp.int32(wp.floor(pq[2] / h[2])), 0, N[2] - 1)  # int

    return imps.vsnf_values[idx, idy, idz]


#######################################################################################
# Visualization
#######################################################################################


@wp.kernel
def eval_sdf_k(
    p: wp.array(dtype=wp.vec3),  # type: ignore
    sdf_out: wp.array(dtype=float),  # type: ignore
    imps: ImplicitSurface,
):
    """
    p: shape(N, 3)
    sdf_out: shape(N,)

    The signed distance at each point, by `eval_sdf`.

    Launch: dim=num_p, a thread per point, by `eval_sdf_np`.
    In: p (num_p,) wp.vec3, the points (m); imps, reads as `eval_sdf`. Out: sdf_out (num_p,) float32 (m).
    """

    tid = wp.tid()  # int: the point
    pq = p[tid]  # wp.vec3
    sdf_out[tid] = eval_sdf(pq, imps)


@wp.kernel
def eval_n_k(
    p: wp.array(dtype=wp.vec3),  # type: ignore
    n_out: wp.array(dtype=wp.vec3),  # type: ignore
    imps: ImplicitSurface,
):
    """
    p: shape(N, 3)
    n_out: shape(N, 3)

    The unit normal at each point, by `eval_n`.

    Launch: dim=num_p, a thread per point, by `eval_n_np`.
    In: p (num_p,) wp.vec3, the points (m); imps, reads as `eval_sdf`. Out: n_out (num_p,) wp.vec3.
    """

    tid = wp.tid()  # int: the point
    pq = p[tid]  # wp.vec3
    n = eval_n(pq, imps)  # wp.vec3
    n_out[tid] = n


@wp.kernel
def eval_nsdf_k(
    p: wp.array(dtype=wp.vec3),  # type: ignore
    sdf_out: wp.array(dtype=float),  # type: ignore
    n_out: wp.array(dtype=wp.vec3),  # type: ignore
    imps: ImplicitSurface,
):
    """
    p: shape(N, 3)
    sdf_out: shape(N,)
    n_out: shape(N, 3)

    The signed distance and the unit normal at each point, by `eval_nsdf`.

    Launch: dim=num_p, a thread per point, by `eval_nsdf_np`.
    In: p (num_p,) wp.vec3, the points (m); imps, reads as `eval_sdf`.
    Out: sdf_out (num_p,) float32 (m), n_out (num_p,) wp.vec3.
    """

    tid = wp.tid()  # int: the point
    pq = p[tid]  # wp.vec3
    sdf, n = eval_nsdf(pq, imps)  # float, wp.vec3
    sdf_out[tid] = sdf
    n_out[tid] = n


@wp.kernel
def eval_vsdf_k(
    p: wp.array(dtype=wp.vec3),  # type: ignore
    vsdf_out: wp.array(dtype=float),  # type: ignore
    imps: ImplicitSurface,
):
    """
    p: shape(N, 3)
    vsdf_out: shape(N,)

    The vox signed distance at each point, by `eval_vsdf`.

    Launch: a thread per point; not launched in the repository.
    In: p (n,) wp.vec3, the points (m); imps, reads grid_params and vsdf_values (Nx, Ny, Nz) float32.
    Out: vsdf_out (n,) float32 (m).
    """

    tid = wp.tid()  # int: the point
    pq = p[tid]  # wp.vec3
    vsdf_out[tid] = eval_vsdf(pq, imps)


def eval_sdf_np(x, imps):
    """
    The signed distance at the points `x`, by `eval_sdf_k`. Synchronizes: it copies the result back.

    Called by `compute_sdf_batch`.
    In: x (num_p, 3) float numpy array (m); imps, an `ImplicitSurface`. Out: (num_p,) float32 numpy array (m).
    """
    num_p = x.shape[0]  # int: the number of points
    x = wp.array(x, dtype=wp.vec3)  # (num_p,) wp.vec3
    sdf_out = wp.zeros(num_p, dtype=float)  # (num_p,) float32
    wp.launch(eval_sdf_k, num_p, inputs=[x, sdf_out, imps])
    sdf = sdf_out.numpy()  # (num_p,) float32

    return sdf


def eval_n_np(x, imps):
    """
    The unit normal at the points `x`, by `eval_n_k`. Synchronizes: it copies the result back.

    Called by `compute_n_batch`.
    In: x (num_p, 3) float numpy array (m); imps, an `ImplicitSurface`. Out: (num_p, 3) float32 numpy array.
    """
    num_p = x.shape[0]  # int: the number of points
    x = wp.array(x, dtype=wp.vec3)  # (num_p,) wp.vec3
    n_out = wp.zeros(num_p, dtype=wp.vec3)  # (num_p,) wp.vec3
    wp.launch(eval_n_k, num_p, inputs=[x, n_out, imps])
    n = n_out.numpy()  # (num_p, 3) float32

    return n


def eval_nsdf_np(x, imps):
    """
    The signed distance and the unit normal at the points `x`, by `eval_nsdf_k`. Synchronizes: it copies the
    results back.

    Not called in the repository.
    In: x (num_p, 3) float numpy array (m); imps, an `ImplicitSurface`.
    Out: (num_p,) float32 (m) and (num_p, 3) float32 numpy arrays.
    """
    num_p = x.shape[0]  # int: the number of points
    x = wp.array(x, dtype=wp.vec3)  # (num_p,) wp.vec3
    sdf_out = wp.zeros(num_p, dtype=float)  # (num_p,) float32
    n_out = wp.zeros(num_p, dtype=wp.vec3)  # (num_p,) wp.vec3
    wp.launch(eval_nsdf_k, num_p, inputs=[x, sdf_out, n_out, imps])
    sdf = sdf_out.numpy()  # (num_p,) float32
    n = n_out.numpy()  # (num_p, 3) float32

    return sdf, n


def compute_sdf_batch(x, imps, batch_size=None, verbose=False):
    """
    x: shape(N, 3)

    The signed distance at the points `x`, `batch_size` of them per launch (all at once by default), with a
    progress bar when `verbose`.

    Called by `tools.mesh_processing.create_surface_data.isosurface`, `tools.mesh_processing.nutbolt.bake.build`,
    `to_pv_mesh` and 2 others.
    In: x (N, 3) float numpy array (m); imps, an `ImplicitSurface`. Out: (N,) float32 numpy array (m).
    """

    if x.shape[0] == 0:
        return np.zeros(0)

    if batch_size is None:
        batch_size = x.shape[0]  # int: every point in one batch

    batch_size = int(batch_size)

    from tqdm import tqdm  # the `mesh` extra: only making surfaces evaluates in batches

    sdfs = [  # list of (batch_size,) float32, the last shorter when batch_size does not divide N
        eval_sdf_np(x[start : start + batch_size], imps)
        for start in tqdm(range(0, x.shape[0], batch_size), disable=not verbose)  # start: int, a batch's first
    ]

    return np.concatenate(sdfs)


def compute_n_batch(x, imps, batch_size=None, verbose=False):
    """
    x: shape(N, 3)

    The unit normal at the points `x`, `batch_size` of them per launch (all at once by default), with a progress
    bar when `verbose`.

    Called by `tools.mesh_processing.create_surface_data.main`.
    In: x (N, 3) float numpy array (m); imps, an `ImplicitSurface`. Out: (N, 3) float32 numpy array.
    """

    if x.shape[0] == 0:
        return np.zeros((0, 3))

    if batch_size is None:
        batch_size = x.shape[0]  # int: every point in one batch

    batch_size = int(batch_size)

    from tqdm import tqdm  # the `mesh` extra, as above

    ns = [  # list of (batch_size, 3) float32, the last shorter when batch_size does not divide N
        eval_n_np(x[start : start + batch_size], imps)
        for start in tqdm(range(0, x.shape[0], batch_size), disable=not verbose)  # start: int, a batch's first
    ]

    return np.concatenate(ns)


def create_pv_grid(imps, grid_upsample=5, shrinkage_factor=0.9):
    """
    A pyvista image grid to contour the field on: the middle `shrinkage_factor` of the surface's grid box, sampled
    about `grid_upsample` times as finely as its voxels.

    Called by `to_pv_mesh` and `tools.mesh_processing.create_surface_data.isosurface`.
    In: imps, an `ImplicitSurface` (its grid_params). Out: a `pv.ImageData` of N[a] * grid_upsample + 1 points
    along each axis `a`.
    """
    import pyvista as pv  # the `mesh` extra: only making surfaces draws them

    p_min = np.array(imps.grid_params.p_min)  # (3,) float32
    p_max = np.array(imps.grid_params.p_max)  # (3,) float32
    N = np.array(imps.grid_params.N)  # (3,) int32
    Nx = int(N[0] * grid_upsample) + 1  # int: points along x
    Ny = int(N[1] * grid_upsample) + 1  # int: points along y
    Nz = int(N[2] * grid_upsample) + 1  # int: points along z

    bb_delta = (p_max - p_min) * shrinkage_factor  # (3,): the drawn box's extent
    bb_mid = (p_max + p_min) / 2  # (3,): its centre

    bb_origin = bb_mid - bb_delta / 2  # (3,): its lower corner

    grid = pv.ImageData(  # pv.ImageData
        dimensions=(Nx, Ny, Nz),
        spacing=(bb_delta[0] / (Nx - 1), bb_delta[1] / (Ny - 1), bb_delta[2] / (Nz - 1)),
        origin=bb_origin,
    )

    return grid


def to_pv_mesh(imps, batch_size=1e5, grid_upsample=5):
    """
    The surface's zero level set, by marching cubes over the field sampled on `create_pv_grid`'s grid.

    Not called in the repository.
    In: imps, an `ImplicitSurface`. Out: a `pv.PolyData`, in the surface's frame (m).
    """
    grid = create_pv_grid(imps, grid_upsample)  # pv.ImageData
    values = compute_sdf_batch(  # (grid.n_points,) float32
        np.array(grid.points), imps, batch_size=batch_size, verbose=True
    )
    pv_mesh = grid.contour([0], values, method="marching_cubes")  # pv.PolyData

    return pv_mesh
