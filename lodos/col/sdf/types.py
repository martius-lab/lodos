import numpy as np
import warp as wp

from lodos.utils import scan as lscan
# `PlaneSurface` is `sdf`'s: a surface a pipeline is handed, like a baked one.
# Imported here, so that `types.PlaneSurface` works for every caller that
# builds a scene through this module.
from lodos.col.sdf import spline_kind, PlaneSurface, _is_plane
from lodos.col.sdf.spline import c0 as spline_c0

# Where a contact is placed, in the order `CollisionModel.contact_point` encodes.
# Here rather than in `narrowphase`, which builds the kernels: that module imports
# this one, so the names have to live on this side of the dependency.
CONTACT_POINTS = ("surface", "midpoint")

# Stage-1 slots per block of the block-partition resolve. A thread reads the tile
# its block starts in and walks forward, so the walk is `PARTITION_BLK` over the
# mean tile size. One warp, so that a warp does not straddle several tiles and
# diverge. Here rather than in `compact`, which builds the kernels, since that
# module imports this one.
PARTITION_BLK = 32

#######################################################################################
# Main Data Structures
#######################################################################################


@wp.struct
class CollisionModel:
    """
    - All the information about a collision geometry that does not depend on the state of the rigid-body system.
    - Non-batched.

    **Pairs and segments.** A collision pair names two geometries, the first
    supplying the vertices and the second the SDF. That direction is fixed, so an
    interface tested both ways is two declared pairs, `[a, b]` and `[b, a]`, each
    with its own contacts.

    A pair owns one or more *segments*, a segment being a pair together with a
    *direction*: the vertices of one of the two geometries, tested against the
    other's SDF. An asymmetric pair owns one; a symmetric pair owns both.
    Segments of a pair are one contiguous run, which `lodos.cd` depends on.

    The split exists because the two kinds of work scale differently. The OBB test
    and the relative transform depend only on the *pair*, so those stay indexed by
    `nCPR`. Everything from the vertex loop onward is per direction, so it is
    indexed by `nSEG`.

    A segment is the unit the contact reduction and the contact groups work on
    (`lodos.cd.reduce`, `lodos.cd.groups`). With every pair asymmetric,
    `nSEG == nCPR` and every segment-indexed array is in effect pair-indexed.

    A surface bake may carry `v_group`, a per-vertex region label; nothing
    reads it.

    **Spheres, and why a vertex is one.** A geometry may carry *inner spheres*
    beside its vertices -- centres and radii in a `spheres.npz` sidecar -- meant
    to be strictly inside it, so that a deep overlap is caught before the mesh's
    own vertices have passed through the other surface. A geometry's primitives
    are its vertices followed by its sphere centres, and `prim_r` gives each one
    a radius, zero for a vertex. Every stage then reads the same thing: a point
    and a radius, kept when `sdf(p) - r` is negative.

    That is the whole of the feature. Tiles, all three stages, the compaction, the
    contact ordering and `lodos.cd` need nothing more, because a sphere contact is
    just another contact in its segment's contiguous run -- which is the property
    `build_T` and `collect_pairs` depend on. `has_spheres` is 0 when no geometry
    carries any, and the two hot kernels are then compiled without the radius
    read at all, so a scene without spheres pays nothing for them.

    **SDF representation.** Every non-plane geometry of a context is evaluated the
    same way: `spline` is 0 for the tricubic B-spline and 1 for trilinear
    interpolation, taken from the surfaces' `spline_type` and required to agree
    across them. Homogeneity is what lets `narrowphase` specialize its kernels
    once at build time instead of branching per candidate -- and per candidate is
    the whole cost model here, since a step evaluates millions of them. A context
    mixing the two is rejected.

    Built by `make_collision_context`; read by the kernels of `lodos.col.sdf.broadphase`, `compact` and
    `narrowphase`, and by `lodos.cd`.
    """

    B: wp.int32  # batchsize (number of parallel simulations)
    nB: wp.int32  # number of bodies = number of geometries
    nV: wp.int32  # total number of primitives (vertices + spheres), i.e. len(v_b)
    nCPR: wp.int32  # total number of declared collision pairs
    nSEG: wp.int32  # total number of (pair, direction) segments
    nC: wp.int32  # total number of contact candidates across all segments
    nVox: wp.int32  # total number of voxels across all geometries
    spline: wp.int32  # 0 = tricubic (c2) SDF, 1 = trilinear (c0); see below
    # --------------------------------------------------------------------------------------
    # Vertex tiles, for the two-level broadphase.
    tile_K: wp.int32  # primitives per tile where every segment agrees; else the largest
    max_nT: wp.int32  # tiles of the largest segment; diagnostic only
    nT_world: wp.int32  # sum(seg_nT) -- the tiles one world actually has
    n_tile_slots: wp.int32  # B * nT_world
    # sum(seg_nT * seg_K) -- the stage-1 slots one world's tiles ask for, padding
    # included, and so the worst case `capacity_tile` is a fraction of. It is
    # `nT_world * tile_K` wherever every segment shares a size; read it rather
    # than reconstructing that product, which is wrong as soon as one does not.
    n_slot_world: wp.int32
    nG_world: wp.int32  # sum(seg_nG) -- the generators one world has; see gen_seg
    # 1 when the tile stage runs over a compacted list of the *active* segments'
    # tiles rather than over every tile every world has. `capacity1` sizes that
    # list, and 1.0 turns this off and restores the dense launch exactly.
    seg_compact: wp.int32
    # 1 when every segment shares one tile size, which lets a stage-1 slot's tile
    # be `i / tile_K`; with sizes that differ it is looked up instead
    # (`compact.resolve_nCa1_partition`). Host-side, read by
    # `broadphase.sdf_filter_kernel` and by `compact_nTa` to skip the tables the
    # lookup needs.
    uniform_K: wp.int32
    # 1 when some segment carries super-tiles. Host-side: it decides whether the
    # generator level does any work at all, and with it 0 that level *is* the
    # plain per-segment tile count.
    has_super: wp.int32
    # --------------------------------------------------------------------------------------
    # Index Maps
    idm_nC_to_nSEG: wp.array(
        dtype=int
    )  # (nC,), the index s of the segment corresponding to contact point i #type:ignore
    idm_nC_to_prim: wp.array(
        dtype=int
    )  # (nC,), the vertex of candidate i, as an index into v_b # type: ignore
    idul_nSEG_to_nC: wp.array(
        dtype=int
    )  # (nSEG+1,), segment i generates the contacts idul_nSEG_to_nC[i]:idul_nSEG_to_nC[i+1] #type:ignore
    # --------------------------------------------------------------------------------------
    cprs: wp.array(
        dtype=wp.vec2i
    )  # (nCPR), collision pairs (verts on body 1, penetrate into body 2) # type: ignore
    # --------------------------------------------------------------------------------------
    # Segments. A pair's segments are adjacent and in kind order, so a pair's
    # contacts are one contiguous run. Also _beg stands for beginning.
    cpr_seg_beg: wp.array(
        dtype=int
    )  # (nCPR+1), the segments of pair j are cpr_seg_beg[j]:cpr_seg_beg[j+1] # type: ignore
    seg_cpr: wp.array(
        dtype=int
    )  # (nSEG), the pair this segment belongs to # type: ignore
    seg_g_v: wp.array(
        dtype=int
    )  # (nSEG), geometry supplying the vertices # type: ignore
    seg_g_s: wp.array(dtype=int)  # (nSEG), geometry supplying the SDF # type: ignore
    seg_nC: wp.array(
        dtype=int
    )  # (nSEG), vertices this segment's geometry contributes # type: ignore
    seg_dir: wp.array(
        dtype=int
    )  # (nSEG), 0 = cprs[j][0] supplies the vertices, 1 = cprs[j][1] does. # type: ignore
    # Only a symmetric pair ever has segments of both directions.
    g_nVox: wp.array(dtype=int)  # (nB), number of voxels per geometry # type: ignore
    g_nVox_cumsum: wp.array(
        dtype=int
    )  # (nB+1), cumsum of g_nVox, with a leading zero # type: ignore
    g_nVox_c2_cumsum: wp.array(  # type: ignore
        dtype=int
    )  # (nB+1), cumsum of the c2 coefficient counts prod(N+3) per geom, leading zero
    g_nVox_c0_cumsum: wp.array(  # type: ignore
        dtype=int
    )  # (nB+1), cumsum of the c0 knot counts prod(N+1) per geom, leading zero
    g_Vox_pmin: wp.array(
        dtype=wp.vec3
    )  # (nB, 3), voxel grid pmin per geom # type: ignore
    g_Vox_pmax: wp.array(
        dtype=wp.vec3
    )  # (nB, 3), voxel grid pmax per geom # type: ignore
    g_Vox_N: wp.array(
        dtype=wp.vec3i
    )  # (nB, 3,), number of voxels per axis per geom # type: ignore
    g_Vox_h: wp.array(
        dtype=wp.vec3
    )  # (nB, 3) voxel edge lengths per axis per geom # type: ignore
    g_Vox_r: wp.array(
        dtype=float
    )  # (nB,), voxel circumsphere radius per geom # type: ignore
    g_nV: wp.array(dtype=int)  # (nB), number of vertices per geometry # type: ignore
    g_nS: wp.array(dtype=int)  # (nB), number of inner spheres per geometry # type: ignore
    g_nP: wp.array(
        dtype=int
    )  # (nB), primitives per geometry -- g_nV + g_nS, the run length in v_b # type: ignore
    # 1 when some geometry carries spheres. Read on the *host* to choose which
    # filter and narrowphase kernels a context runs, never inside one.
    has_spheres: wp.int32
    # Where a contact is placed on the segment between its two witness points:
    # 0 the primitive's own surface, 1 the midpoint. Host-side, like the above --
    # `narrowphase.contact_kernel` picks the specialization once per context.
    contact_point: wp.int32
    # 1 to resolve a stage-1 slot by block partition rather than by dividing by
    # `tile_K`. A measurement knob: the two produce identical contacts while
    # tile sizes are uniform, and the partition is what per-segment sizes need.
    # Host-side, like the fields above.
    partition: wp.int32
    # 1 when some pair is marked `toa` (time of arrival): such a pair's candidates are kept
    # wherever their motion over `toa_h` could close the distance a test measures, and its
    # vertices become speculative contacts before they penetrate. Host-side, like the fields
    # above: it picks the kernels, so a context without a `toa` pair runs the kernels it ran
    # before the option existed.
    has_toa: wp.int32
    toa_h: wp.float32  # the horizon (s) a `toa` segment's motion is bounded over, the timestep by default
    cpr_toa: wp.array(dtype=int)  # (nCPR,), 1 when a segment of the pair is `toa` # type: ignore
    seg_toa: wp.array(dtype=int)  # (nSEG,), 1 for a `toa` segment # type: ignore
    # (nSEG,) each with a `toa` pair, (1,) without: how a `toa` segment's speculative contacts are traced along
    # their path (`toa_trace`: 0 not, 1 sphere tracing, 2 over-relaxed) and at most how many field evaluations
    # a trace takes (`toa_trace_iters`). Read only by the `toa` narrowphase kernels, at run time.
    seg_trace: wp.array(dtype=int)  # type: ignore
    seg_trace_iters: wp.array(dtype=int)  # type: ignore
    # (nSEG,) with a `toa` pair, (1,) without: 1 where a `toa` segment's contacts carry their predicted
    # second-order drift (`toa_drift`, `narrowphase._make_trace_kernel`); and the timestep (s) it is taken over.
    seg_drift: wp.array(dtype=int)  # type: ignore
    toa_dt: wp.float32
    # (nSEG,) with a `toa` pair, (1,) without: 1 where a `toa` segment's penetrating contacts keep the side they
    # came from (`toa_side`, `narrowphase.trace_toa`).
    seg_side: wp.array(dtype=int)  # type: ignore
    has_side: wp.int32  # 1 when some segment has `toa_side`: the last step's candidates are remembered (host-side)
    # (nB,), the root body of each geometry's kinematic tree, whose subtree COM the
    # rigid-body velocities `cvel` are taken about; read only by `toa` kernels.
    g_root: wp.array(dtype=int)  # type: ignore
    # The kinematic tree, for a body's velocity at the end of the free (contact-free) step: the
    # sum of `cdof * qacc_smooth` over the dofs from the body to its root (`broadphase.body_acc`).
    # (nbody,) each, with a `toa` pair; (1,) zeros without. Read only by `toa` kernels.
    body_parentid: wp.array(dtype=int)  # type: ignore
    body_dofadr: wp.array(dtype=int)  # type: ignore
    body_dofnum: wp.array(dtype=int)  # type: ignore
    g_nV_cumsum: wp.array(
        dtype=int
    )  # (nB+1), cumsum of g_nV, leading zero # type: ignore
    g_bid: wp.array(
        dtype=int
    )  # (nB), MuJoCo body id each geometry is attached to # type: ignore
    g_plane: wp.array(
        dtype=int
    )  # (nB), 1 where the geometry is a half-space rather than a spline surface # type: ignore
    # --------------------------------------------------------------------------------------
    # Body-frame oriented bounding boxes, precomputed from the 8 stored corners so
    # that the broadphase never has to re-derive a frame from corner coordinates.
    g_obb_c: wp.array(dtype=wp.vec3)  # (nB,), OBB centre in body coords # type: ignore
    g_obb_A: wp.array(
        dtype=wp.mat33
    )  # (nB,), OBB axes as ROWS (`obb_frame_np`'s) # type: ignore
    g_obb_h: wp.array(dtype=wp.vec3)  # (nB,), OBB half-extents # type: ignore
    # --------------------------------------------------------------------------------------
    # Every primitive of every geometry, laid out per geometry as its vertices
    # followed by its sphere centres. `prim_r` is the matching radius, zero for a
    # vertex -- which is what lets one point-and-radius test serve both.
    v_b: wp.array(
        dtype=wp.vec3
    )  # (nP, 3), body coords of every primitive across all geoms # type: ignore
    prim_r: wp.array(
        dtype=float
    )  # (nP,), 0 for a vertex, the sphere's radius for a sphere # type: ignore
    # --------------------------------------------------------------------------------------
    # Tile bounding spheres, in the vertex geometry's body coords (`sphere_survives`
    # moves them into the SDF geometry's frame by `T_rel`). Tile t of segment s
    # covers that segment's primitives [t*seg_K[s], (t+1)*seg_K[s]): a contiguous
    # run of `v_b`, spatially compact because `lodos/col/sdf/io.py` Hilbert-orders
    # each geometry's vertices.
    #
    # A tile's radius covers its members' own radii -- `max(|p_i - c| + r_i)` --
    # so `tile_survives` needs no knowledge that spheres exist.
    seg_K: wp.array(dtype=int)  # (nSEG,), primitives per tile of this segment # type: ignore
    seg_nT: wp.array(dtype=int)  # (nSEG,), tiles per segment # type: ignore
    seg_nT_beg: wp.array(
        dtype=int
    )  # (nSEG,), where a segment's tiles start in tile_c/tile_r # type: ignore
    tile_c: wp.array(dtype=wp.vec3)  # (sum seg_nT,), sphere centre # type: ignore
    tile_r: wp.array(dtype=float)  # (sum seg_nT,), sphere radius # type: ignore
    # The owning segment of each dense tile, i.e. the inverse of `seg_nT_beg`:
    # what lets the tile stage be indexed by the tiles a world has.
    tile_seg: wp.array(dtype=int)  # (sum seg_nT,), owning segment # type: ignore
    # --------------------------------------------------------------------------------------
    # Tile *generators*: the level above the tiles. A generator contributes a run
    # of consecutive tiles of one segment to the tile filter's candidate list. A
    # segment that wants super-tiles has one generator per super-tile, each
    # contributing its children only if its own bounding sphere survives; a
    # segment that does not has exactly one generator, contributing all of its
    # tiles unconditionally and running no SDF test.
    #
    # Generators are laid out per segment and in tile order, so the candidate
    # list they expand into is still ordered (batch, segment, tile) -- the
    # contiguity `lodos.cd` requires.
    #
    # `gen_c`/`gen_r` are a bounding sphere in the same body coords and with the
    # same meaning as `tile_c`/`tile_r`: a super-tile is the tile of size
    # `seg_K * super_K`, tested by the same `sphere_survives`. Both are zero for
    # a pass-through generator, which nothing reads.
    seg_nG: wp.array(dtype=int)  # (nSEG,), generators per segment # type: ignore
    seg_nG_beg: wp.array(dtype=int)  # (nSEG,), where they start # type: ignore
    gen_seg: wp.array(dtype=int)  # (nG_world,), owning segment # type: ignore
    gen_sup: wp.array(dtype=int)  # (nG_world,), 1 to test gen_c/gen_r first # type: ignore
    gen_tile_beg: wp.array(
        dtype=int
    )  # (nG_world,), its first tile, within its segment # type: ignore
    gen_nT: wp.array(dtype=int)  # (nG_world,), tiles it covers # type: ignore
    gen_c: wp.array(dtype=wp.vec3)  # (nG_world,), sphere centre # type: ignore
    gen_r: wp.array(dtype=float)  # (nG_world,), sphere radius # type: ignore
    # The owning generator of each dense tile -- the inverse of `seg_nG_beg` plus
    # the super-tile stride, and the dense launch's counterpart to `tile_seg`.
    tile_gen: wp.array(dtype=int)  # (sum seg_nT,), owning generator # type: ignore
    # --------------------------------------------------------------------------------------
    # All voxel parameters for the vox lookup (one memory load) used for broadphase
    # and the narrowphase SDF, which is either the c2 spline (tricubic, 64
    # coefficients per query) or the c0 one (trilinear, 8 knot samples per query).
    # Voxels are flattened into a single axis (nVox,).
    #
    # One table, one channel: the signed distance. Both consumers read the same 64
    # coefficients -- `eval_sdf_c2` sums them, `eval_nsdf_c2` also accumulates the
    # three derivative sums that give the normal.
    #
    # `spline` says which of the two narrowphase tables is live, and only that one
    # is filled: the other is a single-element placeholder, since these are the
    # largest arrays in the model (a 100^3 grid is 4 MB per geometry).
    vox_c_sdf_c2: wp.array(dtype=float)  # (sum prod(N+3),) tricubic coefficients
    vox_c_sdf_c0: wp.array(dtype=float)  # (sum prod(N+1),) knot samples
    vox_c_sdf_vox: wp.array(dtype=float)  # (nVox,)


@wp.func
def idm_nB_nx_ny_nz_to_nVox_vox(
    iB: int, ix: int, iy: int, iz: int, mcol: CollisionModel  # type: ignore
):
    """
    Index into the vox (voxel-centre) table, stored row-major.

    Bricking this table measured no benefit: each thread reads a single 4-byte
    cell rather than a 4x4x4 stencil, so there is no intra-thread locality to
    gain, and the table is far too large to cache regardless of layout.

    Called by `lodos.col.sdf.broadphase.eval_sdf_vox`.
    In: iB, int, the geometry; ix, iy, iz, int, its voxel's indices; mcol reads g_Vox_N (nB,) wp.vec3i and
    g_nVox_cumsum (nB+1,) int32. Out: int, the index into mcol.vox_c_sdf_vox (nVox,) float32.
    """
    N_B = mcol.g_Vox_N[iB]  # wp.vec3i: the geometry's voxels per axis
    Ny, Nz = N_B[1], N_B[2]  # int, int

    return mcol.g_nVox_cumsum[iB] + ix * Ny * Nz + iy * Nz + iz


@wp.func
def idm_nB_nx_ny_nz_to_nVox_c2(
    iB: int, ix: int, iy: int, iz: int, mcol: CollisionModel  # type: ignore
):
    """
    Index into the c2 (tricubic coefficient) table, stored row-major.

    Called by `lodos.col.sdf.narrowphase.eval_sdf_c2` and `eval_nsdf_c2`.
    In: iB, int, the geometry; ix, iy, iz, int, a coefficient's indices on its (N+3)-per-axis grid; mcol reads
    g_Vox_N (nB,) wp.vec3i and g_nVox_c2_cumsum (nB+1,) int32. Out: int, the index into mcol.vox_c_sdf_c2
    (sum prod(N+3),) float32.
    """
    # The c2 coefficient grid has (N+3) entries per axis, so geoms stride by
    # prod(N+3) -- not by prod(N), which is the vox (voxel-centre) stride.
    N_B = mcol.g_Vox_N[iB]  # wp.vec3i: the geometry's voxels per axis
    Ny, Nz = N_B[1], N_B[2]  # int, int

    return mcol.g_nVox_c2_cumsum[iB] + ix * (Ny + 3) * (Nz + 3) + iy * (Nz + 3) + iz


@wp.func
def idm_nB_nx_ny_nz_to_nVox_c0(
    iB: int, ix: int, iy: int, iz: int, mcol: CollisionModel  # type: ignore
):
    """
    Index into the c0 (trilinear knot sample) table, stored row-major.

    Called by `lodos.col.sdf.narrowphase.eval_sdf_c0` and `eval_nsdf_c0`.
    In: iB, int, the geometry; ix, iy, iz, int, a knot's indices on its (N+1)-per-axis lattice; mcol reads
    g_Vox_N (nB,) wp.vec3i and g_nVox_c0_cumsum (nB+1,) int32. Out: int, the index into mcol.vox_c_sdf_c0
    (sum prod(N+1),) float32.
    """
    # The c0 grid is the knot lattice, (N+1) entries per axis -- one more than the
    # voxel centres (vox) and two fewer than the c2 coefficients. Three tables, three
    # strides, and getting one wrong reads plausible values from the wrong geometry.
    N_B = mcol.g_Vox_N[iB]  # wp.vec3i: the geometry's voxels per axis
    Ny, Nz = N_B[1], N_B[2]  # int, int

    return mcol.g_nVox_c0_cumsum[iB] + ix * (Ny + 1) * (Nz + 1) + iy * (Nz + 1) + iz


@wp.struct
class CollisionData:
    """
    - All the information about a collision geometry that depends on the state of the rigid-body system.
    - Batched.

    Every array below is allocated once (see `make_collision_context`) and reused
    across steps. The three survivor counts live on the *device* in `ct`, so no
    stage needs a host round trip and the whole Warp section is CUDA-graph
    capturable. Stage kernels are launched at the fixed capacities `nCa*_max` and
    exit early on `tid >= ct[i]`.

    Built by `make_collision_context`, its capacities regrown by `lodos.col.sdf.collide.resize_context`; written
    each step by the kernels `collide.collide` launches, and read by `lodos.cd`.
    """

    # --------------------------------------------------------------------------------------
    # Kinematic inputs.
    #
    # Body poses, bound to `RbdData`'s rather than copied -- `xpos` and `xquat` are the whole
    # of what the pipeline needs from the rigid-body side. `compute_T_w` gathers
    # them per collision geometry into `T_w`; contacts are then evaluated from the
    # static body-frame vertices `mcol.v_b` via the relative pose `T_rel`.
    xpos: wp.array(dtype=wp.vec3, ndim=2)  # (B, nbody) # type: ignore
    xquat: wp.array(
        dtype=wp.vec4, ndim=2
    )  # (B, nbody), MuJoCo (w, x, y, z) order # type: ignore
    T_w: wp.array(
        dtype=wp.transformf, ndim=2
    )  # (B, nB), 6DOF poses of all bodies across all sims # type: ignore
    T_rel: wp.array(
        dtype=wp.transformf, ndim=2
    )  # (B, nSEG), pose of a segment's vertex geom in its SDF geom's frame # type: ignore
    # Velocities, which only `toa` kernels read. `cvel` (each body's [angular, linear]
    # velocity about its tree root's subtree COM) and `subtree_com` are bound to
    # `RbdData`'s like the poses. `twist_rel` is a segment's relative motion in its SDF
    # geometry's body frame, `broadphase.compute_twist_rel`'s: the vertex geometry's
    # material point at y moves at `a + Omega x y` relative to the SDF geometry, stored as
    # the spatial vector (Omega, a); (1, 1) without a `toa` pair.
    cvel: wp.array(dtype=wp.spatial_vector, ndim=2)  # (B, nbody) # type: ignore
    subtree_com: wp.array(dtype=wp.vec3, ndim=2)  # (B, nbody) # type: ignore
    twist_rel: wp.array(dtype=wp.spatial_vector, ndim=2)  # (B, nSEG) # type: ignore
    # The same at the end of the horizon under the free motion, `cvel + toa_h J qacc_smooth`
    # (`cdof` and `qacc_smooth` bound to `RbdData`'s): `toa_margin` bounds a point's speed by
    # the larger of the two, so that a step's own acceleration is inside the margin.
    cdof: wp.array(dtype=wp.spatial_vector, ndim=2)  # (B, nv) # type: ignore
    qacc_smooth: wp.array(dtype=float, ndim=2)  # (B, nv) # type: ignore
    twist_end: wp.array(dtype=wp.spatial_vector, ndim=2)  # (B, nSEG), (1, 1) without a `toa` pair # type: ignore
    # (nB,) sticky flags, 1 once a `toa` contact's gap to SDF geometry g was read outside g's
    # grid, where the narrowphase falls back to the distance to g's bounding box (a bound that
    # can only under-estimate); (1,) without a `toa` pair. `collide.check_toa_band` reports them.
    toa_band: wp.array(dtype=int)  # type: ignore
    # --------------------------------------------------------------------------------------
    # Capacities (host-side constants, mirrored here so kernels can clamp).
    nCa1_max: wp.int32  # stage-1 slots: the primitives of the surviving tiles
    nCa2_max: wp.int32  # stage-2 slots: the voxel-SDF (vox) survivors
    nCa3_max: wp.int32  # stage-3 slots: the spline-SDF survivors, i.e. the contacts
    nTa_max: wp.int32  # slots for the surviving tiles
    nTb_max: wp.int32  # tiles of active segments; n_tile_slots when not compacting
    # Stage 1's *launch width*, which is a separate thing from its allocation:
    # `nCa1_max` when one thread owns one slot, and a hardware-sized cap when it
    # walks them (`make_collision_context`'s `grid_stride`). It is also the
    # stride, since a grid-strided thread steps by the width of its own grid.
    nCa1_grid: wp.int32
    # 1 when the capacity-sized scans take their length from `ct` instead of from
    # the allocation. See `lodos.utils.scan`; `scan_buf` is its scratch, shared
    # by every site because they run one after another.
    dyn_scan: wp.int32
    scan_buf: wp.array(dtype=int)  # (scan.workspace(...),) # type: ignore
    # Device-side survivor counts and overflow bookkeeping.
    ct: wp.array(
        dtype=int
    )  # (5,), [nCa1, nCa2, nCa3, nTa, nTb], clamped to capacity # type: ignore
    ct_raw: wp.array(dtype=int)  # (5,), the same counts before clamping # type: ignore
    ct_peak: wp.array(
        dtype=int
    )  # (5,), the largest pre-clamp count each stage has reached since the # type: ignore
    # last `reset_overflow`. The overflow *flags* are sticky and `ct_raw` holds
    # only the latest step, so the flags are reported against this.
    overflow: wp.array(
        dtype=int
    )  # (5,), 1 where a stage exceeded its capacity # type: ignore
    # --------------------------------------------------------------------------------------
    # Bounding-box filtering
    cpr_active: wp.array(
        dtype=wp.bool, ndim=2
    )  # (B, nCPR) whether a pair survived the OBB broadphase # type: ignore
    gen_w: wp.array(
        dtype=int
    )  # (B*nG_world,), tiles each (batch, generator) contributes # type: ignore
    gen_s: wp.array(
        dtype=int
    )  # (B*nG_world,), exclusive scan of gen_w == where they start in nTb # type: ignore
    # The tile survivors, counted and scanned per generator rather than per
    # candidate. Both lists are ordered (batch, segment, tile), so a survivor's
    # rank is its generator's offset plus its rank within that generator, and
    # `B*nG_world` is fixed by the scene where the candidate list is a capacity.
    # See `compact.compact_nTa`.
    gcnt: wp.array(dtype=int)  # (B*nG_world,), survivors per generator # type: ignore
    goff: wp.array(dtype=int)  # (B*nG_world,), exclusive scan of gcnt # type: ignore
    idul_B_nSEG_to_nCa1: wp.array(dtype=int, ndim=2)  # (B, nSEG+1) # type: ignore
    # idul_B_nSEG_to_nCa1[b, i]: idul_B_nSEG_to_nCa1[b, i+1] are the contact points in nCa1 sized arrays that correspond to segment i in batch b
    # --------------------------------------------------------------------------------------
    # Tile filtering. `tflag` is (nTb_max,) and `idm_nTa_to_slot` holds
    # the surviving slots. Without segment compaction `nTb_max == n_tile_slots`
    # and a thread's index
    # *is* its dense tile slot; with it the two differ and `tslot` carries the
    # dense slot across, so everything downstream of `idm_nTa_to_slot` -- every
    # `resolve_nTa` caller, all of stages 1 to 3 -- is unchanged either way.
    tflag: wp.array(dtype=int)  # (nTb_max,) # type: ignore
    tslot: wp.array(dtype=int)  # (nTb_max,), dense tile slot; 1 elem when dense # type: ignore
    # Stage-1 -> stage-2 compaction at *tile* granularity. Every surviving tile
    # owns `tile_K` contiguous stage-1 slots, so the survivor offsets can be
    # scanned over tiles and ranked within one, instead of scanned over every
    # vertex slot -- a `tile_K` times shorter scan. See `compact_nCa2`.
    tcnt: wp.array(dtype=int)  # (nTa_max,), survivors per surviving tile # type: ignore
    toff1: wp.array(dtype=int)  # (nTa_max,), exclusive scan of tcnt # type: ignore
    # The block-partition resolve's tables (see `PARTITION_BLK`), built only where `partition`
    # is set, by default where tile sizes differ. `tile_sz[t]` is a surviving tile's own size,
    # `tile_off` its exclusive scan -- so `tile_off[t]` is tile t's first stage-1
    # slot, and `tile_off[nTa_max]` is the whole stage-1 count that
    # `ct[3] * tile_K` gives when the sizes agree. `blk_base[k]` is the tile
    # owning slot `k * PARTITION_BLK`, so a thread starts there and walks.
    tile_sz: wp.array(dtype=int)  # (nTa_max + 1,), 1 elem without `partition` # type: ignore
    tile_off: wp.array(dtype=int)  # (nTa_max + 1,), 1 elem without `partition` # type: ignore
    blk_base: wp.array(dtype=int)  # (nCa1_max // PARTITION_BLK + 2,) # type: ignore
    idm_nTa_to_slot: wp.array(
        dtype=int
    )  # (nTa_max,), surviving tile -> (b, segment, tile) slot # type: ignore
    # Where a segment's tiles start among the surviving ones. Stage 2's segment
    # bounds are the tile scan read at a segment's first tile, which cannot be
    # recovered from the stage-1 bound by dividing once tile sizes differ.
    idul_B_nSEG_to_nTa: wp.array(dtype=int, ndim=2)  # (B, nSEG+1) # type: ignore
    # --------------------------------------------------------------------------------------
    # SDF (vox) filtering
    flag1: wp.array(dtype=int)  # (nCa1_max,), 1 where sdf_vox < r # type: ignore
    idm_nCa2_to_nCa1: wp.array(
        dtype=int
    )  # (nCa2_max,), maps nCa2 index to nCa1 index # type: ignore
    idul_B_nSEG_to_nCa2: wp.array(dtype=int, ndim=2)  # (B, nSEG+1) # type: ignore
    # idul_B_nSEG_to_nCa2[b, i]: idul_B_nSEG_to_nCa2[b, i+1] are the contact points in nCa2 sized arrays that correspond to segment i in batch b
    # --------------------------------------------------------------------------------------
    # NSDF (c2 or c0) narrowphase
    sdf_nCa2: wp.array(dtype=float)  # (nCa2_max,), # type: ignore
    n_nCa2: wp.array(dtype=wp.vec3)  # (nCa2_max, 3), # type: ignore
    flag2: wp.array(dtype=int)  # (nCa2_max,), 1 where sdf < 0 # type: ignore
    off2: wp.array(dtype=int)  # (nCa2_max,), exclusive scan of flag2 # type: ignore
    # A `toa` context's speculative candidates to trace (`narrowphase.trace_toa`), listed by the narrowphase as
    # stage-2 slots: (nCa2_max,), at most every candidate, (1,) without a `toa` pair; and their count, (1,).
    trace_list: wp.array(dtype=int)  # type: ignore
    trace_ct: wp.array(dtype=int)  # type: ignore
    # (nCa2_max,) with a `toa` pair, (1,) without: each stage-2 contact's predicted second-order drift (m), 0 unless
    # its segment has `toa_drift`; read by the export (`lodos.cd.driver.export_lodos_toa`).
    e_nCa2: wp.array(dtype=float)  # type: ignore
    # The last step's stage-2 candidates, for `toa_side` -- every vertex near a surface, contact or not (a trace
    # drops a vertex whose straight path misses, though its arc may not): each one's vertex, its normal in its SDF
    # geometry's body frame, and its plane, gap(y) = side_nB . y + side_c; each (world, segment)'s run of them,
    # sorted by vertex (`idul_B_nSEG_to_nCa2`'s). (nCa2_max,) and (B, nSEG+1) with a `toa` pair, (1,) and (1, 1)
    # without.
    side_v: wp.array(dtype=int)  # type: ignore
    side_nB: wp.array(dtype=wp.vec3)  # type: ignore
    side_c: wp.array(dtype=float)  # type: ignore
    side_idul: wp.array(dtype=int, ndim=2)  # type: ignore
    idm_nCa3_to_nCa2: wp.array(
        dtype=int
    )  # (nCa3_max,), maps nCa3 index to nCa2 index # type: ignore
    idul_B_nSEG_to_nCa3: wp.array(dtype=int, ndim=2)  # (B, nSEG+1) # type: ignore
    # --------------------------------------------------------------------------------------
    # Final contact list, in world coordinates. Positions are formed over the nCa3
    # survivors only; normals are rotated out of the SDF geometry's body frame,
    # which is the frame eval_nsdf_c2 returns them in.
    p_nCa3: wp.array(
        dtype=wp.vec3
    )  # (nCa3_max,), world contact position # type: ignore
    n_nCa3: wp.array(dtype=wp.vec3)  # (nCa3_max,), world surface normal # type: ignore
    sdf_nCa3: wp.array(
        dtype=float
    )  # (nCa3_max,), penetration depth (< 0) # type: ignore
    b_nCa3: wp.array(dtype=int)  # (nCa3_max,), batch index # type: ignore
    seg_nCa3: wp.array(dtype=int)  # (nCa3_max,), segment index # type: ignore
    v_nCa3: wp.array(dtype=int)  # (nCa3_max,), index into mcol.v_b # type: ignore


#######################################################################################
# Creation
#######################################################################################


def obb_frame_np(boxes):
    """
    The static body-frame OBBs, computed once when the context is built: the
    broadphase reads them (`g_obb_c`, `g_obb_A`, `g_obb_h`) and derives no frame itself.

    boxes: (nG, 8, 3), corners ordered i = 4*xbit + 2*ybit + zbit.
    returns (centers (nG,3), axes (nG,3,3) with axes as ROWS, half_extents (nG,3)).

    Called by `make_collision_context`, for the non-plane geometries' `bbox`es. All three outputs are float64.
    """
    boxes = np.asarray(boxes, dtype=np.float64)  # (nG, 8, 3) float64
    centers = boxes.mean(axis=1)  # (nG, 3) float64
    edges = np.stack(
        [
            boxes[:, 4] - boxes[:, 0],
            boxes[:, 2] - boxes[:, 0],
            boxes[:, 1] - boxes[:, 0],
        ],
        axis=1,
    )  # (nG, 3, 3) float64: corner 0's edges along the box's x, y and z, as rows
    lengths = np.linalg.norm(edges, axis=-1)  # (nG, 3)
    axes = edges / lengths[..., None]  # (nG, 3, 3) float64: unit axes, as rows
    return centers, axes, 0.5 * lengths


# Threads to launch stage 1 with when it grid-strides: roughly nine times what
# an RTX 4080 holds resident (76 SMs x 1536), enough to keep every SM fed and
# to balance the tail. Sized in threads rather than derived from the device so
# that a context is reproducible across machines.
STAGE1_GRID = 1 << 20


def _stage1_grid(nCa1_max: int, grid_stride: bool) -> int:
    """Stage 1's launch width: capped when striding, the allocation when not.

    Below the cap the two are equal whatever the flag says, so a small context
    is one-thread-per-slot either way and pays nothing for the loop that would
    go around it.

    Called by `make_collision_context` and `lodos.col.sdf.collide.resize_context`.
    In: nCa1_max, int, stage 1's allocation; grid_stride, bool. Out: int, the width `CollisionData.nCa1_grid`.
    """
    return min(int(nCa1_max), STAGE1_GRID) if grid_stride else int(nCa1_max)


def make_collision_context(
    m,
    drbd,
    cprs,
    surfaces,
    surface_body_ids,
    cpr_is_symmetric=None,
    contact_point="surface",
    partition=None,
    capacity1=1.0,
    capacity2=1.0,
    tile_size=8,
    super_size=0,
    capacity_tile=1.0,
    dyn_scan=True,
    grid_stride=True,
    mempool_release_threshold=1 << 30,
    toa=None,
    toa_horizon=None,
    toa_trace="relaxed",
    toa_trace_iters=128,
    toa_drift=False,
    toa_side=False,
):
    """
    - `cprs` indexes `surfaces`, not the MuJoCo model's geoms

    m: any model with MuJoCo's field names, a `mujoco.MjModel` for one. Only
        `nbody` is read, and with a `toa` pair `body_rootid` and `opt.timestep`.

    drbd: the batched Warp rigid-body state from
        `lodos.rbd.types.make_rbd_context`. Its `xpos`/`xquat` are *bound*
        here, not copied: the poses collision reads are the ones `kinematics`
        wrote, with nothing in between.

    surfaces: the processed surfaces, one per collision geometry, or a
        `PlaneSurface` for a half-space. Each carries a `spline_type` -- "cubic"
        for the tricubic B-spline SDF, "linear" for trilinear interpolation of
        the same knot samples -- and the non-plane ones must agree, since the
        context evaluates one representation and `narrowphase` specializes its
        kernels on it. Set the type when loading the surface rather than
        afterwards; `surf.load_surface_data` takes it as an argument.

    contact_point: where a contact sits on the segment between its two witness
        points -- "surface", the primitive's own (the default), or "midpoint",
        MuJoCo's convention for a primitive pair. Depth and normal are the same
        either way; only the lever arm differs, so this is a modelling choice
        rather than an accuracy one. It matters only where a primitive has a
        radius: at r = 0 both leave the contact on the vertex itself, since the
        midpoint is placed only for a primitive with a radius. Compiled, not branched --
        `narrowphase.contact_kernel`.

    tile_size: vertices per tile of the two-level broadphase, in which a tile is
        rejected whole against the other geometry's voxel SDF before any of its
        vertices are tested.

        This is where the pipeline's time goes, so it is where the speed is. The
        per-pair box filter admits *all* of a geometry's vertices whenever the
        boxes overlap, and almost all of those are then rejected one at a time
        by the voxel prefilter, whose launch and whose compaction scan are both
        sized by that count. Rejecting a tile costs one gather and removes
        `tile_size` vertices from both.

        A tile is a *contiguous run* of `v_b`: `lodos/col/sdf/io.py` Hilbert-orders
        each geometry's vertices, so a run is already a spatially compact
        cluster, nothing is reindexed, and the segment-contiguous ordering
        `lodos.cd` depends on is preserved exactly.

        Tiling changes no result: the survivors are exactly those of a
        per-vertex pipeline. What it changes is how many candidates the stages
        after the tile test are launched and scanned over.

        **It is not free everywhere, and it is the only path.** The tile
        level costs four launches and two scans where a per-vertex one costs one
        of each, so on a small scene, with few candidates per world, that fixed
        overhead dominates. A geometry with fewer than `2 * tile_size` vertices
        is one or two tiles that cull nothing, and it pays that anyway. Lower
        `tile_size` for such a scene; at `tile_size = 1` a tile is a vertex and
        the tile test is the vertex test.

        8 by default, between the two ways a size goes wrong: smaller tiles make
        the tile level itself the cost, and larger ones let too many vertices
        survive it.

        **Per directional pair, not per scene.** An int applies to every segment;
        a dict sizes them individually, keyed by the ordered tuple
        `(vertex geometry, SDF geometry)` -- which is what a segment *is*, and
        the only key that separates a symmetric pair's two directions, the case
        most likely to want different sizes on each side. The optional key `None`
        gives the default for segments the dict does not name; without it they
        take 8. So `tile_size={None: 8, (1, 0): 16}` is "16 for geometry 1's
        vertices against geometry 0's SDF, 8 everywhere else".

        This exists because one global size is an average over segments that
        want different ones. The total cost bottoms out between two steep walls
        -- small tiles, where the tile level is the cost, and large ones, where
        stage 1 is -- and a geometry of many tiles and one of a few sit at
        opposite ends of that curve.

        Sizes that differ are not free: a stage-1 slot's tile is then a lookup
        and a short walk rather than `i / tile_size`. A context whose segments
        all agree keeps the division.

    super_size: tiles per *super-tile*, the level above -- 0, the default, for
        none. Same spelling as `tile_size`: an int for every segment, or a dict
        keyed by `(vertex geometry, SDF geometry)` with an optional `None`
        default.

        A super-tile is bounded and tested exactly like a tile, so it *is* the
        tile of size `tile_size * super_size`; what differs is only that
        rejecting one removes its children from the tile filter's launch rather
        than from stage 1. That is its use: fine tiles cull far better (far
        fewer stage-1 slots stay live), and the tile level's own cost, which
        grows as the tiles shrink, is what stops you using them. A level above
        pays for the fine ones.

        **Off by default, per directional pair.** A segment of one or two tiles
        has nothing to gain. A segment with super-tiles off runs the tile level
        alone; see `CollisionModel.gen_seg`. Silently ignored where a segment has
        one tile or fewer, since a super-tile over it can reject nothing.

    capacity_tile: fraction of `B * sum(seg_nT)` -- the tiles the scene has -- to
        preallocate for the surviving ones. As with `capacity1`, 1.0 makes
        overflow impossible and `collide.calibrate_capacity` measures what a
        scene actually reaches.

        A fraction of the tiles the scene has rather than of `B * nSEG * max_nT`,
        a rectangle sized by the longest segment, which would over-allocate by the
        segment-length spread: the longest segment's tile count over the mean.

    capacity1: fraction of the B*nC worst case to preallocate for stage 1 -- the
        *candidates* of every pair whose boxes overlap, not the box test itself.
        The box test writes `cpr_active`, one bool per (batch, pair), which is
        statically sized and cannot overflow; `capacity1` sizes the expansion of
        that result into one candidate per primitive, which is where the
        millions are.

        **Tiled, it sizes the same expansion in tiles rather than in
        primitives**: the fraction of `B * sum(seg_nT)` to preallocate for the
        tiles of the segments whose boxes overlap. The tile filter is then
        launched over that compacted list instead of over every tile every world
        has. Where the box test rejects most segments, a dense filter, its scan
        and its scatter each visit a grid several times larger than the one that
        can contribute anything.

        **1.0 skips the compaction entirely** and launches densely -- so 1.0 is
        not merely safe but bit-identical to the dense launch, and the only way
        to be slower than the dense launch is to ask for it. The bound is
        structural (a world cannot have more active tiles than tiles), so 1.0
        cannot overflow.

        Below 1.0 it can, like any capacity: `collide.calibrate_capacity`
        measures what a scene reaches, and `check_overflow` names `nTb` when it
        does not fit. Stage 1 is still sized by `capacity_tile` and `tile_size`
        -- this knob sizes what the tile *filter* runs over, not its output.

        Note that `capacity2` is a fraction of *`nCa1_max`*, and tiling shrinks
        `nCa1_max` by roughly the culling factor, so a `capacity2` that fits one
        tile size can overflow stage 2 at another. Recalibrate both together;
        `collide.calibrate_capacity` returns a consistent set.
    capacity2: fraction of nCa1_max to preallocate for the stage-2 (voxel-SDF)
        and stage-3 (spline-SDF) survivors.

    Both default to 1.0, which makes overflow impossible but allocates for a
    worst case no real scene reaches.

    **What over-sizing costs**, with `grid_stride` off:
    **`capacity_tile` is the only one worth getting right**, and the reason is
    not scans -- with `dyn_scan` every compaction costs what is live. It is
    **launch width**: most of what an over-sized `capacity_tile` costs is
    `sdf_filter`'s idle threads, and the rest is the downstream arenas, which
    move with it because `nCa2_max` is a fraction of `nCa1_max`.

    An early-exiting thread really is nearly free, but this is by far the
    widest launch in the pipeline, so an over-sized one holds a great many of
    them, and the kernel can then spend longer retiring empty threads than
    doing the work. Grid-striding *this one kernel* (`grid_stride`) is what
    recovers it.

    Beware extrapolating a per-thread cost: it is not constant in the launch
    width.

    `capacity1` sizes the tile filter's launch and `capacity2` the narrowphase's,
    both far narrower -- and `capacity1` is flattest of all because no scan runs
    over what it sizes, where the others' scans are merely bounded
    (`compact.compact_nTa`).

    So grid-striding is the lever that flattens the rest, but only for stage 1
    rather than everywhere -- and it is *safe*, since nothing reads a stage's
    tail.

    With the CUB scan instead (`dyn_scan=False`) the same over-sizing costs
    more, and the difference is the part of each slope that was the scan.

    Three ways to size these, in increasing order of not needing a human:
    `collide.calibrate_capacity` (offline, over representative states),
    `collide.suggest_capacity` (from the peaks a live context has reached), and
    `collide.Autosizer` (the same, on a loop, growing again if a stage
    overflows).

    A stage that exceeds its capacity drops the overflowing tail and raises a
    flag in `dcol.overflow`; call `collide.check_overflow(dcol)` to inspect it
    (that syncs, so keep it off the hot path).

    dyn_scan: take each capacity-sized scan's *length* from the device-side
        survivor count rather than from the allocation (`lodos.utils.scan`).
        This is what stops a capacity from being a per-step cost: with it off,
        every compaction scans its whole arena because CUB needs a host-side
        length, so a capacity at twice the true peak costs twice the scan. The
        model-sized scans -- the two over `B * nG_world` -- keep CUB either way,
        since every entry of those is live by construction and a bounded scan
        would only add launches.

    grid_stride: launch stage 1 at a width the hardware can hold rather than at
        `nCa1_max`, each thread walking its slots a grid apart until it passes
        `ct[0]`. The other half of taking a capacity out of the step time:
        `dyn_scan` stops the compactions from costing the arena, and this stops
        the widest launch in the pipeline from doing so.

        It exists for an over-sized stage 1, where `sdf_filter` can spend
        longer over its capacity than over the live count: an idle thread is
        nearly free, but an over-sized launch holds a great many of them.

        With it off, `nCa1_grid == nCa1_max`, the kernel's loop runs at most once
        and the launch is one thread per slot -- so this is structurally the
        same kernel rather than a branch.

    mempool_release_threshold: bytes Warp's CUDA memory pool may retain instead
        of returning to the driver. Warp defaults this to 0, which makes the CUB
        temporary allocation inside every `wp.utils.array_scan` round-trip to
        the driver -- a fixed ~0.6 ms per scan, i.e. far more than the scans
        themselves cost. Pass None to leave the device setting untouched.

    toa: which pairs run on time of arrival, for thin features: None or False for
        none (the default), True for every pair, a list of nCPR bools (a symmetric
        pair's two directions take its flag), or `tile_size`'s dict keyed by
        `(vertex geometry, SDF geometry)` with an optional `None` default. A `toa`
        pair's every test -- the box test, super-tiles, tiles, the voxel prefilter --
        also keeps what its relative motion could bring within reach over
        `toa_horizon` (each level bounding its sphere's speed by the segment's relative
        twist, `broadphase.compute_twist_rel`), and its narrowphase admits a vertex
        whose distance that motion could close: a speculative contact with a positive
        gap, which the exact mode's Baumgarte term lets close by at most the gap in a
        step. The bounds are path-length bounds on a 1-Lipschitz field, so they hold
        under rotation; a point's speed is the larger of its speed at the start of the
        step and at the horizon's end under the free motion (`qacc_smooth`), so the
        step's own acceleration is inside them, though not its contact forces. A
        speculative contact is kept only if its vertex's path over the step can reach
        the surface (a sphere trace along it, `toa_trace`), and is linearized where the
        path arrives: a vertex passing an edge is not stopped. With no
        `toa` pair the context compiles and runs exactly the kernels it did before.
        **Bake `toa` surfaces with a wide band.** The baked field exists only inside
        its grid, the mesh's box plus a band (`create_surface_data.py`'s `--dilation`
        and `--d-max`). A speculative gap read outside it comes from the bounding box
        instead (`narrowphase.box_gap`), which is safe but can stop a fast body a step
        early; give a `toa` surface a band wider than the largest speed x timestep it
        meets. `collide.check_toa_band` reports the geometries where it happened.

    toa_horizon: the horizon (s) of the motion bound, `m.opt.timestep` if None.

    toa_trace: how a `toa` pair's speculative contacts are traced along their path over the
        horizon (`narrowphase._make_toa_trace`), in `toa`'s spellings (one value for every
        pair, a list of nCPR, or the dict). False: not at all -- every admitted contact is
        kept, linearized at its start, which stops a body passing an edge as if it hit it.
        "sphere": sphere tracing, each step the distance to the surface, so no surface is
        skipped; a path proved clear drops its contact. "relaxed" or True (the default):
        over-relaxed sphere tracing, steps of `narrowphase.TOA_RELAX` (1.6) times that,
        each kept only if the clear spheres about its two ends overlap and else redone as a
        plain step: the same contacts in fewer steps, about half when it grazes a surface,
        where plain steps shrink to the clearance.

    toa_trace_iters: the most field evaluations one trace takes, in `toa`'s spellings (128 by
        default). A graze takes about its path length over its clearance; a trace that runs
        out keeps its contact, linearized at the last point it reached.

    toa_drift: whether a `toa` pair's contacts carry their predicted second-order drift, in
        `toa`'s spellings (False by default): how much faster than first order the gap of a
        vertex sliding along a concave surface closes in a step, as the surface curves into its
        path -- e = dt^2/2 |u_t|^2 max(0, -kappa), about v^2 dt^2/(2 rho) on a curve of radius
        rho (`narrowphase._make_trace_kernel`'s `toa_drift`). The exact mode's bias corrects it as a
        feed-forward in the step it happens; uncorrected, the depenetration gain lets it settle at
        about e / baumgarte_alpha in sustained sliding. Measured: a cube sliding inside a 10 cm
        ring with a 1 cm wall holds at 40 m/s with it, sticks at 10 m/s without. Off by default:
        on a hand scene it raised the mean depth by about 15% (the one-voxel curvature reads
        sharp detail as curving the whole step). A rotating body's own arc is not covered.

    toa_side: whether a `toa` pair's penetrating contacts keep the side they came from, in
        `toa`'s spellings (False by default): a safety net for thin features. A vertex pushed
        past a thin feature's midsurface, by whatever error, reads the field of the far side,
        whose normal turned against the one it had, and would be pushed on through. Such a
        contact (the normal turned by more than 90 degrees since the step before, and the vertex
        behind the plane it had then) takes the last step's normal instead, its depth measured
        from that plane, and is pushed back the way it came (`narrowphase.trace_toa`; the last
        step's candidates kept by `narrowphase.side_snapshot`). It catches a step's overshoot
        between the feature's half-thickness and its thickness; a vertex carried past the far face
        has no flip left to catch. Off by default: the snapshot sweeps every stage-2 candidate a
        step (about a tenth of the collision time on a scene of many candidates).

    Called by `lodos.sim.make_sim` and `tools.performance_tuning.tune.time_collision`.
    In: m, reads nbody; drbd, reads qpos (B, nq) float32 for B and binds xpos (B, nbody) wp.vec3 and xquat
    (B, nbody) wp.vec4; cprs, nCPR (vertex geometry, SDF geometry) pairs of indices into `surfaces`; surfaces, nB
    surfaces; surface_body_ids, nB ints, each geometry's MuJoCo body; cpr_is_symmetric, nCPR bools or None.
    Out: (mcol, dcol), the context's `CollisionModel` and `CollisionData`, every array but the bound poses
    allocated here.
    """
    if mempool_release_threshold is not None:
        device = wp.get_device()  # wp.context.Device: the current device
        if device.is_cuda and wp.is_mempool_enabled(device):
            wp.set_mempool_release_threshold(
                device,
                max(
                    wp.get_mempool_release_threshold(device), mempool_release_threshold
                ),
            )

    # --------------------------------------------------------------------------------------
    # Geometry tables

    cprs = [list(c) for c in cprs]  # nCPR [vertex geometry, SDF geometry] lists
    nCPR = len(cprs)  # int

    # A half-space can only be the SDF side of a pair, since it has no vertices to
    # test. Caught here rather than as a silently empty contact segment.
    if cpr_is_symmetric is None:
        cpr_is_symmetric = [False] * nCPR  # nCPR bools
    cpr_is_symmetric = [bool(x) for x in cpr_is_symmetric]
    assert len(cpr_is_symmetric) == nCPR, (
        f"cpr_is_symmetric has {len(cpr_is_symmetric)} entries for {nCPR} pairs"
    )

    for j, cpr in enumerate(cprs):  # j: int, the pair; cpr: [int, int], its geometries
        assert not _is_plane(surfaces[cpr[0]]), (
            f"collision pair {j} = {list(cpr)} has a plane as its vertex geometry; "
            "a half-space has no vertices, so it can only be the SDF side"
        )
        if cpr_is_symmetric[j]:
            assert not _is_plane(surfaces[cpr[1]]), (
                f"collision pair {j} = {list(cpr)} is marked symmetric but contains "
                "a plane; a half-space has no vertices, so it can never take the "
                "vertex role and a symmetric declaration is meaningless"
            )

    # A geometry's primitives are its vertices followed by its sphere centres,
    # and `prim_r` is the radius of each -- zero for a vertex. Kept in one array,
    # per geometry, so that `seg_prim` is one contiguous run, a tile is
    # `tile_K` consecutive primitives, and no stage below has to know which
    # kind it is holding.
    def _spheres(surf, g):
        """
        A surface's inner spheres, checked: as many radii as centres, every radius positive.

        Called by `make_collision_context`, once per surface.
        In: surf, a surface (its sph_c and sph_r, none if absent); g, int, its index, for the messages.
        Out: c (nS, 3) float64, the centres, and r (nS,) float64, the radii.
        """
        c = np.asarray(getattr(surf, "sph_c", np.zeros((0, 3))), dtype=np.float64)  # float64, reshaped to (nS, 3)
        r = np.asarray(getattr(surf, "sph_r", np.zeros(0)), dtype=np.float64)  # float64, reshaped to (nS,)
        c, r = c.reshape(-1, 3), r.reshape(-1)
        assert c.shape[0] == r.shape[0], (
            f"surface {g} has {c.shape[0]} sphere centres and {r.shape[0]} radii"
        )
        assert r.size == 0 or r.min() > 0.0, (
            f"surface {g} carries a sphere of radius <= 0; a vertex is how a "
            "zero-radius primitive is spelled"
        )
        return c, r

    _sph = [_spheres(surf, g) for g, surf in enumerate(surfaces)]  # nB ((nS, 3), (nS,)) float64 pairs
    v_b = np.concatenate(  # (nV, 3) float64: each geometry's vertices, then its sphere centres
        [
            np.concatenate([np.asarray(surf.v, dtype=np.float64), c])
            for surf, (c, _) in zip(surfaces, _sph)
        ]
    )
    prim_r = np.concatenate(  # (nV,) float64: 0 for a vertex, the radius for a sphere
        [
            np.concatenate([np.zeros(np.asarray(surf.v).shape[0]), r])
            for surf, (_, r) in zip(surfaces, _sph)
        ]
    )
    g_nS_np = np.array([r.shape[0] for _, r in _sph], dtype=int)  # (nB,) int: spheres per geometry
    is_plane = [_is_plane(surf) for surf in surfaces]  # nB bools: True for a half-space

    for g, p in enumerate(is_plane):  # g: int, a geometry; p: bool, whether it is a plane
        assert not (p and g_nS_np[g]), (
            f"surface {g} is a half-space and carries {g_nS_np[g]} spheres; a "
            "plane can only ever be the SDF side of a pair, so its primitives "
            "would never be tested"
        )

    # The SDF representation, which every non-plane geometry must share -- see
    # `CollisionModel`. Planes carry no SDF table and so hold no opinion.
    kinds = {  # {int: int}: each non-plane geometry's spline code, 0 tricubic, 1 trilinear
        g: spline_kind(surf)
        for g, (p, surf) in enumerate(zip(is_plane, surfaces))
        if not p
    }
    assert len(set(kinds.values())) <= 1, (
        "a collision context evaluates one SDF representation, but its surfaces "
        "disagree: "
        + ", ".join(
            f"geom {g} is {'linear' if k else 'cubic'}" for g, k in sorted(kinds.items())
        )
    )
    spline = next(iter(kinds.values()), 0)  # int: the context's spline code, 0 when every geometry is a plane

    # A surface whose `spline_type` was flipped after `load_surface_data` built its
    # per-surface evaluator would run the packed tables one way and every reference
    # check against `imps` the other. Caught here; pass `spline_type=` to the loader.
    for g, k in kinds.items():  # k: int, the geometry's spline code
        imps = getattr(surfaces[g], "imps", None)  # implicit.ImplicitSurface or None: the surface's own evaluator
        if imps is not None:
            assert int(imps.spline) == k, (
                f"surface {g} says spline_type={surfaces[g].spline_type!r} but its "
                f"`imps` was built for the other representation; set the type when "
                f"loading rather than afterwards"
            )

    B = int(drbd.qpos.shape[0])  # int: worlds
    nB = len(surface_body_ids)  # int: geometries
    # The length of `v_b`, i.e. primitives rather than vertices: a geometry's
    # spheres sit in the same array right after its vertices.
    nV = v_b.shape[0]  # int

    g_nV_np = np.array([surf.v.shape[0] for surf in surfaces])  # (nB,) int: vertices per geometry

    # --------------------------------------------------------------------------------------
    # Segments
    #
    # `(pair, direction)`, ordered by pair. That ordering is load-bearing: a
    # pair's contacts must form one contiguous run, which is what `build_T`
    # relies on.

    # The global primitive indices each geometry owns, ascending -- the run a
    # segment enumerates, its vertices then its spheres. `idm_nC_to_prim` is an
    # explicit indirection rather than a base and a count, so a segment can hold
    # any subset of its geometry's primitives without a kernel knowing how it was
    # chosen.
    g_nP_np = g_nV_np + g_nS_np  # (nB,) int: primitives per geometry
    _v_beg = np.concatenate([[0], np.cumsum(g_nP_np)]).astype(int)  # (nB + 1,) int: each geometry's first primitive
    g_prims = [  # nB (g_nP,) int arrays: each geometry's primitives, as indices into v_b
        np.arange(int(g_nP_np[gi]), dtype=int) + int(_v_beg[gi]) for gi in range(nB)
    ]

    seg_cpr, seg_dir = [], []  # nSEG ints each: a segment's pair and direction
    cpr_seg_beg = [0]  # nCPR + 1 ints: each pair's first segment, then nSEG

    for j in range(nCPR):
        # A symmetric pair owns both directions: each geometry's vertices are
        # tested against the other's SDF. It stays *one* pair, so it costs one
        # entry in every per-pair table and one OBB test, not two.
        for dr in (0, 1) if cpr_is_symmetric[j] else (0,):  # dr: int, a direction
            seg_cpr.append(j)
            seg_dir.append(dr)
        cpr_seg_beg.append(len(seg_cpr))

    nSEG = len(seg_cpr)  # int
    cpr_seg_beg = np.asarray(cpr_seg_beg, dtype=int)

    seg_g_v = [cprs[seg_cpr[s]][seg_dir[s]] for s in range(nSEG)]  # nSEG ints: the geometry supplying the vertices
    seg_g_s = [cprs[seg_cpr[s]][1 - seg_dir[s]] for s in range(nSEG)]  # nSEG ints: the geometry supplying the SDF
    seg_prim = [g_prims[g] for g in seg_g_v]  # nSEG int arrays: each segment's primitives, as indices into v_b
    seg_nC = [int(pr.shape[0]) for pr in seg_prim]  # nSEG ints: candidates per segment


    # --------------------------------------------------------------------------------------
    # Collision Model

    mcol = CollisionModel()  # CollisionModel

    mcol.B = wp.int32(B)
    mcol.nB = wp.int32(nB)
    mcol.nV = wp.int32(nV)
    mcol.nCPR = wp.int32(nCPR)
    mcol.nSEG = wp.int32(nSEG)
    mcol.nC = wp.int32(np.sum(seg_nC))
    mcol.spline = wp.int32(spline)

    # Half-spaces contribute nothing to any of the packed voxel tables, so their
    # per-geometry entries are zeros and their table strides are zero -- which keeps
    # the cumsum bookkeeping of the spline geometries around them unchanged.
    g_nVox = [0 if p else int(np.prod(surf.N)) for p, surf in zip(is_plane, surfaces)]  # nB ints: voxels per geometry
    mcol.nVox = wp.int32(np.sum(g_nVox))

    def _per_geom(attr, fill):
        """
        A surface attribute per geometry, `fill` for a plane, which carries none.

        Called by `make_collision_context`, for the voxel grids' `p_min`, `p_max`, `N`, `h` and `r`.
        In: attr, str, the attribute; fill, its value for a plane. Out: a list of nB values.
        """
        return [
            fill if p else getattr(surf, attr) for p, surf in zip(is_plane, surfaces)
        ]

    mcol.cprs = wp.array(np.asarray(cprs, dtype=int), dtype=wp.vec2i)
    mcol.cpr_seg_beg = wp.array(np.asarray(cpr_seg_beg, dtype=int), dtype=wp.int32)
    mcol.seg_cpr = wp.array(seg_cpr, dtype=wp.int32)
    mcol.seg_g_v = wp.array(seg_g_v, dtype=wp.int32)
    mcol.seg_g_s = wp.array(seg_g_s, dtype=wp.int32)
    mcol.seg_nC = wp.array(seg_nC, dtype=wp.int32)
    mcol.seg_dir = wp.array(seg_dir, dtype=wp.int32)
    mcol.g_plane = wp.array(np.array(is_plane, dtype=int), dtype=wp.int32)
    mcol.g_nVox = wp.array(g_nVox, dtype=wp.int32)
    mcol.g_nVox_cumsum = wp.array(np.cumsum([0] + g_nVox), dtype=wp.int32)
    mcol.g_Vox_pmin = wp.array(_per_geom("p_min", np.zeros(3)), dtype=wp.vec3)
    mcol.g_Vox_pmax = wp.array(_per_geom("p_max", np.zeros(3)), dtype=wp.vec3)
    mcol.g_Vox_N = wp.array(_per_geom("N", np.zeros(3, dtype=int)), dtype=wp.vec3i)
    mcol.g_Vox_h = wp.array(_per_geom("h", np.ones(3)), dtype=wp.vec3)
    mcol.g_Vox_r = wp.array(_per_geom("r", 0.0), dtype=wp.float32)
    g_nV = g_nV_np  # (nB,) int: vertices per geometry
    g_nV_cumsum = np.concat([[0], np.cumsum(g_nV)])  # (nB + 1,) int, leading zero
    mcol.g_nV = wp.array(g_nV, dtype=wp.int32)
    mcol.g_nV_cumsum = wp.array(g_nV_cumsum, dtype=wp.int32)
    mcol.g_nS = wp.array(g_nS_np, dtype=wp.int32)
    mcol.g_nP = wp.array(g_nP_np, dtype=wp.int32)
    mcol.has_spheres = wp.int32(1 if int(g_nS_np.sum()) else 0)
    assert contact_point in CONTACT_POINTS, (
        f"contact_point must be one of {CONTACT_POINTS}, got {contact_point!r}"
    )
    mcol.contact_point = wp.int32(CONTACT_POINTS.index(contact_point))
    mcol.g_bid = wp.array(np.array(surface_body_ids, dtype=int), dtype=wp.int32)

    # `obb_frame_np` normalizes by edge length, so it cannot be handed a degenerate
    # box; planes get an identity frame instead, which no kernel reads.
    obb_c = np.zeros((nB, 3))  # (nB, 3) float64: OBB centres
    obb_A = np.tile(np.eye(3), (nB, 1, 1))  # (nB, 3, 3) float64: OBB axes, as rows
    obb_h = np.zeros((nB, 3))  # (nB, 3) float64: OBB half-extents
    real = [g for g in range(nB) if not is_plane[g]]  # nr ints: the non-plane geometries
    if real:
        c, A, h = obb_frame_np(np.stack([surfaces[g].bbox for g in real]))  # (nr, 3), (nr, 3, 3), (nr, 3) float64
        obb_c[real], obb_A[real], obb_h[real] = c, A, h
    mcol.g_obb_c = wp.array(obb_c, dtype=wp.vec3)
    mcol.g_obb_A = wp.array(obb_A, dtype=wp.mat33)
    mcol.g_obb_h = wp.array(obb_h, dtype=wp.vec3)

    # --------------------------------------------------------------------------------------
    # Vertex tiles. A segment under `2 * tile_size` vertices gains nothing from tiling: lower `tile_size` for such a
    # scene.
    assert 0.0 < capacity1 <= 1.0, "capacity1 must be in (0, 1]"

    def _per_seg(spec, name, fallback, conv=int):
        """A scalar, or a {(vertex geom, SDF geom): value} dict, read per segment.

        The tuple is the key because a segment *is* a directional pair, and it is
        the only key that tells a symmetric pair's two directions apart.

        Called by `make_collision_context`, for `tile_size` and `super_size`, and by `_per_pair`.
        In: spec, an int or a dict; name, str, the option's name, for the message; fallback, int, the value for a
        segment the dict does not name and no `None` key covers; conv, a value to its int (int by default).
        Out: a list of nSEG ints.
        """
        if not isinstance(spec, dict):
            return [conv(spec)] * nSEG

        keys = {(seg_g_v[s], seg_g_s[s]) for s in range(nSEG)}  # set of (int, int): the directional pairs
        unknown = sorted(  # list of str: the keys that name no directional pair
            str(k) for k in spec if k is not None and tuple(k) not in keys
        )
        assert not unknown, (
            f"{name} names {', '.join(unknown)}, which are not directional pairs "
            f"of this context. Keys are (vertex geometry, SDF geometry) tuples; "
            f"this context has {sorted(keys)}. `None` sets the default for the "
            "rest."
        )
        named = {tuple(k): v for k, v in spec.items() if k is not None}  # {(int, int): value}: the named pairs
        dflt = spec.get(None, fallback)  # the value for the pairs not named
        return [conv(named.get((seg_g_v[s], seg_g_s[s]), dflt)) for s in range(nSEG)]

    seg_K = _per_seg(tile_size, "tile_size", 8)  # nSEG ints: primitives per tile
    seg_SK = _per_seg(super_size, "super_size", 0)  # nSEG ints: tiles per super-tile, 0 for none

    # Time of arrival, per segment: a bool for every pair, a list of nCPR bools (a symmetric pair's
    # two directions take its flag), or `tile_size`'s per-direction dict.
    def _per_pair(spec, name, fallback, conv):
        """`toa`'s spellings, read per segment through `conv`: one value for every pair, a list of nCPR values (a
        symmetric pair's two directions take its value), or `_per_seg`'s dict.

        Called by `make_collision_context`, for `toa`, `toa_trace` and `toa_trace_iters`.
        In: spec; name str, the option's name, for messages; fallback, a dict's value for the segments it does not
        name; conv, a value to its int code. Out: a list of nSEG ints.
        """
        if isinstance(spec, dict):
            return _per_seg(spec, name, fallback, conv)
        if isinstance(spec, (list, tuple, np.ndarray)):
            assert len(spec) == nCPR, f"{name} has {len(spec)} entries for {nCPR} pairs"
            return [conv(spec[seg_cpr[s]]) for s in range(nSEG)]
        return [conv(spec)] * nSEG

    def _trace_mode(x):
        """`toa_trace`'s value as its code: 0 no trace, 1 sphere tracing, 2 over-relaxed (True's)."""
        if isinstance(x, str):
            assert x in ("sphere", "relaxed"), f"toa_trace: {x!r}, expected False, True, 'sphere' or 'relaxed'"
            return 1 if x == "sphere" else 2
        return 2 if bool(x) else 0

    def _trace_iters(x):
        """`toa_trace_iters`'s value, checked."""
        assert int(x) >= 1, f"toa_trace_iters must be at least 1, got {x}"
        return int(x)

    seg_toa = _per_pair(toa, "toa", False, lambda x: int(bool(x)))  # nSEG ints: 1 for a `toa` segment
    seg_trace = _per_pair(toa_trace, "toa_trace", "relaxed", _trace_mode)  # nSEG ints: each segment's trace
    seg_trace_iters = _per_pair(toa_trace_iters, "toa_trace_iters", 128, _trace_iters)  # nSEG ints
    seg_drift = [x & t for x, t in zip(_per_pair(toa_drift, "toa_drift", False, lambda x: int(bool(x))), seg_toa)]
    seg_side = [x & t for x, t in zip(_per_pair(toa_side, "toa_side", False, lambda x: int(bool(x))), seg_toa)]
    cpr_toa = [int(any(seg_toa[cpr_seg_beg[j]:cpr_seg_beg[j + 1]])) for j in range(nCPR)]  # nCPR ints
    has_toa = int(any(seg_toa))  # int: 1 when some segment is `toa`
    toa_h = 0.0  # float: the motion bound's horizon (s)
    g_root = [0] * nB  # nB ints: each geometry's tree root, read only by `toa` kernels
    if has_toa:
        toa_h = float(m.opt.timestep) if toa_horizon is None else float(toa_horizon)
        assert toa_h > 0.0, f"toa_horizon must be positive, got {toa_h}"
        g_root = [int(m.body_rootid[int(b)]) for b in surface_body_ids]
    mcol.has_toa = wp.int32(has_toa)
    mcol.toa_h = toa_h
    mcol.cpr_toa = wp.array(cpr_toa, dtype=wp.int32)
    mcol.seg_toa = wp.array(seg_toa, dtype=wp.int32)
    mcol.seg_trace = wp.array(seg_trace if has_toa else [0], dtype=wp.int32)
    mcol.seg_trace_iters = wp.array(seg_trace_iters if has_toa else [0], dtype=wp.int32)
    mcol.seg_drift = wp.array(seg_drift if has_toa else [0], dtype=wp.int32)
    mcol.seg_side = wp.array(seg_side if has_toa else [0], dtype=wp.int32)
    mcol.has_side = wp.int32(int(any(seg_side)))
    mcol.toa_dt = float(m.opt.timestep)
    mcol.g_root = wp.array(g_root, dtype=wp.int32)
    tree = [np.zeros(1, dtype=np.int32)] * 3  # body_parentid, body_dofadr, body_dofnum: (1,) without `toa`
    if has_toa:
        tree = [np.asarray(x, dtype=np.int32) for x in (m.body_parentid, m.body_dofadr, m.body_dofnum)]
    mcol.body_parentid, mcol.body_dofadr, mcol.body_dofnum = (wp.array(x, dtype=wp.int32) for x in tree)

    for s in range(nSEG):  # s: int, a segment
        assert seg_K[s] >= 1, (
            f"tile_size is {seg_K[s]} for the segment testing geometry "
            f"{seg_g_v[s]}'s primitives against {seg_g_s[s]}; a tile holds at "
            "least one primitive"
        )
        assert seg_SK[s] == 0 or seg_SK[s] >= 2, (
            f"super_size is {seg_SK[s]} for the segment testing geometry "
            f"{seg_g_v[s]}'s primitives against {seg_g_s[s]}; a super-tile over "
            "one tile bounds the same primitives twice, so 0 is how the level is "
            "turned off"
        )

    seg_nT = [int(np.ceil(n / k)) for n, k in zip(seg_nC, seg_K)]  # nSEG ints: tiles per segment
    max_nT = max(seg_nT) if seg_nT else 0  # int

    def _bound(idx):
        """The bounding sphere of a run of primitives, in their own body frame.

        `+ qr`: the bound has to enclose the members' own spheres, or
        `tile_survives` would reject a tile holding a sphere that reaches the
        surface. Every radius is zero without spheres, so this is then the plain
        vertex bound.

        Shared by both levels: a super-tile is this same sphere over more
        primitives.

        Called by `make_collision_context`, for every tile and super-tile.
        In: idx, (n,) int, the primitives, as indices into v_b. Out: c (3,) float64, the centre; float, the radius.
        """
        q, qr = v_b[idx], prim_r[idx]  # (n, 3) float64, (n,) float64: the primitives and their radii
        c = 0.5 * (q.min(axis=0) + q.max(axis=0))  # (3,) float64: the centre of their bounding box
        return c, float((np.linalg.norm(q - c, axis=1) + qr).max())

    tile_c, tile_r = [], []  # nT_world (3,) float64 centres and float radii
    for pr, nt, k in zip(seg_prim, seg_nT, seg_K):  # pr: (seg_nC,) int, a segment's primitives; nt, k: int
        for t in range(nt):  # t: int, a tile of the segment
            c, r = _bound(pr[t * k : (t + 1) * k])  # (3,) float64, float: the tile's bounding sphere
            tile_c.append(c)
            tile_r.append(r)

    # The generators, laid out per segment and in tile order so that the
    # candidate list they expand into stays ordered (batch, segment, tile).
    # A segment with super-tiles contributes one generator each; a segment
    # without contributes a single pass-through covering all of its tiles, which
    # is the per-pair disable and costs no SDF test. See `CollisionModel.gen_seg`.
    seg_nG, gen_seg, gen_sup, gen_tile_beg, gen_nT = [], [], [], [], []  # nSEG ints, then nG_world ints each
    gen_c, gen_r, tile_gen = [], [], []  # nG_world (3,) float64 centres and float radii; nT_world ints
    for s in range(nSEG):
        pr, nt, k, sk = seg_prim[s], seg_nT[s], seg_K[s], seg_SK[s]  # sk: int, the segment's super-tile size
        g0 = len(gen_seg)  # int: the segment's first generator

        if sk and nt > 1:
            for u in range(int(np.ceil(nt / sk))):  # u: int, a super-tile of the segment
                tb = u * sk  # int: its first tile
                n = min(sk, nt - tb)  # int: its tiles
                c, r = _bound(pr[tb * k : (tb + n) * k])
                gen_seg.append(s)
                gen_sup.append(1)
                gen_tile_beg.append(tb)
                gen_nT.append(n)
                gen_c.append(c)
                gen_r.append(r)
            tile_gen.extend(g0 + t // sk for t in range(nt))
        else:
            # Including a segment that asked for super-tiles but has one tile or
            # none: there is nothing for the level to reject.
            gen_seg.append(s)
            gen_sup.append(0)
            gen_tile_beg.append(0)
            gen_nT.append(nt)
            gen_c.append(np.zeros(3))
            gen_r.append(0.0)
            tile_gen.extend([g0] * nt)

        seg_nG.append(len(gen_seg) - g0)

    # The tile stage is indexed by the tiles a world *has*, densely: segment `s`
    # owns `[seg_nT_beg[s], seg_nT_beg[s] + seg_nT[s])`, which is the same run
    # `tile_c`/`tile_r` are already stored in.
    nT_world = int(sum(seg_nT))  # int: tiles per world
    nG_world = len(gen_seg)  # int: generators per world

    # One shared tile size lets a stage-1 slot's tile be `i / tile_K`:
    # `broadphase.sdf_filter_kernel` compiles that resolve in, and `compact_nTa`
    # skips the tables the lookup needs.
    uniform_K = len(set(seg_K)) <= 1  # bool: every segment has the same tile size
    if partition is None:
        partition = not uniform_K  # bool: the lookup wherever the tile sizes differ
    assert partition or uniform_K, (
        f"partition=False, but this context's segments have tile sizes "
        f"{sorted(set(seg_K))}; a stage-1 slot's tile is only `i / tile_K` where "
        "they agree"
    )

    mcol.tile_K = wp.int32(seg_K[0] if uniform_K and seg_K else max(seg_K, default=1))
    mcol.uniform_K = wp.int32(1 if uniform_K else 0)
    mcol.partition = wp.int32(1 if partition else 0)
    mcol.max_nT = wp.int32(max_nT)
    mcol.nT_world = wp.int32(nT_world)
    mcol.n_tile_slots = wp.int32(B * nT_world)
    mcol.n_slot_world = wp.int32(sum(nt * k for nt, k in zip(seg_nT, seg_K)))
    mcol.nG_world = wp.int32(nG_world)
    mcol.has_super = wp.int32(1 if any(gen_sup) else 0)
    mcol.seg_compact = wp.int32(1 if capacity1 < 1.0 else 0)
    mcol.seg_K = wp.array(seg_K, dtype=wp.int32)
    mcol.seg_nT = wp.array(seg_nT, dtype=wp.int32)
    mcol.seg_nT_beg = wp.array(np.cumsum([0] + seg_nT[:-1]), dtype=wp.int32)
    mcol.tile_c = wp.array(np.asarray(tile_c), dtype=wp.vec3)
    mcol.tile_r = wp.array(np.asarray(tile_r), dtype=wp.float32)
    mcol.tile_seg = wp.array(
        np.repeat(np.arange(nSEG, dtype=np.int32), seg_nT), dtype=wp.int32
    )

    mcol.seg_nG = wp.array(seg_nG, dtype=wp.int32)
    mcol.seg_nG_beg = wp.array(np.cumsum([0] + seg_nG[:-1]), dtype=wp.int32)
    mcol.gen_seg = wp.array(gen_seg, dtype=wp.int32)
    mcol.gen_sup = wp.array(gen_sup, dtype=wp.int32)
    mcol.gen_tile_beg = wp.array(gen_tile_beg, dtype=wp.int32)
    mcol.gen_nT = wp.array(gen_nT, dtype=wp.int32)
    mcol.gen_c = wp.array(np.asarray(gen_c, dtype=float).reshape(-1, 3), dtype=wp.vec3)
    mcol.gen_r = wp.array(np.asarray(gen_r, dtype=float), dtype=wp.float32)
    mcol.tile_gen = wp.array(np.asarray(tile_gen, dtype=int), dtype=wp.int32)

    mcol.v_b = wp.array(v_b, dtype=wp.vec3)
    mcol.prim_r = wp.array(prim_r, dtype=wp.float32)

    # One channel, the signed distance: the narrowphase differentiates the field
    # for the normal rather than reading a baked one.
    #
    # Both packed tables stride each geometry by its own grid size -- prod(N + 3)
    # of tricubic coefficients, prod(N + 1) of knot samples -- so a surface whose
    # array does not have exactly that many entries offsets every geometry after
    # it, silently and with plausible-looking values. Checked rather than trusted.
    #
    # Only the live representation is packed. The trilinear samples are derived
    # from the tricubic coefficients rather than loaded, since a B-spline
    # interpolates its knots and `spline.c0.knots_from_c_3d` inverts the fit exactly,
    # so any bake can be evaluated either way.
    def pack(what, per_geom_size, values):
        """
        One SDF table over every geometry, each surface's entry count checked against its grid.

        Called by `make_collision_context`, for the live representation's table.
        In: what, str, the entries' name, for the message; per_geom_size, surface -> int, the count a grid asks
        for; values, surface -> array, its entries. Out: (sum sizes,) float, the planes contributing nothing,
        and (nB + 1,) int, the cumsum of the sizes with a leading zero.
        """
        sizes = [0 if p else per_geom_size(surf) for p, surf in zip(is_plane, surfaces)]  # nB ints, 0 for a plane
        arrays = [  # nB flat arrays: each geometry's entries
            np.zeros(0) if p else np.asarray(values(surf)).reshape(-1)
            for p, surf in zip(is_plane, surfaces)
        ]
        for g, (c, n) in enumerate(zip(arrays, sizes)):  # g: int, a geometry; c its entries; n: int, its size
            assert c.size == n, (
                f"surface {g} has {c.size} {what} where its grid N="
                f"{tuple(np.asarray(surfaces[g].N))} asks for {n}"
            )
        return np.concat(arrays, axis=0), np.cumsum([0] + sizes)

    # What the unused table is left as: one element, read by nothing.
    def unused():
        """
        A placeholder table and cumsum for the representation not packed.

        Called by `make_collision_context`. Out: (1,) float32 and (1,) int32 zeros.
        """
        return wp.zeros(1, dtype=wp.float32), wp.zeros(1, dtype=wp.int32)

    if spline == 1:
        knots, cumsum = pack(  # (sum prod(N + 1),) float64, (nB + 1,) int: the knot samples, their cumsum
            "knot samples",
            lambda surf: int(np.prod(np.asarray(surf.N) + 1)),
            lambda surf: spline_c0.knots_from_c_3d(surf.sdf_coeffs),
        )
        mcol.vox_c_sdf_c0 = wp.array(knots, dtype=wp.float32)
        mcol.g_nVox_c0_cumsum = wp.array(cumsum, dtype=wp.int32)
        mcol.vox_c_sdf_c2, mcol.g_nVox_c2_cumsum = unused()
    else:
        coeffs, cumsum = pack(  # (sum prod(N + 3),) float: the tricubic coefficients
            "spline coefficients",
            lambda surf: int(np.prod(np.asarray(surf.N) + 3)),
            lambda surf: surf.sdf_coeffs,
        )
        mcol.vox_c_sdf_c2 = wp.array(coeffs, dtype=wp.float32)
        mcol.g_nVox_c2_cumsum = wp.array(cumsum, dtype=wp.int32)
        mcol.vox_c_sdf_c0, mcol.g_nVox_c0_cumsum = unused()
    mcol.vox_c_sdf_vox = wp.array(
        np.concat(
            [
                np.zeros(0) if p else surf.vsdf_values.flatten()
                for p, surf in zip(is_plane, surfaces)
            ],
            axis=0,
        ),
        dtype=wp.float32,
    )

    mcol.idm_nC_to_nSEG = wp.array(
        np.concat([seg_nC[s] * [s] for s in range(nSEG)]), dtype=wp.int32
    )

    # A segment enumerates global vertex indices -- the whole of its geometry's
    # vertices. The indirection, rather than an `arange`, is what lets a
    # segment hold a subset without a kernel caring.
    mcol.idm_nC_to_prim = wp.array(
        (
            np.concat(seg_prim).astype(int)
            if nSEG
            else np.zeros(0, dtype=int)
        ),
        dtype=wp.int32,
    )
    mcol.idul_nSEG_to_nC = wp.array(np.cumsum([0] + seg_nC), dtype=wp.int32)

    # --------------------------------------------------------------------------------------
    # Collision Data

    dcol = CollisionData()  # CollisionData

    # Stage 1 is the primitive slots of the surviving tiles, `nTa_max * tile_size`
    # where the tile sizes agree.
    n_tile_slots = B * nT_world  # int: tiles over all worlds
    nTa_max = max(1, int(np.ceil(capacity_tile * n_tile_slots)))  # int: slots for the surviving tiles
    # `capacity1` sizes the box filter's expansion into one slot per *tile* of
    # an active segment. At 1.0 the list is the whole grid, which is what the
    # dense launch already visits, so the compaction is skipped rather than run
    # for nothing: 1.0 is exactly the pipeline without it.
    seg_compact = capacity1 < 1.0  # bool
    nTb_max = (  # int: slots for the tiles of active segments
        max(1, int(np.ceil(capacity1 * n_tile_slots))) if seg_compact
        else n_tile_slots
    )
    # Every surviving tile contributes its own size in stage-1 slots, so where
    # the sizes agree the stage-1 allocation is that product exactly. Where they
    # do not, which tiles survive decides how many slots they want, so
    # `capacity_tile` is applied to the slot total instead -- the same fraction
    # of the same worst case, and the identical number when the sizes agree.
    # Stage 1 can then overflow on its own, which is what `overflow[0]` reports.
    if uniform_K:
        nCa1_max = nTa_max * int(mcol.tile_K)  # int: stage-1 slots
    else:
        nCa1_max = max(
            1, int(np.ceil(capacity_tile * B * int(mcol.n_slot_world)))
        )
    nCa2_max = int(np.ceil(capacity2 * nCa1_max))  # int: stage-2 slots
    nCa3_max = nCa2_max  # int: stage-3 slots, the contacts

    dcol.dyn_scan = wp.int32(1 if dyn_scan else 0)
    # Two of the three sites are `nTa_max` long (`tile_off` by one more), the
    # third `nCa2_max`; one buffer covers all of them, and it is sized for the
    # *worst case* rather than for this context's capacities so that
    # `collide.resize_context` can grow one without a second allocation to
    # remember: the tiles at `n_tile_slots`, stage 2 at stage 1's whole grid,
    # which a resize cannot take it past.
    nCa1_worst = (  # int: stage-1 slots at capacity_tile 1.0
        n_tile_slots * int(mcol.tile_K) if uniform_K else B * int(mcol.n_slot_world)
    )
    dcol.scan_buf = wp.zeros(
        max(1, lscan.workspace(n_tile_slots + 1, nCa1_worst)), dtype=wp.int32
    )

    dcol.nCa1_grid = wp.int32(_stage1_grid(nCa1_max, grid_stride))
    dcol.nTa_max = wp.int32(nTa_max)
    dcol.nTb_max = wp.int32(nTb_max)
    dcol.nCa1_max = wp.int32(nCa1_max)
    dcol.nCa2_max = wp.int32(nCa2_max)
    dcol.nCa3_max = wp.int32(nCa3_max)

    # Five slots: the three stage counts, the surviving tiles, then the tiles of
    # active segments the tile filter is launched over.
    dcol.ct = wp.zeros(5, dtype=wp.int32)
    dcol.ct_raw = wp.zeros(5, dtype=wp.int32)
    dcol.ct_peak = wp.zeros(5, dtype=wp.int32)
    dcol.overflow = wp.zeros(5, dtype=wp.int32)

    # Body poses, shared with the rigid-body context rather than copied from it.
    # `broadphase.compute_T_w` gathers them into the per-geometry `T_w`.
    assert drbd.xpos.shape == (B, int(m.nbody)), "rbd context is for another model"
    dcol.xpos = drbd.xpos
    dcol.xquat = drbd.xquat
    dcol.T_w = wp.zeros((B, nB), dtype=wp.transformf)
    dcol.T_rel = wp.zeros((B, nSEG), dtype=wp.transformf)
    # Velocities, bound like the poses; only `toa` kernels read them.
    dcol.cvel = drbd.cvel
    dcol.subtree_com = drbd.subtree_com
    dcol.twist_rel = wp.zeros((B, nSEG) if has_toa else (1, 1), dtype=wp.spatial_vector)
    dcol.cdof = drbd.cdof
    dcol.qacc_smooth = drbd.qacc_smooth
    dcol.twist_end = wp.zeros((B, nSEG) if has_toa else (1, 1), dtype=wp.spatial_vector)
    dcol.toa_band = wp.zeros(nB if has_toa else 1, dtype=wp.int32)

    dcol.cpr_active = wp.zeros((B, nCPR), dtype=wp.bool)
    # `B * nG_world`, which is `B * nSEG` exactly when no segment has
    # super-tiles: a generator is then a segment.
    dcol.gen_w = wp.zeros(B * nG_world, dtype=wp.int32)
    dcol.gen_s = wp.zeros(B * nG_world, dtype=wp.int32)
    dcol.gcnt = wp.zeros(B * nG_world, dtype=wp.int32)
    dcol.goff = wp.zeros(B * nG_world, dtype=wp.int32)
    dcol.idul_B_nSEG_to_nCa1 = wp.zeros((B, nSEG + 1), dtype=wp.int32)
    dcol.idul_B_nSEG_to_nTa = wp.zeros((B, nSEG + 1), dtype=wp.int32)

    dcol.tflag = wp.zeros(nTb_max, dtype=wp.int32)
    dcol.tslot = wp.zeros(nTb_max if seg_compact else 1, dtype=wp.int32)
    dcol.tcnt = wp.zeros(nTa_max, dtype=wp.int32)
    dcol.toff1 = wp.zeros(nTa_max, dtype=wp.int32)
    dcol.idm_nTa_to_slot = wp.zeros(nTa_max, dtype=wp.int32)
    # Only the lookup resolve reads these, so a context that kept the division
    # allocates neither the tables nor the scan over them. Gated on `partition`
    # rather than on differing sizes, since the lookup can also be switched on
    # over a uniform scene.
    dcol.tile_sz = wp.zeros(nTa_max + 1 if partition else 1, dtype=wp.int32)
    dcol.tile_off = wp.zeros(nTa_max + 1 if partition else 1, dtype=wp.int32)
    dcol.blk_base = wp.zeros(
        nCa1_max // PARTITION_BLK + 2 if partition else 1, dtype=wp.int32
    )

    dcol.flag1 = wp.zeros(nCa1_max, dtype=wp.int32)
    dcol.idm_nCa2_to_nCa1 = wp.zeros(nCa2_max, dtype=wp.int32)
    dcol.idul_B_nSEG_to_nCa2 = wp.zeros((B, nSEG + 1), dtype=wp.int32)

    dcol.sdf_nCa2 = wp.zeros(nCa2_max, dtype=wp.float32)
    dcol.n_nCa2 = wp.zeros(nCa2_max, dtype=wp.vec3)

    # The stage-2 -> stage-3 compaction tables, which every context allocates.
    dcol.flag2 = wp.zeros(nCa2_max, dtype=wp.int32)
    dcol.off2 = wp.zeros(nCa2_max, dtype=wp.int32)
    dcol.trace_list = wp.zeros(nCa2_max if has_toa else 1, dtype=wp.int32)
    dcol.e_nCa2 = wp.zeros(nCa2_max if has_toa else 1, dtype=wp.float32)
    dcol.trace_ct = wp.zeros(1, dtype=wp.int32)
    dcol.idm_nCa3_to_nCa2 = wp.zeros(nCa3_max, dtype=wp.int32)
    dcol.side_v = wp.zeros(nCa2_max if has_toa else 1, dtype=wp.int32)
    dcol.side_nB = wp.zeros(nCa2_max if has_toa else 1, dtype=wp.vec3)
    dcol.side_c = wp.zeros(nCa2_max if has_toa else 1, dtype=wp.float32)
    dcol.side_idul = wp.zeros((B, nSEG + 1) if has_toa else (1, 1), dtype=wp.int32)
    dcol.idul_B_nSEG_to_nCa3 = wp.zeros((B, nSEG + 1), dtype=wp.int32)

    dcol.p_nCa3 = wp.zeros(nCa3_max, dtype=wp.vec3)
    dcol.n_nCa3 = wp.zeros(nCa3_max, dtype=wp.vec3)
    dcol.sdf_nCa3 = wp.zeros(nCa3_max, dtype=wp.float32)
    dcol.b_nCa3 = wp.zeros(nCa3_max, dtype=wp.int32)
    dcol.seg_nCa3 = wp.zeros(nCa3_max, dtype=wp.int32)
    dcol.v_nCa3 = wp.zeros(nCa3_max, dtype=wp.int32)

    return mcol, dcol
