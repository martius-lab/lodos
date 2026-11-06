import copy
import importlib
from pathlib import Path

import numpy as np
import warp as wp

from . import implicit
from .spline import c0 as spline_c0


class PlaneSurface:
    """
    A half-space collision geometry: the SDF is the z coordinate in the geometry's
    own body frame and the normal is +z.

    One can pass a PlaneSurface() object to `make_collision_context` where a processed spline surface
    would otherwise go. It carries no vertices, which is why it can only ever be
    the *SDF* side of a collision pair -- and why it needs no voxel grid, no spline
    coefficients and no bounding box. Every kernel that would read those is guarded
    by `CollisionModel.g_plane`.

    Read by `lodos.col.sdf.types.make_collision_context`.
    """

    v = np.zeros((0, 3), dtype=float)  # (0, 3) float64: no vertices
    sph_c = np.zeros((0, 3), dtype=float)  # (0, 3) float64: no sphere centres
    sph_r = np.zeros(0, dtype=float)  # (0,) float64: no sphere radii


def _is_plane(surf):
    """
    Whether a surface is a PlaneSurface() instance.

    Called by `lodos.col.sdf.types.make_collision_context`, `lodos.sim.make_sim` and
    `lodos.utils.tuning.surface_fingerprint`.

    In: 
    surf, a surface 

    Out: 
    bool
    """
    return isinstance(surf, PlaneSurface)


class SurfaceData:
    """
    A loaded surface: every array of its `surface_data.npz` as an attribute -- v (n, 3) float64, f (nf, 3) int64,
    e (ne, 2) int64 and e_l (ne,) float64, the grid's p_min, p_max and h (3,) float32, N (3,) int32 and r () float64,
    sdf_coeffs (Nx+3, Ny+3, Nz+3) float32, vsdf_values (Nx, Ny, Nz) float32, and bbox (8, 3) float64 where the
    bake ships one -- and what `load_surface_data` adds: source str, spline_type str, sph_c (nS, 3) and sph_r
    (nS,) float64, and imps, an `implicit.ImplicitSurface`.

    Built by `load_surface_data`.
    """

    def __init__(self, dictionary):
        """
        Sets each of `dictionary`'s items as an attribute.

        Called by `load_surface_data`.
        In: dictionary, a mapping of names to arrays, the bake's `np.load` result.
        """
        for k, v in dictionary.items():  # k: str, a key; v: np.ndarray, its array
            setattr(self, k, v)


#######################################################################################
# Spline type
#######################################################################################

# The SDF representations a surface may be evaluated with, and the integer the
# Warp side carries. Both are fitted to the *same* knot samples, so a bake is not
# specific to either: `spline.c0.knots_from_c_3d` recovers the samples from the tricubic
# coefficients exactly, which is why "linear" needs no new asset.
#
#   "cubic"   tricubic B-spline (`spline.c2`), 64 coefficients per query: C2 field, C1 normals
#   "linear"  trilinear interpolation (`spline.c0`), 8 samples per query: C0 field, normals discontinuous
#             across cell faces
#
# See `spline.c0` for what that continuity difference means for a sliding contact.
SPLINE_TYPES = {"cubic": 0, "linear": 1}

DEFAULT_SPLINE_TYPE = "cubic"  # the representation of a surface that names none


def spline_kind(surf):
    """
    The integer spline code of a surface: 0 tricubic, 1 trilinear.

    Reads `spline_type`, defaulting to the tricubic when the attribute is absent
    -- as it is on every bake, since `create_surface_data` writes no such key. The
    value survives an `npz` round trip as a 0-d array, so it is unwrapped rather
    than compared as-is.

    Called by `_fill_spline_type`, `implicit_surface` and `lodos.col.sdf.types.make_collision_context`.

    In: 
    surf, a surface 

    Out: 
    int, 0 or 1; raises ValueError on a name not in `SPLINE_TYPES`
    """
    t = getattr(surf, "spline_type", DEFAULT_SPLINE_TYPE)  # str, or a 0-d array of one
    t = str(np.asarray(t).item())

    if t not in SPLINE_TYPES:
        raise ValueError(
            f"unknown spline_type {t!r}; expected one of {sorted(SPLINE_TYPES)}"
        )

    return SPLINE_TYPES[t]


def _fill_spline_type(sdata, override=None):
    """
    Normalizes `spline_type` to one of `SPLINE_TYPES`' plain strings.

    `override` wins over whatever the file carried, which is how a caller picks a
    representation for a bake that pins none -- the ordinary case, since the
    choice is about how to *evaluate* a surface.

    Called by `load_surface_data` and `with_spline_type`.

    In: 
    sdata, a surface 
    override, str or None 
    
    Out: 
    sdata, its `spline_type` set, in place, to a str
    """
    if override is not None:
        t = override  # str, or a 0-d array of one
    elif hasattr(sdata, "spline_type"):
        t = sdata.spline_type
    else:
        t = DEFAULT_SPLINE_TYPE

    sdata.spline_type = str(np.asarray(t).item())
    spline_kind(sdata)  # rejects an unknown name here rather than at context build

    return sdata


def _fill_spheres(sdata, path):
    """The geometry's inner spheres, from `spheres.npz` beside the bake.

    A sidecar, so that adding spheres to a geometry does not mean rewriting its
    bake, to which every calibrated capacity in the tree is fitted.

    The file holds `c` (N, 3) body-frame centres and `r` (N,) radii. Absent, a
    geometry simply has none, and `make_collision_context` then builds the
    pipeline without them.

    Nothing here checks that a sphere is actually *inside* its geometry. It is
    the caller's guarantee, and a sphere poking out of the surface is not a
    detectable error at this level -- it just collides earlier than the mesh
    does. `lodos.col.sdf.io` has no sphere packer; the centres come from whatever
    produced them.

    Called by `load_surface_data`.

    In: 
    sdata, the surface 
    path, str or Path, its `surface_data.npz`

    Writes: 
    sdata.sph_c (nS, 3) float64 
    sdata.sph_r (nS,) float64, (0, 3) and (0,) without the file
    """
    f = Path(path).parent / "spheres.npz"  # Path: the sidecar
    if not f.exists():
        sdata.sph_c = np.zeros((0, 3), dtype=np.float64)
        sdata.sph_r = np.zeros(0, dtype=np.float64)
        return

    z = np.load(f)  # NpzFile: c and r
    c = np.asarray(z["c"], dtype=np.float64).reshape(-1, 3)  # (nS, 3) float64: the centres
    r = np.asarray(z["r"], dtype=np.float64).reshape(-1)  # (nS,) float64: the radii
    assert c.shape[0] == r.shape[0], (
        f"{f}: {c.shape[0]} sphere centres against {r.shape[0]} radii"
    )
    assert r.size == 0 or r.min() > 0.0, f"{f}: a sphere radius is not positive"
    sdata.sph_c, sdata.sph_r = c, r


def load_surface_data(PATH, spline_type=None):
    """
    A processed surface, with its per-surface Warp evaluator context `imps`.

    Inner spheres are read from `spheres.npz` beside the bake, when there is one;
    see `_fill_spheres`.

    spline_type: "cubic" or "linear", overriding whatever the file says. The
        representation is a choice about evaluation -- both are fitted to the
        same knot samples -- so this is where it is usually made.

        It is set here, and not afterwards on the returned object, because `imps`
        is built from it. `make_collision_context` checks the two agree, so that
        a surface whose attribute was flipped after loading cannot run with a
        stale evaluator.

    Called by `load_surface_file`, `viewer.scenes.nist.load`, `viewer.scenes.nutbolt.load` and 2 others.
    
    In: 
    PATH, str or Path, a `surface_data.npz`; 
    spline_type, str or None

    Out: 
    `SurfaceData`, the bake's arrays with source, spline_type, sph_c, sph_r and imps (see the class)
    """
    surface_data = np.load(PATH)  # NpzFile: the bake's arrays

    sdata = SurfaceData(surface_data)  # SurfaceData
    sdata.source = str(PATH)  # where it was baked, for `lodos.utils.tuning.surface_fingerprint`
    _fill_spline_type(sdata, spline_type)
    _fill_spheres(sdata, PATH)
    sdata.sdf_coeffs = np.ascontiguousarray(surface_data["sdf_coeffs"])
    sdata.imps = implicit_surface(sdata)

    return sdata


def with_spline_type(sdata, spline_type):
    """
    A copy of the loaded surface `sdata` evaluated as `spline_type` ("cubic" or "linear"): the same
    bake -- both representations read the same coefficients -- and whatever was set on it after
    loading (a scene's inner spheres, say), with its evaluator `imps` rebuilt for the new type, as
    `load_surface_data(..., spline_type=)` would build it. `sdata` itself is unchanged.

    Called by `viewer.scenes.catalog.respline`.

    In: 
    sdata, a loaded surface
    spline_type, str 
    
    Out: 
    a shallow copy of `sdata`, its arrays shared
    """
    out = copy.copy(sdata)  # SurfaceData: a shallow copy, its own attributes over the same arrays
    _fill_spline_type(out, spline_type)
    out.imps = implicit_surface(out)
    return out


def implicit_surface(sdata):
    """
    The per-surface Warp evaluator context `imps` for a surface's arrays as they stand --
    built by `load_surface_data`, and again by `scale_surface` after it has rescaled them.

    Called by `load_surface_data`, `with_spline_type` and `scale_surface`.

    In: 
    sdata, reads p_min, p_max, h (3,) float, N (3,) int, r, spline_type, sdf_coeffs (Nx+3, Ny+3, Nz+3) float32
    and vsdf_values (Nx, Ny, Nz) float32 
    
    Out: 
    `implicit.ImplicitSurface`, with its own device copies:
    sdf_coeffs, sdf_knots (Nx+1, Ny+1, Nz+1) and vsdf_values, float32
    """
    voxel_grid_params = implicit.GridParams()  # implicit.GridParams
    voxel_grid_params.p_min = wp.vec3(np.asarray(sdata.p_min, dtype=np.float32))
    voxel_grid_params.p_max = wp.vec3(np.asarray(sdata.p_max, dtype=np.float32))
    voxel_grid_params.N = wp.vec3i(np.asarray(sdata.N))
    voxel_grid_params.h = wp.vec3(np.asarray(sdata.h, dtype=np.float32))
    voxel_grid_params.r = wp.float32(float(sdata.r))

    tmp = np.ascontiguousarray(sdata.sdf_coeffs)  # (Nx+3, Ny+3, Nz+3) float32: the tricubic coefficients
    imps = implicit.ImplicitSurface()  # implicit.ImplicitSurface
    imps.grid_params = voxel_grid_params
    imps.spline = wp.int32(spline_kind(sdata))
    imps.sdf_coeffs = wp.array(tmp)
    # Both representations, unlike the packed tables in `make_collision_context`,
    # which carry only the live one: this is the per-surface diagnostic path, one
    # surface at a time rather than a whole scene, and having both here is what
    # makes comparing them a matter of two evaluations.
    imps.sdf_knots = wp.array(
        np.ascontiguousarray(spline_c0.knots_from_c_3d(tmp), dtype=np.float32)
    )
    imps.vsdf_values = wp.array(np.ascontiguousarray(sdata.vsdf_values))
    return imps


def load_surface_file(path, spline_type=None):
    """
    A baked surface from its `surface_data.npz` (`load_surface_data`), with an oriented
    bounding box synthesized where the bake ships none, and without the inner spheres a
    bake may carry. A scene's shipped surfaces come through here by name
    (`viewer.scenes.assets.load_surface`).

    Corners are ordered i = 4*xbit + 2*ybit + zbit, as `types.obb_frame_np` reads them.

    `spline_type` picks the SDF representation -- "cubic" or "linear" -- and is passed to
    the loader rather than set afterwards, since the per-surface evaluator is built from it.

    Called by `viewer.scenes.assets.load_surface`, `viewer.scenes.shadow.surfaces`,
    `viewer.scenes.shadow_kong.surfaces` and 2 others.

    In: 
    path, str or Path
    spline_type, str or None
    
    Out: 
    `SurfaceData`, with bbox (8, 3) float64, sph_c (0, 3)
    and sph_r (0,)
    """
    surf = load_surface_data(str(path), spline_type=spline_type)  # SurfaceData

    if not hasattr(surf, "bbox"):
        lo, hi = np.asarray(surf.v).min(0), np.asarray(surf.v).max(0)  # (3,) float64 each: the vertices' box
        corners = [  # 8 [x, y, z] lists
            [hi[0] if i & 4 else lo[0], hi[1] if i & 2 else lo[1], hi[2] if i & 1 else lo[2]]
            for i in range(8)
        ]
        surf.bbox = np.array(corners, dtype=np.float64)

    # Whatever `spheres.npz` happens to sit beside the bake is dropped. A scene built
    # from this loader is a controlled one, and every timing recorded with it is meant
    # to be comparable across bakes. Callers that want spheres pass them explicitly.
    surf.sph_c = np.zeros((0, 3), dtype=np.float64)
    surf.sph_r = np.zeros(0, dtype=np.float64)

    return surf


def scale_surface(surf, scale, shift=(0.0, 0.0, 0.0)):
    """
    A loaded surface resized by `scale` and moved by `shift` (applied after the scaling,
    in its body frame), in place; returns it.

    Everything in a bake is a length or a field of lengths, so a uniform scaling touches
    all of it: the vertices, edges and box, the grid's corners, voxel size and
    circumsphere, the spline coefficients and the vertex-SDF values -- a signed distance
    is a length. `N` is a count and stays. A shift moves only what carries an origin.
    Inner spheres are lengths too: left unscaled, they would no longer be inside the
    scaled surface -- a shrunken one could sit inside spheres larger than itself, and
    the first contact would throw it.
    `imps`, the per-surface evaluator, is rebuilt from the scaled arrays so
    that evaluating the surface directly sees the field the simulation will.

    Called by `viewer.scenes.sharpa_armadillo.armadillo_surface` and `notebooks/05_make_a_surface.ipynb`.

    In: 
    surf, a loaded surface
    scale, float 
    shift, (3,) float 
    
    Writes: 
    surf.v (n, 3), 
    sph_c (nS, 3), 
    sph_r (nS,) 
    bbox (8, 3) 
    e_l (ne,) float64 
    p_min, p_max and h (3,) float32
    r float64
    sdf_coeffs and vsdf_values float32 
    imps 
    
    Out: 
    surf
    """
    s = float(scale)  # float
    shift = np.asarray(shift, dtype=np.float64).reshape(3)  # (3,) float64
    surf.v = np.asarray(surf.v, dtype=np.float64) * s + shift
    if len(getattr(surf, "sph_r", ())):
        surf.sph_c = np.asarray(surf.sph_c, dtype=np.float64) * s + shift
        surf.sph_r = np.asarray(surf.sph_r, dtype=np.float64) * s
    surf.bbox = np.asarray(surf.bbox, dtype=np.float64) * s + shift
    surf.e_l = np.asarray(surf.e_l, dtype=np.float64) * s
    surf.p_min = (np.asarray(surf.p_min, dtype=np.float64) * s + shift).astype(np.float32)
    surf.p_max = (np.asarray(surf.p_max, dtype=np.float64) * s + shift).astype(np.float32)
    surf.h = (np.asarray(surf.h, dtype=np.float64) * s).astype(np.float32)
    surf.r = np.float64(float(surf.r) * s)
    surf.sdf_coeffs = np.ascontiguousarray(np.asarray(surf.sdf_coeffs, dtype=np.float32) * np.float32(s))
    surf.vsdf_values = np.ascontiguousarray(np.asarray(surf.vsdf_values, dtype=np.float32) * np.float32(s))
    surf.imps = implicit_surface(surf)
    return surf


def write_surface_obj(surf, path):
    """
    The collision surface as a Wavefront OBJ, for MuJoCo to take inertia from.

    Called by `viewer.scenes.assets.surface_obj`.

    In: 
    surf, reads v (n, 3) float and f (nf, 3) int 
    path, str or Path, the file, its directories made.

    Out: 
    path, as a Path
    """
    v, f = np.asarray(surf.v), np.asarray(surf.f)  # (n, 3) float, (nf, 3) int
    path = Path(path)  # Path
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:  # fh: the file, open for writing
        fh.write("# lodos collision surface\n")
        for p in v:  # p: (3,) float, a vertex
            fh.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        for t in f:  # t: (3,) int, a face's vertices, from 0
            fh.write(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")
    return path


# The collision pipeline. `types` takes `PlaneSurface`, `spline_kind` and `_is_plane` from above,
# so it is imported last.
from . import broadphase, collide, narrowphase, types  # noqa: E402


def __getattr__(name):
    """
    The package's lazy submodules, `io` and `serialize`, imported on first access (PEP 562).

    Called by Python for an attribute of `lodos.col.sdf` it does not find otherwise.

    In: 
    name, str 
    
    Out: 
    the submodule; raises AttributeError for any other name.
    """
    # `io` and `serialize` make new surfaces and need the `mesh` extra (torch, trimesh), so
    # they load on first use rather than with every simulation.
    if name in ("io", "serialize"):
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
