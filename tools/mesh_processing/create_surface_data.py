#!/usr/bin/env python3
"""
Step 5 -- bake each body into the surface data lodos loads, one `.npz` a body.

    python create_surface_data.py path/to/robot_wt_cp_fused_im_man

        ->  path/to/robot_wt_cp_fused_im_man_sdata/

Step 5 of five. Each step appends its own suffix to whatever its input carried, so a directory's
name is the record of how it was made:

    robot.xml -> _wt -> _wt_cp_fused -> _wt_cp_fused_im -> ..._im_man -> ..._man_sdata

`mjcf.model_stem` strips the lot to recover `robot`, which is how each step finds
the MJCF its world-frame poses need. No step needs an output argument;
`run_pipeline.py` runs the whole chain from the XML.

This is the bake `notebooks/05_make_a_surface.ipynb` walks through, run as a batch, with
the notebook's constants turned into arguments. Per body:

**Reorder.** `sdf_io.process_mesh` quantizes the vertices, sorts them along a
space-filling curve and renumbers the edges and faces to match. The geometry is
unchanged, its vertices only permuted, and their order makes the broadphase's
gathers local, since neighbouring indices then hold neighbouring points. `--encoding` picks Hilbert
(the default, and the better locality) or Morton.

**Grid.** The mesh AABB dilated by `--dilation` about its own centre, `--voxels`
cubic voxels along the longest edge. The dilation is what gives the field somewhere
to be positive: a grid clipped to the mesh's own AABB has no exterior, so a vertex
approaching from outside leaves the grid before the SDF ever reads greater than
zero, and the spline is extrapolating exactly where the narrowphase needs it.

`N` is computed here and rounded up rather than handing a scalar `h` to
`create_voxel_grid_from_verts`, whose per-axis `ceil` sits on a rounding boundary
when the longest edge divides evenly -- which at 100 voxels it does by construction.
Rounding up also means the grid contains the dilated box rather than landing within
half a voxel of it; cubic voxels and an integer count cannot honour the dilation
exactly on the short axes, and erring outwards is the harmless direction.

**Bake.** Signed distance at every knot, fitted with a tricubic B-spline, plus the
distance at each voxel centre for the voxel prefilter. The sign comes from
`wp.mesh_query_point`, which needs a closed mesh; bodies that are not are reported
rather than refused, since a geometry that only ever supplies vertices to an
asymmetric pair never has its own field sampled.

The viewer then shows one body at a time, in its own body frame, chosen by the
radio column. Six toggles over it, each answering a question the table cannot:

    wire     the mesh's own edges, for seeing what is inside one without making it
             see-through at all.
    verts    the vertices as points, joined in the order the curve put them.
    edges    the same for the edge midpoints, in the order the edges are stored:
             `process_mesh` renumbers them to the new vertices but leaves them in
             the order `sdf_io.load_mesh` read them (trimesh's `edges_unique`), so
             their walk is not the curve's. The vertices' walk is what makes a bad
             `--depth` visible: a truncated code shows up as long jumps across the
             body instead of a tight walk.
    grid     the voxel grid's knots along the box, against the mesh inside it.
    iso      the zero level set of the *baked spline*, marching-cubed. Not the
             input mesh: the difference between the two is the bake's error.
    normals  `grad(sdf)` at each vertex, which is the direction a contact pushes.

The `transparency` slider is real alpha, not a fade: importing
`viewer.draw` patches Newton's shape and shadow shaders to dither a
fragment out by an alpha smuggled through the material's free `z` component. On a
Newton whose shaders have moved on, the patch declines to apply, the slider says
`mesh fade (no alpha)` and blends the colour towards the background instead --
which de-emphasizes a solid without revealing anything behind it.

This step imports lodos -- `lodos.col.sdf.io` and `lodos.col.sdf` -- and the
viewer's `viewer.draw`, so it is the one step that is not self-contained: it bakes
with lodos's own spline fit and curve encoders, so that what it writes agrees
exactly with what lodos reads.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import trimesh
import warp as wp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from lodos.col.sdf import io as sdf_io  # noqa: E402
from lodos.col import sdf  # noqa: E402
# Importing this patches Newton's shape and shadow shaders to carry an alpha, which
# has to happen before any `ViewerGL` is built because the program is compiled there.
from viewer.draw import TRANSPARENCY  # noqa: E402
from mjcf import PIPELINE_SUFFIXES, step_dir  # noqa: E402
from mesh_viewer import Body, Lines, MeshViewer, Points, body_color  # noqa: E402

ORDER_V_RGB = (0.10, 0.85, 0.35)
ORDER_E_RGB = (0.95, 0.60, 0.10)
GRID_RGB = (0.35, 0.75, 1.00)
NORMAL_RGB = (0.40, 0.90, 0.75)


def bbox_corners(v):
    """The AABB's 8 corners, ordered i = 4*xbit + 2*ybit + zbit, as `lodos.col.sdf.types.obb_frame_np` reads them."""
    lo, hi = np.asarray(v).min(0), np.asarray(v).max(0)
    return np.array(
        [[hi[0] if i & 4 else lo[0],
          hi[1] if i & 2 else lo[1],
          hi[2] if i & 1 else lo[2]] for i in range(8)],
        dtype=np.float64,
    )


def grid_for(v, voxels, dilation, d_max):
    """`(N, grid_aabb)` for cubic voxels, `voxels` of them along the longest grid edge."""
    lo, hi = np.asarray(v).min(0), np.asarray(v).max(0)
    center = (lo + hi) / 2
    extent = (hi - lo) * dilation + 2.0 * d_max
    h = extent.max() / voxels
    # The tolerance keeps the longest axis at exactly `voxels`, where the quotient is
    # `voxels` up to float rounding and a bare ceil would sometimes return one more.
    N = np.maximum(1, np.ceil(extent / h - 1e-9)).astype(int)
    half = N * h / 2
    return N, np.stack([center - half, center + half])


def bake(path, args):
    """One body: `(data, imps)`, where `data` is exactly what is written to the npz."""
    v, e, f = sdf_io.load_mesh(path)
    # `quantize_v` recomputes what `process_mesh` would print; called here so the
    # recommended depth can be compared against `--depth` rather than only logged.
    with contextlib.redirect_stdout(io.StringIO()):
        _, recommended = sdf_io.quantize_v(v, args.eps_divisor)
        v, e, f, e_l = sdf_io.process_mesh(
            v, e, f, eps_divisor=args.eps_divisor, depth=args.depth,
            encoding=args.encoding,
        )

    N, grid_aabb = grid_for(v, args.voxels, args.dilation, args.d_max)
    params, knots, centers, grid_vis = sdf.implicit.create_voxel_grid_from_verts(
        v=v, N=[int(n) for n in N], grid_aabb=grid_aabb, verbose=False
    )
    imps = sdf.implicit.create_implicit_surface_from_mesh(v, f, params, knots, centers)

    data = dict(
        v=v, e=e, f=f, e_l=e_l,
        p_min=np.array(params.p_min), p_max=np.array(params.p_max),
        N=np.array(params.N), h=np.array(params.h), r=float(params.r),
        sdf_coeffs=imps.sdf_coeffs.numpy(), vsdf_values=imps.vsdf_values.numpy(),
        bbox=bbox_corners(v),
    )
    return data, imps, grid_vis, int(recommended)


def grid_outline(p_min, p_max, N):
    """
    The knots along six edges of the grid box -- what `create_voxel_grid_from_verts`
    returns for drawing, rebuilt here from a bake that is already on disk.
    """
    axes = []
    for a in range(3):
        knots = np.linspace(p_min[a], p_max[a], int(N[a]) + 1)
        for corner in (p_min, p_max):
            pts = np.tile(corner, (len(knots), 1))
            pts[:, a] = knots
            axes.append(pts)
    return np.concatenate(axes, axis=0)


def reuse(path, out):
    """A bake already written, as `bake` would have returned it. None if absent."""
    npz = out / f"{path.stem}.npz"
    if not npz.exists():
        return None
    with np.load(npz) as z:
        data = {k: z[k] for k in z.files}
    sdata = sdf.load_surface_data(str(npz))
    return data, sdata.imps, grid_outline(data["p_min"], data["p_max"], data["N"]), 0


def isosurface(imps, args):
    """The baked spline's zero level set, or None where it does not cross zero."""
    grid = sdf.implicit.create_pv_grid(
        imps, grid_upsample=args.iso_upsample, shrinkage_factor=args.iso_shrink
    )
    values = sdf.implicit.compute_sdf_batch(
        np.array(grid.points), imps, batch_size=args.batch
    )
    surface = grid.contour([0.0], values, method="marching_cubes")
    if surface.n_points == 0 or surface.n_faces_strict == 0:
        return None
    # pyvista packs faces as [n, i, j, k]; `contour` only ever emits triangles.
    faces = np.asarray(surface.faces).reshape(-1, 4)[:, 1:]
    # Reversed, because VTK winds a contour so its normals point along *decreasing*
    # scalar and this scalar is a signed distance, which increases outwards. Left as
    # it comes, the surface is inside-out: lit from within, backfaces to the camera,
    # and a negative enclosed volume.
    return trimesh.Trimesh(np.asarray(surface.points), faces[:, ::-1], process=False)


class SurfaceViewer(MeshViewer):
    """One body at a time, in its own frame, with the camera following the choice."""

    def __init__(self, bodies, out, args):
        super().__init__(
            bodies, ["mesh", "wire", "verts", "edges", "grid", "iso", "normals"],
            title="Surface data",
            subtitle=f"{len(bodies)} bodies from {args.input.name}"
                     f"   {args.encoding}, depth {args.depth}, {args.voxels} voxels",
            default_on=["mesh"], wire=["wire"], line_width=args.line_width,
            exclusive=True, transparency=TRANSPARENCY,
        )
        self.out = out
        self._framed = ...          # not None, which is a valid focus

    def on_frame(self):
        if self.focused is None and self._framed is ...:
            self.focus(self.bodies[0].name)
        if self.focused != self._framed:
            self._framed = self.focused
            chosen = [b for b in self.bodies if b.name == self.focused]
            self.frame_camera(chosen or None)

    def extra_header(self, imgui):
        imgui.text(f"{sum(len(b.variants['mesh'].vertices) for b in self.bodies):,d}"
                   f" contact vertices   ->  {self.out.name}/")

        imgui.set_next_item_width(200)
        label = "transparency" if self.alpha_ok else "mesh fade (no alpha)"
        changed, self.fade = imgui.slider_float(label, self.fade, 0.0, 1.0, "%.2f")
        redraw = changed

        # Pixels, because that is what Newton's line shader takes; see `line_px`.
        # No redraw on change -- it is a uniform, so it takes effect on the next frame.
        imgui.set_next_item_width(200)
        _, self.line_px = imgui.slider_float("line width [px]", self.line_px, 0.5, 16.0, "%.1f")
        if redraw:
            self.redraw_all()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="a directory of per-body OBJs")
    ap.add_argument("--voxels", type=int, default=100,
                    help="voxels along the longest grid edge")
    ap.add_argument("--dilation", type=float, default=2.0, help="grid AABB / mesh AABB")
    ap.add_argument("--d-max", type=float, default=0.0,
                    help="extra margin on every side [m], beyond the dilation; a surface in a `toa` pair "
                         "wants more than the largest speed x timestep it meets")
    ap.add_argument("--encoding", choices=("hilbert", "morton"), default="hilbert")
    ap.add_argument("--depth", type=int, default=16, help="curve encoding depth")
    ap.add_argument("--eps-divisor", type=int, default=50,
                    help="quantization step, as a fraction of the closest vertex pair")
    ap.add_argument("--iso-upsample", type=float, default=1.0,
                    help="isosurface sampling, relative to the voxel grid")
    ap.add_argument("--iso-shrink", type=float, default=1.0,
                    help="fraction of the grid the isosurface is sampled over")
    ap.add_argument("--normal-scale", type=float, default=0.5,
                    help="drawn normal length, in mean edge lengths")
    ap.add_argument("--point-scale", type=float, default=0.08,
                    help="drawn vertex radius, in mean edge lengths")
    ap.add_argument("--batch", type=float, default=1e5, help="points per SDF batch")
    ap.add_argument("--line-width", type=float, default=2.0e-4)
    ap.add_argument("--rebake", action="store_true",
                    help="re-run the bake even where an .npz is already written")
    ap.add_argument("--no-view", dest="view", action="store_false")
    args = ap.parse_args()

    paths = sorted(args.input.glob("*.obj"))
    if not paths:
        raise SystemExit(f"no .obj files in {args.input}")
    out = step_dir(args.input, "_sdata")
    out.mkdir(parents=True, exist_ok=True)

    # A body takes seconds to fit, so the row is printed in two halves -- the name
    # before the work, the numbers after -- and the stream is line-buffered so the
    # first half arrives even when stdout is a pipe rather than a terminal.
    sys.stdout.reconfigure(line_buffering=True)
    wp.init()          # so its banner lands above the table rather than inside it
    print(f"{args.input.name}: {len(paths)} bodies -> {out.name}/")
    print(f"{'':7s} {'body':24s} {'V':>5s} {'E':>5s} {'grid':>14s} {'h [mm]':>7s} "
          f"{'|sdf(v)|/h':>10s} {'MB':>6s}  wt")
    print("-" * 90)

    rows, bodies, open_mesh, shallow, reused = [], [], [], [], []
    for i, path in enumerate(paths):
        print(f"[{i + 1:2d}/{len(paths)}] {path.stem:24s} ", end="", flush=True)
        watertight = trimesh.load_mesh(path, process=False).is_watertight
        if not watertight:
            open_mesh.append(path.stem)

        # Re-baking is the whole cost of this script -- a tricubic fit over (N+3)^3
        # coefficients per body -- and it is pure waste when the answer is already
        # beside us. `load_surface_data` rebuilds the spline the viewer samples, so a
        # second run opens its window in seconds rather than minutes.
        done = None if args.rebake else reuse(path, out)
        if done is not None:
            data, imps, grid_vis, recommended = done
            reused.append(path.stem)
        else:
            data, imps, grid_vis, recommended = bake(path, args)
            if recommended > args.depth:
                shallow.append((path.stem, recommended))
            np.savez(out / f"{path.stem}.npz", **data)

        v, N, h = data["v"], data["N"], data["h"]
        # The spline should read ~0 on the surface it was fitted to, so this is the
        # bake's error, reported against the scale its O(h^2) interpolation lives on.
        err = float(np.abs(sdf.implicit.compute_sdf_batch(
            v, imps, batch_size=args.batch)).mean() / h.max())
        nbytes = 4 * (int(np.prod(N)) + int(np.prod(N + 3)))

        rows.append(dict(
            body=path.stem, v=int(len(v)), e=int(len(data["e"])),
            N=[int(n) for n in N], h_mm=round(float(h.max()) * 1e3, 4),
            sdf_error_h=round(err, 4), bytes=nbytes,
            watertight=bool(watertight), recommended_depth=recommended,
            # what the bake was asked for, so that the record says how the surface was made
            settings=dict(voxels=args.voxels, dilation=args.dilation, d_max=args.d_max, encoding=args.encoding,
                          depth=args.depth, eps_divisor=args.eps_divisor, reused=done is not None),
        ))
        if args.view:
            mesh = trimesh.Trimesh(v, data["f"], process=False)
            mid = v[data["e"]].mean(axis=1)
            scale = args.normal_scale * float(np.mean(data["e_l"]))
            dot = args.point_scale * float(np.mean(data["e_l"]))
            n = sdf.implicit.compute_n_batch(v, imps, batch_size=args.batch)
            bodies.append(Body(
                path.stem,
                {
                    "mesh": mesh,
                    "wire": mesh,
                    # Points and the walk between them, as the notebook draws it:
                    # the points say where the samples are, the line says in what order.
                    "verts": [Points(v, dot, ORDER_V_RGB),
                              Lines(v[:-1], v[1:], ORDER_V_RGB)],
                    "edges": [Points(mid, dot, ORDER_E_RGB),
                              Lines(mid[:-1], mid[1:], ORDER_E_RGB)],
                    "grid": Points(grid_vis, 0.35 * float(h.max()), GRID_RGB),
                    "iso": isosurface(imps, args),
                    "normals": Lines(v, v + n * scale, NORMAL_RGB),
                },
                body_color(i), None,
                info=f"{len(v):5d} V  {'x'.join(str(int(x)) for x in N):>12s}"
                     f"  {h.max() * 1e3:5.3f} mm  err {err:.3f} h",
            ))

        print(f"{len(v):5d} {len(data['e']):5d} "
              f"{'x'.join(str(int(n)) for n in N):>14s} {h.max() * 1e3:7.3f} "
              f"{err:10.4f} {nbytes / 1e6:6.2f}  {'y' if watertight else 'n'}"
              f"{'  reused' if done is not None else ''}")

    (out / "create_surface_data.json").write_text(json.dumps(rows, indent=2))
    copied = gather_records(args.input, out)
    print("-" * 90)
    print(f"{len(rows)} surfaces -> {out}"
          + (f"   ({len(reused)} reused, --rebake to redo)" if reused else ""))
    print(f"packed voxel tables: {sum(np.prod(r['N']) for r in rows) / 1e6:.2f} M voxels, "
          f"{sum(r['bytes'] for r in rows) / 1e6:.1f} MB")
    # A reused bake reports no recommended depth, so the warning would be wrong.
    if shallow and not reused:
        print(f"\n--depth {args.depth} is below what these need, so their curve codes "
              f"are truncated and the reordering is worse than it could be:")
        print("  " + ", ".join(f"{n} ({d})" for n, d in shallow))
    if open_mesh:
        print(f"\nnot watertight ({len(open_mesh)}): {', '.join(open_mesh)}")
        print("  Their baked sign is unreliable near the offending edges. Harmless")
        print("  while they only ever supply vertices to an asymmetric pair.")

    if copied:
        print(f"the earlier steps' records, beside the bakes: {', '.join(copied)}")
    if args.view:
        SurfaceViewer(bodies, out, args).run()


# Each step's record, by the directory it is written to: a list of rows, one per body.
RECORDS = {"_wt": "fuse_meshes_by_body.json", "_cp": "calibrate_cp_decomp.json", "_im": "interactive_remesh.json",
           "_man": "fix_nonmanifold_meshes.json"}


def gather_records(src, out):
    """
    Copies the earlier steps' records into `out`, beside the bakes: the threshold and
    hull cap each body was decomposed at, its vertex target, its repair, how it was made
    watertight. A surface then carries how it was made wherever it is installed, and
    `lodos.utils.tuning.surface_fingerprint` reads each body's rows from them -- the vertex
    budget being the largest single lever on what a scene costs. The directories are
    found by stripping the suffixes off the input's name one at a time.
    """
    copied = []
    name, parent = Path(src).name, Path(src).parent
    while True:
        for suffix, record in RECORDS.items():
            d = parent / name
            if name.endswith(suffix) and (d / record).exists() and record not in copied:
                shutil.copy2(d / record, out / record)
                copied.append(record)
            # step 2 writes its record into `_cp/`, beside the `_cp_fused/` the chain continues from
            if suffix == "_cp" and name.endswith("_cp_fused"):
                cp = parent / name[: -len("_fused")]
                if (cp / record).exists() and record not in copied:
                    shutil.copy2(cp / record, out / record)
                    copied.append(record)
        stripped = next((name[: -len(sfx)] for sfx in PIPELINE_SUFFIXES if name.endswith(sfx)), None)
        if stripped is None:
            return copied
        name = stripped


if __name__ == "__main__":
    main()
