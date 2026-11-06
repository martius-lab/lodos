#!/usr/bin/env python3
"""
Step 6 -- an inner-sphere packing for one baked surface.

    python pack_spheres.py path/to/surface_data.npz            # calibrate, then Export
    python pack_spheres.py path/to/body_dir --max-spheres 200
    python pack_spheres.py path/to/surface_data.npz --write --no-view

`lodos.col.sdf` collides a geometry's vertices against the other's SDF, which
catches an overlap only once a vertex has passed through the surface. Spheres
strictly *inside* a body are the guard against the deep case: they reach the
other surface while the mesh is still clear of it, and one field evaluation gives
their exact gap, since a sphere of radius `r` about `c` is the r-offset of `c`.
This writes the `spheres.npz` sidecar that `lodos.col.sdf.load_surface_data` reads.

**The bake already contains the answer.** `vsdf_values` is the signed distance at
every voxel centre, so a voxel with `sdf < 0` is a point inside the body and
`|sdf|` is the radius of the largest sphere that fits there -- the medial-axis
sphere at that point, with no packing search of its own. All that is left is
choosing which of them to keep.

**Greedy on added volume.** Spheres are added one at a time, each time the one
covering the most interior voxels that no kept sphere covers yet. Union volume is
*submodular* -- a candidate's gain can only fall as others are added -- which is
what makes the greedy tractable: a stale gain is an upper bound, so a priority
queue that re-checks the top entry before accepting it gives the exact greedy
answer while recomputing a small fraction of the candidates. Without that, one
pick over a 100^3 grid is a few hundred thousand ball counts.

**Radii are shrunk by one voxel radius.** `vsdf` is sampled at voxel centres, so
the true surface can be nearer than the sample says; the SDF is 1-Lipschitz and
the sample is within the voxel's circumsphere radius `r` of the truth, so
`|sdf| - r` is provably inside. Tangent spheres would cover more and would
sometimes poke out, which defeats the point of packing them.

**Volume is counted on the bake's own grid.** A gain is a count of interior
voxels, and the coverage reported is those covered over those that exist. The
mesh's true volume is printed beside it so the discretization is visible rather
than assumed -- on a coarse grid against a thin feature the two differ.

**The window is the calibration**, like steps 2 and 3. The greedy produces an
*ordered* list, so its first `k` is the answer for a budget of `k` and the slider
costs nothing to drag: coverage per `k` is computed once, up front. `Export` is
the only thing that writes, and closing the window writes nothing.
"""

from __future__ import annotations

import argparse
import heapq
import os
import sys
import time

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from mesh_viewer import Body, MeshViewer, Points, body_color  # noqa: E402


#######################################################################################
# The packing
#######################################################################################


def ball(centre_idx, radius, h, shape):
    """`(slices, mask)` for the voxels whose centres lie within `radius`.

    In metres, not voxels: `h` is per-axis and these grids are not isotropic, so a
    ball is an ellipsoid in index space and treating it as a cube of voxels would
    over-cover the fine axes.
    """
    d = np.ceil(radius / h).astype(int)
    lo = np.maximum(centre_idx - d, 0)
    hi = np.minimum(centre_idx + d + 1, shape)
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))

    grids = np.ogrid[tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))]
    q = sum(((g - c) * s) ** 2 for g, c, s in zip(grids, centre_idx, h))
    return sl, q <= radius * radius


def pack_spheres(sdf, h, r_vox, max_spheres, stride=1, verbose=True):
    """The greedy order, and the coverage each prefix reaches.

    Returns `(centres_idx, radii, covered_counts, n_inside, ceiling)`;
    `covered_counts[k]` is how many interior voxels the first `k + 1` spheres
    cover between them, and `ceiling` is what *every* candidate together covers.

    The ceiling is the number to read first during calibration. It is well under
    the whole body and no budget can beat it, because shrinking each radius by a
    voxel leaves a shell the packing cannot enter: on the `sphere` asset one
    sphere reaches 76% and a hundred reach 82%, and the missing 18% is that shell
    rather than anything the greedy failed to find. Without it a low coverage
    reads as "raise --max-spheres" when the answer is "the grid is too coarse".
    """
    shape = np.array(sdf.shape)
    inside = sdf < 0.0
    n_inside = int(inside.sum())
    assert n_inside, "no voxel of this bake is inside the surface"

    radius = -sdf - r_vox
    cand = inside & (radius > 0.0)
    if stride > 1:
        keep = np.zeros_like(cand)
        keep[::stride, ::stride, ::stride] = True
        cand &= keep
    idx = np.argwhere(cand)
    rad = radius[cand]
    assert len(idx), (
        "every interior voxel is within one voxel radius of the surface; the grid "
        "is too coarse for this body to carry a sphere at all"
    )
    if verbose:
        print(f"  {n_inside} interior voxels, {len(idx)} sphere candidates, "
              f"largest radius {rad.max():.5f} m")

    # An upper bound on a candidate's gain, needed before any true gain is known.
    # Every counted voxel centre owns a disjoint cell contained in the ball of
    # radius `R + r_vox`, so that ball's volume over one cell's bounds the count.
    vox = float(np.prod(h))
    ub = (4.0 / 3.0) * np.pi * (rad + r_vox) ** 3 / vox

    heap = [(-float(u), int(i)) for i, u in enumerate(ub)]
    heapq.heapify(heap)

    covered = np.zeros_like(inside)
    order, radii, counts = [], [], []
    n_cov, n_eval = 0, 0
    t0 = time.perf_counter()

    while heap and len(order) < max_spheres:
        # Lazy: the top entry's stored gain may be stale, but it is an upper
        # bound, so recomputing it and re-inserting is enough to find the true
        # maximum without touching the rest.
        neg, i = heapq.heappop(heap)
        sl, mask = ball(idx[i], rad[i], h, shape)
        gain = int(np.count_nonzero(inside[sl] & ~covered[sl] & mask))
        n_eval += 1
        if gain <= 0:
            continue
        if heap and gain < -heap[0][0]:
            heapq.heappush(heap, (-float(gain), i))
            continue

        covered[sl] |= mask
        n_cov += gain
        order.append(idx[i])
        radii.append(float(rad[i]))
        counts.append(n_cov)

    if verbose:
        print(f"  {len(order)} spheres in {time.perf_counter() - t0:.1f} s "
              f"({n_eval} gain evaluations, {n_eval / max(len(order), 1):.1f} per sphere)")

    # What every candidate together reaches -- the bound no budget can pass.
    reach = np.zeros_like(inside)
    for c, R in zip(idx, rad):
        sl, mask = ball(c, R, h, shape)
        reach[sl] |= mask
    ceiling = int(np.count_nonzero(reach & inside))

    return np.asarray(order), np.asarray(radii), np.asarray(counts), n_inside, ceiling


def centres_world(idx, p_min, h):
    """Voxel indices to body-frame centres: the grid samples cell centres."""
    return p_min + (np.asarray(idx, dtype=float) + 0.5) * h


#######################################################################################
# The window
#######################################################################################


class SphereViewer(MeshViewer):
    """`MeshViewer` plus the sphere-count slider and `Export`."""

    def __init__(self, out, mesh, centres, radii, counts, n_inside, ceiling,
                 mesh_volume, vox, k0, **kw):
        self.out = out
        self.centres, self.radii = centres, radii
        self.counts, self.n_inside, self.ceiling = counts, n_inside, ceiling
        self.mesh_volume, self.vox = mesh_volume, vox
        self.k = int(k0)
        self.applied = -1
        self.written = ""
        # Fixed once, not from the shown subset: a point radius derived from the
        # visible spheres would resize every centre as the slider moved, which
        # reads as the packing changing when only the count did.
        self.point_r = 0.25 * float(np.min(radii)) if len(radii) else 1e-3

        # Two bodies rather than two variants of one: the viewer's alpha is per
        # body, and this window shows the spheres *through* the mesh.
        # The surface starts half transparent, since a solid shell hides the
        # thing this window exists to show; the spheres start slightly so their
        # overlaps read as depth rather than as one blob.
        self.obj = Body("surface", {"mesh": mesh}, body_color(0), fade=0.5)
        self.sph = Body("spheres", {}, body_color(4), fade=0.2)
        super().__init__([self.obj, self.sph], ["mesh", "spheres", "centres"],
                         default_on=("mesh", "spheres", "centres"), **kw)
        self._rebuild()

    def coverage(self, k):
        return 0.0 if k == 0 else self.counts[k - 1] / self.n_inside

    def _rebuild(self):
        k = self.k
        # At k = 0 the variants are dropped rather than emptied: `log_points` and
        # `log_mesh` are handed a zero-length array otherwise, and the slider's
        # left end is a position a user reaches by dragging, not an error.
        self.sph.variants = {} if k == 0 else {
            "spheres": [
                trimesh.creation.icosphere(subdivisions=2, radius=r).apply_translation(c)
                for c, r in zip(self.centres[:k], self.radii[:k])
            ],
            "centres": Points(self.centres[:k], self.point_r, color=(1.0, 1.0, 1.0)),
        }
        self.sph.dirty = True
        self.sph.info = (f"{k} spheres, {100 * self.coverage(k):.1f}% of "
                         f"{self.n_inside} interior voxels")
        self.applied = k

    def on_frame(self):
        if self.k != self.applied:
            self._rebuild()

    def extra_header(self, imgui):
        _, self.k = imgui.slider_int("spheres", self.k, 0, len(self.radii))

        # Two sliders, because the picture shows one body *through* another.
        # `self.fade` is the viewer's own and moves everything together; the mesh's
        # is a `Body.fade` offset on top of it, so the shell can be taken away
        # without taking the spheres with it. Both are baked into the material at
        # log time, so a change has to force a redraw.
        redraw = False
        imgui.set_next_item_width(200)
        label = "surface" if self.alpha_ok else "surface fade (no alpha)"
        changed, self.obj.fade = imgui.slider_float(
            label, self.obj.fade, 0.0, 1.0, "%.2f")
        redraw |= changed
        imgui.same_line()
        imgui.set_next_item_width(160)
        changed, self.sph.fade = imgui.slider_float(
            "spheres##fade", self.sph.fade, 0.0, 1.0, "%.2f")
        redraw |= changed

        imgui.set_next_item_width(200)
        changed, self.fade = imgui.slider_float(
            "everything", self.fade, 0.0, 1.0, "%.2f")
        if changed:
            self.redraw_all()
        elif redraw:
            self.obj.dirty = self.sph.dirty = True

        imgui.text(f"coverage {100 * self.coverage(self.k):6.2f}% of the body, "
                   f"{100 * self.counts[self.k - 1] / self.ceiling if self.k else 0.0:6.2f}%"
                   f" of reachable   "
                   f"volume {self.counts[self.k - 1] * self.vox if self.k else 0.0:.6g}"
                   f" / {self.mesh_volume:.6g} m^3 (mesh)")
        if imgui.button("Export"):
            write(self.out, self.centres[:self.k], self.radii[:self.k])
            self.written = f"wrote {self.k} spheres to {self.out}"
        if self.written:
            imgui.same_line()
            imgui.text(self.written)


#######################################################################################
# Entry point
#######################################################################################


def write(out, centres, radii):
    np.savez(out, c=np.asarray(centres, dtype=np.float64),
             r=np.asarray(radii, dtype=np.float64))


def resolve(path):
    """A `surface_data.npz`, given either it or the directory holding it."""
    if os.path.isdir(path):
        path = os.path.join(path, "surface_data.npz")
    assert os.path.exists(path), f"no surface data at {path}"
    return path


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="a surface_data.npz, or the directory holding one")
    ap.add_argument("--max-spheres", dest="max_spheres", type=int, default=128,
                    help="how many the greedy computes; the slider chooses from "
                         "these, so raising it costs time and lowering it costs "
                         "range. Default 128.")
    ap.add_argument("--start", type=int, default=None,
                    help="where the slider starts. Default: all of them")
    ap.add_argument("--stride", type=int, default=1,
                    help="subsample the candidate voxels by this factor per axis. "
                         "The greedy is exact over whatever it is given; this "
                         "trades a little placement freedom for speed on a fine grid")
    ap.add_argument("--write", action="store_true",
                    help="write spheres.npz for --max-spheres without opening a window")
    ap.add_argument("--no-view", action="store_true", dest="no_view")
    args = ap.parse_args()

    path = resolve(args.path)
    z = np.load(path)
    h = np.asarray(z["h"], dtype=float)
    p_min = np.asarray(z["p_min"], dtype=float)
    r_vox = float(z["r"])
    sdf = np.asarray(z["vsdf_values"], dtype=float)
    mesh = trimesh.Trimesh(vertices=np.asarray(z["v"]), faces=np.asarray(z["f"]),
                           process=False)

    print(f"{path}\n  grid {tuple(sdf.shape)}  h {h}  voxel radius {r_vox:.5f} m")
    idx, radii, counts, n_inside, ceiling = pack_spheres(
        sdf, h, r_vox, args.max_spheres, stride=args.stride)
    centres = centres_world(idx, p_min, h)

    vox = float(np.prod(h))
    print(f"  interior volume {n_inside * vox:.6g} m^3 on the grid, "
          f"{mesh.volume:.6g} m^3 from the mesh "
          f"({100 * n_inside * vox / mesh.volume:.1f}%)")
    for k in (1, 5, 10, 25, 50, 100, len(radii)):
        if 0 < k <= len(radii):
            print(f"  {k:4d} spheres -> {100 * counts[k - 1] / n_inside:6.2f}% "
                  f"of the interior volume "
                  f"({100 * counts[k - 1] / ceiling:6.2f}% of what is reachable)")
    print(f"  ceiling      {100 * ceiling / n_inside:6.2f}% -- every candidate "
          f"sphere at once. The rest is the shell within one voxel radius "
          f"({r_vox:.4f} m) of the surface, which the shrink puts out of reach.")

    out = os.path.join(os.path.dirname(path), "spheres.npz")
    if args.write:
        write(out, centres, radii)
        print(f"\nwrote {len(radii)} spheres to {out}")
    if args.no_view:
        return

    # `enable_transparency` patches the shader and must run before any ViewerGL.
    from viewer.draw import enable_transparency  # noqa: PLC0415
    alpha = enable_transparency()

    SphereViewer(
        out, mesh, centres, radii, counts, n_inside, ceiling,
        float(mesh.volume), vox,
        k0=len(radii) if args.start is None else args.start,
        title="Inner spheres",
        subtitle=f"{os.path.basename(os.path.dirname(path))} -- "
                 f"Export writes {os.path.basename(out)}",
        exclusive=False, transparency=alpha,
    ).run()


if __name__ == "__main__":
    main()
