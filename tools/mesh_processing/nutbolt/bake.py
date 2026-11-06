"""A Lodos surface: a remesh's vertices (Hilbert-ordered, as the bake step does) with the SDF of the repaired original."""
import sys, contextlib, io, numpy as np, trimesh, warp as wp
from pathlib import Path
from lodos.col.sdf import io as sdf_io
from lodos.col.sdf import implicit
wp.config.quiet = True


def bbox_corners(v):
    lo, hi = np.asarray(v).min(0), np.asarray(v).max(0)
    return np.array([[hi[0] if i & 4 else lo[0], hi[1] if i & 2 else lo[1], hi[2] if i & 1 else lo[2]] for i in range(8)])


def build(remesh_obj, orig_obj, h, margin, out_dir, depth=16, eps_divisor=50):
    m = trimesh.load(remesh_obj, force="mesh", process=True)
    v, e, f = np.asarray(m.vertices), np.asarray(m.edges_unique), np.asarray(m.faces)
    with contextlib.redirect_stdout(io.StringIO()):
        _, rec = sdf_io.quantize_v(v, eps_divisor)
        v, e, f, e_l = sdf_io.process_mesh(v, e, f, eps_divisor=eps_divisor, depth=max(depth, int(rec)), encoding="hilbert")
    o = trimesh.load(orig_obj, force="mesh", process=True)
    vo, fo = np.asarray(o.vertices, np.float64), np.asarray(o.faces, np.int64)
    lo = np.minimum(vo.min(0), v.min(0)) - margin
    hi = np.maximum(vo.max(0), v.max(0)) + margin
    params, knots, centers, _ = implicit.create_voxel_grid_from_verts(v=vo, h=float(h), grid_aabb=np.stack([lo, hi]), verbose=False)
    imps = implicit.create_implicit_surface_from_mesh(vo, fo, params, knots, centers)
    data = dict(v=v, e=e, f=f, e_l=e_l, p_min=np.array(params.p_min), p_max=np.array(params.p_max), N=np.array(params.N),
                h=np.array(params.h), r=float(params.r), sdf_coeffs=imps.sdf_coeffs.numpy(), vsdf_values=imps.vsdf_values.numpy(),
                bbox=bbox_corners(v))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    np.savez(Path(out_dir) / "surface_data.npz", **data)
    sdf_v = implicit.compute_sdf_batch(v, imps, batch_size=1e6)
    print(f"{out_dir}: {len(v)} vertices (depth {max(depth, int(rec))}), grid {np.array(params.N).tolist()} at h {h * 1e3:.2f} mm "
          f"({(data['sdf_coeffs'].nbytes + data['vsdf_values'].nbytes) / 1e6:.0f} MB); baked SDF at the vertices (um): "
          f"median {np.median(np.abs(sdf_v)) * 1e6:.2f} p99 {np.percentile(np.abs(sdf_v), 99) * 1e6:.2f} max {np.abs(sdf_v).max() * 1e6:.1f}", flush=True)


if __name__ == "__main__":
    # Usage: bake.py ASSET_DIR H_MM [--nuts 0.2 0.5 ...] [--no-bolt]
    # Bakes the bolt (from im/bolt_s1.0.obj) and each nut im/nut_s<EDGE>.obj -- by default every
    # one present -- with the SDF of the repaired original, into ASSET_DIR/surfaces/.
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("asset_dir")
    ap.add_argument("h_mm", type=float)
    ap.add_argument("--nuts", nargs="*", default=None, help="edge lengths in mm, as in im/nut_s<EDGE>.obj")
    ap.add_argument("--no-bolt", action="store_true")
    a = ap.parse_args()
    D, h = Path(a.asset_dir), a.h_mm * 1e-3
    if not a.no_bolt:
        build(D / "im/bolt_s1.0.obj", D / "repair/factory_bolt_m20_loose_wt.obj", h, 0.003, D / "surfaces/bolt")
    nuts = a.nuts if a.nuts is not None else sorted(p.stem.removeprefix("nut_s") for p in (D / "im").glob("nut_s*.obj")
                                                    if not p.stem.endswith("_raw"))
    for s in nuts:
        build(D / f"im/nut_s{s}.obj", D / "repair/factory_nut_m20_loose_subdiv_3x_wt.obj", h, 0.003, D / f"surfaces/nut_s{s}")
