#!/usr/bin/env python3
"""
Step 1 -- one watertight solid per body, from an MJCF's meshes.

    python fuse_meshes_by_body.py path/to/robot.xml   ->  path/to/robot_wt/
                                                          path/to/robot_wt_world/

Step 1 of five. Each step appends its own suffix to whatever its input carried, so a directory's
name is the record of how it was made:

    robot.xml -> _wt -> _wt_cp_fused -> _wt_cp_fused_im -> ..._im_man -> ..._man_sdata

`mjcf.model_stem` strips the lot to recover `robot`, which is how each step finds
the MJCF its world-frame poses need. There is no output argument anywhere;
`run_pipeline.py` runs the whole chain from the XML.

The input is the model, not a directory: which mesh belongs to which body is stated
in the MJCF and nowhere else, so a folder of OBJs could not be grouped at all.

MuJoCo ships a model the usual way round: the meshes you see are marked
`contype="0" conaffinity="0"` and the physics runs against hand-fitted capsules.
lodos collides the mesh itself, so the visual set is the input here and the
shipped collision primitives are ignored -- `--geoms` changes that.

Two things happen, and the second is the hard one.

**Merge by body.** A geom's mesh lives in its own file frame; MuJoCo's compiler
recenters mesh vertices on their inertial frame and folds the correction into
`geom_pos`/`geom_quat`, so

    v_body = R(geom_quat) @ v_mesh + geom_pos

is exact. Everything downstream happens in body frame, which is the frame a
collision geometry has to be authored in: the simulator supplies the body pose
and nothing else. Merging is concatenation, not a boolean union -- the parts of
one body are separate closed solids that touch, and a union of raw visual meshes
is not even well posed when some of them are open.

**Make it watertight,** which the bake needs: `create_implicit_surface_from_mesh`
signs its field with `wp.mesh_query_point`, and that is undefined on open or
inside-out geometry. Two passes, because they fail in opposite directions.

  The *generalized winding number* says which solid a set of sheets was meant to
  bound. It is what works where the bake's own sign does not, staying meaningful
  on arbitrary triangle soup; Warp has it as
  `mesh_query_point_sign_winding_number`. Contoured at zero it leaves specks --
  measured on a robot hand, 91 to 218 stray components together holding under 0.1%
  of the volume -- so the contour is half a voxel out and a volume filter removes
  the rest.

  *CGAL alpha wrapping* then supplies the topology: one closed manifold shell,
  unconditionally. On its own it cannot be used, because the wrap contains the
  input *triangles* rather than the solid, so an alpha below the size of a hole
  tunnels through and shrink-wraps the sheets into a membrane -- 39.3 cm3 at
  alpha 5.6 mm against 2.5 cm3 at 3.4 mm, on a part whose convex hull is 39.5.
  Handed something already closed it cannot tunnel, which is why it runs second.

  A body that is already a solid skips the winding-number pass: it would put an
  intact mesh through a 0.25 mm grid and a half-voxel dilation for nothing.

CGAL is not a lodos dependency, and nothing under `tools/mesh_processing/` may reach
outside it, so the local install lives here:

    pip install --no-deps --target tools/mesh_processing/.pylibs cgal

`robot_wt/` is in body frame, which is the frame a collision geometry has to be
authored in; `robot_wt_world/` is the same meshes through their body pose at
`qpos0`, for anything that wants the assembled scene. A rigid transform of what is
already on disk, so the two cannot drift.

The viewer then opens on the result -- `input` and `watertight` for every body,
toggled per body and per variant, with focus mode to isolate one. It writes
nothing; close it and the run is done.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import trimesh
import warp as wp
from skimage.measure import marching_cubes

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from mjcf import body_poses, merge_by_body, step_dir  # noqa: E402
from mesh_viewer import Body, MeshViewer, body_color  # noqa: E402


#######################################################################################
# Watertight
#######################################################################################


@wp.kernel
def _sdf_winding(mesh: wp.uint64, lo: wp.vec3, h: float, out: wp.array3d(dtype=float)):
    i, j, k = wp.tid()
    p = lo + wp.vec3(float(i) * h, float(j) * h, float(k) * h)
    q = wp.mesh_query_point_sign_winding_number(mesh, p, 1.0e6, 2.0, 0.5)
    if q.result:
        out[i, j, k] = q.sign * wp.length(p - wp.mesh_eval_position(mesh, q.face, q.u, q.v))


def is_solid(mesh):
    return mesh.is_watertight and all(
        c.is_watertight for c in mesh.split(only_watertight=False)
    )


def winding_solid(mesh, h=2.5e-4, level=0.5, min_vol_frac=1e-3):
    """The solid the sheets bound, as a mesh. See the module docstring."""
    lo = mesh.vertices.min(axis=0) - 4 * h
    n = np.ceil((mesh.vertices.max(axis=0) + 4 * h - lo) / h).astype(int) + 1

    # Named, not inlined: a temporary `wp.Mesh` is collected as soon as `.id` is read,
    # freeing its device memory under a kernel that is still queued.
    wm = wp.Mesh(
        points=wp.array(np.ascontiguousarray(mesh.vertices, np.float32), dtype=wp.vec3),
        indices=wp.array(np.ascontiguousarray(mesh.faces, np.int32).ravel(), dtype=int),
        support_winding_number=True,
    )
    grid = wp.zeros(tuple(int(x) for x in n), dtype=float)
    wp.launch(_sdf_winding, dim=grid.shape,
              inputs=(wm.id, wp.vec3(*lo.astype(float)), float(h), grid))
    wp.synchronize()

    v, f, _, _ = marching_cubes(grid.numpy(), level=level * h, spacing=(h, h, h))
    out = trimesh.Trimesh(v + lo, f, process=False)

    parts = out.split(only_watertight=False)
    if len(parts) > 1:
        total = sum(abs(p.volume) for p in parts)
        keep = [p for p in parts if abs(p.volume) >= min_vol_frac * total]
        out = trimesh.util.concatenate(keep) if len(keep) > 1 else keep[0]
    return out


def _cgal():
    try:
        from CGAL import CGAL_Alpha_wrap_3  # noqa: F401
    except ImportError:
        sys.path.insert(0, str(HERE / ".pylibs"))
    from CGAL import CGAL_Alpha_wrap_3 as aw
    from CGAL.CGAL_Kernel import Point_3
    from CGAL.CGAL_Polyhedron_3 import Polyhedron_3

    return aw, Point_3, Polyhedron_3


def alpha_wrap(mesh, alpha, offset=None, tmp=None):
    """CGAL's alpha wrap of `mesh`. `offset` defaults to alpha/30, CGAL's own ratio."""
    aw, Point_3, Polyhedron_3 = _cgal()
    offset = alpha / 30.0 if offset is None else offset

    pts = aw.Point_3_Vector()
    pts.reserve(len(mesh.vertices))
    for p in mesh.vertices:
        pts.append(Point_3(float(p[0]), float(p[1]), float(p[2])))
    polys = aw.Polygon_Vector()
    polys.reserve(len(mesh.faces))
    for face in mesh.faces:
        iv = aw.Int_Vector()
        iv.reserve(3)
        for i in face:
            iv.append(int(i))
        polys.append(iv)

    out = Polyhedron_3()
    aw.alpha_wrap_3(pts, polys, float(alpha), float(offset), out)

    # The SWIG binding exposes no vertex accessor that survives round-tripping, so the
    # OFF file is the way back out. It is deleted immediately.
    tmp = Path(tmp or HERE / ".tmp" / "alpha_wrap.off")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    out.write_to_file(str(tmp))
    wrapped = trimesh.load_mesh(tmp, process=False)
    tmp.unlink(missing_ok=True)
    return wrapped


def make_watertight(mesh, alpha=5.0e-4, voxel=2.5e-4, level=0.5):
    """`(solid, used_winding)` -- the watertight version of `mesh`."""
    used_winding = not is_solid(mesh)
    solid = winding_solid(mesh, voxel, level) if used_winding else mesh
    return alpha_wrap(solid, alpha), used_winding


#######################################################################################


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="the MJCF to read")
    ap.add_argument("--geoms", choices=("visual", "collision", "all"), default="visual")
    ap.add_argument("--alpha", type=float, default=5.0e-4, help="alpha wrap size [m]")
    ap.add_argument("--voxel", type=float, default=2.5e-4, help="winding-number grid [m]")
    ap.add_argument("--level", type=float, default=0.5, help="contour offset, in voxels")
    ap.add_argument("--no-view", dest="view", action="store_false",
                    help="skip the viewer, which otherwise opens on the result")
    args = ap.parse_args()

    out = step_dir(args.input, "_wt")
    world = out.parent / f"{out.name}_world"
    out.mkdir(parents=True, exist_ok=True)
    world.mkdir(parents=True, exist_ok=True)

    wp.init()
    model = mujoco.MjModel.from_xml_path(str(args.input))
    merged = merge_by_body(model, args.geoms)
    if not merged:
        raise SystemExit(f"no {args.geoms} mesh geoms in {args.input}")
    poses = body_poses(model)
    print(f"{args.input.name}: {len(merged)} bodies with {args.geoms} meshes\n")

    print(f"{'body':<24s} {'V in':>7s} {'shells':>6s} {'solid':>5s} | {'V out':>7s} {'wt':>5s} "
          f"{'genus':>5s} {'vol cm3':>8s} {'wn':>3s}")
    solids, rows = {}, []
    for name, mesh in merged.items():
        solid, used_wn = make_watertight(mesh, args.alpha, args.voxel, args.level)
        solids[name] = solid
        solid.export(out / f"{name}.obj")
        r, t = poses[name]
        posed = solid.copy()
        posed.vertices = posed.vertices @ r.T + t
        posed.export(world / f"{name}.obj")

        shells = len(solid.split(only_watertight=False))
        row = dict(
            body=name, v_in=int(len(mesh.vertices)),
            shells_in=int(len(mesh.split(only_watertight=False))),
            solid_in=bool(is_solid(mesh)), v_out=int(len(solid.vertices)),
            watertight=bool(solid.is_watertight), shells=shells,
            genus=int((2 - solid.euler_number) // 2) if solid.is_watertight and shells == 1 else None,
            volume_cm3=round(float(solid.volume) * 1e6, 3),
            winding_used=bool(used_wn),
        )
        rows.append(row)
        print(f"{name:<24s} {row['v_in']:7d} {row['shells_in']:6d} {str(row['solid_in']):>5s} |"
              f" {row['v_out']:7d} {str(row['watertight']):>5s} {str(row['genus']):>5s}"
              f" {row['volume_cm3']:8.2f} {'Y' if used_wn else '-':>3s}")

    (out / "fuse_meshes_by_body.json").write_text(json.dumps(rows, indent=2))
    print(f"\n{out} and {world.name}/ : {len(rows)} solids, "
          f"{sum(r['v_out'] for r in rows):,d} vertices")
    print(f"  watertight {sum(r['watertight'] for r in rows)}/{len(rows)}"
          f"   single shell {sum(r['shells'] == 1 for r in rows)}/{len(rows)}"
          f"   winding-number pass used on {sum(r['winding_used'] for r in rows)}")

    if args.view:
        bodies = [
            Body(name, {"input": merged[name], "watertight": solids[name]},
                 body_color(i), poses.get(name),
                 info=f"{r['v_in']:6d} -> {r['v_out']:6d} V   {r['volume_cm3']:7.2f} cm3"
                      + ("   wn" if r["winding_used"] else ""))
            for i, (name, r) in enumerate(zip(merged, rows))
        ]
        MeshViewer(bodies, ["input", "watertight"],
                   title="Fuse meshes by body",
                   subtitle=f"{args.input.name}   {len(bodies)} bodies   -> {out.name}/",
                   default_on=["watertight"]).run()


if __name__ == "__main__":
    main()
