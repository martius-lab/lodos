"""
Isaac Lab's NIST parts as OBJ meshes in each part's body frame: metres, Z up.

Reads a part's USD as Isaac Lab ships it (`Factory/NIST/<part>.usd`, asset release 6.2) and writes,
into `<out>/src/`:

    <part>_collision.obj   the mesh Newton bakes the part's SDF from (the prim with CollisionAPI)
    <part>_visual.obj      what the demo renders
    <part>.json            where they came from: prims, transforms, units, counts, the authored mass

Each mesh is carried through every transform between it and the part's rigid body. The M16 bolt's
collision prim, for one, is scaled 0.95 across its axis: the "loose" pair's clearance, which a
mesh read without its prim's transform would lose. Points are in the stage's units times
`metersPerUnit`. A Y-up stage is turned Z-up, (x, y, z) -> (x, -z, y), and the JSON records that it
was. Faces are triangulated as fans, and wound so that normals point out: a left-handed mesh, or a
transform that mirrors, reverses the winding.

Needs `pxr`, which Lodos's environment lacks; Isaac Lab's has it. With `$PXR_PY` an interpreter that has it:

    $PXR_PY tools/mesh_processing/nist/extract.py DIR/usd/*.usd --out DIR
"""

import argparse
import json
import pathlib
from typing import Any

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics


def body_prim(stage):
    """The part's one rigid body; its frame is the part's frame."""
    bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(bodies) != 1:
        raise ValueError(f"expected one rigid body, found {[str(p.GetPath()) for p in bodies]}")
    return bodies[0]


def mesh_in_body(prim, to_body, scale, y_up):
    """A mesh prim's points and triangles in the body frame, metres, Z up."""
    mesh = UsdGeom.Mesh(prim)
    pts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    # USD multiplies row vectors on the left: p_body = [p 1] (local-to-world) (body-to-world)^-1
    X = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim)) @ to_body
    v = (np.c_[pts, np.ones(len(pts))] @ X)[:, :3] * scale
    if y_up:
        v = np.c_[v[:, 0], -v[:, 2], v[:, 1]]
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    tris = np.array([(idx[s], idx[s + k], idx[s + k + 1]) for s, n in zip(starts, counts) for k in range(1, n - 1)])
    left = mesh.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded
    mirrored = np.linalg.det(X[:3, :3]) < 0.0
    if left != mirrored:
        tris = tris[:, ::-1]
    return v, tris, X


def cube_in_body(prim, to_body, scale, y_up):
    """A collision `Cube` (the board's) as a 12-triangle box in the body frame: its prim's scale is its size."""
    h = 0.5 * float(UsdGeom.Cube(prim).GetSizeAttr().Get())
    pts = np.array([[sx * h, sy * h, sz * h] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    # corner k = 4 i + 2 j + l for signs (i, j, l); each face's two triangles wound outward
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    tris = np.array([t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))])
    X = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim)) @ to_body
    v = (np.c_[pts, np.ones(len(pts))] @ X)[:, :3] * scale
    if y_up:
        v = np.c_[v[:, 0], -v[:, 2], v[:, 1]]
    if np.linalg.det(X[:3, :3]) < 0.0:
        tris = tris[:, ::-1]
    return v, tris, X


def write_obj(path, v, f):
    with open(path, "w") as fh:
        fh.writelines(f"v {x:.9g} {y:.9g} {z:.9g}\n" for x, y, z in v)
        fh.writelines(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in f)


def extract(usd_path, out_dir):
    stage = Usd.Stage.Open(str(usd_path))
    name = pathlib.Path(usd_path).stem
    body = body_prim(stage)
    to_body = np.linalg.inv(np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(body)))
    scale = float(UsdGeom.GetStageMetersPerUnit(stage))
    y_up = UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.y
    record: dict[str, Any] = dict(source=str(usd_path), body=str(body.GetPath()), meters_per_unit=scale,
                                  y_up_turned=bool(y_up), authored_mass=UsdPhysics.MassAPI(body).GetMassAttr().Get(),
                                  meshes={})
    groups = {"collision": [], "visual": []}
    for p in Usd.PrimRange(body):
        if p.IsA(UsdGeom.Cube) and p.HasAPI(UsdPhysics.CollisionAPI):  # the board collides as one box
            groups["collision"].append((p, *cube_in_body(p, to_body, scale, y_up)))
            continue
        if not p.IsA(UsdGeom.Mesh):
            continue
        kind = "collision" if p.HasAPI(UsdPhysics.CollisionAPI) else "visual"
        if kind == "visual" and UsdGeom.Imageable(p).ComputePurpose() not in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render):
            continue
        groups[kind].append((p, *mesh_in_body(p, to_body, scale, y_up)))
    # Some parts (nut_m16) ship one mesh, which is both what collides and what is drawn.
    record["visual_is_collision"] = not groups["visual"]
    if not groups["visual"]:
        groups["visual"] = groups["collision"]
    for kind, items in groups.items():
        if not items:
            raise ValueError(f"{name}: no {kind} mesh")
        vs, fs, n = [], [], 0
        for p, v, f, X in items:
            vs.append(v)
            fs.append(f + n)
            n += len(v)
        v, f = np.concatenate(vs), np.concatenate(fs)
        write_obj(out_dir / f"{name}_{kind}.obj", v, f)
        record["meshes"][kind] = dict(
            prims=[str(p.GetPath()) for p, *_ in items],
            approximation=[UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get() if p.HasAPI(UsdPhysics.MeshCollisionAPI)
                           else None for p, *_ in items],
            prim_scale=[np.linalg.norm(X[:3, :3], axis=1).round(6).tolist() for *_, X in items],
            vertices=len(v), triangles=len(f), bbox_min=v.min(0).tolist(), bbox_max=v.max(0).tolist())
    (out_dir / f"{name}.json").write_text(json.dumps(record, indent=1))
    c = record["meshes"]["collision"]
    size = (np.array(c["bbox_max"]) - np.array(c["bbox_min"])) * 1e3
    print(f"{name}: collision {c['vertices']} vertices, {c['triangles']} triangles, {size.round(2).tolist()} mm, "
          f"z {c['bbox_min'][2] * 1e3:.2f} to {c['bbox_max'][2] * 1e3:.2f} mm; scale {c['prim_scale']}; "
          f"visual {record['meshes']['visual']['vertices']} vertices", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("usd", nargs="+")
    ap.add_argument("--out", required=True, help="the NIST asset directory; meshes go to its src/")
    a = ap.parse_args()
    out = pathlib.Path(a.out) / "src"
    out.mkdir(parents=True, exist_ok=True)
    for u in a.usd:
        extract(u, out)
