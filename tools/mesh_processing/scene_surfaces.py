#!/usr/bin/env python3
"""
The surfaces of the viewer's primitive scenes -- boxes and spheres -- made without a
remesher.

A box's vertices sit on a regular lattice over its faces and a sphere's are an icosphere
subdivided down to the spacing. Either mesh is watertight and outward-wound, as the bake
(`create_surface_data.py`, step 5) needs, and its vertices are evenly spaced, which is
all the remesh steps are for (see the README). So the chain reduces to writing the OBJ
and baking it:

    python tools/mesh_processing/scene_surfaces.py [NAME ...]     # default: every shape

writes `.tmp/scene_surfaces/NAME.obj`, bakes the directory, and installs each result as
`viewer/assets/surfaces/NAME/surface_data.npz`, where `viewer.scenes.assets.load_surface(NAME)` finds it.

The arch's stones (`viewer/assets/arch`) are convex hexahedra of eight vertices; their faces
are subdivided to the spacing instead, which keeps them planar and the mesh watertight.

The `toa` scenes (`viewer.scenes.toa`) add a 2 cm cube, a 1 cm plate, a 10 cm ring with a 1 cm wall and a
1 cm card of 0.6 by 1.2 m (2.4 MB: a card the shipped one's size would be tens of MB at its voxels),
each fine enough to resolve its thin feature (four voxels across it) and with a band of 1-2 cm; the ring is
trimesh's annulus, its vertices on its two rims alone, which as an SDF geometry alone it needs no more of.
Installed files are compressed, as the shipped ones are.

The shipped `cube` surface is its eight corners alone, so two of them touch only corner
to face: a face resting on a face, or a card's edge against another card, needs vertices
along the edges and across the faces, which these have.

Each shape may pass the bake its own grid. A card is 4 cm thick, thinner than a voxel of
the default grid (100 voxels over twice its 2.5 m), where the fitted field would not go
negative inside it: its grid is 1 cm voxels over the card itself plus 10 cm all round.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parents[1] / "viewer" / "assets" / "surfaces"

# name: (kind, size, spacing, bake options) -- ("box", half extents, ...), ("sphere", radius,
# ...) or ("stone", 1-based index, ...) -- in metres.
SHAPES = {
    "cube_0.8": ("box", (0.4, 0.4, 0.4), 0.1, {}),  # `viewer.scenes.drop`'s cubes
    "cube_0.2": ("box", (0.1, 0.1, 0.1), 0.025, {}),  # `viewer.scenes.fbf`'s incline
    "sphere_0.25": ("sphere", 0.25, 0.05, {}),  # its backspin ball
    "box_0.3x0.6x1.2": ("box", (0.15, 0.3, 0.6), 0.05, {}),  # its Painleve box
    "card_0.04x1.25x2.5": ("box", (0.02, 0.625, 1.25), 0.1,  # its cards
                           {"--voxels": "270", "--dilation": "1.0", "--d-max": "0.1"}),
}
SHAPES.update({f"arch25_stone{i:02d}": ("stone", i, 0.75, {}) for i in range(1, 26)})  # its arch
SHAPES.update({  # `viewer.scenes.toa`'s: its small cubes, its hammers' plates, its rings (inner radius, wall, height)
    "cube_0.02": ("box", (0.01, 0.01, 0.01), 0.005, {"--voxels": "32", "--dilation": "1.0", "--d-max": "0.01"}),
    "plate_0.01x0.12x0.06": ("box", (0.005, 0.06, 0.03), 0.005,
                             {"--voxels": "64", "--dilation": "1.0", "--d-max": "0.02"}),
    "ring_0.1x0.01x0.06": ("ring", (0.1, 0.01, 0.06), 0.005,
                           {"--voxels": "104", "--dilation": "1.0", "--d-max": "0.02"}),
    # its bullet's card, 1 cm thick: 2.5 mm voxels, four across it, and a 2 cm band
    "card_0.01x0.6x1.2": ("box", (0.005, 0.3, 0.6), 0.05, {"--voxels": "496", "--dilation": "1.0", "--d-max": "0.02"}),
})


def box_mesh(half, spacing):
    """`(v, f)`: a box of half extents `half`, its faces a lattice about `spacing` apart."""
    half = np.asarray(half, dtype=float)
    n = np.maximum(1, np.round(2.0 * half / spacing)).astype(int)  # segments per axis
    index, verts, faces = {}, [], []

    def vid(ijk):
        key = tuple(int(x) for x in ijk)
        if key not in index:
            index[key] = len(verts)
            verts.append(-half + 2.0 * half * np.array(key) / n)
        return index[key]

    for a in range(3):
        # (u, v, a) is right-handed, so a counter-clockwise quad in (u, v) faces +a.
        u, w = (a + 1) % 3, (a + 2) % 3
        for side in (0, 1):
            for p in range(n[u]):
                for q in range(n[w]):
                    def lat(pp, qq):
                        ijk = [0, 0, 0]
                        ijk[a], ijk[u], ijk[w] = side * n[a], pp, qq
                        return vid(ijk)

                    c00, c10, c11, c01 = lat(p, q), lat(p + 1, q), lat(p + 1, q + 1), lat(p, q + 1)
                    if side == 1:
                        faces += [(c00, c10, c11), (c00, c11, c01)]
                    else:
                        faces += [(c00, c11, c10), (c00, c01, c11)]
    return np.array(verts), np.array(faces, dtype=np.int64)


def sphere_mesh(radius, spacing):
    """`(v, f)`: an icosphere of `radius`, subdivided until its edges are at most `spacing`."""
    for level in range(8):
        m = trimesh.creation.icosphere(subdivisions=level, radius=radius)
        if m.edges_unique_length.max() <= spacing:
            break
    return np.asarray(m.vertices), np.asarray(m.faces, dtype=np.int64)


def ring_mesh(size, spacing):
    """`(v, f)`: a ring about z -- `size` its inner radius, wall and height -- its rims cut about `spacing` apart."""
    r, wall, height = size
    m = trimesh.creation.annulus(r, r + wall, height, sections=int(np.ceil(2.0 * np.pi * (r + wall) / spacing)))
    return np.asarray(m.vertices), np.asarray(m.faces, dtype=np.int64)


def stone_mesh(i, spacing):
    """`(v, f)`: the arch's stone `i` in its body frame, its faces subdivided to `spacing`."""
    from viewer.scenes import fbf

    v, f, _ = fbf.load_stone(i)
    return trimesh.remesh.subdivide_to_size(v, f, max_edge=spacing)


def write_obj(path, v, f):
    with open(path, "w") as out:
        out.writelines(f"v {x:.9g} {y:.9g} {z:.9g}\n" for x, y, z in v)
        out.writelines(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in f)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", default=list(SHAPES), help=f"of {list(SHAPES)}")
    ap.add_argument("--out", type=Path, default=ASSETS, help="where the surfaces are installed")
    a = ap.parse_args()
    make = {"box": box_mesh, "sphere": sphere_mesh, "stone": stone_mesh, "ring": ring_mesh}
    # One bake per set of grid options: the bake takes one set for its whole directory.
    groups = {}
    for name in a.names:
        groups.setdefault(tuple(sorted(SHAPES[name][3].items())), []).append(name)
    for g, (opts, names) in enumerate(groups.items()):
        work = HERE / ".tmp" / f"scene_surfaces{g}"
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        for name in names:
            kind, size, spacing, _ = SHAPES[name]
            v, f = make[kind](size, spacing)
            mesh = trimesh.Trimesh(v, f, process=False)
            assert mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0, name
            write_obj(work / f"{name}.obj", v, f)
            print(f"{name}: {kind} {size}, {len(v)} vertices, {len(f)} faces")
        args = [x for kv in opts for x in kv]
        subprocess.run([sys.executable, str(HERE / "create_surface_data.py"), str(work), "--no-view", "--rebake", *args],
                       check=True)
        baked = work.with_name(work.name + "_sdata")
        for name in names:
            dest = a.out / name
            dest.mkdir(parents=True, exist_ok=True)
            with np.load(baked / f"{name}.npz") as z:  # compressed, as the shipped surfaces are
                np.savez_compressed(dest / "surface_data.npz", **{k: z[k] for k in z.files})
            print(f"installed {dest / 'surface_data.npz'}")


if __name__ == "__main__":
    main()
