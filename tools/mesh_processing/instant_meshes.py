"""
Driving the Instant Meshes binary.

Instant Meshes fits a smooth orientation and position field over a surface and
extracts a mesh aligned to it. What that buys over quadric collapse is
*isotropy* -- triangles of roughly equal size, so the vertices are evenly spread.
For a contact set that matters more than it does for rendering, because the
vertices *are* the samples and their spacing is the resolution; quadric collapse
leaves long thin triangles along flat regions, which is a fine surface and a lumpy
sampling of it.

`-r 6 -p 6` is the pure triangle configuration (6-RoSy orientation, 6-PoSy
position), the tools' default; `-r 4 -p 4` is the quad one.

The two need separate calibrations. At `-v 1492` the armadillo comes out as 1483
vertices of triangles and 5924 of quads -- roughly four times as many -- so a
vertex budget tuned on one configuration means nothing on the other.

The quads are *not* planar. Measured on the armadillo at the density matching its
triangle mesh: the worst vertex is a median 2.6% of an edge length off the quad's
best-fit plane, 20% at the 99th percentile, and 27% of quads exceed 5%. Anything
downstream that needs a face plane has to either fit one and carry the error, or
split the warped quads into triangles.

It guarantees nothing about topology. It is a remesher, not a repair tool, and at
low vertex counts it can leave holes -- so callers check its result, and this runs
late in the pipeline, on the genus-0 solids a convex union produces rather than on
raw CAD.

Not pip-installable. Pre-built binaries are linked from the project README:
https://github.com/wjakob/instant-meshes
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
BINARY = HERE / ".instant-meshes"


def read_obj(path):
    """
    Positions and faces from an OBJ, ignoring normals and texture coordinates.

    `trimesh.load_mesh` is the wrong tool for what Instant Meshes writes: it splits a
    vertex whenever the per-corner normal indices attached to it differ, which turns a
    997-vertex file into 5,994 and makes every count and topology check meaningless.
    n-gons are fanned into a triangle trimesh, which is what the topology checks
    and the viewer want. It is lossy for `-p 4` output: use `read_obj_polys` when
    the polygons themselves matter.
    """
    v, f = [], []
    for line in Path(path).read_text().splitlines():
        if line.startswith("v "):
            v.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
            f.extend([[idx[0], idx[i], idx[i + 1]] for i in range(1, len(idx) - 1)])
    return trimesh.Trimesh(np.array(v), np.array(f, dtype=np.int64), process=False)


def remesh(src, dst, target_v, binary=BINARY, rosy=6, posy=6, deterministic=True):
    """Remesh `src` to roughly `target_v` vertices; returns the result as a `Trimesh`."""
    cmd = [str(binary), "-r", str(rosy), "-p", str(posy), "-v", str(int(target_v)),
           "-o", str(dst)]
    if deterministic:
        cmd.append("-d")
    cmd.append(str(src))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not Path(dst).exists():
        tail = (proc.stderr or proc.stdout or "no output").strip().splitlines()[-1:]
        raise RuntimeError(tail[0] if tail else "instant meshes failed")
    return read_obj(dst)


def target_vertices(mesh, spacing):
    """
    How many vertices `mesh` gets at a fixed areal density.

    A ratio of the input is the knob most simplifiers offer and the wrong one here:
    bodies span 15 to 318 cm2, so one ratio gives the palm eight times the contact
    resolution of a fingertip. Fixing the spacing gives every part the same distance
    between contact candidates.
    """
    return max(int(mesh.area / spacing**2), 20)


def read_obj_polys(path):
    """
    Positions and faces from an OBJ, with polygons left intact.

    `read_obj` fans every n-gon into triangles, which is right for the topology
    checks and wrong for anything that wants the quads `-r 4 -p 4` produces.
    Returns `(vertices (V, 3), faces)` where `faces` is a list of index lists,
    each of length 3 or 4 in practice.
    """
    v, f = [], []
    for line in Path(path).read_text().splitlines():
        if line.startswith("v "):
            v.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            f.append([int(t.split("/")[0]) - 1 for t in line.split()[1:]])
    return np.array(v), f


def write_obj_polys(path, verts, faces):
    """An OBJ with its polygons intact, which `trimesh.export` cannot write."""
    lines = [f"v {x:.9g} {y:.9g} {z:.9g}" for x, y, z in verts]
    lines += ["f " + " ".join(str(i + 1) for i in f) for f in faces]
    Path(path).write_text("\n".join(lines) + "\n")
