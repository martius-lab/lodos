"""
The Lodos surfaces of the NIST nuts and bolts (`viewer.scenes.nist`), from `extract.py`'s meshes.

    .venv/bin/python tools/mesh_processing/nist/make_surfaces.py viewer/assets/nist [--sizes m8 m12 m16]
    .venv/bin/python tools/mesh_processing/nist/make_surfaces.py viewer/assets/nist --edges 0.1 --no-bolt   # one more nut
    .venv/bin/python tools/mesh_processing/nist/make_surfaces.py viewer/assets/nist --sizes --board           # the board alone
    .venv/bin/python tools/mesh_processing/nist/make_surfaces.py viewer/assets/nist --sizes --pad             # the Franka's finger pad

For each size, into the asset directory:

    repair/<part>.obj         its collision mesh, watertight
    im/<nut>_s<EDGE>.obj      the nut remeshed at a fixed edge length, EDGE in mm
    surfaces/<nut>_s<EDGE>/   the nut's vertices with its field
    surfaces/<bolt>/          the bolt's field

Why each step:

- **Repair.** Three of the six collision meshes (bolt_m8, bolt_m12, nut_m16) are a watertight shell
  plus one or two stray single triangles, of no volume, glued along an edge of it. The repair keeps
  the largest piece, and refuses a piece that is still open.
- **The nut's vertices.** The collision mesh's own vertices lie on the thread's crest and root lines,
  none across the flanks that slide (529 on the M8 nut), as on the M20 CAD nut.
  So the nut is remeshed at a fixed edge length by
  `nutbolt/remesh.py` (Instant Meshes, each vertex projected back onto the repaired mesh): a fifth
  and two fifths of the pitch, as the M20 nut's 0.5 and 1 mm stand to its 2.5 mm, and a tenth, for a
  denser nut.
- **The bolt's vertices are its collision mesh's own.** Its bake carries the repaired mesh's
  vertices, on the thread's crest and root lines, where a flat face meets a faceted thread first:
  the Franka's pads read them (`viewer.scenes.nist.franka_surfaces`), and the nut reads its field.
  The field has 12.5 voxels a pitch, as the M20 bolt's 0.2 mm has: 0.1, 0.14 and 0.16 mm.
- **Both fields are the repaired collision mesh's** (`nutbolt/bake.py`), the geometry Newton bakes
  for Isaac Lab's demo. The nut's own field (0.2 mm) is for the pairs that will need it later.
- **The board** (`--board`) collides as one box, 385 x 384.6 x 30 mm (`nistboard.usd`'s collision
  `Cube`). Its field is baked from that box at 2 mm, which is exact on its faces: the field is linear
  there, and a tricubic spline reproduces it, rounding only within a couple of voxels of an edge.
  Nothing reads its vertices, a lattice 10 mm apart (`scene_surfaces.box_mesh`).
- **The Franka's finger pad** (`--pad`) is the box Isaac Lab's gripper collides with,
  17.5 x 15.2 x 16.5 mm (`viewer.scenes.franka.PAD_HALF`), placed in its finger's frame as the finger
  carries it, since a surface's frame is its body's; both fingers share it. Its field is baked at
  0.25 mm, exact on its faces as the board's is; nothing reads its vertices, a lattice 0.5 mm apart.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "nutbolt"))
import bake  # noqa: E402  (tools/mesh_processing/nutbolt/bake.py)

SIZES = {
    "m8": dict(nut="nut_m8", bolt="bolt_m8", pitch=1.25e-3),
    "m12": dict(nut="nut_m12", bolt="bolt_m12", pitch=1.75e-3),
    "m16": dict(nut="nut_m16", bolt="bolt_m16", pitch=2.0e-3),
}
NUT_EDGES = (0.1, 0.2, 0.4)  # fractions of the pitch
NUT_H = 0.2e-3
BOLT_VOXELS_PER_PITCH = 12.5
MARGIN = 3e-3
BOARD_H, BOARD_MARGIN, BOARD_SPACING = 2e-3, 6e-3, 10e-3
PAD_H, PAD_MARGIN, PAD_SPACING = 0.25e-3, 2e-3, 0.5e-3


def repair(src, dst):
    m = trimesh.load(src, force="mesh", process=True)
    pieces = m.split(only_watertight=False)
    big = max(pieces, key=lambda p: len(p.faces))
    if not big.is_watertight:
        raise ValueError(f"{src}: its largest piece is not watertight")
    dst.parent.mkdir(parents=True, exist_ok=True)
    big.export(dst)
    print(f"{dst.name}: {len(pieces)} piece(s), kept {len(big.faces)} of {len(m.faces)} faces, watertight, "
          f"euler {big.euler_number}, volume {big.volume * 1e6:.4f} cm^3", flush=True)


def make_board(D):
    """The board's field from its collision box, with a lattice of vertices over the box's faces."""
    sys.path.insert(0, str(HERE.parent))
    from scene_surfaces import box_mesh

    box = D / "src" / "nistboard_collision.obj"
    lo, hi = trimesh.load(box, force="mesh", process=True).bounds
    v, f = box_mesh((hi - lo) / 2, BOARD_SPACING)
    lattice = D / "im" / "nistboard_lattice.obj"
    lattice.parent.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(v + (lo + hi) / 2, f, process=False).export(lattice)
    bake.build(lattice, box, BOARD_H, BOARD_MARGIN, D / "surfaces" / "nistboard")


def make_pad(D):
    """The Franka's finger pad: a box's field in its finger's frame, with a lattice of vertices over the box's faces."""
    sys.path.insert(0, str(HERE.parent))
    from scene_surfaces import box_mesh

    from viewer.scenes import franka as F

    v, f = box_mesh(np.asarray(F.PAD_HALF), PAD_SPACING)
    box = D / "im" / f"{F.PAD}.obj"
    box.parent.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(v + np.asarray(F.PAD_POS), f, process=False).export(box)
    bake.build(box, box, PAD_H, PAD_MARGIN, D / "surfaces" / F.PAD)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("asset_dir")
    ap.add_argument("--sizes", nargs="*", default=list(SIZES))
    ap.add_argument("--edges", nargs="*", type=float, default=list(NUT_EDGES), help="the nuts' edge lengths, as fractions of the pitch")
    ap.add_argument("--no-bolt", action="store_true", help="leave the bolts' fields as they are")
    ap.add_argument("--board", action="store_true", help="bake the board too (src/nistboard_collision.obj)")
    ap.add_argument("--pad", action="store_true", help="bake the Franka's finger pad too")
    a = ap.parse_args()
    D = Path(a.asset_dir)
    if a.board:
        make_board(D)
    if a.pad:
        make_pad(D)
    for size in a.sizes:
        p = SIZES[size]
        for part in (p["nut"], p["bolt"]):
            repair(D / "src" / f"{part}_collision.obj", D / "repair" / f"{part}.obj")
        nut_mesh, bolt_mesh = D / "repair" / f"{p['nut']}.obj", D / "repair" / f"{p['bolt']}.obj"
        for frac in a.edges:
            edge = round(frac * p["pitch"] * 1e3, 3)
            im = D / "im" / f"{p['nut']}_s{edge:g}.obj"
            im.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([sys.executable, str(HERE.parent / "nutbolt" / "remesh.py"), str(nut_mesh), f"{edge * 1e-3}", str(im)],
                           check=True)
            bake.build(im, nut_mesh, NUT_H, MARGIN, D / "surfaces" / f"{p['nut']}_s{edge:g}")
        if not a.no_bolt:
            bake.build(bolt_mesh, bolt_mesh, p["pitch"] / BOLT_VOXELS_PER_PITCH, MARGIN, D / "surfaces" / p["bolt"])


if __name__ == "__main__":
    main()
