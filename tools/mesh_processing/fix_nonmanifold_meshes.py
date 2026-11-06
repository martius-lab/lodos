#!/usr/bin/env python3
"""
Step 4 -- make every remeshed body a single closed 2-manifold.

    python fix_nonmanifold_meshes.py path/to/robot_wt_cp_fused_im

        ->  path/to/robot_wt_cp_fused_im_man/  and  ..._man_world/

Step 4 of five, `_man` for manifold. Each step appends its own suffix to whatever its input carried, so a directory's
name is the record of how it was made:

    robot.xml -> _wt -> _wt_cp_fused -> _wt_cp_fused_im -> ..._im_man -> ..._man_sdata

`mjcf.model_stem` strips the lot to recover `robot`, which is how each step finds
the MJCF its world-frame poses need. There is no output argument anywhere;
`run_pipeline.py` runs the whole chain from the XML.

Instant Meshes is a remesher, not a repair tool, and at a contact-vertex budget it
sometimes returns a surface that touches itself. `interactive_remesh.py` reports
that but cannot fix it: the slider is the only control it has, and the defect is
not monotone in the slider, so hunting for a clean count by hand means dragging
back and forth watching a label. This does the hunt.

Two repairs, cheapest first, per body.

**Surgery.** Delete every copy of each repeated face. Those repeats are
zero-thickness flaps -- oppositely wound, enclosing nothing -- so removing them
takes the flap's edges from four incident faces down to two, leaves the volume
identical to the last bit, and moves no surviving vertex. On the SHARPA hand this
alone fixes four of the five bad bodies at a cost of two to fourteen faces, and of
the nought to six vertices the flap alone was holding up. It is tried first because
a repair that cannot move a contact candidate is worth more than one that produces
a slightly nicer mesh.

**Re-remesh.** What surgery cannot fix is a genuine pinch: an edge where two sheets
cross, which no local edit separates without tearing the vertex fans around its
endpoints. But the defect is an accident of the target count rather than of the
geometry -- a body can be clean at 100 vertices, pinched at 105, holed at 110 and
clean again at 150 -- so the fix is to ask Instant Meshes for a different number.
This searches outward from the count the body already has, `+1, -1, +2, -2, ...`,
and takes the first target whose result survives surgery as a closed manifold.
Nearest first, so the contact budget moves as little as the topology allows.

The search remeshes `--remesh-from`, which must be the *input* Instant Meshes was
given, not its output: remeshing an already-decimated mesh decimates it twice.
That is this directory's name without the `_im` it ends in, which is where the
default looks.

A body repaired by surgery keeps its vertices bit for bit, so anything measured
against the earlier steps still holds. A body repaired by re-remesh gets *new
vertices in new places*, so its own numbers -- volume, spacing, whatever step 3
reported -- no longer describe it.

Everything is written before the viewer opens, so closing the window cannot alter
what was produced. The window is the check, not the control: `before` and `after`
per body, with the same visibility and focus panel as the other steps.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mujoco
import trimesh

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import instant_meshes as im  # noqa: E402
from mjcf import body_poses, model_stem, step_dir  # noqa: E402
from topology import drop_flaps_polys, fan, topo_tag, topology_polys  # noqa: E402
from mesh_viewer import Body, MeshViewer, body_color  # noqa: E402


def search_offsets(limit):
    """`0, +1, -1, +2, -2, ...` -- nearest counts first, so the budget moves least."""
    yield 0
    for d in range(1, limit + 1):
        yield d
        yield -d


def repair(name, pm, remesh_from, args):
    """
    One body: `((verts, polys), action, detail)`, the best result found.

    `action` is one of `clean` (nothing was wrong), `surgery`, `remesh` or `failed`;
    `failed` returns the input untouched rather than a half-repair, so a body this
    cannot fix is left exactly as it was for the viewer to show.

    Everything here works on the polygons. Checking a fanned triangulation instead
    invents defects on a quad mesh -- see `topology.fan` -- and the armadillo quad
    remesh is the case in point: fanned it looks like 9 shells and 8 non-manifold
    edges, and it is in fact a single closed surface needing no repair at all.
    """
    verts, polys = pm
    if topology_polys(verts, polys).ok:
        return pm, "clean", None

    cv, cp, nf, nv = drop_flaps_polys(verts, polys)
    if topology_polys(cv, cp).ok:
        return (cv, cp), "surgery", f"-{nf}f" + (f" -{nv}v" if nv else "")

    if remesh_from is None or not remesh_from.exists():
        return pm, "failed", "no source to remesh"

    have = len(verts)
    for offset in search_offsets(args.search):
        target = have + offset
        if target < 20:
            continue
        dst = args.tmp / f"{name}_{target}.obj"
        try:
            im.remesh(remesh_from, dst, target, args.binary, args.rosy, args.posy)
        except RuntimeError:
            continue
        rv, rp = im.read_obj_polys(dst)
        rv, rp, _, _ = drop_flaps_polys(rv, rp)
        if topology_polys(rv, rp).ok:
            return (rv, rp), "remesh", f"target {target}"

    return pm, "failed", f"no clean target within +/-{args.search}"


class Inspector(MeshViewer):
    """`MeshViewer` over the before/after pair. Nothing recomputes while it is open."""

    def __init__(self, bodies, out, counts, args):
        super().__init__(
            bodies, ["before", "after", "wire"],
            title="Manifold repair",
            subtitle=f"{len(bodies)} bodies from {args.input.name}",
            default_on=["after", "wire"], wire=["wire"], line_width=args.line_width,
        )
        self.out = out
        self.counts = counts

    def extra_header(self, imgui):
        imgui.text("  ".join(f"{n} {k}" for k, n in self.counts.items() if n)
                   + f"   ->  {self.out.name}/")


def row_info(r):
    """One body's line in the panel and on the terminal."""
    if r["action"] == "clean":
        return f"{r['v_before']:5d} V   closed manifold"
    arrow = f"{r['v_before']:5d} ->{r['v_after']:5d} V"
    after = "closed" if r["manifold"] else r["after_tag"]
    return (f"{arrow}   {r['before_tag']}  ->  {after}"
            + f"   [{r['action']}{' ' + r['detail'] if r['detail'] else ''}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="a directory of per-body OBJs")
    ap.add_argument("--remesh-from", type=Path, default=None,
                    help="the meshes Instant Meshes should re-read; "
                         "default: <input> without the `_im` it ends in")
    ap.add_argument("--xml", type=Path, default=None,
                    help="the MJCF the bodies come from, for the world-frame poses")
    ap.add_argument("--search", type=int, default=40,
                    help="how far either side of the current count to look")
    ap.add_argument("--binary", type=Path, default=im.BINARY)
    ap.add_argument("--rosy", type=int, default=6)
    ap.add_argument("--posy", type=int, default=6)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--line-width", type=float, default=2.0e-4)
    ap.add_argument("--out", type=Path, default=None,
                    help="where to write; defaults to the input's name plus `_man`, "
                         "beside it")
    ap.add_argument("--no-view", dest="view", action="store_false")
    args = ap.parse_args()

    args.tmp = HERE / ".tmp" / "fix"
    args.tmp.mkdir(parents=True, exist_ok=True)

    out = args.out if args.out is not None else step_dir(args.input, "_man")
    world = out.parent / f"{out.name}_world"

    # The meshes Instant Meshes was *given*, not what it returned: under the naming
    # convention that is this directory's own name with the `_im` it appended removed.
    src_dir = args.remesh_from
    if src_dir is None and args.input.name.endswith("_im"):
        sibling = args.input.parent / args.input.name[: -len("_im")]
        src_dir = sibling if sibling.is_dir() else None

    meshes = {p.stem: im.read_obj_polys(p) for p in sorted(args.input.glob("*.obj"))}
    if not meshes:
        raise SystemExit(f"no .obj files in {args.input}")

    xml = args.xml or args.input.parent / f"{model_stem(args.input.name)}.xml"
    if not xml.exists():
        raise SystemExit(f"no MJCF at {xml} -- pass --xml")
    poses = body_poses(mujoco.MjModel.from_xml_path(str(xml)))

    broken = [n for n, m in meshes.items() if not topology_polys(*m).ok]
    print(f"{args.input.name}: {len(meshes)} bodies, {len(broken)} not a closed manifold"
          + (f" ({', '.join(broken)})" if broken else ""))
    if broken and src_dir is not None:
        print(f"re-remeshing, where needed, from {src_dir.name}/")
    elif broken:
        print("no --remesh-from: surgery only, pinched bodies will be left alone")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(
            lambda n: repair(n, meshes[n],
                             (src_dir / f"{n}.obj") if src_dir else None, args),
            meshes,
        ))

    out.mkdir(parents=True, exist_ok=True)
    world.mkdir(parents=True, exist_ok=True)

    rows, bodies = [], []
    for i, (name, (fixed, action, detail)) in enumerate(zip(meshes, results)):
        before = meshes[name]
        t0, t1 = topology_polys(*before), topology_polys(*fixed)
        fv, fp = fixed
        im.write_obj_polys(out / f"{name}.obj", fv, fp)
        pose = poses.get(name)
        pv = fv @ pose[0].T + pose[1] if pose else fv
        im.write_obj_polys(world / f"{name}.obj", pv, fp)
        # trimeshes only for the volume and the viewer, never for the checks
        m_before, m_after = fan(*before), fan(fv, fp)

        rows.append(dict(
            body=name, action=action, detail=detail,
            v_before=int(len(before[0])), v_after=int(len(fv)),
            f_before=int(len(before[1])), f_after=int(len(fp)),
            before_tag=topo_tag(t0, "closed"), after_tag=topo_tag(t1, "closed"),
            manifold=bool(t1.ok),
            volume_change=round(float(m_after.volume / m_before.volume) - 1.0, 6),
        ))
        bodies.append(Body(name, {"before": m_before, "after": m_after, "wire": m_after},
                           body_color(i), pose, row_info(rows[-1])))

    (out / "fix_nonmanifold_meshes.json").write_text(json.dumps(rows, indent=2))

    counts = {k: sum(1 for r in rows if r["action"] == k)
              for k in ("clean", "surgery", "remesh", "failed")}
    for r in rows:
        if r["action"] != "clean":
            print(f"  {r['body']:24s} {row_info(r)}")
    print(f"\n{len(rows)} meshes, {sum(r['v_after'] for r in rows):,d} vertices -> "
          f"{out.name}/ and {world.name}/")
    print("  " + ", ".join(f"{n} {k}" for k, n in counts.items() if n))
    if counts["remesh"]:
        moved = [r["body"] for r in rows if r["action"] == "remesh"]
        print(f"  re-remeshed, so new vertices on: {', '.join(moved)}")
    if counts["failed"]:
        print(f"  STILL BROKEN: {', '.join(r['body'] for r in rows if r['action'] == 'failed')}")

    if args.view:
        Inspector(bodies, out, counts, args).run()


if __name__ == "__main__":
    main()
