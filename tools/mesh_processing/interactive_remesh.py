#!/usr/bin/env python3
"""
Step 3 -- choose a vertex budget per body, by eye, and remesh to it.

    python interactive_remesh.py path/to/robot_wt_cp_fused

        ->  path/to/robot_wt_cp_fused_im/  and  ..._im_world/

Step 3 of five. Each step appends its own suffix to whatever its input carried, so a directory's
name is the record of how it was made:

    robot.xml -> _wt -> _wt_cp_fused -> _wt_cp_fused_im -> ..._im_man -> ..._man_sdata

`mjcf.model_stem` strips the lot to recover `robot`, which is how each step finds
the MJCF its world-frame poses need. There is no output argument anywhere;
`run_pipeline.py` runs the whole chain from the XML.

The input is a directory of OBJs directly -- step 2's `_cp_fused/`, not its `_cp/`,
which holds a directory of hulls per body.

These vertices are the contact candidates. Every one of them is tested against the
other half of a collision pair, so the count is the per-step cost of the model, and
their *spacing* is its resolution. Instant Meshes is here for the spacing: it fits a
field over the surface and extracts a mesh aligned to it, giving triangles of
roughly equal size where quadric collapse gives long thin ones along the flats.
Measured over a hand, edge-length CV 0.13 against 0.48.

The slider is a vertex count, logarithmic, because the useful range spans two
decades -- 40 for a fingertip against 900 for the palm -- and a linear slider would
spend nine tenths of its travel above anything you would choose. Initial values
come from `--spacing`, a fixed areal density.

What is drawn: the result solid with its own wireframe on it, which is the view
that answers the question the tool is for. `source` is off by default: the remesh
sits *inside* its input -- Instant Meshes shrinks, here by 1 to 20% of volume --
so both solid shows only the input and hides what is being judged. Turn it on with
`remesh` off to see the wireframe against the original surface, which is how to
spot a feature the remesh has cut off.

Instant Meshes guarantees nothing about topology, and it fails in three different
ways that want different answers, so each row names which one it hit rather than
reporting one "not closed" flag. `topology.py` says what the labels mean.

The distinction matters because the count is not monotone in the target: a body
can be clean at 100 vertices, pinched at 105 and holed at 110. A row reading
`nonman` or `dup f` is usually fixed by nudging the slider a few vertices either
way, or afterwards by `fix_nonmanifold_meshes.py`, which does that search; one
reading `open` has been asked for too few and needs a real increase.

Where these meshes are the *vertex* half of an asymmetric pair, no field is baked
from them and `make_collision_context` never reads their faces, so the defects are
inert.

Work runs in threads, not processes: each job is a `subprocess.run` of the Instant
Meshes binary, which releases the GIL for its whole duration.

`Export` is the only thing that writes, and closing the window writes nothing. The
sliders start from `--spacing` rather than from whatever is already in the output
directory, so a run that wrote on close would replace a set of hand-chosen counts
with the default ones on the way out -- and the counts are the whole product of
this step. Deciding to keep a result should take a click, not the absence of one.

**Without the window.** `--targets FILE` starts each body from the vertex target a
previous run chose -- that run's `interactive_remesh.json`, in `_im/` -- and `--no-view`
remeshes to those targets, writes, and exits: the step's choices, replayed. Instant
Meshes is deterministic, so the meshes come back the same.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mujoco
import trimesh

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import instant_meshes as im  # noqa: E402
from mjcf import body_poses, model_stem, step_dir  # noqa: E402
from topology import topo_tag, topology  # noqa: E402
from mesh_viewer import Body, MeshViewer, body_color  # noqa: E402


class Remesher(MeshViewer):
    """`MeshViewer` plus one Instant Meshes job per body and the slider that drives it."""

    def __init__(self, bodies, sources, pool, args, out):
        super().__init__(
            bodies, ["source", "remesh", "wire"],
            title="Interactive remesh",
            subtitle=f"{len(bodies)} bodies from {args.input.name}   (-r 6 -p 6, triangles)",
            default_on=["remesh", "wire"], wire=["wire"], line_width=args.line_width,
        )
        self.sources = sources
        self.pool = pool
        self.args = args
        self.out = out
        for b in self.bodies:
            b.target = im.target_vertices(sources[b.name], args.spacing)
            b.applied = None
            b.topo = None
            b.job_dirty = True
            b.touched = 0.0
            b.future = None
            b.seconds = None
            b.error = None

    #######################################################################################
    # Work
    #######################################################################################

    def on_frame(self):
        now = time.monotonic()
        for b in self.bodies:
            if b.future is not None and b.future.done():
                try:
                    result = b.future.result()
                    b.variants["remesh"] = result
                    b.variants["wire"] = result
                    # Cached, not recomputed per frame: `split` walks the face
                    # adjacency, and the panel redraws at frame rate.
                    b.topo = topology(result)
                    b.error, b.seconds = None, now - b.started
                except Exception as exc:
                    b.variants["remesh"] = b.variants["wire"] = None
                    b.topo = None
                    b.error = f"{type(exc).__name__}: {exc}"
                b.applied = b.submitted
                b.future = None
                b.job_dirty = b.target != b.applied
                b.dirty = True
                self._label(b)

        in_flight = sum(1 for b in self.bodies if b.future is not None)
        for b in self.bodies:
            if b.future is not None or not b.job_dirty or in_flight >= self.args.workers:
                continue
            if now - b.touched < self.args.debounce:
                continue
            b.started, b.submitted = now, b.target
            b.future = self.pool.submit(
                im.remesh, self.sources[b.name].tmp_path, self.args.tmp / f"{b.name}.obj",
                b.target, self.args.binary, self.args.rosy, self.args.posy,
            )
            b.job_dirty = False
            in_flight += 1

    def _label(self, b):
        result = b.variants.get("remesh")
        if b.error:
            b.info = f"! {b.error[:34]}"
            return
        state = ("working..." if b.future is not None
                 else "pending" if b.job_dirty
                 else f"{b.seconds:.1f}s" if b.seconds is not None else "")
        if result is None:
            b.info = f"{'':>5s}  {state}"
            return
        b.info = (f"{len(result.vertices):5d} V  "
                  f"{result.volume / self.sources[b.name].volume - 1:+6.1%}  "
                  f"{topo_tag(b.topo)}  {state}")

    #######################################################################################
    # Panel
    #######################################################################################

    def extra_header(self, imgui):
        done = [b for b in self.bodies if b.variants.get("remesh") is not None]
        total = sum(len(b.variants["remesh"].vertices) for b in done)
        holed = sum(1 for b in done if b.topo.bnd)
        pinched = sum(1 for b in done if not b.topo.bnd and not b.topo.ok)
        bad = ", ".join(
            s for s in (f"{holed} holed" if holed else "", f"{pinched} pinched" if pinched else "") if s
        )
        imgui.text(f"{total:,d} contact vertices"
                   + (f"   {bad}" if bad else "   all closed manifolds")
                   + f"   ->  {self.out.name}/")
        if imgui.button("Export"):
            self.message = self.write()
            print(self.message)
        imgui.same_line()
        imgui.text("(closing writes nothing)")
        if self.message:
            imgui.text_wrapped(self.message)

    def extra_row(self, imgui, i, b):
        imgui.set_next_item_width(165)
        changed, b.target = imgui.slider_int(
            f"##v{i}", b.target, 20, 5000, "%d", imgui.SliderFlags_.logarithmic
        )
        if changed:
            b.job_dirty, b.touched = True, time.monotonic()
            self._label(b)
        return True

    #######################################################################################
    # Output
    #######################################################################################

    def write(self):
        """Export. In quad mode the polygons are preserved; see below."""
        world = self.out.parent / f"{self.out.name}_world"
        self.out.mkdir(parents=True, exist_ok=True)
        world.mkdir(parents=True, exist_ok=True)

        rows, skipped = [], []
        for b in self.bodies:
            mesh = b.variants.get("remesh")
            if mesh is None:
                skipped.append(b.name)
                continue
            # Under `-p 4` the file Instant Meshes wrote holds quads, while
            # `mesh` is the fanned triangle version the topology checks and the
            # viewer run on. Exporting `mesh` would throw the quads away before
            # anything downstream ever saw them, so in quad mode the polygons are
            # carried across from the raw output instead.
            raw = self.args.tmp / f"{b.name}.obj"
            if self.args.posy == 4 and raw.exists():
                rv, rf = im.read_obj_polys(raw)
                im.write_obj_polys(self.out / f"{b.name}.obj", rv, rf)
                im.write_obj_polys(world / f"{b.name}.obj", b.posed(rv), rf)
            else:
                mesh.export(self.out / f"{b.name}.obj")
                w = mesh.copy()
                w.vertices = b.posed(w.vertices)
                w.export(world / f"{b.name}.obj")

            t = b.topo or topology(mesh)
            rows.append(dict(
                body=b.name, target=b.applied or b.target,
                v=int(len(mesh.vertices)), f=int(len(mesh.faces)), shells=t.shells,
                manifold=bool(t.ok),
                boundary_edges=t.bnd, nonmanifold_edges=t.nm, duplicate_faces=t.dup,
                volume_cm3=round(float(mesh.volume) * 1e6, 3),
                volume_change=round(float(mesh.volume / self.sources[b.name].volume) - 1.0, 4),
                stale=bool(b.job_dirty or b.future is not None),
                rosy=int(self.args.rosy), posy=int(self.args.posy),
            ))
        (self.out / "interactive_remesh.json").write_text(json.dumps(rows, indent=2))

        bad = [r for r in rows if not r["manifold"]]
        stale = [r for r in rows if r["stale"]]
        return (f"{len(rows)} meshes, {sum(r['v'] for r in rows):,d} vertices -> "
                f"{self.out.name}/ and {world.name}/"
                + (f"; {len(bad)} not a closed manifold ({', '.join(r['body'] for r in bad)})" if bad else "")
                + (f"; {len(stale)} stale" if stale else "")
                + (f"; skipped {len(skipped)}" if skipped else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="a directory of per-body OBJs")
    ap.add_argument("--xml", type=Path, default=None,
                    help="the MJCF the bodies come from, for the world-frame poses")
    ap.add_argument("--binary", type=Path, default=im.BINARY)
    ap.add_argument("--spacing", type=float, default=5.648e-3,
                    help="metres between contact candidates, for the initial targets")
    ap.add_argument("--out", type=Path, default=None,
                    help="where to write; defaults to the input's name plus `_im`, "
                         "beside it. Give it explicitly when remeshing the same "
                         "input a second way -- quads and triangles are separate "
                         "calibrations and must not land on top of each other.")
    ap.add_argument("--rosy", type=int, default=6)
    ap.add_argument("--posy", type=int, default=6)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--debounce", type=float, default=0.35)
    ap.add_argument("--line-width", type=float, default=2.0e-4)
    ap.add_argument("--targets", type=Path, default=None,
                    help="a previous run's interactive_remesh.json: each body's vertex target")
    ap.add_argument("--no-view", dest="view", action="store_false",
                    help="remesh to the targets given, write, and exit without the window")
    args = ap.parse_args()

    if not args.binary.exists():
        raise SystemExit(
            f"Instant Meshes binary not found at {args.binary}\n"
            f"    pre-built binaries: https://github.com/wjakob/instant-meshes"
        )
    args.tmp = HERE / ".tmp" / "remesh"
    args.tmp.mkdir(parents=True, exist_ok=True)

    out = args.out if args.out is not None else step_dir(args.input, "_im")

    # The MJCF is only wanted for the `qpos0` poses. Its stem is the source name with
    # every pipeline suffix stripped, which is how the earlier steps named things.
    xml = args.xml or args.input.parent / f"{model_stem(args.input.name)}.xml"
    if not xml.exists():
        raise SystemExit(f"no MJCF at {xml} -- pass --xml")

    sources = {}
    for p in sorted(args.input.glob("*.obj")):
        mesh = trimesh.load_mesh(p, process=False)
        mesh.tmp_path = p            # Instant Meshes reads a file, so keep the path
        sources[p.stem] = mesh
    if not sources:
        raise SystemExit(f"no .obj files in {args.input}")

    poses = body_poses(mujoco.MjModel.from_xml_path(str(xml)))
    bodies = [
        Body(name, {"source": sources[name], "remesh": None, "wire": None},
             body_color(i), poses.get(name))
        for i, name in enumerate(sources)
    ]
    print(f"{args.input.name}: {len(bodies)} bodies, "
          f"{sum(len(m.vertices) for m in sources.values()):,d} vertices")

    pool = ThreadPoolExecutor(max_workers=args.workers)
    rem = Remesher(bodies, sources, pool, args, out)
    chosen = {}
    if args.targets is not None:
        chosen = {r["body"]: int(r["target"]) for r in json.loads(args.targets.read_text())}
        for b in rem.bodies:
            if b.name in chosen:
                b.target = chosen[b.name]
    print(f"targets from {args.spacing*1e3:.3f} mm spacing"
          + (f", {len(chosen)} from {args.targets.name}" if chosen else "")
          + f" ({min(b.target for b in bodies)}-{max(b.target for b in bodies)} vertices)\n")
    try:
        while any(b.job_dirty or b.future is not None for b in rem.bodies):
            rem.on_frame()
            time.sleep(0.05)
        for b in rem.bodies:
            rem._label(b)
        if args.view:
            rem.run()
        else:
            print(rem.write())
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    main()
