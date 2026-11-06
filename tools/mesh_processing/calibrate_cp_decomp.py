#!/usr/bin/env python3
"""
Step 2 -- choose a convex decomposition per body, and write it out.

    python calibrate_cp_decomp.py path/to/robot_wt

        ->  path/to/robot_wt_cp/          the hulls, one directory per body
            path/to/robot_wt_cp_world/
            path/to/robot_wt_cp_fused/    their union, one solid per body
            path/to/robot_wt_cp_fused_world/

Step 2 of five. Each step appends its own suffix to whatever its input carried, so a directory's
name is the record of how it was made:

    robot.xml -> _wt -> _wt_cp_fused -> _wt_cp_fused_im -> ..._im_man -> ..._man_sdata

`mjcf.model_stem` strips the lot to recover `robot`, which is how each step finds
the MJCF its world-frame poses need. There is no output argument anywhere;
`run_pipeline.py` runs the whole chain from the XML.

Two things happen before the window opens.

**PaMO decimates.** The watertight solids from step 1 carry half a million
vertices between them, and CoACD's cost scales with its input. PaMO takes them to
a few thousand while keeping every body watertight and intersection-free -- see
`pamo_driver.py` for why quadric collapse cannot. Its output is cached in `.tmp/`, so a
second run of this script skips straight to the window.

**CoACD decomposes** every body once at `--threshold`, so the window opens on a
decomposition.

Then the panel: a threshold slider and a hull cap per body, the hulls drawn over
the mesh they came from, and `foc` to look at one body alone. What a threshold
does to a knuckle ring is not what it does to a fingertip -- 5 hulls against 73 at
the same setting -- so each body has a slider of its own.

Recomputation happens in worker *processes*: CoACD is an in-process C++ extension
that holds the GIL, so a thread would freeze the window for the 0.5 to 8 s of each
run. Slider motion is debounced and coalesced to one job per body, or a drag would
queue hundreds.

Closing the window writes all four directories. `Export` writes them early without
closing. `_fused` is the boolean union of each body's hulls -- one watertight solid
per body, which is what a bake consumes; the hulls themselves are kept because the
union cannot be undone.

**Without the window.** `--choices FILE` starts each body from the threshold and hull
cap a previous run chose -- that run's `calibrate_cp_decomp.json`, in `_cp/` -- and
`--no-view` decomposes at those settings, writes, and exits. CoACD is not reproducible
at a fixed seed, so this recovers a decomposition's *settings*, not its hulls: the
counts come back within a few percent.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mujoco
import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pamo_driver  # noqa: E402
from mjcf import body_poses, model_stem, step_dir  # noqa: E402
from mesh_viewer import Body, MeshViewer, body_color  # noqa: E402


def _decompose(vertices, faces, threshold, max_hulls, resolution, seed):
    """Runs in a worker process. Returns a list of (vertices, faces).

    The clamp keeps the worker alive. CoACD rejects a threshold outside
    [0.01, 1] by *throwing from C++*, which does not raise in Python -- it calls
    `terminate` and takes the worker down, and one dead worker breaks the whole
    `ProcessPoolExecutor` permanently. The window then cannot recompute anything
    and the run is lost, several minutes of PaMO decimation with it. So a value
    the slider can produce must never reach `run_coacd` unclamped.
    """
    import coacd

    coacd.set_log_level("error")
    parts = coacd.run_coacd(
        coacd.Mesh(vertices, faces), threshold=min(max(threshold, 0.01), 1.0),
        max_convex_hull=max_hulls,
        resolution=resolution, seed=seed,
    )
    return [(np.asarray(v), np.asarray(f)) for v, f in parts]


def fuse(hulls, dilate=1.0005):
    """
    The boolean union of one body's hulls.

    Dilated first. CoACD's hulls *partition* the solid:
    they meet face to face rather than overlapping, so their shared interfaces have
    zero thickness and an exact boolean sees disjoint pieces -- unioned as they are,
    most bodies come back as several components with the cut interfaces still inside
    them, and the union volume equals the sum of the hull volumes exactly, which is
    the giveaway that nothing merged. Scaling each hull about its own centroid by
    five parts in ten thousand turns contact into overlap; it costs +0.1% by volume
    and takes every body to a single genus-0 shell.
    """
    if dilate and dilate != 1.0:
        scaled = []
        for h in hulls:
            g = h.copy()
            c = g.centroid
            g.vertices = (g.vertices - c) * dilate + c
            scaled.append(g)
        hulls = scaled
    return hulls[0].copy() if len(hulls) == 1 else trimesh.boolean.union(hulls)


class Calibrator(MeshViewer):
    """`MeshViewer` plus a CoACD job per body and the widgets that drive it."""

    def __init__(self, bodies, sources, pool, args, out_stem):
        super().__init__(bodies, ["mesh", "hulls"], title="Convex decomposition",
                         subtitle=f"{len(bodies)} bodies from {args.input.name}",
                         default_on=["mesh", "hulls"])
        self.sources = sources          # {name: the mesh CoACD runs on}
        self.pool = pool
        self.args = args
        self.out_stem = out_stem
        for b in self.bodies:
            b.threshold = args.threshold
            b.max_hulls = args.max_hulls
            b.applied = (args.threshold, args.max_hulls)
            b.job_dirty = False
            b.touched = time.monotonic()
            b.future = None
            b.seconds = None

    #######################################################################################
    # Work
    #######################################################################################

    def on_frame(self):
        now = time.monotonic()
        for b in self.bodies:
            if b.future is not None and b.future.done():
                try:
                    parts = b.future.result()
                    b.variants["hulls"] = [trimesh.Trimesh(v, f, process=False) for v, f in parts]
                    b.seconds = now - b.started
                    b.applied = b.submitted
                except Exception as exc:
                    print(f"[{b.name}] {type(exc).__name__}: {exc}")
                    b.variants["hulls"] = []
                b.future = None
                b.job_dirty = (b.threshold, b.max_hulls) != b.applied
                b.dirty = True
                self._label(b)

        in_flight = sum(1 for b in self.bodies if b.future is not None)
        for b in self.bodies:
            if b.future is not None or not b.job_dirty or in_flight >= self.args.workers:
                continue
            if now - b.touched < self.args.debounce:
                continue
            b.started, b.submitted = now, (b.threshold, b.max_hulls)
            b.future = self.pool.submit(
                _decompose, self.sources[b.name].vertices, self.sources[b.name].faces,
                b.threshold, b.max_hulls, self.args.resolution, self.args.seed,
            )
            b.job_dirty = False
            in_flight += 1

    def _label(self, b):
        n = len(b.variants.get("hulls") or [])
        state = ("working..." if b.future is not None
                 else "pending" if b.job_dirty
                 else f"{b.seconds:.1f}s" if b.seconds is not None else "")
        b.info = f"{n:4d} hulls  {state}"

    #######################################################################################
    # Panel
    #######################################################################################

    def extra_header(self, imgui):
        total = sum(len(b.variants.get("hulls") or []) for b in self.bodies)
        imgui.text(f"{total} convex primitives   ->  {self.out_stem.name}_cp/ and _cp_fused/")
        if imgui.button("Export now"):
            self.message = self.write()
            print(self.message)
        if self.message:
            imgui.text_wrapped(self.message)

    def extra_row(self, imgui, i, b):
        imgui.set_next_item_width(140)
        # 0.01, not lower: CoACD's own valid range is [0.01, 1] and it enforces
        # that by throwing from C++, which kills the worker rather than raising.
        changed, b.threshold = imgui.slider_float(f"##thr{i}", b.threshold, 0.01, 1.0, "%.3f")
        if changed:
            b.job_dirty, b.touched = True, time.monotonic()
            self._label(b)
        imgui.same_line()
        imgui.set_next_item_width(55)
        changed, value = imgui.input_int(f"##cap{i}", b.max_hulls, 0, 0)
        if changed:
            b.max_hulls = max(-1, value)
            b.job_dirty, b.touched = True, time.monotonic()
            self._label(b)
        return True

    #######################################################################################
    # Output
    #######################################################################################

    def write(self):
        """
        All four directories: the hulls and their union, each in body and world frame.

        The hulls are written as well as the union because the union cannot be undone,
        and because CoACD is not reproducible at a fixed seed -- three runs of one input
        gave 74, 76 and 78 hulls -- so re-running from a saved threshold does not
        recover the decomposition that was approved by eye. The meshes are the record.
        """
        cp = self.out_stem.parent / f"{self.out_stem.name}_cp"
        cpw = self.out_stem.parent / f"{self.out_stem.name}_cp_world"
        fused = self.out_stem.parent / f"{self.out_stem.name}_cp_fused"
        fusedw = self.out_stem.parent / f"{self.out_stem.name}_cp_fused_world"
        for d in (cp, cpw, fused, fusedw):
            d.mkdir(parents=True, exist_ok=True)

        rows, skipped = [], []
        for b in self.bodies:
            hulls = b.variants.get("hulls") or []
            if not hulls:
                skipped.append(b.name)
                continue

            for root, world in ((cp, False), (cpw, True)):
                d = root / b.name
                if d.exists():
                    shutil.rmtree(d)
                d.mkdir(parents=True)
                for i, h in enumerate(hulls):
                    mesh = h
                    if world:
                        mesh = h.copy()
                        mesh.vertices = b.posed(mesh.vertices)
                    mesh.export(d / f"hull_{i:03d}.obj")

            u = fuse(hulls, self.args.dilate)
            u.export(fused / f"{b.name}.obj")
            uw = u.copy()
            uw.vertices = b.posed(uw.vertices)
            uw.export(fusedw / f"{b.name}.obj")

            applied = b.applied or (b.threshold, b.max_hulls)
            rows.append(dict(
                body=b.name, threshold=round(applied[0], 4), max_hulls=applied[1],
                hulls=len(hulls), hull_vertices=int(sum(len(h.vertices) for h in hulls)),
                fused_vertices=int(len(u.vertices)),
                fused_watertight=bool(u.is_watertight),
                fused_shells=int(len(u.split(only_watertight=False))),
                fused_volume_cm3=round(float(u.volume) * 1e6, 3),
                source_volume_cm3=round(float(self.sources[b.name].volume) * 1e6, 3),
                stale=bool(b.job_dirty or b.future is not None),
            ))
        (cp / "calibrate_cp_decomp.json").write_text(json.dumps(rows, indent=2))

        bad = [r["body"] for r in rows if not r["fused_watertight"] or r["fused_shells"] != 1]
        stale = [r for r in rows if r["stale"]]
        return (f"{len(rows)} bodies, {sum(r['hulls'] for r in rows)} hulls -> "
                f"{cp.name}/ {cpw.name}/ {fused.name}/ {fusedw.name}/"
                + (f"; {len(bad)} fused not closed" if bad else "")
                + (f"; {len(stale)} stale" if stale else "")
                + (f"; skipped {len(skipped)}" if skipped else ""))

    def on_close(self):
        print("\n" + self.write())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="a directory of per-body OBJs, e.g. robot_wt/")
    ap.add_argument("--xml", type=Path, default=None,
                    help="the MJCF the bodies come from, for the world-frame poses; "
                         "default: <input with every pipeline suffix stripped>.xml beside it")
    ap.add_argument("--threshold", type=float, default=0.05, help="CoACD concavity tolerance")
    ap.add_argument("--max-hulls", type=int, default=-1, help="-1 for unlimited")
    ap.add_argument("--resolution", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dilate", type=float, default=1.0005, help="hull scaling before the union")
    ap.add_argument("--spacing", type=float, default=5.0e-3, help="PaMO vertex spacing [m]")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--debounce", type=float, default=0.35)
    ap.add_argument("--redo-pamo", action="store_true", help="ignore the cached decimation")
    ap.add_argument("--choices", type=Path, default=None,
                    help="a previous run's calibrate_cp_decomp.json: each body's threshold and hull cap")
    ap.add_argument("--no-view", dest="view", action="store_false",
                    help="decompose at the settings given, write, and exit without the window")
    args = ap.parse_args()

    out_stem = step_dir(args.input, "")
    xml = args.xml or args.input.parent / f"{model_stem(args.input.name)}.xml"
    if not xml.exists():
        raise SystemExit(f"no MJCF at {xml} -- pass --xml")

    meshes = {p.stem: trimesh.load_mesh(p, process=False)
              for p in sorted(args.input.glob("*.obj"))}
    if not meshes:
        raise SystemExit(f"no .obj files in {args.input}")
    print(f"{args.input.name}: {len(meshes)} bodies, "
          f"{sum(len(m.vertices) for m in meshes.values()):,d} vertices")

    cache = HERE / ".tmp" / f"{args.input.name}_pamo"
    if args.redo_pamo and cache.exists():
        shutil.rmtree(cache)
    cached = {p.stem: p for p in cache.glob("*.obj")} if cache.exists() else {}
    if set(cached) >= set(meshes):
        print(f"\nreusing the cached PaMO decimation in {cache.relative_to(HERE)}/"
              f"  (--redo-pamo to rebuild)")
        decimated = {n: trimesh.load_mesh(cached[n], process=False) for n in meshes}
    else:
        print(f"\nPaMO -> {cache.relative_to(HERE)}/")
        decimated = pamo_driver.run_batch(meshes, cache, spacing=args.spacing)
        if not decimated:
            raise SystemExit("PaMO produced nothing")
    print(f"  {sum(len(m.vertices) for m in decimated.values()):,d} vertices after decimation")

    poses = body_poses(mujoco.MjModel.from_xml_path(str(xml)))
    pool = ProcessPoolExecutor(max_workers=args.workers,
                               mp_context=multiprocessing.get_context("spawn"))
    bodies = [
        Body(name, {"mesh": decimated[name], "hulls": []}, body_color(i), poses.get(name))
        for i, name in enumerate(decimated)
    ]

    cal = Calibrator(bodies, decimated, pool, args, out_stem)
    chosen = {}
    if args.choices is not None:
        chosen = {r["body"]: r for r in json.loads(args.choices.read_text())}
    for b in cal.bodies:
        if b.name in chosen:
            b.threshold, b.max_hulls = float(chosen[b.name]["threshold"]), int(chosen[b.name]["max_hulls"])
        b.job_dirty, b.touched = True, 0.0
    print(f"\nCoACD at threshold {args.threshold}"
          + (f", {len(chosen)} bodies at their own from {args.choices.name}" if chosen else "") + " ...")
    try:
        while any(b.job_dirty or b.future is not None for b in cal.bodies):
            cal.on_frame()
            time.sleep(0.05)
        for b in cal.bodies:
            cal._label(b)
        print(f"  {sum(len(b.variants['hulls']) for b in cal.bodies)} hulls\n")
        if args.view:
            cal.run()
        else:
            print(cal.write())
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    main()
