#!/usr/bin/env python3
"""
The whole chain, from an MJCF to baked surfaces.

    python run_pipeline.py path/to/robot.xml

Five steps, each its own script, each reading what the last one wrote:

    1  fuse_meshes_by_body.py       robot.xml               ->  robot_wt
    2  calibrate_cp_decomp.py       robot_wt                ->  robot_wt_cp_fused
    3  interactive_remesh.py        robot_wt_cp_fused       ->  ..._im
    4  fix_nonmanifold_meshes.py    ..._im                  ->  ..._im_man
    5  create_surface_data.py       ..._im_man              ->  ..._im_man_sdata

Nothing is passed between them but a path, because a step's output name is a pure
function of its input name (`mjcf.step_dir`), so this driver is only spelling out a
chain the scripts already agree on. Run any of them by hand at any point and the
next one still finds its input.

**Steps 2 and 3 open a window and wait for you.** They are calibration, not
computation: what a CoACD threshold does to a knuckle ring is not what it does to a
fingertip, and the same is true of a vertex budget. Step 3 writes only when you
press `Export`, so this checks for its output and stops with an explanation rather
than running step 4 on a directory that was never written.

**Or replay a run that was calibrated.** `--replay DIR` runs every step without its
window, steps 2 and 3 from the choices a previous run recorded in `DIR` --
`calibrate_cp_decomp.json`'s threshold and hull cap per body, `interactive_remesh.json`'s
vertex target -- such as a previous run's `_sdata/`, where step 5 gathers them.
Instant Meshes is deterministic, so the remesh comes back the same; CoACD is not, so
the decomposition comes back with the same settings and about the same hull counts.
`--no-view` alone runs steps 2 and 3 at their defaults.

Each step runs as a subprocess rather than an import. They are separate programs
with their own argument parsing, and two of them build a GL context and a Newton
viewer that does not expect to be created twice in one interpreter; a crash in one
step also then leaves the others runnable.

`--from`/`--to` run part of the chain, by step number, which is what you want after
changing your mind about a threshold four steps back. `--dry-run` prints the
commands and the paths without running anything.

No other step options are forwarded through here, and there is nowhere to put them:
the five have almost disjoint flags, so anything general enough to pass to all of
them would abort four. To change a threshold, an alpha or a vertex budget, run that
one script by hand -- the naming convention means the next step still finds it --
and then `--from` the one after.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from mjcf import step_dir  # noqa: E402


# (number, script, suffix it appends, whether it waits for you, whether it takes --no-view)
STEPS = [
    (1, "fuse_meshes_by_body.py", "_wt", False, True),
    (2, "calibrate_cp_decomp.py", "_cp_fused", True, True),
    (3, "interactive_remesh.py", "_im", True, True),
    (4, "fix_nonmanifold_meshes.py", "_man", False, True),
    (5, "create_surface_data.py", "_sdata", False, True),
]
# The records a step replays from -- this pipeline's name first, then the one Lodos-v1's
# pipeline gave it (`viewer/assets/sharpa/final_sdata/` has those), whose rows have the
# same fields -- and the flag that hands one over.
REPLAY = {2: (("calibrate_cp_decomp.json", "decompose.json"), "--choices"),
          3: (("interactive_remesh.json", "instant_meshes.json"), "--targets")}


def plan(xml):
    """`[(n, script, input, output, interactive, has_no_view)]` for the whole chain."""
    out = []
    current = Path(xml)
    for n, script, suffix, waits, no_view in STEPS:
        nxt = step_dir(current, suffix)
        out.append((n, script, current, nxt, waits, no_view))
        current = nxt
    return out


def run(step, args):
    """One step. Returns True if it produced its output directory."""
    n, script, src, dst, waits, has_no_view = step

    cmd = [sys.executable, str(HERE / script), str(src)]
    if args.no_view and has_no_view:
        cmd.append("--no-view")
        waits = False
    if args.replay is not None and n in REPLAY:
        names, flag = REPLAY[n]
        found = [args.replay / name for name in names if (args.replay / name).exists()]
        if found:
            cmd += [flag, str(found[0])]
        else:
            print(f"  (no {' or '.join(names)} in {args.replay}: step {n} runs at its defaults)")

    print(f"\n{'=' * 78}\n[{n}/{len(STEPS)}] {script}   {src.name} -> {dst.name}"
          + ("   (opens a window, waits for you)" if waits else "")
          + f"\n{'=' * 78}", flush=True)
    if args.dry_run:
        print("  " + " ".join(cmd))
        return True

    started = time.monotonic()
    code = subprocess.run(cmd).returncode
    elapsed = time.monotonic() - started

    if code != 0:
        print(f"\n{script} exited {code} after {elapsed:.0f}s -- stopping.")
        return False
    if not dst.is_dir() or not (any(dst.glob("*.obj")) or any(dst.glob("*.npz"))):
        print(f"\n{script} exited cleanly but wrote no {dst.name}/.")
        if script == "interactive_remesh.py":
            print("  That step writes only when you press `Export`; closing the window\n"
                  "  discards the result. Re-run it and export before closing:\n"
                  f"      python {script} {src}")
        return False
    print(f"\n  {dst.name}/ written in {elapsed:.0f}s")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("xml", type=Path, help="the MJCF to start from")
    ap.add_argument("--from", dest="first", type=int, default=1, metavar="N",
                    help="first step to run (default 1)")
    ap.add_argument("--to", dest="last", type=int, default=len(STEPS), metavar="N",
                    help="last step to run (default %d)" % len(STEPS))
    ap.add_argument("--no-view", action="store_true",
                    help="no windows: steps 2 and 3 run at their defaults (or, with --replay, "
                         "at a previous run's choices)")
    ap.add_argument("--replay", type=Path, default=None, metavar="DIR",
                    help="replay steps 2 and 3 from the records in DIR (a previous run's _sdata/); "
                         "implies --no-view")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands and the paths, run nothing")
    args = ap.parse_args()
    if args.replay is not None:
        args.no_view = True

    if not args.xml.exists():
        raise SystemExit(f"no such file: {args.xml}")
    if not 1 <= args.first <= args.last <= len(STEPS):
        raise SystemExit(f"--from/--to must satisfy 1 <= from <= to <= {len(STEPS)}")

    steps = plan(args.xml)
    print(f"{args.xml.name}  ->  {steps[-1][3].name}/")
    for n, script, src, dst, waits, _ in steps:
        mark = " " if args.first <= n <= args.last else "-"
        print(f"  {mark} {n}  {script:28s} {src.name:38s} -> {dst.name}"
              + ("   [waits]" if waits else ""))

    for step in steps:
        if not args.first <= step[0] <= args.last:
            continue
        if not run(step, args):
            raise SystemExit(1)

    if not args.dry_run:
        print(f"\n{'=' * 78}\ndone: {steps[args.last - 1][3]}")


if __name__ == "__main__":
    main()
