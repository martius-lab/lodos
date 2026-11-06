"""
Runs `tune.py` over every built-in viewer scene, one process per scene.

    python tools/performance_tuning/tune_all.py [--only NAME ...] [--skip-done] [--timeout 3600] [--mode mj]

Logs go to `<out-dir>/logs/<scene>.log`; a scene that fails or times out is listed at the end and the
rest go on. `--mode` tunes every scene under that mode (`tune.py --mode`), its files, state sets and
logs in `<out-dir>/<mode>/`.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def order(names):
    """The scenes in the order they are tuned: the worked example first, then by family."""
    first = ["sharpa_armadillo_sdf", "sharpa_armadillo_polytope"]
    return [n for n in first if n in names] + [n for n in names if n not in first]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--out-dir", default="tools/performance_tuning/tuned")
    ap.add_argument("--mode", help="tune under this mode (tune.py --mode); off by default, each scene's own")
    a = ap.parse_args()

    from viewer.scenes import catalog as vscenes

    names = order(list(vscenes.BUILTIN))
    if a.only:
        names = [n for n in names if n in a.only]
    out = Path(a.out_dir)
    done = out / a.mode if a.mode else out  # where tune.py writes a mode's files
    (done / "logs").mkdir(parents=True, exist_ok=True)
    failed = []
    t0 = time.time()
    for n in names:
        if a.skip_done and (done / f"{n}.tuning.json").exists() and (done / f"{n}.report.json").exists():
            continue
        cmd = [sys.executable, str(ROOT / "tools" / "performance_tuning" / "tune.py"), n, "--out-dir", str(out)]
        if a.mode:
            cmd += ["--mode", a.mode]
        print(f"[{time.time() - t0:7.0f} s] {n}", flush=True)
        with open(done / "logs" / f"{n}.log", "w") as log:
            try:
                r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=a.timeout, cwd=ROOT)
                if r.returncode:
                    failed.append((n, f"exit {r.returncode}"))
            except subprocess.TimeoutExpired:
                failed.append((n, "timeout"))
        tail = (done / "logs" / f"{n}.log").read_text().strip().splitlines()[-1:]
        print(f"           {tail[0] if tail else ''}", flush=True)
    print(f"done in {time.time() - t0:.0f} s; failed: {failed}", flush=True)


if __name__ == "__main__":
    main()
