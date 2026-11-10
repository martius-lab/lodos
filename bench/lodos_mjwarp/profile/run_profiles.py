"""
Every measurement of `docs/speedup_analysis.pdf`, one process at a time, into `results/lodos_mjwarp/profile/`
(`common.OUT`); a measurement whose file is there is skipped. From the repository root:

    .venv/bin/python bench/lodos_mjwarp/profile/run_profiles.py [--scenes ...] [--phases ...]

Phases, in order:

    prepare  the benchmark's own prepare, tune and ab phases for the scenes (`bench.lodos_mjwarp.run`), and starts
             for the largest batch size
    stages   every configuration at the batch size the benchmark found best (`report/best.json`):
             `lodos_stages.py`, `newton_stages.py`
    equal    Lodos (per contact, and ray where the scene has surfaces) at MuJoCo Warp's best batch size
    weld     MuJoCo Warp with the pairs of bodies fixed to the world left out, at its best batch size and twice it
             (`newton_stages.py --weld-filter`), on the scenes in `WELD`
    kernels  the collision stages' kernels at MuJoCo Warp's best batch size (`collision_kernels.py`)
    solve    the solve's kernels, the same way (`solve_kernels.py`)
    census   Newton's triangle pairs and contacts per pair of shapes (`newton_pairs.py`), and Lodos's hull pipeline's
             counters (`lodos_hybrid.py`)

Then `summary.py` prints the tables and writes `results.json`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from bench.lodos_mjwarp import common as C  # noqa: E402

OUT = C.OUT / "profile"
LODOS_PY = ROOT / ".venv" / "bin" / "python"
NEWTON_PY = C.NEWTON_PY
SCENES = ("nist_threading_m8", "arm_walled_5_sdf", "arm_walled_5_polytope")
WELD = ("nist_threading_m8",)  # where Newton's importer makes bodies fixed to the world collide with each other
PAIRS = {"nist_threading_m8": 256}  # newton_pairs.py's scenes and batch sizes
HYBRID = ("arm_walled_5_polytope",)  # lodos_hybrid.py's scenes
PHASES = ("prepare", "stages", "equal", "weld", "kernels", "solve", "census")


def best():
    return json.loads((ROOT / "bench" / "lodos_mjwarp" / "report" / "best.json").read_text())


def scene_info(scene):
    return json.loads((C.OUT / "scenes" / f"{scene}.json").read_text())


def variant(scene, cfg):
    p = C.OUT / "ab" / f"{scene}.{cfg}.json"
    return json.loads(p.read_text()).get("chosen", "default") if p.exists() else "default"


def caps(scene, cfg):
    """A configuration's capacities: the benchmark's tuned ones, or as a stage run of MuJoCo Warp grew them."""
    if cfg == "mjwarp":
        grown = OUT / "caps" / f"{scene}.mjwarp.json"
        if grown.exists():
            return grown
    return C.OUT / "caps" / f"{scene}.{cfg}.json"


def run(cmd, out, log):
    """`cmd` in a process of its own unless `out` exists; its output in `log`."""
    if out is not None and Path(out).exists():
        print(f"have {Path(out).name}", flush=True)
        return 0
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.run([str(c) for c in cmd], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    tail = Path(log).read_text().strip().splitlines()[-1:] or [""]
    print(f"{Path(log).stem}: rc={rc} in {time.time() - t0:.0f} s  {tail[0][:160]}", flush=True)
    return rc


def stages(scene, cfg, B, weld=False):
    tag = ".weldfilter" if weld else ""
    out = OUT / f"{scene}.{cfg}{tag}.B{B}.json"
    every = max(1, scene_info(scene)["steps"] // 200)
    if cfg == "mjwarp":
        cmd = [NEWTON_PY, HERE / "newton_stages.py", scene, out, "--sizes", caps(scene, cfg), "--B", B,
               "--every", every] + (["--weld-filter"] if weld else [])
    else:
        cmd = [LODOS_PY, HERE / "lodos_stages.py", scene, cfg.split("_", 1)[1], out, "--sizes", caps(scene, cfg),
               "--B", B, "--every", every, "--starts", C.OUT / "scenes" / f"{scene}.starts.npz",
               "--variant", variant(scene, cfg)]
    rc = run(cmd, out, out.with_suffix(".log"))
    if rc == 0 and cfg == "mjwarp" and not weld:
        d = json.loads(out.read_text())
        if d.get("attempts", 1) > 1:  # it grew MuJoCo Warp's capacities: every later run of the scene takes them
            p = OUT / "caps" / f"{scene}.mjwarp.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(dict(ok=True, sizes=d["sizes"], grown_at=B), indent=1))


def kernel_pass(script, prefix, scene, B):
    for mode in ("lodos", "newton"):
        out = OUT / f"{prefix}.{scene}.{mode}.B{B}.json"
        if mode == "lodos":
            cfg = "lodos_contact"
            cmd = [LODOS_PY, HERE / script, "lodos", scene, "contact", out, "--sizes", caps(scene, cfg), "--B", B,
                   "--starts", C.OUT / "scenes" / f"{scene}.starts.npz", "--variant", variant(scene, cfg)]
        else:
            cmd = [NEWTON_PY, HERE / script, "newton", scene, out, "--sizes", caps(scene, "mjwarp"), "--B", B]
            if prefix == "ksolve" and scene in WELD:
                cmd.append("--weld-filter")
        run(cmd, out, out.with_suffix(".log"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=SCENES)
    ap.add_argument("--phases", default=",".join(PHASES))
    a = ap.parse_args()
    phases = a.phases.split(",")
    OUT.mkdir(parents=True, exist_ok=True)
    b = best()
    if "prepare" in phases:
        subprocess.run([LODOS_PY, "-m", "bench.lodos_mjwarp.run", "--only", *a.scenes, "--phases",
                        "prepare,tune,ab"], cwd=ROOT, check=True)
        for s in a.scenes:
            n = max(v["B"] for v in b[s].values())
            subprocess.run([LODOS_PY, "-m", "bench.lodos_mjwarp.prepare", s, "--starts", str(n)], cwd=ROOT,
                           check=True)
    for s in a.scenes:
        cfgs = scene_info(s)["configs"]
        Bm = b[s]["mjwarp"]["B"]
        if "stages" in phases:
            for cfg in ["mjwarp"] + [c for c in cfgs if c.startswith("lodos")]:
                stages(s, cfg, b[s][cfg]["B"])
        if "equal" in phases:
            for cfg in ("lodos_contact", "lodos_ray"):
                if cfg in cfgs and b[s][cfg]["B"] != Bm:
                    stages(s, cfg, Bm)
        if "weld" in phases and s in WELD:
            for B in (Bm, 2 * Bm):
                stages(s, "mjwarp", B, weld=True)
        if "kernels" in phases:
            kernel_pass("collision_kernels.py", "kern", s, Bm)
        if "solve" in phases:
            kernel_pass("solve_kernels.py", "ksolve", s, Bm)
        if "census" in phases:
            if s in PAIRS:
                out = OUT / f"pairs.{s}.B{PAIRS[s]}.json"
                run([NEWTON_PY, HERE / "newton_pairs.py", s, "--sizes", caps(s, "mjwarp"), "--B", PAIRS[s],
                     "--json", out], out, out.with_suffix(".log"))
            if s in HYBRID:
                out = OUT / f"hybrid.{s}.B{Bm}.json"
                run([LODOS_PY, HERE / "lodos_hybrid.py", s, "--sizes", caps(s, "lodos_contact"), "--B", Bm,
                     "--starts", C.OUT / "scenes" / f"{s}.starts.npz", "--variant", variant(s, "lodos_contact"),
                     "--json", out], out, out.with_suffix(".log"))


if __name__ == "__main__":
    main()
