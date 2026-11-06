"""
The comparison's driver: every measurement a process of its own, none holding the GPU here.

    python -m bench.lodos_kamino.run [--only SCENE ...] [--phases prepare,tune,ab,ladder] [--jobs 3]

As `bench.lodos_mjwarp.run`, for Lodos's exact mode and Kamino (on the scenes without Lodos surfaces):

1. prepare: each scene's starts, and Kamino's package, options and controls (`prepare.py`).
2. tune: each configuration's capacities, a world's largest demand over `common.N_TUNE` starts, 1.5 times: Lodos's
   each start simulated alone for the whole run (`lodos_point.py measure`), Kamino's the starts as the worlds of one
   batch, each world's contacts counted (`kamino_point.py measure`); untimed, `--jobs` at once.
3. ab: the execution options against the default, timed from the half-way states of step 2 -- Lodos's shared
   Hessian and fused Newton solve, Kamino's sparse Jacobians and reordered factorization; an option is kept where it
   is 3 % faster (`lodos_point.py ab`, `kamino_point.py ab`).
4. ladder: Lodos's on every scene, then Kamino's; per scene, batch sizes from 64, doubled, each configuration in turn
   (the order reversed every other batch size), until its throughput drops or it runs out of memory. A run that overflows a capacity is discarded,
   the capacity doubled and the run redone. Each timed run waits for a GPU without other processes.

Everything lands in `results/lodos_kamino/` (`common.OUT`); a phase skips what it finds done, so a run cut short
resumes where it stopped.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bench.lodos_kamino import common as C
from bench.lodos_mjwarp.run import grow

PY = sys.executable
LOCK = threading.Lock()
TIMEOUT = 4 * 3600  # s, one process
D = C.OUT
MAX_B = 1 << 20  # the ladders' largest batch size; `--max-B` lowers it for a smoke test


def log(msg):
    line = f"{time.strftime('%m-%d %H:%M:%S')} {msg}"
    with LOCK:
        print(line, flush=True)
        with open(D / "run.log", "a") as f:
            f.write(line + "\n")


def run(cmd, name):
    """`cmd` in a process of its own, its output in `logs/name.log`; the return code, or None on a timeout."""
    (D / "logs").mkdir(parents=True, exist_ok=True)
    with open(D / "logs" / f"{name}.log", "w") as f:
        try:
            return subprocess.run(cmd, cwd=C.ROOT, stdout=f, stderr=subprocess.STDOUT, timeout=TIMEOUT).returncode
        except subprocess.TimeoutExpired:
            return None


def result(path, rc):
    p = Path(path)
    if p.exists():
        return C.read_json(p)
    return dict(ok=False, error="timed out" if rc is None else f"the process ended with {rc} and no result")


def others_on_gpu():
    """The processes on the GPU (none of ours runs while this is asked)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=60).stdout
    except Exception:  # noqa: BLE001
        return []
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def wait_for_gpu(what):
    t0 = time.time()
    while True:
        busy = others_on_gpu()
        if not busy:
            return 0.0
        if time.time() - t0 > 2 * 3600:
            log(f"  the GPU still has {busy} after 2 h; timing {what} anyway, marked shared")
            return -1.0
        if time.time() - t0 < 61:
            log(f"  waiting for the GPU before {what}: {busy}")
        time.sleep(60)


def scene_info(scene):
    return C.read_json(D / "scenes" / f"{scene}.json")


# -------------------------------------------------------------------------------------------------------------------


def prepare(scene, n=0, force=False):
    """The scene's files, its first `n` starts among them; `force`: the process asked even so, for what it adds."""
    path = D / "scenes" / f"{scene}.json"
    have = 0
    st = D / "scenes" / f"{scene}.starts.npz"
    if st.exists():
        import numpy as np

        have = len(np.load(st)["qpos"])
    if path.exists() and have >= n and not force:
        return
    rc = run([PY, "-m", "bench.lodos_kamino.prepare", scene, "--starts", str(max(n, have))], f"prepare.{scene}")
    if rc != 0 or not path.exists():
        raise RuntimeError(f"preparing {scene} failed ({rc}): see logs/prepare.{scene}.log")


def lodos_cmd(task, scene, groups, out, *extra):
    return [PY, "-m", "bench.lodos_kamino.lodos_point", task, scene, groups, str(out), *map(str, extra)]


def kamino_cmd(task, scene, starts, out, *extra):
    s = D / "scenes"
    ctl = s / f"{scene}.controls.npz"
    return [str(C.NEWTON_PY), str(C.ROOT / "bench" / "lodos_kamino" / "kamino_point.py"), task, str(s / f"{scene}.pkg"),
            str(s / f"{scene}.kamino.json"), str(starts), str(ctl) if ctl.exists() else "-", str(out),
            *map(str, extra)]


def cmd_of(task, scene, cfg, out, *extra, starts=None):
    """A measurement's command: Kamino's (`kamino_point.py`, its starts the ladder's or the capacities') or Lodos's."""
    if cfg == "kamino":
        return kamino_cmd(task, scene, starts or D / "scenes" / f"{scene}.tune.npz", out, *extra)
    return lodos_cmd(task, scene, cfg.split("_", 1)[1], out, *extra, *(["--starts", starts] if starts else []))


def caps_path(scene, cfg, grown=False):
    return D / "caps" / f"{scene}.{cfg}{'.grown' if grown else ''}.json"


def grown_sizes(cfg, sizes, over, B):
    """`sizes` with each capacity that overflowed at `B` worlds doubled: Kamino's contacts a world, or Lodos's."""
    if cfg != "kamino":
        return grow(cfg, sizes, over, B)
    return dict(sizes, contacts_per_world=2 * int(sizes["contacts_per_world"])) if "contacts" in over else sizes


def tune_one(scene, cfg):
    out = caps_path(scene, cfg)
    if out.exists() and C.read_json(out).get("ok"):
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    r = result(out, run(cmd_of("measure", scene, cfg, out), f"tune.{scene}.{cfg}"))
    log(f"tune {scene} {cfg}: " + (f"{r['sizes']} in {time.time() - t0:.0f} s" if r.get("ok")
                                   else f"FAILED {r.get('error')}"))


def ab_one(scene, cfg):
    out = D / "ab" / f"{scene}.{cfg}.json"
    if out.exists() and C.read_json(out).get("ok"):
        return C.read_json(out)
    if not caps_path(scene, cfg).exists() or not C.read_json(caps_path(scene, cfg)).get("ok"):
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    wait_for_gpu(f"ab {scene} {cfg}")
    r = result(out, run(cmd_of("ab", scene, cfg, out, "--sizes", caps_path(scene, cfg)), f"ab.{scene}.{cfg}"))
    if r.get("ok"):
        log(f"ab {scene} {cfg}: " + ", ".join(f"{k} {v['world_steps_per_s']:.4g}" if "world_steps_per_s" in v else
                                              f"{k} n/a" for k, v in r["variants"].items()) + f" -> {r['chosen']}")
    else:
        log(f"ab {scene} {cfg}: FAILED {r.get('error')}")
    return r


# -------------------------------------------------------------------------------------------------------------------


def point(scene, cfg, B, variant):
    """One timed run (`--B` worlds), redone with grown capacities while it overflows (three times at most)."""
    out = D / "points" / f"{scene}.{cfg}.B{B}.json"
    if out.exists() and C.read_json(out).get("ok"):
        return C.read_json(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prepare(scene, B)
    starts = D / "scenes" / f"{scene}.starts.npz"
    for attempt in range(4):
        grown = caps_path(scene, cfg, grown=True)
        sizes_file = grown if grown.exists() else caps_path(scene, cfg)
        shared = wait_for_gpu(f"{scene} {cfg} B={B}")
        cmd = cmd_of("point", scene, cfg, out, "--sizes", sizes_file, "--B", B, "--variant", variant, starts=starts)
        r = result(out, run(cmd, f"point.{scene}.{cfg}.B{B}"))
        r["gpu_shared"] = shared < 0
        C.write_json(out, r)
        if not (r.get("ok") and r.get("overflowed")):
            return r
        # discarded: the capacities grown for this and every larger batch, the run redone
        out.rename(out.with_suffix(f".overflowed{attempt}.json"))
        sizes = C.read_json(sizes_file)
        C.write_json(grown, dict(sizes, sizes=grown_sizes(cfg, sizes["sizes"], r["overflow"], B), grown_at=B))
        log(f"  {scene} {cfg} B={B} overflowed {r['overflow']}: grown, redone")
    return dict(ok=False, error="kept overflowing")


def ladder(scene, kamino):
    """The scene's ladder of Kamino (`kamino`) or of Lodos's configurations."""
    info = scene_info(scene)
    configs = [c for c in info["configs"] if (c == "kamino") == kamino]
    variants = {}
    for cfg in configs:
        ab = C.read_json(D / "ab" / f"{scene}.{cfg}.json") if (D / "ab" / f"{scene}.{cfg}.json").exists() else {}
        variants[cfg] = ab.get("chosen", "default") if ab.get("ok") else "default"
    active = [c for c in configs if caps_path(scene, c).exists() and C.read_json(caps_path(scene, c)).get("ok")]
    best, B, flip = {}, C.B0, False
    while active and B <= MAX_B:
        for cfg in (reversed(active) if flip else list(active)):
            r = point(scene, cfg, B, variants.get(cfg, "default"))
            if not r.get("ok"):
                e = str(r.get("error"))
                oom = any(s in e for s in ("Failed to allocate", "out of memory", "cudaErrorMemory"))
                big = "Array shapes must" in e  # Kamino's dense Delassus matrices past 2^31 entries in all
                why = "out of memory" if oom else "past Kamino's 2^31 entries" if big else "FAILED " + e[:200]
                log(f"{scene} {cfg} B={B}: {why}; its ladder ends")
                active.remove(cfg)
                continue
            v = r["world_steps_per_s"]
            log(f"{scene} {cfg} B={B}: {v:,.0f} world-steps/s ({r['sim_seconds_per_s']:.3g} sim-s/s), "
                f"{r['seconds']:.0f} s, non-finite {r['nan_worlds']}, escaped {r['escaped_worlds']}")
            if v < best.get(cfg, (0.0, 0))[0]:
                log(f"{scene} {cfg}: throughput dropped at B={B}; its ladder ends, best "
                    f"{best[cfg][0]:,.0f} at B={best[cfg][1]}")
                active.remove(cfg)
            else:
                best[cfg] = (v, B)
        B, flip = 2 * B, not flip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--phases", default="prepare,tune,ab,ladder")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--max-B", type=int, default=MAX_B)
    a = ap.parse_args()
    globals()["MAX_B"] = a.max_B
    D.mkdir(parents=True, exist_ok=True)
    scenes = [s for s in C.SCENES if not a.only or s in a.only]
    phases = a.phases.split(",")
    log(f"=== {a.phases} on {len(scenes)} scenes")
    if "prepare" in phases:
        for s in scenes:
            prepare(s, C.B0, force=True)
            log(f"prepared {s}: {scene_info(s)['configs']}")
    if "tune" in phases:
        # Untimed, so at once: the twenty-body scenes one at a time (at one world Lodos's arenas are a world's
        # worst case, the twenty hands' some 8 GB), the rest `--jobs` - 1 at a time beside them.
        jobs = [(s, c) for s in scenes for c in scene_info(s)["configs"]]
        heavy = [j for j in jobs if "_20_" in j[0]]
        light = [j for j in jobs if j not in heavy]
        with ThreadPoolExecutor(1) as hx, ThreadPoolExecutor(max(1, a.jobs - 1)) as lx:
            done = [hx.submit(tune_one, *j) for j in heavy] + [lx.submit(tune_one, *j) for j in light]
            for f in done:
                f.result()
    if "ab" in phases:
        for s in scenes:
            for c in scene_info(s)["configs"]:
                ab_one(s, c)
    if "ladder" in phases:
        for kamino in (False, True):  # every scene's Lodos ladders, then Kamino's
            for s in scenes:
                ladder(s, kamino)
    log("=== done")


if __name__ == "__main__":
    main()
