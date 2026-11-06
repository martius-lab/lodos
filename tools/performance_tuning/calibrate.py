"""
Sizes every capacity of a scene from a state set and writes them to its tuning file.

    python tools/performance_tuning/calibrate.py arm_5_sdf --states results/tuning/states/arm_5_sdf.states.npz \\
        --B 4096 [--layout independent|shared] [--horizon 200] \\
        [--controls hold|none|random] [--safety 1.5] [--out arm_5_sdf.tuning.json] [--update]

A pre-pass at a small batch (`--B-small`, the scene's own capacities and
the collision defaults, which hold at that size) gives the fractions to start the real
measurement from; the measurement at `--B` rolls every batch of the state set out
`--horizon` steps and reads every counter on the device after every step
(`lodos.utils.calibrate.Tracker`); whatever overflowed is grown and the measurement repeated
until nothing does, since a peak read behind an overflow is not the demand. The
capacities are then `--safety` times the peaks (`lodos.utils.calibrate.settings`).

`--update` keeps what the file already holds that this does not measure -- tile sizes,
solver budgets, throughput -- and replaces the capacities.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

from lodos.utils import calibrate as lcalibrate  # noqa: E402
from lodos.utils import states as lstates  # noqa: E402
from lodos.utils import tuning as ltuning  # noqa: E402

CAPACITY_KEYS = ltuning.CAPACITIES


def controls_of(name, sigma=0.5, seed=0):
    return lstates.Controls.random(sigma, seed=seed) if name == "random" else lstates.Controls(name)


def calibrate_section(src, sset, B, layout="independent", horizon=200, controls=None, safety=1.5,
                      B_small=64, keep=None, progress=None, log=C.log):
    """
    `(section, report, rounds)`: the capacities at `B`, the peaks they came from. `keep` is
    a section (or a tuning file) whose other settings (tile sizes, budgets) the build uses
    and the result keeps.
    """
    keep = {k: dict(v) for k, v in (keep or {}).items() if k in ltuning.SECTIONS}
    non_caps = {k: {n: v for n, v in sec.items() if n not in CAPACITY_KEYS.get(k, ())} for k, sec in keep.items()}
    # `capacity1` below 1.0 is also a choice of launch -- the tile filter over a compacted
    # list -- so a kept one is built with until the measurement sizes it.
    cap1 = keep.get("collision", {}).get("capacity1")

    def with_kept(sec):
        out = {k: dict(v) for k, v in sec.items()}
        for k, v in non_caps.items():
            out.setdefault(k, {}).update(v)
        if cap1 is not None and "capacity1" not in out.get("collision", {}):
            out.setdefault("collision", {})["capacity1"] = cap1
        return out

    def build(sec, b):
        return C.build(src, b, section=with_kept(sec))

    t0 = time.time()
    B_small = min(B_small, B)
    rng = np.random.default_rng(0)
    base = C.baseline_section(src)
    small_batches = max(1, min(8, len(sset) // B_small))
    _, rep_small, r_small = lcalibrate.calibrate(build, sset, B_small, base, layout, horizon, controls, safety,
                                           cover=False if layout == "independent" else True,
                                           max_batches=small_batches if layout == "independent" else None,
                                           rng=rng)
    log(f"  pre-pass at B={B_small}: {r_small} round(s), {time.time() - t0:.0f} s")
    if B == B_small:
        sec, rep, rounds = lcalibrate.settings(rep_small, safety), rep_small, r_small
    else:
        start = C.start_section(rep_small, B_small, B)
        sec, rep, rounds = lcalibrate.calibrate(build, sset, B, start, layout, horizon, controls, safety,
                                          progress=progress)
    log(f"  B={B}: {rounds} round(s), {time.time() - t0:.0f} s")
    sec = with_kept(sec)
    sec["B"] = int(B)
    return sec, rep, rounds


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene")
    ap.add_argument("--states", required=True)
    ap.add_argument("--B", type=int, default=4096)
    ap.add_argument("--B-small", type=int, default=64)
    ap.add_argument("--layout", default="independent", choices=["independent", "shared"])
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--controls", default="hold", choices=["hold", "none", "random", "sequence"])
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--safety", type=float, default=1.5)
    ap.add_argument("--out")
    ap.add_argument("--update", action="store_true")
    a = ap.parse_args()

    src = C.scene(a.scene)
    sset = lstates.load(a.states, src.mjm)
    out = Path(a.out or f"results/tuning/{C.scene_name(a.scene)}.tuning.json")
    old = ltuning.load(out) if a.update and out.exists() else None
    t = ltuning.new(C.scene_name(a.scene), a.B, a.layout, ltuning.scene_fingerprint(src.mjm, src.surfaces),
                    states=dict(path=str(a.states), digest=sset.digest(), size=len(sset), source=sset.source,
                                horizon=a.horizon, controls=controls_of(a.controls, a.sigma).describe()),
                    safety=a.safety)
    if old is not None:
        for k in ("baseline", "notes", "tiles", "throughput", "artifacts"):
            if k in old:
                t[k] = old[k]
    controls = controls_of(a.controls, a.sigma)
    C.log(f"{C.scene_name(a.scene)}: calibrating at B={a.B} over {len(sset)} states ({a.layout})")
    sec, rep, rounds = calibrate_section(src, sset, a.B, a.layout, a.horizon, controls, a.safety, a.B_small, old)
    t.update(sec, peaks=rep, rounds=rounds)
    ltuning.save(t, out)
    C.log(ltuning.describe(t))
    C.log(f"-> {out}")


if __name__ == "__main__":
    main()
