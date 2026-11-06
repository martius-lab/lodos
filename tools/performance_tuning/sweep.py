"""
Times any knob, or several, over a grid of values, from a tuning file's settings.

    python tools/performance_tuning/sweep.py arm_5_sdf --B 4096 \\
        --tuning tools/performance_tuning/tuned/arm_5_sdf.tuning.json \\
        --knob solver_kwargs.prox_iters_per_qp=10,5,3 --knob sim.contact_reduction=false,true \\
        [--states S.npz] [--rounds 3] [--guard] [--recalibrate] [--json out.json]

A knob is `<section>.<name>` in one of a tuning file's sections (`collision`, `runtime`,
`sim`, `solver_kwargs`); values are parsed as JSON where they can be (`8`, `0.1`, `true`,
`{"default": 8}`). Every configuration of the grid is timed once per round, the rounds
interleaved -- A B C, A B C, ... -- so that a drift in the GPU's clock or temperature falls on
all of them alike; the table gives each configuration's median.

`--guard` also runs the artifact metrics (`common.artifacts`) and holds each configuration
to the first one's (`common.no_worse`). `--recalibrate` sizes the capacities again for
each configuration -- needed for a knob that changes the demand (a budget, a tile size).
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibrate as CAL  # noqa: E402
import common as C  # noqa: E402
import tune as T  # noqa: E402

from lodos.utils import states as lstates  # noqa: E402
from lodos.utils import tuning as ltuning  # noqa: E402


def parse_value(v):
    try:
        return json.loads(v)
    except ValueError:
        return v


def parse_knob(spec):
    name, values = spec.split("=", 1)
    sec, key = name.split(".", 1)
    if sec not in ltuning.SECTIONS:
        raise SystemExit(f"{name}: the section must be one of {ltuning.SECTIONS}")
    return sec, key, [parse_value(v) for v in values.split(",")]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene")
    ap.add_argument("--B", type=int, default=1024)
    ap.add_argument("--tuning")
    ap.add_argument("--states")
    ap.add_argument("--knob", action="append", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--guard", action="store_true")
    ap.add_argument("--recalibrate", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args()

    src = C.scene(a.scene)
    sset = (lstates.load(a.states, src.mjm) if a.states
            else lstates.from_qpos(src.mjm, src.qpos0[None], qvel=src.qvel0[None], ctrl=src.ctrl))
    base = {k: {} for k in ltuning.SECTIONS}
    if a.tuning:
        t = ltuning.load(a.tuning)
        base = {k: dict(t.get(k, {})) for k in ltuning.SECTIONS}
        base["B"] = t["B"]
    knobs = [parse_knob(k) for k in a.knob]
    grid = list(itertools.product(*[v for _, _, v in knobs]))
    configs = []
    for values in grid:
        sec = copy.deepcopy(base)
        for (s, k, _), v in zip(knobs, values):
            sec[s][k] = v
        if a.recalibrate:
            sec, _, _ = CAL.calibrate_section(src, sset, a.B, keep=sec, log=lambda *x: None)
            sec = T.section_without_peaks(sec)
        configs.append((dict(zip([f"{s}.{k}" for s, k, _ in knobs], values)), sec))
    idx = T.timing_batch(sset, a.B)
    times = {i: [] for i in range(len(configs))}
    errors = {}
    for r in range(a.rounds):
        for i, (label, sec) in enumerate(configs):
            if i in errors:
                continue
            res, err = T.measure(src, a.B, sset, idx, sec, steps=a.steps, art=False)
            if res is None:
                errors[i] = err
                continue
            times[i].append(res["throughput"]["world_steps_per_s"])
        C.log(f"round {r + 1}/{a.rounds} done")
    arts = {}
    if a.guard:
        rest = C.settled(sset)
        for i, (label, sec) in enumerate(configs):
            if i in errors:
                continue
            res, err = T.measure(src, a.B, sset, idx, sec, rest=rest, steps=10)
            arts[i] = res["artifacts"] if res else None
    rows = []
    ref = arts.get(0)
    for i, (label, sec) in enumerate(configs):
        med = statistics.median(times[i]) if times[i] else None
        row = dict(knobs=label, world_steps_per_s=med, spread=(max(times[i]) - min(times[i])) / med if med else None,
                   error=errors.get(i))
        if a.guard and arts.get(i) is not None and ref is not None:
            ok, why = C.no_worse(arts[i], ref)
            row.update(artifacts=C.public(arts[i]), ok=ok, why=why)
        rows.append(row)
    best = max((r["world_steps_per_s"] or 0) for r in rows) or 1.0
    for r in rows:
        thr = r["world_steps_per_s"]
        mark = "" if not a.guard or r.get("ok", True) else f"  ARTIFACTS: {r.get('why')}"
        print(f"  {json.dumps(r['knobs'], default=str):<60} "
              + (f"{thr:10.4g} world-steps/s ({thr / best:5.1%})" if thr else f"  {r['error']}") + mark)
    if a.json:
        C.write_json(a.json, dict(scene=C.scene_name(a.scene), B=a.B, rows=rows))


if __name__ == "__main__":
    main()
