"""
Times a variant of a scene's tuned settings -- its tuning file's, with other solver
options -- as `tune.py` times them: every capacity recalibrated at each
batch size, from the scene family's tuning batch up 2x, 4x, 8x and 16x while throughput
grows by more than 5 %, then the artifact guard at the guard's batch. The result goes to
`--out`; the tuning file is left alone.

    python tools/performance_tuning/time_variant.py arm_5_sdf [SCENE ... | all] \\
        [--kw mode=mj] [--label lodos_mj] [--out results/newton/ours]

The default is `lodos_mj`: the mj mode, MuJoCo's soft problem on Lodos's contacts solved
by MJWarp's Newton solver, without the proximal loop -- what Newton's MuJoCo Warp solves on
Newton's contacts (`viewer/newton/`). The tuned proximal budget is dropped, since it is
the loop the variant leaves out.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibrate as CAL  # noqa: E402
import common as C  # noqa: E402
import tune as T  # noqa: E402

from lodos.utils import states as lstates  # noqa: E402
from lodos.utils import tuning as ltuning  # noqa: E402
from viewer.scenes import catalog as vscenes  # noqa: E402

PROX_KNOBS = ("prox_iters_per_qp", "prox_tol_z", "prox_tol_sigma", "prox_rho", "refresh_de_saxce_per_prox")


def parse_kw(items):
    """`key=value` pairs, each value a Python literal or else a string."""
    out = {}
    for item in items:
        k, v = item.split("=", 1)
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v
    return out


def time_variant(name, kw, label, tuned_dir, states_dir, safety=1.5, log=C.log):
    t0 = time.time()
    src = C.scene(name)
    _, B = T.family(name)
    tuned = ltuning.load(Path(tuned_dir) / f"{name}.tuning.json")
    sset = lstates.load(Path(states_dir) / f"{name}.states.npz", src.mjm)
    rest = C.settled(sset)
    sec = T.section_without_peaks(tuned)
    sec.pop("B", None)
    sec["solver_kwargs"] = {k: v for k, v in sec.get("solver_kwargs", {}).items() if k not in PROX_KNOBS} | kw
    res = dict(scene=name, label=label, solver_kwargs=sec["solver_kwargs"], sim=sec.get("sim", {}),
               tuned=dict(B=tuned.get("B"), world_steps_per_s=(tuned.get("throughput") or {}).get("world_steps_per_s"),
                          artifacts=(tuned.get("artifacts") or {}).get("tuned")),
               batch=[], best=None)
    best = None
    for mult in (1, 2, 4, 8, 16):
        Bm = B * mult
        cal, err = T.guarded(CAL.calibrate_section, src, sset, Bm, safety=safety, keep=sec, log=lambda *x: None)
        r = thr = None
        if cal is not None:
            secm = T.section_without_peaks(cal[0])
            r, err = T.measure(src, Bm, sset, T.timing_batch(sset, Bm), secm, art=False)
            thr = r["throughput"]["world_steps_per_s"] if r else None
        res["batch"].append(dict(B=Bm, world_steps_per_s=thr, ms_per_step=r["throughput"]["ms_per_step"] if r else None,
                                 error=err))
        log(f"{name}: {label}: B={Bm}: {'-' if thr is None else f'{thr:.4g} world-steps/s'} {err or ''}")
        if thr is None or (best is not None and thr < 1.05 * best[1]):
            break
        best = (Bm, thr, secm)
    if best is not None:
        res["best"] = dict(B=best[0], world_steps_per_s=best[1])
        art, err = T.guard(src, T.GUARD_B, sset, best[2], rest)
        if art is None:
            # Below the section's own batch its batch totals fall back to the scene's own,
            # which can ask for more than the card has: capacities measured at the guard's.
            cal, err2 = T.guarded(CAL.calibrate_section, src, sset, T.GUARD_B, safety=safety, keep=best[2],
                                  log=lambda *x: None)
            if cal is not None:
                art, err = T.guard(src, T.GUARD_B, sset, T.section_without_peaks(cal[0]), rest)
            else:
                err = f"{err}; at the guard's batch: {err2}"
        res["artifacts"] = None if art is None else C.public(art)
        res["artifacts_error"] = err
        log(f"{name}: {label}: best {best[1]:.4g} world-steps/s at B={best[0]}; "
            f"{T.fmt_art(art) if art is not None else err}")
    res["seconds"] = time.time() - t0
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenes", nargs="+", help="built-in scenes, or `all`")
    ap.add_argument("--kw", nargs="*", default=["mode=mj"], help="the variant's solver options, key=value")
    ap.add_argument("--label", default="lodos_mj")
    ap.add_argument("--out", default="results/newton/ours")
    ap.add_argument("--tuned-dir", default="tools/performance_tuning/tuned")
    ap.add_argument("--states-dir", default="results/tuning/states")
    ap.add_argument("--safety", type=float, default=1.5)
    ap.add_argument("--force", action="store_true", help="redo scenes that already have a result")
    a = ap.parse_args()
    names = list(vscenes.BUILTIN) if a.scenes == ["all"] else a.scenes
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    kw = parse_kw(a.kw)
    for name in names:
        path = out / f"{name}.{a.label}.json"
        if path.exists() and not a.force:
            continue
        res = time_variant(name, kw, a.label, a.tuned_dir, a.states_dir, a.safety)
        C.write_json(path, res)


if __name__ == "__main__":
    main()
