"""
The tables of `docs/speedup_analysis.pdf`, from `run_profiles.py`'s measurements: per scene, each configuration's
time per world-step by category (the step's GPU time over B), the speed-up's decomposition, the workload, the
collision and solve kernels and the censuses. Prints them, and writes them to `results.json` beside this file, the
record the write-up quotes. No GPU.

    .venv/bin/python bench/lodos_mjwarp/profile/summary.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from bench.lodos_mjwarp import common as C  # noqa: E402

OUT = C.OUT / "profile"
SCENES = ("nist_threading_m8", "arm_walled_5_sdf", "arm_walled_5_polytope")

# category -> the stages (paths under "step/") whose times it sums, per simulator
LODOS = {
    "collision": ["prim_poses", "prim_collision", "lodos_collision", "contact_dynamics/reduce"],
    "rigid body": ["rbd", "integrate"],
    "assembly": ["contact_dynamics/make_constraint", "contact_dynamics/(self)", "contact_dynamics/groups_before",
                 "contact_dynamics/groups_export", "contact_dynamics/rowspace_prepare"],
    "solve": ["contact_dynamics/solve"],
    "other": ["track", "(self)"],
}
NEWTON = {
    "collision": ["collide"],
    "rigid body": ["clear_forces", "assign", "solver_step/control", "solver_step/state_in", "solver_step/state_out",
                   "solver_step/mjw_step/forward/position/kinematics", "solver_step/mjw_step/forward/position/crb",
                   "solver_step/mjw_step/forward/position/transmission", "solver_step/mjw_step/forward/position/(self)",
                   "solver_step/mjw_step/forward/velocity", "solver_step/mjw_step/forward/actuation",
                   "solver_step/mjw_step/forward/acceleration", "solver_step/mjw_step/integrate"],
    "assembly": ["solver_step/convert_contacts", "solver_step/mjw_step/forward/position/make_constraint"],
    "solve": ["solver_step/mjw_step/forward/solve"],
    "other": ["(self)", "solver_step/(self)", "solver_step/mjw_step/(self)", "solver_step/mjw_step/forward/(self)"],
}
CATS = list(LODOS)
COLLISION = {  # the collision stages, per simulator
    "lodos": ["prim_poses", "prim_collision/broadphase", "prim_collision/hybrid", "prim_collision/primitive_narrow",
              "prim_collision/convex_narrow", "prim_collision/(self)", "lodos_collision/bbox",
              "lodos_collision/sdf_prefilter", "lodos_collision/narrow", "lodos_collision/(self)",
              "contact_dynamics/reduce"],
    "mjwarp": ["collide/(self)", "collide/broadphase", "collide/narrowphase"],
}


def load(name):
    p = OUT / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def us(d, path):
    """A stage's microseconds per world-step (its self time for a path ending in "(self)")."""
    t = d["traced"]
    ms = t["self_ms"] if path.endswith("(self)") else t["ms"]
    return 1e3 * ms.get("step/" + path, 0.0) / d["B"]


def entry(d):
    """One run's record: throughput, time per world-step by category and collision stage, workload."""
    table = LODOS if d["sim"].startswith("lodos") else NEWTON
    cats = {c: sum(us(d, p) for p in paths) for c, paths in table.items()}
    coll = {p: us(d, p) for p in COLLISION["lodos" if d["sim"].startswith("lodos") else "mjwarp"]}
    return dict(B=d["B"], world_steps_per_s=d["plain"]["world_steps_per_s"],
                us_plain=1e6 / d["plain"]["world_steps_per_s"], us_traced=1e3 * d["traced"]["ms"]["step"] / d["B"],
                categories=cats, collision_stages={k: v for k, v in coll.items() if v > 5e-4},
                counts={k: dict(mean=v["mean"], max=v["max"]) for k, v in d["traced"]["counts"].items()},
                variant=d.get("variant"), sizes=d.get("sizes"), attempts=d.get("attempts", 1),
                weld_filter=bool(d.get("weld_filter")))


def compare(lo, mj):
    """Lodos against MuJoCo Warp: the speed-up, each category's ratio and its share of the time saved."""
    gap = mj["us_traced"] - lo["us_traced"]
    return dict(speedup=mj["us_traced"] / lo["us_traced"],
                speedup_plain=lo["world_steps_per_s"] / mj["world_steps_per_s"],
                ratio={c: mj["categories"][c] / lo["categories"][c] for c in CATS},
                share={c: (mj["categories"][c] - lo["categories"][c]) / gap for c in CATS},
                with_lodos_collision=(mj["us_traced"] - mj["categories"]["collision"] + lo["categories"]["collision"])
                / lo["us_traced"])


def kernels(prefix, scene, B):
    out = {}
    for mode in ("lodos", "newton"):
        d = load(f"{prefix}.{scene}.{mode}.B{B}")
        if d is None:
            continue
        parts = d.get("stages") or d.get("parts")
        out[mode] = {k: dict(ms=v["ms"], top=[[r["name"].replace("forward kernel ", ""), r["ms"], r["launches"]]
                                              for r in v["kernels"][:8]]) for k, v in parts.items()}
        if prefix == "ksolve":
            out[mode].update(iters=d["iters_mean"], rows=d["rows_mean"])
    return out


def main():
    best = json.loads((ROOT / "bench" / "lodos_mjwarp" / "report" / "best.json").read_text())
    res = dict(source="bench/lodos_mjwarp/profile (run_profiles.py, summary.py)", scenes={})
    for s in SCENES:
        b = best[s]
        Bm = b["mjwarp"]["B"]
        rec = dict(best={}, equal={}, weld={}, compare={}, compare_equal={}, compare_weld={})
        for cfg, v in b.items():
            d = load(f"{s}.{cfg}.B{v['B']}")
            if d is not None:
                rec["best"][cfg] = dict(entry(d), best_json=v["world_steps_per_s"])
        for cfg in ("lodos_contact", "lodos_ray"):
            d = load(f"{s}.{cfg}.B{Bm}")
            if d is not None:
                rec["equal"][cfg] = entry(d)
        for B in (Bm, 2 * Bm):
            d = load(f"{s}.mjwarp.weldfilter.B{B}")
            if d is not None:
                rec["weld"][f"B{B}"] = entry(d)
        mj = rec["best"].get("mjwarp")
        for cfg, e in rec["best"].items():
            if cfg != "mjwarp" and mj:
                rec["compare"][cfg] = compare(e, mj)
        for cfg, e in rec["equal"].items():
            rec["compare_equal"][cfg] = compare(e, mj)
        if rec["weld"]:
            w = rec["weld"][f"B{Bm}"]
            for cfg in ("lodos_contact", "lodos_ray"):
                if cfg in rec["best"]:
                    rec["compare_weld"][cfg] = compare(rec["best"][cfg], w)
                if cfg in rec["equal"]:
                    rec["compare_weld"][cfg + "_equal"] = compare(rec["equal"][cfg], w)
        rec["collision_kernels"] = kernels("kern", s, Bm)
        rec["solve_kernels"] = kernels("ksolve", s, Bm)
        for name in (f"pairs.{s}.B256", f"hybrid.{s}.B{Bm}"):
            d = load(name)
            if d is not None:
                rec[name.split(".")[0]] = d
        res["scenes"][s] = rec
        show(s, rec)
    (HERE / "results.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"\nwrote {HERE / 'results.json'}")


def show(s, rec):
    print(f"\n=== {s}")
    rows = [("best B", rec["best"]), ("equal B", rec["equal"]), ("weld filter", rec["weld"])]
    for title, runs in rows:
        for cfg, e in runs.items():
            print(f"  {title:11s} {cfg:14s} B={e['B']:5d} {e['world_steps_per_s']:>12,.0f}/s {e['us_traced']:7.3f} us: "
                  + " ".join(f"{c} {e['categories'][c]:.3f}" for c in CATS))
    for title in ("compare", "compare_equal", "compare_weld"):
        for cfg, c in rec[title].items():
            print(f"  {title:14s} {cfg:20s} {c['speedup']:6.2f}x | " + " ".join(
                f"{k} {c['ratio'][k]:.2f}x/{100 * c['share'][k]:.0f}%" for k in CATS))


if __name__ == "__main__":
    main()
