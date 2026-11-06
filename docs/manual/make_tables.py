"""
The guides' tables of tuned scenes, from tools/performance_tuning/tuned/ (no GPU).

    python docs/manual/make_tables.py

writes into docs/manual/tables/:

    builtin.tex          every built-in scene: baseline and tuned batch size and throughput,
                         the ratio, the budgets kept, the final artifact check
    builtin_summary.tex  the ratios' range and median, what the budgets did
    extune.tex           the worked example's two scenes (creating_new_sim_from_mjcf.tex)
    profiles.tex         stage profiles, from tables/profiles/*.json (profile_scene.py --json)
"""

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TUNED = ROOT / "tools" / "performance_tuning" / "tuned"
OUT = Path(__file__).resolve().parent / "tables"
sys.path.insert(0, str(ROOT))


def fmt(x):
    """World-steps per second, three digits: 62.2k, 1.18M."""
    if x is None:
        return "--"
    if x >= 1e6:
        return f"{x / 1e6:.3g}M"
    if x >= 1e3:
        return f"{x / 1e3:.3g}k"
    return f"{x:.3g}"


def tex(s):
    return s.replace("_", r"\_")


def budgets(t):
    """What the tuning changed from the solver defaults, short."""
    out = []
    kw = t.get("solver_kwargs", {})
    if "prox_iters_per_qp" in kw:
        out.append(f"prox {kw['prox_iters_per_qp']}")
    if t.get("sim", {}).get("contact_reduction"):
        out.append("red")
    col = t.get("collision", {})
    if isinstance(col.get("tile_size"), dict):
        out.append("tiles")
    if "capacity1" in col and col["capacity1"] < 1.0:
        out.append("compact")
    return ", ".join(out)


def order(names):
    from viewer.scenes import catalog as vscenes

    known = list(vscenes.BUILTIN)

    def key(n):
        base = n.split(" (")[0]  # a file tuned under another mode follows its scene's
        return (known.index(base) if base in known else len(known), n != base, n)

    return sorted(names, key=key)


def load_all():
    """Every tuned file and its report: the scenes' own, and those tuned under another mode
    (`tune.py --mode`, in a subdirectory named for it), named `scene (mode)`."""
    from lodos.cd.driver import MODES

    rows = {}
    dirs = [(TUNED, "")] + [(TUNED / m, f" ({m})") for m in MODES if (TUNED / m).is_dir()]
    for folder, tag in dirs:
        for f in folder.glob("*.report.json"):
            name = f.name[: -len(".report.json")]
            rep = json.loads(f.read_text())
            tf = folder / f"{name}.tuning.json"
            t = json.loads(tf.read_text()) if tf.exists() else {}
            rows[name + tag] = (rep, t)
    return rows


def scene_row(name, t):
    if not t:
        return None
    base = t.get("baseline") or {}
    b_thr = (base.get("throughput") or {}).get("world_steps_per_s")
    thr = (t.get("throughput") or {}).get("world_steps_per_s")
    art = t.get("artifacts") or {}
    return dict(scene=name, B_base=base.get("B"), base=b_thr, B=t.get("B"), tuned=thr,
                ratio=(thr / b_thr) if (thr and b_thr) else None, budgets=budgets(t), ok=art.get("ok"),
                why=art.get("why", ""))


def main():
    OUT.mkdir(exist_ok=True)
    data = load_all()
    lines, allrows = [], []
    for name in order(list(data)):
        rep, t = data[name]
        r = scene_row(name, t)
        if r is None:
            continue
        allrows.append(r)
        check = r"\checkmark" if r["ok"] else r"\textbf{no}"
        note = r["budgets"] + ("; " if r["budgets"] else "") + check
        lines.append(" & ".join([
            r"\code{" + name + "}", str(r["B_base"] or "--"), fmt(r["base"]), str(r["B"] or "--"), fmt(r["tuned"]),
            f"{r['ratio']:.2f}" if r["ratio"] else "--", note]) + r" \\")
    head = [r"\begin{longtable}{@{}L{44mm}R{9mm}R{13mm}R{9mm}R{13mm}R{9mm}L{36mm}@{}}",
            r"\caption{Every built-in scene, before and after tuning: batch size, world-steps per second and the "
            r"ratio; what the tuning changed besides the capacities (``prox'' $n$: \code{prox_iters_per_qp}; "
            r"``red'': \code{contact_reduction}; ``tiles'': tile sizes per pair; ``compact'': the "
            r"compacted tile filter) and the final artifact check (\checkmark{} passed); ``(mj)'': a file tuned under mj "
            r"(\code{tune.py --mode mj}).}\label{tab:builtin}\\",
            r"\toprule",
            r"scene & \multicolumn{2}{c}{baseline} & \multicolumn{2}{c}{tuned} & & \\",
            r" & $B$ & steps/s & $B$ & steps/s & ratio & changed, check \\",
            r"\midrule", r"\endfirsthead", r"\toprule",
            r"scene & $B$ & steps/s & $B$ & steps/s & ratio & changed, check \\",
            r"\midrule", r"\endhead"]
    (OUT / "builtin.tex").write_text("\n".join(head + lines + [r"\bottomrule", r"\end{longtable}"]) + "\n")

    # the summary: the ratios, the budgets, the checks -- of the scenes' own files, not those tuned under another mode
    scenes = [r for r in allrows if " (" not in r["scene"]]
    own = [r for r in scenes if r["ratio"]]
    rs = sorted(r["ratio"] for r in own)
    parts = [f"{rs[0]:.2f}--{rs[-1]:.2f}$\\times$ (median {statistics.median(rs):.2f}, {len(rs)} scenes)"] if rs else []
    failed = [r for r in scenes if not r["ok"]]
    kept = {}
    for r in own:
        for b in [x.strip() for x in r["budgets"].split(",") if x.strip()]:
            kept[b] = kept.get(b, 0) + 1
    with open(OUT / "builtin_summary.tex", "w") as f:
        f.write("Over all scenes, the tuned settings against the baseline: " + "; ".join(parts) + ".")
        if kept:
            f.write(" Changes kept: " + ", ".join(f"{k} in {v}" for k, v in sorted(kept.items())) + " scenes.")
        f.write(f" {len(scenes) - len(failed)} of {len(scenes)} passed the final artifact check" +
                (": the rest are listed below." if failed else ".") + "\n")

    # the worked example
    ex = [r for r in allrows if r["scene"].startswith("sharpa_armadillo")]
    ex_lines = [r"\begin{tabularx}{\linewidth}{@{}L{50mm}R{8mm}R{13mm}R{8mm}R{13mm}R{9mm}Y@{}}", r"\toprule",
                r"scene & \multicolumn{2}{c}{before} & \multicolumn{2}{c}{after} & ratio & kept \\",
                r" & $B$ & steps/s & $B$ & steps/s & & \\", r"\midrule"]
    for r in ex:
        ex_lines.append(" & ".join([r"\code{" + r["scene"] + "}", str(r["B_base"]), fmt(r["base"]), str(r["B"]),
                                    fmt(r["tuned"]), f"{r['ratio']:.2f}" if r["ratio"] else "--",
                                    r["budgets"] or "--"]) + r" \\")
    ex_lines += [r"\bottomrule", r"\end{tabularx}"]
    (OUT / "extune.tex").write_text("\n".join(ex_lines) + "\n")

    # stage profiles
    prof = sorted((OUT / "profiles").glob("*.json")) if (OUT / "profiles").exists() else []
    if prof:
        cols = ["rbd", "prim_collision", "lodos", "contact_dynamics", "integrate", "whole_step"]
        heads = ["rigid body", "MJWarp coll.", "Lodos coll.", "solver", "integrate", "step"]
        pl = [r"\begin{tabularx}{\linewidth}{@{}Y" + "r" * (len(cols) + 1) + "@{}}", r"\toprule",
              "scene, $B$ & " + " & ".join(heads) + r" & steps/s \\", r"\midrule"]
        for p in prof:
            d = json.loads(p.read_text())
            ms = dict(d["ms"])
            ms["lodos"] = sum(ms.pop(k, 0.0) for k in ("lodos_bbox", "lodos_sdf", "lodos_narrow")) or None
            cells = [(f"{ms[c]:.3f}" if ms.get(c) else "--") for c in cols]
            pl.append(r"\code{" + d["scene"] + "}" + f", {d['B']} & " + " & ".join(cells)
                      + f" & {fmt(d['world_steps_per_s'])}" + r" \\")
        pl += [r"\bottomrule", r"\end{tabularx}"]
        (OUT / "profiles.tex").write_text("\n".join(pl) + "\n")
    elif not (OUT / "profiles.tex").exists():
        (OUT / "profiles.tex").write_text("(no profiles yet)\n")
    print(f"{len(allrows)} rows from {len(data)} scenes; {len(failed)} failed the final check; "
          f"{len(prof)} profiles -> {OUT}")


if __name__ == "__main__":
    main()
