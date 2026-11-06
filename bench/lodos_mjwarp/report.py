"""
The comparison's tables and figure, from what `run.py` wrote (`common.OUT`), for `report.tex`:

    python -m bench.lodos_mjwarp.report [--out bench/lodos_mjwarp/report]

`tables.tex` (the setup, the capacities, the execution options, the best throughput of each configuration --
by scene, and by configuration with its batch size, against real time and against MuJoCo Warp -- and every
ladder) and `throughput.pdf` (each scene's throughput against the batch size), then `report.tex` compiled with
pdflatex where it is installed.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

from bench.lodos_mjwarp import common as C

NAMES = {"lodos_contact": "Lodos, contact", "lodos_cone": "Lodos, cone", "lodos_ray": "Lodos, ray",
         "mjwarp": "MuJoCo Warp", "mjwarp_sap": "MuJoCo Warp, SAP"}
SAP_MARK = r"$^\S$"  # MuJoCo Warp's best under Newton's sweep-and-prune broad phase
D = C.OUT


def tt(s):
    return r"\texttt{" + str(s).replace("_", r"\_") + "}"


def k(v):
    """A throughput, compactly: 1.23M, 456k, 7.8k."""
    if v is None:
        return "--"
    for f, u in ((1e6, "M"), (1e3, "k")):
        if v >= f:
            return f"{v / f:.3g}{u}"
    return f"{v:.3g}"


def times(v):
    """A factor to three significant figures, its thousands set apart: 3\\,720, 552, 1.25."""
    r = float(f"{v:.3g}")
    return f"{r:,.0f}".replace(",", r"\,") if r >= 1000 else f"{r:.3g}"


def redone():
    """`{(scene, cfg): n}`: the runs discarded after an overflow, as `run.log` has them (a restarted driver numbers
    its discarded records from 0 again, over those it finds)."""
    n = {}
    log = D / "run.log"
    for line in log.read_text().splitlines() if log.exists() else []:
        m = re.match(r"\S+ \S+\s+(\S+) (\S+) B=\d+ overflowed", line)
        if m:
            n[m[1], m[2]] = n.get((m[1], m[2]), 0) + 1
    return n


def load():
    """`{scene: dict(info, caps={cfg: ...}, ab={cfg: ...}, points={cfg: {B: ...}}, overflowed={cfg: n})}`."""
    out, n = {}, redone()
    for scene in C.SCENES:
        p = D / "scenes" / f"{scene}.json"
        if not p.exists():
            continue
        info = C.read_json(p)
        caps, ab, points, notes = {}, {}, {}, {}
        for cfg in info["configs"]:
            c = D / "caps" / f"{scene}.{cfg}.json"
            caps[cfg] = C.read_json(c) if c.exists() else None
            a = D / "ab" / f"{scene}.{cfg}.json"
            ab[cfg] = C.read_json(a) if a.exists() else None
            pts = {}
            for f in sorted((D / "points").glob(f"{scene}.{cfg}.B*.json")):
                if ".overflowed" in f.name:
                    continue
                r = C.read_json(f)
                pts[int(f.name.split(".B")[1].split(".")[0])] = r
            points[cfg] = dict(sorted(pts.items()))
            notes[cfg] = n.get((scene, cfg), 0)
        out[scene] = dict(info=info, caps=caps, ab=ab, points=points, overflowed=notes)
    return out


def best(points):
    """The fastest run that did not overflow: `(B, result)`, or None."""
    ok = [(B, r) for B, r in points.items() if r.get("ok") and not r.get("overflowed")]
    return max(ok, key=lambda x: x[1]["world_steps_per_s"]) if ok else None


def lost(r):
    """The worlds a run lost: non-finite or escaped."""
    return r["nan_worlds"] + r["escaped_worlds"]


def mjwarp_best(points):
    """MuJoCo Warp's best over its broad phases: `(cfg, (B, result))`, or None."""
    bs = [(cfg, best(pts)) for cfg, pts in points.items() if cfg.startswith("mjwarp")]
    bs = [x for x in bs if x[1]]
    return max(bs, key=lambda x: x[1][1]["world_steps_per_s"]) if bs else None


def end_of(points):
    """How a ladder ended: its last run's out of memory or failure, a drop, or nothing."""
    if not points:
        return "not run"
    B, r = list(points.items())[-1]
    if not r.get("ok"):
        e = str(r.get("error"))
        return f"out of memory at {B}" if any(s in e for s in ("allocate", "out of memory")) else f"failed at {B}"
    b = best(points)
    return f"dropped at {B}" if b and b[0] != B else "--"


def tables(data):
    L = []
    # the setup of each scene
    L += [r"\begin{longtable}{@{}lrrrrl@{}}", r"\toprule",
          r"scene & s & $\Delta t$ (ms) & steps & solref (ms) & Lodos's solver \\", r"\midrule", r"\endhead"]
    for scene, d in data.items():
        i, s = d["info"], d["info"]["lodos_setting"]
        L.append(f"{tt(scene)} & {i['seconds']:g} & {1e3 * i['dt']:g} & {i['steps']} & {1e3 * i['solref'][0]:g} & "
                 f"{s['iterations']} it, tol {s['tolerance']:g}, ls {s['ls_iterations']} it" + r" \\")
    L += [r"\bottomrule", r"\caption{The scenes: the simulated seconds a run times, the step, the steps, every "
          r"contact's solref time constant (damping ratio 1), and the solver settings both sims run (the viewer's "
          r"defaults for Lodos's mj mode).}\label{tab:setup}", r"\end{longtable}", ""]
    # capacities per world
    L += [r"\begin{longtable}{@{}llp{9.5cm}@{}}", r"\toprule", r"scene & configuration & a world's capacities "
          r"(1.5 times its peak over 16 starts) \\", r"\midrule", r"\endhead"]
    for scene, d in data.items():
        for cfg, c in d["caps"].items():
            if not c:
                txt = "not measured"
            elif not c.get("ok"):
                txt = "failed: " + str(c.get("error"))[:80].replace("_", r"\_")
            else:
                sz = c["sizes"]
                if cfg.startswith("mjwarp"):
                    txt = (f"contacts {sz['contacts_per_world']}, triangle pairs {sz['triangle_pairs_per_world']}, "
                           f"rows {sz['njmax']}")
                    if "shape_pairs_per_world" in sz:
                        txt += f", shape pairs {sz['shape_pairs_per_world']}"
                else:
                    txt = ", ".join(f"{a} {b}" for a, b in sz["caps"].items())
                    if sz["collision"]:  # Lodos's arenas, where it has surfaces
                        txt += "; " + ", ".join(f"{a.replace('_', r'\_')} {b:.3g}" for a, b in sz["collision"].items())
            L.append(f"{tt(scene)} & {NAMES[cfg]} & {txt}" + r" \\")
    L += [r"\bottomrule", r"\caption{Each configuration's capacities per world, times the worlds at a batch size: "
          r"MuJoCo Warp's contacts, rows and CCD slots, Lodos's arenas as fractions of a world's worst case; "
          r"Newton's contacts and mesh triangle pairs (at least a million in all) and rows, and under its "
          r"sweep-and-prune broad phase (SAP) its candidate shape pairs.}\label{tab:caps}",
          r"\end{longtable}", ""]
    # the demand per world
    L += [r"\begin{longtable}{@{}lrrrr@{}}", r"\toprule", r"scene & Lodos, contact & Lodos, cone & Lodos, ray & "
          r"MuJoCo Warp \\", r"\midrule", r"\endhead"]
    for scene, d in data.items():
        cells = []
        for cfg in ("lodos_contact", "lodos_cone", "lodos_ray", "mjwarp"):
            c = d["caps"].get(cfg)
            if cfg not in d["caps"]:
                cells.append("n/a")
            elif not c or not c.get("ok"):
                cells.append("--")
            else:
                p = c["peaks"]
                cells.append(f"{p['contacts']} / {p['rows']}" if cfg == "mjwarp" else f"{p['nconmax']} / {p['njmax']}")
        L.append(f"{tt(scene)} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\caption{A world's largest demand over the 16 starts: contacts / constraint rows of the "
          r"contact solve (Lodos's: MJWarp's contacts and rows, its surfaces' contacts among them, a group of them a "
          r"contact). n/a: a polytope scene has no Lodos surfaces, so no groups.}\label{tab:demand}",
          r"\end{longtable}", ""]
    # execution options
    kept = {"default": "default", "no_shared_hessian": "no shared H.", "fused_newton": "fused Newton"}
    L += [r"\begin{longtable}{@{}llrrrrl@{}}", r"\toprule",
          r"scene & groups & B & default & no shared H. & fused Newton & kept \\", r"\midrule", r"\endhead"]
    for scene, d in data.items():
        for cfg, a in d["ab"].items():
            if cfg.startswith("mjwarp") or not a:
                continue
            if not a.get("ok"):
                L.append(f"{tt(scene)} & {cfg.split('_')[1]} & \\multicolumn{{5}}{{l}}{{failed}}" + r" \\")
                continue
            v = [k(a["variants"][n].get("world_steps_per_s")) for n in ("default", "no_shared_hessian", "fused_newton")]
            L.append(f"{tt(scene)} & {cfg.split('_')[1]} & {a['B']} & {v[0]} & {v[1]} & {v[2]} & "
                     f"{kept[a['chosen']]}" + r" \\")
    L += [r"\bottomrule", r"\caption{Lodos's execution options, world-steps/s at B worlds, from the capacity runs' "
          r"half-way states: the default with MJWarp's Hessian summed in shared memory (on by default where it "
          r"builds), without it (no shared H.), and the fused Newton solve; an option is kept where it is 3\,\% "
          r"faster (``--'': refused, the fused solve takes models of at most 16 degrees of freedom).}\label{tab:ab}",
          r"\end{longtable}", ""]
    # the best of each configuration
    L += [r"\begin{longtable}{@{}lrrrrr@{}}", r"\toprule", r"scene & Lodos, contact & Lodos, cone & Lodos, ray & "
          r"MuJoCo Warp & ratio \\", r"\midrule", r"\endhead"]
    dagger = r"$^\dagger$"
    for scene, d in data.items():
        cells, kept, rest = [], 0.0, 0.0
        for cfg in ("lodos_contact", "lodos_cone", "lodos_ray"):
            b = best(d["points"].get(cfg, {})) if cfg in d["points"] else None
            if cfg not in d["points"]:
                cells.append("n/a")
                continue
            cells.append(f"{k(b[1]['world_steps_per_s'])} @{b[0]}{dagger if lost(b[1]) else ''}" if b else "--")
            if b:
                v = b[1]["world_steps_per_s"]
                kept, rest = (max(kept, v), rest) if not lost(b[1]) else (kept, max(rest, v))
        m = mjwarp_best(d["points"])  # MuJoCo Warp's best over its broad phases
        mj = m[1][1]["world_steps_per_s"] if m else None
        cells.append(f"{k(mj)} @{m[1][0]}{dagger if lost(m[1][1]) else ''}{SAP_MARK if m[0] == 'mjwarp_sap' else ''}"
                     if m else "--")
        lodos = kept or rest  # the fastest Lodos configuration that lost no world, else the fastest
        ratio = f"{lodos / mj:.2f}{dagger if not kept else ''}" if mj and lodos else "--"
        L.append(f"{tt(scene)} & " + " & ".join(cells) + f" & {ratio}" + r" \\")
    L += [r"\bottomrule", r"\caption{The best throughput of each configuration, world-steps/s at the batch size "
          r"after @, and the ratio of Lodos's to MuJoCo Warp's, Lodos's the fastest of its configurations that lost no "
          r"world, MuJoCo Warp's the faster of its broad phases where it ran both. "
          r"$\dagger$: the run lost worlds, non-finite or escaped (\cref{tab:ladders}). " + SAP_MARK + r": under "
          r"Newton's sweep-and-prune broad phase (\cref{tab:max}). n/a: a polytope scene has no Lodos surfaces, so "
          r"no groups.}\label{tab:best}", r"\end{longtable}", ""]
    # the best of each configuration, its batch size, against real time and against MuJoCo Warp's best
    L += [r"\begin{longtable}{@{}llrrrr@{}}", r"\toprule", r"scene & configuration & world-steps/s & B & "
          r"$\times$ real time & $\times$ MuJoCo Warp \\", r"\midrule", r"\endhead"]
    for n, (scene, d) in enumerate(data.items()):
        bests = {cfg: best(pts) for cfg, pts in d["points"].items()}
        mj = (mjwarp_best(d["points"]) or (None, None))[1]  # over MuJoCo Warp's broad phases
        L += [r"\addlinespace"] if n else []
        for i, (cfg, b) in enumerate(bests.items()):
            end = r" \\" if i == len(bests) - 1 else r" \\*"  # no page break inside a scene
            row = f"{tt(scene) if i == 0 else ''} & {NAMES[cfg]} & "
            if not b:
                L.append(row + "-- & & &" + end)
                continue
            v = b[1]["world_steps_per_s"]
            ratio = f"{v / mj[1]['world_steps_per_s']:.2f}" if mj else "--"
            L.append(row + f"{k(v)}{dagger if lost(b[1]) else ''} & {b[0]} & {times(b[1]['sim_seconds_per_s'])} & "
                     f"{ratio}" + end)
    L += [r"\bottomrule", r"\caption{Each configuration's best throughput: world-steps/s at a batch of B worlds; how "
          r"many times real time it ran, the seconds simulated per second summed over the worlds (world-steps/s "
          r"times the step, 2\,ms, the threading's 1.25\,ms; each world's own is that over B); and its ratio to "
          r"MuJoCo Warp's best on the scene, over its broad phases. MuJoCo Warp, SAP: under Newton's sweep-and-prune "
          r"broad phase, its candidate shape pairs a capacity measured as the others, in place of the explicit pairs, "
          r"whose narrow-phase buffers Newton sizes for every pair of shapes in every world. $\dagger$: the run lost "
          r"worlds, non-finite or escaped (\cref{tab:ladders}).}\label{tab:max}", r"\end{longtable}", ""]
    # every ladder
    L += [r"\begin{longtable}{@{}llp{9.5cm}@{}}", r"\toprule", r"scene & configuration & world-steps/s by batch "
          r"size; how the ladder ended; worlds lost at the best \\", r"\midrule", r"\endhead"]
    for scene, d in data.items():
        for cfg, pts in d["points"].items():
            run = ", ".join(f"{B}: {k(r['world_steps_per_s'])}" if r.get("ok") and not r.get("overflowed")
                            else f"{B}: --" for B, r in pts.items())
            b = best(pts)
            gone = f"{b[1]['nan_worlds']} non-finite, {b[1]['escaped_worlds']} escaped" if b else ""
            over = f"; {d['overflowed'][cfg]} redone after an overflow" if d["overflowed"].get(cfg) else ""
            L.append(f"{tt(scene)} & {NAMES[cfg]} & {run}; {end_of(pts)}; {gone}{over}" + r" \\")
    L += [r"\bottomrule", r"\caption{Every ladder: each batch size's world-steps/s, from 64 doubled, how it ended, "
          r"and the worlds the best run lost.}\label{tab:ladders}", r"\end{longtable}"]
    # a table's last row kept on the page of its caption: no break before its bottom rule (\\*) or after it
    # (booktabs' \bottomrule ends in glue, at which a page may break; a bare rule does not)
    end = "\\\\*\n\\noalign{\\vskip\\aboverulesep\\hrule height\\heavyrulewidth}"
    return "\n".join(L).replace("\\\\\n\\bottomrule", end)


def figure(data, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenes = [s for s, d in data.items() if any(d["points"].values())]
    n = len(scenes)
    if not n:
        return False
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(11, 2.6 * rows), squeeze=False)
    colors = {"lodos_contact": "#1f77b4", "lodos_cone": "#2ca02c", "lodos_ray": "#9467bd", "mjwarp": "#d62728",
              "mjwarp_sap": "#ff7f0e"}
    for ax, scene in zip(axes.flat, scenes):
        for cfg, pts in data[scene]["points"].items():
            xs = [B for B, r in pts.items() if r.get("ok") and not r.get("overflowed")]
            ys = [pts[B]["world_steps_per_s"] for B in xs]
            if xs:  # dashed: a run lost worlds; MuJoCo Warp's explicit pairs over SAP's, which run alike to their end
                ax.plot(xs, ys, "o--" if any(lost(pts[B]) for B in xs) else "o-", ms=3, lw=1.2, color=colors[cfg],
                        label=NAMES[cfg], zorder=3 if cfg == "mjwarp" else 2)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(scene, fontsize=9)
        ax.tick_params(which="both", labelsize=7)
        ax.grid(True, which="both", lw=0.3, alpha=0.5)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    from matplotlib.lines import Line2D

    keys = [Line2D([], [], color=c, marker="o", ms=3, lw=1.2, label=NAMES[cfg]) for cfg, c in colors.items()]
    keys.append(Line2D([], [], color="gray", ls="--", lw=1.2, label="a run lost worlds"))
    fig.legend(handles=keys, loc="upper center", ncol=len(keys), fontsize=8, frameon=False)
    fig.supxlabel("worlds", fontsize=9)
    fig.supylabel("world-steps / s", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.3 / (2.6 * rows)))
    fig.savefig(path)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(C.ROOT / "bench" / "lodos_mjwarp" / "report"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = load()
    (out / "tables.tex").write_text("\\begingroup\\small\\setlength{\\tabcolsep}{4pt}\n" + tables(data) + "\n\\endgroup\n")
    figure(data, out / "throughput.pdf")
    summary = {s: {cfg: (lambda b: None if b is None else dict(B=b[0], world_steps_per_s=b[1]["world_steps_per_s"],
                                                                    sim_seconds_per_s=b[1]["sim_seconds_per_s"]))(
        best(pts)) for cfg, pts in d["points"].items()} for s, d in data.items()}
    (out / "best.json").write_text(json.dumps(summary, indent=1))
    tex = out / "report.tex"
    if tex.exists() and shutil.which("pdflatex"):
        for _ in range(2):
            subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex.name], cwd=out,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"wrote {out / 'report.pdf'}")
    print(f"wrote {out / 'tables.tex'}, {out / 'throughput.pdf'}, {out / 'best.json'}")


if __name__ == "__main__":
    main()
