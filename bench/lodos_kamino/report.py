"""
The comparison's tables and figure, from what `run.py` wrote (`common.OUT`), for `report.tex`:

    python -m bench.lodos_kamino.report [--out bench/lodos_kamino/report]

As `bench.lodos_mjwarp.report`, Kamino in MuJoCo Warp's place: `tables.tex` (the setup, the capacities and the
demand, the execution options, Kamino's solves, the best throughput of each configuration -- by scene, and by
configuration with its batch size, against real time and against Kamino on the same bodies -- and every ladder) and
`throughput.pdf` (each scene's throughput against the batch size), then `report.tex` compiled with pdflatex where it
is installed.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

from bench.lodos_kamino import common as C
from bench.lodos_mjwarp.report import best, k, lost, times, tt

NAMES = {"lodos_contact": "Lodos, contact", "lodos_cone": "Lodos, cone", "lodos_ray": "Lodos, ray", "kamino": "Kamino"}
D = C.OUT
DAGGER, TWIN = r"$^\dagger$", r"$^\ddagger$"


def redone():
    """`{(scene, cfg): n}`: the runs discarded after an overflow, as `run.log` has them."""
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
                pts[int(f.name.split(".B")[1].split(".")[0])] = C.read_json(f)
            points[cfg] = dict(sorted(pts.items()))
            notes[cfg] = n.get((scene, cfg), 0)
        out[scene] = dict(info=info, caps=caps, ab=ab, points=points, overflowed=notes)
    return out


def kamino_best(data, scene):
    """Kamino's best run on the scene's bodies, `((B, result), twin)`: on the scene, or, `twin`, on its polytope
    version where the scene has Lodos surfaces, which Kamino does not run; None where it ran on neither."""
    for s in (scene, scene.replace("_sdf", "_polytope")):
        b = best(data.get(s, {}).get("points", {}).get("kamino", {}))
        if b:
            return b, s != scene
    return None


def end_of(points):
    """How a ladder ended: its last run's out of memory, Kamino's 2^31 entries or failure, a drop, or nothing."""
    if not points:
        return "not run"
    B, r = list(points.items())[-1]
    if not r.get("ok"):
        e = str(r.get("error"))
        if any(s in e for s in ("allocate", "out of memory", "cudaErrorMemory")):
            return f"out of memory at {B}"
        return f"past $2^{{31}}$ entries at {B}" if "Array shapes must" in e else f"failed at {B}"
    b = best(points)
    return f"dropped at {B}" if b and b[0] != B else "--"


def tables(data, rows):
    L = []
    # the setup of each scene
    L += [r"\begin{longtable}{@{}lrrrlr@{}}", r"\toprule",
          r"scene & s & $\Delta t$ (ms) & steps & Lodos's Newton solver & Kamino's iterations \\", r"\midrule",
          r"\endhead"]
    for scene, d in data.items():
        i, s = d["info"], d["info"]["lodos_setting"]
        kam = (str(i["kamino_options"]["max_iterations"]) if "kamino" in i["configs"] else
               "left out" if scene in C.KAMINO_EXCLUDED else "--")
        L.append(f"{tt(scene)} & {i['seconds']:g} & {1e3 * i['dt']:.3g} & {i['steps']} & {s['iterations']} it, tol "
                 f"{s['tolerance']:g}, ls {s['ls_iterations']} it & {kam}" + r" \\")
    L += [r"\bottomrule", r"\caption{The scenes: the simulated seconds a run times, the step, the steps, Lodos's "
          r"Newton solver (its iterations, tolerance and line-search iterations) and Kamino's PADMM iterations (--: "
          r"a scene with Lodos surfaces, which Kamino does not run; left out: the twenty hands, where 16 worlds took "
          r"7.2\,s a step).}\label{tab:setup}", r"\end{longtable}", ""]
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
                if cfg == "kamino":
                    txt, r = f"contacts {sz['contacts_per_world']}", rows.get(scene)
                    if r:
                        jl = f" ({r['joint_and_limit_rows']} of joints and limits)" if r["joint_and_limit_rows"] else ""
                        txt += (f"; its Delassus matrix {r['rows']:,} rows{jl}, {r['mb_per_world']:g}\\,MB, so at "
                                f"most {r['batch_cap']:,} worlds")
                else:
                    txt = ", ".join(f"{a} {b}" for a, b in sz["caps"].items())
                    if sz["collision"]:
                        txt += "; " + ", ".join(f"{a.replace('_', r'\_')} {b:.3g}" for a, b in sz["collision"].items())
            L.append(f"{tt(scene)} & {NAMES[cfg]} & {txt}" + r" \\")
    L += [r"\bottomrule", r"\caption{Each configuration's capacities per world, times the worlds at a batch size: "
          r"Lodos's MJWarp contacts, rows and CCD slots and its arenas as fractions of a world's worst case; Kamino's "
          r"contacts, which size its dense Delassus matrix too, three rows a contact besides the joints' and limits'. "
          r"One array holds every world's matrix, at most $2^{31}$ entries, Warp's limit, which caps Kamino's "
          r"batch.}\label{tab:caps}", r"\end{longtable}", ""]
    # the demand per world
    L += [r"\begin{longtable}{@{}lrrrr@{}}", r"\toprule", r"scene & Lodos, contact & Lodos, cone & Lodos, ray & "
          r"Kamino \\", r"\midrule", r"\endhead"]
    for scene, d in data.items():
        cells = []
        for cfg in NAMES:
            c = d["caps"].get(cfg)
            if cfg not in d["caps"]:
                cells.append("left out" if cfg == "kamino" and scene in C.KAMINO_EXCLUDED else "n/a")
            elif not c or not c.get("ok"):
                cells.append("--")
            else:
                p = c["peaks"]
                cells.append(f"{p['world_contacts']}" if cfg == "kamino" else f"{p['nconmax']} / {p['njmax']}")
        L.append(f"{tt(scene)} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\caption{A world's largest demand over the 16 starts: Lodos's contacts / constraint rows "
          r"of the contact solve (MJWarp's contacts and rows, its surfaces' contacts among them, a group of them a "
          r"contact); Kamino's contacts. n/a: a polytope scene has no Lodos surfaces, so no groups, and Kamino does "
          r"not run the scenes with them.}\label{tab:demand}", r"\end{longtable}", ""]
    # Lodos's execution options
    kept = {"default": "default", "no_shared_hessian": "no shared H.", "fused_newton": "fused Newton"}
    L += [r"\begin{longtable}{@{}llrrrrl@{}}", r"\toprule",
          r"scene & groups & B & default & no shared H. & fused Newton & kept \\", r"\midrule", r"\endhead"]
    for scene, d in data.items():
        for cfg, a in d["ab"].items():
            if cfg == "kamino" or not a:
                continue
            if not a.get("ok"):
                L.append(f"{tt(scene)} & {cfg.split('_')[1]} & \\multicolumn{{5}}{{l}}{{failed}}" + r" \\")
                continue
            v = [k(a["variants"][n].get("world_steps_per_s")) for n in ("default", "no_shared_hessian", "fused_newton")]
            L.append(f"{tt(scene)} & {cfg.split('_')[1]} & {a['B']} & {v[0]} & {v[1]} & {v[2]} & "
                     f"{kept[a['chosen']]}" + r" \\")
    L += [r"\bottomrule", r"\caption{Lodos's execution options, world-steps/s at B worlds, from the capacity runs' "
          r"half-way states: the default with MJWarp's Hessian summed in shared memory (on by default where it "
          r"builds, on models of at most 144 degrees of freedom, padded), without it (no shared H.), and the fused "
          r"Newton solve; an option is kept where it is 3\,\% faster (``--'': refused, the fused solve takes models "
          r"of at most 16 degrees of freedom).}\label{tab:ab}", r"\end{longtable}", ""]
    # Kamino's execution options
    kept = {"default": "default", "sparse_jacobian": "sparse J.", "rcm": "RCM", "sparse_jacobian_rcm": "both"}
    L += [r"\begin{longtable}{@{}lrrrrrl@{}}", r"\toprule",
          r"scene & B & default & sparse J. & RCM & both & kept \\", r"\midrule", r"\endhead"]
    for scene, d in data.items():
        a = d["ab"].get("kamino")
        if "kamino" not in d["ab"]:
            continue
        if not a or not a.get("ok"):
            L.append(f"{tt(scene)} & \\multicolumn{{6}}{{l}}{{{'failed' if a else 'not timed'}}}" + r" \\")
            continue
        v = [k(r["world_steps_per_s"]) if "world_steps_per_s" in r else
             "OOM" if "allocate" in r.get("error", "") else "--" for r in (a["variants"].get(n, {}) for n in kept)]
        L.append(f"{tt(scene)} & {a['B']} & " + " & ".join(v) + f" & {kept[a['chosen']]}" + r" \\")
    L += [r"\bottomrule", r"\caption{Kamino's execution options, world-steps/s at B worlds, from the capacity runs' "
          r"half-way states: the default (dense Jacobians, its Delassus matrices factored by blocked Cholesky), "
          r"sparse Jacobians (sparse J.), the factorization after a reverse Cuthill--McKee reordering (RCM), and "
          r"both: the same solve each. An option is kept where it is 3\,\% faster. B is halved from 1024 until the "
          r"default fits. OOM: out of memory.}\label{tab:abk}", r"\end{longtable}", ""]
    # Kamino's solves
    L += [r"\begin{longtable}{@{}lrrrrl@{}}", r"\toprule",
          r"scene & cap & best & iterations & converged & ladder \\", r"\midrule", r"\endhead"]
    for scene, d in data.items():
        if "kamino" not in d["points"]:
            continue
        pts, cap = d["points"]["kamino"], d["info"]["kamino_options"]["max_iterations"]
        b = best(pts)
        if not b:
            L.append(f"{tt(scene)} & {cap} & -- & & & {end_of(pts)}" + r" \\")
            continue
        st = b[1].get("status", {})
        L.append(f"{tt(scene)} & {cap} & {k(b[1]['world_steps_per_s'])} @{b[0]} & {st['mean_iterations']:.3g} & "
                 f"{100 * st['converged']:.3g}\\,\\% & {end_of(pts)}" + r" \\")
    L += [r"\bottomrule", r"\caption{Kamino's solves: its PADMM iterations' cap, its best throughput (world-steps/s "
          r"at the batch size after @), and in that run's last step its worlds' mean PADMM iterations and the share "
          r"of them whose solve met its tolerances; how its ladder ended.}\label{tab:kamino}", r"\end{longtable}", ""]
    # the best of each configuration
    L += [r"\begin{longtable}{@{}lrrrrr@{}}", r"\toprule", r"scene & Lodos, contact & Lodos, cone & Lodos, ray & "
          r"Kamino & ratio \\", r"\midrule", r"\endhead"]
    for scene, d in data.items():
        cells, kept, rest = [], 0.0, 0.0
        for cfg in NAMES:
            if cfg not in d["points"]:
                cells.append("left out" if cfg == "kamino" and scene in C.KAMINO_EXCLUDED else "n/a")
                continue
            b = best(d["points"][cfg])
            cells.append(f"{k(b[1]['world_steps_per_s'])} @{b[0]}{DAGGER if lost(b[1]) else ''}" if b else "--")
            if b and cfg != "kamino":
                v = b[1]["world_steps_per_s"]
                kept, rest = (max(kept, v), rest) if not lost(b[1]) else (kept, max(rest, v))
        lodos = kept or rest  # the fastest Lodos configuration that lost no world, else the fastest
        kb = kamino_best(data, scene)
        ratio = (f"{lodos / kb[0][1]['world_steps_per_s']:.2f}{DAGGER if not kept else ''}{TWIN if kb[1] else ''}"
                 if kb and lodos else "--")
        L.append(f"{tt(scene)} & " + " & ".join(cells) + f" & {ratio}" + r" \\")
    L += [r"\bottomrule", r"\caption{The best throughput of each configuration, world-steps/s at the batch size "
          r"after @, and the ratio of Lodos's to Kamino's on the same bodies, Lodos's the fastest of its "
          r"configurations that lost no world. $\ddagger$: against Kamino on the scene's polytope version, which "
          r"collides the bodies as convex hulls, the scene having Lodos surfaces. $\dagger$: the run lost worlds, "
          r"non-finite or escaped (\cref{tab:ladders}). n/a: a polytope scene has no Lodos surfaces, so no groups, "
          r"and Kamino does not run the scenes with them.}\label{tab:best}", r"\end{longtable}", ""]
    # the best of each configuration, its batch size, against real time and against Kamino's best
    L += [r"\begin{longtable}{@{}llrrrr@{}}", r"\toprule", r"scene & configuration & world-steps/s & B & "
          r"$\times$ real time & $\times$ Kamino \\", r"\midrule", r"\endhead"]
    for n, (scene, d) in enumerate(data.items()):
        bests = {cfg: best(pts) for cfg, pts in d["points"].items()}
        kb = kamino_best(data, scene)
        L += [r"\addlinespace"] if n else []
        for i, (cfg, b) in enumerate(bests.items()):
            end = r" \\" if i == len(bests) - 1 else r" \\*"  # no page break inside a scene
            row = f"{tt(scene) if i == 0 else ''} & {NAMES[cfg]} & "
            if not b:
                L.append(row + "-- & & &" + end)
                continue
            v = b[1]["world_steps_per_s"]
            ratio = f"{v / kb[0][1]['world_steps_per_s']:.2f}{TWIN if kb[1] else ''}" if kb else "--"
            L.append(row + f"{k(v)}{DAGGER if lost(b[1]) else ''} & {b[0]} & {times(b[1]['sim_seconds_per_s'])} & "
                     f"{ratio}" + end)
    L += [r"\bottomrule", r"\caption{Each configuration's best throughput: world-steps/s at a batch of B worlds; how "
          r"many times real time it ran, the seconds simulated per second summed over the worlds (world-steps/s "
          r"times the step: 2\,ms, the threading's 1.25\,ms, the arch's and the house of cards' 4.17\,ms; each "
          r"world's own is that over B); and its ratio to Kamino's best on the same bodies. $\ddagger$: against "
          r"Kamino on the scene's polytope version, as convex hulls. $\dagger$: the run lost worlds, non-finite or "
          r"escaped (\cref{tab:ladders}).}\label{tab:max}", r"\end{longtable}", ""]
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
            shared = [B for B, r in pts.items() if r.get("gpu_shared")]
            shared = f"; timed beside another process at {', '.join(map(str, shared))}" if shared else ""
            L.append(f"{tt(scene)} & {NAMES[cfg]} & {run}; {end_of(pts)}; {gone}{over}{shared}" + r" \\")
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
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(11, 2.5 * rows), squeeze=False)
    colors = {"lodos_contact": "#1f77b4", "lodos_cone": "#2ca02c", "lodos_ray": "#9467bd", "kamino": "#d62728"}
    for ax, scene in zip(axes.flat, scenes):
        for cfg, pts in data[scene]["points"].items():
            xs = [B for B, r in pts.items() if r.get("ok") and not r.get("overflowed")]
            ys = [pts[B]["world_steps_per_s"] for B in xs]
            if xs:  # dashed: a run lost worlds
                ax.plot(xs, ys, "o--" if any(lost(pts[B]) for B in xs) else "o-", ms=3, lw=1.2, color=colors[cfg],
                        label=NAMES[cfg])
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
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.3 / (2.5 * rows)))
    fig.savefig(path)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(C.ROOT / "bench" / "lodos_kamino" / "report"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = load()
    rows = D / "caps" / "kamino_rows.json"  # each scene's Delassus rows a world, from Kamino's build at its capacity
    rows = C.read_json(rows) if rows.exists() else {}
    (out / "tables.tex").write_text("\\begingroup\\small\\setlength{\\tabcolsep}{4pt}\n" + tables(data, rows) +
                                    "\n\\endgroup\n")
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
