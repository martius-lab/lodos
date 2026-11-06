"""
Writes `tuned/README.md`: every tuned scene's settings and throughput before and after.

    python tools/performance_tuning/summarize.py [--dir tools/performance_tuning/tuned]
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def fmt(x):
    if x is None:
        return "--"
    if x >= 1e6:
        return f"{x / 1e6:.3g}M"
    if x >= 1e3:
        return f"{x / 1e3:.3g}k"
    return f"{x:.3g}"


def changed(sec):
    out = []
    kw = sec.get("solver_kwargs", {})
    if "prox_iters_per_qp" in kw:
        out.append(f"prox_iters_per_qp={kw['prox_iters_per_qp']}")
    if sec.get("sim", {}).get("contact_reduction"):
        out.append("contact_reduction")
    col = sec.get("collision", {})
    if isinstance(col.get("tile_size"), dict):
        out.append("tile sizes per pair")
    if col.get("capacity1", 1.0) < 1.0:
        out.append("compacted tile filter")
    return ", ".join(out)


def caps(sec):
    col, mw = sec.get("collision", {}), sec.get("runtime", {})
    parts = []
    if "capacity_tile" in col:
        parts.append(f"tile {col['capacity_tile']:.3g}")
    if "capacity2" in col:
        parts.append(f"c2 {col['capacity2']:.3g}")
    if "naconmax" in mw:
        parts.append(f"nacon {mw['naconmax']}")
    if mw.get("njmax", 0) > 16:
        parts.append(f"njmax {mw['njmax']}")
    return ", ".join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", default=str(Path(__file__).resolve().parent / "tuned"))
    a = ap.parse_args()
    d = Path(a.dir)
    try:
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from viewer.scenes import catalog as vscenes

        order = list(vscenes.BUILTIN)
    except Exception:  # noqa: BLE001
        order = []
    files = sorted(d.glob("*.tuning.json"), key=lambda f: order.index(f.name[:-12]) if f.name[:-12] in order else 999)
    rows, ratios = [], []
    for f in files:
        t = json.loads(f.read_text())
        name = t["scene"]
        base = t.get("baseline") or {}
        bt = (base.get("throughput") or {}).get("world_steps_per_s")
        tt = (t.get("throughput") or {}).get("world_steps_per_s")
        r = tt / bt if (tt and bt) else None
        if r:
            ratios.append(r)
        ok = (t.get("artifacts") or {}).get("ok")
        check = "yes" if ok else "no: " + (t.get("artifacts") or {}).get("why", "")
        rows.append(f"| `{name}` | {base.get('B', '--')} | {fmt(bt)} | {t.get('B', '--')} | {fmt(tt)} | "
                    f"{f'{r:.2f}' if r else '--'} | {changed(t) or '--'} | {caps(t)} | {check} |")
    lines = [
        "# Tuned built-in scenes",
        "",
        "Every built-in viewer scene, tuned by `tools/performance_tuning/tune_all.py`",
        "on an RTX 4080 (16 GiB). Load one with `make_sim(..., tuning=...)` or",
        "`viewer.scenes.catalog.builtin(name).make_sim(B=..., tuning=...)`; the viewer's",
        "Performance panel shows the current scene's.",
        "",
        "- **baseline**: the scene's own settings -- the per-world MJWarp capacities the viewer sized",
        "  for one world, the collision defaults (fractions 1.0, tile size 8), the solver defaults --",
        "  at the family's batch size, or the largest that fit.",
        "- **tuned**: the file's settings at the batch size the sweep found fastest. *Changed* lists",
        "  what besides the capacities the tuning changed, each kept only past the artifact guard",
        "  (`common.no_worse`); *capacities* the main ones (Lodos's tile and stage-2 fractions,",
        "  MJWarp's contacts and rows).",
        "- **check**: the final artifact check of the tuned settings against the baseline's.",
        "",
        "World-steps per second (k = 10^3, M = 10^6).",
        "",
        "| scene | B | baseline | B | tuned | ratio | changed | capacities | check |",
        "|---|---|---|---|---|---|---|---|---|",
    ] + rows + [""]
    if ratios:
        lines += [f"Ratios, tuned against baseline: {min(ratios):.2f}-{max(ratios):.2f}x, median "
                  f"{statistics.median(ratios):.2f}x over {len(ratios)} scenes.", ""]
    (d / "README.md").write_text("\n".join(lines))
    print(f"{len(rows)} rows -> {d / 'README.md'}")


if __name__ == "__main__":
    main()
