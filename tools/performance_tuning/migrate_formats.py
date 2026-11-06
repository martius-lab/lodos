"""
Migrate files in the first of Lodos's stored formats to the names its formats use.

In the first formats, the sections holding the sizes and a run's use of the buffers every pipeline shares (the contacts
the solve sees, the broadphase's pairs, the rows, the hybrid's CCD arena) are "mjwarp" and "mjw", where Lodos's are
"runtime"; the counters runtime_* are mjw_contacts_world, mjw_overflow and mjw_contact_overflow; and the viewer's pair
kind "prim" and its frame stat ncon_prim are "mjw" and ncon_mjw. Lodos refuses such files rather than read them under
those names: a tuning file with an "mjwarp" or "mjw" section raises, and so does a take of format 1; takes and exported
viewer configurations are of format 2 (lodos-viewer-config/2).

    python tools/performance_tuning/migrate_formats.py PATH...          files or directories (*.json, *.jsonl, *.npz)
    python tools/performance_tuning/migrate_formats.py --dry PATH...    report what would change, write nothing

What is renamed is decided by content, because the same names mean MuJoCo Warp itself elsewhere (bench/lodos_mjwarp's
"mjwarp" entries are MuJoCo Warp's throughput, and stay):

  "mjwarp": the capacities (naconmax, nconmax, njmax, nccdmax, njmax_nnz)    -> "runtime"
  "mjwarp": {contacts, rows_per_world}, tools/performance_tuning/profile_scene's    -> "runtime"
  "mjw": the calibration's counters (nacon, naconmax, contacts_per_world...)  -> "runtime"
  "mjw" under a viewer configuration's pairs or overrides                     -> "prim"
  mjw_contacts_world, mjw_overflow, mjw_contact_overflow                      -> runtime_*
  ncon_mjw -> ncon_prim; mjw_geom_mode -> prim_geom_mode; mjw_collision, mjw_poses -> prim_collision, prim_poses
  a take (.npz): format 1 -> 2; an exported viewer configuration: lodos-viewer-config/1 -> /2

A file is written back in its own layout (json's indent and trailing newline, found by reproducing it), so that its
diff is only the renamed keys; a log of JSON lines is migrated line by line.
"""

import argparse
import io
import json
from pathlib import Path

import numpy as np

CAPS = {"naconmax", "nconmax", "njmax", "nccdmax", "njmax_nnz"}
PROFILE = {"contacts", "rows_per_world"}
REPORT = {"nacon", "ncollision", "naconmax", "contacts_per_world", "overflow", "nefc_per_world", "njmax",
          "efc_overflow", "ccd", "nacon_peak", "ncollision_peak", "nefc_peak_per_world", "nccdmax"}
PAIRS = {"pairs", "overrides"}
PLAIN = {"mjw_contacts_world": "runtime_contacts_world", "mjw_overflow": "runtime_overflow",
         "mjw_contact_overflow": "runtime_contact_overflow", "ncon_mjw": "ncon_prim", "mjw_geom_mode": "prim_geom_mode",
         "mjw_collision": "prim_collision", "mjw_poses": "prim_poses"}
CONFIG_OLD, CONFIG_NEW = "lodos-viewer-config/1", "lodos-viewer-config/2"


def new_name(key, value, parent):
    if key in PLAIN:
        return PLAIN[key]
    if key == "mjwarp" and isinstance(value, dict) and value and (set(value) <= CAPS or set(value) <= PROFILE):
        return "runtime"
    if key == "mjw" and isinstance(value, dict):
        if parent in PAIRS:
            return "prim"
        if value and set(value) <= REPORT:
            return "runtime"
    return None


def migrate(x, path="", parent="", done=None, left=None):
    """A copy of `x` under the new names; `done` and `left` collect what was renamed and what was not."""
    done = [] if done is None else done
    left = [] if left is None else left
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            nk = new_name(k, v, parent) if isinstance(k, str) else None
            if nk:
                done.append(f"{path}{k} -> {nk}")
            elif isinstance(k, str) and "mjw" in k:
                left.append(f"{path}{k}")
            out[nk or k] = migrate(v, f"{path}{nk or k}.", k, done, left)
        if out.get("format") == CONFIG_OLD:
            out["format"] = CONFIG_NEW
            done.append(f"{path}format -> {CONFIG_NEW}")
        return out
    if isinstance(x, list):
        return [migrate(v, f"{path}[{i}].", parent, done, left) for i, v in enumerate(x)]
    return x


def layout(text):
    data = json.loads(text)
    for indent in (1, 2, 4, None):
        for nl in ("\n", ""):
            if json.dumps(data, indent=indent) + nl == text:
                return indent, nl
    return None


def migrate_json(path, dry):
    text = path.read_text()
    try:
        data = json.loads(text)
    except ValueError:
        return [], [], False
    done, left = [], []
    out = migrate(data, done=done, left=left)
    reformatted = False
    if done and not dry:
        lay = layout(text)
        reformatted = lay is None
        indent, nl = lay if lay else (1, "\n")
        path.write_text(json.dumps(out, indent=indent) + nl)
    return done, left, reformatted


def migrate_jsonl(path, dry):
    lines = path.read_text().splitlines(keepends=True)
    try:
        records = [json.loads(line) if line.strip() else None for line in lines]
    except ValueError:
        return [], [], False
    done, left, out, reformatted = [], [], [], False
    for i, (line, r) in enumerate(zip(lines, records)):
        if r is None:
            out.append(line)
            continue
        d = []
        m = migrate(r, f"[{i}].", done=d, left=left)
        done += d
        body = line.rstrip("\n")
        reformatted |= bool(d) and json.dumps(r) != body
        out.append((json.dumps(m) if d else body) + line[len(body):])
    if done and not dry:
        path.write_text("".join(out))
    return done, left, reformatted


def migrate_take(path, dry):
    with np.load(path, allow_pickle=False) as z:
        if "meta" not in z.files:
            return [], [], False
        meta = json.loads(str(z["meta"]))
        arrays = {k: z[k] for k in z.files if k != "meta"}
    done, left = [], []
    out = migrate(meta, done=done, left=left)
    if out.get("format") == 1:
        out["format"] = 2
        done.append("format 1 -> 2")
    if done and not dry:
        buf = io.BytesIO()
        np.savez_compressed(buf, meta=np.array(json.dumps(out)), **arrays)
        path.write_bytes(buf.getvalue())
    return done, left, False


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--dry", action="store_true", help="report, write nothing")
    ap.add_argument("--quiet", action="store_true", help="only the totals")
    a = ap.parse_args()
    files = []
    for p in map(Path, a.paths):
        files += [f for g in ("*.json", "*.jsonl", "*.npz") for f in sorted(p.rglob(g))] if p.is_dir() else [p]
    changed = reformatted = 0
    for f in files:
        read = {".npz": migrate_take, ".jsonl": migrate_jsonl}.get(f.suffix, migrate_json)
        done, left, refmt = read(f, a.dry)
        changed += bool(done)
        reformatted += bool(refmt)
        if (done or left) and not a.quiet:
            print(f"{f}: {len(done)} renamed" + (f"; left as they are: {', '.join(left)}" if left else ""))
    print(f"{changed} of {len(files)} files {'would change' if a.dry else 'migrated'}"
          + (f", {reformatted} reformatted (their own layout could not be reproduced)" if reformatted else ""))


if __name__ == "__main__":
    main()
