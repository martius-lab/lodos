"""
Driving PaMO from outside its container.

PaMO remeshes to a watertight manifold, simplifies with a guarantee that the
result stays intersection-free, then projects back onto the original to undo the
shift and restore sharp features. The guarantee is why it is here: quadric
collapse at this budget tears most of a hand's bodies open, because the watertight
solids are genus > 0 -- the knuckle rings reach genus 10 -- so there are handles
to destroy and destroying one tears the surface.

The budget is areal, not a ratio. A ratio is the knob every simplifier offers and
the wrong one: bodies span 15 to 318 cm2, so one ratio gives the palm eight times
the contact resolution of a fingertip. Fixing the density instead gives every part
the same spacing between contact candidates.

Named `pamo_driver`, not `pamo`: `pamo_worker.py` does `from pamo import PaMO` with
its own directory first on `sys.path`, so a module here called `pamo.py` shadows the
package it is trying to reach and every container run dies on the import.

PaMO stops early rather than tearing, so the vertex counts it returns overshoot
the targets -- most on the parts that cannot be represented at the asked-for count
without self-intersecting, which is the guarantee doing its job.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
SIF = HERE / ".pamo.sif"


def target_vertices(mesh, spacing, min_verts=64, max_verts=512):
    return int(np.clip(mesh.area / spacing**2, min_verts, max_verts))


def run_batch(meshes, out_dir, sif=SIF, spacing=5.0e-3, min_verts=64, max_verts=512,
              stage1=True, stage3=True, tmp=None, log=print):
    """
    `{name: Trimesh}` decimated by PaMO, written to `out_dir` and returned.

    One `singularity exec` per body; see `pamo_worker.py` for why.
    """
    if not Path(sif).exists():
        raise SystemExit(
            f"PaMO container not found at {sif}\n"
            f"    singularity pull {sif} docker://sarahwei0210/pamo:0.0.2"
        )

    # Resolved, not just wrapped: the worker runs with its own directory as `cwd`, so a
    # relative output path would be written relative to `tools/mesh_processing/` instead of to
    # the caller's directory, and every job would fail on a path that does not exist.
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tmp or HERE / ".tmp")
    tmp.mkdir(parents=True, exist_ok=True)

    jobs = []
    for name, mesh in meshes.items():
        src = tmp / f"in_{name}.obj"
        mesh.export(src)
        tv = target_vertices(mesh, spacing, min_verts, max_verts)
        jobs.append(dict(
            body=name, input=str(src), output=str(out_dir / f"{name}.obj"),
            target_verts=tv, target_faces=2 * tv, ratio=2 * tv / len(mesh.faces),
            stage1=stage1, stage3=stage3,
        ))
    jobs_path = tmp / "pamo_jobs.json"
    jobs_path.write_text(json.dumps(jobs, indent=2))

    results, out = {}, {}
    for i, j in enumerate(jobs):
        result_path = tmp / "pamo_result.json"
        proc = subprocess.run(
            ["singularity", "exec", "--nv", "-B", f"{HERE}:{HERE}", str(sif),
             "python", str(HERE / "pamo_worker.py"), str(jobs_path), str(result_path), str(i)],
            capture_output=True, text=True, cwd=str(HERE),
        )
        if proc.returncode != 0 or not result_path.exists():
            tail = (proc.stderr or proc.stdout or "no output").strip().splitlines()[-1:]
            log(f"  {j['body']:<26s} container exited {proc.returncode}: {tail[0][:80] if tail else ''}")
            results[j["body"]] = {"body": j["body"], "error": tail[0] if tail else "no output"}
            continue
        row = json.loads(result_path.read_text())
        result_path.unlink()
        results[row["body"]] = row
        if "error" in row:
            log(f"  {j['body']:<26s} FAILED: {row['error'].strip().splitlines()[-1][:70]}")
            continue
        out[j["body"]] = trimesh.load_mesh(j["output"], process=False)
        log(f"  {j['body']:<26s} {len(meshes[j['body']].vertices):7d} -> {row['v_out']:5d} V"
            f"  (target {j['target_verts']})  {row['seconds']:5.1f}s")

    for j in jobs:
        Path(j["input"]).unlink(missing_ok=True)
    jobs_path.unlink(missing_ok=True)
    (out_dir / "pamo.json").write_text(json.dumps(list(results.values()), indent=2))
    return out
