#!/usr/bin/env python3
"""
The half of a PaMO run that happens inside the container. Not importable outside it.

    python pamo_worker.py jobs.json result.json INDEX

PaMO wants Python 3.10, numpy 1.26, triton 2.0, a CUDA build of its own
simplification kernel and a Warp fork built from source, so it cannot share an
environment with lodos and runs in its published image instead.

**One body per process, by index.** `PaMO.__init__` takes the mesh and builds a
`Stage3System` holding ~400 MB of CUDA arrays, so the object is per-body by
construction. Run over a batch in one process they accumulate and the third body
dies allocating; dropped explicitly with `torch.cuda.empty_cache()` the teardown
corrupts the context instead ("CUDA free failed: cudaErrorIllegalAddress") -- the
Warp fork's arrays and torch's allocator do not survive each other's cleanup. A
fresh process per body sidesteps both, and `singularity exec` costs 2.8 s.

One API trap, from `simp_cuda/pamo/__init__.py:97`:

    self.target_faces = max(int(ratio * len(triangles)), min_verts)

`min_verts` floors **faces**, not vertices, despite the name -- and its default is
1e10, so a `run()` that does not pass it explicitly silently does no
simplification at all. The driver sends both numbers and they agree, which pins
the target exactly rather than leaving it to `int()` rounding on the ratio.
"""

import json
import sys
import time
import traceback

import numpy as np
import torch
import trimesh
from pamo import PaMO


def main():
    jobs_path, result_path, index = sys.argv[1], sys.argv[2], int(sys.argv[3])
    j = json.load(open(jobs_path))[index]

    row = {"body": j["body"]}
    try:
        mesh = trimesh.load(j["input"], force="mesh", process=False)
        pamo = PaMO(mesh, use_stage1=j["stage1"], use_stage3=j["stage3"])

        t = time.time()
        v, f = pamo.run(
            torch.from_numpy(mesh.vertices).float().cuda(),
            torch.from_numpy(mesh.faces).int().cuda(),
            ratio=j["ratio"],
            min_verts=j["target_faces"],
        )
        row["seconds"] = round(time.time() - t, 2)

        out = trimesh.Trimesh(vertices=np.asarray(v), faces=np.asarray(f), process=False)
        out.export(j["output"])
        row.update(v_out=int(len(out.vertices)), f_out=int(len(out.faces)))
    except Exception:
        row["error"] = traceback.format_exc(limit=3)

    json.dump(row, open(result_path, "w"), indent=2)
    print(f"[worker] {j['body']}: {row.get('v_out', 'FAILED')}", flush=True)


if __name__ == "__main__":
    main()
