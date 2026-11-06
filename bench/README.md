# Benchmarks

This folder contains the batched-throughput comparisons presented in the extended abstract
(`papers/lodos_v2.pdf`, Section 7), and the code needed to reproduce them. There are two
comparisons. Each runs its simulator as Newton 1.7.0 runs it, on Newton's own collision pipeline:

| folder | Lodos-v2 | compared with | scenes | in the abstract |
|---|---|---|---|---|
| `lodos_mjwarp/` | the Mujoco mode | MuJoCo Warp: Newton's `SolverMuJoCo`, on upstream MuJoCo Warp 3.15 | 15 | Table 1 |
| `lodos_kamino/` | the exact mode | Kamino: Newton's `SolverKamino` | 19, Kamino on 8 | Table 2 |

Each folder's README says what is compared and how it is measured. Its `report/report.pdf` has, as tables, the exact settings,
the capacities, the execution options and the throughput at every batch size for each scene. The two tables from the extended abstract
are collated from the two reports' own tables, and `report/best.json` holds the corresponding records.

The file `lodos_setting.py` is used by both comparisons to set Lodos up.

## Reproducing the results

### 1. What it takes

- Linux 
- [uv](https://docs.astral.sh/uv/) (`docs/installation_guide.pdf`).
- pdflatex
- An NVIDIA GPU whose driver runs CUDA 13. The committed results are from an RTX 4080 with 16 GB.
- Patience. The committed runs took about 11 hours for the MuJoCo Warp comparison and about 8 hours for the Kamino one.

### 2. Install the two environments

From the repository root:

```bash
uv sync --locked                          # Lodos-v2's environment
(cd viewer/newton && uv sync --locked)    # Newton 1.7.0's, for MuJoCo Warp and Kamino: Python 3.12, about 0.7 GB
```

Lodos runs in the first environment and the two simulators from Newton run in the second (`viewer/newton/.venv/bin/python`,
`NEWTON_PY` in `lodos_mjwarp/common.py`). Every measurement runs in a process of its own.

### 3. Run the two comparisons

From the repository root, please run (one after the other, since each needs the GPU to itself):

```bash
.venv/bin/python -m bench.lodos_mjwarp.run      # Table 1: Lodos-v2's mj mode and MuJoCo Warp, 15 scenes
.venv/bin/python -m bench.lodos_kamino.run      # Table 2: Lodos-v2's exact mode and Kamino, 19 scenes
```

The output goes to `results/lodos_mjwarp/` and `results/lodos_kamino/` (`LODOS_MJWARP_OUT` and `LODOS_KAMINO_OUT`
change where):

- `scenes/`: each scene's starts and Newton package;
- `caps/`: each configuration's capacities;
- `ab/`: the execution options' timings;
- `points/`: every timed run;
- `logs/`: each process's output;
- `run.log`: the progress.

### 4. Build the tables and compare

```bash
.venv/bin/python -m bench.lodos_mjwarp.report --out results/lodos_mjwarp/report
.venv/bin/python -m bench.lodos_kamino.report --out results/lodos_kamino/report
```

Each command writes three files from the runs:

- `tables.tex`: the report's tables;
- `throughput.pdf`: each scene's throughput against the batch size;
- `best.json`: each configuration's best batch size and world-steps per second.

Once the run is complete, you can compare `best.json` with the committed `bench/lodos_mjwarp/report/best.json` (or `lodos_kamino`'s), which holds the
abstract's numbers.

Without `--out`, the commands overwrite the committed files in `bench/*/report/` and, where pdflatex is installed,
compile `report.pdf` there.

## Looking at the scenes

Every scene here is a built-in scene for the live viewer, by the same name. `viewer/README.md` shows how to open each
one under Lodos, MuJoCo Warp or Kamino with these settings, and from any world's start.
