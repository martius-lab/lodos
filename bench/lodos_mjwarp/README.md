# Benchmark: Lodos's mj mode and MuJoCo Warp

Lodos in its mj mode
and MuJoCo Warp as Newton runs it (Newton's `SolverMuJoCo` on Newton's collision pipeline), on fifteen scenes,
each simulator's capacities tuned and its batch size laddered up to its best. The results are the tables of
`report/report.pdf`, and Table 1 of the extended abstract (`papers/lodos_v2.pdf`); `bench/README.md` says how they were measured.

## Replicating

From the repository root:

```bash
.venv/bin/python -m bench.lodos_mjwarp.run                       # everything: prepare, tune, ab, ladder (~11 h)
.venv/bin/python -m bench.lodos_mjwarp.run --only cube_walled_5_sdf --phases prepare,tune,ab,ladder
.venv/bin/python -m bench.lodos_mjwarp.run --only arm_walled_20_polytope shadow_walled_20_polytope \
    --configs mjwarp_sap --phases tune,ladder                    # MuJoCo Warp under SAP alone, as rerun
.venv/bin/python -m bench.lodos_mjwarp.report                    # tables.tex, throughput.pdf, report.pdf
```

The runs write to `results/lodos_mjwarp/` (`LODOS_MJWARP_OUT` overrides it): per scene its starts, Newton package
and options (`scenes/`), each configuration's capacities (`caps/`), the execution options' timings (`ab/`), every
timed run (`points/`, an overflowed one kept as `*.overflowed*.json`), each process's output (`logs/`) and
`run.log`.

## Files

- `common.py`: the scenes, the settings, the starts.
- `prepare.py`: a scene's starts, Newton package, options and controls.
- `lodos_point.py`: one Lodos measurement (`measure`, `ab`, `point`).
- `newton_point.py`: one MuJoCo Warp measurement (`measure`, `point`), in Newton's environment.
- `run.py`: the driver.
- `report.py`, `report/report.tex`: the report.
- `profile/`: where each simulator's step spends its time, kernel (recorded in `docs/speedup_analysis.pdf`)
