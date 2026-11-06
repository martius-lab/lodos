# Benchmark: Lodos's exact mode and Kamino

Lodos in its exact mode against Kamino (Newton's `SolverKamino`, proximal ADMM, on Newton's collision pipeline), on nineteen scenes,
each simulator's capacities tuned and its batch size laddered up to its best. The results are the tables of `report/report.pdf`, and Table 2 of the
extended abstract (`papers/lodos_v2.pdf`).

## Replicating

From the repository root:

```bash
.venv/bin/python -m bench.lodos_kamino.run                       # everything: prepare, tune, ab, ladder
.venv/bin/python -m bench.lodos_kamino.run --phases prepare,tune,ab   # capacities and options alone
.venv/bin/python -m bench.lodos_kamino.run --only house_of_cards_polytope --phases prepare,tune,ab,ladder
.venv/bin/python -m bench.lodos_kamino.report                    # tables.tex, throughput.pdf, report.pdf
```

The runs write to `results/lodos_kamino/` (`LODOS_KAMINO_OUT` overrides it): per scene its starts, settings and
Kamino's Newton package, options and controls (`scenes/`), each configuration's capacities (`caps/`), the execution
options' timings (`ab/`), every timed run (`points/`, an overflowed one kept as `*.overflowed*.json`), each
process's output (`logs/`) and `run.log`. 

## Files

- `common.py`: the scenes, the settings, the lost-world test (the rest is `bench/lodos_mjwarp/common.py`'s).
- `prepare.py`: a scene's starts, settings, and Kamino's package, options and controls.
- `lodos_point.py`: one Lodos measurement (`measure`, `ab`, `point`).
- `kamino_point.py`: one Kamino measurement (`measure`, `ab`, `point`), in Newton's environment.
- `run.py`: the driver.
- `report.py`, `report/report.tex`: the report.
