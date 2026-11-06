# Performance Tuning

Measures the minimum capacities (e.g. for memory allocation) a scene needs and writes it into a **tuning file**
that `make_sim` reads:

```python
sim = make_sim(mjm, B, surfaces=surf, tuning="tools/performance_tuning/tuned/arm_5_sdf.tuning.json")
# or, for a built-in viewer scene
sim = viewer.scenes.catalog.builtin("arm_5_sdf").make_sim(B=4096, tuning=...)
```

**Nothing here is needed to run a simulation.** A scene runs at its defaults without a
tuning file. Those defaults are safe, but not fast. For example, on the five-armadillo drop scene, Lodos
runs 3.4x faster on calibrated capacities.

## The tools

Every tool takes a scene: a built-in viewer scene's name, or an MJCF file the viewer can
load. Every tool runs from a **state set**, so its capacities, timings and accuracy are all
about the same states.

| script | does | writes |
|---|---|---|
| `explore.py` | makes a state set: the scene's starts rolled out, sampled every few steps | `<scene>.states.npz` (+ `.json` report) |
| `calibrate.py` | sizes every capacity at a batch size from a state set | the tuning file's capacities |
| `tile_tuner.py` | picks the tile size of each kind of Lodos segment | the tuning file's `tile_size` |
| `sweep.py` | times any knobs over a grid, interleaved, optionally guarded | a table |
| `profile_scene.py` | times each stage of a step | a table |
| `tune.py` | all of the above for one scene, before and after; `--keep-tiles` takes the file's tile sizes instead of searching them | `tuned/<scene>.tuning.json`, `.report.json` |
| `tune_all.py` | `tune.py` over every built-in scene | `tuned/` |
| `summarize.py` | the table of every tuned scene, before and after | `tuned/README.md` |
| `time_variant.py` | times a variant of a scene's tuned settings (other solver options: the mj mode without the proximal loop, for the Newton comparison), recalibrated at each batch as `tune.py` does | `<out>/<scene>.<label>.json` |


A typical session on a new scene:

```bash
python tools/performance_tuning/explore.py my_scene.xml --out states/my_scene.states.npz --strategy scatter perturb
python tools/performance_tuning/calibrate.py my_scene.xml --states states/my_scene.states.npz --B 4096 --out tuning/my_scene.tuning.json
python tools/performance_tuning/profile_scene.py my_scene.xml --B 4096 --tuning tuning/my_scene.tuning.json
python tools/performance_tuning/sweep.py my_scene.xml --B 4096 --tuning ... --knob solver_kwargs.prox_iters_per_qp=10,5 --guard --recalibrate
# or all of it at once
python tools/performance_tuning/tune.py my_scene.xml --out-dir tuning
```
