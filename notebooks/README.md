# Notebooks

Tutorials for Lodos's Python API. Each one is stored with its outputs, from an RTX 4080.

| notebook | what it shows |
|---|---|
| `01_quickstart` | five armadillos falling in a box across 64 parallel worlds: stepping with CUDA graphs, reading state and contacts, rendering |
| `02_build_a_scene` | the viewer's `cube_5_sdf` assembled from scratch and rolled out |
| `03_modes_and_friction` | a cube on an inclined plane: the solver's different modes and contact groups |
| `04_batched_rollouts` | a nut screwing itself into its bolt under gravity: graph replay against eager stepping, throughput against batch size under either mode, capacity tuning |
| `05_make_a_surface` | making a Lodos surface from a triangle mesh|

To run the notebooks:

```bash
pip install ".[vis,notebooks,mesh]"   # mesh makes surfaces (05); or, with uv: uv sync --group notebooks
jupyter lab notebooks/
```
