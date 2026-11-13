# Lodos

Lodos is a *research codebase* for running massively parallelized simulations of rigid-body dynamics with contact. It is particularly optimized for achieving high-throughput in scenes involving collisions between large numbers of detailed meshes.

The framework builds upon two of our previous works:
* `papers/lodos_v1.pdf`: The first version of the framework, under review. The current codebase incorporates significant extensions beyond this initial version, as described in `papers/lodos_v2.pdf`. These developments will eventually find their way into a follow-up publication.
* `papers/contact_tutorial.pdf`: A tutorial on discrete time-stepping methods for simulating cone-complementarity-based contact dynamics. It concludes by proposing a general proximal minimization blueprint, and this codebase essentially constitutes a concrete implementation.


From an implementation perspective, Lodos is built entirely using [NVIDIA Warp](https://nvidia.github.io/warp/). Its architecture builds upon a core subset of the [MJWarp](https://github.com/google-deepmind/mujoco_warp) codebase, particularly its constraint model and its excellent GPU implementation of a nonsmooth Newton solver with exact line-search.

The main repository is https://github.com/martius-lab/lodos

## How to Get Started?

* Read through `papers/lodos_v1.pdf` and `papers/lodos_v2.pdf` to understand the context around Lodos and its contributions.
* Read through `papers/contact_tutorial.pdf` to understand the theory behind Lodos.
* Read `viewer/README.md` to run interactive simulations of an existing catalogue of scenes.
* Read `bench/README.md` to reproduce the benchmark comparing Lodos to [MJWarp](https://github.com/google-deepmind/mujoco_warp) and [Kamino](https://disneyresearch.github.io/kamino-site/).
* Run the examples in `notebooks/` to familiarize yourself with the API.
* Read `tools/performance_tuning/README.md` to understand how to tune Lodos simulations to maximize throughput.
* Read `tools/mesh_processing/README.md` to understand how to preprocess, or "bake", existing meshes so that they can be used as collision geometries in Lodos.

<br />

* If you are using a coding agent, have it read through the contents of `docs/`. It should find everything it needs to make full use of the Lodos codebase. The docs themselves are AI generated, and are perhaps too long (and unbearable) for human eyes.

## Installation

Lodos is developed and tested on Linux (x86_64) with an NVIDIA GPU. \
The uv installation needs NVIDIA driver 580 or newer (Warp's CUDA 13 build). \
The pip installation needs NVIDIA driver 525 or newer (PyPI's CUDA 12.9 build).

```bash
git clone https://github.com/martius-lab/lodos
cd lodos
```

With uv (the groups `vis` and `mesh` come by default):

```bash
uv sync --locked                 # Python 3.14, Warp 1.16 (CUDA 13 build: NVIDIA driver 580+), MuJoCo 3.14.0
uv sync --locked --all-groups    # + the notebooks group, which a plain `uv sync` removes again
(cd viewer/newton && uv sync --locked)   # optional: Newton 1.7.0's environment, for the viewer's MuJoCo Warp and Kamino and for bench/
```

With pip, from a clone (Python 3.13 or newer):

```bash
pip install .                    # the simulator, the viewer's scenes and their assets
pip install ".[vis]"             # + the live viewer (lodos-viewer) and the recording tools
pip install ".[vis,notebooks]"   # + JupyterLab, for notebooks/
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install ".[mesh]"            # + what making new surfaces needs; torch's CPU build is enough
pip install -e ".[vis,mesh]"     # editable
```
## Contributions
This section provides a concise summary of the main contributions of the Lodos framework. For a more detailed discussion, please refer to the papers in the `papers` directory. 

### Collision Detection

Lodos's main contribution to collision detection is an efficient approach to vertex-SDF collision handling, with computational and memory requirements that scale as $O(\log |V|)$, where $|V|$ denotes the number of vertices.

Lodos deliberately does not implement edge-SDF collisions (even though `papers/lodos_precursor.pdf` proposes a way to do it efficiently). The rationale is that its ability to efficiently handle arbitrarily large numbers of vertices against geometries of arbitrary complexity makes explicit edge-SDF collision handling unnecessary in most practical scenarios. In particular, an edge-SDF collision can often be captured by simply isotropically remeshing a surface at a higher and higher resolution until the corresponding collision is detected through a vertex-SDF query. 

That said, to accommodate less common applications in which explicitly modeling edge-edge contacts is desirable, Lodos also provides an implementation of the standard GJK + EPA + polygon-clipping pipeline used by most modern simulation frameworks, optimized for GPU throughput (with warmstarting and a modified polygon-clipping routine).

### Contact Dynamics
Lodos has two main contributions to contact dynamics.

The first is an efficient primal-form algorithm for solving the Signorini–Coulomb conditions for frictional contact exactly, resulting in improved physical accuracy. In contrast, Kamino solves the same problem using a dual-form algorithm, which scales less favorably with batch size on the GPU. MJWarp, on the other hand, solves a convex relaxation of the problem, giving rise to the well-known artifacts of compliant contact, spurious lift-off, and tangential creeping.

The second contribution is a reparameterization of contact forces that substantially reduces the number of constraints in the contact model. This translates into lower memory requirements and, consequently, higher throughput, particularly in scenes involving large numbers of collision pairs between geometrically detailed, non-convex meshes.


## Live viewer

```bash
lodos-viewer                     # empty window: load a built-in scene or search for an MJCF
lodos-viewer arm_box_6 --sim lodos
lodos-viewer viewer/assets/mjcf/joints_demo.xml
```

(`lodos-viewer` is `python -m viewer`, and it needs the `vis` extra.)

## Benchmarks

`bench/README.md` reproduces the batched-throughput comparisons of `papers/lodos_v2.pdf`
(Section 7): the mj mode against MuJoCo Warp, and the exact mode against Kamino.
