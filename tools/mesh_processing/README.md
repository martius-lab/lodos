# Mesh Processing

Turns a MuJoCo model's meshes into "Lodos surfaces", the collision geometry `lodos.col.sdf` collides: a watertight mesh with isotropically spaced vertices and the baked signed distance field corresponding to it, both stored in one `.npz`
that `lodos.col.sdf.load_surface_data` reads.

**Nothing here is needed to run a simulation.** The built-in scenes ship their surfaces baked, and these tools are provided to make
new ones.

## The steps

Each script implements one step of the pipeline. It reads the previous step's output folder and writes its own output folder beside it, under the input's name
with a suffix appended, so folder names trace how a surface is constructed.

| step | script | does | writes |
|---|---|---|---|
| 1 | `fuse_meshes_by_body.py` | merges each body's meshes, in its body frame, into one watertight solid using CGAL's alpha wrap | `robot_wt/` |
| 2 | `calibrate_cp_decomp.py` | decimates each solid (PaMO), decomposes it into convex hulls (CoACD) at a threshold and hull cap chosen per body in its window, and fuses the hulls | `robot_wt_cp/`, `robot_wt_cp_fused/` |
| 3 | `interactive_remesh.py` | remeshes each solid into isotropically distributed vertices (Instant Meshes), at a vertex count chosen per body in its window | `..._im/` |
| 4 | `fix_nonmanifold_meshes.py` | repairs what the remesh left non-manifold: removes repeated faces, or remeshes at a nearby vertex count | `..._im_man/` |
| 5 | `create_surface_data.py` | "bakes" the mesh by: i) sorting its vertices in space-filling-curve order, ii) measuring signed distance values on a voxel grid, and iii) fits a trilinear/tricubic spline on the measurements | `..._im_man_sdata/` |
| 6 | `pack_spheres.py` | optional: packs spheres inside a baked surface, to further robustify the collision pipeline against deep penetrations (which normally shouldn't happen if the contact parameters, solver settings, and timestep size are chosen appropriately) | `spheres.npz`, beside the bake |

Steps 1-4 write each body in its body frame and, under `<output>_world/`, in the world frame at `qpos0`. Every step
writes a JSON record of its settings, and step 5 copies the earlier records beside the bakes, so a surface carries
how it was made. Every step opens a window on its result, which `--no-view` skips. In steps 2 and 3 the window is
where the settings are chosen, and step 3 writes only when `Export` is pressed.

## Installation

Steps 3-5 run in Lodos's environment (the `mesh` group) with the Instant Meshes binary at
`tools/mesh_processing/.instant-meshes`. Step 2 also needs CoACD and manifold3d (`uv pip install coacd manifold3d`)
and PaMO's container at `tools/mesh_processing/.pamo.sif`. Step 1 needs CGAL and scikit-image on Python 3.12, the
newest Python that CGAL's bindings support, so it runs from an environment of its own. `docs/installation_guide.pdf`
(its section "The mesh pipeline's tools") sets up each of them.

## Running the pipeline

`run_pipeline.py` runs steps 1-5 in order, each with the interpreter that runs it, so from Lodos's environment it
starts at step 2:

```bash
~/venvs/lodos-step1/bin/python tools/mesh_processing/fuse_meshes_by_body.py path/to/robot.xml   # step 1, on Python 3.12
.venv/bin/python tools/mesh_processing/run_pipeline.py path/to/robot.xml --from 2                # steps 2-5
.venv/bin/python tools/mesh_processing/run_pipeline.py path/to/robot.xml --from 2 --replay old_sdata/
# the same without windows, steps 2 and 3 from the choices an earlier run recorded in old_sdata/
```

`--from N` and `--to N` run part of the chain, and `--dry-run` prints the commands. Steps 2 and 3 wait for their
choices; the rest takes a few minutes on an RTX 4080.