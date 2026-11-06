# Viewer

The command `python -m viewer` simulates one world of a scene in a window. It allows loading a built-in scene or an MJCF file, setting its
state, and playing it under Lodos or, for comparison, under Newton's MuJoCo Warp or Kamino. One can switch the simulator or any option (e.g. solver tolerances or contact model parameters) at any frame.

The full manual is `docs/visualizer.pdf`.

*(Please note that this viewer application is entirely AI coded, and that we have not examined the source code line by line)*

## Install

From the repository root:

```bash
uv sync --locked                          # Lodos-v2's environment, the viewer's among it
(cd viewer/newton && uv sync --locked)    # Newton 1.7.0's, for MuJoCo Warp and Kamino (Python 3.12, about 0.7 GB)
```

The viewer needs an NVIDIA GPU and a display. Without Newton's environment, its sim menu offers Lodos alone.

## Start

```bash
.venv/bin/python -m viewer                                       # an empty window: pick a scene in the Scene panel
.venv/bin/python -m viewer cube_walled_5_sdf                     # a built-in scene, under Lodos
.venv/bin/python -m viewer cube_walled_5_sdf --sim mjwarp        # ... under MuJoCo Warp (Newton)
.venv/bin/python -m viewer cube_walled_5_polytope --sim kamino   # ... under Kamino (Newton)
.venv/bin/python -m viewer arm_walled_20_sdf --seed 1000         # ... from the stacks seed 1000 draws
.venv/bin/python -m viewer --help                                # every option
```

- **Keys.** Space plays and pauses, Right and Left step one frame, F frames the camera, H hides the panels, and
  Escape quits.
- **Takes.** Every run is kept as a *take*. Changing the sim, an option or the state makes a draft from the frame
  on screen, and Play runs the draft as a new take. One start can thus be run under all three sims, and the takes
  compared in the timeline.
- **First switch.** The first switch to a configuration builds it, which takes seconds, and the very first build
  also compiles Warp's kernels. Switching back is free.

## The benchmarks' scenes

Each is a built-in scene by the same name (`viewer.scenes.catalog.BUILTIN`):

| scenes | what | compared with |
|---|---|---|
| `cube_walled_5_sdf`, `cube_walled_5_polytope`, `cube_walled_20_sdf`, `cube_walled_20_polytope` | five cubes dropped in a walled box, or twenty in four stacks | MuJoCo Warp, Kamino |
| `arm_walled_5_sdf`, `arm_walled_5_polytope`, `arm_walled_20_sdf`, `arm_walled_20_polytope` | five armadillos, or twenty in four stacks; as polytopes, 34 convex pieces each | MuJoCo Warp, Kamino |
| `shadow_walled_5_sdf`, `shadow_walled_5_polytope`, `shadow_walled_20_sdf`, `shadow_walled_20_polytope` | five Shadow hands, or twenty in four stacks, with joint limits and friction loss; as polytopes, 70 convex hulls each | MuJoCo Warp; Kamino the five as polytopes |
| `shadow_grasp_sdf`, `shadow_grasp_polytope` | the Shadow hand squeezing and lifting an 81 g toy 100 mm in a delicate four-finger prismatic grasp, position controls replayed open loop from a recording | MuJoCo Warp; Kamino as polytopes |
| `nist_threading_m8` | A Franka arm threading an M8 nut onto its bolt, position controls (recorded from the NIST policy in Isaac Lab) replayed open loop  | MuJoCo Warp |
| `masonry_arch_sdf`, `masonry_arch_polytope` | an arch of 25 stones, the end ones are welded to the ground  | Kamino as polytopes |
| `house_of_cards_sdf`, `house_of_cards_polytope` | a house of 40 cards in five levels | Kamino as polytopes |

The `_sdf` and `_polytope` versions of a scene have the same bodies, inertias and starts.
