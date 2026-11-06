# viewer/newton

Newton's own solvers on the viewer's built-in scenes, for comparison: Newton 1.7.0's
`SolverMuJoCo` (upstream MuJoCo Warp 3.15) and `SolverKamino`, both on Newton's own
`CollisionPipeline`. Nothing here imports `lodos`. The environment is a uv project of its own,
because Newton 1.7.0 brings MuJoCo 3.15, upstream MuJoCo Warp 3.15 and Warp 1.18, where the main
environment pins MuJoCo 3.14.0 and Warp 1.16.0:

```bash
cd viewer/newton && uv sync --locked     # Python 3.12, newton[sim]==1.7.0, trimesh, scipy (about 0.7 GB)
```

## What runs where

- `viewer/newton_link.py` (main environment) writes a scene as a **package**: its MJCF;
  each Lodos surface's triangle mesh, its bake's voxel size and its contact parameters; each
  Lodos plane's contact parameters; the time step, the control and the scene's start.
- `newton_scene.py` builds the Newton model from a package, converts states between MuJoCo's
  layout and Newton's (as `SolverMuJoCo`'s own kernels do), and steps it.
- `runner.py` serves the live viewer: `python -m viewer SCENE --sim mjwarp`
  (or `kamino`, or the sim menu) starts it in this environment and exchanges states
  with it every step.
- The benchmarks' Newton side runs in this environment too: `bench/lodos_mjwarp/newton_point.py`
  and `bench/lodos_kamino/kamino_point.py` time a scene's package through `newton_scene.py`.

## How the scenes are imported

As close to the MJCF's own meaning as Newton allows. These are the only changes, each needed
for Newton to run the scenes as MuJoCo reads them:

- **The MJCF's MuJoCo settings come through, under MuJoCo Warp.** SolverMuJoCo's custom
  attributes are registered before the import (`builder_for`), which carries the MJCF's
  `<option>`s, each geom's solref, solimp, condim and priority, the joints' armature, damping
  and limits, and its actuators. Without them Newton runs its own defaults: pyramidal cones
  where every built-in scene asks for elliptic ones, impratio 1 where the SHARPA and Shadow hand
  scenes ask for 10, default contact softness. Where an MJCF leaves an option out, Newton's
  default is its own (implicitfast, not MuJoCo's Euler), so the solver gets the options as MuJoCo
  compiles them.
- **Two of the importer's readings are replaced by MuJoCo's compiled model** (`_mujoco_params`).
  It fills `solref="0.01"` to (0.01, 0) where MuJoCo's missing damping ratio is 1, which left the
  SHARPA fingers' contacts undamped; and it limits a joint that has a `range` and
  `limited="false"`, every joint of the SHARPA hands without `_jlimit`.
- **Gravity is MuJoCo's**, which the importer leaves at Newton's default: none where the MJCF
  turns it off (the Shadow hand's grasp of the Kong toy).
- **Planes on bodies MuJoCo welds to the world are the world's** (`_static_planes_to_world`).
  The walls and the incline's slope are planes on bodies without joints, which Newton's importer
  keeps as bodies of their own on fixed joints, and SolverMuJoCo takes a plane only on the world:
  ten scenes did not build without it.
- **The polytope scenes' convex pieces are convex hulls.** MuJoCo collides every mesh geom as
  its convex hull. Newton's importer makes them triangle meshes, which on two SHARPA hands
  produced over 10,000 contacts per world and diverged.
- **Each Lodos surface is a mesh shape with Newton's mesh SDF** at the surface's own bake's
  voxel size, or at the settings the scene names for Newton (Isaac Lab's, for the NIST parts).
  It takes the friction of its Lodos pairs (the smallest, since a contact takes the larger of its
  two shapes'), and their solref and solimp where its pairs agree, else those of the geom it is
  keyed to. That geom collides with nothing, as in the scene.
- **Every colliding shape's contact gap is 1 % of the scene's typical body size**, unless its
  MJCF geom has a gap of its own (the NIST demo's). Newton's default, 0.1 m, makes contacts
  between every pair of shapes within 10 cm: 5,185 within one SHARPA hand, where MuJoCo
  (margin 0) has a handful. The viewer's panel sets it.
- **The bodies' masses and inertias are MuJoCo's.** Kamino refuses a moving body without mass,
  so under Kamino the grasp's massless wrist gets one gram.
- **Under Kamino the servos are Newton's position drives** with MuJoCo's compiled gains: the
  hands' actuators are `<general>`, which Newton's importer does not make drives of. What Kamino
  does not model, the MJCF's forces on sites and its gravity compensation, is applied as forces
  on the bodies each step.
- **Capacities are measured.** Contacts per world: twice the busiest world's over the starting
  states, at least 256, for the pipeline, MuJoCo Warp's `nconmax` and Kamino's
  `max_contacts_per_world`; MuJoCo Warp's constraint rows (`njmax`) are what those contacts
  need. Newton's own estimates come from the geometry: Kamino's overflowed 32-bit array sizes on
  the larger scenes, and MuJoCo Warp's rows asked for 23 GB at a thousand worlds of the five
  armadillos. The measurement collides the states 64 worlds at a time, since the pipeline's
  default capacity grows with the worlds. The mesh contacts' triangle pairs are 4,096 per world
  (Newton's default, a million in all, is full at a thousand worlds of twenty armadillos).
- **Everything else is Newton's default**, among it the broad phase (explicit: every pair of
  shapes that may collide, precomputed; `options["broad_phase"]` takes Newton's sweep-and-prune
  or all-pairs one instead, as one of the benchmarks does) and Kamino's configuration from the
  model.
