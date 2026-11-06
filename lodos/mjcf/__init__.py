"""
The MJCF model and data on the device, which the prim collision (`lodos.col.prim`) and the constraint
solver (`lodos.cd.solver`) share; adapted from MuJoCo Warp 3.14.0's mujoco_warp/_src/types.py and io.py
(Apache-2.0). Imports no submodule.

  types   Model, Data and Option, and the enums and constants more than one package uses
  io      put_model and put_data (from MuJoCo's MjModel and MjData), override_model, the capacity
          estimates
"""
