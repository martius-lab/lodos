# Copyright 2025 The Newton Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Adapted for Lodos from MuJoCo Warp 3.14.0's mujoco_warp/_src/types.py: the parts Lodos uses, its MJ_ constants without
# the prefix (MJ_MINIMP and MJ_MAXIMP keep it); every class keeps its MuJoCo Warp name.

"""The prim collision's own types and constants (MuJoCo Warp's)."""

import enum

import mujoco

import warp as wp


MAXCONPAIR = mujoco.mjMAXCONPAIR


# maximum size (by number of edges) of an horizon in EPA algorithm
MAX_EPAHORIZON = 24


# maximum average number of trianglarfaces EPA can insert at each iteration
MAX_EPAFACES = 5


class SleepState(enum.IntEnum):
  """Sleep state for bodies.

  Attributes:
    STATIC: body is static (world body or mocap)
    ASLEEP: body is asleep
    AWAKE: body is awake
  """

  STATIC = mujoco.mjtSleepState.mjS_STATIC
  ASLEEP = mujoco.mjtSleepState.mjS_ASLEEP
  AWAKE = mujoco.mjtSleepState.mjS_AWAKE


class mat23f(wp.types.matrix(shape=(2, 3), dtype=float)):
  pass


class mat43f(wp.types.matrix(shape=(4, 3), dtype=float)):
  pass


class mat63f(wp.types.matrix(shape=(6, 3), dtype=float)):
  pass


mat23 = mat23f


mat43 = mat43f


mat63 = mat63f
