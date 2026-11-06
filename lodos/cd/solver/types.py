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

"""The constraint solver's own types and constants (MuJoCo Warp's): its contexts, the
constraint types and states."""

import dataclasses
import enum
from typing import Callable, Optional

import mujoco

import warp as wp

from lodos.mjcf.types import Data
from lodos.mjcf.types import Model


MJ_MINIMP = mujoco.mjMINIMP  # minimum constraint impedance


MJ_MAXIMP = mujoco.mjMAXIMP  # maximum constraint impedance


class JointType(enum.IntEnum):
  """Type of degree of freedom.

  Attributes:
    FREE:  global position and orientation (quat)       (7,)
    BALL:  orientation (quat) relative to parent        (4,)
    SLIDE: sliding distance along body-fixed axis       (1,)
    HINGE: rotation angle (rad) around body-fixed axis  (1,)
  """

  FREE = mujoco.mjtJoint.mjJNT_FREE
  BALL = mujoco.mjtJoint.mjJNT_BALL
  SLIDE = mujoco.mjtJoint.mjJNT_SLIDE
  HINGE = mujoco.mjtJoint.mjJNT_HINGE


  # unsupported: PGS


class ConstraintState(enum.IntEnum):
  """State of constraint.

  Attributes:
    SATISFIED: constraint satisfied, zero cost (limit, contact)
    QUADRATIC: quadratic cost (equality, friction, limit, contact)
    LINEARNEG: linear cost, negative side (friction)
    LINEARPOS: linear cost, positive side (friction)
    CONE: square distance to cone cost (elliptic contact)
  """

  SATISFIED = mujoco.mjtConstraintState.mjCNSTRSTATE_SATISFIED
  QUADRATIC = mujoco.mjtConstraintState.mjCNSTRSTATE_QUADRATIC
  LINEARNEG = mujoco.mjtConstraintState.mjCNSTRSTATE_LINEARNEG
  LINEARPOS = mujoco.mjtConstraintState.mjCNSTRSTATE_LINEARPOS
  CONE = mujoco.mjtConstraintState.mjCNSTRSTATE_CONE


class ConstraintType(enum.IntEnum):
  """Type of constraint.

  Attributes:
    EQUALITY: equality constraint
    FRICTION_DOF: dof friction
    FRICTION_TENDON: tendon friction
    LIMIT_JOINT: joint limit
    LIMIT_TENDON: tendon limit
    CONTACT_FRICTIONLESS: frictionless contact
    CONTACT_PYRAMIDAL: frictional contact, pyramidal friction cone
    CONTACT_ELLIPTIC: frictional contact, elliptic friction cone
  """

  EQUALITY = mujoco.mjtConstraint.mjCNSTR_EQUALITY
  FRICTION_DOF = mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF
  FRICTION_TENDON = mujoco.mjtConstraint.mjCNSTR_FRICTION_TENDON
  LIMIT_JOINT = mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT
  LIMIT_TENDON = mujoco.mjtConstraint.mjCNSTR_LIMIT_TENDON
  CONTACT_FRICTIONLESS = mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS
  CONTACT_PYRAMIDAL = mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL
  CONTACT_ELLIPTIC = mujoco.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC


class vec6i(wp.types.vector(length=6, dtype=int)):
  pass


class vec16f(wp.types.vector(length=16, dtype=float)):
  pass


class vec16i(wp.types.vector(length=16, dtype=int)):
  pass


vec16 = vec16f


@dataclasses.dataclass
class InverseContext:
  """Workspace arrays for inverse dynamics."""

  Jaref: wp.array2d[float]
  search_dot: wp.array[float]
  done: wp.array[bool]
  quad_changed_ids: wp.array2d[int]
  quad_changed_count: wp.array[int]
  state_changed_count: wp.array[int]
  ls_exhausted: wp.array[bool]
  # the full-coordinate Data, set by solve_compact (None natively)
  compact_m_full: Optional["Model"] = None
  compact_d_full: Optional["Data"] = None


@dataclasses.dataclass
class SolverContext:
  """Workspace arrays for constraint solver."""

  Jaref: wp.array2d[float]
  search_dot: wp.array[float]
  done: wp.array[bool]
  grad: wp.array2d[float]
  grad_dot: wp.array[float]
  newton_decrement: wp.array[float]
  Mgrad: wp.array2d[float]
  search: wp.array2d[float]
  mv: wp.array2d[float]
  jv: wp.array2d[float]
  quad: wp.array2d[wp.vec3]
  alpha: wp.array[float]
  grad_scale: wp.array[float]
  state_changed_count: wp.array[int]
  improvement: wp.array[float]
  ls_exhausted: wp.array[bool]
  search_unchanged: wp.array[bool]
  prev_grad: wp.array2d[float]
  prev_Mgrad: wp.array2d[float]
  beta: wp.array[float]
  h: wp.array3d[float]
  hfactor: wp.array3d[float]
  quad_changed_ids: wp.array2d[int]
  quad_changed_count: wp.array[int]
  # exact-contact (prox) solver state; empty unless opt.prox_enable
  # (nworld, njmax) proximal anchors z, efc-row layout
  prox_z: wp.array2d[float] = dataclasses.field(default_factory=lambda: wp.empty((0, 0), dtype=float))
  # (nworld, njmax) stabilization-only reference accelerations aref0
  prox_aref0: wp.array2d[float] = dataclasses.field(default_factory=lambda: wp.empty((0, 0), dtype=float))
  # (naconmax,) De Saxce corrections Gamma_n = mu*||sigma_t||
  prox_gamma: wp.array[float] = dataclasses.field(default_factory=lambda: wp.empty((0,), dtype=float))
  # (naconmax,) opt.prox_soft: the proximal weight rho_i and the normal row's damping b
  prox_rho_i: wp.array[float] = dataclasses.field(default_factory=lambda: wp.empty((0,), dtype=float))
  prox_bdamp: wp.array[float] = dataclasses.field(default_factory=lambda: wp.empty((0,), dtype=float))
  # (nworld,) max |f - z| anchor residual (atomic-max accumulator)
  prox_dz: wp.array[float] = dataclasses.field(default_factory=lambda: wp.empty((0,), dtype=float))
  # (nworld,) max |f| anchor scale
  prox_znorm: wp.array[float] = dataclasses.field(default_factory=lambda: wp.empty((0,), dtype=float))
  # (nworld,) max |dGamma| De Saxce residual
  prox_dsigma: wp.array[float] = dataclasses.field(default_factory=lambda: wp.empty((0,), dtype=float))
  # (nworld,) max Gamma De Saxce scale
  prox_signorm: wp.array[float] = dataclasses.field(default_factory=lambda: wp.empty((0,), dtype=float))
  # (nworld,) sticky per-world convergence flags (frozen worlds skip all work)
  prox_done: wp.array[bool] = dataclasses.field(default_factory=lambda: wp.empty((0,), dtype=bool))
  # (nworld,) Newton iterations accumulated across the step's inner solves
  prox_niter: wp.array[int] = dataclasses.field(default_factory=lambda: wp.empty((0,), dtype=int))
  # (1,) count of not-yet-frozen worlds (seeds the Newton loop's nsolving)
  prox_nactive: wp.array[int] = dataclasses.field(default_factory=lambda: wp.empty((0,), dtype=int))
  # the full-coordinate Data, set by solve_compact (None natively)
  compact_m_full: Optional["Model"] = None
  compact_d_full: Optional["Data"] = None
  # a caller's solve of the Newton system, in place of the dense Hessian's (nlsp.solve's
  # newton_solve): reads ctx.grad and the efc state, writes ctx.search, ctx.search_dot and
  # ctx.newton_decrement; None natively
  newton_solve: Optional[Callable] = None
