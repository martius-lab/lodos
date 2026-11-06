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
# Adapted for Lodos from MuJoCo Warp 3.14.0's mujoco_warp/_src/solver.py, with the parts of island.py, smooth.py and
# support.py it uses inlined, Lodos's proximal solve fused in and Lodos's Hessian summed in shared memory added
# (opt.shared_hessian). Every function and class from MuJoCo Warp keeps its MuJoCo Warp name; one from another file
# than solver.py says so above it, as does each of the proximal solve's.

"""Constraint solve: nonlinear least squares with proximal minimization (nlsp).

MuJoCo Warp's constraint solver (solver.py) and Lodos's exact-contact prox solver,
fused. `solve` runs the prox solver's loop around the Newton solve under opt.prox_enable
(`solve_prox`), MuJoCo's CG or Newton solve otherwise.

Constraint solver: CG and Newton iterations with an exact linesearch over efc rows.

Minimizes MuJoCo's soft-constraint cost over accelerations. Each iteration forms the
gradient grad = Ma - qfrc_smooth - qfrc_constraint and, for Newton, the Hessian
H = M + J.T @ diag(D * active) @ J (plus elliptic cone curvature), solves for a search
direction, and takes an exact linesearch step along it.

Conventions used throughout: a linesearch "point" is a wp.vec3 (cost, slope, curvature)
of the total cost along the search ray at a step alpha; a "quad" is a wp.vec3 of
quadratic coefficients (c0, c1, c2) meaning cost = c0 + c1*alpha + c2*alpha^2.
Per-world scalars are (nworld,), efc row arrays are (nworld, njmax), the dense Jacobian
is (nworld, njmax_pad, nv_pad) (CSR arrays J_rownnz/J_rowadr/J_colind when sparse), and
the Newton Hessian ctx.h is (nworld, nv_pad, nv_pad).

Exact contact solve: proximal-point + De Saxce iterations around the Newton solver.

Solves the cone complementarity problem exactly (no impedance regularization of the
contact rows) by iterating the Newton solver on a sequence of rewritten (D, aref):

  D(contact rows)  = 1 / (rho_hat * invweight)                    [prox weight dt/rho_i]
  aref(contact i)  = aref0_i - Gamma_i(sigma)/dt + z_i/D_i

where aref0 is the stabilization-only reference: on the normal row the Baumgarte bias
alpha*min(0, (pos + slop)/dt) + max(0, pos/dt), with slop = opt.prox_baumgarte_slop the
penetration it leaves alone, or none with opt.prox_baumgarte false (the velocity-level
target), and on the tangent rows the pure -v_t/dt. Gamma(sigma) =
mu*||sigma_t|| on the normal row is the De Saxce correction at the last outer iterate
sigma = J(qvel + dt*qacc), and z is the proximal anchor, updated to the contact force f*
after every inner solve.  At the fixed point (z, sigma), f* solves the unregularized CCP
with exact Coulomb friction (the contact tutorial, papers/contact_tutorial.pdf, Section 5.2.6).

Requires Newton solver, elliptic cone, condim <= 3, no flex, no sleeping.  Isotropic
effective friction mu = friction[0] * impratio_invsqrt is assumed (guaranteed for
geom-derived contacts; validated for explicit pairs in put_model).
"""

import dataclasses
from math import ceil
from typing import Any
from typing import Optional
from typing import Tuple

import warp as wp


from lodos.cd.solver.block_cholesky import create_blocked_cholesky_augmented_factorize_solve_newton_func
from lodos.cd.solver.block_cholesky import create_blocked_cholesky_factorize_solve_func
from lodos.cd.solver.block_cholesky import create_blocked_cholesky_solve_newton_func
from lodos.cd.solver.block_cholesky import solve_search_sums
from lodos.cd.solver.constraint import _contact_kbimp
from lodos.cd.solver.types import ConstraintState
from lodos.cd.solver.types import ConstraintType
from lodos.cd.solver.types import InverseContext
from lodos.cd.solver.types import SolverContext
from lodos.cd.solver.types import vec16
from lodos.cd.solver.types import vec6i
from lodos.mjcf.types import MAXVAL
from lodos.mjcf.types import MINVAL
from lodos.mjcf.types import Q_LD_BLOCK_COMPACT
from lodos.mjcf.types import Q_LD_BLOCK_SPARSE
from lodos.mjcf.types import TILE_SIZE_JTDAJ_DENSE
from lodos.mjcf.types import ConeType
from lodos.mjcf.types import ContactType
from lodos.mjcf.types import Data
from lodos.mjcf.types import DisableBit
from lodos.mjcf.types import EnableBit
from lodos.mjcf.types import EqType
from lodos.mjcf.types import Model
from lodos.mjcf.types import ObjType
from lodos.mjcf.types import OverflowType
from lodos.mjcf.types import SolverType
from lodos.mjcf.types import TileSet
from lodos.mjcf.types import vec5
from lodos.utils import math
from lodos.utils.kernels import cache_kernel
from lodos.utils.kernels import event_scope

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


# From MuJoCo Warp's support.py.

# From mujoco_warp/_src/support.py.
@cache_kernel
def mul_m_kernel(check_skip: bool):
  @wp.kernel(module="unique", grid_stride=False)
  def _mul_m(
    # Model:
    M_mulm_rowadr: wp.array[int],
    M_mulm_col: wp.array[int],
    M_mulm_madr: wp.array[int],
    # Data in:
    M_in: wp.array2d[float],
    # In:
    vec: wp.array2d[float],
    skip: wp.array[bool],
    # Out:
    res: wp.array2d[float],
  ):
    """Sparse matmul: one thread per DOF, gather-based (no atomics)."""
    worldid, dofid = wp.tid()

    if wp.static(check_skip):
      if skip[worldid]:
        return

    # Gather all contributions (diagonal + off-diagonal).
    acc = float(0.0)
    start = M_mulm_rowadr[dofid]
    end = M_mulm_rowadr[dofid + 1]
    for k in range(start, end):
      col = M_mulm_col[k]
      madr = M_mulm_madr[k]
      acc += M_in[worldid, madr] * vec[worldid, col]

    res[worldid, dofid] = acc

  return _mul_m


# From mujoco_warp/_src/support.py.
@cache_kernel
def mul_m_dense(nv: int, check_skip: bool):
  @wp.kernel(module="unique", grid_stride=False)
  def _mul_m_dense(
    # Data in:
    M_in: wp.array3d[float],  # kernel_analyzer: ignore
    # In:
    vec: wp.array2d[float],
    skip: wp.array[bool],
    # Out:
    res: wp.array2d[float],
  ):
    """Dense matmul for the compact active-DOF inertia block (nworld, nv, nv)."""
    worldid, i = wp.tid()

    if wp.static(check_skip):
      if skip[worldid]:
        return

    acc = float(0.0)
    for j in range(wp.static(nv)):
      acc += M_in[worldid, i, j] * vec[worldid, j]
    res[worldid, i] = acc

  return _mul_m_dense


# From mujoco_warp/_src/support.py.
@event_scope
def mul_m(
  m: Model,
  d: Data,
  res: wp.array2d[float],
  vec: wp.array2d[float],
  skip: Optional[wp.array] = None,
  M: Optional[wp.array] = None,
):
  """Multiply vectors by inertia matrix; optionally skip per world.

  Args:
    m: The model containing kinematic and dynamic information (device).
    d: The data object containing the current state and output arrays (device).
    res: Result: M @ vec.
    vec: Input vector to multiply by M.
    skip: Per-world bitmask to skip computing output.
    M: Input matrix: M @ vec.
  """
  check_skip = skip is not None
  skip = skip or wp.empty(0, dtype=bool)

  if M is None:
    M = d.M

  if M.ndim == 3:
    # Dense compact active-DOF block (nworld, nv, nv) used by the compact solver.
    wp.launch(
      mul_m_dense(m.nv, check_skip),
      dim=(d.nworld, m.nv),
      inputs=[M, vec, skip],
      outputs=[res],
    )
  else:
    wp.launch(
      mul_m_kernel(check_skip),
      dim=(d.nworld, m.nv),
      inputs=[m.M_mulm_rowadr, m.M_mulm_col, m.M_mulm_madr, M, vec, skip],
      outputs=[res],
    )


# From MuJoCo Warp's smooth.py.

# From mujoco_warp/_src/smooth.py.
@cache_kernel
def _solve_LD_sparse_fused(nv: int, nlevels: int):
  """Fused sparse backsubstitution: UP + diag + DOWN in one kernel."""

  @wp.func_native(snippet="WP_TILE_SYNC();")
  def _syncthreads():
    pass

  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Model:
    qLD_block_adr: wp.array[int],
    # In:
    L: wp.array2d[float],
    D: wp.array2d[float],
    all_updates: wp.array[wp.vec3i],
    level_offsets: wp.array[int],
    y: wp.array2d[float],
    # Out:
    x_out: wp.array2d[float],
  ):
    worldid, tid = wp.tid()
    NV = wp.static(nv)
    NLEVELS = wp.static(nlevels)
    BLOCK_DIM = wp.block_dim()

    # Copy y to x_out for sparse-block dofs only.
    for dofid in range(tid, NV, BLOCK_DIM):
      if qLD_block_adr[dofid] == Q_LD_BLOCK_SPARSE:
        x_out[worldid, dofid] = y[worldid, dofid]
    _syncthreads()

    # Forward substitution (all_updates only references sparse-block dofs)
    for level in range(NLEVELS):
      level_idx = NLEVELS - 1 - level
      level_offset = level_offsets[level_idx]
      level_size = level_offsets[level_idx + 1] - level_offset

      for u in range(tid, level_size, BLOCK_DIM):
        update = all_updates[level_offset + u]
        i, k, Madr_ki = update[0], update[1], update[2]
        wp.atomic_sub(x_out[worldid], i, L[worldid, Madr_ki] * x_out[worldid, k])
      _syncthreads()

    # Diagonal multiply (sparse-block dofs only)
    for dofid in range(tid, NV, BLOCK_DIM):
      if qLD_block_adr[dofid] == Q_LD_BLOCK_SPARSE:
        x_out[worldid, dofid] *= D[worldid, dofid]
    _syncthreads()

    # Backward substitution
    for level in range(NLEVELS):
      level_idx = level
      level_offset = level_offsets[level_idx]
      level_size = level_offsets[level_idx + 1] - level_offset

      for u in range(tid, level_size, BLOCK_DIM):
        update = all_updates[level_offset + u]
        i, k, Madr_ki = update[0], update[1], update[2]
        wp.atomic_sub(x_out[worldid], k, L[worldid, Madr_ki] * x_out[worldid, i])
      _syncthreads()

  return kernel


# From mujoco_warp/_src/smooth.py.
def _solve_LD_sparse(
  m: Model,
  d: Data,
  L: wp.array2d[float],
  D: wp.array2d[float],
  x: wp.array2d[float],
  y: wp.array2d[float],
):
  """Computes sparse backsubstitution: x = inv(L'*D*L)*y."""
  nlevels = len(m.qLD_updates)
  if wp.get_device().is_cuda:
    dim_block = m.block_dim.solve_LD_sparse_fused
  else:
    # Fallback for CPU
    dim_block = 1

  wp.launch(
    _solve_LD_sparse_fused(m.nv, nlevels),
    dim=(d.nworld, dim_block),
    inputs=[m.qLD_block_adr, L, D, m.qLD_all_updates, m.qLD_level_offsets, y],
    outputs=[x],
    block_dim=dim_block,
  )


# From mujoco_warp/_src/smooth.py.
@wp.func
def _small_cholesky_solve(
  # In:
  block_size: int,
  worldid: int,
  factor_adr: int,
  start: int,
  L_in: wp.array2d[float],
  y_in: wp.array2d[float],
  # Out:
  x_out: wp.array2d[float],
):
  for i in range(block_size):
    value = y_in[worldid, start + i]
    for k in range(i):
      value -= L_in[worldid, factor_adr + k * block_size + i] * x_out[worldid, start + k]
    x_out[worldid, start + i] = value / L_in[worldid, factor_adr + i * block_size + i]

  for reverse_i in range(block_size):
    i = block_size - 1 - reverse_i
    value = x_out[worldid, start + i]
    for k in range(i + 1, block_size):
      value -= L_in[worldid, factor_adr + i * block_size + k] * x_out[worldid, start + k]
    x_out[worldid, start + i] = value / L_in[worldid, factor_adr + i * block_size + i]


# From mujoco_warp/_src/smooth.py.
@cache_kernel
def _small_cholesky_solve_block(block_size: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Model:
    qLD_block_adr: wp.array[int],
    # In:
    block_dof: wp.array[int],
    D_in: wp.array2d[float],
    L_in: wp.array2d[float],
    y_in: wp.array2d[float],
    # Out:
    x_out: wp.array2d[float],
  ):
    worldid, blk = wp.tid()
    start = block_dof[blk]
    size = wp.static(block_size)
    factor_adr = qLD_block_adr[start]
    if factor_adr == Q_LD_BLOCK_COMPACT:
      for i in range(wp.static(block_size)):
        x_out[worldid, start + i] = D_in[worldid, start + i] * y_in[worldid, start + i]
    else:
      _small_cholesky_solve(size, worldid, factor_adr, start, L_in, y_in, x_out)

  return kernel


# From mujoco_warp/_src/smooth.py.
@cache_kernel
def _tile_cholesky_solve_block(tile: TileSet):
  # One diagonal block per (world, block) thread group; no densify, so a 2D grid suffices.
  block_size = tile.size
  block_area = block_size * block_size

  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Model:
    qLD_block_adr: wp.array[int],
    # In:
    block_dof: wp.array[int],
    L_in: wp.array2d[float],
    y: wp.array2d[float],
    # Out:
    x: wp.array2d[float],
  ):
    worldid, blk = wp.tid()
    start = block_dof[blk]

    L = wp.tile_reshape(
      wp.tile_load(L_in[worldid], shape=(block_area,), offset=(qLD_block_adr[start],)), (block_size, block_size)
    )
    rhs = wp.tile_load(y[worldid], shape=(block_size,), offset=(start,))
    sol = wp.tile_cholesky_solve(L, rhs, fill_mode="upper")
    wp.tile_store(x[worldid], sol, offset=(start,))

  return kernel


# From mujoco_warp/_src/smooth.py.
def _solve_blocks(
  m: Model,
  d: Data,
  L: wp.array2d[float],
  D: wp.array2d[float],
  x: wp.array2d[float],
  y: wp.array2d[float],
):
  for tile in m.M_tiles:
    if tile.elemid.size == 0:
      wp.launch(
        _small_cholesky_solve_block(tile.size),
        dim=(d.nworld, tile.adr.size),
        inputs=[m.qLD_block_adr, tile.adr, D, L, y],
        outputs=[x],
        block_dim=m.block_dim.small_cholesky,
      )
    else:
      # The triangular back-substitution is largely sequential, so large blocks prefer fewer threads
      # for better occupancy while moderate blocks still want a couple warps (16/27->64, 60->32).
      block_dim = m.block_dim.cholesky_solve if tile.size <= 40 else 32
      wp.launch_tiled(
        _tile_cholesky_solve_block(tile),
        dim=(d.nworld, tile.adr.size),
        inputs=[m.qLD_block_adr, tile.adr, L, y],
        outputs=[x],
        block_dim=block_dim,
      )


# From mujoco_warp/_src/smooth.py.
def solve_LD(
  m: Model,
  d: Data,
  L: wp.array2d[float],
  D: wp.array2d[float],
  x: wp.array2d[float],
  y: wp.array2d[float],
):
  """Computes backsubstitution for the inertia factorization.

  Compact blocks use reciprocal diagonals, full small blocks use scalar Cholesky, dense blocks use
  tile Cholesky, and sparse blocks use the LDL region.

  Args:
    m: The model containing factorization and sparsity information.
    d: The data object containing workspace and factorization results.
    L: The factor: packed dense region followed by the nC LDL region.
    D: Reciprocal diagonal for compact and sparse blocks.
    x: Output array for the solution.
    y: Input right-hand side array.
  """
  if m.M_tiles:
    _solve_blocks(m, d, L, D, x, y)
  if L.shape[1] > m.qLD_block_total:
    _solve_LD_sparse(m, d, L[:, m.qLD_block_total :], D, x, y)


# From mujoco_warp/_src/smooth.py.
@event_scope
def solve_m(m: Model, d: Data, x: wp.array2d[float], y: wp.array2d[float]):
  """Computes backsubstitution: x = qLD * y.

  Args:
    m: The model containing inertia and factorization information.
    d: The data object containing factorization results.
    x: Output array for the solution.
    y: Input right-hand side array.
  """
  solve_LD(m, d, d.qLD, d.qLDiagInv, x, y)


# From MuJoCo Warp's island.py.

# From mujoco_warp/_src/island.py.
@wp.kernel
def _island_count_dofs(
  # Model:
  dof_treeid: wp.array[int],
  # Data in:
  tree_island_in: wp.array2d[int],
  # Data out:
  dof_island_out: wp.array2d[int],
  island_nv_out: wp.array2d[int],
):
  worldid, dofid = wp.tid()

  island_id = tree_island_in[worldid, dof_treeid[dofid]]
  dof_island_out[worldid, dofid] = island_id
  if island_id >= 0:
    wp.atomic_add(island_nv_out, worldid, island_id, 1)


# From mujoco_warp/_src/island.py.
@wp.kernel
def _island_scan_sizes(
  # Data in:
  nisland_in: wp.array[int],
  island_nv_in: wp.array2d[int],
  # Data out:
  nidof_out: wp.array[int],
  island_idofadr_out: wp.array2d[int],
  island_nefc_out: wp.array2d[int],
  island_iefcadr_out: wp.array2d[int],
):
  worldid = wp.tid()

  nisland = nisland_in[worldid]
  if nisland == 0:
    nidof_out[worldid] = 0
    return

  # Scan DOFs and Constraints
  island_idofadr_out[worldid, 0] = 0
  island_iefcadr_out[worldid, 0] = 0
  for i in range(1, nisland):
    island_idofadr_out[worldid, i] = island_idofadr_out[worldid, i - 1] + island_nv_in[worldid, i - 1]
    island_iefcadr_out[worldid, i] = island_iefcadr_out[worldid, i - 1] + island_nefc_out[worldid, i - 1]

  nidof = island_idofadr_out[worldid, nisland - 1] + island_nv_in[worldid, nisland - 1]
  nidof_out[worldid] = nidof

  # Reset for recount ( island_nv is final after _island_count_dofs and stays put )
  for i in range(nisland):
    island_nefc_out[worldid, i] = 0


# From mujoco_warp/_src/island.py.
@wp.kernel
def _island_map_dofs(
  # Model:
  dof_treeid: wp.array[int],
  tree_dofadr: wp.array[int],
  tree_dofnum: wp.array[int],
  # Data in:
  nidof_in: wp.array[int],
  tree_island_in: wp.array2d[int],
  island_idofadr_in: wp.array2d[int],
  # Data out:
  island_dofadr_out: wp.array2d[int],
  map_dof2idof_out: wp.array2d[int],
  map_idof2dof_out: wp.array2d[int],
  # Out:
  idof_islandid_out: wp.array2d[int],
):
  """Map DOFs in ascending global order for deterministic solver inputs.

  A DOF's rank inside its island is fixed by the tree layout: DOFs are contiguous per
  tree and trees are ordered, so the rank is the DOF count of every preceding tree
  carrying the same label. That closed form makes the ascending order reproducible
  without a serial per-world scan.
  """
  worldid, dofid = wp.tid()

  tree = dof_treeid[dofid]
  island_id = tree_island_in[worldid, tree]

  local_idx = dofid - tree_dofadr[tree]
  for t in range(tree):
    if tree_island_in[worldid, t] == island_id:
      local_idx += tree_dofnum[t]

  if island_id >= 0:
    idof = island_idofadr_in[worldid, island_id] + local_idx
    idof_islandid_out[worldid, idof] = island_id
    if local_idx == 0:
      island_dofadr_out[worldid, island_id] = dofid
  else:
    idof = nidof_in[worldid] + local_idx

  map_dof2idof_out[worldid, dofid] = idof
  map_idof2dof_out[worldid, idof] = dofid


# From mujoco_warp/_src/island.py.
@wp.kernel
def _island_count_constraints(
  # Data in:
  nefc_in: wp.array[int],
  efc_type_in: wp.array2d[int],
  tree_island_in: wp.array2d[int],
  njmax_in: int,
  # In:
  efc_tree_in: wp.array2d[int],
  # Data out:
  efc_island_out: wp.array2d[int],
  island_nefc_out: wp.array2d[int],
  island_ne_out: wp.array2d[int],
  island_nf_out: wp.array2d[int],
):
  worldid, efcid = wp.tid()
  if efcid >= wp.min(njmax_in, nefc_in[worldid]):
    return

  efc_tree = efc_tree_in[worldid, efcid]
  if efc_tree < 0:
    efc_island_out[worldid, efcid] = -1
    return
  island_id = tree_island_in[worldid, efc_tree]
  efc_island_out[worldid, efcid] = island_id

  if island_id >= 0:
    wp.atomic_add(island_nefc_out, worldid, island_id, 1)

    efc_type = efc_type_in[worldid, efcid]
    if efc_type == ConstraintType.EQUALITY:
      wp.atomic_add(island_ne_out, worldid, island_id, 1)
    elif efc_type == ConstraintType.FRICTION_DOF or efc_type == ConstraintType.FRICTION_TENDON:
      wp.atomic_add(island_nf_out, worldid, island_id, 1)


# From mujoco_warp/_src/island.py.
@wp.kernel
def _island_map_constraints(
  # Data in:
  nefc_in: wp.array[int],
  nisland_in: wp.array[int],
  efc_type_in: wp.array2d[int],
  efc_island_in: wp.array2d[int],
  island_ne_in: wp.array2d[int],
  island_nf_in: wp.array2d[int],
  island_iefcadr_in: wp.array2d[int],
  # In:
  njmax_in: int,
  # Data out:
  island_nefc_out: wp.array2d[int],
  map_efc2iefc_out: wp.array2d[int],
  map_iefc2efc_out: wp.array2d[int],
  # Out:
  iefc_islandid_out: wp.array2d[int],
):
  """Map constraints in ascending EFC order within each MuJoCo category.

  One thread owns one island and keeps its three category counters in registers, so the
  ascending scan stays serial per island (and therefore reproducible) while different
  islands and worlds advance in parallel.
  """
  worldid, islandid = wp.tid()
  if islandid >= nisland_in[worldid]:
    return

  iefcadr = island_iefcadr_in[worldid, islandid]
  fadr = iefcadr + island_ne_in[worldid, islandid]
  oadr = fadr + island_nf_in[worldid, islandid]
  ne_mapped = int(0)
  nf_mapped = int(0)
  nother_mapped = int(0)

  nefc = wp.min(njmax_in, nefc_in[worldid])
  for efcid in range(nefc):
    if efc_island_in[worldid, efcid] == islandid:
      efc_type = efc_type_in[worldid, efcid]

      # 1. Determine absolute index ic based on category
      if efc_type == ConstraintType.EQUALITY:
        ic = iefcadr + ne_mapped
        ne_mapped += 1
      elif efc_type == ConstraintType.FRICTION_DOF or efc_type == ConstraintType.FRICTION_TENDON:
        ic = fadr + nf_mapped
        nf_mapped += 1
      else:
        ic = oadr + nother_mapped
        nother_mapped += 1

      # 2. Store mappings
      map_efc2iefc_out[worldid, efcid] = ic
      map_iefc2efc_out[worldid, ic] = efcid
      iefc_islandid_out[worldid, ic] = islandid

  # 3. Reconstruct d.island_nefc from the three category counts
  island_nefc_out[worldid, islandid] = ne_mapped + nf_mapped + nother_mapped


# From mujoco_warp/_src/island.py.
@wp.kernel
def _compute_efc_tree(
  # Model:
  nv: int,
  body_treeid: wp.array[int],
  jnt_dofadr: wp.array[int],
  dof_treeid: wp.array[int],
  geom_bodyid: wp.array[int],
  site_bodyid: wp.array[int],
  eq_type: wp.array[int],
  eq_obj1id: wp.array[int],
  eq_obj2id: wp.array[int],
  eq_objtype: wp.array[int],
  is_sparse: bool,
  # Data in:
  nefc_in: wp.array[int],
  contact_geom_in: wp.array[wp.vec2i],
  efc_type_in: wp.array2d[int],
  efc_id_in: wp.array2d[int],
  efc_J_rownnz_in: wp.array2d[int],
  efc_J_rowadr_in: wp.array2d[int],
  efc_J_colind_in: wp.array3d[int],
  efc_J_in: wp.array3d[float],
  njmax_in: int,
  # Out:
  efc_tree_out: wp.array2d[int],
):
  """Compute the first non-negative tree for each constraint."""
  worldid, efcid = wp.tid()

  if efcid >= wp.min(njmax_in, nefc_in[worldid]):
    return

  efc_type = efc_type_in[worldid, efcid]
  efc_id = efc_id_in[worldid, efcid]

  tree = int(-1)
  use_generic = int(0)

  # equality (connect/weld)
  if efc_type == ConstraintType.EQUALITY:
    eq_t = eq_type[efc_id]

    if eq_t == EqType.CONNECT or eq_t == EqType.WELD:
      b1 = eq_obj1id[efc_id]
      b2 = eq_obj2id[efc_id]

      if eq_objtype[efc_id] == ObjType.SITE:
        b1 = site_bodyid[b1]
        b2 = site_bodyid[b2]

      t1 = body_treeid[b1]
      t2 = body_treeid[b2]
      if t1 >= 0:
        tree = t1
      else:
        tree = t2
    else:
      # JOINT, TENDON, FLEX: generic scan
      use_generic = 1

  # joint friction
  elif efc_type == ConstraintType.FRICTION_DOF:
    tree = dof_treeid[efc_id]

  # joint limit
  elif efc_type == ConstraintType.LIMIT_JOINT:
    tree = dof_treeid[jnt_dofadr[efc_id]]

  # contact
  elif (
    efc_type == ConstraintType.CONTACT_FRICTIONLESS
    or efc_type == ConstraintType.CONTACT_PYRAMIDAL
    or efc_type == ConstraintType.CONTACT_ELLIPTIC
  ):
    geom_pair = contact_geom_in[efc_id]
    g1 = geom_pair[0]
    g2 = geom_pair[1]

    if g1 >= 0 and g2 >= 0:
      t1 = body_treeid[geom_bodyid[g1]]
      t2 = body_treeid[geom_bodyid[g2]]
      if t1 >= 0:
        tree = t1
      else:
        tree = t2
    else:
      # flex contacts: generic scan
      use_generic = 1

  else:
    # generic: scan Jacobian row
    use_generic = 1

  if use_generic:
    count = nv
    rowadr = 0
    if is_sparse:
      count = efc_J_rownnz_in[worldid, efcid]
      rowadr = efc_J_rowadr_in[worldid, efcid]

    for i in range(count):
      dof = i
      if is_sparse:
        sparseid = rowadr + i
        dof = efc_J_colind_in[worldid, 0, sparseid]
      else:
        J_val = efc_J_in[worldid, efcid, dof]
        if J_val == 0.0:
          continue

      t = dof_treeid[dof]
      if t >= 0:
        tree = t
        break

  efc_tree_out[worldid, efcid] = tree


# From mujoco_warp/_src/island.py.
@wp.kernel
def _init_island_arrays(
  # Data out:
  nidof_out: wp.array[int],
  island_idofadr_out: wp.array2d[int],
  island_nv_out: wp.array2d[int],
  island_nefc_out: wp.array2d[int],
  island_ne_out: wp.array2d[int],
  island_nf_out: wp.array2d[int],
  island_iefcadr_out: wp.array2d[int],
):
  worldid, islandid = wp.tid()
  island_nv_out[worldid, islandid] = 0
  island_nefc_out[worldid, islandid] = 0
  island_ne_out[worldid, islandid] = 0
  island_nf_out[worldid, islandid] = 0
  island_idofadr_out[worldid, islandid] = 0
  island_iefcadr_out[worldid, islandid] = 0
  if islandid == 0:
    nidof_out[worldid] = 0


# From mujoco_warp/_src/island.py.
@wp.kernel
def _init_dof_arrays(
  # Data out:
  dof_island_out: wp.array2d[int],
  map_dof2idof_out: wp.array2d[int],
  map_idof2dof_out: wp.array2d[int],
  # Out:
  idof_islandid_out: wp.array2d[int],
):
  worldid, dofid = wp.tid()
  dof_island_out[worldid, dofid] = -1
  map_dof2idof_out[worldid, dofid] = 0
  map_idof2dof_out[worldid, dofid] = 0
  idof_islandid_out[worldid, dofid] = -1


# From mujoco_warp/_src/island.py.
@wp.kernel
def _init_efc_arrays(
  # Data out:
  efc_island_out: wp.array2d[int],
  map_efc2iefc_out: wp.array2d[int],
  map_iefc2efc_out: wp.array2d[int],
  # Out:
  iefc_islandid_out: wp.array2d[int],
  efc_tree_out: wp.array2d[int],
):
  worldid, efcid = wp.tid()
  efc_island_out[worldid, efcid] = -1
  map_efc2iefc_out[worldid, efcid] = 0
  map_iefc2efc_out[worldid, efcid] = 0
  iefc_islandid_out[worldid, efcid] = -1
  efc_tree_out[worldid, efcid] = -1


# From mujoco_warp/_src/island.py.
@event_scope
def compute_island_mapping(m: Model, d: Data):
  """Compute DOF/constraint island mappings after island discovery.

  Populates d.dof_island, d.efc.island, d.island_idofadr, d.island_dofadr,
  d.island_nv, d.island_nefc, d.island_ne, d.island_nf, d.island_iefcadr,
  d.nidof, d.map_dof2idof, d.map_idof2dof, d.dof_islandid, d.map_efc2iefc,
  d.map_iefc2efc, d.efc_islandid.

  Args:
    m: Model.
    d: Data.
  """
  # Ensure dof_islandid / efc_islandid are allocated at the right shape
  if d.dof_islandid.shape[1] != m.nv:
    d.dof_islandid = wp.empty((d.nworld, m.nv), dtype=int)
  if d.efc_islandid.shape[1] != d.njmax:
    d.efc_islandid = wp.empty((d.nworld, d.njmax), dtype=int)
  if d.island_idofadr.shape[1] != m.ntree:
    d.island_idofadr = wp.empty((d.nworld, m.ntree), dtype=int)

  wp.launch(
    _init_island_arrays,
    dim=(d.nworld, m.ntree),
    inputs=[],
    outputs=[d.nidof, d.island_idofadr, d.island_nv, d.island_nefc, d.island_ne, d.island_nf, d.island_iefcadr],
  )
  wp.launch(
    _init_dof_arrays,
    dim=(d.nworld, m.nv),
    inputs=[],
    outputs=[d.dof_island, d.map_dof2idof, d.map_idof2dof, d.dof_islandid],
  )
  efc_tree = wp.empty((d.nworld, d.njmax), dtype=int)
  wp.launch(
    _init_efc_arrays,
    dim=(d.nworld, d.njmax),
    inputs=[],
    outputs=[d.efc.island, d.map_efc2iefc, d.map_iefc2efc, d.efc_islandid, efc_tree],
  )

  wp.launch(
    _compute_efc_tree,
    dim=(d.nworld, d.njmax),
    inputs=[
      m.nv,
      m.body_treeid,
      m.jnt_dofadr,
      m.dof_treeid,
      m.geom_bodyid,
      m.site_bodyid,
      m.eq_type,
      m.eq_obj1id,
      m.eq_obj2id,
      m.eq_objtype,
      m.is_sparse,
      d.nefc,
      d.contact.geom,
      d.efc.type,
      d.efc.id,
      d.efc.J_rownnz,
      d.efc.J_rowadr,
      d.efc.J_colind,
      d.efc.J,
      d.njmax,
    ],
    outputs=[efc_tree],
  )

  # 1. Count DOFs per island
  wp.launch(
    _island_count_dofs,
    dim=(d.nworld, m.nv),
    inputs=[m.dof_treeid, d.tree_island],
    outputs=[d.dof_island, d.island_nv],
  )

  # 2. Count Constraints per island
  wp.launch(
    _island_count_constraints,
    dim=(d.nworld, d.njmax),
    inputs=[d.nefc, d.efc.type, d.tree_island, d.njmax, efc_tree],
    outputs=[d.efc.island, d.island_nefc, d.island_ne, d.island_nf],
  )

  # 3. Scan sizes and reset counters for mapping
  wp.launch(
    _island_scan_sizes,
    dim=d.nworld,
    inputs=[d.nisland, d.island_nv],
    outputs=[d.nidof, d.island_idofadr, d.island_nefc, d.island_iefcadr],
  )

  # 4. Map DOFs
  d.island_dofadr.fill_(m.nv)
  wp.launch(
    _island_map_dofs,
    dim=(d.nworld, m.nv),
    inputs=[m.dof_treeid, m.tree_dofadr, m.tree_dofnum, d.nidof, d.tree_island, d.island_idofadr],
    outputs=[d.island_dofadr, d.map_dof2idof, d.map_idof2dof, d.dof_islandid],
  )

  # 5. Map Constraints
  wp.launch(
    _island_map_constraints,
    dim=(d.nworld, m.ntree),
    inputs=[d.nefc, d.nisland, d.efc.type, d.efc.island, d.island_ne, d.island_nf, d.island_iefcadr, d.njmax],
    outputs=[d.island_nefc, d.map_efc2iefc, d.map_iefc2efc, d.efc_islandid],
  )


# Active-DOF compaction (nvmax < nv).
#
# The active set is tracked per kinematic tree (the unit of coupling in the mass matrix)
# via the same tree_awake bookkeeping used by the sleep solver. Each step the active trees'
# DOFs are packed into a contiguous [0, ncdof) range so the dense factor/solve can run at
# size nvmax instead of nv. This is the active-set analog of compute_island_mapping.


# From mujoco_warp/_src/island.py.
@wp.kernel
def _reset_compact_maps(
  # Model:
  nv: int,
  # Data in:
  nvmax_pad_in: int,
  # Data out:
  dof_cdof_out: wp.array2d[int],
  cdof_dof_out: wp.array2d[int],
):
  worldid, idx = wp.tid()
  if idx < nv:
    dof_cdof_out[worldid, idx] = -1
  # cdof_dof is nvmax_pad-wide: clear the whole row so the padded tail [ncdof, nvmax_pad)
  # reads as -1 (the gather/solve run over nvmax_pad, not just the active ncdof).
  if idx < nvmax_pad_in:
    cdof_dof_out[worldid, idx] = -1


# From mujoco_warp/_src/island.py.
@cache_kernel
def _compact_dofs_builder(warn_overflow: int):
  @wp.kernel(module="unique", enable_backward=False)
  def _compact_dofs(
    # Model:
    ntree: int,
    tree_dofadr: wp.array[int],
    tree_dofnum: wp.array[int],
    # Data in:
    tree_awake_in: wp.array2d[int],
    nvmax_in: int,
    # Data out:
    ncdof_out: wp.array[int],
    dof_cdof_out: wp.array2d[int],
    cdof_dof_out: wp.array2d[int],
    overflow_out: wp.array[int],
  ):
    worldid = wp.tid()
    count = int(0)
    for t in range(ntree):
      if tree_awake_in[worldid, t] == 1:
        adr = tree_dofadr[t]
        num = tree_dofnum[t]
        for j in range(num):
          dof = adr + j
          if count < nvmax_in:
            dof_cdof_out[worldid, dof] = count
            cdof_dof_out[worldid, count] = dof
          count += 1

    if count > nvmax_in:
      if wp.static(bool(warn_overflow & OverflowType.NVMAX)):
        wp.printf(
          "nvmax overflow: world %d needs %d active DOFs but nvmax = %d (behavior undefined)\n"
          "To disable the print warning: m.opt.warn_overflow &= ~mjw.OverflowType.NVMAX (or = 0 for all)\n",
          worldid,
          count,
          nvmax_in,
        )
      overflow_out[worldid] = overflow_out[worldid] | OverflowType.NVMAX
      ncdof_out[worldid] = nvmax_in
    else:
      ncdof_out[worldid] = count

  return _compact_dofs


# From mujoco_warp/_src/island.py.
@event_scope
def update_active_dofs(m: Model, d: Data):
  """Rebuild the compaction maps (dof_cdof / cdof_dof) from tree_awake."""
  wp.launch(
    _reset_compact_maps,
    dim=(d.nworld, max(m.nv, d.nvmax_pad)),
    inputs=[m.nv, d.nvmax_pad],
    outputs=[d.dof_cdof, d.cdof_dof],
  )
  wp.launch(
    _compact_dofs_builder(int(m.opt.warn_overflow)),
    dim=(d.nworld,),
    inputs=[m.ntree, m.tree_dofadr, m.tree_dofnum, d.tree_awake, d.nvmax],
    outputs=[d.ncdof, d.dof_cdof, d.cdof_dof, d.overflow],
  )

_BLOCK_CHOLESKY_DIM = 32


def create_inverse_context(m: Model, d: Data) -> InverseContext:
  """Create an InverseContext with allocated workspace arrays.

  Args:
    m: Model.
    d: Data.

  Returns:
    InverseContext with allocated arrays: Jaref (nworld, njmax), search_dot (nworld,),
    done (nworld,); the change-tracking arrays are left empty (unused by inverse).
  """
  nworld = d.nworld
  njmax = d.njmax

  return InverseContext(
    Jaref=wp.empty((nworld, njmax), dtype=float),
    search_dot=wp.empty((nworld,), dtype=float),
    done=wp.empty((nworld,), dtype=bool),
    quad_changed_ids=wp.empty((nworld, 0), dtype=int),
    quad_changed_count=wp.empty((0,), dtype=int),
    state_changed_count=wp.empty((0,), dtype=int),
    ls_exhausted=wp.empty((0,), dtype=bool),
  )


def _create_solver_context(m: Model, d: Data, newton_solve=None) -> SolverContext:
  """Create a SolverContext with allocated workspace arrays.

  Args:
    m: Model.
    d: Data.
    newton_solve: None, or the caller's solve of the Newton system (solve's newton_solve):
      then no dense Hessian is allocated and no incremental tracking runs.

  Returns:
    SolverContext with the per-solver workspaces allocated: Newton gets the dense
    Hessian h (nworld, nv_pad, nv_pad) (plus hfactor for the blocked Cholesky), CG gets
    Mgrad/prev_grad/prev_Mgrad/beta, incremental tracking gets the change counters, and
    prox gets the ctx.prox_* scratch arrays (shapes documented on types.SolverContext).
  """
  nworld = d.nworld
  nv = m.nv
  nv_pad = m.nv_pad
  njmax = d.njmax

  # bool: Newton needs the dense Hessian, unless the caller solves its system
  alloc_h = m.opt.solver == SolverType.NEWTON and newton_solve is None
  alloc_hfactor = alloc_h and nv > _BLOCK_CHOLESKY_DIM  # bool: blocked Cholesky caches its factor
  alloc_mgrad = m.opt.solver == SolverType.CG  # bool: CG preconditioned-gradient state
  alloc_incremental = _use_incremental(m) and newton_solve is None  # bool: constraint-state change tracking enabled
  alloc_prox = bool(m.opt.prox_enable)  # bool: exact-contact (prox) scratch state
  naconmax = d.contact.dist.shape[0]  # scalar: contact capacity across all worlds

  return SolverContext(
    prox_z=wp.zeros((nworld, njmax), dtype=float) if alloc_prox else wp.empty((0, 0), dtype=float),
    prox_aref0=wp.zeros((nworld, njmax), dtype=float) if alloc_prox else wp.empty((0, 0), dtype=float),
    prox_gamma=wp.zeros((naconmax,), dtype=float) if alloc_prox else wp.empty((0,), dtype=float),
    prox_rho_i=wp.zeros((naconmax,), dtype=float) if alloc_prox else wp.empty((0,), dtype=float),
    prox_bdamp=wp.zeros((naconmax,), dtype=float) if alloc_prox else wp.empty((0,), dtype=float),
    prox_dz=wp.zeros((nworld,), dtype=float) if alloc_prox else wp.empty((0,), dtype=float),
    prox_znorm=wp.zeros((nworld,), dtype=float) if alloc_prox else wp.empty((0,), dtype=float),
    prox_dsigma=wp.zeros((nworld,), dtype=float) if alloc_prox else wp.empty((0,), dtype=float),
    prox_signorm=wp.zeros((nworld,), dtype=float) if alloc_prox else wp.empty((0,), dtype=float),
    prox_done=wp.zeros((nworld,), dtype=bool) if alloc_prox else wp.empty((0,), dtype=bool),
    prox_niter=wp.zeros((nworld,), dtype=int) if alloc_prox else wp.empty((0,), dtype=int),
    prox_nactive=wp.zeros((1,), dtype=int) if alloc_prox else wp.empty((0,), dtype=int),
    Jaref=wp.empty((nworld, njmax), dtype=float),
    search_dot=wp.empty((nworld,), dtype=float),
    done=wp.empty((nworld,), dtype=bool),
    grad=wp.zeros((nworld, nv_pad), dtype=float),
    grad_dot=wp.empty((nworld,), dtype=float),
    newton_decrement=wp.empty((nworld,), dtype=float),
    Mgrad=wp.empty((nworld, nv_pad), dtype=float) if alloc_mgrad else wp.empty((nworld, 0), dtype=float),
    search=wp.empty((nworld, nv), dtype=float),
    mv=wp.empty((nworld, nv), dtype=float),
    jv=wp.empty((nworld, njmax), dtype=float),
    quad=wp.empty((nworld, njmax), dtype=wp.vec3),
    alpha=wp.empty((nworld,), dtype=float),
    grad_scale=wp.empty((nworld,), dtype=float),
    improvement=wp.empty((nworld,), dtype=float),
    ls_exhausted=wp.zeros((nworld,), dtype=bool),
    search_unchanged=wp.empty((nworld,), dtype=bool),
    prev_grad=wp.empty((nworld, nv), dtype=float) if alloc_mgrad else wp.empty((nworld, 0), dtype=float),
    prev_Mgrad=wp.empty((nworld, nv), dtype=float) if alloc_mgrad else wp.empty((nworld, 0), dtype=float),
    beta=wp.empty((nworld,), dtype=float) if alloc_mgrad else wp.empty((0,), dtype=float),
    h=wp.empty((nworld, nv_pad, nv_pad), dtype=float) if alloc_h else wp.empty((nworld, 0, 0), dtype=float),
    hfactor=wp.empty((nworld, nv_pad, nv_pad), dtype=float) if alloc_hfactor else wp.empty((nworld, 0, 0), dtype=float),
    quad_changed_ids=wp.empty((nworld, njmax), dtype=int) if alloc_incremental else wp.empty((nworld, 0), dtype=int),
    quad_changed_count=wp.empty((nworld,), dtype=int) if alloc_incremental else wp.empty((0,), dtype=int),
    state_changed_count=wp.empty((nworld,), dtype=int) if alloc_incremental else wp.empty((0,), dtype=int),
    newton_solve=newton_solve,
  )


@wp.func
def _rescale(nv: int, meaninertia: float, value: float) -> float:
  """Normalizes a cost/gradient magnitude by the problem scale meaninertia * nv."""
  return value / (meaninertia * float(nv))


@wp.func
def _in_bracket(x: wp.vec3, y: wp.vec3) -> bool:
  """True if y's slope lies strictly between x's slope and 0 (same sign, smaller
  magnitude), i.e. y tightens the bracket. x and y are (cost, slope, curvature) vec3s."""
  return (x[1] < y[1] and y[1] < 0.0) or (x[1] > y[1] and y[1] > 0.0)


@wp.func
def _eval_pt_direct_alpha_zero(jaref: float, jv: float, d: float) -> wp.vec3:
  """Eval quadratic constraint at alpha=0.

  Args:
    jaref: scalar constraint residual (J @ qacc - aref) of the row.
    jv: scalar directional velocity (J @ search) of the row.
    d: scalar constraint weight efc_D.

  Returns:
    vec3 (cost, slope, curvature) of 0.5 * d * (jaref + alpha*jv)^2 at alpha = 0.
  """
  jvD = jv * d
  return wp.vec3(0.5 * d * jaref * jaref, jvD * jaref, jv * jvD)


@wp.func
def _eval_pt_direct(jaref: float, jv: float, d: float, alpha: float) -> wp.vec3:
  """Eval quadratic constraint.

  Args:
    jaref: scalar constraint residual of the row.
    jv: scalar directional velocity of the row.
    d: scalar constraint weight efc_D.
    alpha: scalar step along the search ray.

  Returns:
    vec3 (cost, slope, curvature) of 0.5 * d * (jaref + alpha*jv)^2 at alpha.
  """
  x = jaref + alpha * jv
  jvD = jv * d
  return wp.vec3(0.5 * d * x * x, jvD * x, jv * jvD)


@wp.func
def _eval_pt_direct_cost_alpha_zero(jaref: float, d: float) -> float:
  """Cost of the quadratic constraint at alpha=0: 0.5 * d * jaref^2."""
  return 0.5 * d * jaref * jaref


@wp.func
def _eval_pt_direct_shifted(jaref: float, jv: float, d: float, alpha: float, offset: float) -> wp.vec3:
  """Eval quadratic constraint shifted by alpha=0, plus a constant cost offset.

  Returns:
    vec3 (cost, slope, curvature) with cost = quad(alpha) - quad(0) + offset; slope and
    curvature are those of the unshifted quadratic.
  """
  jvD = jv * d
  hessian = jv * jvD
  alpha_h = alpha * hessian
  return wp.vec3(alpha * (jvD * jaref + 0.5 * alpha_h) + offset, jvD * jaref + alpha_h, hessian)


@wp.func
def _eval_pt_direct_3alphas(
  jaref: float, jv: float, d: float, lo_alpha: float, hi_alpha: float, mid_alpha: float
) -> tuple[wp.vec3, wp.vec3, wp.vec3]:
  """Eval quadratic constraint for 3 alphas.

  Returns:
    Three (cost, slope, curvature) vec3s of 0.5 * d * (jaref + alpha*jv)^2, evaluated
    at (lo_alpha, hi_alpha, mid_alpha) with the shared products computed once.
  """
  x_lo = jaref + lo_alpha * jv
  x_hi = jaref + hi_alpha * jv
  x_mid = jaref + mid_alpha * jv
  jvD = jv * d
  hessian = jv * jvD
  half_d = 0.5 * d
  return (
    wp.vec3(half_d * x_lo * x_lo, jvD * x_lo, hessian),
    wp.vec3(half_d * x_hi * x_hi, jvD * x_hi, hessian),
    wp.vec3(half_d * x_mid * x_mid, jvD * x_mid, hessian),
  )


@wp.func
def _eval_pt_direct_shifted_3alphas(
  jaref: float, jv: float, d: float, lo_alpha: float, hi_alpha: float, mid_alpha: float, offset: float
) -> tuple[wp.vec3, wp.vec3, wp.vec3]:
  """Eval shifted quadratic constraint for 3 alphas, plus a constant cost offset.

  Returns:
    Three (cost, slope, curvature) vec3s with cost = quad(alpha) - quad(0) + offset,
    at (lo_alpha, hi_alpha, mid_alpha).
  """
  jvD = jv * d
  grad0 = jvD * jaref
  hessian = jv * jvD
  lo_ah = lo_alpha * hessian
  hi_ah = hi_alpha * hessian
  mid_ah = mid_alpha * hessian
  return (
    wp.vec3(lo_alpha * (grad0 + 0.5 * lo_ah) + offset, grad0 + lo_ah, hessian),
    wp.vec3(hi_alpha * (grad0 + 0.5 * hi_ah) + offset, grad0 + hi_ah, hessian),
    wp.vec3(mid_alpha * (grad0 + 0.5 * mid_ah) + offset, grad0 + mid_ah, hessian),
  )


@wp.func
def _eval_cost(quad: wp.vec3, alpha: float) -> float:
  """Evaluates quad = (c0, c1, c2) at alpha: cost = c0 + c1*alpha + c2*alpha^2."""
  return alpha * alpha * quad[2] + alpha * quad[1] + quad[0]


@wp.func
def _eval_pt(quad: wp.vec3, alpha: float) -> wp.vec3:
  """Eval quad polynomial at alpha, return (cost, grad, hessian).

  quad holds coefficients (c0, c1, c2) with cost = c0 + c1*alpha + c2*alpha^2; the
  returned vec3 is the (cost, slope, curvature) point of that polynomial at alpha.
  """
  aq2 = alpha * quad[2]
  return wp.vec3(
    alpha * aq2 + alpha * quad[1] + quad[0],
    2.0 * aq2 + quad[1],
    2.0 * quad[2],
  )


@wp.func
def _eval_pt_3alphas(quad: wp.vec3, lo_alpha: float, hi_alpha: float, mid_alpha: float) -> tuple[wp.vec3, wp.vec3, wp.vec3]:
  """Eval quad polynomial for 3 alphas.

  Returns:
    Three (cost, slope, curvature) vec3s of quad = (c0, c1, c2) at (lo, hi, mid)
    alphas, sharing the constant curvature 2*c2.
  """
  q0, q1, q2 = quad[0], quad[1], quad[2]
  hessian = 2.0 * q2
  lo_aq2 = lo_alpha * q2
  hi_aq2 = hi_alpha * q2
  mid_aq2 = mid_alpha * q2
  return (
    wp.vec3(lo_alpha * lo_aq2 + lo_alpha * q1 + q0, 2.0 * lo_aq2 + q1, hessian),
    wp.vec3(hi_alpha * hi_aq2 + hi_alpha * q1 + q0, 2.0 * hi_aq2 + q1, hessian),
    wp.vec3(mid_alpha * mid_aq2 + mid_alpha * q1 + q0, 2.0 * mid_aq2 + q1, hessian),
  )


@wp.func
def _shift_cost(pt: wp.vec3, cost0: float) -> wp.vec3:
  """Subtracts the alpha=0 cost cost0 from pt's cost; pt is (cost, slope, curvature)."""
  return wp.vec3(pt[0] - cost0, pt[1], pt[2])


@wp.func
def _eval_frictionloss_pt(x: float, f: float, rf: float, jv: float, d: float) -> wp.vec3:
  """Eval frictionloss and return (cost, grad, hessian). x = Jaref + alpha * jv.

  Huber-style cost: quadratic 0.5*d*x^2 inside |x| < rf (rf = frictionloss/D), linear
  with force -+f outside. Returns the (cost, slope, curvature) vec3 along the ray.
  """
  if (-rf < x) and (x < rf):
    jvD = jv * d
    return wp.vec3(0.5 * d * x * x, jvD * x, jv * jvD)
  elif x <= -rf:
    return wp.vec3(f * (-0.5 * rf - x), -f * jv, 0.0)
  else:
    return wp.vec3(f * (-0.5 * rf + x), f * jv, 0.0)


@wp.func
def _eval_frictionloss_cost(x: float, f: float, rf: float, d: float) -> float:
  """Frictionloss cost only: quadratic 0.5*d*x^2 inside |x| < rf, linear f-sloped outside."""
  if (-rf < x) and (x < rf):
    return 0.5 * d * x * x
  elif x <= -rf:
    return f * (-0.5 * rf - x)
  return f * (-0.5 * rf + x)


@wp.func
def _eval_frictionloss_pt_one(x: float, f: float, rf: float, half_d: float, jvD: float, hessian: float, f_jv: float) -> wp.vec3:
  """Eval frictionloss with precomputed shared values.

  Same math as _eval_frictionloss_pt; half_d, jvD, hessian and f_jv are hoisted by the
  caller so the 3-alpha evaluation computes them once. Returns (cost, slope, curvature).
  """
  if (-rf < x) and (x < rf):
    return wp.vec3(half_d * x * x, jvD * x, hessian)
  elif x <= -rf:
    return wp.vec3(f * (-0.5 * rf - x), -f_jv, 0.0)
  else:
    return wp.vec3(f * (-0.5 * rf + x), f_jv, 0.0)


@wp.func
def _eval_frictionloss_pt_3alphas(
  x_lo: float, x_hi: float, x_mid: float, f: float, rf: float, jv: float, d: float
) -> tuple[wp.vec3, wp.vec3, wp.vec3]:
  """Eval frictionloss for 3 x values with shared precomputation.

  Returns:
    Three (cost, slope, curvature) vec3s at x_lo, x_hi, x_mid (x = Jaref + alpha*jv).
  """
  jvD = jv * d
  half_d = 0.5 * d
  hessian = jv * jvD
  f_jv = f * jv
  return (
    _eval_frictionloss_pt_one(x_lo, f, rf, half_d, jvD, hessian, f_jv),
    _eval_frictionloss_pt_one(x_hi, f, rf, half_d, jvD, hessian, f_jv),
    _eval_frictionloss_pt_one(x_mid, f, rf, half_d, jvD, hessian, f_jv),
  )


@wp.func
def _eval_elliptic_reference(
  # In:
  mu: float,
  quad: wp.vec3,
  quad1: wp.vec3,
  quad2: wp.vec3,
) -> tuple[float, float, float, int]:
  """Classifies the elliptic-cone contact at alpha=0 and returns its reference state.

  Args:
    mu: scalar effective friction of the circularized cone.
    quad: vec3 (c0, c1, c2) quadratic cost coefficients summed over the contact's rows.
    quad1: vec3 (u0, v0, uu) = (mu-scaled normal Jaref, its ray derivative, ||u_t||^2).
    quad2: vec3 (uv, vv, dm) = (u_t . v_t, ||v_t||^2, D0 / (mu^2 * (1 + mu^2))).

  Returns:
    (cost0, T0, r0, state0): cost at alpha=0, tangential norm T0 = sqrt(uu), cone
    residual r0 = u0 - mu*T0 (middle zone only), and the ConstraintState zone
    (SATISFIED inside the cone, QUADRATIC inside the polar cone, CONE in between).
  """
  u0 = quad1[0]  # scalar: mu-scaled normal component N at alpha=0
  uu = quad1[2]  # scalar: squared tangential norm ||u_t||^2 at alpha=0
  dm = quad2[2]  # scalar: middle-zone curvature weight D0 / (mu^2 * (1 + mu^2))

  if uu <= 0.0:
    if u0 < 0.0:
      return quad[0], 0.0, 0.0, int(ConstraintState.QUADRATIC)
    return 0.0, 0.0, 0.0, int(ConstraintState.SATISFIED)

  T0 = wp.sqrt(uu)  # scalar: tangential norm at alpha=0
  if u0 >= mu * T0:
    return 0.0, T0, 0.0, int(ConstraintState.SATISFIED)
  if mu * u0 + T0 <= 0.0:
    return quad[0], T0, 0.0, int(ConstraintState.QUADRATIC)

  r0 = u0 - mu * T0  # scalar: cone residual (negative inside the middle zone)
  return 0.5 * dm * r0 * r0, T0, r0, int(ConstraintState.CONE)


@wp.func
def _eval_elliptic_alpha_zero(mu: float, quad: wp.vec3, quad1: wp.vec3, quad2: wp.vec3) -> wp.vec3:
  """(cost, slope, curvature) of the elliptic-cone contact cost at alpha=0.

  QUADRATIC zone: the plain quadratic point. CONE (middle) zone: ray derivatives of
  0.5 * dm * r^2 with r = N - mu*T. SATISFIED: zero vec3. Argument conventions as in
  _eval_elliptic_reference.
  """
  cost0, T0, r0, state0 = _eval_elliptic_reference(mu, quad, quad1, quad2)
  if state0 == int(ConstraintState.QUADRATIC):
    return _eval_pt(quad, 0.0)
  if state0 == int(ConstraintState.CONE):
    T0_inv = 1.0 / T0  # scalar: 1 / tangential norm
    T1 = quad2[0] * T0_inv  # scalar: dT/dalpha at 0 = (u_t . v_t) / T0
    T2 = (quad2[1] - T1 * T1) * T0_inv  # scalar: d2T/dalpha2 at 0
    r1 = quad1[1] - mu * T1  # scalar: dr/dalpha at 0
    dm = quad2[2]  # scalar: middle-zone curvature weight
    return wp.vec3(cost0, dm * r0 * r1, dm * (r1 * r1 - mu * r0 * T2))
  return wp.vec3(0.0)


@wp.func
def _eval_elliptic_quadratic_cone_gap(mu: float, N: float, T: float, dm: float) -> float:
  """Return quadratic cost minus cone cost at the same point.

  The gap 0.5 * dm * (mu*N + T)^2 vanishes on the polar-cone boundary, making the
  shifted cost continuous across QUADRATIC <-> CONE zone changes along the ray.
  """
  boundary = mu * N + T  # scalar: distance factor to the polar-cone boundary
  return 0.5 * dm * boundary * boundary


@wp.func
def _eval_elliptic_quadratic_shifted(
  # In:
  mu: float,
  quad: wp.vec3,
  alpha: float,
  N: float,
  Tsqr: float,
  u0: float,
  T0: float,
  dm: float,
  state0: int,
) -> wp.vec3:
  """Shifted quadratic-zone point at alpha for a contact whose alpha=0 zone was state0.

  Returns the (cost, slope, curvature) of the full quadratic with the cost rebased so
  that cost(alpha) - cost(0) stays exact across zone changes: a CONE start adds the
  quadratic-minus-cone gap at alpha=0, a SATISFIED start (cost(0) = 0) reconstructs
  the absolute quadratic cost from (N, Tsqr).
  """
  aq2 = alpha * quad[2]  # scalar: alpha * c2
  cost = alpha * (aq2 + quad[1])  # scalar: quad(alpha) - quad(0)
  if state0 == int(ConstraintState.CONE):
    cost += _eval_elliptic_quadratic_cone_gap(mu, u0, T0, dm)
  elif state0 == int(ConstraintState.SATISFIED):
    cost = 0.5 * dm * (1.0 + mu * mu) * (N * N + wp.max(Tsqr, 0.0))
  return wp.vec3(cost, 2.0 * aq2 + quad[1], 2.0 * quad[2])


@wp.func
def _eval_elliptic_shifted(
  # In:
  mu: float,
  quad: wp.vec3,
  quad1: wp.vec3,
  quad2: wp.vec3,
  alpha: float,
  cost0: float,
  T0: float,
  r0: float,
  state0: int,
) -> wp.vec3:
  """Shifted elliptic-cone point at alpha: (cost(alpha) - cost(0), slope, curvature).

  Re-classifies the contact at alpha (polar cone -> quadratic, middle -> cone,
  inside -> satisfied) and returns the point relative to the alpha=0 cost;
  (cost0, T0, r0, state0) come from _eval_elliptic_reference. The CONE-to-CONE path
  forms the cost change from the rationalized residual change to avoid cancellation.
  """
  u0 = quad1[0]  # scalar: mu-scaled normal N at alpha=0
  v0 = quad1[1]  # scalar: dN/dalpha
  uu = quad1[2]  # scalar: ||u_t||^2 at alpha=0
  uv = quad2[0]  # scalar: u_t . v_t
  vv = quad2[1]  # scalar: ||v_t||^2
  dm = quad2[2]  # scalar: middle-zone curvature weight

  N = u0 + alpha * v0  # scalar: mu-scaled normal at alpha
  Tsqr_delta = alpha * (2.0 * uv + alpha * vv)  # scalar: change of ||u_t||^2 along the ray
  Tsqr = uu + Tsqr_delta  # scalar: ||u_t||^2 at alpha

  if Tsqr <= 0.0:
    if N < 0.0:
      return _eval_elliptic_quadratic_shifted(mu, quad, alpha, N, Tsqr, u0, T0, dm, state0)
  else:
    T = wp.sqrt(Tsqr)
    if N >= mu * T:
      pass
    elif mu * N + T <= 0.0:
      return _eval_elliptic_quadratic_shifted(mu, quad, alpha, N, Tsqr, u0, T0, dm, state0)
    else:
      T_inv = 1.0 / T  # scalar: 1 / tangential norm at alpha
      T1 = (uv + alpha * vv) * T_inv  # scalar: dT/dalpha at alpha
      T2 = (vv - T1 * T1) * T_inv  # scalar: d2T/dalpha2 at alpha
      r = N - mu * T  # scalar: cone residual at alpha
      r1 = v0 - mu * T1  # scalar: dr/dalpha at alpha

      if state0 == int(ConstraintState.CONE):
        # Rationalize T - T0 before forming the small cone residual change.
        T_delta = Tsqr_delta / (T + T0)  # scalar: T - T0 without cancellation
        r_delta = alpha * v0 - mu * T_delta  # scalar: r - r0
        cost = 0.5 * dm * r_delta * (2.0 * r0 + r_delta)
      elif state0 == int(ConstraintState.QUADRATIC):
        aq2 = alpha * quad[2]
        cost = alpha * (aq2 + quad[1]) - _eval_elliptic_quadratic_cone_gap(mu, N, T, dm)
      else:
        cost = 0.5 * dm * r * r

      return wp.vec3(
        cost,
        dm * r * r1,
        dm * (r1 * r1 + r * (-mu * T2)),
      )

  # satisfied at alpha: cost(alpha) = 0, so the shifted cost is -cost0
  return wp.vec3(-cost0, 0.0, 0.0)


@wp.func
def _eval_elliptic_middle(
  # In:
  N: float,
  T: float,
  D0: float,
  mu: float,
  ufrictionj: float,
  is_normal: bool,
) -> wp.vec2:
  """Computes the elliptic-cone middle-zone (force, cost) for one row (cost 0 on tangent rows).

  Args:
    N: scalar mu-scaled normal residual of the contact.
    T: scalar tangential norm.
    D0: scalar efc_D of the normal row.
    mu: scalar effective friction of the circularized cone.
    ufrictionj: scalar u_j * friction_j of this row (tangent rows only).
    is_normal: bool; True on the contact's normal row.

  Returns:
    vec2 (force, cost): the normal row carries the cone cost 0.5 * dm * (N - mu*T)^2;
    a tangent row redistributes the normal force along -u_t / T.
  """
  dm = math.safe_div(D0, mu * mu * (1.0 + mu * mu))  # scalar: middle-zone curvature weight
  nmt = N - mu * T  # scalar: cone residual
  force_normal = -dm * nmt * mu  # scalar: normal force in the middle zone
  if is_normal:
    return wp.vec2(force_normal, 0.5 * dm * nmt * nmt)
  return wp.vec2(-math.safe_div(force_normal, T) * ufrictionj, 0.0)


@wp.func
def _eval_constraint(
  # In:
  is_equality: bool,
  is_friction: bool,
  is_elliptic: bool,
  jaref: float,
  D: float,
  frictionloss: float,
  efcid: int,
  efcid0: int,
  jaref0: float,
  D0: float,
  mu: float,
  ufrictionj: float,
  TT: float,
) -> wp.vec3:
  """Force, state, and cost of one efc row at the current Jaref.

  Args:
    is_equality: bool; row is an equality constraint (always-active quadratic).
    is_friction: bool; row is a dof/tendon frictionloss row (Huber cost).
    is_elliptic: bool; row belongs to an elliptic contact cone.
    jaref: scalar residual J @ qacc - aref of this row.
    D: scalar constraint weight of this row.
    frictionloss: scalar friction loss (friction rows only).
    efcid: scalar efc id of this row.
    efcid0: scalar efc id of the contact's normal row (elliptic only).
    jaref0: scalar residual of the normal row (elliptic only).
    D0: scalar weight of the normal row (elliptic only).
    mu: scalar effective friction of the circularized cone (elliptic only).
    ufrictionj: scalar u_j * friction_j of this row (elliptic tangent rows).
    TT: scalar squared tangential norm ||u_t||^2 (elliptic only).

  Returns:
    vec3 (force, state, cost) with state a float-encoded ConstraintState.
  """
  if is_equality:
    force = -D * jaref
    cost = 0.5 * D * jaref * jaref
    return wp.vec3(force, float(ConstraintState.QUADRATIC.value), cost)

  if is_friction:
    rf = math.safe_div(frictionloss, D)
    if jaref <= -rf:
      return wp.vec3(frictionloss, float(ConstraintState.LINEARNEG.value), -frictionloss * (0.5 * rf + jaref))
    elif jaref >= rf:
      return wp.vec3(-frictionloss, float(ConstraintState.LINEARPOS.value), -frictionloss * (0.5 * rf - jaref))
    else:
      return wp.vec3(-D * jaref, float(ConstraintState.QUADRATIC.value), 0.5 * D * jaref * jaref)

  if is_elliptic:
    N = jaref0 * mu
    if TT <= 0.0:
      T = 0.0
    else:
      T = wp.sqrt(TT)

    # Top zone
    if (N >= mu * T) or ((T <= 0.0) and (N >= 0.0)):
      return wp.vec3(0.0, float(ConstraintState.SATISFIED.value), 0.0)
    # Bottom zone
    elif (mu * N + T <= 0.0) or ((T <= 0.0) and (N < 0.0)):
      return wp.vec3(-D * jaref, float(ConstraintState.QUADRATIC.value), 0.5 * D * jaref * jaref)
    # Middle zone
    else:
      is_normal = efcid == efcid0
      fc = _eval_elliptic_middle(N, T, D0, mu, ufrictionj, is_normal)
      return wp.vec3(fc[0], float(ConstraintState.CONE.value), fc[1])

  if jaref >= 0.0:
    return wp.vec3(0.0, float(ConstraintState.SATISFIED.value), 0.0)
  else:
    return wp.vec3(-D * jaref, float(ConstraintState.QUADRATIC.value), 0.5 * D * jaref * jaref)


# kernel_analyzer: off
@wp.func
def _compute_efc_eval_pt_pyramidal(
  efcid: int,
  alpha: float,
  ne: int,
  nf: int,
  # Per-row data:
  efc_D: float,
  efc_frictionloss: wp.array[float],
  ctx_Jaref: float,
  ctx_jv: float,
) -> wp.vec3:
  """Compute shifted cost, gradient, and hessian for pyramidal cones.

  Dispatches on the row's constraint class (equality / friction / limit-contact) from
  (efcid, ne, nf) and evaluates directly from (Jaref, jv, efc_D, alpha); the array arg
  efc_frictionloss is the world's (njmax,) row slice.

  Returns (cost(alpha) - cost(0), grad(alpha), hessian(alpha)) summed across the row.
  """
  # Limit/other constraint
  if efcid >= ne + nf:
    x = ctx_Jaref + alpha * ctx_jv
    quad0 = _eval_pt_direct_cost_alpha_zero(ctx_Jaref, efc_D)
    cost0 = wp.where(ctx_Jaref < 0.0, quad0, 0.0)
    # _eval_pt_direct_shifted returns quad(alpha) - quad(0); add back quad(0) when the
    # constraint was inactive at alpha=0 (i.e. cost(0) = 0) so we get quad(alpha) - 0.
    offset = quad0 - cost0
    if x < 0.0:
      return _eval_pt_direct_shifted(ctx_Jaref, ctx_jv, efc_D, alpha, offset)
    return wp.vec3(-cost0, 0.0, 0.0)

  # Friction constraint - needs quad for frictionloss computation
  if efcid >= ne:
    f = efc_frictionloss[efcid]
    x = ctx_Jaref + alpha * ctx_jv
    rf = math.safe_div(f, efc_D)
    return _shift_cost(_eval_frictionloss_pt(x, f, rf, ctx_jv, efc_D), _eval_frictionloss_cost(ctx_Jaref, f, rf, efc_D))

  # Equality constraint
  return _eval_pt_direct_shifted(ctx_Jaref, ctx_jv, efc_D, alpha, 0.0)


@wp.func
def _compute_efc_eval_pt_elliptic(
  efcid: int,
  alpha: float,
  ne: int,
  nf: int,
  impratio_invsqrt: float,
  # Per-row data (arrays for deferred load):
  efc_type: int,
  efc_D_in: wp.array[float],
  efc_frictionloss: wp.array[float],
  ctx_Jaref: float,
  ctx_jv: float,
  ctx_quad: wp.vec3,
  # Contact data (for elliptic):
  contact_friction: vec5,
  efc_address0: int,
  quad1: wp.vec3,
  quad2: wp.vec3,
) -> wp.vec3:
  """Compute shifted cost, gradient, and hessian for elliptic cones.

  Elliptic contact rows contribute only through the contact's primary (normal) row,
  which consumes the precomputed quads (ctx_quad, quad1, quad2); secondary rows return
  zero. Other rows evaluate directly from (Jaref, jv, efc_D). Array args efc_D_in and
  efc_frictionloss are the world's (njmax,) row slices.

  Returns (cost(alpha) - cost(0), grad(alpha), hessian(alpha)) summed across the row.
  """
  # Contact/limit/other constraints
  if efcid >= ne + nf:
    if efc_type == ConstraintType.CONTACT_ELLIPTIC:
      if efcid != efc_address0:  # Not primary row
        return wp.vec3(0.0)
      mu = contact_friction[0] * impratio_invsqrt
      cost0, T0, r0, state0 = _eval_elliptic_reference(mu, ctx_quad, quad1, quad2)
      return _eval_elliptic_shifted(mu, ctx_quad, quad1, quad2, alpha, cost0, T0, r0, state0)

    # Limit/other constraint — direct eval (no quad read)
    x = ctx_Jaref + alpha * ctx_jv
    efc_D = efc_D_in[efcid]
    quad0 = _eval_pt_direct_cost_alpha_zero(ctx_Jaref, efc_D)
    cost0 = wp.where(ctx_Jaref < 0.0, quad0, 0.0)
    # See _compute_efc_eval_pt_pyramidal for the offset rationale.
    offset = quad0 - cost0
    if x < 0.0:
      return _eval_pt_direct_shifted(ctx_Jaref, ctx_jv, efc_D, alpha, offset)
    return wp.vec3(-cost0, 0.0, 0.0)

  # Friction constraint - load D and frictionloss only here
  if efcid >= ne:
    efc_D = efc_D_in[efcid]
    f = efc_frictionloss[efcid]
    x = ctx_Jaref + alpha * ctx_jv
    rf = math.safe_div(f, efc_D)
    return _shift_cost(_eval_frictionloss_pt(x, f, rf, ctx_jv, efc_D), _eval_frictionloss_cost(ctx_Jaref, f, rf, efc_D))

  # Equality constraint — direct eval (no quad read)
  efc_D = efc_D_in[efcid]
  return _eval_pt_direct_shifted(ctx_Jaref, ctx_jv, efc_D, alpha, 0.0)


@wp.func
def _compute_efc_eval_pt_alpha_zero_pyramidal(
  efcid: int,
  ne: int,
  nf: int,
  # Per-row data:
  efc_D: float,
  efc_frictionloss: wp.array[float],
  ctx_Jaref: float,
  ctx_jv: float,
) -> wp.vec3:
  """Optimized version for alpha=0.0, pyramidal cones.

  Returns the unshifted (cost, slope, curvature) of the row at alpha=0.
  """
  # Limit/other constraint
  if efcid >= ne + nf:
    if ctx_Jaref < 0.0:
      return _eval_pt_direct_alpha_zero(ctx_Jaref, ctx_jv, efc_D)
    return wp.vec3(0.0)

  # Friction constraint - needs quad for frictionloss computation
  if efcid >= ne:
    f = efc_frictionloss[efcid]
    rf = math.safe_div(f, efc_D)
    return _eval_frictionloss_pt(ctx_Jaref, f, rf, ctx_jv, efc_D)

  # Equality constraint
  return _eval_pt_direct_alpha_zero(ctx_Jaref, ctx_jv, efc_D)


@wp.func
def _compute_efc_eval_pt_alpha_zero_elliptic(
  efcid: int,
  ne: int,
  nf: int,
  impratio_invsqrt: float,
  # Per-row data (arrays for deferred load):
  efc_type: int,
  efc_D_in: wp.array[float],
  efc_frictionloss: wp.array[float],
  ctx_Jaref: float,
  ctx_jv: float,
  ctx_quad: wp.vec3,
  # Contact data (for elliptic):
  contact_friction: vec5,
  efc_address0: int,
  quad1: wp.vec3,
  quad2: wp.vec3,
) -> wp.vec3:
  """Optimized version for alpha=0.0, elliptic cones.

  Returns the unshifted (cost, slope, curvature) of the row at alpha=0; elliptic
  contacts contribute through their primary row only (see _compute_efc_eval_pt_elliptic).
  """
  # Contact/limit/other constraints
  if efcid >= ne + nf:
    if efc_type == ConstraintType.CONTACT_ELLIPTIC:
      if efcid != efc_address0:  # Not primary row
        return wp.vec3(0.0)
      mu = contact_friction[0] * impratio_invsqrt
      return _eval_elliptic_alpha_zero(mu, ctx_quad, quad1, quad2)

    # Limit/other constraint — direct eval (no quad read)
    if ctx_Jaref < 0.0:
      return _eval_pt_direct_alpha_zero(ctx_Jaref, ctx_jv, efc_D_in[efcid])
    return wp.vec3(0.0)

  # Friction constraint - load D and frictionloss only here
  if efcid >= ne:
    efc_D = efc_D_in[efcid]
    f = efc_frictionloss[efcid]
    rf = math.safe_div(f, efc_D)
    return _eval_frictionloss_pt(ctx_Jaref, f, rf, ctx_jv, efc_D)

  # Equality constraint — direct eval (no quad read)
  return _eval_pt_direct_alpha_zero(ctx_Jaref, ctx_jv, efc_D_in[efcid])


@wp.func
def _compute_efc_eval_pt_3alphas_pyramidal(
  efcid: int,
  lo_alpha: float,
  hi_alpha: float,
  mid_alpha: float,
  ne: int,
  nf: int,
  # Per-row data:
  efc_D: float,
  efc_frictionloss: wp.array[float],
  ctx_Jaref: float,
  ctx_jv: float,
) -> tuple[wp.vec3, wp.vec3, wp.vec3]:
  """Compute shifted cost, gradient, and hessian for 3 alphas, pyramidal cones.

  Returns a tuple of 3 (cost, slope, curvature) vec3s for (lo_alpha, hi_alpha,
  mid_alpha), each shifted by the row's alpha=0 cost.
  Constraint types checked in order: limit/other -> friction -> equality.
  """
  # Limit/other constraints: active only when x < 0
  if efcid >= ne + nf:
    x_lo = ctx_Jaref + lo_alpha * ctx_jv
    x_hi = ctx_Jaref + hi_alpha * ctx_jv
    x_mid = ctx_Jaref + mid_alpha * ctx_jv
    quad0 = _eval_pt_direct_cost_alpha_zero(ctx_Jaref, efc_D)
    cost0 = wp.where(ctx_Jaref < 0.0, quad0, 0.0)
    # See _compute_efc_eval_pt_pyramidal for the offset rationale.
    offset = quad0 - cost0
    pt_lo, pt_hi, pt_mid = _eval_pt_direct_shifted_3alphas(ctx_Jaref, ctx_jv, efc_D, lo_alpha, hi_alpha, mid_alpha, offset)
    inactive = wp.vec3(-cost0, 0.0, 0.0)
    return (
      wp.where(x_lo < 0.0, pt_lo, inactive),
      wp.where(x_hi < 0.0, pt_hi, inactive),
      wp.where(x_mid < 0.0, pt_mid, inactive),
    )

  # Friction constraint - needs quad for frictionloss computation
  if efcid >= ne:
    x_lo = ctx_Jaref + lo_alpha * ctx_jv
    x_hi = ctx_Jaref + hi_alpha * ctx_jv
    x_mid = ctx_Jaref + mid_alpha * ctx_jv
    f = efc_frictionloss[efcid]
    rf = math.safe_div(f, efc_D)
    cost0 = _eval_frictionloss_cost(ctx_Jaref, f, rf, efc_D)
    lo, hi, mid = _eval_frictionloss_pt_3alphas(x_lo, x_hi, x_mid, f, rf, ctx_jv, efc_D)
    return (_shift_cost(lo, cost0), _shift_cost(hi, cost0), _shift_cost(mid, cost0))

  # Equality constraint: always active
  return _eval_pt_direct_shifted_3alphas(ctx_Jaref, ctx_jv, efc_D, lo_alpha, hi_alpha, mid_alpha, 0.0)


@wp.func
def _compute_efc_eval_pt_3alphas_elliptic(
  efcid: int,
  lo_alpha: float,
  hi_alpha: float,
  mid_alpha: float,
  ne: int,
  nf: int,
  impratio_invsqrt: float,
  # Per-row data (arrays for deferred load):
  efc_type: int,
  efc_D_in: wp.array[float],
  efc_frictionloss: wp.array[float],
  ctx_Jaref: float,
  ctx_jv: float,
  ctx_quad: wp.vec3,
  # Contact data (for elliptic):
  contact_friction: vec5,
  efc_address0: int,
  quad1: wp.vec3,
  quad2: wp.vec3,
) -> tuple[wp.vec3, wp.vec3, wp.vec3]:
  """Compute shifted cost, gradient, and hessian for 3 alphas, elliptic cones.

  Returns a tuple of 3 (cost, slope, curvature) vec3s for (lo_alpha, hi_alpha,
  mid_alpha), each shifted by the row's alpha=0 cost.
  Constraint types checked in order: contact elliptic/limit/other -> friction -> equality.
  """
  # Contact/limit/other constraints
  if efcid >= ne + nf:
    # Contact elliptic: uses special elliptic cone evaluation
    if efc_type == ConstraintType.CONTACT_ELLIPTIC:
      if efcid != efc_address0:  # secondary rows contribute nothing
        return (wp.vec3(0.0), wp.vec3(0.0), wp.vec3(0.0))
      mu = contact_friction[0] * impratio_invsqrt
      cost0, T0, r0, state0 = _eval_elliptic_reference(mu, ctx_quad, quad1, quad2)
      return (
        _eval_elliptic_shifted(mu, ctx_quad, quad1, quad2, lo_alpha, cost0, T0, r0, state0),
        _eval_elliptic_shifted(mu, ctx_quad, quad1, quad2, hi_alpha, cost0, T0, r0, state0),
        _eval_elliptic_shifted(mu, ctx_quad, quad1, quad2, mid_alpha, cost0, T0, r0, state0),
      )

    # Limit/other constraints — direct eval (no quad read)
    x_lo = ctx_Jaref + lo_alpha * ctx_jv
    x_hi = ctx_Jaref + hi_alpha * ctx_jv
    x_mid = ctx_Jaref + mid_alpha * ctx_jv
    efc_D = efc_D_in[efcid]
    quad0 = _eval_pt_direct_cost_alpha_zero(ctx_Jaref, efc_D)
    cost0 = wp.where(ctx_Jaref < 0.0, quad0, 0.0)
    # See _compute_efc_eval_pt_pyramidal for the offset rationale.
    offset = quad0 - cost0
    pt_lo, pt_hi, pt_mid = _eval_pt_direct_shifted_3alphas(ctx_Jaref, ctx_jv, efc_D, lo_alpha, hi_alpha, mid_alpha, offset)
    inactive = wp.vec3(-cost0, 0.0, 0.0)
    return (
      wp.where(x_lo < 0.0, pt_lo, inactive),
      wp.where(x_hi < 0.0, pt_hi, inactive),
      wp.where(x_mid < 0.0, pt_mid, inactive),
    )

  # Friction constraint - load D and frictionloss only here
  if efcid >= ne:
    x_lo = ctx_Jaref + lo_alpha * ctx_jv
    x_hi = ctx_Jaref + hi_alpha * ctx_jv
    x_mid = ctx_Jaref + mid_alpha * ctx_jv
    efc_D = efc_D_in[efcid]
    f = efc_frictionloss[efcid]
    rf = math.safe_div(f, efc_D)
    cost0 = _eval_frictionloss_cost(ctx_Jaref, f, rf, efc_D)
    lo, hi, mid = _eval_frictionloss_pt_3alphas(x_lo, x_hi, x_mid, f, rf, ctx_jv, efc_D)
    return (_shift_cost(lo, cost0), _shift_cost(hi, cost0), _shift_cost(mid, cost0))

  # Equality constraint — direct eval (no quad read)
  efc_D = efc_D_in[efcid]
  return _eval_pt_direct_shifted_3alphas(ctx_Jaref, ctx_jv, efc_D, lo_alpha, hi_alpha, mid_alpha, 0.0)


# kernel_analyzer: on

# =============================================================================
# Iterative Linesearch
# =============================================================================
#
# Iterative linesearch implementation using Warp's tiled execution model with
# parallel reductions over constraint (EFC) rows.
#
# Key optimizations:
#
# 1. KERNEL FUSION - Reduces kernel launch overhead by combining:
#    - linesearch_jv_fused: jv = J @ search (for small nv <= 50)
#    - linesearch_prepare_quad: quad coefficients (pyramidal: computed directly,
#      elliptic: computed in a prepare phase with __syncthreads barrier)
#    - linesearch_prepare_gauss: quad_gauss via tile reduction over DOFs
#    - linesearch_qacc_ma: qacc and Ma updates at kernel end
#    - linesearch_jaref: Jaref update at kernel end
#
# 2. PARALLEL REDUCTIONS - Uses wp.tile_reduce for summing cost/gradient/hessian
#    contributions across EFC rows within each world. The main iteration loop
#    packs 3 vec3 reductions into a single mat33 reduction for efficiency.
#
# 3. COMPILE-TIME SPECIALIZATION via factory parameters:
#    - cone_type: Eliminates elliptic cone branches for pyramidal-only models
#    - ls_iterations: Enables loop unrolling for the main bracket search
#    - fuse_jv: Conditionally includes jv computation based on nv size
#
# 4. DIRECT EVALUATION (pyramidal only) - For equality and limit constraints,
#    computes cost/gradient/hessian directly from (Jaref, jv, efc_D, alpha)
#    without intermediate quad coefficients, using _eval_pt_direct functions.
#
# 5. BATCHED 3-ALPHA EVALUATION - The main iteration loop evaluates 3 alpha
#    values per iteration (lo_next, hi_next, mid). Instead of calling
#    _compute_efc_eval_pt 3 times per constraint row (which would repeat
#    constraint type checks and data loads), we use _compute_efc_eval_pt_3alphas
#    which:
#    - Performs constraint type branching once per row
#    - Loads efc_D, efc_frictionloss, contact data once
#    - Computes x = Jaref + alpha * jv for all 3 alphas
#    - For pyramidal direct evaluation: shares jvD = jv * efc_D, hessian = jv * jvD
#    - For quad-based evaluation: uses _eval_pt_3alphas which computes the
#      constant hessian (2.0 * quad[2]) once and reuses for all 3 alphas
#
# 6. DEFERRED DATA LOADING - efc_D and efc_frictionloss are only loaded
#    inside the constraint branches where they're needed, reducing register
#    pressure for other constraint types.
#
# Trade-offs:
# - Requires block synchronization (__syncthreads) for elliptic quad preparation
# - Separate kernel compilation for each (block_dim, ls_iterations, cone_type,
#   fuse_jv) combination (cached by Warp)
#
# Optimizations attempted but not beneficial:
# - Caching EFC data (Jaref, jv, quad, etc.) in shared memory tiles for reuse
#   across the p0, lo_in, and main iteration loops.
#
# =============================================================================


@cache_kernel
def _linesearch_iterative_kernel(
  ls_iterations: int,
  cone_type: ConeType,
  fuse_jv: bool,
  is_sparse: bool,
  incremental: bool,
  warn_overflow: int,
):
  """Factory for iterative linesearch kernel.

  Args:
    ls_iterations: Max linesearch iterations (compile-time constant for loop optimization).
    cone_type: Friction cone type (PYRAMIDAL or ELLIPTIC) for compile-time optimization.
    fuse_jv: Whether to compute jv = J @ search in-kernel (efficient for small nv).
    is_sparse: Use sparse matrix representation for constraint Jacobian.
    incremental: Use incremental linesearch updates.
    warn_overflow: Overflow warning bitmask.
  """
  LS_ITERATIONS = ls_iterations
  IS_ELLIPTIC = cone_type == ConeType.ELLIPTIC
  FUSE_JV = fuse_jv
  INCREMENTAL = incremental
  IS_SPARSE = is_sparse

  # Native snippet for CUDA __syncthreads()
  @wp.func_native(snippet="WP_TILE_SYNC();")
  def _syncthreads():
    pass

  # Select specialized helper functions based on cone type
  if IS_ELLIPTIC:
    _compute_efc_eval_pt = _compute_efc_eval_pt_elliptic
    _compute_efc_eval_pt_alpha_zero = _compute_efc_eval_pt_alpha_zero_elliptic
    _compute_efc_eval_pt_3alphas = _compute_efc_eval_pt_3alphas_elliptic
  else:
    _compute_efc_eval_pt = _compute_efc_eval_pt_pyramidal
    _compute_efc_eval_pt_alpha_zero = _compute_efc_eval_pt_alpha_zero_pyramidal
    _compute_efc_eval_pt_3alphas = _compute_efc_eval_pt_3alphas_pyramidal

  @wp.kernel(module="unique", enable_backward=False, grid_stride=True)
  def kernel(
    # Model:
    nv: int,
    opt_tolerance: wp.array[float],
    opt_ls_tolerance: wp.array[float],
    opt_impratio_invsqrt: wp.array[float],
    contact_impratio_invsqrt_in: wp.array[float],
    stat_meaninertia: wp.array[float],
    # Data in:
    ne_in: wp.array[int],
    nf_in: wp.array[int],
    nefc_in: wp.array[int],
    qfrc_smooth_in: wp.array2d[float],
    contact_friction_in: wp.array[vec5],
    contact_dim_in: wp.array[int],
    contact_efc_address_in: wp.array2d[int],
    efc_type_in: wp.array2d[int],
    efc_id_in: wp.array2d[int],
    efc_J_rownnz_in: wp.array2d[int],
    efc_J_rowadr_in: wp.array2d[int],
    efc_J_colind_in: wp.array3d[int],
    efc_J_in: wp.array3d[float],
    efc_D_in: wp.array2d[float],
    efc_frictionloss_in: wp.array2d[float],
    njmax_in: int,
    nacon_in: wp.array[int],
    # In:
    ctx_search_unchanged_in: wp.array[bool],
    ctx_Jaref_in: wp.array2d[float],
    ctx_search_in: wp.array2d[float],
    ctx_search_dot_in: wp.array[float],
    ctx_mv_in: wp.array2d[float],
    ctx_jv_in: wp.array2d[float],
    ctx_quad_in: wp.array2d[wp.vec3],
    ctx_done_in: wp.array[bool],
    # Data out:
    qacc_out: wp.array2d[float],
    efc_Ma_out: wp.array2d[float],
    overflow_out: wp.array[int],
    # Out:
    ctx_Jaref_out: wp.array2d[float],
    ctx_jv_out: wp.array2d[float],
    ctx_quad_out: wp.array2d[wp.vec3],
    ctx_improvement_out: wp.array[float],
    ctx_alpha_out: wp.array[float],
    ctx_ls_exhausted_out: wp.array[bool],
  ):
    """One world's full iterative linesearch: brackets the ray minimum and applies it.

    Runs tiled over one world (block_dim lanes named tid): optionally recomputes jv
    and the elliptic quads, reduces the gauss and p0 quadratics, Newton-steps to
    lo_in, then iterates 3-alpha bracket refinement until a slope is below gtol.
    Finishes by advancing qacc, Ma, and Jaref by the accepted alpha.

    Args:
      nv: scalar number of dofs.
      opt_tolerance: (1 or nworld,) main solver tolerance.
      opt_ls_tolerance: (1 or nworld,) linesearch tolerance (scales gtol).
      opt_impratio_invsqrt: (1 or nworld,) world 1/sqrt(impratio).
      contact_impratio_invsqrt_in: (naconmax,) per-contact cone warp 1/sqrt(impratio_i).
      stat_meaninertia: (1 or nworld,) mean diagonal inertia (cost scale).
      ne_in: (nworld,) equality row counts.
      nf_in: (nworld,) friction row counts.
      nefc_in: (nworld,) total constraint row counts.
      qfrc_smooth_in: (nworld, nv) smooth (unconstrained) force.
      contact_friction_in: (naconmax,) vec5 contact friction.
      contact_dim_in: (naconmax,) contact dimensionalities.
      contact_efc_address_in: (naconmax, ncondmax) efc row id of each contact row.
      efc_type_in: (nworld, njmax) ConstraintType per row.
      efc_id_in: (nworld, njmax) owning (contact) id per row.
      efc_J_*: constraint Jacobian in the model's sparse or dense layout.
      efc_D_in: (nworld, njmax_pad) constraint weights.
      efc_frictionloss_in: (nworld, njmax) frictionloss per row.
      njmax_in: scalar maximum rows per world.
      nacon_in: (1,) number of active contacts.
      ctx_search_unchanged_in: (nworld,) mv/jv reuse flags (incremental fast path).
      ctx_Jaref_in: (nworld, njmax) J @ qacc - aref at alpha=0.
      ctx_search_in: (nworld, nv) search direction.
      ctx_search_dot_in: (nworld,) search . search.
      ctx_mv_in: (nworld, nv) M @ search.
      ctx_jv_in: (nworld, njmax) J @ search.
      ctx_quad_in: (nworld, njmax) vec3 elliptic quad coefficients.
      ctx_done_in: (nworld,) converged flags; done worlds exit immediately.
      qacc_out: (nworld, nv) accelerations, advanced by alpha * search.
      efc_Ma_out: (nworld, nv) M @ qacc, advanced by alpha * mv.
      overflow_out: (nworld,) OverflowType bitmask (LS_ITERATIONS on failure).
      ctx_Jaref_out: (nworld, njmax) Jaref, advanced by alpha * jv.
      ctx_jv_out: (nworld, njmax) jv, written when FUSE_JV recomputes it.
      ctx_quad_out: (nworld, njmax) vec3 elliptic quads, written in the prepare phase.
      ctx_improvement_out: (nworld,) accepted cost improvement.
      ctx_alpha_out: (nworld,) accepted step length.
      ctx_ls_exhausted_out: (nworld,) step-was-rounding-noise flags (incremental).
    """
    worldid, tid = wp.tid()  # scalars: world, lane within the world's block

    if ctx_done_in[worldid]:
      return

    ne = ne_in[worldid]  # scalar: equality row count
    nf = nf_in[worldid]  # scalar: friction row count
    nefc = wp.min(njmax_in, nefc_in[worldid])  # scalar: clamped total row count

    # jv = J @ search (fused for small nv); on unchanged search the buffer
    # already holds jv (see _linesearch).
    if wp.static(FUSE_JV):
      recompute_jv = True  # bool: stale-jv check (incremental reuse)
      if wp.static(INCREMENTAL):
        recompute_jv = not ctx_search_unchanged_in[worldid]
      if recompute_jv:
        for efcid in range(tid, nefc, wp.block_dim()):
          jv = float(0.0)  # scalar: running (J @ search)[efcid]
          if wp.static(IS_SPARSE):
            rownnz = efc_J_rownnz_in[worldid, efcid]  # scalar: nonzeros in this row
            rowadr = efc_J_rowadr_in[worldid, efcid]  # scalar: row start in the sparse buffer
            for k in range(rownnz):
              sparseid = rowadr + k  # scalar: sparse entry index
              colind = efc_J_colind_in[worldid, 0, sparseid]  # scalar: dof column
              jv += efc_J_in[worldid, 0, sparseid] * ctx_search_in[worldid, colind]
          else:
            for i in range(nv):
              jv += efc_J_in[worldid, efcid, i] * ctx_search_in[worldid, i]
          ctx_jv_out[worldid, efcid] = jv

      _syncthreads()  # ensure all jv values are written before reading

    # quad coefficients (elliptic contacts only, requires barrier sync)
    # Non-elliptic constraints (equality, friction, limit) now use direct
    # evaluation from (Jaref, jv, efc_D), avoiding quad reads entirely.
    if wp.static(IS_ELLIPTIC):
      # elliptic-only config values (cone warp is per contact; see Data docs)
      nacon = nacon_in[0]  # scalar: active contact count

      for efcid in range(tid, nefc, wp.block_dim()):
        # Only compute and store quad for CONTACT_ELLIPTIC (needs inter-row data)
        if efc_type_in[worldid, efcid] == ConstraintType.CONTACT_ELLIPTIC:
          conid = efc_id_in[worldid, efcid]  # scalar: owning contact
          if conid < nacon:
            efcid0 = contact_efc_address_in[conid, 0]  # scalar: efc row of the normal component
            if efcid == efcid0:
              Jaref = ctx_Jaref_in[worldid, efcid]  # scalar: normal-row residual
              jv = ctx_jv_in[worldid, efcid]  # scalar: normal-row J @ search
              efc_D = efc_D_in[worldid, efcid]  # scalar: normal-row weight

              jvD = jv * efc_D  # scalar: shared product
              # vec3 (c0, c1, c2): the contact's quadratic cost coefficients along the ray
              quad = wp.vec3(0.5 * Jaref * Jaref * efc_D, jvD * Jaref, 0.5 * jv * jvD)

              # primary row: accumulate secondary rows and write quad, quad1, quad2
              dim = contact_dim_in[conid]  # scalar: contact dimensionality
              friction = contact_friction_in[conid]  # vec5: contact friction
              mu = friction[0] * contact_impratio_invsqrt_in[conid]  # scalar: effective friction

              u0 = Jaref * mu  # scalar: mu-scaled normal residual N
              v0 = jv * mu  # scalar: dN/dalpha

              uu = float(0.0)  # scalar: ||u_t||^2 accumulator
              uv = float(0.0)  # scalar: u_t . v_t accumulator
              vv = float(0.0)  # scalar: ||v_t||^2 accumulator
              for j in range(1, dim):
                efcidj = contact_efc_address_in[conid, j]  # scalar: efc row of tangent j
                if efcidj >= 0:
                  jvj = ctx_jv_in[worldid, efcidj]  # scalar: tangent-row J @ search
                  jarefj = ctx_Jaref_in[worldid, efcidj]  # scalar: tangent-row residual
                  dj = efc_D_in[worldid, efcidj]  # scalar: tangent-row weight
                  DJj = dj * jarefj  # scalar: shared product

                  quad += wp.vec3(0.5 * jarefj * DJj, jvj * DJj, 0.5 * jvj * dj * jvj)

                  # rescale to make primal cone circular
                  frictionj = friction[j - 1]  # scalar: friction scale of tangent row j
                  uj = jarefj * frictionj  # scalar: scaled tangential residual
                  vj = jvj * frictionj  # scalar: scaled tangential ray derivative

                  uu += uj * uj
                  uv += uj * vj
                  vv += vj * vj

              ctx_quad_out[worldid, efcid] = quad

              # quad1 = (u0, v0, uu) on the tangent-1 row (see _eval_elliptic_reference)
              efcid1 = contact_efc_address_in[conid, 1]  # scalar: efc row of tangent 1
              ctx_quad_out[worldid, efcid1] = wp.vec3(u0, v0, uu)

              # quad2 = (uv, vv, dm) on the tangent-2 row
              mu2 = mu * mu  # scalar
              efcid2 = contact_efc_address_in[conid, 2]  # scalar: efc row of tangent 2
              ctx_quad_out[worldid, efcid2] = wp.vec3(uv, vv, efc_D / (mu2 * (1.0 + mu2)))

      _syncthreads()  # ensure all quads are written before reading

    # gtol (tolerance values loaded here, deferred from kernel start)
    tolerance = opt_tolerance[worldid % opt_tolerance.shape[0]]  # scalar: solver tolerance
    ls_tolerance = opt_ls_tolerance[worldid % opt_ls_tolerance.shape[0]]  # scalar: ls tolerance
    snorm = wp.sqrt(ctx_search_dot_in[worldid])  # scalar: search direction norm
    meaninertia = stat_meaninertia[worldid % stat_meaninertia.shape[0]]  # scalar: cost scale
    scale = meaninertia * wp.float(nv)  # scalar: problem scale
    gtol = wp.max(tolerance * ls_tolerance * snorm * scale, 1e-6)  # scalar: slope tolerance

    # quad_gauss = [0, search.T @ Ma - search.T @ qfrc_smooth, 0.5 * search.T @ mv] + equality
    local_gauss = wp.vec2(0.0)  # vec2: this lane's partial (c1, c2) of the gauss quadratic
    for dofid in range(tid, nv, wp.block_dim()):
      search = ctx_search_in[worldid, dofid]  # scalar: search entry of this dof
      local_gauss += wp.vec2(
        search * (efc_Ma_out[worldid, dofid] - qfrc_smooth_in[worldid, dofid]),
        0.5 * search * ctx_mv_in[worldid, dofid],
      )

    # equality rows are always active: fold them into the gauss quadratic
    for efcid in range(tid, ne, wp.block_dim()):
      jv = ctx_jv_in[worldid, efcid]  # scalar: equality-row J @ search
      jvD = jv * efc_D_in[worldid, efcid]  # scalar: shared product
      local_gauss += wp.vec2(
        jvD * ctx_Jaref_in[worldid, efcid],
        0.5 * jv * jvD,
      )

    gauss_tile = wp.tile(local_gauss, preserve_type=True)  # tile: per-lane partials
    gauss_sum = wp.tile_reduce(wp.add, gauss_tile)
    gauss_reduced = gauss_sum[0]  # vec2: reduced (c1, c2)
    ctx_quad_gauss = wp.vec3(0.0, gauss_reduced[0], gauss_reduced[1])  # vec3: gauss (c0, c1, c2)

    # p0 via parallel reduction over non-equality constraints
    local_p0 = wp.vec3(0.0)  # vec3: this lane's partial (cost, slope, curvature) at alpha=0
    for efcid in range(ne + tid, nefc, wp.block_dim()):
      if wp.static(IS_ELLIPTIC):
        # per-row elliptic inputs (defaults for non-contact rows)
        efc_type = efc_type_in[worldid, efcid]  # scalar: ConstraintType of the row
        efc_id = 0  # scalar: owning contact id
        contact_friction = vec5(0.0)  # vec5: contact friction
        efc_addr0 = int(0)  # scalar: efc row of the normal component
        ctx_quad = wp.vec3(0.0)  # vec3: (c0, c1, c2) contact quadratic
        quad1 = wp.vec3(0.0)  # vec3: (u0, v0, uu) cone linearization
        quad2 = wp.vec3(0.0)  # vec3: (uv, vv, dm) cone linearization

        if efc_type == ConstraintType.CONTACT_ELLIPTIC:
          efc_id = efc_id_in[worldid, efcid]
          contact_friction = contact_friction_in[efc_id]
          efc_addr0 = contact_efc_address_in[efc_id, 0]
          efc_addr1 = contact_efc_address_in[efc_id, 1]
          efc_addr2 = contact_efc_address_in[efc_id, 2]
          ctx_quad = ctx_quad_in[worldid, efcid]
          quad1 = ctx_quad_in[worldid, efc_addr1]
          quad2 = ctx_quad_in[worldid, efc_addr2]

        local_p0 += _compute_efc_eval_pt_alpha_zero(
          efcid,
          ne,
          nf,
          contact_impratio_invsqrt_in[efc_id],
          efc_type,
          efc_D_in[worldid],
          efc_frictionloss_in[worldid],
          ctx_Jaref_in[worldid, efcid],
          ctx_jv_in[worldid, efcid],
          ctx_quad,
          contact_friction,
          efc_addr0,
          quad1,
          quad2,
        )
      else:
        # direct evaluation for pyramidal cones (no intermediate quad)
        local_p0 += _compute_efc_eval_pt_alpha_zero(
          efcid,
          ne,
          nf,
          efc_D_in[worldid, efcid],
          efc_frictionloss_in[worldid],
          ctx_Jaref_in[worldid, efcid],
          ctx_jv_in[worldid, efcid],
        )

    # at this point, every thread has computed some contributions to p0 in local_p0
    # we now create a tile of all local_p0 contributions and reduce them to a single value
    # this is done in parallel using a tile reduction
    p0_tile = wp.tile(local_p0, preserve_type=True)  # tile: per-lane partials
    p0_sum = wp.tile_reduce(wp.add, p0_tile)

    # add quad_gauss contribution to p0
    # vec3: total (cost, slope, curvature) at alpha=0 (gauss curvature doubled: point form)
    p0 = wp.vec3(ctx_quad_gauss[0], ctx_quad_gauss[1], 2.0 * ctx_quad_gauss[2]) + p0_sum[0]
    p0_delta = wp.vec3(0.0, p0[1], p0[2])  # vec3: p0 as a shifted point (cost rebased to 0)

    # The bracketing search reads derivative sums whose rounding noise is ~eps
    # times their gross magnitude, so roots below eps * |q1| / p0[2] (and never
    # below 8 ulps of the unit ray anchor) are noise, not descent: an accepted
    # step under this floor means the stale ray is exhausted and the fast path
    # must rebuild the world. Per quadratic row |q1| = sqrt(2 * cost * hessian),
    # so Cauchy-Schwarz bounds the row total by sums already reduced for p0; the
    # smooth term is added exactly. Friction linear rows fall outside the bound,
    # which only lowers the floor toward the fixed 8-ulp base.
    rows = p0_sum[0]  # vec3: constraint-row-only (cost, slope, curvature) at alpha=0
    # scalar: Cauchy-Schwarz bound on the gross derivative-term magnitude |q1|
    q1_abs = wp.sqrt(2.0 * wp.max(rows[0], 0.0) * wp.max(rows[2], 0.0)) + wp.abs(ctx_quad_gauss[1])

    # set acceptance tolerance to avoid exceeding gtol in f32
    gtol_accept = wp.max(gtol, _ALPHA_NOISE_EPS * q1_abs)  # scalar: noise-aware slope tolerance

    noise_floor = float(0.0)  # scalar: smallest alpha distinguishable from rounding noise
    if wp.static(INCREMENTAL):
      noise_floor = _ALPHA_NOISE_EPS * wp.max(1.0, math.safe_div(q1_abs, p0[2]))

    # lo_in at lo_alpha_in = -p0[1] / p0[2]
    lo_alpha_in = -math.safe_div(p0[1], p0[2])  # scalar: Newton step from alpha=0

    local_lo_in = wp.vec3(0.0)  # vec3: this lane's partial shifted point at lo_alpha_in
    for efcid in range(ne + tid, nefc, wp.block_dim()):
      if wp.static(IS_ELLIPTIC):
        # per-row elliptic inputs (defaults for non-contact rows)
        efc_type = efc_type_in[worldid, efcid]  # scalar: ConstraintType of the row
        efc_id = 0  # scalar: owning contact id
        contact_friction = vec5(0.0)  # vec5: contact friction
        efc_addr0 = int(0)  # scalar: efc row of the normal component
        ctx_quad = wp.vec3(0.0)  # vec3: (c0, c1, c2) contact quadratic
        quad1 = wp.vec3(0.0)  # vec3: (u0, v0, uu) cone linearization
        quad2 = wp.vec3(0.0)  # vec3: (uv, vv, dm) cone linearization

        if efc_type == ConstraintType.CONTACT_ELLIPTIC:
          efc_id = efc_id_in[worldid, efcid]
          contact_friction = contact_friction_in[efc_id]
          efc_addr0 = contact_efc_address_in[efc_id, 0]
          efc_addr1 = contact_efc_address_in[efc_id, 1]
          efc_addr2 = contact_efc_address_in[efc_id, 2]
          ctx_quad = ctx_quad_in[worldid, efcid]
          quad1 = ctx_quad_in[worldid, efc_addr1]
          quad2 = ctx_quad_in[worldid, efc_addr2]

        local_lo_in += _compute_efc_eval_pt(
          efcid,
          lo_alpha_in,
          ne,
          nf,
          contact_impratio_invsqrt_in[efc_id],
          efc_type,
          efc_D_in[worldid],
          efc_frictionloss_in[worldid],
          ctx_Jaref_in[worldid, efcid],
          ctx_jv_in[worldid, efcid],
          ctx_quad,
          contact_friction,
          efc_addr0,
          quad1,
          quad2,
        )
      else:
        # direct evaluation for pyramidal cones (no intermediate quad)
        local_lo_in += _compute_efc_eval_pt(
          efcid,
          lo_alpha_in,
          ne,
          nf,
          efc_D_in[worldid, efcid],
          efc_frictionloss_in[worldid],
          ctx_Jaref_in[worldid, efcid],
          ctx_jv_in[worldid, efcid],
        )

    lo_in_tile = wp.tile(local_lo_in, preserve_type=True)  # tile: per-lane partials
    lo_in_sum = wp.tile_reduce(wp.add, lo_in_tile)
    lo_in = _eval_pt(ctx_quad_gauss, lo_alpha_in) + lo_in_sum[0]  # vec3: shifted point at lo_alpha_in

    # accept Newton step if derivative is small and cost improved
    initial_converged = wp.abs(lo_in[1]) < gtol_accept and lo_in[0] < 0.0  # bool
    ls_converged = initial_converged  # bool: linesearch termination flag

    # main iterative loop - skip if already converged
    if not initial_converged:
      alpha = float(0.0)  # scalar: best accepted step so far
      improvement = float(0.0)  # scalar: best cost improvement so far

      # initialize bounds: lo/hi are shifted (cost, slope, curvature) bracket points
      lo_less = lo_in[1] < p0[1]  # bool: order the two candidates by slope
      lo = wp.where(lo_less, lo_in, p0_delta)
      lo_alpha = wp.where(lo_less, lo_alpha_in, 0.0)  # scalar: lo bracket step
      hi = wp.where(lo_less, p0_delta, lo_in)
      hi_alpha = wp.where(lo_less, 0.0, lo_alpha_in)  # scalar: hi bracket step

      for _ in range(LS_ITERATIONS):
        lo_next_alpha = lo_alpha - math.safe_div(lo[1], lo[2])  # scalar: Newton step from lo
        hi_next_alpha = hi_alpha - math.safe_div(hi[1], hi[2])  # scalar: Newton step from hi
        mid_alpha = 0.5 * (lo_alpha + hi_alpha)  # scalar: bisection midpoint

        local_lo = wp.vec3(0.0)  # vec3: this lane's partial shifted point at lo_next_alpha
        local_hi = wp.vec3(0.0)  # vec3: this lane's partial shifted point at hi_next_alpha
        local_mid = wp.vec3(0.0)  # vec3: this lane's partial shifted point at mid_alpha

        for efcid in range(ne + tid, nefc, wp.block_dim()):
          if wp.static(IS_ELLIPTIC):
            # per-row elliptic inputs (defaults for non-contact rows)
            efc_type = efc_type_in[worldid, efcid]  # scalar: ConstraintType of the row
            efc_id = 0  # scalar: owning contact id
            contact_friction = vec5(0.0)  # vec5: contact friction
            efc_addr0 = int(0)  # scalar: efc row of the normal component
            ctx_quad = wp.vec3(0.0)  # vec3: (c0, c1, c2) contact quadratic
            quad1 = wp.vec3(0.0)  # vec3: (u0, v0, uu) cone linearization
            quad2 = wp.vec3(0.0)  # vec3: (uv, vv, dm) cone linearization

            if efc_type == ConstraintType.CONTACT_ELLIPTIC:
              efc_id = efc_id_in[worldid, efcid]
              contact_friction = contact_friction_in[efc_id]
              efc_addr0 = contact_efc_address_in[efc_id, 0]
              efc_addr1 = contact_efc_address_in[efc_id, 1]
              efc_addr2 = contact_efc_address_in[efc_id, 2]
              ctx_quad = ctx_quad_in[worldid, efcid]
              quad1 = ctx_quad_in[worldid, efc_addr1]
              quad2 = ctx_quad_in[worldid, efc_addr2]

            r_lo, r_hi, r_mid = _compute_efc_eval_pt_3alphas(
              efcid,
              lo_next_alpha,
              hi_next_alpha,
              mid_alpha,
              ne,
              nf,
              contact_impratio_invsqrt_in[efc_id],
              efc_type,
              efc_D_in[worldid],
              efc_frictionloss_in[worldid],
              ctx_Jaref_in[worldid, efcid],
              ctx_jv_in[worldid, efcid],
              ctx_quad,
              contact_friction,
              efc_addr0,
              quad1,
              quad2,
            )
          else:
            # direct evaluation for pyramidal cones (no intermediate quad)
            r_lo, r_hi, r_mid = _compute_efc_eval_pt_3alphas(
              efcid,
              lo_next_alpha,
              hi_next_alpha,
              mid_alpha,
              ne,
              nf,
              efc_D_in[worldid, efcid],
              efc_frictionloss_in[worldid],
              ctx_Jaref_in[worldid, efcid],
              ctx_jv_in[worldid, efcid],
            )
          local_lo += r_lo
          local_hi += r_hi
          local_mid += r_mid

        # reduce with packed mat33 (3 vec3s into columns: col0=lo, col1=hi, col2=mid)
        local_combined = wp.mat33(
          local_lo[0],
          local_hi[0],
          local_mid[0],
          local_lo[1],
          local_hi[1],
          local_mid[1],
          local_lo[2],
          local_hi[2],
          local_mid[2],
        )

        # reduce with packed mat33 (3 vec3s into columns: col0=lo, col1=hi, col2=mid)
        # this is faster than 3 vec3 reductions because it avoids synchronization barriers
        combined_tile = wp.tile(local_combined, preserve_type=True)  # tile: per-lane partials
        combined_sum = wp.tile_reduce(wp.add, combined_tile)
        result = combined_sum[0]  # mat33: reduced candidate points (columns lo/hi/mid)

        # extract columns back to vec3s and add quad_gauss contributions
        gauss_lo, gauss_hi, gauss_mid = _eval_pt_3alphas(ctx_quad_gauss, lo_next_alpha, hi_next_alpha, mid_alpha)
        lo_next = gauss_lo + wp.vec3(result[0, 0], result[1, 0], result[2, 0])  # vec3: point at lo_next_alpha
        hi_next = gauss_hi + wp.vec3(result[0, 1], result[1, 1], result[2, 1])  # vec3: point at hi_next_alpha
        mid = gauss_mid + wp.vec3(result[0, 2], result[1, 2], result[2, 2])  # vec3: point at mid_alpha

        # accept the lowest-cost converged candidate regardless of the sign of its derivative
        conv_lo = wp.abs(lo_next[1]) < gtol_accept and lo_next[0] < 0.0  # bool
        conv_hi = wp.abs(hi_next[1]) < gtol_accept and hi_next[0] < 0.0  # bool
        conv_mid = wp.abs(mid[1]) < gtol_accept and mid[0] < 0.0  # bool
        converged = conv_lo or conv_hi or conv_mid  # bool: some candidate converged
        swap_lo = False  # bool: lo bracket point replaced this iteration
        swap_hi = False  # bool: hi bracket point replaced this iteration
        if converged:
          conv_pt = wp.vec3(MAXVAL, 0.0, 0.0)  # vec3: best converged point so far
          conv_alpha = float(0.0)  # scalar: its step
          if conv_lo and lo_next[0] < conv_pt[0]:
            conv_pt = lo_next
            conv_alpha = lo_next_alpha
          if conv_hi and hi_next[0] < conv_pt[0]:
            conv_pt = hi_next
            conv_alpha = hi_next_alpha
          if conv_mid and mid[0] < conv_pt[0]:
            conv_pt = mid
            conv_alpha = mid_alpha
          lo = conv_pt
          lo_alpha = conv_alpha
          hi = conv_pt
          hi_alpha = conv_alpha
        else:
          # bracket swapping
          # swap lo:
          swap_lo_lo_next = _in_bracket(lo, lo_next)
          lo = wp.where(swap_lo_lo_next, lo_next, lo)
          lo_alpha = wp.where(swap_lo_lo_next, lo_next_alpha, lo_alpha)
          swap_lo_mid = _in_bracket(lo, mid)
          lo = wp.where(swap_lo_mid, mid, lo)
          lo_alpha = wp.where(swap_lo_mid, mid_alpha, lo_alpha)
          swap_lo_hi_next = _in_bracket(lo, hi_next)
          lo = wp.where(swap_lo_hi_next, hi_next, lo)
          lo_alpha = wp.where(swap_lo_hi_next, hi_next_alpha, lo_alpha)
          swap_lo = swap_lo_lo_next or swap_lo_mid or swap_lo_hi_next

          # swap hi:
          # also accept a Newton step that crosses the minimizer from an undershooting hi,
          # turning the one-sided search into an opposite-sign bracket
          swap_hi_hi_next = _in_bracket(hi, hi_next) or (hi[1] < 0.0 and hi_next[1] > 0.0)
          hi = wp.where(swap_hi_hi_next, hi_next, hi)
          hi_alpha = wp.where(swap_hi_hi_next, hi_next_alpha, hi_alpha)
          swap_hi_mid = _in_bracket(hi, mid)
          hi = wp.where(swap_hi_mid, mid, hi)
          hi_alpha = wp.where(swap_hi_mid, mid_alpha, hi_alpha)
          swap_hi_lo_next = _in_bracket(hi, lo_next)
          hi = wp.where(swap_hi_lo_next, lo_next, hi)
          hi_alpha = wp.where(swap_hi_lo_next, lo_next_alpha, hi_alpha)
          swap_hi = swap_hi_hi_next or swap_hi_mid or swap_hi_lo_next

        # check for convergence
        ls_done = (
          converged
          or (not swap_lo and not swap_hi)
          or (lo[0] < 0.0 and lo[1] < 0.0 and lo[1] > -gtol)
          or (hi[0] < 0.0 and hi[1] > 0.0 and hi[1] < gtol)
        )

        # update alpha if improved
        improved = lo[0] < 0.0 or hi[0] < 0.0  # bool: some bracket point lowered the cost
        lo_better = lo[0] < hi[0]  # bool: lo is the cheaper bracket point
        best_alpha = wp.where(lo_better, lo_alpha, hi_alpha)  # scalar: cheaper point's step
        best_delta = wp.where(lo_better, lo[0], hi[0])  # scalar: its (negative) cost change
        alpha = wp.where(improved, best_alpha, alpha)
        improvement = wp.where(improved, -best_delta, improvement)

        if ls_done:
          ls_converged = True
          break
    else:
      alpha = lo_alpha_in
      improvement = -lo_in[0]

    # qacc and Ma update
    for dofid in range(tid, nv, wp.block_dim()):
      qacc_out[worldid, dofid] += alpha * ctx_search_in[worldid, dofid]
      efc_Ma_out[worldid, dofid] += alpha * ctx_mv_in[worldid, dofid]

    # Jaref update
    for efcid in range(tid, nefc, wp.block_dim()):
      ctx_Jaref_out[worldid, efcid] += alpha * ctx_jv_in[worldid, efcid]

    if tid == 0:
      ctx_improvement_out[worldid] = improvement
      ctx_alpha_out[worldid] = alpha
      if wp.static(INCREMENTAL):
        ctx_ls_exhausted_out[worldid] = wp.abs(alpha) < noise_floor
      if not ls_converged:
        if wp.static(bool(warn_overflow & OverflowType.LS_ITERATIONS)):
          wp.printf(
            "linesearch iterations limit reached - please increase ls_iterations beyond %u\n"
            "To disable the print warning: m.opt.warn_overflow &= ~mjw.OverflowType.LS_ITERATIONS (or = 0 for all)\n",
            wp.static(LS_ITERATIONS),
          )
        wp.atomic_or(overflow_out, worldid, wp.static(OverflowType.LS_ITERATIONS))

  return kernel


def _linesearch_iterative(m: Model, d: Data, ctx: SolverContext, fuse_jv: bool):
  """Iterative linesearch with parallel reductions over efc rows and dofs.

  Args:
    m: Model.
    d: Data.
    ctx: SolverContext.
    fuse_jv: Whether jv is computed in-kernel (True) or pre-computed (False).
  """
  wp.launch_tiled(
    _linesearch_iterative_kernel(
      m.opt.ls_iterations,
      m.opt.cone,
      fuse_jv,
      m.is_sparse,
      _use_incremental(m, ctx),
      int(m.opt.warn_overflow),
    ),
    dim=d.nworld,
    inputs=[
      m.nv,
      m.opt.tolerance,
      m.opt.ls_tolerance,
      m.opt.impratio_invsqrt,
      d.contact_impratio_invsqrt,
      m.stat.meaninertia,
      d.ne,
      d.nf,
      d.nefc,
      d.qfrc_smooth,
      d.contact.friction,
      d.contact.dim,
      d.contact.efc_address,
      d.efc.type,
      d.efc.id,
      d.efc.J_rownnz,
      d.efc.J_rowadr,
      d.efc.J_colind,
      d.efc.J,
      d.efc.D,
      d.efc.frictionloss,
      d.njmax,
      d.nacon,
      ctx.search_unchanged,
      ctx.Jaref,
      ctx.search,
      ctx.search_dot,
      ctx.mv,
      ctx.jv,
      ctx.quad,
      ctx.done,
    ],
    outputs=[
      d.qacc,
      d.efc.Ma,
      d.overflow,
      ctx.Jaref,
      ctx.jv,
      ctx.quad,
      ctx.improvement,
      ctx.alpha,
      ctx.ls_exhausted,
    ],
    block_dim=m.block_dim.linesearch_iterative,
  )


@wp.kernel
def _linesearch_zero_jv(
  # Data in:
  nefc_in: wp.array[int],
  # In:
  skip_in: wp.array[bool],
  # Out:
  ctx_jv_out: wp.array2d[float],
):
  """Zeroes jv rows of non-skipped worlds before a multi-thread-per-row accumulate.

  Args:
    nefc_in: (nworld,) constraint row counts.
    skip_in: (nworld,) worlds to leave untouched (done, or search unchanged).
    ctx_jv_out: (nworld, njmax) accumulation target J @ search.
  """
  worldid, efcid = wp.tid()  # scalars: world, efc row

  if efcid >= nefc_in[worldid]:
    return

  if skip_in[worldid]:
    return

  ctx_jv_out[worldid, efcid] = 0.0


@cache_kernel
def _linesearch_jv_fused_kernel(is_sparse: bool, nv: int, dofs_per_thread: int, compact: bool):
  """Builds the jv = J @ search kernel, specialized on the Jacobian layout.

  Args:
    is_sparse: bool; CSR constraint Jacobian layout, compiled in.
    nv: scalar number of dofs, compiled in.
    dofs_per_thread: scalar dofs accumulated per thread (dense split path).
    compact: bool; map Jacobian columns through dof_cdof (compacted solve).

  Returns:
    A kernel over (nworld, njmax, threads_per_efc) writing ctx_jv_out (nworld, njmax);
    with threads_per_efc > 1 the rows are atomically accumulated (caller zeroes first).
  """
  COMPACT = compact  # bool: compile-time compaction switch (must be a local for wp.static)

  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def kernel(
    # Data in:
    nefc_in: wp.array[int],
    efc_J_rownnz_in: wp.array2d[int],
    efc_J_rowadr_in: wp.array2d[int],
    efc_J_colind_in: wp.array3d[int],
    efc_J_in: wp.array3d[float],
    dof_cdof_in: wp.array2d[int],
    # In:
    ctx_search_in: wp.array2d[float],
    skip_in: wp.array[bool],
    # Out:
    ctx_jv_out: wp.array2d[float],
  ):
    """Computes jv[worldid, efcid] = (J @ search)[efcid] for non-skipped worlds.

    Args:
      nefc_in: (nworld,) constraint row counts.
      efc_J_*: constraint Jacobian in the model's sparse or dense layout.
      dof_cdof_in: (nworld, nv) global dof -> compacted dof map (compact only).
      ctx_search_in: (nworld, nv) search direction.
      skip_in: (nworld,) worlds whose jv is kept (done, or search unchanged).
      ctx_jv_out: (nworld, njmax) output J @ search.
    """
    worldid, efcid, dofstart = wp.tid()  # scalars: world, efc row, dense dof block

    if efcid >= nefc_in[worldid]:
      return

    if skip_in[worldid]:
      return

    jv_out = float(0.0)  # scalar: running (J @ search)[efcid]

    if wp.static(dofs_per_thread >= nv):
      if wp.static(is_sparse):
        # Sparse: iterate over non-zero entries in the row
        rownnz = efc_J_rownnz_in[worldid, efcid]  # scalar: nonzeros in this row
        rowadr = efc_J_rowadr_in[worldid, efcid]  # scalar: row start in the sparse buffer
        for k in range(rownnz):
          sparseid = rowadr + k  # scalar: sparse entry index
          colind = efc_J_colind_in[worldid, 0, sparseid]  # scalar: dof column
          if wp.static(COMPACT):
            colind = dof_cdof_in[worldid, colind]
            if colind < 0:
              continue
          jv_out += efc_J_in[worldid, 0, sparseid] * ctx_search_in[worldid, colind]
      else:
        for i in range(wp.static(min(dofs_per_thread, nv))):
          jv_out += efc_J_in[worldid, efcid, i] * ctx_search_in[worldid, i]
      ctx_jv_out[worldid, efcid] = jv_out

    else:
      if wp.static(is_sparse):
        # Sparse: thread 0 handles entire row (sparse entries << nv typically)
        if dofstart == 0:
          rownnz = efc_J_rownnz_in[worldid, efcid]  # scalar: nonzeros in this row
          rowadr = efc_J_rowadr_in[worldid, efcid]  # scalar: row start in the sparse buffer
          for k in range(rownnz):
            sparseid = rowadr + k  # scalar: sparse entry index
            colind = efc_J_colind_in[worldid, 0, sparseid]  # scalar: dof column
            if wp.static(COMPACT):
              colind = dof_cdof_in[worldid, colind]
              if colind < 0:
                continue
            jv_out += efc_J_in[worldid, 0, sparseid] * ctx_search_in[worldid, colind]
          ctx_jv_out[worldid, efcid] = jv_out
      else:
        for i in range(wp.static(dofs_per_thread)):
          ii = dofstart * wp.static(dofs_per_thread) + i  # scalar: dof index of this thread's block
          if ii < nv:
            jv_out += efc_J_in[worldid, efcid, ii] * ctx_search_in[worldid, ii]
        wp.atomic_add(ctx_jv_out, worldid, efcid, jv_out)

  return kernel


@event_scope
def _linesearch(m: Model, d: Data, ctx: SolverContext):
  """Linesearch for constraint solver.

  When state changes are tracked, worlds with ctx.search_unchanged reuse last
  iteration's mv/jv, so a fresh search requires clearing the flag (see the
  invalidation in _solve and the writer in _update_gradient_zero_grad_dot).

  Args:
    m: Model
    d: Data
    ctx: SolverContext
  """
  # mv and jv are pure functions of the search direction, and M and J are
  # constant within a solve, so worlds whose search was kept reuse last
  # iteration's values.
  skip = ctx.search_unchanged if _use_incremental(m, ctx) else ctx.done  # (nworld,) worlds whose mv/jv are kept

  # mv = M @ search (common to both parallel and iterative)
  _mul_m_compact_aware(m, d, ctx, ctx.mv, ctx.search, skip)

  # Fuse jv computation in-kernel for small nv (iterative only, dense only)
  # Sparse mode requires pre-computed jv since in-kernel uses dense indexing
  # the sparse-compact J path reads the full model's sparse J structures;
  # dense full models keep the gathered dense cJ path
  sc = _sparse_compact(ctx)  # bool: compact solve over a sparse full model
  fuse_jv = m.nv <= 50 and not m.is_sparse and not sc  # bool: compute jv inside the linesearch kernel

  # jv = J @ search (when not fused into iterative kernel)
  if not fuse_jv:
    dj = ctx.compact_d_full if sc else d  # data holding the J structures to read
    if sc or m.is_sparse:
      # Sparse J has few nonzeros per row, one thread handles them all.
      dofs_per_thread = m.nv  # scalar: dofs each thread accumulates
      threads_per_efc = 1  # scalar: threads cooperating on one efc row
    else:
      dofs_per_thread = 20 if m.nv > 50 else 50
      threads_per_efc = ceil(m.nv / dofs_per_thread)

    if threads_per_efc > 1:
      wp.launch(
        _linesearch_zero_jv,
        dim=(d.nworld, d.njmax),
        inputs=[d.nefc, skip],
        outputs=[ctx.jv],
      )

    wp.launch(
      _linesearch_jv_fused_kernel(sc or m.is_sparse, m.nv, dofs_per_thread, sc),
      dim=(d.nworld, d.njmax, threads_per_efc),
      inputs=[d.nefc, dj.efc.J_rownnz, dj.efc.J_rowadr, dj.efc.J_colind, dj.efc.J, dj.dof_cdof, ctx.search, skip],
      outputs=[ctx.jv],
    )

  _linesearch_iterative(m, d, ctx, fuse_jv)


@cache_kernel
def _solve_init_dof(warmstart: bool, sparse: bool):
  """Builds the per-dof solve init kernel, specialized on (warmstart, sparse).

  Args:
    warmstart: bool; seed qacc from qacc_warmstart instead of qacc_smooth.
    sparse: bool; also zero qfrc_constraint of constraint-free worlds (the sparse
      accumulator never touches them).

  Returns:
    A kernel over (nworld, nv) writing qacc (nworld, nv) and, when sparse,
    qfrc_constraint (nworld, nv).
  """
  WARMSTART = warmstart  # bool: compile-time warmstart switch
  SPARSE = sparse  # bool: compile-time sparsity switch

  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def kernel(
    # Data in:
    nefc_in: wp.array[int],
    qacc_warmstart_in: wp.array2d[float],
    qacc_smooth_in: wp.array2d[float],
    # Data out:
    qacc_out: wp.array2d[float],
    qfrc_constraint_out: wp.array2d[float],
  ):
    """Seeds one dof's qacc from the warmstart (or smooth) acceleration.

    Args:
      nefc_in: (nworld,) constraint row counts.
      qacc_warmstart_in: (nworld, nv) warmstart accelerations.
      qacc_smooth_in: (nworld, nv) unconstrained accelerations.
      qacc_out: (nworld, nv) solver iterate, seeded here.
      qfrc_constraint_out: (nworld, nv) zeroed for constraint-free worlds (sparse).
    """
    worldid, dofid = wp.tid()  # scalars: world, dof

    if wp.static(WARMSTART):
      qacc_out[worldid, dofid] = qacc_warmstart_in[worldid, dofid]
    else:
      qacc_out[worldid, dofid] = qacc_smooth_in[worldid, dofid]

    if wp.static(SPARSE):
      if nefc_in[worldid] == 0:
        qfrc_constraint_out[worldid, dofid] = 0.0

  return kernel


@wp.kernel(grid_stride=True)
def _solve_init_efc(
  # Data out:
  solver_niter_out: wp.array[int],
  # Out:
  ctx_search_dot_out: wp.array[float],
  ctx_done_out: wp.array[bool],
):
  """Resets the per-world solver state before a solve.

  Args:
    solver_niter_out: (nworld,) solver iteration counters, reset to 0.
    ctx_search_dot_out: (nworld,) search-direction dot products, reset to 0.
    ctx_done_out: (nworld,) convergence flags, reset to False.
  """
  worldid = wp.tid()  # scalar: world index
  solver_niter_out[worldid] = 0
  ctx_done_out[worldid] = False
  ctx_search_dot_out[worldid] = 0.0


@cache_kernel
def _solve_init_jaref_kernel(is_sparse: bool, nv: int, dofs_per_thread: int, compact: bool):
  """Builds the Jaref init kernel, specialized on the Jacobian layout.

  Args:
    is_sparse: bool; CSR constraint Jacobian layout, compiled in.
    nv: scalar number of dofs, compiled in.
    dofs_per_thread: scalar dofs accumulated per thread (dense split path).
    compact: bool; map Jacobian columns through dof_cdof (compacted solve).

  Returns:
    A kernel over (nworld, njmax, threads_per_efc) writing ctx_Jaref_out
    (nworld, njmax); with threads_per_efc > 1 the rows are atomically accumulated
    (caller zeroes ctx.Jaref first).
  """
  COMPACT = compact  # bool: compile-time compaction switch (must be a local for wp.static)

  @wp.kernel(module="unique", enable_backward=False, grid_stride=True)
  def kernel(
    # Data in:
    nefc_in: wp.array[int],
    qacc_in: wp.array2d[float],
    efc_J_rownnz_in: wp.array2d[int],
    efc_J_rowadr_in: wp.array2d[int],
    efc_J_colind_in: wp.array3d[int],
    efc_J_in: wp.array3d[float],
    efc_aref_in: wp.array2d[float],
    dof_cdof_in: wp.array2d[int],
    # Out:
    ctx_Jaref_out: wp.array2d[float],
  ):
    """Computes Jaref[worldid, efcid] = (J @ qacc - aref)[efcid].

    Args:
      nefc_in: (nworld,) constraint row counts.
      qacc_in: (nworld, nv) accelerations.
      efc_J_*: constraint Jacobian in the model's sparse or dense layout.
      efc_aref_in: (nworld, njmax) reference accelerations.
      dof_cdof_in: (nworld, nv) global dof -> compacted dof map (compact only).
      ctx_Jaref_out: (nworld, njmax) output residuals.
    """
    worldid, efcid, dofstart = wp.tid()  # scalars: world, efc row, dense dof block

    if efcid >= nefc_in[worldid]:
      return

    jaref = float(0.0)  # scalar: running dot product (J @ qacc)[efcid]
    if wp.static(is_sparse):
      rownnz = efc_J_rownnz_in[worldid, efcid]  # scalar: nonzeros in this row
      rowadr = efc_J_rowadr_in[worldid, efcid]  # scalar: row start in the sparse buffer
      for i in range(rownnz):
        sparseid = rowadr + i  # scalar: sparse entry index
        colind = efc_J_colind_in[worldid, 0, sparseid]  # scalar: dof column
        if wp.static(COMPACT):
          colind = dof_cdof_in[worldid, colind]
          if colind < 0:
            continue
        jaref += efc_J_in[worldid, 0, sparseid] * qacc_in[worldid, colind]
      ctx_Jaref_out[worldid, efcid] = jaref - efc_aref_in[worldid, efcid]
    else:
      if wp.static(dofs_per_thread >= nv):
        for i in range(wp.static(min(dofs_per_thread, nv))):
          jaref += efc_J_in[worldid, efcid, i] * qacc_in[worldid, i]
        ctx_Jaref_out[worldid, efcid] = jaref - efc_aref_in[worldid, efcid]

      else:
        for i in range(wp.static(dofs_per_thread)):
          ii = dofstart * wp.static(dofs_per_thread) + i  # scalar: dof index of this thread's block
          if ii < nv:
            jaref += efc_J_in[worldid, efcid, ii] * qacc_in[worldid, ii]

        if dofstart == 0:
          wp.atomic_add(ctx_Jaref_out, worldid, efcid, jaref - efc_aref_in[worldid, efcid])
        else:
          wp.atomic_add(ctx_Jaref_out, worldid, efcid, jaref)

  return kernel


@wp.kernel
def _solve_init_search_cg_tiled(
  # Model:
  nv: int,
  # In:
  ctx_grad_in: wp.array2d[float],
  ctx_Mgrad_in: wp.array2d[float],
  # Out:
  ctx_search_out: wp.array2d[float],
  ctx_search_dot_out: wp.array[float],
  ctx_prev_grad_out: wp.array2d[float],
  ctx_prev_Mgrad_out: wp.array2d[float],
):
  """Initializes the CG search direction: search = -Mgrad; stores prev grad/Mgrad.

  Args:
    nv: scalar number of dofs.
    ctx_grad_in: (nworld, nv_pad) cost gradient.
    ctx_Mgrad_in: (nworld, nv_pad) preconditioned gradient M^-1 grad.
    ctx_search_out: (nworld, nv) search direction -Mgrad.
    ctx_search_dot_out: (nworld,) search . search (tile-reduced).
    ctx_prev_grad_out: (nworld, nv) gradient saved for Polak-Ribiere.
    ctx_prev_Mgrad_out: (nworld, nv) preconditioned gradient saved for Polak-Ribiere.
  """
  worldid, tid = wp.tid()  # scalars: world, lane within the world's block

  local_search_dot = float(0.0)  # scalar: this lane's partial search . search
  BLOCK_DIM = wp.block_dim()  # scalar: lanes per world

  for dofid in range(tid, nv, BLOCK_DIM):
    mgrad = ctx_Mgrad_in[worldid, dofid]  # scalar: preconditioned gradient entry
    search = -1.0 * mgrad  # scalar: steepest-descent entry
    ctx_search_out[worldid, dofid] = search
    local_search_dot += search * search

    ctx_prev_grad_out[worldid, dofid] = ctx_grad_in[worldid, dofid]
    ctx_prev_Mgrad_out[worldid, dofid] = mgrad

  search_dot_tile = wp.tile(local_search_dot, preserve_type=True)
  search_dot_sum = wp.tile_reduce(wp.add, search_dot_tile)

  if tid == 0:
    ctx_search_dot_out[worldid] = search_dot_sum[0]


@cache_kernel
def _update_constraint_efc(track_changes: bool):
  """Builds the per-row constraint update kernel, specialized on change tracking.

  Args:
    track_changes: bool; record QUADRATIC-flag flips (quad_changed_*) and any state
      change (state_changed_count) for the incremental Newton fast path.

  Returns:
    A kernel over (nworld, njmax) writing efc_force/efc_state and, when tracking,
    the change counters.
  """
  TRACK_CHANGES = track_changes  # bool: compile-time tracking switch

  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def kernel(
    # Model:
    opt_impratio_invsqrt: wp.array[float],
    contact_impratio_invsqrt_in: wp.array[float],
    # Data in:
    ne_in: wp.array[int],
    nf_in: wp.array[int],
    nefc_in: wp.array[int],
    contact_friction_in: wp.array[vec5],
    contact_dim_in: wp.array[int],
    contact_efc_address_in: wp.array2d[int],
    efc_type_in: wp.array2d[int],
    efc_id_in: wp.array2d[int],
    efc_D_in: wp.array2d[float],
    efc_frictionloss_in: wp.array2d[float],
    nacon_in: wp.array[int],
    # In:
    ctx_Jaref_in: wp.array2d[float],
    ctx_ls_exhausted_in: wp.array[bool],
    ctx_done_in: wp.array[bool],
    # Data out:
    efc_force_out: wp.array2d[float],
    efc_state_out: wp.array2d[int],
    # Out:
    quad_changed_ids_out: wp.array2d[int],
    quad_changed_count_out: wp.array[int],
    state_changed_count_out: wp.array[int],
  ):
    """Re-evaluates one efc row's force, state, and change flags at the current Jaref.

    Args:
      opt_impratio_invsqrt: (1 or nworld,) world 1/sqrt(impratio), unread: the kernel
        reads the per-contact warp below.
      contact_impratio_invsqrt_in: (naconmax,) per-contact cone warp.
      ne_in: (nworld,) equality row counts.
      nf_in: (nworld,) friction row counts.
      nefc_in: (nworld,) total constraint row counts.
      contact_friction_in: (naconmax,) vec5 contact friction.
      contact_dim_in: (naconmax,) contact dimensionalities.
      contact_efc_address_in: (naconmax, ncondmax) efc row id of each contact row.
      efc_type_in: (nworld, njmax) ConstraintType per row.
      efc_id_in: (nworld, njmax) owning contact id per row.
      efc_D_in: (nworld, njmax_pad) constraint weights.
      efc_frictionloss_in: (nworld, njmax) frictionloss per row.
      nacon_in: (1,) number of active contacts.
      ctx_Jaref_in: (nworld, njmax) J @ qacc - aref.
      ctx_ls_exhausted_in: (nworld,) linesearch noise flags (forces a rebuild).
      ctx_done_in: (nworld,) converged flags; done worlds are skipped.
      efc_force_out: (nworld, njmax) constraint forces.
      efc_state_out: (nworld, njmax_pad) ConstraintState per row.
      quad_changed_ids_out: (nworld, njmax) rows whose QUADRATIC flag flipped.
      quad_changed_count_out: (nworld,) count of flipped rows (atomic).
      state_changed_count_out: (nworld,) count of any state changes (atomic).
    """
    worldid, efcid = wp.tid()  # scalars: world, efc row

    if ctx_done_in[worldid]:
      return

    # The linesearch flags worlds whose accepted step was rounding noise (stale
    # ray exhausted); count it as a state change so the fast path rebuilds them.
    if wp.static(TRACK_CHANGES):
      if efcid == 0 and ctx_ls_exhausted_in[worldid]:
        wp.atomic_add(state_changed_count_out, worldid, 1)

    if efcid >= nefc_in[worldid]:
      return

    # Read old state before overwriting
    if wp.static(TRACK_CHANGES):
      old_state = efc_state_out[worldid, efcid]  # scalar: state before this update

    efc_D = efc_D_in[worldid, efcid]  # scalar: constraint weight
    Jaref = ctx_Jaref_in[worldid, efcid]  # scalar: row residual

    ne = ne_in[worldid]  # scalar: equality row count
    nf = nf_in[worldid]  # scalar: friction row count

    is_equality = efcid < ne  # bool: rows [0, ne) are equalities
    is_friction = (not is_equality) and (efcid < ne + nf)  # bool: rows [ne, ne+nf) are frictionloss
    is_elliptic = efc_type_in[worldid, efcid] == ConstraintType.CONTACT_ELLIPTIC  # bool

    frictionloss = efc_frictionloss_in[worldid, efcid] if is_friction else 0.0  # scalar

    efcid0 = -1  # scalar: efc row of the contact's normal component (elliptic)
    jaref0 = float(0.0)  # scalar: normal-row residual (elliptic)
    D0 = float(0.0)  # scalar: normal-row weight (elliptic)
    mu = float(0.0)  # scalar: effective friction of the circularized cone (elliptic)
    ufrictionj = float(0.0)  # scalar: u_j * friction_j of this row (elliptic tangents)
    TT = float(0.0)  # scalar: squared tangential norm ||u_t||^2 (elliptic)

    if is_elliptic:
      conid = efc_id_in[worldid, efcid]  # scalar: owning contact
      if conid >= nacon_in[0]:
        return
      efcid0 = contact_efc_address_in[conid, 0]
      if efcid0 < 0:
        return

      dim = contact_dim_in[conid]  # scalar: contact dimensionality
      friction = contact_friction_in[conid]  # vec5: contact friction
      mu = friction[0] * contact_impratio_invsqrt_in[conid]
      jaref0 = ctx_Jaref_in[worldid, efcid0]
      D0 = efc_D_in[worldid, efcid0]

      for j in range(1, dim):
        efcidj = contact_efc_address_in[conid, j]  # scalar: efc row of tangent j
        if efcidj < 0:
          return
        frictionj = friction[j - 1]  # scalar: friction scale of tangent row j
        uj = ctx_Jaref_in[worldid, efcidj] * frictionj  # scalar: scaled tangential residual
        TT += uj * uj
        if efcid == efcidj:
          ufrictionj = uj * frictionj

    # vec3 (force, state, cost) of this row at the current Jaref
    res = _eval_constraint(
      is_equality,
      is_friction,
      is_elliptic,
      Jaref,
      efc_D,
      frictionloss,
      efcid,
      efcid0,
      jaref0,
      D0,
      mu,
      ufrictionj,
      TT,
    )

    new_state = int(res[1])  # scalar: ConstraintState after the update
    efc_force_out[worldid, efcid] = res[0]
    efc_state_out[worldid, efcid] = new_state

    if wp.static(TRACK_CHANGES):
      old_quad = old_state == ConstraintState.QUADRATIC.value  # bool: was QUADRATIC
      new_quad = new_state == ConstraintState.QUADRATIC.value  # bool: is QUADRATIC
      if old_quad != new_quad:
        idx = wp.atomic_add(quad_changed_count_out, worldid, 1)  # scalar: slot in the changed list
        quad_changed_ids_out[worldid, idx] = efcid
      # LINEARNEG <-> LINEARPOS friction transitions change the force without
      # changing the quadratic flag (or H); the fast path must still see them.
      if old_state != new_state:
        wp.atomic_add(state_changed_count_out, worldid, 1)

  return kernel


@wp.kernel
def _zero_qfrc_constraint_sparse(
  # In:
  state_changed_count_in: wp.array[int],
  ctx_done_in: wp.array[bool],
  # Data out:
  qfrc_constraint_out: wp.array2d[float],
):
  """Zeroes qfrc_constraint for worlds the sparse rebuild will repopulate.

  Args:
    state_changed_count_in: (nworld,) state-change counts; unchanged worlds skip.
    ctx_done_in: (nworld,) converged flags.
    qfrc_constraint_out: (nworld, nv) accumulation target.
  """
  # Only zero worlds the rebuild will repopulate; done worlds keep their value.
  worldid, dofid = wp.tid()  # scalars: world, dof

  if ctx_done_in[worldid]:
    return

  if state_changed_count_in[worldid] == 0:
    return

  qfrc_constraint_out[worldid, dofid] = 0.0


@cache_kernel
def _update_constraint_init_qfrc_constraint_sparse(compact: bool):
  """Builds the sparse qfrc_constraint accumulator, specialized on compaction.

  Args:
    compact: bool; map Jacobian columns through dof_cdof (compacted solve).

  Returns:
    A kernel over (nworld, njmax) atomically accumulating
    qfrc_constraint += efc_J.T @ efc_force (caller zeroes the target first).
  """
  COMPACT = compact  # bool: compile-time compaction switch (must be a local for wp.static)

  @wp.kernel(module="unique", enable_backward=False, grid_stride=True)
  def kernel(
    # Data in:
    nefc_in: wp.array[int],
    efc_J_rownnz_in: wp.array2d[int],
    efc_J_rowadr_in: wp.array2d[int],
    efc_J_colind_in: wp.array3d[int],
    efc_J_in: wp.array3d[float],
    efc_force_in: wp.array2d[float],
    dof_cdof_in: wp.array2d[int],
    # In:
    state_changed_count_in: wp.array[int],
    ctx_done_in: wp.array[bool],
    # Data out:
    qfrc_constraint_out: wp.array2d[float],
  ):
    """Scatters one row's force through its sparse Jacobian row into qfrc_constraint.

    Args:
      nefc_in: (nworld,) constraint row counts.
      efc_J_*: CSR constraint Jacobian.
      efc_force_in: (nworld, njmax) constraint forces.
      dof_cdof_in: (nworld, nv) global dof -> compacted dof map (compact only).
      state_changed_count_in: (nworld,) state-change counts; unchanged worlds skip.
      ctx_done_in: (nworld,) converged flags.
      qfrc_constraint_out: (nworld, nv) output J.T @ force (atomic adds).
    """
    worldid, efcid = wp.tid()  # scalars: world, efc row

    if ctx_done_in[worldid]:
      return

    if state_changed_count_in[worldid] == 0:
      return

    if efcid >= nefc_in[worldid]:
      return

    force = efc_force_in[worldid, efcid]  # scalar: this row's constraint force
    if force == 0.0:
      return

    rownnz = efc_J_rownnz_in[worldid, efcid]  # scalar: nonzeros in this row
    rowadr = efc_J_rowadr_in[worldid, efcid]  # scalar: row start in the sparse buffer
    for i in range(rownnz):
      sparseid = rowadr + i  # scalar: sparse entry index
      colind = efc_J_colind_in[worldid, 0, sparseid]  # scalar: dof column
      if wp.static(COMPACT):
        colind = dof_cdof_in[worldid, colind]
        if colind < 0:
          continue
      efc_J = efc_J_in[worldid, 0, sparseid]  # scalar: Jacobian entry
      wp.atomic_add(qfrc_constraint_out[worldid], colind, efc_J * force)

  return kernel


@wp.kernel
def _qfrc_constraint_from_grad(
  # Data in:
  qfrc_smooth_in: wp.array2d[float],
  efc_Ma_in: wp.array2d[float],
  # In:
  ctx_grad_in: wp.array2d[float],
  ctx_grad_scale_in: wp.array[float],
  # Data out:
  qfrc_constraint_out: wp.array2d[float],
):
  """Recovers qfrc_constraint = Ma - qfrc_smooth - grad_scale * grad after a solve.

  The incremental fast path leaves qfrc_constraint stale; the gradient identity
  grad = Ma - qfrc_smooth - qfrc_constraint (up to the tracked scale) inverts it.

  Args:
    qfrc_smooth_in: (nworld, nv) smooth force.
    efc_Ma_in: (nworld, nv) M @ qacc.
    ctx_grad_in: (nworld, nv_pad) last rebuilt gradient g.
    ctx_grad_scale_in: (nworld,) scale of the true gradient relative to g.
    qfrc_constraint_out: (nworld, nv) recovered constraint force.
  """
  worldid, dofid = wp.tid()  # scalars: world, dof

  grad = ctx_grad_scale_in[worldid] * ctx_grad_in[worldid, dofid]  # scalar: true gradient entry
  qfrc_constraint_out[worldid, dofid] = efc_Ma_in[worldid, dofid] - qfrc_smooth_in[worldid, dofid] - grad


@cache_kernel
def _update_constraint_init_qfrc_constraint_dense(stable_fast: bool):
  """Builds the dense qfrc_constraint kernel, specialized on the stable fast path.

  Args:
    stable_fast: bool; worlds with no state flips skip the rebuild (their value is
      recovered from the gradient after the solve, see _qfrc_constraint_from_grad).

  Returns:
    A kernel over (nworld, nv) writing qfrc_constraint = efc_J.T @ efc_force.
  """
  STABLE_FAST = stable_fast  # bool: compile-time fast-path switch

  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def kernel(
    # Data in:
    nefc_in: wp.array[int],
    efc_J_in: wp.array3d[float],
    efc_force_in: wp.array2d[float],
    njmax_in: int,
    # In:
    state_changed_count_in: wp.array[int],
    ctx_done_in: wp.array[bool],
    # Data out:
    qfrc_constraint_out: wp.array2d[float],
  ):
    """Computes one dof's qfrc_constraint = sum over rows of J[efcid, dofid] * force.

    Args:
      nefc_in: (nworld,) constraint row counts.
      efc_J_in: (nworld, njmax_pad, nv_pad) dense constraint Jacobian.
      efc_force_in: (nworld, njmax) constraint forces.
      njmax_in: scalar maximum rows per world.
      state_changed_count_in: (nworld,) state-change counts (fast-path gate).
      ctx_done_in: (nworld,) converged flags.
      qfrc_constraint_out: (nworld, nv) output J.T @ force.
    """
    worldid, dofid = wp.tid()  # scalars: world, dof

    if ctx_done_in[worldid]:
      return

    # Fast path: stale qfrc_constraint is never read; recovered after the solve.
    if wp.static(STABLE_FAST):
      if state_changed_count_in[worldid] == 0:
        return

    sum_qfrc = float(0.0)  # scalar: running (J.T @ force)[dofid]
    for efcid in range(min(njmax_in, nefc_in[worldid])):
      efc_J = efc_J_in[worldid, efcid, dofid]  # scalar: Jacobian entry
      force = efc_force_in[worldid, efcid]  # scalar: row force
      sum_qfrc += efc_J * force

    qfrc_constraint_out[worldid, dofid] = sum_qfrc

  return kernel


@wp.kernel
def _update_gradient_h_incremental(
  # Data in:
  efc_J_in: wp.array3d[float],
  efc_D_in: wp.array2d[float],
  efc_state_in: wp.array2d[int],
  # In:
  quad_changed_ids_in: wp.array2d[int],
  quad_changed_count_in: wp.array[int],
  # Out:
  ctx_h_out: wp.array3d[float],
):
  """Incrementally update upper triangle of H for changed constraints.

  Each thread handles one unique (i, j) element and writes it to the upper triangle.
  For each changed constraint, adds or subtracts D * J[i] * J[j].

  Args:
    efc_J_in: (nworld, njmax_pad, nv_pad) dense constraint Jacobian.
    efc_D_in: (nworld, njmax_pad) constraint weights.
    efc_state_in: (nworld, njmax_pad) ConstraintState per row.
    quad_changed_ids_in: (nworld, njmax) rows whose QUADRATIC flag flipped.
    quad_changed_count_in: (nworld,) number of flipped rows.
    ctx_h_out: (nworld, nv_pad, nv_pad) Hessian, upper triangle updated in place.
  """
  worldid, elementid = wp.tid()  # scalars: world, upper-triangle element

  n_changes = quad_changed_count_in[worldid]  # scalar: flipped rows this iteration
  if n_changes == 0:
    return

  # Upper-triangle enumeration: elementid -> (row, col) where row <= col.
  col = (int(wp.sqrt(float(1 + 8 * elementid))) - 1) // 2
  row = elementid - (col * (col + 1)) // 2

  delta = float(0.0)  # scalar: accumulated Hessian change at (row, col)
  for change_idx in range(n_changes):
    efcid = quad_changed_ids_in[worldid, change_idx]  # scalar: flipped row
    Jrow = efc_J_in[worldid, efcid, row]  # scalar: J entry at (efcid, row)
    if Jrow == 0.0:
      continue
    Jcol = efc_J_in[worldid, efcid, col]  # scalar: J entry at (efcid, col)
    if Jcol == 0.0:
      continue

    D = efc_D_in[worldid, efcid]  # scalar: row weight
    if efc_state_in[worldid, efcid] == ConstraintState.QUADRATIC.value:
      delta += D * Jrow * Jcol
    else:
      delta -= D * Jrow * Jcol

  if delta != 0.0:
    ctx_h_out[worldid, row, col] += delta


@cache_kernel
def _update_gradient_h_incremental_sparse(compact: bool):
  """Builds the sparse incremental H update, specialized on compaction.

  Args:
    compact: bool; map Jacobian columns through dof_cdof (compacted solve).

  Returns:
    A kernel over (nworld, slots_per_world, _JTDAJ_THREADS_PER_GROUP) updating
    ctx_h's upper triangle in place (atomic adds).
  """
  COMPACT = compact  # bool: compile-time compaction switch (must be a local for wp.static)

  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def kernel(
    # Data in:
    efc_J_rownnz_in: wp.array2d[int],
    efc_J_rowadr_in: wp.array2d[int],
    efc_J_colind_in: wp.array3d[int],
    efc_J_in: wp.array3d[float],
    efc_D_in: wp.array2d[float],
    efc_state_in: wp.array2d[int],
    dof_cdof_in: wp.array2d[int],
    # In:
    quad_changed_ids_in: wp.array2d[int],
    quad_changed_count_in: wp.array[int],
    slots_per_world: int,
    # Out:
    ctx_h_out: wp.array3d[float],
  ):
    """Incrementally update upper triangle of H for changed constraints (sparse J).

    One warp per changed constraint row: the lanes split the row's upper-triangular
    entries (same sqrt triangular-number decode as _JTDACJ_sparse), replacing the
    serial nnz^2 loop that dominated this kernel.

    Args:
      efc_J_*: CSR constraint Jacobian.
      efc_D_in: (nworld, njmax_pad) constraint weights.
      efc_state_in: (nworld, njmax_pad) ConstraintState per row.
      dof_cdof_in: (nworld, nv) global dof -> compacted dof map (compact only).
      quad_changed_ids_in: (nworld, njmax) rows whose QUADRATIC flag flipped.
      quad_changed_count_in: (nworld,) number of flipped rows.
      slots_per_world: scalar warp groups striding the changed rows per world.
      ctx_h_out: (nworld, nv_pad, nv_pad) Hessian, upper triangle updated (atomic).
    """
    worldid, slot, lane = wp.tid()  # scalars: world, row group, lane within the group

    n_changes = quad_changed_count_in[worldid]  # scalar: flipped rows this iteration
    for change_idx in range(slot, n_changes, slots_per_world):
      efcid = quad_changed_ids_in[worldid, change_idx]  # scalar: flipped row
      D = efc_D_in[worldid, efcid]  # scalar: row weight
      sign = float(0.0)  # scalar: +D on activation, -D on deactivation
      if efc_state_in[worldid, efcid] == ConstraintState.QUADRATIC.value:
        sign = D
      else:
        sign = -D

      rownnz = efc_J_rownnz_in[worldid, efcid]  # scalar: nonzeros in this row
      rowadr = efc_J_rowadr_in[worldid, efcid]  # scalar: row start in the sparse buffer
      n_entries = rownnz * (rownnz + 1) // 2  # scalar: upper-triangle pairs of the row

      for entry in range(lane, n_entries, wp.static(_JTDAJ_THREADS_PER_GROUP)):
        ii = int((wp.sqrt(float(8 * entry + 1)) - 1.0) * 0.5)  # scalar: triangular decode
        jj = entry - ii * (ii + 1) // 2  # scalar: pair member with jj <= ii
        Ji = efc_J_in[worldid, 0, rowadr + ii]  # scalar: J entry i
        Jj = efc_J_in[worldid, 0, rowadr + jj]  # scalar: J entry j
        h = sign * Ji * Jj  # scalar: Hessian delta of this pair
        if h != 0.0:
          colindi = efc_J_colind_in[worldid, 0, rowadr + ii]  # scalar: dof column i
          colindj = efc_J_colind_in[worldid, 0, rowadr + jj]  # scalar: dof column j
          if wp.static(COMPACT):
            colindi = dof_cdof_in[worldid, colindi]
            colindj = dof_cdof_in[worldid, colindj]
            if colindi < 0 or colindj < 0:
              continue
          wp.atomic_add(ctx_h_out[worldid, wp.min(colindi, colindj)], wp.max(colindi, colindj), h)

  return kernel


def _update_constraint(
  m: Model,
  d: Data,
  ctx: SolverContext | InverseContext,
  track_changes: bool = False,
  stable_fast: bool = False,
):
  """Update constraint arrays after each solve iteration.

  Recomputes efc.force/efc.state from ctx.Jaref, then rebuilds
  qfrc_constraint = efc_J.T @ efc_force in the model's dense or sparse layout.

  Args:
    m: model.
    d: data; writes efc.force (nworld, njmax), efc.state (nworld, njmax_pad), and
      qfrc_constraint (nworld, nv).
    ctx: solver or inverse context; reads ctx.Jaref (nworld, njmax) and ctx.done.
    track_changes: record state changes for the incremental Newton path.
    stable_fast: worlds with no state flips skip the qfrc_constraint rebuild.
  """
  efc_inputs = [
    m.opt.impratio_invsqrt,
      d.contact_impratio_invsqrt,
    d.ne,
    d.nf,
    d.nefc,
    d.contact.friction,
    d.contact.dim,
    d.contact.efc_address,
    d.efc.type,
    d.efc.id,
    d.efc.D,
    d.efc.frictionloss,
    d.nacon,
    ctx.Jaref,
    ctx.ls_exhausted,
    ctx.done,
  ]

  wp.launch(
    _update_constraint_efc(track_changes),
    dim=(d.nworld, d.njmax),
    inputs=efc_inputs,
    outputs=[d.efc.force, d.efc.state, ctx.quad_changed_ids, ctx.quad_changed_count, ctx.state_changed_count],
  )

  # qfrc_constraint = efc_J.T @ efc_force. Fast-path worlds with no state flips
  # skip the rebuild; the public value is recovered after the solve.
  changed = ctx.state_changed_count if stable_fast else d.nefc  # (nworld,) gate (placeholder when off)
  sc = _sparse_compact(ctx)  # bool: compact solve over a sparse full model
  if m.is_sparse or sc:
    dj = ctx.compact_d_full if sc else d  # data holding the J structures to read
    wp.launch(
      _zero_qfrc_constraint_sparse,
      dim=(d.nworld, m.nv),
      inputs=[changed, ctx.done],
      outputs=[d.qfrc_constraint],
    )
    wp.launch(
      _update_constraint_init_qfrc_constraint_sparse(sc),
      dim=(d.nworld, d.njmax),
      inputs=[d.nefc, dj.efc.J_rownnz, dj.efc.J_rowadr, dj.efc.J_colind, dj.efc.J, d.efc.force, dj.dof_cdof, changed, ctx.done],
      outputs=[d.qfrc_constraint],
    )
  else:
    wp.launch(
      _update_constraint_init_qfrc_constraint_dense(stable_fast),
      dim=(d.nworld, m.nv),
      inputs=[d.nefc, d.efc.J, d.efc.force, d.njmax, changed, ctx.done],
      outputs=[d.qfrc_constraint],
    )


@cache_kernel
def _update_gradient_zero_grad_dot(stable_fast: bool):
  """Builds the pre-gradient reset kernel, specialized on the stable fast path.

  Args:
    stable_fast: bool; unchanged worlds keep a rescaled stale gradient instead of a
      rebuild (see the in-kernel comments).

  Returns:
    A kernel over (nworld,) resetting grad_dot/newton_decrement/grad_scale and
    setting the search_unchanged reuse flag.
  """
  STABLE_FAST = stable_fast  # bool: compile-time fast-path switch

  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def kernel(
    # In:
    state_changed_count_in: wp.array[int],
    ctx_alpha_in: wp.array[float],
    ctx_done_in: wp.array[bool],
    # Out:
    ctx_grad_dot_out: wp.array[float],
    ctx_newton_decrement_out: wp.array[float],
    ctx_grad_scale_out: wp.array[float],
    ctx_search_unchanged_out: wp.array[bool],
  ):
    """Resets one world's gradient accumulators, or rescales them on the fast path.

    Args:
      state_changed_count_in: (nworld,) state-change counts of the last update.
      ctx_alpha_in: (nworld,) last accepted linesearch step.
      ctx_done_in: (nworld,) converged flags.
      ctx_grad_dot_out: (nworld,) grad . grad accumulator.
      ctx_newton_decrement_out: (nworld,) grad . search of the Newton solve.
      ctx_grad_scale_out: (nworld,) scale of the true gradient vs the stale one.
      ctx_search_unchanged_out: (nworld,) mv/jv reuse flags for the linesearch.
    """
    worldid = wp.tid()  # scalar: world index

    if wp.static(STABLE_FAST):
      ctx_search_unchanged_out[worldid] = ctx_done_in[worldid] or state_changed_count_in[worldid] == 0
    else:
      ctx_search_unchanged_out[worldid] = False

    if ctx_done_in[worldid]:
      return

    # Fast path: grad stays stale at its last rebuilt value g. The true
    # gradient is grad_scale * g, and a linesearch step t along the (equally
    # stale) search direction changes it to (grad_scale - t) * g.
    if wp.static(STABLE_FAST):
      if state_changed_count_in[worldid] == 0:
        sigma = ctx_grad_scale_out[worldid]  # scalar: current gradient scale
        new_sigma = sigma - ctx_alpha_in[worldid]  # scalar: scale after the step
        ratio = float(0.0)  # scalar: new/old scale
        if sigma != 0.0:
          ratio = new_sigma / sigma
        ratio_sq = ratio * ratio  # scalar: quadratic quantities rescale by ratio^2
        ctx_grad_dot_out[worldid] *= ratio_sq
        ctx_newton_decrement_out[worldid] *= ratio_sq
        ctx_grad_scale_out[worldid] = new_sigma
        return

    ctx_grad_dot_out[worldid] = 0.0
    ctx_newton_decrement_out[worldid] = 0.0
    ctx_grad_scale_out[worldid] = 1.0

  return kernel


@cache_kernel
def _update_gradient_grad(stable_fast: bool):
  """Builds the per-dof gradient kernel, specialized on the stable fast path.

  Args:
    stable_fast: bool; unchanged worlds keep the stale gradient
      (see _update_gradient_zero_grad_dot).

  Returns:
    A kernel over (nworld, nv) writing ctx_grad and atomically accumulating grad_dot.
  """
  STABLE_FAST = stable_fast  # bool: compile-time fast-path switch

  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def kernel(
    # Data in:
    qfrc_smooth_in: wp.array2d[float],
    qfrc_constraint_in: wp.array2d[float],
    efc_Ma_in: wp.array2d[float],
    # In:
    state_changed_count_in: wp.array[int],
    ctx_done_in: wp.array[bool],
    # Out:
    ctx_grad_out: wp.array2d[float],
    ctx_grad_dot_out: wp.array[float],
  ):
    """Computes one dof's grad = Ma - qfrc_smooth - qfrc_constraint and its dot.

    Args:
      qfrc_smooth_in: (nworld, nv) smooth force.
      qfrc_constraint_in: (nworld, nv) constraint force.
      efc_Ma_in: (nworld, nv) M @ qacc.
      state_changed_count_in: (nworld,) state-change counts (fast-path gate).
      ctx_done_in: (nworld,) converged flags.
      ctx_grad_out: (nworld, nv_pad) cost gradient.
      ctx_grad_dot_out: (nworld,) grad . grad (atomic adds).
    """
    worldid, dofid = wp.tid()  # scalars: world, dof

    if ctx_done_in[worldid]:
      return

    # Fast path: grad stays stale (see _update_gradient_zero_grad_dot).
    if wp.static(STABLE_FAST):
      if state_changed_count_in[worldid] == 0:
        return

    grad = efc_Ma_in[worldid, dofid] - qfrc_smooth_in[worldid, dofid] - qfrc_constraint_in[worldid, dofid]  # scalar
    ctx_grad_out[worldid, dofid] = grad
    wp.atomic_add(ctx_grad_dot_out, worldid, grad * grad)

  return kernel


@wp.kernel
def _update_gradient_grad_tiled(
  # Model:
  nv: int,
  # Data in:
  qfrc_smooth_in: wp.array2d[float],
  qfrc_constraint_in: wp.array2d[float],
  efc_Ma_in: wp.array2d[float],
  # In:
  ctx_done_in: wp.array[bool],
  # Out:
  ctx_grad_out: wp.array2d[float],
  ctx_grad_dot_out: wp.array[float],
):
  """Tiled (CG) gradient: grad = Ma - qfrc_smooth - qfrc_constraint, tile-reduced dot.

  Args:
    nv: scalar number of dofs.
    qfrc_smooth_in: (nworld, nv) smooth force.
    qfrc_constraint_in: (nworld, nv) constraint force.
    efc_Ma_in: (nworld, nv) M @ qacc.
    ctx_done_in: (nworld,) converged flags.
    ctx_grad_out: (nworld, nv_pad) cost gradient.
    ctx_grad_dot_out: (nworld,) grad . grad.
  """
  worldid, tid = wp.tid()  # scalars: world, lane within the world's block

  if ctx_done_in[worldid]:
    return

  local_grad_dot = float(0.0)  # scalar: this lane's partial grad . grad
  BLOCK_DIM = wp.block_dim()  # scalar: lanes per world

  for dofid in range(tid, nv, BLOCK_DIM):
    grad = efc_Ma_in[worldid, dofid] - qfrc_smooth_in[worldid, dofid] - qfrc_constraint_in[worldid, dofid]  # scalar
    ctx_grad_out[worldid, dofid] = grad
    local_grad_dot += grad * grad

  grad_dot_tile = wp.tile(local_grad_dot, preserve_type=True)
  grad_dot_sum = wp.tile_reduce(wp.add, grad_dot_tile)

  if tid == 0:
    ctx_grad_dot_out[worldid] = grad_dot_sum[0]


@cache_kernel
def _update_gradient_init_h_sparse(compact: bool):
  """Builds the H initializer that densifies the sparse inertia M into the Hessian.

  Args:
    compact: bool; index through cdof_dof and pad inactive dofs with identity.

  Returns:
    A kernel over (nworld, nv_pad, nv_pad) writing ctx_h's upper triangle = M.
  """
  COMPACT = compact  # bool: compile-time compaction switch (must be a local for wp.static)

  @wp.kernel(module="unique", enable_backward=False, grid_stride=True)
  def kernel(
    # Model:
    nv: int,
    M_elemid: wp.array2d[int],
    # Data in:
    M_in: wp.array2d[float],
    cdof_dof_in: wp.array2d[int],
    # In:
    ctx_done_in: wp.array[bool],
    # Out:
    ctx_h_out: wp.array3d[float],
  ):
    """Writes one upper-triangle entry of H from the sparse inertia M.

    Args:
      nv: scalar number of dofs of the (full) model.
      M_elemid: (nv, nv) CSR element index of each lower-triangle dof pair, -1 if absent.
      M_in: (nworld, nC) CSR inertia values.
      cdof_dof_in: (nworld, nvmax_pad) compacted dof -> global dof map (compact only).
      ctx_done_in: (nworld,) converged flags.
      ctx_h_out: (nworld, nv_pad, nv_pad) Hessian; upper triangle initialized to M.
    """
    worldid, i, j = wp.tid()  # scalars: world, Hessian row, Hessian column

    if ctx_done_in[worldid]:
      return

    # only write the upper triangle; Cholesky reads the upper triangle only
    if j < i:
      return

    if wp.static(COMPACT):
      dof_i = cdof_dof_in[worldid, i]  # scalar: global dof of row i (-1 on padding)
      dof_j = cdof_dof_in[worldid, j]  # scalar: global dof of column j (-1 on padding)
      if dof_i < 0 or dof_j < 0:
        # per-world padded block: identity keeps the factorization well conditioned
        ctx_h_out[worldid, i, j] = wp.where(i == j, 1.0, 0.0)
        return
    else:
      dof_i = i
      dof_j = j
      if i >= nv or j >= nv:
        ctx_h_out[worldid, i, j] = 0.0
        return

    # sparse M is stored in the lower triangle, so look up (larger, smaller)
    elemid = M_elemid[wp.max(dof_i, dof_j), wp.min(dof_i, dof_j)]  # scalar: CSR entry (-1 if zero)
    if elemid >= 0:
      ctx_h_out[worldid, i, j] = M_in[worldid, elemid]
    else:
      ctx_h_out[worldid, i, j] = 0.0

  return kernel


@wp.func
def _state_check(D: float, state: int) -> float:
  """Masks a constraint weight: D for QUADRATIC (active) rows, 0 otherwise."""
  if state == ConstraintState.QUADRATIC.value:
    return D
  else:
    return 0.0


@wp.func
def _active_check(tid: int, threshold: int) -> float:
  """1.0 for in-range tile lanes (tid < threshold), 0.0 for padding lanes."""
  if tid >= threshold:
    return 0.0
  else:
    return 1.0


@cache_kernel
def _update_gradient_JTDAJ_dense_tiled_compact(nv_pad: int, tile_size: int, njmax: int):
  """Compact-path variant of _update_gradient_JTDAJ_dense_tiled.

  Takes M_in as a dense 3D array (nworld, nv_pad, nv_pad) -- the compacted active-DOF
  inertia block cM -- instead of a 2D CSR array.  Cholesky reads fill_mode="upper";
  cM is full-symmetric so the tile_load covers both triangles and the upper triangle
  of the result is correct.
  """
  if njmax < tile_size:
    tile_size = njmax

  TILE_SIZE_K = tile_size  # scalar: constraint rows per slab

  @wp.kernel(module="unique", enable_backward=False, grid_stride=False, module_options={"enable_mathdx_gemm": False})
  def kernel(
    # Data in:
    nefc_in: wp.array[int],
    M_in: wp.array3d[float],  # kernel_analyzer: ignore; compact dense inertia (nworld, nv_pad, nv_pad)
    efc_J_in: wp.array3d[float],
    efc_D_in: wp.array2d[float],
    efc_state_in: wp.array2d[int],
    # In:
    ctx_done_in: wp.array[bool],
    # Out:
    ctx_h_out: wp.array3d[float],
  ):
    """Builds one world's H = cM + J.T @ diag(D * active) @ J with tile operations.

    Args:
      nefc_in: (nworld,) constraint row counts.
      M_in: (nworld, nv_pad, nv_pad) compact dense inertia.
      efc_J_in: (nworld, njmax_pad, nv_pad) dense constraint Jacobian.
      efc_D_in: (nworld, njmax_pad) constraint weights.
      efc_state_in: (nworld, njmax_pad) ConstraintState per row.
      ctx_done_in: (nworld,) converged flags.
      ctx_h_out: (nworld, nv_pad, nv_pad) output Hessian.
    """
    worldid, rank = wp.tid()  # scalars: world, lane within the world's block

    if ctx_done_in[worldid]:
      return

    nefc = nefc_in[worldid]  # scalar: active rows

    # Load full symmetric compact inertia tile (Cholesky only reads upper triangle).
    sum_val = wp.tile_load(M_in[worldid], shape=(wp.static(nv_pad), wp.static(nv_pad)), bounds_check=True)

    for k in range(0, njmax, TILE_SIZE_K):
      if k >= nefc:
        break

      # J rows of this slab (TILE_SIZE_K, nv_pad), masked weights, and the slab product
      J_kj = wp.tile_load(efc_J_in[worldid], shape=(TILE_SIZE_K, nv_pad), offset=(k, 0), bounds_check=False)

      D_k = wp.tile_load(efc_D_in[worldid], shape=TILE_SIZE_K, offset=k, bounds_check=False)
      state = wp.tile_load(efc_state_in[worldid], shape=TILE_SIZE_K, offset=k, bounds_check=False)

      D_k = wp.tile_map(_state_check, D_k, state)

      tid_tile = wp.tile_arange(TILE_SIZE_K, dtype=int)
      threshold_tile = wp.tile_ones(shape=TILE_SIZE_K, dtype=int) * (nefc - k)

      active_tile = wp.tile_map(_active_check, tid_tile, threshold_tile)
      D_k = wp.tile_map(wp.mul, active_tile, D_k)

      J_ki = wp.tile_map(wp.mul, wp.tile_transpose(J_kj), wp.tile_broadcast(D_k, shape=(nv_pad, TILE_SIZE_K)))

      sum_val += wp.tile_matmul(J_ki, J_kj)

    wp.tile_store(ctx_h_out[worldid], sum_val, bounds_check=False)

  return kernel


@cache_kernel
def _update_gradient_JTDAJ_dense_tiled(nv_pad: int, tile_size: int, njmax: int, nC: int):
  """Builds the dense tiled H = M + J.T @ diag(D * active) @ J kernel.

  Args:
    nv_pad: scalar padded dof count (tile edge), compiled in.
    tile_size: scalar constraint-slab size (clamped to njmax), compiled in.
    njmax: scalar maximum rows per world, compiled in.
    nC: scalar number of CSR inertia entries, compiled in.

  Returns:
    A tiled kernel over (nworld,) writing ctx_h (nworld, nv_pad, nv_pad); M is
    densified from CSR into the upper triangle first.
  """
  if njmax < tile_size:
    tile_size = njmax

  TILE_SIZE_K = tile_size  # scalar: constraint rows per slab

  @wp.kernel(module="unique", enable_backward=False, grid_stride=False, module_options={"enable_mathdx_gemm": False})
  def kernel(
    # Model:
    M_colind: wp.array[int],  # column index of each CSR entry
    M_hinit_i: wp.array[int],  # row index of each CSR entry
    # Data in:
    nefc_in: wp.array[int],
    M_in: wp.array2d[float],  # CSR M (nworld, nC)
    efc_J_in: wp.array3d[float],
    efc_D_in: wp.array2d[float],
    efc_state_in: wp.array2d[int],
    # In:
    ctx_done_in: wp.array[bool],
    # Out:
    ctx_h_out: wp.array3d[float],
  ):
    """Builds one world's H: densifies CSR M into a shared tile, then accumulates
    J.T @ diag(D * active) @ J over constraint slabs.

    Args:
      M_colind: (nC,) column index of each CSR inertia entry.
      M_hinit_i: (nC,) row index of each CSR inertia entry.
      nefc_in: (nworld,) constraint row counts.
      M_in: (nworld, nC) CSR inertia values.
      efc_J_in: (nworld, njmax_pad, nv_pad) dense constraint Jacobian.
      efc_D_in: (nworld, njmax_pad) constraint weights.
      efc_state_in: (nworld, njmax_pad) ConstraintState per row.
      ctx_done_in: (nworld,) converged flags.
      ctx_h_out: (nworld, nv_pad, nv_pad) output Hessian.
    """
    worldid, rank = wp.tid()  # scalars: world, lane within the world's block

    if ctx_done_in[worldid]:
      return

    nefc = nefc_in[worldid]  # scalar: active rows

    # Densify M's upper triangle from CSR into the shared H tile (Cholesky reads fill_mode="upper").
    # Entry (i, col<=i) of M goes to the upper position (col, i); the lower triangle stays zero.
    m_tile = wp.tile_zeros(shape=(wp.static(nv_pad * nv_pad),), dtype=float, storage="shared")
    # Uniform trip count from the RUNTIME lane count: every lane iterates iters times (so the
    # collective tile_scatter_add is always called by all lanes -- a divergent range(rank, nC, bd)
    # deadlocks). wp.block_dim() is 1 on the CPU backend, so lane 0 then covers every entry.
    lanes = wp.block_dim()  # scalar: lanes per world
    iters = (nC + lanes - 1) // lanes  # scalar: uniform trip count over CSR entries
    for it in range(iters):
      e = it * lanes + rank  # scalar: CSR entry index for this lane
      enable = e < nC  # bool: in-range lanes contribute
      ec = wp.where(enable, e, 0)  # scalar: clamped index (safe load)
      pos = M_colind[ec] * nv_pad + M_hinit_i[ec]  # scalar: flat upper-triangle position
      wp.tile_scatter_add(m_tile, pos, wp.where(enable, M_in[worldid, ec], 0.0), enable)
    sum_val = wp.tile_reshape(m_tile, (nv_pad, nv_pad))  # tile (nv_pad, nv_pad): running H

    # Each tile processes one output tile by looping over all constraints
    for k in range(0, njmax, TILE_SIZE_K):
      if k >= nefc:
        break

      # AD: leaving bounds-check disabled here because I'm not entirely sure that
      # everything always hits the fast path. The padding takes care of any
      #  potential OOB accesses.
      J_kj = wp.tile_load(efc_J_in[worldid], shape=(TILE_SIZE_K, nv_pad), offset=(k, 0), bounds_check=False)

      # state check
      D_k = wp.tile_load(efc_D_in[worldid], shape=TILE_SIZE_K, offset=k, bounds_check=False)
      state = wp.tile_load(efc_state_in[worldid], shape=TILE_SIZE_K, offset=k, bounds_check=False)

      D_k = wp.tile_map(_state_check, D_k, state)

      # force unused elements to be zero
      tid_tile = wp.tile_arange(TILE_SIZE_K, dtype=int)
      threshold_tile = wp.tile_ones(shape=TILE_SIZE_K, dtype=int) * (nefc - k)

      active_tile = wp.tile_map(_active_check, tid_tile, threshold_tile)
      D_k = wp.tile_map(wp.mul, active_tile, D_k)

      J_ki = wp.tile_map(wp.mul, wp.tile_transpose(J_kj), wp.tile_broadcast(D_k, shape=(nv_pad, TILE_SIZE_K)))

      sum_val += wp.tile_matmul(J_ki, J_kj)

    wp.tile_store(ctx_h_out[worldid], sum_val, bounds_check=False)

  return kernel


@wp.func
def _elliptic_hessian_entry_from_projections(
  # In:
  dm: float,
  mu: float,
  mu_n_over_t: float,
  tangent_diag: float,
  z01: float,
  z02: float,
  projection1: float,
  projection2: float,
  tangent_dot: float,
) -> float:
  """One (dof1, dof2) entry of the elliptic middle-zone cone curvature.

  Implements dm * (grad_r[dof1] * grad_r[dof2] - mu * r * hess_T[dof1, dof2]) for the
  cone cost 0.5 * dm * r^2, r = N - mu*T, folded into the precomputed scalars; see
  _update_gradient_JTCJ_dense for the projection/tangent_dot definitions.
  """
  # Contract the diagonal-plus-rank-one curvature without materializing the cone Hessian.
  return dm * (
    z01 * z02
    - mu * (z01 * projection2 + z02 * projection1)
    + mu_n_over_t * projection1 * projection2
    + tangent_diag * tangent_dot
  )


@wp.kernel
def _update_gradient_JTCJ_dense(
  # Model:
  opt_impratio_invsqrt: wp.array[float],
  contact_impratio_invsqrt_in: wp.array[float],
  dof_tri_row: wp.array[int],
  dof_tri_col: wp.array[int],
  # Data in:
  contact_dist_in: wp.array[float],
  contact_includemargin_in: wp.array[float],
  contact_friction_in: wp.array[vec5],
  contact_dim_in: wp.array[int],
  contact_efc_address_in: wp.array2d[int],
  contact_worldid_in: wp.array[int],
  efc_J_in: wp.array3d[float],
  efc_D_in: wp.array2d[float],
  efc_state_in: wp.array2d[int],
  naconmax_in: int,
  nacon_in: wp.array[int],
  # In:
  ctx_Jaref_in: wp.array2d[float],
  ctx_done_in: wp.array[bool],
  nblocks_perblock: int,
  dim_block: int,
  # Out:
  ctx_h_out: wp.array3d[float],
):
  """Adds the elliptic middle-zone cone curvature to H (dense path).

  Each thread owns one upper-triangle (dof1, dof2) pair and strides over contacts,
  accumulating the cone term for contacts whose normal row is in the CONE state.

  Args:
    opt_impratio_invsqrt: (1 or nworld,) world 1/sqrt(impratio).
    contact_impratio_invsqrt_in: (naconmax,) per-contact cone warp.
    dof_tri_row: (nv*(nv+1)/2,) row of each upper-triangle dof pair.
    dof_tri_col: (nv*(nv+1)/2,) column of each upper-triangle dof pair.
    contact_dist_in: (naconmax,) contact distances.
    contact_includemargin_in: (naconmax,) inclusion margins.
    contact_friction_in: (naconmax,) vec5 contact friction.
    contact_dim_in: (naconmax,) contact dimensionalities.
    contact_efc_address_in: (naconmax, ncondmax) efc row id of each contact row.
    contact_worldid_in: (naconmax,) world of each contact.
    efc_J_in: (nworld, njmax_pad, nv_pad) dense constraint Jacobian.
    efc_D_in: (nworld, njmax_pad) constraint weights.
    efc_state_in: (nworld, njmax_pad) ConstraintState per row.
    naconmax_in: scalar contact capacity.
    nacon_in: (1,) number of active contacts.
    ctx_Jaref_in: (nworld, njmax) J @ qacc - aref.
    ctx_done_in: (nworld,) converged flags.
    nblocks_perblock: scalar contacts strided per launch block.
    dim_block: scalar launch blocks over contacts.
    ctx_h_out: (nworld, nv_pad, nv_pad) Hessian, upper triangle updated in place.
  """
  conid_start, elementid = wp.tid()  # scalars: first contact of this block, dof pair

  dof1id = dof_tri_row[elementid]  # scalar: first dof of the pair
  dof2id = dof_tri_col[elementid]  # scalar: second dof of the pair

  for i in range(nblocks_perblock):
    conid = conid_start + i * dim_block  # scalar: contact handled this round

    if conid >= min(nacon_in[0], naconmax_in):
      return

    worldid = contact_worldid_in[conid]  # scalar: world of this contact
    if ctx_done_in[worldid]:
      continue

    condim = contact_dim_in[conid]  # scalar: contact dimensionality

    if condim == 1:
      continue

    # check contact status
    if contact_dist_in[conid] - contact_includemargin_in[conid] >= 0.0:
      continue

    efcid0 = contact_efc_address_in[conid, 0]
    if efcid0 < 0:
      continue
    if efc_state_in[worldid, efcid0] != ConstraintState.CONE:
      continue

    fri = contact_friction_in[conid]  # vec5: contact friction
    mu = fri[0] * contact_impratio_invsqrt_in[conid]  # scalar: effective friction

    mu2 = mu * mu  # scalar
    dm = math.safe_div(efc_D_in[worldid, efcid0], mu2 * (1.0 + mu2))  # scalar: cone curvature weight

    if dm == 0.0:
      continue

    n = ctx_Jaref_in[worldid, efcid0] * mu  # scalar: mu-scaled normal residual N
    z01 = mu * efc_J_in[worldid, efcid0, dof1id]  # scalar: mu-scaled normal-row J at dof1
    z02 = mu * efc_J_in[worldid, efcid0, dof2id]  # scalar: mu-scaled normal-row J at dof2
    tt = float(0.0)  # scalar: squared tangential norm ||u_t||^2
    projection1 = float(0.0)  # scalar: u_t . z_t at dof1 (normalized by inv_t below)
    projection2 = float(0.0)  # scalar: u_t . z_t at dof2
    tangent_dot = float(0.0)  # scalar: z_t(dof1) . z_t(dof2)
    for dim in range(1, condim):
      efcid = contact_efc_address_in[conid, dim]  # scalar: efc row of tangent dim
      if efcid >= 0:
        scale = fri[dim - 1]  # scalar: friction scale of this tangent row
        u = ctx_Jaref_in[worldid, efcid] * scale  # scalar: scaled tangential residual
        z1 = scale * efc_J_in[worldid, efcid, dof1id]  # scalar: scaled J at dof1
        z2 = scale * efc_J_in[worldid, efcid, dof2id]  # scalar: scaled J at dof2
        tt += u * u
        projection1 += u * z1
        projection2 += u * z2
        tangent_dot += z1 * z2

    t = wp.max(wp.sqrt(tt), MINVAL)  # scalar: tangential norm
    inv_t = math.safe_div(1.0, t)  # scalar
    mu_n_over_t = mu * n * inv_t  # scalar: mu*N/T factor of the cone Hessian
    # scalar: cone-curvature entry at (dof1, dof2)
    h = _elliptic_hessian_entry_from_projections(
      dm,
      mu,
      mu_n_over_t,
      mu2 - mu_n_over_t,
      z01,
      z02,
      projection1 * inv_t,
      projection2 * inv_t,
      tangent_dot,
    )

    ctx_h_out[worldid, dof1id, dof2id] += h


@cache_kernel
def _update_gradient_cholesky(tile_size: int, skip_noflip: bool = False):
  """Builds the single-tile Cholesky solve kernel (nv <= _BLOCK_CHOLESKY_DIM).

  Args:
    tile_size: scalar tile edge (= nv), compiled in.
    skip_noflip: bool; worlds with no state changes keep their previous search.

  Returns:
    A tiled kernel over (nworld,) factorizing ctx.h in registers and writing
    search = -H^-1 grad.
  """
  SKIP_NOFLIP = skip_noflip  # bool: compile-time fast-path switch

  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def kernel(
    # In:
    ctx_grad_in: wp.array2d[float],
    h_in: wp.array3d[float],
    state_changed_count_in: wp.array[int],
    ctx_done_in: wp.array[bool],
    # Out:
    ctx_search_out: wp.array2d[float],
    ctx_search_dot_out: wp.array[float],
    ctx_newton_decrement_out: wp.array[float],
  ):
    """Factorizes one world's H and forms the Newton search direction.

    Args:
      ctx_grad_in: (nworld, nv_pad) cost gradient.
      h_in: (nworld, nv_pad, nv_pad) Hessian (upper triangle valid).
      state_changed_count_in: (nworld,) state-change counts (fast-path gate).
      ctx_done_in: (nworld,) converged flags.
      ctx_search_out: (nworld, nv) search direction -H^-1 grad.
      ctx_search_dot_out: (nworld,) search . search.
      ctx_newton_decrement_out: (nworld,) grad . H^-1 grad.
    """
    worldid = wp.tid()  # scalar: world index
    TILE_SIZE = wp.static(tile_size)  # scalar: tile edge

    if ctx_done_in[worldid]:
      return

    # Fast path: skip the solve (see the blocked skip_unchanged variant).
    if wp.static(SKIP_NOFLIP):
      if state_changed_count_in[worldid] == 0:
        return

    mat_tile = wp.tile_load(h_in[worldid], shape=(TILE_SIZE, TILE_SIZE))  # tile: Hessian
    wp.tile_cholesky_inplace(mat_tile, fill_mode="upper")
    input_tile = wp.tile_load(ctx_grad_in[worldid], shape=TILE_SIZE)  # tile: gradient
    output_tile = wp.tile_cholesky_solve(mat_tile, input_tile, fill_mode="upper")  # tile: H^-1 grad
    # vec2: tile-reduced (search . search, grad . H^-1 grad)
    sums = wp.tile_reduce(wp.add, wp.tile_map(solve_search_sums, input_tile, output_tile))[0]
    ctx_search_dot_out[worldid] = sums[0]
    ctx_newton_decrement_out[worldid] = sums[1]
    wp.tile_store(ctx_search_out[worldid], wp.tile_map(wp.mul, output_tile, -1.0))

  return kernel


@cache_kernel
def _update_gradient_cholesky_blocked(tile_size: int, matrix_size: int, vector_size: int):
  """Builds the blocked Cholesky factorize+solve kernel (nv > _BLOCK_CHOLESKY_DIM).

  Args:
    tile_size: scalar Cholesky block edge, compiled in.
    matrix_size: scalar padded matrix size nv_pad, compiled in.
    vector_size: scalar true vector size nv, compiled in.

  Returns:
    A tiled kernel over (nworld,) factorizing ctx.h into ctx.hfactor and writing
    search = -H^-1 grad plus (search_dot, newton_decrement).
  """

  @wp.kernel(module="unique", enable_backward=False, grid_stride=False, module_options={"enable_mathdx_gemm": False})
  def kernel(
    # In:
    ctx_done_in: wp.array[bool],
    ctx_grad_in: wp.array3d[float],
    ctx_h_in: wp.array3d[float],
    ctx_hfactor: wp.array3d[float],
    # Out:
    ctx_search_out: wp.array3d[float],
    ctx_search_dot_out: wp.array[float],
    ctx_newton_decrement_out: wp.array[float],
  ):
    """Blocked factorize+solve of one world's Newton system.

    Args:
      ctx_done_in: (nworld,) converged flags.
      ctx_grad_in: (nworld, nv_pad, 1) gradient (reshaped column).
      ctx_h_in: (nworld, nv_pad, nv_pad) Hessian.
      ctx_hfactor: (nworld, nv_pad, nv_pad) Cholesky factor (written).
      ctx_search_out: (nworld, nv, 1) search direction (reshaped column).
      ctx_search_dot_out: (nworld,) search . search.
      ctx_newton_decrement_out: (nworld,) grad . H^-1 grad.
    """
    worldid = wp.tid()  # scalar: world index
    TILE_SIZE = wp.static(tile_size)  # scalar: Cholesky block edge

    if ctx_done_in[worldid]:
      return

    # We need matrix size both as a runtime input as well as a static input:
    # static input is needed to specify the tile sizes for the compiler
    # runtime input is needed for the loop bounds, otherwise warp will unroll
    # unconditionally leading to shared memory capacity issues.

    # vec2: (search . search, grad . H^-1 grad) of the solve
    sums = wp.static(create_blocked_cholesky_augmented_factorize_solve_newton_func(TILE_SIZE, matrix_size, vector_size))(
      ctx_h_in[worldid],
      ctx_grad_in[worldid],
      matrix_size,
      ctx_hfactor[worldid],
      ctx_search_out[worldid],
    )
    ctx_search_dot_out[worldid] = sums[0]
    ctx_newton_decrement_out[worldid] = sums[1]

  return kernel


@cache_kernel
def _cholesky_factorize_solve_blocked(tile_size: int, matrix_size: int):
  """Builds a plain blocked Cholesky A x = b factorize+solve (no solver state).

  Args:
    tile_size: scalar Cholesky block edge, compiled in.
    matrix_size: scalar padded matrix size, compiled in.

  Returns:
    A tiled kernel over (nworld,) writing the factor U_out and solution x_out.
  """

  @wp.kernel(module="unique", enable_backward=False, grid_stride=False, module_options={"enable_mathdx_gemm": False})
  def kernel(
    # In:
    A_in: wp.array3d[float],
    b_in: wp.array3d[float],
    # Out:
    U_out: wp.array3d[float],
    x_out: wp.array3d[float],
  ):
    """Factorizes one world's A and solves A x = b.

    Args:
      A_in: (nworld, n_pad, n_pad) symmetric positive-definite matrices.
      b_in: (nworld, n_pad, 1) right-hand sides.
      U_out: (nworld, n_pad, n_pad) upper Cholesky factors.
      x_out: (nworld, n_pad, 1) solutions.
    """
    worldid = wp.tid()  # scalar: world index
    TILE_SIZE = wp.static(tile_size)  # scalar: Cholesky block edge

    wp.static(create_blocked_cholesky_factorize_solve_func(TILE_SIZE, matrix_size))(
      A_in[worldid],
      b_in[worldid],
      matrix_size,
      U_out[worldid],
      x_out[worldid],
    )

  return kernel


@cache_kernel
def _update_gradient_cholesky_blocked_skip_unchanged(
  tile_size: int, matrix_size: int, vector_size: int, skip_noflip: bool = False
):
  """Blocked Cholesky that skips factorization when no constraints changed.

  Args:
    tile_size: scalar Cholesky block edge, compiled in.
    matrix_size: scalar padded matrix size nv_pad, compiled in.
    vector_size: scalar true vector size nv, compiled in.
    skip_noflip: bool; worlds with no state changes keep their previous search.

  Returns:
    A tiled kernel over (nworld,): quad flips refactorize into ctx.hfactor,
    otherwise the cached factor is reused for the solve.
  """
  SKIP_NOFLIP = skip_noflip  # bool: compile-time fast-path switch

  @wp.kernel(module="unique", enable_backward=False, grid_stride=False, module_options={"enable_mathdx_gemm": False})
  def kernel(
    # In:
    ctx_done_in: wp.array[bool],
    ctx_grad_in: wp.array3d[float],
    ctx_h_in: wp.array3d[float],
    quad_changed_count_in: wp.array[int],
    state_changed_count_in: wp.array[int],
    ctx_hfactor: wp.array3d[float],
    # Out:
    ctx_search_out: wp.array3d[float],
    ctx_search_dot_out: wp.array[float],
    ctx_newton_decrement_out: wp.array[float],
  ):
    """Solves one world's Newton system, reusing the cached factor when H is unchanged.

    Args:
      ctx_done_in: (nworld,) converged flags.
      ctx_grad_in: (nworld, nv_pad, 1) gradient (reshaped column).
      ctx_h_in: (nworld, nv_pad, nv_pad) Hessian.
      quad_changed_count_in: (nworld,) QUADRATIC-flip counts (refactorize gate).
      state_changed_count_in: (nworld,) state-change counts (fast-path gate).
      ctx_hfactor: (nworld, nv_pad, nv_pad) cached Cholesky factor.
      ctx_search_out: (nworld, nv, 1) search direction (reshaped column).
      ctx_search_dot_out: (nworld,) search . search.
      ctx_newton_decrement_out: (nworld,) grad . H^-1 grad.
    """
    worldid = wp.tid()  # scalar: world index
    TILE_SIZE = wp.static(tile_size)  # scalar: Cholesky block edge

    if ctx_done_in[worldid]:
      return

    # The linesearch is invariant to direction scale, so an unchanged ray can
    # keep its previous search direction.
    if wp.static(SKIP_NOFLIP):
      if state_changed_count_in[worldid] == 0:
        return

    if quad_changed_count_in[worldid] > 0:
      sums = wp.static(create_blocked_cholesky_augmented_factorize_solve_newton_func(TILE_SIZE, matrix_size, vector_size))(
        ctx_h_in[worldid],
        ctx_grad_in[worldid],
        matrix_size,
        ctx_hfactor[worldid],
        ctx_search_out[worldid],
      )
    else:
      sums = wp.static(create_blocked_cholesky_solve_newton_func(TILE_SIZE, matrix_size, vector_size))(
        ctx_hfactor[worldid],
        ctx_grad_in[worldid],
        matrix_size,
        ctx_search_out[worldid],
      )

    ctx_search_dot_out[worldid] = sums[0]
    ctx_newton_decrement_out[worldid] = sums[1]

  return kernel


@wp.kernel(grid_stride=True)
def _padding_h(nv: int, ctx_done_in: wp.array[bool], ctx_h_out: wp.array3d[float]):
  """Writes identity onto the padded diagonal rows [nv, nv_pad) of the Hessian.

  Args:
    nv: scalar number of real dofs.
    ctx_done_in: (nworld,) converged flags.
    ctx_h_out: (nworld, nv_pad, nv_pad) Hessian; padding keeps the Cholesky well posed.
  """
  worldid, elementid = wp.tid()  # scalars: world, padded-diagonal offset

  if ctx_done_in[worldid]:
    return

  dofid = nv + elementid  # scalar: padded diagonal index
  ctx_h_out[worldid, dofid, dofid] = 1.0


def _cholesky_factorize_solve(
  m: Model, d: Data, ctx: SolverContext, skip_unchanged: bool = False, skip_noflip: bool = False
):
  """Cholesky factorize ctx.h and form the Newton search direction.

  If skip_unchanged is True (blocked path only), worlds where no constraints
  changed reuse the cached factorization in hfactor instead of refactorizing.

  Args:
    m: model; nv selects the single-tile vs blocked path.
    d: data.
    ctx: solver context; reads ctx.grad/ctx.h, writes ctx.search (nworld, nv),
      ctx.search_dot (nworld,), ctx.newton_decrement (nworld,).
    skip_unchanged: reuse the cached hfactor when no quad flips occurred.
    skip_noflip: skip the solve entirely for worlds with no state changes.
  """
  if m.nv <= _BLOCK_CHOLESKY_DIM:
    wp.launch_tiled(
      _update_gradient_cholesky(m.nv, skip_noflip),
      dim=d.nworld,
      inputs=[ctx.grad, ctx.h, ctx.state_changed_count if skip_noflip else d.nefc, ctx.done],
      outputs=[ctx.search, ctx.search_dot, ctx.newton_decrement],
      block_dim=m.block_dim.update_gradient_cholesky,
    )
  else:
    wp.launch(
      _padding_h,
      dim=(d.nworld, m.nv_pad - m.nv),
      inputs=[m.nv, ctx.done],
      outputs=[ctx.h],
    )

    if skip_unchanged:
      wp.launch_tiled(
        _update_gradient_cholesky_blocked_skip_unchanged(TILE_SIZE_JTDAJ_DENSE, m.nv_pad, m.nv, skip_noflip),
        dim=d.nworld,
        inputs=[
          ctx.done,
          ctx.grad.reshape(shape=(d.nworld, ctx.grad.shape[1], 1)),
          ctx.h,
          ctx.quad_changed_count,
          ctx.state_changed_count if skip_noflip else ctx.quad_changed_count,
          ctx.hfactor,
        ],
        outputs=[
          ctx.search.reshape(shape=(d.nworld, m.nv, 1)),
          ctx.search_dot,
          ctx.newton_decrement,
        ],
        block_dim=m.block_dim.update_gradient_cholesky_blocked,
      )
    else:
      wp.launch_tiled(
        _update_gradient_cholesky_blocked(TILE_SIZE_JTDAJ_DENSE, m.nv_pad, m.nv),
        dim=d.nworld,
        inputs=[ctx.done, ctx.grad.reshape(shape=(d.nworld, ctx.grad.shape[1], 1)), ctx.h, ctx.hfactor],
        outputs=[
          ctx.search.reshape(shape=(d.nworld, m.nv, 1)),
          ctx.search_dot,
          ctx.newton_decrement,
        ],
        block_dim=m.block_dim.update_gradient_cholesky_blocked,
      )


# ---------------------------------------------------------------------------
# Constraint groups contain consecutive rows with identical sparse support. Each thread group
# accumulates their upper-triangular Hessian block, including elliptic cone curvature.
# ---------------------------------------------------------------------------
_JTDAJ_THREADS_PER_GROUP = 32
_JTDAJ_OVERSUBSCRIBE_WAVES = 6


@cache_kernel
def _JTDACJ_sparse(compact: bool, cone_type: ConeType, max_condim: int):
  """Builds the sparse-J Hessian accumulator H += J.T @ diag(D*active) @ J + cone terms.

  Constraint groups are consecutive rows with identical sparse support; each thread
  group accumulates one group's upper-triangular block (see the section comment above).

  Args:
    compact: bool; map Jacobian columns through dof_cdof (compacted solve).
    cone_type: friction cone, compiled in; ELLIPTIC adds the middle-zone curvature.
    max_condim: scalar largest contact dim, specializing the cone helpers (3, 4 or 6).

  Returns:
    A kernel over (nworld, groups_per_world, threads_per_group) updating ctx.h's
    upper triangle in place (atomic adds).
  """
  COMPACT = compact  # bool: compile-time compaction switch
  ELLIPTIC = cone_type == ConeType.ELLIPTIC  # bool: compile-time cone switch
  MAX_CONDIM = max_condim  # scalar: compile-time condim bound

  def make_curvature_terms(condim: int):
    """Returns a wp.func computing the condim-specialized cone curvature terms."""

    @wp.func
    def func(
      # In:
      impratio_invsqrt_c: float,
      # Data in:
      efc_D_in: wp.array2d[float],
      # In:
      fri: vec5,
      ctx_Jaref_in: wp.array2d[float],
      worldid: int,
      efcid0: int,
      block_rows: int,
    ) -> vec16:
      """Packs the shared cone-curvature quantities of one contact into a vec16
      (layout in the comment below); zero when the middle-zone weight dm vanishes."""
      mu = fri[0] * impratio_invsqrt_c  # scalar: effective friction
      mu2 = mu * mu  # scalar
      dm = math.safe_div(efc_D_in[worldid, efcid0], mu2 * (1.0 + mu2))  # scalar: curvature weight
      if dm == 0.0:
        return vec16()

      n = ctx_Jaref_in[worldid, efcid0] * mu  # scalar: mu-scaled normal residual N
      terms = vec16()  # vec16: packed output (see layout comment below)
      terms[6] = mu
      tt = float(0.0)  # scalar: squared tangential norm ||u_t||^2
      for dim in range(1, wp.static(condim)):
        if dim < block_rows:
          efcid = efcid0 + dim  # scalar: efc row of tangent dim (consecutive in a group)
          scale = fri[dim - 1]  # scalar: friction scale of this tangent row
          u = ctx_Jaref_in[worldid, efcid] * scale  # scalar: scaled tangential residual
          terms[dim] = u
          terms[6 + dim] = scale
          tt += u * u

      t = wp.max(wp.sqrt(tt), MINVAL)  # scalar: tangential norm
      inv_t = math.safe_div(1.0, t)  # scalar
      for dim in range(1, wp.static(condim)):
        terms[dim] = terms[dim] * inv_t

      mu_n_over_t = mu * n * inv_t  # scalar: mu*N/T factor of the cone Hessian
      tangent_diag = mu2 - mu_n_over_t  # scalar: diagonal tangential term

      # Layout: normalized tangent u/t[1:6], scales[6:12], dm, mu, mu*n/t, tangent diagonal.
      terms[12] = dm
      terms[13] = mu
      terms[14] = mu_n_over_t
      terms[15] = tangent_diag
      return terms

    return func

  def make_hessian_entry(condim: int):
    """Returns a wp.func computing one (pos1, pos2) cone-curvature Hessian entry."""

    @wp.func
    def func(
      # Data in:
      efc_J_in: wp.array3d[float],
      # In:
      terms: Any,
      rowadr: vec6i,
      worldid: int,
      pos1: int,
      pos2: int,
    ) -> float:
      """Contracts the packed curvature terms with the group's J entries at
      (pos1, pos2); see _elliptic_hessian_entry_from_projections for the math."""
      z01 = terms[6] * efc_J_in[worldid, 0, rowadr[0] + pos1]  # scalar: mu-scaled normal J at pos1
      z02 = terms[6] * efc_J_in[worldid, 0, rowadr[0] + pos2]  # scalar: mu-scaled normal J at pos2
      projection1 = float(0.0)  # scalar: (u_t/t) . z_t at pos1
      projection2 = float(0.0)  # scalar: (u_t/t) . z_t at pos2
      tangent_dot = float(0.0)  # scalar: z_t(pos1) . z_t(pos2)
      for dim in range(1, wp.static(condim)):
        z1 = terms[6 + dim] * efc_J_in[worldid, 0, rowadr[dim] + pos1]  # scalar: scaled J at pos1
        z2 = terms[6 + dim] * efc_J_in[worldid, 0, rowadr[dim] + pos2]  # scalar: scaled J at pos2
        projection1 += terms[dim] * z1
        projection2 += terms[dim] * z2
        tangent_dot += z1 * z2

      return _elliptic_hessian_entry_from_projections(
        terms[12],
        terms[13],
        terms[14],
        terms[15],
        z01,
        z02,
        projection1,
        projection2,
        tangent_dot,
      )

    return func

  curvature_terms3 = make_curvature_terms(3)
  curvature_terms4 = make_curvature_terms(4)
  curvature_terms6 = make_curvature_terms(6)
  hessian_entry3 = make_hessian_entry(3)
  hessian_entry4 = make_hessian_entry(4)
  hessian_entry6 = make_hessian_entry(6)

  @wp.func
  def curvature_terms(
    # In:
    impratio_invsqrt_c: float,
    # Data in:
    efc_D_in: wp.array2d[float],
    # In:
    fri: vec5,
    ctx_Jaref_in: wp.array2d[float],
    worldid: int,
    efcid0: int,
    condim: int,
    block_rows: int,
  ) -> vec16:
    """Dispatches to the condim-specialized curvature_terms implementation."""
    if wp.static(MAX_CONDIM == 3):
      return curvature_terms3(impratio_invsqrt_c, efc_D_in, fri, ctx_Jaref_in, worldid, efcid0, block_rows)

    if condim == 3:
      return curvature_terms3(impratio_invsqrt_c, efc_D_in, fri, ctx_Jaref_in, worldid, efcid0, block_rows)
    if wp.static(MAX_CONDIM == 4):
      return curvature_terms4(impratio_invsqrt_c, efc_D_in, fri, ctx_Jaref_in, worldid, efcid0, block_rows)
    if condim == 4:
      return curvature_terms4(impratio_invsqrt_c, efc_D_in, fri, ctx_Jaref_in, worldid, efcid0, block_rows)
    return curvature_terms6(impratio_invsqrt_c, efc_D_in, fri, ctx_Jaref_in, worldid, efcid0, block_rows)

  @wp.func
  def hessian_entry(
    # Data in:
    efc_J_in: wp.array3d[float],
    # In:
    terms: Any,
    rowadr: vec6i,
    worldid: int,
    pos1: int,
    pos2: int,
    condim: int,
  ) -> float:
    """Dispatches to the condim-specialized hessian_entry implementation."""
    if wp.static(MAX_CONDIM == 3):
      return hessian_entry3(efc_J_in, terms, rowadr, worldid, pos1, pos2)

    if condim == 3:
      return hessian_entry3(efc_J_in, terms, rowadr, worldid, pos1, pos2)
    if wp.static(MAX_CONDIM == 4):
      return hessian_entry4(efc_J_in, terms, rowadr, worldid, pos1, pos2)
    if condim == 4:
      return hessian_entry4(efc_J_in, terms, rowadr, worldid, pos1, pos2)
    return hessian_entry6(efc_J_in, terms, rowadr, worldid, pos1, pos2)

  @wp.kernel(module="unique", enable_backward=False, grid_stride=True)
  def kernel(
    # Model:
    opt_impratio_invsqrt: wp.array[float],
    contact_impratio_invsqrt_in: wp.array[float],
    # Data in:
    contact_friction_in: wp.array[vec5],
    contact_dim_in: wp.array[int],
    efc_id_in: wp.array2d[int],
    efc_jtdaj_adr_in: wp.array2d[int],
    efc_jtdaj_nrow_in: wp.array2d[int],
    efc_jtdaj_nblock_in: wp.array[int],
    efc_J_rownnz_in: wp.array2d[int],
    efc_J_rowadr_in: wp.array2d[int],
    efc_J_colind_in: wp.array3d[int],
    efc_J_in: wp.array3d[float],
    efc_D_in: wp.array2d[float],
    efc_state_in: wp.array2d[int],
    dof_cdof_in: wp.array2d[int],
    # In:
    ctx_Jaref_in: wp.array2d[float],
    ctx_done_in: wp.array[bool],
    groups_per_world: int,
    # Out:
    h_out: wp.array3d[float],
  ):
    """Accumulates one constraint group's upper-triangular Hessian block.

    Args:
      opt_impratio_invsqrt: (1 or nworld,) world 1/sqrt(impratio).
      contact_impratio_invsqrt_in: (naconmax,) per-contact cone warp.
      contact_friction_in: (naconmax,) vec5 contact friction.
      contact_dim_in: (naconmax,) contact dimensionalities.
      efc_id_in: (nworld, njmax) owning contact id per row.
      efc_jtdaj_adr_in: (nworld, njmax) first efc row of each group.
      efc_jtdaj_nrow_in: (nworld, njmax) rows per group.
      efc_jtdaj_nblock_in: (nworld,) number of groups.
      efc_J_*: CSR constraint Jacobian.
      efc_D_in: (nworld, njmax_pad) constraint weights.
      efc_state_in: (nworld, njmax_pad) ConstraintState per row.
      dof_cdof_in: (nworld, nv) global dof -> compacted dof map (compact only).
      ctx_Jaref_in: (nworld, njmax) J @ qacc - aref.
      ctx_done_in: (nworld,) converged flags.
      groups_per_world: scalar group slots strided per world.
      h_out: (nworld, nv_pad, nv_pad) Hessian, upper triangle accumulated (atomic).
    """
    worldid, slot, lane = wp.tid()  # scalars: world, group slot, lane within the group
    if wp.static(ELLIPTIC):
      lanes = wp.block_dim()  # scalar: lanes per group
    else:
      lanes = wp.static(_JTDAJ_THREADS_PER_GROUP)
    if ctx_done_in[worldid]:
      return
    count = efc_jtdaj_nblock_in[worldid]  # scalar: constraint groups of this world
    for groupid in range(slot, count, groups_per_world):
      head_row = efc_jtdaj_adr_in[worldid, groupid]  # scalar: first efc row of the group
      block_rows = efc_jtdaj_nrow_in[worldid, groupid]  # scalar: rows in the group
      head_adr = efc_J_rowadr_in[worldid, head_row]  # scalar: head row's sparse start
      support = efc_J_rownnz_in[worldid, head_row]  # scalar: shared sparse support size
      n_entries = support * (support + 1) // 2  # scalar: upper-triangle pairs of the block

      is_cone = False
      if wp.static(ELLIPTIC):
        is_cone = efc_state_in[worldid, head_row] == ConstraintState.CONE  # bool
        condim = int(0)  # scalar: contact dimensionality (cone groups only)
        # Clipped cone rows retain the safe head address and a zero scale.
        cone_rowadr = vec6i(head_adr, head_adr, head_adr, head_adr, head_adr, head_adr)  # vec6i
        if is_cone:
          conid = efc_id_in[worldid, head_row]  # scalar: owning contact
          if wp.static(MAX_CONDIM == 3):
            condim = int(3)
          else:
            condim = contact_dim_in[conid]
          for dim in range(1, wp.static(MAX_CONDIM)):
            if dim < block_rows:
              cone_rowadr[dim] = head_adr + dim * support
          local_terms = vec16()  # vec16: lane-0 cone curvature terms
          if lane == 0:
            local_terms = curvature_terms(
              contact_impratio_invsqrt_in[conid],
              efc_D_in,
              contact_friction_in[conid],
              ctx_Jaref_in,
              worldid,
              head_row,
              condim,
              block_rows,
            )
          # tile (16,): curvature terms broadcast to the group through shared memory
          cone_terms = wp.tile_zeros(shape=(16,), dtype=float, storage="shared")
          for term_id in range(1, 16):
            wp.tile_scatter_masked(cone_terms, term_id, local_terms[term_id], lane == 0)

      for entry in range(lane, n_entries, lanes):
        block_col = int((wp.sqrt(float(8 * entry + 1)) - 1.0) * 0.5)  # scalar: triangular decode
        block_row = entry - block_col * (block_col + 1) // 2  # scalar: pair member, row <= col
        dof_row = efc_J_colind_in[worldid, 0, head_adr + block_row]  # scalar: dof of the row entry
        dof_col = efc_J_colind_in[worldid, 0, head_adr + block_col]  # scalar: dof of the col entry
        hval = float(0.0)  # scalar: Hessian contribution of this pair
        if wp.static(ELLIPTIC):
          if is_cone:
            hval = hessian_entry(
              efc_J_in,
              cone_terms,
              cone_rowadr,
              worldid,
              block_row,
              block_col,
              condim,
            )
        if not is_cone:
          for member in range(block_rows):
            member_row = head_row + member  # scalar: efc row of this group member
            if efc_state_in[worldid, member_row] == ConstraintState.QUADRATIC.value:
              member_adr = efc_J_rowadr_in[worldid, member_row]  # scalar: member's sparse start
              j_row = efc_J_in[worldid, 0, member_adr + block_row]  # scalar: J entry at the row dof
              j_col = efc_J_in[worldid, 0, member_adr + block_col]  # scalar: J entry at the col dof
              hval += j_row * efc_D_in[worldid, member_row] * j_col
        if hval != 0.0:
          if wp.static(COMPACT):
            dof_row = dof_cdof_in[worldid, dof_row]
            dof_col = dof_cdof_in[worldid, dof_col]
            if dof_row < 0 or dof_col < 0:
              continue
          wp.atomic_add(h_out[worldid, wp.min(dof_row, dof_col)], wp.max(dof_row, dof_col), hval)

  return kernel


def _jtdaj_groups_per_world(nworld: int, njmax: int) -> int:
  """Number of constraint-group slots each world strides over in _JTDACJ_sparse."""
  # njmax is capacity and often mostly empty, so cap slots at a few resident waves.
  block_size, min_grid_size = wp.get_suggested_block_size(_JTDACJ_sparse(False, ConeType.PYRAMIDAL, 3))
  device_warps = max(1, block_size * min_grid_size // _JTDAJ_THREADS_PER_GROUP)  # scalar: resident groups
  return max(1, min(njmax, _JTDAJ_OVERSUBSCRIBE_WAVES * device_warps // nworld))


# ---------------------------------------------------------------------------
# opt.shared_hessian: the sparse path's Hessian summed per world in shared memory. Every Newton
# iteration under the elliptic cone rebuilds H in global memory: _update_gradient_init_h_sparse
# re-densifies M, one thread per matrix entry of every world, and _JTDACJ_sparse adds J^T D J and
# the cone's curvature by global atomics. Here one block per world sums the same H in shared
# memory -- M's entries, then a warp per constraint group, its lanes splitting the group's entries
# so that a warp never collides with itself -- and writes ctx.h's upper triangle once, four
# columns a store. The Cholesky that follows is the same on both paths.
# ---------------------------------------------------------------------------
# A block per world, sixteen warps. Chosen on an RTX 4080 (Ada: 100 KB of shared memory and 1536 threads a
# multiprocessor), where at 144 padded dofs a block's triangle (42 KB) leaves room for two blocks a
# multiprocessor, and with eight warps a block a world of many contacts stepped slower than with MJWarp's own
# assembly. A GPU with more shared memory a multiprocessor holds more blocks and may prefer another size; 1024
# threads ask for more registers than a block may hold.
_SHARED_HESSIAN_THREADS = 512  # scalar: threads a block
_SHARED_HESSIAN_BYTES = 48 * 1024  # scalar: the static shared memory a block may hold


def shared_hessian_refusal(m: Model) -> str:
  """Why opt.shared_hessian cannot build `m`'s Hessian; empty if it can."""
  if m.opt.solver != SolverType.NEWTON:
    return "it builds the Newton solver's Hessian: it needs solver Newton"
  if not m.is_sparse:
    return "it builds the sparse path's Hessian: it needs the sparse Jacobian (nv > 32, or jacobian sparse)"
  if m.opt.cone == ConeType.ELLIPTIC and m.nmaxcondim > 3:
    return "it takes the elliptic cone's curvature for condim <= 3"
  if m.opt.enableflags & EnableBit.SLEEP:
    return "it does not take sleeping's compacted solve"
  nbytes = 4 * (m.nv_pad * (m.nv_pad + 1) // 2)  # scalar: H's packed upper triangle
  if nbytes > _SHARED_HESSIAN_BYTES:
    return (
      f"H's upper triangle at nv_pad {m.nv_pad}, {nbytes} bytes, exceeds the {_SHARED_HESSIAN_BYTES} bytes "
      "of shared memory a block holds"
    )
  return ""


@cache_kernel
def _shared_hessian(nv: int, nv_pad: int, cone_type: ConeType):
  """Builds the kernel summing each world's Hessian in shared memory.

  Args:
    nv: scalar dofs.
    nv_pad: scalar padded dofs (a multiple of 4).
    cone_type: friction cone, compiled in; ELLIPTIC adds the middle-zone curvature.

  Returns:
    A kernel over (nworld, _SHARED_HESSIAN_THREADS), a block per world, writing ctx.h's upper
    triangle, as vec4 rows: H = M + J.T @ diag(D*active) @ J plus the elliptic cone's
    curvature within nv, zero outside.
  """
  NV = nv
  NVP = nv_pad
  NPK = nv_pad * (nv_pad + 1) // 2  # scalar: H's packed upper triangle, column by column
  THREADS = _SHARED_HESSIAN_THREADS
  WARPS = THREADS // 32
  ELLIPTIC = cone_type == ConeType.ELLIPTIC  # bool: compile-time cone switch

  # The packed triangle: mode 0 reads, 1 stores, 2 adds atomically; returns the element.
  @wp.func_native(
    snippet=f"__shared__ float buf[{NPK}];\nif (mode == 1) buf[i] = v;\n"
    "else if (mode == 2) atomicAdd(&buf[i], v);\nreturn buf[i];"
  )
  def packed(i: int, v: float, mode: int) -> float: ...

  @wp.func_native(snippet="WP_TILE_SYNC();")
  def _syncthreads():
    pass

  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Model:
    M_colind: wp.array[int],
    M_hinit_i: wp.array[int],
    # Data in:
    M_in: wp.array2d[float],
    contact_friction_in: wp.array[vec5],
    contact_impratio_invsqrt_in: wp.array[float],
    efc_id_in: wp.array2d[int],
    efc_jtdaj_adr_in: wp.array2d[int],
    efc_jtdaj_nrow_in: wp.array2d[int],
    efc_jtdaj_nblock_in: wp.array[int],
    efc_J_rownnz_in: wp.array2d[int],
    efc_J_rowadr_in: wp.array2d[int],
    efc_J_colind_in: wp.array3d[int],
    efc_J_in: wp.array3d[float],
    efc_D_in: wp.array2d[float],
    efc_state_in: wp.array2d[int],
    # In:
    ctx_Jaref_in: wp.array2d[float],
    ctx_done_in: wp.array[bool],
    # Out:
    ctx_h_out: wp.array3d[wp.vec4],
  ):
    """Sums one world's Hessian in shared memory and writes its upper triangle.

    Args:
      M_colind, M_hinit_i: (nC,) each CSR entry's column and row, M's lower triangle.
      M_in: (nworld, nC) M's values.
      contact_friction_in: (naconmax,) vec5 contact friction.
      contact_impratio_invsqrt_in: (naconmax,) per-contact cone warp.
      efc_id_in: (nworld, njmax) owning contact id per row.
      efc_jtdaj_adr_in, efc_jtdaj_nrow_in, efc_jtdaj_nblock_in: the constraint groups, as
        _JTDACJ_sparse takes them.
      efc_J_*: CSR constraint Jacobian.
      efc_D_in: (nworld, njmax_pad) constraint weights.
      efc_state_in: (nworld, njmax_pad) ConstraintState per row.
      ctx_Jaref_in: (nworld, njmax) J @ qacc - aref.
      ctx_done_in: (nworld,) converged flags.
      ctx_h_out: (nworld, nv_pad, nv_pad / 4) the Hessian's rows, four columns an element.
    """
    worldid, tid = wp.tid()  # scalars: world (= block), thread within it
    if ctx_done_in[worldid]:
      return
    lane = tid % 32  # scalar: lane within its warp
    warp = tid // 32  # scalar: warp within the block
    for e in range(tid, NPK, THREADS):
      packed(e, 0.0, 1)
    _syncthreads()

    # M: a thread an entry of its lower triangle (column <= row); a warp a row would leave most
    # lanes idle, a row holding as many entries as the dof has ancestors
    for e in range(tid, M_colind.shape[0], THREADS):
      i = M_hinit_i[e]  # scalar: the entry's row
      packed(i * (i + 1) // 2 + M_colind[e], M_in[worldid, e], 2)

    # J.T @ diag(D*active) @ J and the cone's curvature: a warp a constraint group
    for groupid in range(warp, efc_jtdaj_nblock_in[worldid], WARPS):
      head_row = efc_jtdaj_adr_in[worldid, groupid]  # scalar: first efc row of the group
      block_rows = efc_jtdaj_nrow_in[worldid, groupid]  # scalar: rows in the group
      head_adr = efc_J_rowadr_in[worldid, head_row]  # scalar: head row's sparse start
      support = efc_J_rownnz_in[worldid, head_row]  # scalar: the group's shared sparse support

      # the middle zone's curvature, condim 3, as _JTDACJ_sparse's curvature_terms and hessian_entry
      is_cone = False
      dm = float(0.0)
      mu = float(0.0)
      mu_n_over_t = float(0.0)
      tangent_diag = float(0.0)
      ut1 = float(0.0)  # scalar: normalized tangential residual, first tangent
      ut2 = float(0.0)  # scalar: and second
      scale1 = float(0.0)  # scalar: friction scale of the first tangent row (0 when clipped)
      scale2 = float(0.0)
      rowadr1 = head_adr  # scalar: the tangent rows' sparse starts (the head's when clipped)
      rowadr2 = head_adr
      if wp.static(ELLIPTIC):
        is_cone = efc_state_in[worldid, head_row] == ConstraintState.CONE
        if is_cone:
          conid = efc_id_in[worldid, head_row]  # scalar: owning contact
          fri = contact_friction_in[conid]  # vec5: friction scales
          mu = fri[0] * contact_impratio_invsqrt_in[conid]
          mu2 = mu * mu
          dm = math.safe_div(efc_D_in[worldid, head_row], mu2 * (1.0 + mu2))
          n = ctx_Jaref_in[worldid, head_row] * mu  # scalar: mu-scaled normal residual
          if block_rows > 1:
            scale1 = fri[0]
            ut1 = ctx_Jaref_in[worldid, head_row + 1] * scale1
            rowadr1 = head_adr + support
          if block_rows > 2:
            scale2 = fri[1]
            ut2 = ctx_Jaref_in[worldid, head_row + 2] * scale2
            rowadr2 = head_adr + 2 * support
          t = wp.max(wp.sqrt(ut1 * ut1 + ut2 * ut2), MINVAL)  # scalar: tangential norm
          inv_t = math.safe_div(1.0, t)
          ut1 = ut1 * inv_t
          ut2 = ut2 * inv_t
          mu_n_over_t = mu * n * inv_t
          tangent_diag = mu2 - mu_n_over_t

      for entry in range(lane, support * (support + 1) // 2, 32):
        block_col = int((wp.sqrt(float(8 * entry + 1)) - 1.0) * 0.5)  # scalar: triangular decode
        block_row = entry - block_col * (block_col + 1) // 2  # scalar: pair member, row <= col
        if block_row > block_col:  # the float decode one off at a perfect square
          block_col += 1
          block_row = entry - block_col * (block_col + 1) // 2
        hval = float(0.0)  # scalar: Hessian contribution of this pair
        if is_cone:
          if dm != 0.0:
            z01 = mu * efc_J_in[worldid, 0, head_adr + block_row]
            z02 = mu * efc_J_in[worldid, 0, head_adr + block_col]
            z11 = scale1 * efc_J_in[worldid, 0, rowadr1 + block_row]
            z12 = scale1 * efc_J_in[worldid, 0, rowadr1 + block_col]
            z21 = scale2 * efc_J_in[worldid, 0, rowadr2 + block_row]
            z22 = scale2 * efc_J_in[worldid, 0, rowadr2 + block_col]
            hval = _elliptic_hessian_entry_from_projections(
              dm,
              mu,
              mu_n_over_t,
              tangent_diag,
              z01,
              z02,
              ut1 * z11 + ut2 * z21,
              ut1 * z12 + ut2 * z22,
              z11 * z12 + z21 * z22,
            )
        else:
          for member in range(block_rows):
            member_row = head_row + member  # scalar: efc row of this group member
            if efc_state_in[worldid, member_row] == ConstraintState.QUADRATIC.value:
              member_adr = efc_J_rowadr_in[worldid, member_row]  # scalar: member's sparse start
              j_row = efc_J_in[worldid, 0, member_adr + block_row]  # scalar: J entry at the row dof
              j_col = efc_J_in[worldid, 0, member_adr + block_col]  # scalar: J entry at the col dof
              hval += j_row * efc_D_in[worldid, member_row] * j_col
        if hval != 0.0:
          dof_row = efc_J_colind_in[worldid, 0, head_adr + block_row]  # scalar: dof of the row entry
          dof_col = efc_J_colind_in[worldid, 0, head_adr + block_col]  # scalar: dof of the col entry
          hi = wp.max(dof_row, dof_col)
          packed(hi * (hi + 1) // 2 + wp.min(dof_row, dof_col), hval, 2)
    _syncthreads()

    # ctx.h's upper triangle, four columns a store: a row's elements from the one holding its diagonal
    for e in range(tid, NVP * (NVP // 4), THREADS):
      i = e // (NVP // 4)  # scalar: row
      k = e - i * (NVP // 4)  # scalar: its element of four columns
      j0 = 4 * k  # scalar: the element's first column
      if j0 + 3 >= i:
        v4 = wp.vec4(0.0)
        if i < NV:
          for c in range(4):
            j = j0 + c
            if j < NV:
              hi = wp.max(i, j)
              v4[c] = packed(hi * (hi + 1) // 2 + wp.min(i, j), 0.0, 0)
        ctx_h_out[worldid, i, k] = v4

  return kernel


def _shared_hessian_build(m: Model, d: Data, ctx: SolverContext):
  """ctx.h's upper triangle from the shared-memory kernel (opt.shared_hessian)."""
  nworld, rows, cols = ctx.h.shape  # scalars: ctx.h is (nworld, nv_pad, nv_pad), rows 16-byte aligned
  h4 = wp.array(ptr=ctx.h.ptr, dtype=wp.vec4, shape=(nworld, rows, cols // 4), device=ctx.h.device)  # zero-copy view
  wp.launch(
    _shared_hessian(m.nv, m.nv_pad, m.opt.cone),
    dim=(d.nworld, _SHARED_HESSIAN_THREADS),
    inputs=[
      m.M_colind,
      m.M_hinit_i,
      d.M,
      d.contact.friction,
      d.contact_impratio_invsqrt,
      d.efc.id,
      d.efc.jtdaj_adr,
      d.efc.jtdaj_nrow,
      d.efc.jtdaj_nblock,
      d.efc.J_rownnz,
      d.efc.J_rowadr,
      d.efc.J_colind,
      d.efc.J,
      d.efc.D,
      d.efc.state,
      ctx.Jaref,
      ctx.done,
    ],
    outputs=[h4],
    block_dim=_SHARED_HESSIAN_THREADS,
  )


def _newton_hessian_solve(m: Model, d: Data, ctx: SolverContext, compact: bool = False):
  """The Newton search direction from the dense Hessian: H = M + J.T @ diag(D*active) @ J plus the
  elliptic cone curvature, built into ctx.h and factored (_cholesky_factorize_solve)."""
  # h = M + (efc_J.T * efc_D * active) @ efc_J
  sc = _sparse_compact(ctx)  # bool: compact solve over a sparse full model
  if m.opt.shared_hessian:
    reason = shared_hessian_refusal(m) or ("it does not take the compacted solve" if sc or compact else "")
    if reason:
      raise ValueError(f"opt.shared_hessian: {reason}")
    _shared_hessian_build(m, d, ctx)
  elif m.is_sparse or sc:
    mj = ctx.compact_m_full if sc else m  # model holding the sparse M structures
    dj = ctx.compact_d_full if sc else d  # data holding the sparse M/J values
    wp.launch(
      _update_gradient_init_h_sparse(sc),
      dim=(d.nworld, m.nv_pad, m.nv_pad),
      inputs=[mj.nv, mj.M_elemid, dj.M, dj.cdof_dof, ctx.done],
      outputs=[ctx.h],
    )

    groups_per_world = _jtdaj_groups_per_world(d.nworld, d.njmax)  # scalar: slots per world
    max_condim = 3  # scalar: condim bound compiled into the cone helpers
    if m.opt.cone == ConeType.ELLIPTIC and m.nmaxcondim > 3:
      max_condim = int(m.nmaxcondim)
    jtdaj_kernel = _JTDACJ_sparse(sc, m.opt.cone, max_condim)
    jtdaj_inputs = [
      m.opt.impratio_invsqrt,
    d.contact_impratio_invsqrt,
      d.contact.friction,
      d.contact.dim,
      d.efc.id,
      dj.efc.jtdaj_adr,
      dj.efc.jtdaj_nrow,
      dj.efc.jtdaj_nblock,
      dj.efc.J_rownnz,
      dj.efc.J_rowadr,
      dj.efc.J_colind,
      dj.efc.J,
      d.efc.D,
      d.efc.state,
      dj.dof_cdof,
      ctx.Jaref,
      ctx.done,
      groups_per_world,
    ]
    elliptic = m.opt.cone == ConeType.ELLIPTIC  # bool
    threads_per_group = 1 if elliptic and wp.get_device().is_cpu else _JTDAJ_THREADS_PER_GROUP  # scalar
    block_dim = threads_per_group if elliptic else mj.block_dim.update_gradient_JTDAJ_sparse  # scalar
    wp.launch(
      jtdaj_kernel,
      dim=(d.nworld, groups_per_world, threads_per_group),
      inputs=jtdaj_inputs,
      outputs=[ctx.h],
      block_dim=block_dim,
    )
  else:
    if compact:
      # compact path: d.M is the dense 3D compact inertia block (nworld, nv_pad, nv_pad)
      wp.launch_tiled(
        _update_gradient_JTDAJ_dense_tiled_compact(m.nv_pad, TILE_SIZE_JTDAJ_DENSE, d.njmax),
        dim=d.nworld,
        inputs=[
          d.nefc,
          d.M,
          d.efc.J,
          d.efc.D,
          d.efc.state,
          ctx.done,
        ],
        outputs=[ctx.h],
        block_dim=m.block_dim.update_gradient_JTDAJ_dense,
      )
    else:
      wp.launch_tiled(
        _update_gradient_JTDAJ_dense_tiled(m.nv_pad, TILE_SIZE_JTDAJ_DENSE, d.njmax, m.M_colind.shape[0]),
        dim=d.nworld,
        inputs=[
          m.M_colind,
          m.M_hinit_i,
          d.nefc,
          d.M,
          d.efc.J,
          d.efc.D,
          d.efc.state,
          ctx.done,
        ],
        outputs=[ctx.h],
        block_dim=m.block_dim.update_gradient_JTDAJ_dense,
      )

  if m.opt.cone == ConeType.ELLIPTIC and not (m.is_sparse or sc):
    # Optimization: launching update_gradient_JTCJ with limited number of blocks on a GPU.
    # Profiling suggests that only a fraction of blocks out of the original
    # d.njmax blocks do the actual work. It aims to minimize #CTAs with no
    # effective work. It launches with #blocks that's proportional to the number
    # of SMs on the GPU. We can now query the SM count:
    # https://github.com/NVIDIA/warp/commit/f3814e7e5459e5fd13032cf0fddb3daddd510f30

    if wp.get_device().is_cuda:
      sm_count = wp.get_device().sm_count

      # Here we assume one block has 256 threads. We use a factor of 6, which
      # can be changed in the future to fine-tune the perf. The optimal factor will
      # depend on the kernel's occupancy, which determines how many blocks can
      # simultaneously run on the SM. TODO: This factor can be tuned further.
      dim_block = ceil((sm_count * 6 * 256) / m.dof_tri_row.size)  # scalar: launch blocks
    else:
      # fall back for CPU
      dim_block = d.naconmax

    nblocks_perblock = int((d.naconmax + dim_block - 1) / dim_block)  # scalar: contacts per block

    wp.launch(
      _update_gradient_JTCJ_dense,
      dim=(dim_block, m.dof_tri_row.size),
      inputs=[
        m.opt.impratio_invsqrt,
    d.contact_impratio_invsqrt,
        m.dof_tri_row,
        m.dof_tri_col,
        d.contact.dist,
        d.contact.includemargin,
        d.contact.friction,
        d.contact.dim,
        d.contact.efc_address,
        d.contact.worldid,
        d.efc.J,
        d.efc.D,
        d.efc.state,
        d.naconmax,
        d.nacon,
        ctx.Jaref,
        ctx.done,
        nblocks_perblock,
        dim_block,
      ],
      outputs=[ctx.h],
    )

  _cholesky_factorize_solve(m, d, ctx)


def _update_gradient(m: Model, d: Data, ctx: SolverContext, compact: bool = False):
  """Rebuilds the gradient and (Newton) the Hessian plus search direction.

  CG: grad and the preconditioned Mgrad. Newton: grad, H = M + J.T @ diag(D*active) @ J
  (plus elliptic cone curvature), then a Cholesky solve into ctx.search.

  Args:
    m: model; solver type, cone, and sparsity select the kernels.
    d: data.
    ctx: solver context; writes ctx.grad (nworld, nv_pad), ctx.h
      (nworld, nv_pad, nv_pad), ctx.search (nworld, nv), and the per-world scalars.
    compact: dense compacted-DOF path (reads the gathered cM/cJ).
  """
  # grad = Ma - qfrc_smooth - qfrc_constraint
  if m.opt.solver == SolverType.CG:
    wp.launch_tiled(
      _update_gradient_grad_tiled,
      dim=d.nworld,
      inputs=[m.nv, d.qfrc_smooth, d.qfrc_constraint, d.efc.Ma, ctx.done],
      outputs=[ctx.grad, ctx.grad_dot],
      block_dim=m.block_dim.update_gradient_grad,
    )
  else:
    wp.launch(
      _update_gradient_zero_grad_dot(False),
      dim=d.nworld,
      inputs=[d.nefc, ctx.alpha, ctx.done],
      outputs=[ctx.grad_dot, ctx.newton_decrement, ctx.grad_scale, ctx.search_unchanged],
    )
    wp.launch(
      _update_gradient_grad(False),
      dim=(d.nworld, m.nv),
      inputs=[d.qfrc_smooth, d.qfrc_constraint, d.efc.Ma, d.nefc, ctx.done],
      outputs=[ctx.grad, ctx.grad_dot],
    )

  if m.opt.solver == SolverType.CG:
    solve_m(m, d, ctx.Mgrad, ctx.grad)
  elif m.opt.solver == SolverType.NEWTON and ctx.newton_solve is not None:
    # the caller's solve of H search = -grad (solve's newton_solve), in place of forming H
    ctx.newton_solve(m, d, ctx)
  elif m.opt.solver == SolverType.NEWTON:
    _newton_hessian_solve(m, d, ctx, compact)
  else:
    raise ValueError(f"Unknown solver type: {m.opt.solver}")


def _update_gradient_incremental(m: Model, d: Data, ctx: SolverContext, stable_fast: bool = False):
  """Incremental gradient update: update H for changed constraints + re-factorize.

  Skips the full J^T*D*J rebuild by applying only the delta from constraints
  that changed QUADRATIC state, then re-factorizes and solves.

  Args:
    m: model.
    d: data.
    ctx: solver context; ctx.h is updated in place from ctx.quad_changed_* and
      re-solved into ctx.search.
    stable_fast: unchanged worlds keep a rescaled stale gradient and skip the solve.
  """
  changed = ctx.state_changed_count if stable_fast else d.nefc  # (nworld,) gate (placeholder when off)
  wp.launch(
    _update_gradient_zero_grad_dot(stable_fast),
    dim=d.nworld,
    inputs=[changed, ctx.alpha, ctx.done],
    outputs=[ctx.grad_dot, ctx.newton_decrement, ctx.grad_scale, ctx.search_unchanged],
  )

  wp.launch(
    _update_gradient_grad(stable_fast),
    dim=(d.nworld, m.nv),
    inputs=[d.qfrc_smooth, d.qfrc_constraint, d.efc.Ma, changed, ctx.done],
    outputs=[ctx.grad, ctx.grad_dot],
  )

  # Update upper triangle of H with delta from changed constraints.
  sc = _sparse_compact(ctx)  # bool: compact solve over a sparse full model
  if m.is_sparse or sc:
    dj = ctx.compact_d_full if sc else d  # data holding the J structures to read
    slots = _jtdaj_groups_per_world(d.nworld, ctx.quad_changed_ids.shape[1])  # scalar: slots per world
    wp.launch(
      _update_gradient_h_incremental_sparse(sc),
      dim=(d.nworld, slots, _JTDAJ_THREADS_PER_GROUP),
      inputs=[
        dj.efc.J_rownnz,
        dj.efc.J_rowadr,
        dj.efc.J_colind,
        dj.efc.J,
        d.efc.D,
        d.efc.state,
        dj.dof_cdof,
        ctx.quad_changed_ids,
        ctx.quad_changed_count,
        slots,
      ],
      outputs=[ctx.h],
    )
  else:
    tri_dim = m.nv * (m.nv + 1) // 2  # scalar: upper-triangle elements of H
    wp.launch(
      _update_gradient_h_incremental,
      dim=(d.nworld, tri_dim),
      inputs=[
        d.efc.J,
        d.efc.D,
        d.efc.state,
        ctx.quad_changed_ids,
        ctx.quad_changed_count,
      ],
      outputs=[ctx.h],
    )

  _cholesky_factorize_solve(m, d, ctx, skip_unchanged=True, skip_noflip=stable_fast)


@wp.kernel
def _solve_search_update_cg_tiled(
  # Model:
  nv: int,
  # In:
  ctx_grad_in: wp.array2d[float],
  ctx_Mgrad_in: wp.array2d[float],
  ctx_search_in: wp.array2d[float],
  ctx_beta_in: wp.array[float],
  ctx_done_in: wp.array[bool],
  # Out:
  ctx_search_out: wp.array2d[float],
  ctx_search_dot_out: wp.array[float],
  ctx_prev_grad_out: wp.array2d[float],
  ctx_prev_Mgrad_out: wp.array2d[float],
):
  """CG search update: search = -Mgrad + beta * search; stores prev grad/Mgrad.

  Args:
    nv: scalar number of dofs.
    ctx_grad_in: (nworld, nv_pad) cost gradient.
    ctx_Mgrad_in: (nworld, nv_pad) preconditioned gradient.
    ctx_search_in: (nworld, nv) previous search direction.
    ctx_beta_in: (nworld,) Polak-Ribiere coefficients.
    ctx_done_in: (nworld,) converged flags.
    ctx_search_out: (nworld, nv) updated search direction.
    ctx_search_dot_out: (nworld,) search . search (tile-reduced).
    ctx_prev_grad_out: (nworld, nv) gradient saved for the next beta.
    ctx_prev_Mgrad_out: (nworld, nv) preconditioned gradient saved for the next beta.
  """
  worldid, tid = wp.tid()  # scalars: world, lane within the world's block

  if ctx_done_in[worldid]:
    return

  local_search_dot = float(0.0)  # scalar: this lane's partial search . search
  BLOCK_DIM = wp.block_dim()  # scalar: lanes per world
  beta = ctx_beta_in[worldid]  # scalar: Polak-Ribiere coefficient

  for dofid in range(tid, nv, BLOCK_DIM):
    mgrad = ctx_Mgrad_in[worldid, dofid]  # scalar: preconditioned gradient entry
    search = -1.0 * mgrad + beta * ctx_search_in[worldid, dofid]  # scalar: conjugate direction entry

    ctx_search_out[worldid, dofid] = search
    local_search_dot += search * search

    ctx_prev_grad_out[worldid, dofid] = ctx_grad_in[worldid, dofid]
    ctx_prev_Mgrad_out[worldid, dofid] = mgrad

  search_dot_tile = wp.tile(local_search_dot, preserve_type=True)
  search_dot_sum = wp.tile_reduce(wp.add, search_dot_tile)

  if tid == 0:
    ctx_search_dot_out[worldid] = search_dot_sum[0]


@cache_kernel
def _solve_beta_finalize_tiled(warn_overflow: int):
  """Builds the CG finalize kernel: Polak-Ribiere beta plus convergence test.

  Args:
    warn_overflow: scalar OverflowType warning bitmask, compiled in.

  Returns:
    A tiled kernel over (nworld,) writing ctx.beta and the done bookkeeping.
  """
  WARN_OVERFLOW = warn_overflow  # scalar: compile-time warning bitmask

  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Model:
    nv: int,
    opt_tolerance: wp.array[float],
    opt_iterations: int,
    stat_meaninertia: wp.array[float],
    # In:
    ctx_grad_in: wp.array2d[float],
    ctx_Mgrad_in: wp.array2d[float],
    ctx_prev_grad_in: wp.array2d[float],
    ctx_prev_Mgrad_in: wp.array2d[float],
    ctx_improvement_in: wp.array[float],
    ctx_grad_dot_in: wp.array[float],
    ctx_done_in: wp.array[bool],
    # Data out:
    solver_niter_out: wp.array[int],
    overflow_out: wp.array[int],
    # Out:
    ctx_beta_out: wp.array[float],
    nsolving_out: wp.array[int],
    ctx_done_out: wp.array[bool],
  ):
    """Reduces one world's Polak-Ribiere beta and tests convergence.

    Args:
      nv: scalar number of dofs.
      opt_tolerance: (1 or nworld,) main solver tolerance.
      opt_iterations: scalar iteration budget.
      stat_meaninertia: (1 or nworld,) mean diagonal inertia (cost scale).
      ctx_grad_in: (nworld, nv_pad) current gradient.
      ctx_Mgrad_in: (nworld, nv_pad) current preconditioned gradient.
      ctx_prev_grad_in: (nworld, nv) previous gradient.
      ctx_prev_Mgrad_in: (nworld, nv) previous preconditioned gradient.
      ctx_improvement_in: (nworld,) last linesearch improvement.
      ctx_grad_dot_in: (nworld,) grad . grad.
      ctx_done_in: (nworld,) converged flags (entry).
      solver_niter_out: (nworld,) iteration counters, incremented.
      overflow_out: (nworld,) OverflowType bitmask.
      ctx_beta_out: (nworld,) Polak-Ribiere coefficients max(0, num/den).
      nsolving_out: (1,) count of unconverged worlds, atomically decremented.
      ctx_done_out: (nworld,) converged flags (set on convergence or budget).
    """
    worldid, tid = wp.tid()  # scalars: world, lane within the world's block

    if ctx_done_in[worldid]:
      return

    local_num = float(0.0)  # scalar: this lane's partial beta numerator
    local_den = float(0.0)  # scalar: this lane's partial beta denominator
    BLOCK_DIM = wp.block_dim()  # scalar: lanes per world

    for dofid in range(tid, nv, BLOCK_DIM):
      prev_Mgrad = ctx_prev_Mgrad_in[worldid, dofid]  # scalar: previous Mgrad entry
      num = ctx_grad_in[worldid, dofid] * (ctx_Mgrad_in[worldid, dofid] - prev_Mgrad)  # scalar
      den = ctx_prev_grad_in[worldid, dofid] * prev_Mgrad  # scalar
      local_num += num
      local_den += den

    beta_tile = wp.tile(wp.vec2(local_num, local_den), preserve_type=True)  # tile: partials
    beta_sum = wp.tile_reduce(wp.add, beta_tile)
    reduced_beta = beta_sum[0]  # vec2: reduced (numerator, denominator)

    if tid == 0:
      ctx_beta_out[worldid] = wp.max(0.0, reduced_beta[0] / wp.max(MINVAL, reduced_beta[1]))

      solver_niter_out[worldid] += 1
      tolerance = opt_tolerance[worldid % opt_tolerance.shape[0]]
      meaninertia = stat_meaninertia[worldid % stat_meaninertia.shape[0]]

      grad_dot = ctx_grad_dot_in[worldid]  # scalar: grad . grad

      improvement = _rescale(nv, meaninertia, ctx_improvement_in[worldid])  # scalar: normalized
      gradient = _rescale(nv, meaninertia, wp.sqrt(grad_dot))  # scalar: normalized gradient norm
      done = (improvement < tolerance) or (gradient < tolerance)  # bool: converged
      if done or solver_niter_out[worldid] == opt_iterations:
        if not done and solver_niter_out[worldid] == opt_iterations:
          if wp.static(bool(WARN_OVERFLOW & OverflowType.ITERATIONS)):
            wp.printf(
              "solver iterations limit reached - please increase iterations beyond %u\n"
              "To disable the print warning: m.opt.warn_overflow &= ~mjw.OverflowType.ITERATIONS (or = 0 for all)\n",
              opt_iterations,
            )
          overflow_out[worldid] = overflow_out[worldid] | OverflowType.ITERATIONS
        ctx_done_out[worldid] = True
        wp.atomic_add(nsolving_out, 0, -1)

  return kernel


@cache_kernel
def _solve_done(warn_overflow: int):
  """Builds the Newton convergence-test kernel.

  Args:
    warn_overflow: scalar OverflowType warning bitmask, compiled in.

  Returns:
    A kernel over (nworld,) incrementing solver_niter and freezing converged worlds.
  """

  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Model:
    nv: int,
    opt_tolerance: wp.array[float],
    opt_iterations: int,
    stat_meaninertia: wp.array[float],
    # In:
    ctx_grad_dot_in: wp.array[float],
    ctx_newton_decrement_in: wp.array[float],
    ctx_improvement_in: wp.array[float],
    ctx_done_in: wp.array[bool],
    # Data out:
    solver_niter_out: wp.array[int],
    overflow_out: wp.array[int],
    # Out:
    nsolving_out: wp.array[int],
    ctx_done_out: wp.array[bool],
  ):
    """Tests one world's convergence on improvement, gradient, and Newton decrement.

    Args:
      nv: scalar number of dofs.
      opt_tolerance: (1 or nworld,) main solver tolerance.
      opt_iterations: scalar iteration budget.
      stat_meaninertia: (1 or nworld,) mean diagonal inertia (cost scale).
      ctx_grad_dot_in: (nworld,) grad . grad.
      ctx_newton_decrement_in: (nworld,) grad . H^-1 grad.
      ctx_improvement_in: (nworld,) last linesearch improvement.
      ctx_done_in: (nworld,) converged flags (entry).
      solver_niter_out: (nworld,) iteration counters, incremented.
      overflow_out: (nworld,) OverflowType bitmask.
      nsolving_out: (1,) count of unconverged worlds, atomically decremented.
      ctx_done_out: (nworld,) converged flags (set on convergence or budget).
    """
    worldid = wp.tid()  # scalar: world index

    if ctx_done_in[worldid]:
      return

    solver_niter_out[worldid] += 1
    tolerance = opt_tolerance[worldid % opt_tolerance.shape[0]]  # scalar: solver tolerance
    meaninertia = stat_meaninertia[worldid % stat_meaninertia.shape[0]]  # scalar: cost scale

    improvement = _rescale(nv, meaninertia, ctx_improvement_in[worldid])  # scalar: normalized
    gradient = _rescale(nv, meaninertia, wp.sqrt(ctx_grad_dot_in[worldid]))  # scalar: normalized
    model_improvement = _rescale(nv, meaninertia, 0.5 * ctx_newton_decrement_in[worldid])  # scalar
    done = (improvement < tolerance) or (gradient < tolerance) or (model_improvement < tolerance)  # bool
    if done or solver_niter_out[worldid] == opt_iterations:
      # if the solver has converged or the maximum number of iterations has been reached then
      # mark this world as done and remove it from the number of unconverged worlds
      if not done and solver_niter_out[worldid] == opt_iterations:
        if wp.static(bool(warn_overflow & OverflowType.ITERATIONS)):
          wp.printf(
            "solver iterations limit reached - please increase iterations beyond %u\n"
            "To disable the print warning: m.opt.warn_overflow &= ~mjw.OverflowType.ITERATIONS (or = 0 for all)\n",
            opt_iterations,
          )
        overflow_out[worldid] = overflow_out[worldid] | OverflowType.ITERATIONS
      ctx_done_out[worldid] = True
      wp.atomic_add(nsolving_out, 0, -1)

  return kernel


# The linesearch runs in ray units anchored at the last gradient rebuild, and
# its bracketing arithmetic cannot resolve steps below ~eps times the gross
# derivative-term magnitude over the curvature (see _linesearch_iterative_kernel).
# A world whose tolerance sits below that floor could never terminate on the
# stale ray; rebuilding re-anchors the arithmetic at the current (much smaller)
# gradient scale. The 8 is an allowance for the reduction depth of the sums.
_ALPHA_NOISE_EPS = 8.0 * 1.1920929e-07  # 8 * float32 eps


def _use_incremental(m: Model, ctx=None) -> bool:
  """Whether constraint state changes are tracked for incremental H updates: never where the
  caller solves the Newton system (ctx.newton_solve), which builds no H."""
  if ctx is not None and getattr(ctx, "newton_solve", None) is not None:
    return False
  return m.opt.solver == SolverType.NEWTON and m.opt.cone != ConeType.ELLIPTIC


@wp.kernel(grid_stride=True)
def _zero_change_counters(
  # Out:
  quad_changed_count_out: wp.array[int],
  state_changed_count_out: wp.array[int],
):
  """Zeroes the per-world incremental change counters before a constraint update.

  Args:
    quad_changed_count_out: (nworld,) QUADRATIC-flip counters.
    state_changed_count_out: (nworld,) any-state-change counters.
  """
  worldid = wp.tid()  # scalar: world index
  quad_changed_count_out[worldid] = 0
  state_changed_count_out[worldid] = 0


@event_scope
def _solver_iteration(
  m: Model,
  d: Data,
  ctx: SolverContext,
  nsolving: wp.array[int],
  compact: bool = False,
):
  """One solver iteration: linesearch step, constraint/gradient refresh, done test.

  Args:
    m: model.
    d: data; qacc (nworld, nv) and the efc arrays advance in place.
    ctx: solver context of the current solve.
    nsolving: (1,) count of unconverged worlds (drives the capture_while loop).
    compact: dense compacted-DOF path.
  """
  _linesearch(m, d, ctx)

  # Incremental H is only valid for non-elliptic cones. The elliptic cone
  # path in _update_constraint_efc has early returns that skip state change
  # tracking, and the additional JTCJ Hessian term depends on Jaref which
  # changes every iteration.
  incremental = _use_incremental(m, ctx)  # bool: constraint-state change tracking

  if incremental:
    # Must complete before _update_constraint_efc which atomically increments.
    wp.launch(
      _zero_change_counters,
      dim=d.nworld,
      outputs=[ctx.quad_changed_count, ctx.state_changed_count],
    )

  # The tracking also enables the stable-state fast path: worlds with no state
  # flips this iteration were exactly quadratic over the step, so grad/search
  # only changed by a scalar along the same ray. Skip their qfrc/grad/
  # solve/search updates and track the scalar in ctx.grad_scale.
  _update_constraint(m, d, ctx, track_changes=incremental, stable_fast=incremental)

  if incremental:
    _update_gradient_incremental(m, d, ctx, stable_fast=incremental)
  else:
    _update_gradient(m, d, ctx, compact=compact)

  # polak-ribiere
  if m.opt.solver == SolverType.CG:
    wp.launch_tiled(
      _solve_beta_finalize_tiled(int(m.opt.warn_overflow)),
      dim=d.nworld,
      inputs=[
        m.nv,
        m.opt.tolerance,
        m.opt.iterations,
        m.stat.meaninertia,
        ctx.grad,
        ctx.Mgrad,
        ctx.prev_grad,
        ctx.prev_Mgrad,
        ctx.improvement,
        ctx.grad_dot,
        ctx.done,
      ],
      outputs=[
        d.solver_niter,
        d.overflow,
        ctx.beta,
        nsolving,
        ctx.done,
      ],
      block_dim=m.block_dim.solve_beta_accumulate,
    )
    wp.launch_tiled(
      _solve_search_update_cg_tiled,
      dim=d.nworld,
      inputs=[m.nv, ctx.grad, ctx.Mgrad, ctx.search, ctx.beta, ctx.done],
      outputs=[ctx.search, ctx.search_dot, ctx.prev_grad, ctx.prev_Mgrad],
      block_dim=m.block_dim.solve_search_update_cg,
    )

  else:
    wp.launch(
      _solve_done(int(m.opt.warn_overflow)),
      dim=d.nworld,
      inputs=[
        m.nv,
        m.opt.tolerance,
        m.opt.iterations,
        m.stat.meaninertia,
        ctx.grad_dot,
        ctx.newton_decrement,
        ctx.improvement,
        ctx.done,
      ],
      outputs=[d.solver_niter, d.overflow, nsolving, ctx.done],
    )


@wp.kernel(grid_stride=True)
def _fill_contact_impratio(
  # Model:
  opt_impratio_invsqrt: wp.array[float],
  # Data in:
  contact_worldid_in: wp.array[int],
  # Out:
  contact_impratio_invsqrt_out: wp.array[float],
):
  """Resets the per-contact cone warp to the world impratio.

  Args:
    opt_impratio_invsqrt: (1 or nworld,) world 1/sqrt(impratio).
    contact_worldid_in: (naconmax,) world of each contact (may be stale past nacon).
    contact_impratio_invsqrt_out: (naconmax,) per-contact warp values.
  """
  conid = wp.tid()  # scalar: contact slot
  worldid = contact_worldid_in[conid]  # scalar: may be stale garbage past nacon
  if worldid < 0:
    worldid = 0
  contact_impratio_invsqrt_out[conid] = opt_impratio_invsqrt[worldid % opt_impratio_invsqrt.shape[0]]


def init_context(m: Model, d: Data, ctx: SolverContext | InverseContext, grad: bool = True, compact: bool = False):
  """Initializes the solver context against the current (qacc, efc) state.

  Computes ctx.Jaref = efc_J @ qacc - efc_aref (nworld, njmax), Ma = M @ qacc
  (nworld, nv), the constraint forces/states, and optionally the gradient (and, for
  Newton, the Hessian and search direction).

  Args:
    m: model.
    d: data.
    ctx: solver or inverse context; the per-world counters are reset.
    grad: also run _update_gradient.
    compact: dense compacted-DOF path.
  """
  # per-contact cone warp defaults to the world impratio
  wp.launch(
    _fill_contact_impratio,
    dim=d.contact.dist.shape[0],
    inputs=[m.opt.impratio_invsqrt, d.contact.worldid],
    outputs=[d.contact_impratio_invsqrt],
  )

  # initialize some efc arrays
  wp.launch(
    _solve_init_efc,
    dim=d.nworld,
    outputs=[d.solver_niter, ctx.search_dot, ctx.done],
  )

  # jaref = d.efc_J @ d.qacc - d.efc_aref

  # if we are only using 1 thread, it makes sense to do more dofs as we can also skip the
  # init kernel. For more than 1 thread, dofs_per_thread is lower for better load balancing.

  if m.is_sparse:
    # Sparse J has few nonzeros per row, one thread handles them all.
    dofs_per_thread = m.nv  # scalar: dofs each thread accumulates
    threads_per_efc = 1  # scalar: threads cooperating on one efc row
  elif m.nv > 50:
    dofs_per_thread = 20
    threads_per_efc = ceil(m.nv / dofs_per_thread)
  else:
    dofs_per_thread = 50
    threads_per_efc = ceil(m.nv / dofs_per_thread)
  # we need to clear the jaref array if we're doing atomic adds.
  if threads_per_efc > 1:
    ctx.Jaref.zero_()

  sc = _sparse_compact(ctx)  # bool: compact solve over a sparse full model
  dj = ctx.compact_d_full if sc else d  # data holding the J structures to read
  if sc:
    dofs_per_thread = m.nv
    threads_per_efc = 1
  wp.launch(
    _solve_init_jaref_kernel(sc or m.is_sparse, m.nv, dofs_per_thread, sc),
    dim=(d.nworld, d.njmax, threads_per_efc),
    inputs=[d.nefc, d.qacc, dj.efc.J_rownnz, dj.efc.J_rowadr, dj.efc.J_colind, dj.efc.J, d.efc.aref, dj.dof_cdof],
    outputs=[ctx.Jaref],
  )

  # Ma = M @ qacc
  _mul_m_compact_aware(m, d, ctx, d.efc.Ma, d.qacc, ctx.done)

  _update_constraint(m, d, ctx)

  if grad:
    _update_gradient(m, d, ctx, compact=compact)


@event_scope
def solve(m: Model, d: Data, newton_solve=None):
  """Public constraint-solve entry point: dispatches to the configured solver.

  Sleeping models run the compacted solve over awake dofs; prox_enable runs the
  exact-contact prox solver (solve_prox), otherwise the stock CG/Newton solve. Writes qacc,
  qfrc_constraint, the efc arrays, and solver_niter.

  newton_solve: None, or a callable newton_solve(m, d, ctx) that solves the Newton system in
    place of forming and factoring the dense Hessian H = M + J^T C J: after the gradient,
    it reads ctx.grad (and d.efc, ctx.Jaref, the contacts) and writes ctx.search = -H^-1 grad,
    ctx.search_dot and ctx.newton_decrement = grad . H^-1 grad for every world not ctx.done
    (e.g. in the space of the constraint rows). Newton only, not with sleep.
  """
  if m.opt.shared_hessian:
    reason = shared_hessian_refusal(m)
    if not reason and newton_solve is not None:
      reason = "a newton_solve replaces the Hessian it builds"
    if reason:
      raise ValueError(f"opt.shared_hessian: {reason}.")
  if newton_solve is not None:
    if m.opt.solver != SolverType.NEWTON:
      raise ValueError("newton_solve replaces the Newton solver's Hessian solve: it needs the Newton solver.")
    if m.opt.enableflags & EnableBit.SLEEP:
      raise ValueError("newton_solve does not support sleeping.")
  if m.opt.enableflags & EnableBit.SLEEP:
    if m.opt.prox_enable:
      raise ValueError("prox_enable does not support sleeping.")
    # Self-contained like the island branch below: rebuild the active-DOF mapping from
    # tree_awake so solve() works when called directly (not only via fwd_acceleration).
    update_active_dofs(m, d)
    solve_compact(m, d)
    if m.ntree > 1:
      compute_island_mapping(m, d)
    return

  if d.njmax == 0 or m.nv == 0:
    wp.copy(d.qacc, d.qacc_smooth)
    d.solver_niter.fill_(0)
  else:
    ctx = _create_solver_context(m, d, newton_solve)
    if m.opt.prox_enable:
      # exact contact solve: prox + De Saxce fixed point around _solve (solve_prox, below)
      solve_prox(m, d, ctx)
    else:
      _solve(m, d, ctx)


def _solve(m: Model, d: Data, ctx: SolverContext, compact: bool = False):
  """Finds forces that satisfy constraints.

  Seeds qacc from the warmstart, initializes the context, and runs solver iterations
  until every world converges (graph-conditional loop on CUDA, fixed count otherwise).

  Args:
    m: model.
    d: data; writes qacc (nworld, nv), qfrc_constraint (nworld, nv), the efc arrays,
      and solver_niter (nworld,).
    ctx: solver context (freshly created, or reused by solve_prox).
    compact: dense compacted-DOF path (run under solve_compact).
  """
  warmstart = not (m.opt.disableflags & DisableBit.WARMSTART)  # bool: qacc warm start enabled
  wp.launch(
    _solve_init_dof(warmstart, m.is_sparse),
    dim=(d.nworld, m.nv),
    inputs=[d.nefc, d.qacc_warmstart, d.qacc_smooth],
    outputs=[d.qacc, d.qfrc_constraint],
  )

  #  context
  init_context(m, d, ctx, grad=True, compact=compact)

  if _use_incremental(m, ctx):
    # A new solve computes a new search direction: invalidate the mv/jv reuse
    # left over from the previous solve.
    ctx.search_unchanged.zero_()

  # CG search = -Mgrad
  if m.opt.solver == SolverType.CG:
    wp.launch_tiled(
      _solve_init_search_cg_tiled,
      dim=d.nworld,
      inputs=[m.nv, ctx.grad, ctx.Mgrad],
      outputs=[ctx.search, ctx.search_dot, ctx.prev_grad, ctx.prev_Mgrad],
      block_dim=m.block_dim.solve_init_search_cg,
    )

  nsolving = wp.full(shape=(1,), value=d.nworld, dtype=int)  # (1,) int: unconverged world count
  if m.opt.iterations != 0 and m.opt.graph_conditional:
    # Note: the iteration kernel (indicated by while_body) is repeatedly launched
    # as long as condition_iteration is not zero.
    # condition_iteration is a warp array of size 1 and type int, it counts the number
    # of worlds that are not converged, it becomes 0 when all worlds are converged.
    # When the number of iterations reaches m.opt.iterations, solver_niter
    # becomes zero and all worlds are marked as converged to avoid an infinite loop.
    # note: we only launch the iteration kernel if everything is not done
    wp.capture_while(nsolving, while_body=_solver_iteration, m=m, d=d, ctx=ctx, nsolving=nsolving, compact=compact)
  else:
    # This branch is mostly for when JAX is used as it is currently not compatible
    # with CUDA graph conditional.
    # It should be removed when JAX becomes compatible.
    for _ in range(m.opt.iterations):
      _solver_iteration(m, d, ctx, nsolving, compact=compact)

  # Recover qfrc_constraint (the compacted buffer when run under solve_compact):
  # the fast path leaves it stale, and the per-iteration zeroing wiped it for
  # worlds that converged early.
  if _use_incremental(m, ctx):
    wp.launch(
      _qfrc_constraint_from_grad,
      dim=(d.nworld, m.nv),
      inputs=[d.qfrc_smooth, d.efc.Ma, ctx.grad, ctx.grad_scale],
      outputs=[d.qfrc_constraint],
    )


# Active-DOF compaction solve (nvmax < nv).
#
# When fewer than nv DOFs are active, the active set is packed into a contiguous
# [0, ncdof) range (see island.update_active_dofs) and the dense factor/solve runs at
# size nvmax_pad instead of nv. The compacted workspace lives on Data as c* fields
# (mirroring the island-local i* fields). The constrained solve gathers into compacted
# arrays, shallow-replaces (m, d) so the stock dense Newton solver runs at nvmax_pad,
# then scatters qacc/qfrc_constraint back, freezing inactive DOFs to 0.


@wp.kernel
def _init_compact_inertia(
  # Data in:
  ncdof_in: wp.array[int],
  # Out:
  M_c_out: wp.array3d[float],
):
  """Seeds the compact inertia: zero, with identity on the inactive padded diagonal.

  Args:
    ncdof_in: (nworld,) active (compacted) dof counts.
    M_c_out: (nworld, nvmax_pad, nvmax_pad) compact dense inertia.
  """
  worldid, i, j = wp.tid()  # scalars: world, row, column
  val = 0.0  # scalar: 1 on the padded diagonal, else 0
  if i == j and i >= ncdof_in[worldid]:
    val = 1.0
  M_c_out[worldid, i, j] = val


@wp.kernel
def _mul_m_sparse_compact(
  # Model:
  M_mulm_rowadr: wp.array[int],
  M_mulm_col: wp.array[int],
  M_mulm_madr: wp.array[int],
  # Data in:
  M_in: wp.array2d[float],
  dof_cdof_in: wp.array2d[int],
  cdof_dof_in: wp.array2d[int],
  # In:
  vec: wp.array2d[float],
  skip: wp.array[bool],
  # Out:
  res: wp.array2d[float],
):
  """Compacted res = M @ vec via the full-coordinate sparse M (no gathered cM).

  Args:
    M_mulm_rowadr: (nv+1,) row starts of the mul_m CSR structure.
    M_mulm_col: (nnz,) dof column of each entry.
    M_mulm_madr: (nnz,) address of each entry in d.M.
    M_in: (nworld, nC) CSR inertia values.
    dof_cdof_in: (nworld, nv) global dof -> compacted dof map.
    cdof_dof_in: (nworld, nvmax_pad) compacted dof -> global dof map.
    vec: (nworld, nvmax_pad) compacted input vector.
    skip: (nworld,) worlds to leave untouched.
    res: (nworld, nvmax_pad) compacted output M @ vec.
  """
  worldid, ci = wp.tid()  # scalars: world, compacted dof

  if skip[worldid]:
    return

  dof = cdof_dof_in[worldid, ci]  # scalar: global dof (-1 on padding)
  if dof < 0:
    res[worldid, ci] = 0.0
    return

  acc = float(0.0)  # scalar: running (M @ vec)[ci]
  start = M_mulm_rowadr[dof]  # scalar: row start
  end = M_mulm_rowadr[dof + 1]  # scalar: row end
  for k in range(start, end):
    # tree-internal columns are awake with the row; guard like _gather_M_sparse
    # in case M ever carries cross-tree entries (their gathered value is zero)
    cj = dof_cdof_in[worldid, M_mulm_col[k]]  # scalar: compacted column (-1 if inactive)
    if cj >= 0:
      acc += M_in[worldid, M_mulm_madr[k]] * vec[worldid, cj]
  res[worldid, ci] = acc


def _sparse_compact(ctx: SolverContext | InverseContext) -> bool:
  """Whether this solve is compact over a sparse full model (full J structures exist)."""
  return ctx.compact_d_full is not None and ctx.compact_m_full.is_sparse


def _mul_m_compact_aware(m: Model, d: Data, ctx: SolverContext | InverseContext, res, vec, skip):
  """M @ vec: full-coordinate sparse walk under compact, support.mul_m natively.

  Args:
    m: (possibly shallow-replaced compact) model.
    d: data of the running solve.
    ctx: context; compact_m_full/compact_d_full select the compact path.
    res: (nworld, nv) output M @ vec.
    vec: (nworld, nv) input vector.
    skip: (nworld,) worlds to leave untouched.
  """
  dfull = ctx.compact_d_full  # full-coordinate Data under compact, else None
  if dfull is not None:
    mfull = ctx.compact_m_full
    wp.launch(
      _mul_m_sparse_compact,
      dim=(d.nworld, m.nv),
      inputs=[
        mfull.M_mulm_rowadr,
        mfull.M_mulm_col,
        mfull.M_mulm_madr,
        dfull.M,
        dfull.dof_cdof,
        dfull.cdof_dof,
        vec,
        skip,
      ],
      outputs=[res],
    )
  else:
    mul_m(m, d, res, vec, skip=skip)


@wp.kernel
def _gather_M_sparse(
  # Model:
  M_rownnz: wp.array[int],
  M_rowadr: wp.array[int],
  M_colind: wp.array[int],
  # Data in:
  M_in: wp.array2d[float],
  dof_cdof_in: wp.array2d[int],
  # Out:
  M_c_out: wp.array3d[float],
):
  """Gathers the sparse inertia rows of active dofs into the dense compact cM.

  Args:
    M_rownnz: (nv,) nonzeros per inertia row.
    M_rowadr: (nv,) row starts in the CSR buffer.
    M_colind: (nC,) dof column of each entry.
    M_in: (nworld, nC) CSR inertia values.
    dof_cdof_in: (nworld, nv) global dof -> compacted dof map.
    M_c_out: (nworld, nvmax_pad, nvmax_pad) compact dense inertia (both triangles).
  """
  worldid, i = wp.tid()  # scalars: world, global dof row
  ci = dof_cdof_in[worldid, i]  # scalar: compacted row (-1 if inactive)
  if ci < 0:
    return
  rowadr = M_rowadr[i]  # scalar: row start in the CSR buffer
  for k in range(M_rownnz[i]):
    adr = rowadr + k  # scalar: CSR entry index
    cj = dof_cdof_in[worldid, M_colind[adr]]  # scalar: compacted column (-1 if inactive)
    if cj >= 0:
      val = M_in[worldid, adr]  # scalar: inertia entry, mirrored to both triangles
      M_c_out[worldid, ci, cj] = val
      M_c_out[worldid, cj, ci] = val


@wp.kernel
def _gather_rhs_compact(
  # Data in:
  cdof_dof_in: wp.array2d[int],
  # In:
  vec_in: wp.array2d[float],
  # Out:
  rhs_out: wp.array3d[float],
):
  """Gathers a full-coordinate dof vector into a compacted column (inactive -> 0).

  Args:
    cdof_dof_in: (nworld, nvmax_pad) compacted dof -> global dof map.
    vec_in: (nworld, nv) full-coordinate vector.
    rhs_out: (nworld, nvmax_pad, 1) compacted column vector.
  """
  worldid, ci = wp.tid()  # scalars: world, compacted dof
  dof = cdof_dof_in[worldid, ci]  # scalar: global dof (-1 on padding)
  if dof >= 0:
    rhs_out[worldid, ci, 0] = vec_in[worldid, dof]
  else:
    rhs_out[worldid, ci, 0] = 0.0


@wp.kernel
def _scatter_solution(
  # Data in:
  dof_cdof_in: wp.array2d[int],
  # In:
  x_in: wp.array3d[float],
  # Out:
  vec_out: wp.array2d[float],
):
  """Scatters a compacted solution column back to full coordinates (inactive -> 0).

  Args:
    dof_cdof_in: (nworld, nv) global dof -> compacted dof map.
    x_in: (nworld, nvmax_pad, 1) compacted solution.
    vec_out: (nworld, nv) full-coordinate vector.
  """
  worldid, i = wp.tid()  # scalars: world, global dof
  ci = dof_cdof_in[worldid, i]  # scalar: compacted dof (-1 if inactive)
  if ci >= 0:
    vec_out[worldid, i] = x_in[worldid, ci, 0]
  else:
    vec_out[worldid, i] = 0.0  # frozen inactive DOF


@event_scope
def smooth_solve_compact(m: Model, d: Data):
  """Compacted equivalent of solve_m: qacc_smooth[active] = cM^-1 qfrc_smooth[active].

  Inactive DOFs are frozen (qacc_smooth set to 0). Reads the sparse Model inertia.
  """
  wp.launch(
    _init_compact_inertia,
    dim=(d.nworld, d.nvmax_pad, d.nvmax_pad),
    inputs=[d.ncdof],
    outputs=[d.cM],
  )
  wp.launch(
    _gather_M_sparse,
    dim=(d.nworld, m.nv),
    inputs=[m.M_rownnz, m.M_rowadr, m.M_colind, d.M, d.dof_cdof],
    outputs=[d.cM],
  )
  wp.launch(
    _gather_rhs_compact,
    dim=(d.nworld, d.nvmax_pad),
    inputs=[d.cdof_dof, d.qfrc_smooth],
    outputs=[d.crhs],
  )
  wp.launch_tiled(
    _cholesky_factorize_solve_blocked(TILE_SIZE_JTDAJ_DENSE, d.nvmax_pad),
    dim=d.nworld,
    inputs=[d.cM, d.crhs],
    outputs=[d.cqLD, d.cx],
    block_dim=m.block_dim.update_gradient_cholesky_blocked,
  )
  wp.launch(_scatter_solution, dim=(d.nworld, m.nv), inputs=[d.dof_cdof, d.cx], outputs=[d.qacc_smooth])


@wp.kernel
def _gather_dof_vecs_compact(
  # Data in:
  qacc_warmstart_in: wp.array2d[float],
  qfrc_smooth_in: wp.array2d[float],
  qacc_smooth_in: wp.array2d[float],
  cdof_dof_in: wp.array2d[int],
  # Out:
  qfrc_smooth_c_out: wp.array2d[float],
  qacc_smooth_c_out: wp.array2d[float],
  qacc_warmstart_c_out: wp.array2d[float],
):
  """Gathers the warmstart/smooth dof vectors into compacted buffers (inactive -> 0).

  Args:
    qacc_warmstart_in: (nworld, nv) warmstart accelerations.
    qfrc_smooth_in: (nworld, nv) smooth forces.
    qacc_smooth_in: (nworld, nv) unconstrained accelerations.
    cdof_dof_in: (nworld, nvmax_pad) compacted dof -> global dof map.
    qfrc_smooth_c_out: (nworld, nvmax_pad) compacted smooth forces.
    qacc_smooth_c_out: (nworld, nvmax_pad) compacted unconstrained accelerations.
    qacc_warmstart_c_out: (nworld, nvmax_pad) compacted warmstart accelerations.
  """
  worldid, ci = wp.tid()  # scalars: world, compacted dof
  dof = cdof_dof_in[worldid, ci]  # scalar: global dof (-1 on padding)
  if dof >= 0:
    qfrc_smooth_c_out[worldid, ci] = qfrc_smooth_in[worldid, dof]
    qacc_smooth_c_out[worldid, ci] = qacc_smooth_in[worldid, dof]
    qacc_warmstart_c_out[worldid, ci] = qacc_warmstart_in[worldid, dof]
  else:
    qfrc_smooth_c_out[worldid, ci] = 0.0
    qacc_smooth_c_out[worldid, ci] = 0.0
    qacc_warmstart_c_out[worldid, ci] = 0.0


@wp.kernel
def _scatter_dof_vecs(
  # Data in:
  dof_cdof_in: wp.array2d[int],
  # In:
  qacc_c_in: wp.array2d[float],
  qfrc_constraint_c_in: wp.array2d[float],
  # Data out:
  qacc_out: wp.array2d[float],
  qfrc_constraint_out: wp.array2d[float],
):
  """Scatters compacted qacc/qfrc_constraint back to full coordinates (inactive -> 0).

  Args:
    dof_cdof_in: (nworld, nv) global dof -> compacted dof map.
    qacc_c_in: (nworld, nvmax_pad) compacted accelerations.
    qfrc_constraint_c_in: (nworld, nvmax_pad) compacted constraint forces.
    qacc_out: (nworld, nv) accelerations.
    qfrc_constraint_out: (nworld, nv) constraint forces.
  """
  worldid, i = wp.tid()  # scalars: world, global dof
  ci = dof_cdof_in[worldid, i]  # scalar: compacted dof (-1 if inactive)
  if ci >= 0:
    qacc_out[worldid, i] = qacc_c_in[worldid, ci]
    qfrc_constraint_out[worldid, i] = qfrc_constraint_c_in[worldid, ci]
  else:
    qacc_out[worldid, i] = 0.0
    qfrc_constraint_out[worldid, i] = 0.0


@wp.kernel
def _gather_J_sparse(
  # Data in:
  nefc_in: wp.array[int],
  dof_cdof_in: wp.array2d[int],
  # In:
  J_rownnz_in: wp.array2d[int],
  J_rowadr_in: wp.array2d[int],
  J_colind_in: wp.array3d[int],
  J_in: wp.array3d[float],
  # Out:
  J_c_out: wp.array3d[float],
):
  """Gathers the sparse constraint Jacobian into the dense compacted cJ.

  Args:
    nefc_in: (nworld,) constraint row counts.
    dof_cdof_in: (nworld, nv) global dof -> compacted dof map.
    J_*: CSR constraint Jacobian of the full model.
    J_c_out: (nworld, njmax_pad, nvmax_pad) compacted dense Jacobian.
  """
  worldid, efcid = wp.tid()  # scalars: world, efc row
  if efcid >= nefc_in[worldid]:
    return
  rowadr = J_rowadr_in[worldid, efcid]  # scalar: row start in the sparse buffer
  for k in range(J_rownnz_in[worldid, efcid]):
    adr = rowadr + k  # scalar: sparse entry index
    cj = dof_cdof_in[worldid, J_colind_in[worldid, 0, adr]]  # scalar: compacted column
    if cj >= 0:
      J_c_out[worldid, efcid, cj] = J_in[worldid, 0, adr]


@wp.kernel
def _gather_J_dense(
  # Data in:
  nefc_in: wp.array[int],
  dof_cdof_in: wp.array2d[int],
  # In:
  J_in: wp.array3d[float],
  # Out:
  J_c_out: wp.array3d[float],
):
  """Gathers the dense constraint Jacobian columns into the compacted cJ.

  Args:
    nefc_in: (nworld,) constraint row counts.
    dof_cdof_in: (nworld, nv) global dof -> compacted dof map.
    J_in: (nworld, njmax_pad, nv_pad) dense constraint Jacobian.
    J_c_out: (nworld, njmax_pad, nvmax_pad) compacted dense Jacobian (pre-zeroed).
  """
  worldid, efcid = wp.tid()  # scalars: world, efc row
  if efcid >= nefc_in[worldid]:
    return
  nv = dof_cdof_in.shape[1]  # scalar: full dof count
  for j in range(nv):
    cj = dof_cdof_in[worldid, j]  # scalar: compacted column (-1 if inactive)
    if cj >= 0:
      J_c_out[worldid, efcid, cj] = J_in[worldid, efcid, j]


@event_scope
def solve_compact(m: Model, d: Data):
  """Run the dense Newton constraint solver in compacted DOF space.

  Gathers the active-DOF inertia, constraint Jacobian, and smooth/warmstart vectors
  into nvmax_pad-sized dense workspaces, runs the stock dense Newton solver on a
  shallow-replaced (m, d) at nvmax_pad, then scatters qacc/qfrc_constraint back.
  Inactive DOFs are frozen to 0. On the incremental Newton path the solver
  kernels read the sparse M and J directly through the compaction maps.
  """
  _compact_gather(m, d)

  # shallow-replace (m, d) so the stock dense Newton solver runs at nvmax_pad.
  # Keep graph-conditional early-exit on CUDA (matches baseline: stops at convergence
  # instead of running all iterations); fall back to the plain loop on CPU.
  nvp = d.nvmax_pad  # scalar: compacted (padded) dof count
  gc = m.opt.graph_conditional and wp.get_device().is_cuda  # bool: keep the CUDA early exit
  opt2 = dataclasses.replace(m.opt, graph_conditional=gc, tolerance=d.ctol, ls_tolerance=d.cls_tol)
  m2 = dataclasses.replace(
    m, opt=opt2, nv=nvp, nv_pad=nvp, is_sparse=False, dof_tri_row=d.cdof_tri_row, dof_tri_col=d.cdof_tri_col
  )
  efc2 = dataclasses.replace(d.efc, J=d.cJ, Ma=d.cMa)
  d2 = dataclasses.replace(
    d,
    M=d.cM,
    qfrc_smooth=d.cqfrc_smooth,
    qacc_smooth=d.cqacc_smooth,
    qacc_warmstart=d.cqacc_warmstart,
    qacc=d.cqacc,
    qfrc_constraint=d.cqfrc_constraint,
    efc=efc2,
  )

  sctx = _create_solver_context(m2, d2)
  # compact kernels read the full-coordinate sparse structures (M, J) through
  # the compaction maps instead of dense products on gathered blocks
  sctx.compact_m_full = m
  sctx.compact_d_full = d
  _solve(m2, d2, sctx, compact=True)

  _compact_scatter(m, d)


@event_scope
def _compact_gather(m: Model, d: Data):
  """Gathers the inertia, Jacobian, and dof vectors into the compacted buffers.

  Dense models fill cM (nworld, nvmax_pad, nvmax_pad) and cJ
  (nworld, njmax_pad, nvmax_pad); sparse models skip both (the solver kernels read
  the sparse structures through the compaction maps). Always gathers the
  warmstart/smooth vectors into the c* buffers (nworld, nvmax_pad).
  """
  nvp = d.nvmax_pad  # scalar: compacted (padded) dof count
  # gather compacted dense inertia and Jacobian only for dense models;
  # sparse models read sparse M and J directly through compaction maps
  if not m.is_sparse:
    wp.launch(
      _init_compact_inertia,
      dim=(d.nworld, nvp, nvp),
      inputs=[d.ncdof],
      outputs=[d.cM],
    )
    wp.launch(
      _gather_M_sparse,
      dim=(d.nworld, m.nv),
      inputs=[m.M_rownnz, m.M_rowadr, m.M_colind, d.M, d.dof_cdof],
      outputs=[d.cM],
    )
    d.cJ.zero_()
    wp.launch(
      _gather_J_dense,
      dim=(d.nworld, d.njmax),
      inputs=[d.nefc, d.dof_cdof, d.efc.J],
      outputs=[d.cJ],
    )
  # gather compacted DOF-space vectors in a single launch
  wp.launch(
    _gather_dof_vecs_compact,
    dim=(d.nworld, nvp),
    inputs=[
      d.qacc_warmstart,
      d.qfrc_smooth,
      d.qacc_smooth,
      d.cdof_dof,
    ],
    outputs=[
      d.cqfrc_smooth,
      d.cqacc_smooth,
      d.cqacc_warmstart,
    ],
  )


@event_scope
def _compact_scatter(m: Model, d: Data):
  """Scatters qacc/qfrc_constraint back to full coordinates and refreshes efc.Ma."""
  # scatter results back to full DOF space (inactive frozen to 0) in one launch
  wp.launch(
    _scatter_dof_vecs,
    dim=(d.nworld, m.nv),
    inputs=[d.dof_cdof, d.cqacc, d.cqfrc_constraint],
    outputs=[d.qacc, d.qfrc_constraint],
  )

  # Refresh full d.efc.Ma = M @ qacc. The integrators (Euler/implicit damping) use Ma as
  # the RHS; the compact solve only populated the compacted Ma, so recompute it in full
  # space. Inactive DOFs have qacc=0 so their Ma is 0 and they stay frozen.
  mul_m(m, d, d.efc.Ma, d.qacc)


#######################################################################################
# Lodos's proximal solve (solve_prox), around MuJoCo Warp's Newton solve
#######################################################################################

# sentinel that sorts inactive cache slots to the end
_KEY_SENTINEL = wp.constant(wp.int64(9223372036854775807))


# Lodos's proximal solve.
@wp.func
def _pack_key(worldid: int, g0: int, g1: int) -> wp.int64:
  """Packs the cross-step contact identity (world, geom pair) into one sortable key.

  Args:
    worldid: scalar world index (fits in 24 bits).
    g0: scalar first geom id (fits in 20 bits).
    g1: scalar second geom id (fits in 20 bits).

  Returns:
    scalar int64 (worldid << 40) | (g0 << 20) | g1. The key identifies the geom pair
    only; contacts within a pair are re-associated by position (per-pair manifold
    indices permute across steps).
  """
  return (wp.int64(worldid) << wp.int64(40)) | (wp.int64(g0) << wp.int64(20)) | wp.int64(g1)


# Lodos's proximal solve.
@wp.func
def _project_cone(z: wp.vec3, mu: float) -> wp.vec3:
  """Euclidean projection onto the friction cone K_mu = {z : ||z_t|| <= mu * z_n}.

  Args:
    z: vec3 in contact coordinates, ordered (normal, t1, t2).
    mu: scalar isotropic friction coefficient, >= 0.

  Returns:
    vec3 closest point of K_mu to z (z itself inside the cone, the apex 0 inside
    the polar cone, the cone-surface projection in the middle zone).
  """
  zn = z[0]  # scalar: normal component
  nt = wp.sqrt(z[1] * z[1] + z[2] * z[2])  # scalar: tangential norm ||z_t||
  if nt <= mu * zn:  # inside the cone
    return z
  if mu * nt <= -zn:  # inside the polar cone -> apex
    return wp.vec3(0.0)
  # middle zone
  s = (mu * nt + zn) / (1.0 + mu * mu)  # scalar: projected normal component
  scale = mu * s / wp.max(nt, MINVAL)  # scalar: tangential shrink factor mu*s/||z_t||
  return wp.vec3(s, scale * z[1], scale * z[2])


# Lodos's proximal solve.
@wp.func
def _zcache_match(
  # In:
  key: wp.int64,
  pos: wp.vec3,
  radius: float,
  cache_key: wp.array[wp.int64],
  cache_val: wp.array[int],
  cache_pos: wp.array[wp.vec3],
  cache_n: int,
) -> int:
  """Nearest cached contact of the same (world, geom pair) within radius, or -1.

  Binary-searches the sorted keys for the first entry >= key, then scans the (at
  most 32) entries of the matching run for the position-nearest cache slot.

  Args:
    key: scalar int64 packed (world, geom pair) identity (_pack_key).
    pos: vec3 world-frame position of the query contact.
    radius: scalar maximum match distance (0.3 * geom bounding radius).
    cache_key: (naconmax,) int64 keys, radix-sorted ascending (sentinel-padded).
    cache_val: (naconmax,) cache slot of each sorted key (the sort permutation).
    cache_pos: (naconmax,) vec3 cached contact positions, indexed by slot.
    cache_n: scalar number of valid cache entries (last step's nacon).

  Returns:
    scalar cache slot of the nearest same-pair contact within radius, or -1.
  """
  lo = int(0)  # scalar: binary-search lower bound
  hi = cache_n  # scalar: binary-search upper bound
  while lo < hi:
    mid = (lo + hi) >> 1  # scalar: bisection midpoint
    if cache_key[mid] < key:
      lo = mid + 1
    else:
      hi = mid
  best = int(-1)  # scalar: best cache slot so far (-1 = no match)
  best_d = radius  # scalar: distance of the best slot; matches must beat radius
  i = lo  # scalar: first sorted index with cache_key[i] >= key
  while i < cache_n and cache_key[i] == key and i < lo + 32:
    slot = cache_val[i]  # scalar: cache slot of the i-th sorted key
    dist = wp.length(pos - cache_pos[slot])  # scalar: distance to the cached contact
    if dist < best_d:
      best_d = dist
      best = slot
    i += 1
  return best


# Lodos's proximal solve.
@wp.func
def _body_twist(
  # Model:
  body_dofnum: wp.array[int],
  body_dofadr: wp.array[int],
  dof_parentid: wp.array[int],
  # Data in:
  qvel_in: wp.array2d[float],
  qacc_in: wp.array2d[float],
  cdof_in: wp.array2d[wp.spatial_vector],
  # In:
  worldid: int,
  bodyid: int,
  timestep: float,
) -> Tuple[wp.vec3, wp.vec3]:
  """A weld body's angular velocity, and its linear velocity at its root's COM, at qvel + dt qacc.

  Args:
    body_dofnum: (nbody,) dofs of each body.
    body_dofadr: (nbody,) first dof of each body (-1 without dofs).
    dof_parentid: (nv,) parent dof of each dof (-1 at a root).
    qvel_in: (nworld, nv) joint velocities.
    qacc_in: (nworld, nv) joint accelerations: the warm start, or the last inner solve's.
    cdof_in: (nworld, nv) spatial motion of each dof, (angular, linear) about its root's COM.
    worldid: scalar world index.
    bodyid: scalar weld body (body_weldid of a geom's body).
    timestep: scalar dt.

  Returns:
    (vec3 angular velocity, vec3 linear velocity at subtree_com of the body's root).
  """
  w = wp.vec3(0.0)  # vec3: angular velocity
  v = wp.vec3(0.0)  # vec3: linear velocity at the root's COM
  dofid = body_dofadr[bodyid] + body_dofnum[bodyid] - 1  # scalar: the body's last dof, then its ancestors'
  while dofid >= 0:
    qv = qvel_in[worldid, dofid] + timestep * qacc_in[worldid, dofid]  # scalar: dof velocity at the step's end
    cdof = cdof_in[worldid, dofid]  # spatial_vector: the dof's motion
    w += qv * wp.spatial_top(cdof)
    v += qv * wp.spatial_bottom(cdof)
    dofid = dof_parentid[dofid]
  return w, v


# Lodos's proximal solve.
@wp.func
def _group_gamma(
  # Model:
  body_rootid: wp.array[int],
  body_weldid: wp.array[int],
  body_dofnum: wp.array[int],
  body_dofadr: wp.array[int],
  dof_parentid: wp.array[int],
  geom_bodyid: wp.array[int],
  # Data in:
  qvel_in: wp.array2d[float],
  qacc_in: wp.array2d[float],
  cdof_in: wp.array2d[wp.spatial_vector],
  subtree_com_in: wp.array2d[wp.vec3],
  contact_group_adr_in: wp.array[int],
  contact_group_num_in: wp.array[int],
  group_member_pos_in: wp.array[wp.vec3],
  group_member_frame_in: wp.array[wp.mat33],
  group_member_weight_in: wp.array[float],
  group_member_slip_in: wp.array[float],
  # In:
  worldid: int,
  conid: int,
  geom: wp.vec2i,
  timestep: float,
  mu: float,
  ray: bool,
) -> float:
  """A GROUP contact's De Saxce term, per member, at sigma_i, member i's relative velocity at qvel + dt qacc.

  A cone (condim 3): mu sum_i a_i ||sigma_i,t||, each member's own Coulomb term. A ray
  (condim 1): sum_i a_i r_i t1_i . sigma_i, with r_i its friction ratio (group_member_slip)
  along its fixed friction direction -t1_i: it cancels the friction part of the ray's row
  velocity, sum_i a_i (sigma_i,n - r_i t1_i . sigma_i), so that the ray holds its members'
  normal velocities, sum_i a_i (sigma_i,n + gamma_i) >= 0, however their slips turn within the
  step. With mu, or with ||sigma_i,t||, a sliding group would be let into the surface at
  (mu - r_i) ||sigma_i,t||, or by r_i (||sigma_i,t|| - t1_i . sigma_i).

  Args:
    worldid: scalar world index.
    conid: scalar GROUP contact index.
    geom: vec2i the group's geom pair; the normal points from geom[0] to geom[1].
    timestep: scalar dt.
    mu: scalar effective isotropic friction of a cone group.
    ray: bool; a ray group (condim 1).

  Returns:
    scalar Gamma_n of the group's normal row.
  """
  body1 = body_weldid[geom_bodyid[geom[0]]]  # scalar: weld body of geom 0
  body2 = body_weldid[geom_bodyid[geom[1]]]  # scalar: weld body of geom 1
  w1, v1 = _body_twist(body_dofnum, body_dofadr, dof_parentid, qvel_in, qacc_in, cdof_in, worldid, body1, timestep)
  w2, v2 = _body_twist(body_dofnum, body_dofadr, dof_parentid, qvel_in, qacc_in, cdof_in, worldid, body2, timestep)
  com1 = subtree_com_in[worldid, body_rootid[body1]]  # vec3: body 1's root COM
  com2 = subtree_com_in[worldid, body_rootid[body2]]  # vec3: body 2's root COM
  gamma = float(0.0)  # scalar: the weighted sum
  adr = contact_group_adr_in[conid]  # scalar: first member
  for i in range(adr, adr + contact_group_num_in[conid]):
    p = group_member_pos_in[i]  # vec3: member point
    frame = group_member_frame_in[i]  # mat33: member frame, rows (normal, t1, t2)
    n = wp.vec3(frame[0, 0], frame[0, 1], frame[0, 2])  # vec3: member normal
    sigma = (v2 + wp.cross(w2, p - com2)) - (v1 + wp.cross(w1, p - com1))  # vec3: relative velocity
    if ray:
      t1 = wp.vec3(frame[1, 0], frame[1, 1], frame[1, 2])  # vec3: the slip the ray's friction opposes
      gamma += group_member_weight_in[i] * group_member_slip_in[i] * wp.dot(sigma, t1)
    else:
      gamma += group_member_weight_in[i] * mu * wp.length(sigma - wp.dot(sigma, n) * n)
  return gamma


# Lodos's proximal solve.
@wp.func
def _group_bias(
  # Data in:
  contact_group_adr_in: wp.array[int],
  contact_group_num_in: wp.array[int],
  group_member_dist_in: wp.array[float],
  group_member_weight_in: wp.array[float],
  # In:
  conid: int,
  margin: float,
  alpha: float,
  slop: float,
  timestep: float,
) -> float:
  """A GROUP contact's Baumgarte bias: its members' biases at their own depths, weighted, sum_i a_i gamma_n(d_i).

  Args:
    conid: scalar GROUP contact index.
    margin: scalar the group's include margin.
    alpha: scalar depenetration gain.
    slop: scalar penetration the bias leaves alone.
    timestep: scalar dt.

  Returns:
    scalar bias velocity of the group's normal row.
  """
  bias = float(0.0)  # scalar: the weighted sum
  adr = contact_group_adr_in[conid]  # scalar: first member
  for i in range(adr, adr + contact_group_num_in[conid]):
    pos = group_member_dist_in[i] - margin  # scalar: member penetration
    bias += group_member_weight_in[i] * (alpha * wp.min(0.0, (pos + slop) / timestep) + wp.max(0.0, pos / timestep))
  return bias


# Lodos's proximal solve.
@cache_kernel
def _prox_init(baumgarte: bool, carry: bool, desaxce: bool, soft: bool, groups: bool):
  """Builds the per-contact init kernel, specialized on (baumgarte, carry, desaxce, soft).

  Args:
    baumgarte: bool; compiles in the normal row's Baumgarte bias (opt.prox_baumgarte),
      else none: the velocity-level target.
    carry: bool; compiles in the cross-step z-cache lookup (opt.prox_z_carry > 0).
    desaxce: bool; compiles in the De Saxce correction (opt.prox_desaxce), else Gamma = 0.
    soft: bool; keeps MuJoCo's rows (opt.prox_soft): D = 1/(R + rho_i), aref = aref_mj
      - b Gamma + rho_i z on the normal row and aref_mj + rho_i z on the tangent rows, with
      b the normal row's damping from the contact's solref/solimp; else the exact rows.
    groups: bool; compiles in contact groups (io.alloc_contact_groups): a GROUP contact takes
      its Baumgarte bias and De Saxce term from its members (_group_bias, _group_gamma) and
      starts cold.

  Returns:
    A kernel over contacts (dim = naconmax) that overwrites the contact rows of
    (efc_D, efc_aref) with the prox rewrite D = dt/rho_i, aref = aref0 - Gamma/dt + z/D,
    warm-starting sigma from J*(qvel + dt*qacc_warmstart) and z from the previous
    step's anchor cache.
  """
  BAUMGARTE = baumgarte  # bool: compile-time bias switch (must be a local for wp.static)
  CARRY = carry  # bool: compile-time carry switch (must be a local for wp.static)
  DESAXCE = desaxce  # bool: compile-time De Saxce switch
  SOFT = soft  # bool: compile-time MuJoCo-softness switch
  GROUPS = groups  # bool: compile-time contact-group switch

  @wp.kernel(module="unique", enable_backward=False, grid_stride=True)
  def kernel(
    # Model:
    opt_timestep: wp.array[float],
    opt_impratio_invsqrt: wp.array[float],
    opt_prox_rho: wp.array[float],
    opt_prox_baumgarte_alpha: wp.array[float],
    opt_prox_baumgarte_slop: wp.array[float],
    opt_prox_z_carry: float,
    opt_disableflags: int,
    body_rootid: wp.array[int],
    body_weldid: wp.array[int],
    body_dofnum: wp.array[int],
    body_dofadr: wp.array[int],
    body_invweight0: wp.array2d[wp.vec2],
    dof_parentid: wp.array[int],
    geom_bodyid: wp.array[int],
    geom_rbound: wp.array2d[float],
    # Data in:
    contact_pos_in: wp.array[wp.vec3],
    contact_frame_in: wp.array[wp.mat33],
    contact_efc_address_in: wp.array2d[int],
    efc_pos_in: wp.array2d[float],
    efc_margin_in: wp.array2d[float],
    efc_vel_in: wp.array2d[float],
    nacon_in: wp.array[int],
    prox_zcache_key_in: wp.array[wp.int64],
    prox_zcache_val_in: wp.array[int],
    prox_zcache_z_in: wp.array[wp.vec3],
    prox_zcache_pos_in: wp.array[wp.vec3],
    prox_zcache_n_in: wp.array[int],
    qvel_in: wp.array2d[float],
    qacc_in: wp.array2d[float],
    cdof_in: wp.array2d[wp.spatial_vector],
    subtree_com_in: wp.array2d[wp.vec3],
    contact_group_adr_in: wp.array[int],
    contact_group_num_in: wp.array[int],
    group_member_pos_in: wp.array[wp.vec3],
    group_member_frame_in: wp.array[wp.mat33],
    group_member_dist_in: wp.array[float],
    group_member_weight_in: wp.array[float],
    group_member_slip_in: wp.array[float],
    # In:
    condim_in: wp.array[int],
    worldid_in: wp.array[int],
    geom_in: wp.array[wp.vec2i],
    friction_in: wp.array[vec5],
    type_in: wp.array[int],
    jaref_in: wp.array2d[float],
    solref_in: wp.array[wp.vec2],
    solimp_in: wp.array[vec5],
    # Data out:
    efc_D_out: wp.array2d[float],
    efc_aref_out: wp.array2d[float],
    # Out:
    prox_z_out: wp.array2d[float],
    prox_aref0_out: wp.array2d[float],
    prox_gamma_out: wp.array[float],
    prox_rho_i_out: wp.array[float],
    prox_bdamp_out: wp.array[float],
  ):
    """Initializes one contact: prox weight D, aref0, warm (sigma, z), aref rewrite.

    Model fields are per-world broadcastable, shape (1 or nworld,), indexed with
    worldid % shape[0]. Contact arrays have shape (naconmax,); efc arrays have
    shape (nworld, njmax) and are indexed [worldid, efcid].

    Args:
      opt_timestep: (1 or nworld,) timestep dt.
      opt_impratio_invsqrt: (1 or nworld,) 1/sqrt(impratio); mu_eff = friction[0] * this.
      opt_prox_rho: (1 or nworld,) rho_hat; prox weight rho_i = rho_hat * dt * invweight.
      opt_prox_baumgarte_alpha: (1 or nworld,) depenetration gain alpha (Baumgarte bias only).
      opt_prox_baumgarte_slop: (1 or nworld,) penetration the Baumgarte bias leaves alone.
      opt_prox_z_carry: scalar carried-anchor scale (1.0 = full warm start).
      body_invweight0: (1 or nworld, nbody) vec2 (translational, rotational) invweights.
      geom_bodyid: (ngeom,) body id of each geom.
      geom_rbound: (1 or nworld, ngeom) geom bounding radii (0 for planes/hfields).
      contact_pos_in: (naconmax,) vec3 world-frame contact positions.
      contact_frame_in: (naconmax,) mat33 contact frames, rows (normal, t1, t2).
      contact_efc_address_in: (naconmax, ncondmax) efc row id of each contact row.
      efc_pos_in: (nworld, njmax) constraint positions (contact: signed distance).
      efc_margin_in: (nworld, njmax) contact margins.
      efc_vel_in: (nworld, njmax) constraint velocities J @ qvel.
      nacon_in: (1,) number of active contacts.
      prox_zcache_key_in: (naconmax,) int64 sorted keys of the previous step's cache.
      prox_zcache_val_in: (naconmax,) cache slot of each sorted key.
      prox_zcache_z_in: (naconmax,) vec3 cached world-frame anchor forces z.
      prox_zcache_pos_in: (naconmax,) vec3 cached contact positions.
      prox_zcache_n_in: (1,) number of valid cache entries.
      qvel_in, qacc_in: (nworld, nv) velocities and the warm-started accelerations, for a
        GROUP contact's members' sigma.
      cdof_in, subtree_com_in: (nworld, nv) dof motions and (nworld, nbody) subtree COMs.
      contact_group_adr_in, contact_group_num_in, group_member_*_in: the GROUP contacts'
        members (io.alloc_contact_groups); empty without groups.
      condim_in: (naconmax,) contact dimensionalities.
      worldid_in: (naconmax,) world id of each contact.
      geom_in: (naconmax,) vec2i geom pairs.
      friction_in: (naconmax,) vec5 friction; [0] is the sliding coefficient.
      type_in: (naconmax,) ContactType flags.
      jaref_in: (nworld, njmax) J @ qacc - aref computed against the ORIGINAL aref,
        so jaref + aref = J @ qacc stays valid after aref is overwritten below.
      efc_D_out: (nworld, njmax) constraint weights; contact rows <- D = dt/rho_i.
      efc_aref_out: (nworld, njmax) reference accelerations; read as MuJoCo's
        original aref, overwritten with aref0 - Gamma(sigma)/dt + z/D.
      prox_z_out: (nworld, njmax) proximal anchors z, efc-row layout.
      prox_aref0_out: (nworld, njmax) stabilization-only reference aref0.
      prox_gamma_out: (naconmax,) De Saxce corrections Gamma_n = mu*||sigma_t||.
    """
    conid = wp.tid()  # scalar: contact index

    if conid >= nacon_in[0]:
      return

    if not (type_in[conid] & ContactType.CONSTRAINT):
      return

    condim = condim_in[conid]  # scalar: contact dimensionality (1 or 3)
    efcid0 = contact_efc_address_in[conid, 0]  # scalar: efc row of the normal component
    if efcid0 < 0:
      return

    efcid1 = -1  # scalar: efc row of tangent 1 (-1 if frictionless)
    efcid2 = -1  # scalar: efc row of tangent 2 (-1 if frictionless)
    if condim >= 3:
      efcid1 = contact_efc_address_in[conid, 1]
      efcid2 = contact_efc_address_in[conid, 2]
      if efcid1 < 0 or efcid2 < 0:
        return

    worldid = worldid_in[conid]  # scalar: world of this contact
    timestep = opt_timestep[worldid % opt_timestep.shape[0]]  # scalar: dt
    rho_hat = opt_prox_rho[worldid % opt_prox_rho.shape[0]]  # scalar: normalized prox rate

    # prox weight: D = dt / rho_i with rho_i = rho_hat * dt * invweight
    geom = geom_in[conid]  # vec2i: geom pair
    body_invweight0_id = worldid % body_invweight0.shape[0]  # scalar: model world index
    invweight = body_invweight0[body_invweight0_id, geom_bodyid[geom[0]]][0]  # scalar: translational invweight, body 0
    invweight += body_invweight0[body_invweight0_id, geom_bodyid[geom[1]]][0]  # scalar: + body 1
    D = 1.0 / wp.max(rho_hat * invweight, MINVAL)  # scalar: prox weight dt/rho_i

    mu_eff = 0.0  # scalar: effective isotropic friction of the elliptic cone
    if condim >= 3:
      mu_eff = friction_in[conid][0] * opt_impratio_invsqrt[worldid % opt_impratio_invsqrt.shape[0]]

    is_group = False  # bool: a GROUP contact, whose terms come from its members
    if wp.static(GROUPS):
      is_group = (type_in[conid] & ContactType.GROUP) != 0

    # De Saxce term from the warm-started sigma = J*(qvel + dt*qacc) = vel + dt*(jaref + aref)
    # (jaref here was computed against the original aref, so the sum is aref-independent)
    gamma = 0.0  # scalar: De Saxce correction Gamma_n = mu*||sigma_t||
    if wp.static(DESAXCE):
      if wp.static(GROUPS):
        if is_group:
          gamma = _group_gamma(
            body_rootid,
            body_weldid,
            body_dofnum,
            body_dofadr,
            dof_parentid,
            geom_bodyid,
            qvel_in,
            qacc_in,
            cdof_in,
            subtree_com_in,
            contact_group_adr_in,
            contact_group_num_in,
            group_member_pos_in,
            group_member_frame_in,
            group_member_weight_in,
            group_member_slip_in,
            worldid,
            conid,
            geom_in[conid],
            timestep,
            mu_eff,
            condim == 1,
          )
      if condim >= 3 and not is_group:
        sig_t1 = efc_vel_in[worldid, efcid1] + timestep * (jaref_in[worldid, efcid1] + efc_aref_out[worldid, efcid1])  # scalar
        sig_t2 = efc_vel_in[worldid, efcid2] + timestep * (jaref_in[worldid, efcid2] + efc_aref_out[worldid, efcid2])  # scalar
        gamma = mu_eff * wp.sqrt(sig_t1 * sig_t1 + sig_t2 * sig_t2)
    prox_gamma_out[conid] = gamma

    # cross-step anchor warm start: nearest cached contact of the same geom pair
    # (within a bound-relative radius), rotated through world frame, cone-projected; a group
    # starts cold
    z0 = wp.vec3(0.0)  # vec3: warm-started anchor in contact coordinates (n, t1, t2)
    if wp.static(CARRY) and not (type_in[conid] & ContactType.NOCARRY) and not is_group:
      key = _pack_key(worldid, geom[0], geom[1])  # scalar int64: cache lookup key
      rbound_id = worldid % geom_rbound.shape[0]  # scalar: model world index
      # planes and hfields have rbound 0: fall back to the other geom's bound
      r0 = geom_rbound[rbound_id, geom[0]]  # scalar: bounding radius of geom 0
      r1 = geom_rbound[rbound_id, geom[1]]  # scalar: bounding radius of geom 1
      radius = 0.3 * wp.min(wp.where(r0 > 0.0, r0, r1), wp.where(r1 > 0.0, r1, r0))  # scalar: match radius
      slot = _zcache_match(
        key,
        contact_pos_in[conid],
        radius,
        prox_zcache_key_in,
        prox_zcache_val_in,
        prox_zcache_pos_in,
        prox_zcache_n_in[0],
      )  # scalar: matched cache slot, or -1
      if slot >= 0:
        z_local = contact_frame_in[conid] @ prox_zcache_z_in[slot]  # vec3: cached anchor in the NEW contact frame
        z0 = opt_prox_z_carry * _project_cone(z_local, mu_eff)

    if wp.static(SOFT):
      # MuJoCo's own rows, its softness R = 1/D kept beside the proximal weight rho_i, and its
      # reference aref_mj the stabilization: the row's force f = (aref_mj - J a + rho_i z) /
      # (R + rho_i), so at the fixed point z = f MuJoCo's soft solution. The De Saxce shift of
      # the normal velocity enters through the row's own damping: -b Gamma.
      rho_i = rho_hat * invweight  # scalar: proximal weight (0: no proximal term)
      bdamp = _contact_kbimp(opt_disableflags, timestep, solref_in[conid], solimp_in[conid], 0.0)[1]  # scalar: damping b
      prox_rho_i_out[conid] = rho_i
      prox_bdamp_out[conid] = bdamp
      arefm_0 = efc_aref_out[worldid, efcid0]  # scalar: MuJoCo's normal-row aref
      prox_aref0_out[worldid, efcid0] = arefm_0
      prox_z_out[worldid, efcid0] = z0[0]
      efc_D_out[worldid, efcid0] = 1.0 / (1.0 / efc_D_out[worldid, efcid0] + rho_i)
      efc_aref_out[worldid, efcid0] = arefm_0 - bdamp * gamma + rho_i * z0[0]
      if condim >= 3:
        arefm_1 = efc_aref_out[worldid, efcid1]  # scalar: MuJoCo's tangent-1 aref
        arefm_2 = efc_aref_out[worldid, efcid2]  # scalar: MuJoCo's tangent-2 aref
        prox_aref0_out[worldid, efcid1] = arefm_1
        prox_aref0_out[worldid, efcid2] = arefm_2
        prox_z_out[worldid, efcid1] = z0[1]
        prox_z_out[worldid, efcid2] = z0[2]
        efc_D_out[worldid, efcid1] = 1.0 / (1.0 / efc_D_out[worldid, efcid1] + rho_i)
        efc_D_out[worldid, efcid2] = 1.0 / (1.0 / efc_D_out[worldid, efcid2] + rho_i)
        efc_aref_out[worldid, efcid1] = arefm_1 + rho_i * z0[1]
        efc_aref_out[worldid, efcid2] = arefm_2 + rho_i * z0[2]
      return

    # stabilization-only reference acceleration aref0: the normal row's target velocity is
    # -bias_n, the Baumgarte bias or none (the velocity-level target)
    bias_n = 0.0  # scalar: stabilization bias velocity of the normal row
    if wp.static(BAUMGARTE):
      alpha = opt_prox_baumgarte_alpha[worldid % opt_prox_baumgarte_alpha.shape[0]]  # scalar: depenetration gain
      slop = opt_prox_baumgarte_slop[worldid % opt_prox_baumgarte_slop.shape[0]]  # scalar: penetration left alone
      pos = efc_pos_in[worldid, efcid0] - efc_margin_in[worldid, efcid0]  # scalar: margin-corrected penetration
      # Inside the slop a resting contact holds its depth instead of being pushed out to
      # zero, where it would vanish, drop a step and come back: a contact that flickers.
      bias_n = alpha * wp.min(0.0, (pos + slop) / timestep) + wp.max(0.0, pos / timestep)  # scalar: Baumgarte bias velocity
      if wp.static(GROUPS):
        if is_group:
          bias_n = _group_bias(
            contact_group_adr_in,
            contact_group_num_in,
            group_member_dist_in,
            group_member_weight_in,
            conid,
            efc_margin_in[worldid, efcid0],
            alpha,
            slop,
            timestep,
          )
    aref0_0 = -(efc_vel_in[worldid, efcid0] + bias_n) / timestep  # scalar: normal-row aref0

    prox_aref0_out[worldid, efcid0] = aref0_0
    prox_z_out[worldid, efcid0] = z0[0]
    efc_D_out[worldid, efcid0] = D
    efc_aref_out[worldid, efcid0] = aref0_0 - gamma / timestep + z0[0] / D

    if condim >= 3:
      # tangent rows: the pure -v_t/dt target. A tangential bias inside the exact De Saxce
      # complementarity misaligns the slip direction and can inject energy (cf. Tsounis et al.,
      # arXiv:2504.19771, Lemma A.2 and Section V-D).
      aref0_1 = -efc_vel_in[worldid, efcid1] / timestep  # scalar: pure -v_t/dt target (no tangential bias)
      aref0_2 = -efc_vel_in[worldid, efcid2] / timestep  # scalar
      prox_aref0_out[worldid, efcid1] = aref0_1
      prox_aref0_out[worldid, efcid2] = aref0_2
      prox_z_out[worldid, efcid1] = z0[1]
      prox_z_out[worldid, efcid2] = z0[2]
      efc_D_out[worldid, efcid1] = D
      efc_D_out[worldid, efcid2] = D
      efc_aref_out[worldid, efcid1] = aref0_1 + z0[1] / D
      efc_aref_out[worldid, efcid2] = aref0_2 + z0[2] / D

  return kernel


# Lodos's proximal solve.
@wp.kernel(grid_stride=True)
def _prox_cache_fill(
  # Data in:
  contact_pos_in: wp.array[wp.vec3],
  contact_frame_in: wp.array[wp.mat33],
  contact_efc_address_in: wp.array2d[int],
  nacon_in: wp.array[int],
  # In:
  condim_in: wp.array[int],
  worldid_in: wp.array[int],
  geom_in: wp.array[wp.vec2i],
  type_in: wp.array[int],
  prox_z_in: wp.array2d[float],
  # Data out:
  prox_zcache_key_out: wp.array[wp.int64],
  prox_zcache_val_out: wp.array[int],
  prox_zcache_z_out: wp.array[wp.vec3],
  prox_zcache_pos_out: wp.array[wp.vec3],
):
  """Persists the converged anchors z into the world-frame cross-step cache.

  Launched with dim = naconmax; slots without an active contact keep the sort
  sentinel key so the subsequent radix sort moves them past the valid run.

  Args:
    contact_pos_in: (naconmax,) vec3 world-frame contact positions.
    contact_frame_in: (naconmax,) mat33 contact frames, rows (normal, t1, t2).
    contact_efc_address_in: (naconmax, ncondmax) efc row id of each contact row.
    nacon_in: (1,) number of active contacts.
    condim_in: (naconmax,) contact dimensionalities.
    worldid_in: (naconmax,) world id of each contact.
    geom_in: (naconmax,) vec2i geom pairs.
    type_in: (naconmax,) ContactType flags.
    prox_z_in: (nworld, njmax) converged proximal anchors z, efc-row layout.
    prox_zcache_key_out: (naconmax,) int64 pair keys (sentinel where inactive).
    prox_zcache_val_out: (naconmax,) identity permutation, consumed by the sort.
    prox_zcache_z_out: (naconmax,) vec3 anchors rotated to the world frame.
    prox_zcache_pos_out: (naconmax,) vec3 contact positions for position matching.
  """
  conid = wp.tid()  # scalar: contact (= cache slot) index

  prox_zcache_val_out[conid] = conid
  prox_zcache_key_out[conid] = _KEY_SENTINEL

  if conid >= nacon_in[0]:
    return
  if not (type_in[conid] & ContactType.CONSTRAINT) or (type_in[conid] & ContactType.NOCARRY):
    return
  if type_in[conid] & ContactType.GROUP:  # a group starts cold, and is no anchor for its pair's contacts
    return
  efcid0 = contact_efc_address_in[conid, 0]  # scalar: efc row of the normal component
  if efcid0 < 0:
    return

  worldid = worldid_in[conid]  # scalar: world of this contact
  z_local = wp.vec3(prox_z_in[worldid, efcid0], 0.0, 0.0)  # vec3: anchor in contact coordinates
  if condim_in[conid] >= 3:
    efcid1 = contact_efc_address_in[conid, 1]  # scalar: efc row of tangent 1
    efcid2 = contact_efc_address_in[conid, 2]  # scalar: efc row of tangent 2
    if efcid1 >= 0 and efcid2 >= 0:
      z_local = wp.vec3(z_local[0], prox_z_in[worldid, efcid1], prox_z_in[worldid, efcid2])

  geom = geom_in[conid]  # vec2i: geom pair
  prox_zcache_key_out[conid] = _pack_key(worldid, geom[0], geom[1])
  prox_zcache_z_out[conid] = wp.transpose(contact_frame_in[conid]) @ z_local
  prox_zcache_pos_out[conid] = contact_pos_in[conid]


# Lodos's proximal solve.
@wp.kernel(grid_stride=True)
def _prox_solve_init_efc(
  # In:
  prox_done_in: wp.array[bool],
  # Data out:
  solver_niter_out: wp.array[int],
  # Out:
  ctx_search_dot_out: wp.array[float],
  ctx_done_out: wp.array[bool],
):
  """Resets the per-world Newton state before an inner solve; keeps frozen worlds done.

  Args:
    prox_done_in: (nworld,) sticky per-world prox convergence flags.
    solver_niter_out: (nworld,) Newton iteration counters, reset to 0.
    ctx_search_dot_out: (nworld,) search-direction dot products, reset to 0.
    ctx_done_out: (nworld,) Newton done flags, preset to prox_done_in.
  """
  worldid = wp.tid()  # scalar: world index
  solver_niter_out[worldid] = 0
  ctx_search_dot_out[worldid] = 0.0
  # prox-converged worlds stay frozen: the done-gated solver kernels skip them and
  # _solve_done never counts them against nsolving (which starts at prox_nactive)
  ctx_done_out[worldid] = prox_done_in[worldid]


# Lodos's proximal solve.
@wp.kernel(grid_stride=True)
def _prox_zero_residuals(
  # Out:
  prox_dz_out: wp.array[float],
  prox_znorm_out: wp.array[float],
  prox_dsigma_out: wp.array[float],
  prox_signorm_out: wp.array[float],
):
  """Zeroes the per-world prox residual accumulators (atomic-max targets).

  Args:
    prox_dz_out: (nworld,) max |f* - z| over the world's contact rows.
    prox_znorm_out: (nworld,) max |f*| over the world's contact rows.
    prox_dsigma_out: (nworld,) max |Gamma_new - Gamma| over the world's contacts.
    prox_signorm_out: (nworld,) max Gamma_new over the world's contacts.
  """
  worldid = wp.tid()  # scalar: world index
  prox_dz_out[worldid] = 0.0
  prox_znorm_out[worldid] = 0.0
  prox_dsigma_out[worldid] = 0.0
  prox_signorm_out[worldid] = 0.0


# Lodos's proximal solve.
@cache_kernel
def _prox_update(soft: bool, groups: bool):
  """Builds the per-contact update kernel, specialized on soft (opt.prox_soft) and groups (_prox_init)."""
  SOFT = soft  # bool: compile-time MuJoCo-softness switch (must be a local for wp.static)
  GROUPS = groups  # bool: compile-time contact-group switch

  @wp.kernel(module="unique", enable_backward=False, grid_stride=True)
  def kernel(
    # Model:
    opt_timestep: wp.array[float],
    opt_impratio_invsqrt: wp.array[float],
    opt_prox_sigma_relax: wp.array[float],
    body_rootid: wp.array[int],
    body_weldid: wp.array[int],
    body_dofnum: wp.array[int],
    body_dofadr: wp.array[int],
    dof_parentid: wp.array[int],
    geom_bodyid: wp.array[int],
    # Data in:
    qvel_in: wp.array2d[float],
    qacc_in: wp.array2d[float],
    cdof_in: wp.array2d[wp.spatial_vector],
    subtree_com_in: wp.array2d[wp.vec3],
    contact_efc_address_in: wp.array2d[int],
    efc_D_in: wp.array2d[float],
    efc_vel_in: wp.array2d[float],
    efc_force_in: wp.array2d[float],
    nacon_in: wp.array[int],
    contact_group_adr_in: wp.array[int],
    contact_group_num_in: wp.array[int],
    group_member_pos_in: wp.array[wp.vec3],
    group_member_frame_in: wp.array[wp.mat33],
    group_member_weight_in: wp.array[float],
    group_member_slip_in: wp.array[float],
    # In:
    condim_in: wp.array[int],
    worldid_in: wp.array[int],
    geom_in: wp.array[wp.vec2i],
    friction_in: wp.array[vec5],
    type_in: wp.array[int],
    jaref_in: wp.array2d[float],
    prox_aref0_in: wp.array2d[float],
    prox_done_in: wp.array[bool],
    update_sigma: int,
    prox_rho_i_in: wp.array[float],
    prox_bdamp_in: wp.array[float],
    # Data out:
    efc_aref_out: wp.array2d[float],
    # Out:
    prox_z_out: wp.array2d[float],
    prox_gamma_out: wp.array[float],
    prox_dz_out: wp.array[float],
    prox_znorm_out: wp.array[float],
    prox_dsigma_out: wp.array[float],
    prox_signorm_out: wp.array[float],
  ):
    """Prox/De Saxce update of one contact after an inner Newton solve.

    The contact tutorial's Algorithm 13, lines 15 and 16: z <- f* always; sigma (through its only use,
    Gamma_n = mu*||sigma_t||) only when update_sigma is set, i.e. at the end of each
    outer round. Finishes by rewriting aref = aref0 - Gamma/dt + z/D for the next solve.

    Args:
      opt_timestep: (1 or nworld,) timestep dt.
      opt_impratio_invsqrt: (1 or nworld,) 1/sqrt(impratio); mu_eff = friction[0] * this.
      opt_prox_sigma_relax: (1 or nworld,) relaxation of the Gamma update (1 = full step).
      body_*, dof_parentid, geom_bodyid: the model's trees, for a GROUP contact's members.
      qvel_in, qacc_in: (nworld, nv) velocities and the last inner solve's accelerations.
      cdof_in, subtree_com_in: (nworld, nv) dof motions and (nworld, nbody) subtree COMs.
      contact_efc_address_in: (naconmax, ncondmax) efc row id of each contact row.
      efc_D_in: (nworld, njmax) constraint weights (contact rows hold D = dt/rho_i).
      efc_vel_in: (nworld, njmax) constraint velocities J @ qvel.
      efc_force_in: (nworld, njmax) constraint forces f* of the last inner solve.
      nacon_in: (1,) number of active contacts.
      condim_in: (naconmax,) contact dimensionalities.
      worldid_in: (naconmax,) world id of each contact.
      contact_group_*_in, group_member_*_in: the GROUP contacts' members; empty without groups.
      geom_in: (naconmax,) vec2i geom pairs.
      friction_in: (naconmax,) vec5 friction; [0] is the sliding coefficient.
      type_in: (naconmax,) ContactType flags.
      jaref_in: (nworld, njmax) J @ qacc - aref of the last solve (jaref + aref = J @ qacc).
      prox_aref0_in: (nworld, njmax) stabilization-only reference aref0.
      prox_done_in: (nworld,) sticky prox convergence flags; frozen worlds are skipped.
      update_sigma: scalar 0/1; 1 = also take the De Saxce sigma step (outer-round end).
      efc_aref_out: (nworld, njmax) reference accelerations, rewritten in place.
      prox_z_out: (nworld, njmax) proximal anchors, read as old z / overwritten with f*.
      prox_gamma_out: (naconmax,) De Saxce corrections Gamma_n, updated if update_sigma.
      prox_dz_out: (nworld,) atomic-max residual |f* - z|.
      prox_znorm_out: (nworld,) atomic-max scale |f*|.
      prox_dsigma_out: (nworld,) atomic-max residual |Gamma_new - Gamma|.
      prox_signorm_out: (nworld,) atomic-max scale Gamma_new.
    """
    conid = wp.tid()  # scalar: contact index

    if conid >= nacon_in[0]:
      return

    if not (type_in[conid] & ContactType.CONSTRAINT):
      return

    condim = condim_in[conid]  # scalar: contact dimensionality (1 or 3)
    efcid0 = contact_efc_address_in[conid, 0]  # scalar: efc row of the normal component
    if efcid0 < 0:
      return

    efcid1 = -1  # scalar: efc row of tangent 1 (-1 if frictionless)
    efcid2 = -1  # scalar: efc row of tangent 2 (-1 if frictionless)
    if condim >= 3:
      efcid1 = contact_efc_address_in[conid, 1]
      efcid2 = contact_efc_address_in[conid, 2]
      if efcid1 < 0 or efcid2 < 0:
        return

    worldid = worldid_in[conid]  # scalar: world of this contact

    # frozen world: keep (z, sigma, aref) at their converged values
    if prox_done_in[worldid]:
      return

    timestep = opt_timestep[worldid % opt_timestep.shape[0]]  # scalar: dt

    # proximal anchor update: z <- f* of the last inner solve
    f0 = efc_force_in[worldid, efcid0]  # scalar: normal contact force
    dz = wp.abs(f0 - prox_z_out[worldid, efcid0])  # scalar: anchor change |f* - z| (this contact)
    znorm = wp.abs(f0)  # scalar: anchor scale |f*| (this contact)
    f1 = 0.0  # scalar: tangent-1 contact force
    f2 = 0.0  # scalar: tangent-2 contact force
    if condim >= 3:
      f1 = efc_force_in[worldid, efcid1]
      f2 = efc_force_in[worldid, efcid2]
      dz = wp.max(dz, wp.abs(f1 - prox_z_out[worldid, efcid1]))
      dz = wp.max(dz, wp.abs(f2 - prox_z_out[worldid, efcid2]))
      znorm = wp.max(znorm, wp.max(wp.abs(f1), wp.abs(f2)))
    wp.atomic_max(prox_dz_out, worldid, dz)
    wp.atomic_max(prox_znorm_out, worldid, znorm)
    prox_z_out[worldid, efcid0] = f0
    if condim >= 3:
      prox_z_out[worldid, efcid1] = f1
      prox_z_out[worldid, efcid2] = f2

    is_group = False  # bool: a GROUP contact, whose De Saxce term comes from its members
    if wp.static(GROUPS):
      is_group = (type_in[conid] & ContactType.GROUP) != 0

    # De Saxce update: sigma = J*(qvel + dt*qacc) at the current inner solution
    gamma = prox_gamma_out[conid]  # scalar: current De Saxce correction Gamma_n
    if update_sigma and (condim >= 3 or is_group):
      mu_eff = 0.0  # scalar: friction
      if condim >= 3:
        mu_eff = friction_in[conid][0] * opt_impratio_invsqrt[worldid % opt_impratio_invsqrt.shape[0]]
      target = 0.0  # scalar: the term at the current iterate
      if wp.static(GROUPS):
        if is_group:
          target = _group_gamma(
            body_rootid,
            body_weldid,
            body_dofnum,
            body_dofadr,
            dof_parentid,
            geom_bodyid,
            qvel_in,
            qacc_in,
            cdof_in,
            subtree_com_in,
            contact_group_adr_in,
            contact_group_num_in,
            group_member_pos_in,
            group_member_frame_in,
            group_member_weight_in,
            group_member_slip_in,
            worldid,
            conid,
            geom_in[conid],
            timestep,
            mu_eff,
            condim == 1,
          )
      if not is_group:
        sig_t1 = efc_vel_in[worldid, efcid1] + timestep * (jaref_in[worldid, efcid1] + efc_aref_out[worldid, efcid1])  # scalar
        sig_t2 = efc_vel_in[worldid, efcid2] + timestep * (jaref_in[worldid, efcid2] + efc_aref_out[worldid, efcid2])  # scalar
        target = mu_eff * wp.sqrt(sig_t1 * sig_t1 + sig_t2 * sig_t2)
      relax = opt_prox_sigma_relax[worldid % opt_prox_sigma_relax.shape[0]]  # scalar: relaxation factor
      gamma_new = gamma + relax * (target - gamma)  # scalar: relaxed Gamma
      wp.atomic_max(prox_dsigma_out, worldid, wp.abs(gamma_new - gamma))
      wp.atomic_max(prox_signorm_out, worldid, gamma_new)
      prox_gamma_out[conid] = gamma_new
      gamma = gamma_new

    if wp.static(SOFT):
      # rewrite aref = aref_mj - b Gamma + rho_i z (MuJoCo's rows, opt.prox_soft)
      rho_i = prox_rho_i_in[conid]  # scalar: proximal weight
      efc_aref_out[worldid, efcid0] = prox_aref0_in[worldid, efcid0] - prox_bdamp_in[conid] * gamma + rho_i * f0
      if condim >= 3:
        efc_aref_out[worldid, efcid1] = prox_aref0_in[worldid, efcid1] + rho_i * f1
        efc_aref_out[worldid, efcid2] = prox_aref0_in[worldid, efcid2] + rho_i * f2
      return

    # rewrite aref = aref0 - Gamma(sigma)/dt + z/D
    D0 = efc_D_in[worldid, efcid0]  # scalar: prox weight of the normal row
    efc_aref_out[worldid, efcid0] = prox_aref0_in[worldid, efcid0] - gamma / timestep + f0 / D0
    if condim >= 3:
      efc_aref_out[worldid, efcid1] = prox_aref0_in[worldid, efcid1] + f1 / efc_D_in[worldid, efcid1]
      efc_aref_out[worldid, efcid2] = prox_aref0_in[worldid, efcid2] + f2 / efc_D_in[worldid, efcid2]

  return kernel


# Lodos's proximal solve.
@wp.kernel(grid_stride=True)
def _prox_check(
  # Model:
  opt_prox_tol_z: wp.array[float],
  opt_prox_tol_sigma: wp.array[float],
  # Data in:
  solver_niter_in: wp.array[int],
  # In:
  prox_dz_in: wp.array[float],
  prox_znorm_in: wp.array[float],
  prox_dsigma_in: wp.array[float],
  prox_signorm_in: wp.array[float],
  update_sigma: int,
  z_only: int,
  # Data out:
  prox_rounds_out: wp.array[int],
  # Out:
  prox_done_out: wp.array[bool],
  prox_niter_out: wp.array[int],
  prox_nactive_out: wp.array[int],
):
  """Per-world convergence test: freezes worlds whose (z, sigma) fixed point is reached.

  Args:
    opt_prox_tol_z: (1 or nworld,) relative tolerance on the anchor update.
    opt_prox_tol_sigma: (1 or nworld,) relative tolerance on the Gamma update.
    solver_niter_in: (nworld,) Newton iterations of the last inner solve.
    prox_dz_in: (nworld,) max |f* - z| of the last _prox_update.
    prox_znorm_in: (nworld,) max |f*| of the last _prox_update.
    prox_dsigma_in: (nworld,) max |Gamma_new - Gamma| of the last _prox_update.
    prox_signorm_in: (nworld,) max Gamma_new of the last _prox_update.
    update_sigma: scalar 0/1; freezing is only tested at outer-round ends, unless z_only.
    z_only: scalar 0/1; sigma is not refreshed within the step (one outer round, or no De
      Saxce term): z alone is tested, every round.
    prox_rounds_out: (nworld,) rounds the world has run in this step, this one included.
    prox_done_out: (nworld,) sticky convergence flags, set when both residuals pass.
    prox_niter_out: (nworld,) accumulated Newton iterations across the whole step.
    prox_nactive_out: (1,) count of still-active worlds, atomically decremented.
  """
  worldid = wp.tid()  # scalar: world index

  # total Newton iterations across the inner solves (the refresh resets solver_niter)
  prox_niter_out[worldid] += solver_niter_in[worldid]

  # sticky: a frozen world stays frozen for the rest of the step
  if prox_done_out[worldid]:
    return
  prox_rounds_out[worldid] += 1

  if not update_sigma and not z_only:
    return

  tol_z = opt_prox_tol_z[worldid % opt_prox_tol_z.shape[0]]  # scalar: anchor tolerance
  tol_sigma = opt_prox_tol_sigma[worldid % opt_prox_tol_sigma.shape[0]]  # scalar: Gamma tolerance
  z_ok = prox_dz_in[worldid] <= tol_z * (1.0 + prox_znorm_in[worldid])  # bool: anchor fixed point reached
  sigma_ok = True  # bool: sigma fixed point reached (trivially, where sigma did not move)
  if update_sigma:
    sigma_ok = prox_dsigma_in[worldid] <= tol_sigma * (1.0 + prox_signorm_in[worldid])
  if z_ok and sigma_ok:
    prox_done_out[worldid] = True
    wp.atomic_add(prox_nactive_out, 0, -1)


# Lodos's proximal solve.
@cache_kernel
def _prox_jaref_kernel(is_sparse: bool, nv: int, dofs_per_thread: int):
  """Builds the prox-done-gated Jaref kernel (solver's variant minus the compact path).

  Frozen worlds keep their Jaref: their qacc and aref are both frozen, so the
  recompute is pure waste there and every thread of a done world exits at once.

  Args:
    is_sparse: bool; sparse constraint Jacobian layout, compiled in.
    nv: scalar number of dofs, compiled in.
    dofs_per_thread: scalar dofs accumulated per thread (dense split path).

  Returns:
    A kernel over (nworld, njmax, threads_per_efc) writing ctx_Jaref_out.
  """

  @wp.kernel(module="unique", enable_backward=False, grid_stride=True)
  def kernel(
    # Data in:
    nefc_in: wp.array[int],
    qacc_in: wp.array2d[float],
    efc_J_rownnz_in: wp.array2d[int],
    efc_J_rowadr_in: wp.array2d[int],
    efc_J_colind_in: wp.array3d[int],
    efc_J_in: wp.array3d[float],
    efc_aref_in: wp.array2d[float],
    # In:
    prox_done_in: wp.array[bool],
    # Out:
    ctx_Jaref_out: wp.array2d[float],
  ):
    """Computes Jaref[worldid, efcid] = (J @ qacc - aref)[efcid] for ACTIVE worlds.

    Args:
      nefc_in: (nworld,) active constraint-row counts.
      qacc_in: (nworld, nv) accelerations.
      efc_J_*: constraint Jacobian in the model's sparse or dense layout.
      efc_aref_in: (nworld, njmax) reference accelerations.
      prox_done_in: (nworld,) sticky prox convergence flags (done worlds skip).
      ctx_Jaref_out: (nworld, njmax) output residuals.
    """
    worldid, efcid, dofstart = wp.tid()  # scalars: world, efc row, dense dof block

    if prox_done_in[worldid]:
      return
    if efcid >= nefc_in[worldid]:
      return

    jaref = float(0.0)  # scalar: running dot product (J @ qacc)[efcid]
    if wp.static(is_sparse):
      rownnz = efc_J_rownnz_in[worldid, efcid]  # scalar: nonzeros in this row
      rowadr = efc_J_rowadr_in[worldid, efcid]  # scalar: row start in the sparse buffer
      for i in range(rownnz):
        sparseid = rowadr + i  # scalar: sparse entry index
        jaref += efc_J_in[worldid, 0, sparseid] * qacc_in[worldid, efc_J_colind_in[worldid, 0, sparseid]]
      ctx_Jaref_out[worldid, efcid] = jaref - efc_aref_in[worldid, efcid]
    else:
      if wp.static(dofs_per_thread >= nv):
        for i in range(wp.static(min(dofs_per_thread, nv))):
          jaref += efc_J_in[worldid, efcid, i] * qacc_in[worldid, i]
        ctx_Jaref_out[worldid, efcid] = jaref - efc_aref_in[worldid, efcid]
      else:
        for i in range(wp.static(dofs_per_thread)):
          ii = dofstart * wp.static(dofs_per_thread) + i  # scalar: dof index of this thread's block
          if ii < nv:
            jaref += efc_J_in[worldid, efcid, ii] * qacc_in[worldid, ii]
        if dofstart == 0:
          wp.atomic_add(ctx_Jaref_out, worldid, efcid, jaref - efc_aref_in[worldid, efcid])
        else:
          wp.atomic_add(ctx_Jaref_out, worldid, efcid, jaref)

  return kernel


# Lodos's proximal solve.
@wp.kernel(grid_stride=True)
def _prox_jaref_zero(
  # In:
  prox_done_in: wp.array[bool],
  # Out:
  ctx_Jaref_out: wp.array2d[float],
):
  """Zeroes the Jaref rows of ACTIVE worlds before a multi-thread-per-row accumulate.

  Args:
    prox_done_in: (nworld,) sticky prox convergence flags; frozen rows keep their value.
    ctx_Jaref_out: (nworld, njmax) accumulation target.
  """
  worldid, efcid = wp.tid()  # scalars: world, efc row
  if prox_done_in[worldid]:
    return
  ctx_Jaref_out[worldid, efcid] = 0.0


# Lodos's proximal solve.
def _init_jaref(m: Model, d: Data, ctx: SolverContext):
  """Recomputes ctx.Jaref = efc_J @ qacc - efc_aref against the CURRENT efc arrays.

  Mirrors the Jaref launch inside init_context, without the compact path,
  gated on ctx.prox_done (frozen worlds keep their Jaref unchanged). Callers must
  therefore zero ctx.prox_done before the first call of a step.

  Args:
    m: model; consumes is_sparse and nv for the launch geometry.
    d: data; consumes nefc (nworld,), qacc (nworld, nv), the efc Jacobian arrays,
      and efc.aref (nworld, njmax).
    ctx: solver context; writes ctx.Jaref (nworld, njmax), reads ctx.prox_done.
  """
  if m.is_sparse:
    dofs_per_thread = m.nv  # scalar: dofs each thread accumulates
    threads_per_efc = 1  # scalar: threads cooperating on one efc row
  elif m.nv > 50:
    dofs_per_thread = 20
    threads_per_efc = ceil(m.nv / dofs_per_thread)
  else:
    dofs_per_thread = 50
    threads_per_efc = ceil(m.nv / dofs_per_thread)

  if threads_per_efc > 1:
    wp.launch(_prox_jaref_zero, dim=(d.nworld, d.njmax), inputs=[ctx.prox_done], outputs=[ctx.Jaref])

  wp.launch(
    _prox_jaref_kernel(m.is_sparse, m.nv, dofs_per_thread),
    dim=(d.nworld, d.njmax, threads_per_efc),
    inputs=[d.nefc, d.qacc, d.efc.J_rownnz, d.efc.J_rowadr, d.efc.J_colind, d.efc.J, d.efc.aref, ctx.prox_done],
    outputs=[ctx.Jaref],
  )


# Lodos's proximal solve.
def _refresh_after_aref(m: Model, d: Data, ctx: SolverContext):
  """Re-derives the Newton solver state after an aref rewrite (qacc, Ma unchanged).

  Resets the per-world counters, recomputes ctx.Jaref (nworld, njmax) against the
  rewritten aref, and re-evaluates the constraint state and gradient/Hessian.
  Prox-converged worlds keep ctx.done set, so the done-gated constraint/gradient
  kernels (including the Hessian rebuild and factorization) skip them entirely.

  Args:
    m: model.
    d: data; d.solver_niter (nworld,) is reset to 0 by _prox_solve_init_efc.
    ctx: solver context; ctx.done (nworld,) is preset to ctx.prox_done (nworld,).
  """
  wp.launch(
    _prox_solve_init_efc,
    dim=d.nworld,
    inputs=[ctx.prox_done],
    outputs=[d.solver_niter, ctx.search_dot, ctx.done],
  )
  _init_jaref(m, d, ctx)
  _update_constraint(m, d, ctx)
  _update_gradient(m, d, ctx)


# Lodos's proximal solve.
def _newton_iterations(m: Model, d: Data, ctx: SolverContext, nsolving: wp.array):
  """One warm-started Newton solve of the current (D, aref) subproblem.

  nsolving counts the worlds still running prox rounds; frozen worlds enter with
  ctx.done already set and are never counted, so the loop exits as soon as the
  active worlds converge (and never launches when all worlds are frozen).

  Args:
    m: model; consumes opt.iterations (Newton budget) and opt.graph_conditional.
    d: data; qacc (nworld, nv) and the efc arrays are advanced in place.
    ctx: solver context of the current subproblem.
    nsolving: (1,) int scratch counter, seeded from ctx.prox_nactive (1,).
  """
  wp.copy(nsolving, ctx.prox_nactive)
  if m.opt.iterations != 0 and m.opt.graph_conditional:
    wp.capture_while(nsolving, while_body=_solver_iteration, m=m, d=d, ctx=ctx, nsolving=nsolving, compact=False)
  else:
    for _ in range(m.opt.iterations):
      _solver_iteration(m, d, ctx, nsolving, compact=False)


# Lodos's proximal solve.
def _prox_round(m: Model, d: Data, ctx: SolverContext, nsolving: wp.array, update_sigma: int,
                z_only: int):
  """One full prox round: inner Newton solve, (z, sigma) update, convergence check,
  state refresh against the rewritten aref.

  solve_prox wraps rounds after the first in wp.capture_if on ctx.prox_nactive, so
  a fully frozen batch skips the entire round on device (within a round, per-world
  gating via ctx.prox_done / ctx.done handles partially frozen batches).

  Args:
    m: model.
    d: data; qacc, the efc arrays and solver_niter are advanced in place.
    ctx: solver context; all ctx.prox_* state is read and updated.
    nsolving: (1,) int scratch counter for the Newton loop.
    update_sigma: scalar 0/1; 1 = take the De Saxce sigma step (outer-round end).
    z_only: scalar 0/1; freeze on z alone, every round (sigma not refreshed in the step).
  """
  _newton_iterations(m, d, ctx, nsolving)

  wp.launch(
    _prox_zero_residuals,
    dim=d.nworld,
    outputs=[ctx.prox_dz, ctx.prox_znorm, ctx.prox_dsigma, ctx.prox_signorm],
  )
  wp.launch(
    _prox_update(bool(m.opt.prox_soft), d.contact_group_adr.shape[0] > 0),
    dim=d.contact.dist.shape[0],
    inputs=[
      m.opt.timestep,
      m.opt.impratio_invsqrt,
      m.opt.prox_sigma_relax,
      m.body_rootid,
      m.body_weldid,
      m.body_dofnum,
      m.body_dofadr,
      m.dof_parentid,
      m.geom_bodyid,
      d.qvel,
      d.qacc,
      d.cdof,
      d.subtree_com,
      d.contact.efc_address,
      d.efc.D,
      d.efc.vel,
      d.efc.force,
      d.nacon,
      d.contact_group_adr,
      d.contact_group_num,
      d.group_member_pos,
      d.group_member_frame,
      d.group_member_weight,
      d.group_member_slip,
      d.contact.dim,
      d.contact.worldid,
      d.contact.geom,
      d.contact.friction,
      d.contact.type,
      ctx.Jaref,
      ctx.prox_aref0,
      ctx.prox_done,
      update_sigma,
      ctx.prox_rho_i,
      ctx.prox_bdamp,
    ],
    outputs=[d.efc.aref, ctx.prox_z, ctx.prox_gamma, ctx.prox_dz, ctx.prox_znorm, ctx.prox_dsigma, ctx.prox_signorm],
  )
  wp.launch(
    _prox_check,
    dim=d.nworld,
    inputs=[
      m.opt.prox_tol_z,
      m.opt.prox_tol_sigma,
      d.solver_niter,
      ctx.prox_dz,
      ctx.prox_znorm,
      ctx.prox_dsigma,
      ctx.prox_signorm,
      update_sigma,
      z_only,
    ],
    outputs=[d.prox_rounds, ctx.prox_done, ctx.prox_niter, ctx.prox_nactive],
  )
  _refresh_after_aref(m, d, ctx)


# Lodos's proximal solve.
def _validate(m: Model):
  """Raises ValueError unless the model satisfies the prox solver's requirements
  (Newton solver, elliptic cone, no sleeping, and Option.prox_supportable)."""
  if m.opt.solver != SolverType.NEWTON:
    raise ValueError("prox_enable requires the Newton solver.")
  if m.opt.cone != ConeType.ELLIPTIC:
    raise ValueError("prox_enable requires the elliptic friction cone.")
  if m.opt.enableflags & EnableBit.SLEEP:
    raise ValueError("prox_enable does not support sleeping.")
  if not m.opt.prox_supportable:
    raise ValueError(
      "prox_enable requires condim <= 3, isotropic pair friction, and no flex (see Option.prox_supportable)."
    )


# Lodos's proximal solve.
@event_scope
def solve_prox(m: Model, d: Data, ctx: SolverContext):
  """Exact CCP solve: prox + De Saxce fixed point around warm-started Newton solves.

  Runs outer * inner rounds; round k solves the current (D, aref) subproblem with
  Newton, then updates z <- f* (every round) and sigma (every inner-th round) and
  rewrites aref. Converged worlds freeze (sticky prox_done) and skip all later work.

  Args:
    m: model; consumes opt.prox_* (budgets, tolerances, Baumgarte bias, rho, carry),
      opt.iterations (inner Newton budget), and the fields listed in _prox_init.
    d: data; reads the assembled constraint state, writes qacc (nworld, nv),
      qfrc_constraint (nworld, nv), the efc arrays (efc.force holds the exact
      contact forces f*), solver_niter (nworld,), and, with carry, the
      prox_zcache_* arrays (naconmax,) consumed by the NEXT step's warm start.
    ctx: solver context; all ctx.prox_* scratch arrays are (re)written.
  """
  _validate(m)

  warmstart = not (m.opt.disableflags & DisableBit.WARMSTART)  # bool: qacc warm start enabled
  wp.launch(
    _solve_init_dof(warmstart, m.is_sparse),
    dim=(d.nworld, m.nv),
    inputs=[d.nefc, d.qacc_warmstart, d.qacc_smooth],
    outputs=[d.qacc, d.qfrc_constraint],
  )

  # all worlds start active (also un-gates _init_jaref below)
  ctx.prox_done.zero_()
  d.prox_rounds.zero_()

  # jaref against the original aref (only jaref + aref, i.e. J*qacc, is consumed below)
  _init_jaref(m, d, ctx)

  carry = float(m.opt.prox_z_carry) > 0.0  # bool: cross-step anchor warm start enabled
  desaxce = bool(m.opt.prox_desaxce)  # bool: the De Saxce correction
  soft = bool(m.opt.prox_soft)  # bool: MuJoCo's softness kept on the contact rows

  wp.launch(
    _prox_init(bool(m.opt.prox_baumgarte), carry, desaxce, soft, d.contact_group_adr.shape[0] > 0),
    dim=d.contact.dist.shape[0],
    inputs=[
      m.opt.timestep,
      m.opt.impratio_invsqrt,
      m.opt.prox_rho,
      m.opt.prox_baumgarte_alpha,
      m.opt.prox_baumgarte_slop,
      m.opt.prox_z_carry,
      m.opt.disableflags,
      m.body_rootid,
      m.body_weldid,
      m.body_dofnum,
      m.body_dofadr,
      m.body_invweight0,
      m.dof_parentid,
      m.geom_bodyid,
      m.geom_rbound,
      d.contact.pos,
      d.contact.frame,
      d.contact.efc_address,
      d.efc.pos,
      d.efc.margin,
      d.efc.vel,
      d.nacon,
      d.prox_zcache_key,
      d.prox_zcache_val,
      d.prox_zcache_z,
      d.prox_zcache_pos,
      d.prox_zcache_n,
      d.qvel,
      d.qacc,
      d.cdof,
      d.subtree_com,
      d.contact_group_adr,
      d.contact_group_num,
      d.group_member_pos,
      d.group_member_frame,
      d.group_member_dist,
      d.group_member_weight,
      d.group_member_slip,
      d.contact.dim,
      d.contact.worldid,
      d.contact.geom,
      d.contact.friction,
      d.contact.type,
      ctx.Jaref,
      d.contact.solref,
      d.contact.solimp,
    ],
    outputs=[d.efc.D, d.efc.aref, ctx.prox_z, ctx.prox_aref0, ctx.prox_gamma, ctx.prox_rho_i, ctx.prox_bdamp],
  )

  # full context init against the rewritten (D, aref): jaref, Ma, constraint, gradient
  init_context(m, d, ctx, grad=True)

  inner = max(1, m.opt.prox_inner_iterations)  # scalar: prox (z) rounds per sigma update
  outer = max(1, m.opt.prox_outer_iterations)  # scalar: De Saxce (sigma) rounds
  ctx.prox_nactive.fill_(d.nworld)  # (1,) int: worlds still iterating
  nsolving = wp.empty(shape=(1,), dtype=int)  # (1,) int: Newton loop counter scratch

  # rounds after the first run under a conditional graph node: once every world is
  # frozen (prox_nactive == 0), the whole remaining round is skipped on device
  conditional = m.opt.iterations != 0 and m.opt.graph_conditional  # bool
  # sigma refreshed at outer-round ends, if there is a De Saxce term; where it is not refreshed
  # within the step -- one outer round, or no term -- worlds freeze on z alone, every round
  z_only = int(outer == 1 or not desaxce)  # scalar 0/1
  for k in range(outer * inner):
    update_sigma = int((k + 1) % inner == 0 and desaxce)  # scalar 0/1: sigma steps at outer-round ends
    if conditional and k > 0:
      wp.capture_if(ctx.prox_nactive, on_true=_prox_round, m=m, d=d, ctx=ctx, nsolving=nsolving, update_sigma=update_sigma,
                    z_only=z_only)
    else:
      _prox_round(m, d, ctx, nsolving, update_sigma, z_only)

    if m.opt.prox_host_convergence and (update_sigma or z_only):
      if int(ctx.prox_nactive.numpy()[0]) == 0:
        break

  # report the total Newton iterations of the step (the refresh reset the last count), and
  # whether each world's loop converged
  wp.copy(d.solver_niter, ctx.prox_niter)
  wp.copy(d.prox_converged, ctx.prox_done)

  # persist the converged anchors for the next step's warm start
  if carry:
    wp.launch(
      _prox_cache_fill,
      dim=d.contact.dist.shape[0],
      inputs=[
        d.contact.pos,
        d.contact.frame,
        d.contact.efc_address,
        d.nacon,
        d.contact.dim,
        d.contact.worldid,
        d.contact.geom,
        d.contact.type,
        ctx.prox_z,
      ],
      outputs=[d.prox_zcache_key, d.prox_zcache_val, d.prox_zcache_z, d.prox_zcache_pos],
    )
    wp.utils.radix_sort_pairs(d.prox_zcache_key, d.prox_zcache_val, d.contact.dist.shape[0])
    wp.copy(d.prox_zcache_n, d.nacon)
