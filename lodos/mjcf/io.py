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
# Adapted for Lodos from MuJoCo Warp 3.14.0's mujoco_warp/_src/io.py and mujoco_warp/_src/support.py: the parts Lodos
# uses, its MJ_ constants without the prefix (MJ_MINIMP and MJ_MAXIMP keep it). Every function keeps its MuJoCo Warp
# name; one from support.py says so above it.

"""The MJCF model and data on the device, made from MuJoCo's: `put_model`, `put_data`,
`override_model` and the capacity estimates."""

import dataclasses
import warnings
from typing import Any, Optional, Sequence, Tuple

import mujoco
import numpy as np

import warp as wp

from lodos.mjcf import types
from lodos.mjcf.types import COLLISION_TABLE
from lodos.utils.kernels import check_toolkit_driver
from lodos.utils.kernels import is_array_spec
from lodos.utils.kernels import mark_batched

wp.set_module_options({"default_grid_stride": False})


def _create_array(data: Any, spec, sizes: dict[str, int], batch_size: int = 1) -> wp.array | None:
  """Creates a warp array and populates it with data.

  The array shape is determined by a field spec referencing MjModel / MjData array sizes.
  """
  spec_shape = getattr(spec, "shape", (0,))
  if spec_shape == (0,):
    if data is None:
      return None
    return wp.array(np.array(data), dtype=spec.dtype)

  shape = tuple(
    batch_size
    if dim == "*"
    else (int(dim) if isinstance(dim, str) and dim.isdigit() else (sizes[dim] if isinstance(dim, str) else dim))
    for dim in spec_shape
  )

  is_batched = spec_shape[0] in ("*", "nworld")

  if data is None:
    array = wp.zeros(shape, dtype=spec.dtype)
  else:
    data = np.array(data)
    if is_batched and shape[0] != 1:
      target_shape = shape + getattr(spec.dtype, "_shape_", ())
      if data.shape != target_shape:
        tail_shape = target_shape[1:]
        if data.size == np.prod(tail_shape):
          data = data.reshape(tail_shape)
        data = np.broadcast_to(data, target_shape).copy()
    array = wp.array(data, dtype=spec.dtype, shape=shape)

  return array


def _create_constraint(
  mjm,
  nworld: int,
  njmax: int,
  sizes: dict,
  mjd=None,
) -> types.Constraint:
  """Construct a types.Constraint with standard and island local fields allocated properly."""
  efc_kwargs = {"J_rownnz": None, "J_rowadr": None, "J_colind": None, "J": None}
  sparse = is_sparse(mjm)
  # The JTDAJ block list is only consumed by the sparse Newton Hessian assembly (_JTDACJ_sparse).
  jtdaj_active = sparse and mjm.opt.solver == mujoco.mjtSolver.mjSOL_NEWTON

  for f in dataclasses.fields(types.Constraint):
    if f.name in ("jtdaj_adr", "jtdaj_nrow"):
      efc_kwargs[f.name] = wp.empty((nworld, njmax if jtdaj_active else 0), dtype=int)
    elif f.name == "jtdaj_nblock":
      efc_kwargs[f.name] = wp.empty((nworld,), dtype=int)
    else:
      if f.name in efc_kwargs:
        continue

    if mjd is not None:
      shape = tuple(sizes[dim] if isinstance(dim, str) else dim for dim in f.type.shape)
      val = np.full(shape, -1 if f.name == "island" else 0, dtype=f.type.dtype)
      if f.name in ("type", "id", "pos", "margin", "D", "vel", "aref", "frictionloss", "force", "island"):
        val[:, : mjd.nefc] = np.tile(getattr(mjd, "efc_" + f.name), (nworld, 1))
      efc_kwargs[f.name] = wp.array(val, dtype=f.type.dtype)
    else:
      efc_kwargs[f.name] = _create_array(None, f.type, sizes)

  return types.Constraint(**efc_kwargs)


def _jtdaj_groups(mjd: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
  """Group loaded efc rows into JTDAJ blocks: maximal runs sharing (efc_type, efc_id).

  MuJoCo lays each constraint's rows out contiguously, so this reproduces the block list
  make_constraint builds in-kernel. Returns block start rows (adr) and lengths (nrow).
  """
  nefc = mjd.nefc
  if nefc == 0:
    return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
  etype = mjd.efc_type[:nefc]
  eid = mjd.efc_id[:nefc]
  boundary = np.ones(nefc, dtype=bool)
  boundary[1:] = (etype[1:] != etype[:-1]) | (eid[1:] != eid[:-1])
  adr = np.flatnonzero(boundary)
  nrow = np.diff(np.append(adr, nefc))
  return adr, nrow


def _get_nflexintcell(mjm: mujoco.MjModel) -> int:
  nflexintcell = 0
  if mjm.nflex > 0:
    for fi in range(mjm.nflex):
      order = abs(int(mjm.flex_interp[fi]))
      if order == 0:
        continue
      if mjm.flex_edgeequality[fi] == 3:
        continue
      cx, cy, cz = mjm.flex_cellnum[fi]
      nflexintcell += int(cx) * int(cy) * int(cz)
  return nflexintcell


def _get_nflexface(mjm: mujoco.MjModel) -> int:
  nflexface = 0
  if mjm.nflex > 0:
    for fi in range(mjm.nflex):
      order = mjm.flex_interp[fi]
      if order >= 0:
        continue
      cx, cy, cz = mjm.flex_cellnum[fi]
      nfaces = 2 * (cy * cz + cx * cz + cx * cy)
      nflexface += int(nfaces)
  return nflexface


def is_sparse(mjm: mujoco.MjModel) -> bool:
  if mjm.opt.jacobian == mujoco.mjtJacobian.mjJAC_AUTO:
    if mjm.nv > 32:
      return True
    else:
      return False
  else:
    return bool(mujoco.mj_isSparse(mjm))


def _m_blocks(mjm: mujoco.MjModel):
  """The (start, size) diagonal blocks of M: the kinematic trees, each a contiguous dof range.

  M couples a dof only with its tree ancestors, so its diagonal blocks are exactly the trees.
  (dof_simplenum is not used to classify blocks: it is a contiguous-suffix run-length, so it
  misses interspersed decoupled dofs; the M_rownnz coupling check in m_block_layout catches those.)
  """
  return [(int(adr), int(num)) for adr, num in zip(mjm.tree_dofadr, mjm.tree_dofnum) if num > 0]


def m_block_layout(mjm: mujoco.MjModel) -> dict:
  """Per-block dense/sparse layout for M's diagonal blocks.

  Blocks use scalar Cholesky through six DOFs, tile Cholesky through M_BLOCK_DENSE_MAX, and sparse
  LDL beyond that. Compact diagonal blocks also use the scalar path without allocating a factor.
  """
  nv = mjm.nv
  blocks = _m_blocks(mjm)
  dof_adr = np.full(nv, types.Q_LD_BLOCK_SPARSE, dtype=np.int32)
  scalar_tiles = {}
  gather_tiles = {}
  off = 0
  for start, size in blocks:
    last = start + size - 1
    madr = int(mjm.M_rowadr[start])
    nnz = int(mjm.M_rowadr[last] + mjm.M_rownnz[last] - madr)
    compact = nnz == size
    triangular = nnz == size * (size + 1) // 2

    if size <= types.M_BLOCK_SCALAR_MAX and (compact or triangular):
      scalar_tiles.setdefault(size, []).append(start)
      if compact:
        dof_adr[start : start + size] = types.Q_LD_BLOCK_COMPACT
      else:
        dof_adr[start : start + size] = off
        off += size * size
    elif size <= types.M_BLOCK_DENSE_MAX:
      gather_tiles.setdefault(size, []).append(start)
      dof_adr[start : start + size] = off
      off += size * size
  for starts in scalar_tiles.values():
    starts.sort(key=lambda start: dof_adr[start] >= 0)
  return {
    "total": off,
    "dof_adr": dof_adr,
    "scalar_tiles": scalar_tiles,
    "gather_tiles": gather_tiles,
    "has_sparse": bool(np.any(dof_adr == types.Q_LD_BLOCK_SPARSE)),
  }


def put_model(mjm: mujoco.MjModel, batch_sizes: dict[str, int] | None = None) -> types.Model:
  """Creates a model on device.

  Args:
    mjm: The model containing kinematic and dynamic information (host).
    batch_sizes: Optional per-field leading batch sizes for `Model` fields whose
      array spec starts with `*`. Fields not listed here keep the default shared
      leading dimension of 1.

  Returns:
    The model containing kinematic and dynamic information (device).
  """
  # check for compatible cuda toolkit and driver versions
  check_toolkit_driver()

  batch_sizes = batch_sizes or {}
  model_fields = {f.name: f.type for f in dataclasses.fields(types.Model) if is_array_spec(f.type)}
  for name, size in batch_sizes.items():
    field_type = model_fields.get(name)
    spec_shape = getattr(field_type, "shape", ())
    if not spec_shape or spec_shape[0] != "*":
      raise ValueError(f"Model field {name!r} is not a batched array field.")
    if size < 1:
      raise ValueError(f"batch_sizes[{name!r}] must be positive, got {size}.")

  # model: check supported features in array types
  for field, field_type, mj_type in (
    (mjm.actuator_trntype, types.TrnType, mujoco.mjtTrn),
    (mjm.actuator_dyntype, types.DynType, mujoco.mjtDyn),
    (mjm.actuator_gaintype, types.GainType, mujoco.mjtGain),
    (mjm.actuator_biastype, types.BiasType, mujoco.mjtBias),
    (mjm.eq_type, types.EqType, mujoco.mjtEq),
    (mjm.geom_type, types.GeomType, mujoco.mjtGeom),
    (mjm.sensor_type, types.SensorType, mujoco.mjtSensor),
    (mjm.wrap_type, types.WrapType, mujoco.mjtWrap),
    (mjm.tree_sleep_policy, types.SleepPolicy, mujoco.mjtSleepPolicy),
  ):
    missing = ~np.isin(field, field_type)
    if missing.any():
      names = [mj_type(v).name for v in field[missing]]
      raise NotImplementedError(f"{names} not supported.")

  # opt: check supported features in scalar types
  for field, field_type, mj_type in (
    (mjm.opt.integrator, types.IntegratorType, mujoco.mjtIntegrator),
    (mjm.opt.cone, types.ConeType, mujoco.mjtCone),
    (mjm.opt.solver, types.SolverType, mujoco.mjtSolver),
  ):
    if field not in set(field_type):
      raise NotImplementedError(f"{mj_type(field).name} is unsupported.")

  # opt: check supported features in scalar flag types
  for field, field_type, mj_type in (
    (mjm.opt.disableflags, types.DisableBit, mujoco.mjtDisableBit),
    (mjm.opt.enableflags, types.EnableBit, mujoco.mjtEnableBit),
  ):
    unsupported = field & ~np.bitwise_or.reduce(field_type)
    if unsupported:
      raise NotImplementedError(f"{mj_type(unsupported).name} is unsupported.")

  if (mjm.opt.enableflags & mujoco.mjtEnableBit.mjENBL_SLEEP) and (mjm.eq_type == mujoco.mjtEq.mjEQ_FLEX).any():
    raise NotImplementedError("Flex equality constraints are not supported with sleeping enabled.")

  if mjm.opt.noslip_iterations > 0:
    raise NotImplementedError(f"noslip solver not implemented.")

  if (mjm.body_plugin != -1).any():
    raise NotImplementedError("Body plugins not supported.")

  if (mjm.actuator_plugin != -1).any():
    raise NotImplementedError("Actuator plugins not supported.")

  if (mjm.sensor_plugin != -1).any():
    raise NotImplementedError("Sensor plugins not supported.")

  if mjm.nflex > 0:
    for fi in range(mjm.nflex):
      if abs(mjm.flex_interp[fi]) == 2:
        raise NotImplementedError("Quadratic flex interpolation (dof=quadratic) is not supported.")
      if mjm.flex_interp[fi] >= 0:
        continue
      bendingadr = mjm.flex_bendingadr[fi]
      if bendingadr < 0:
        continue
      nedge = int(mjm.flex_bending[bendingadr])
      if nedge > 0 and mjm.flex_damping[fi] > 0.0:
        warnings.warn("Bending damping is not yet supported for interpolated flex shells.")

  # array sizes may change in the future
  if mujoco.mjNPOLY != 2:
    warnings.warn(f"mujoco.mjNPOLY is {mujoco.mjNPOLY}, expected 2. Higher order polynomials may not be supported correctly.")

  # TODO(team): remove after _update_gradient for Newton uses tile operations for islands
  nv_max = 60
  if mjm.nv > nv_max and mjm.opt.jacobian == mujoco.mjtJacobian.mjJAC_DENSE:
    raise ValueError(f"Dense is unsupported for nv > {nv_max} (nv = {mjm.nv}).")

  # sleeping is supported via a dof-compaction approach.  awake dofs are compacted into dense
  # nvmax-sized arrays.  nvmax is chosen to fit the worst-case active dof set. sleeping is only
  # supported for Newton solver and requires nv <= nvmax.
  if (mjm.opt.enableflags & mujoco.mjtEnableBit.mjENBL_SLEEP) and mjm.opt.solver != mujoco.mjtSolver.mjSOL_NEWTON:
    raise ValueError(f"sleeping requires the Newton solver (got solver={types.SolverType(mjm.opt.solver).name})")

  collision_sensors = (mujoco.mjtSensor.mjSENS_GEOMDIST, mujoco.mjtSensor.mjSENS_GEOMNORMAL, mujoco.mjtSensor.mjSENS_GEOMFROMTO)
  is_collision_sensor = np.isin(mjm.sensor_type, collision_sensors)

  def not_implemented(objtype, objid, geomtype):
    if objtype == mujoco.mjtObj.mjOBJ_BODY:
      geomnum = mjm.body_geomnum[objid]
      geomadr = mjm.body_geomadr[objid]
      for geomid in range(geomadr, geomadr + geomnum):
        if mjm.geom_type[geomid] == geomtype:
          return True
    elif objtype == mujoco.mjtObj.mjOBJ_GEOM:
      if mjm.geom_type[objid] == geomtype:
        return True
    return False

  def _check_friction(name: str, id_: int, condim: int, friction, checks):
    for min_condim, indices in checks:
      if condim >= min_condim:
        for idx in indices:
          if friction[idx] < types.MINMU:
            warnings.warn(
              f"{name} {id_}: friction[{idx}] ({friction[idx]}) < MINMU ({types.MINMU}) with condim={condim} may cause NaN"
            )

  for geomid in range(mjm.ngeom):
    _check_friction("geom", geomid, mjm.geom_condim[geomid], mjm.geom_friction[geomid], [(3, [0]), (4, [1]), (6, [2])])

  for pairid in range(mjm.npair):
    _check_friction("pair", pairid, mjm.pair_dim[pairid], mjm.pair_friction[pairid], [(3, [0]), (4, [1, 2]), (6, [3, 4])])

  # create opt
  opt_kwargs = {f.name: getattr(mjm.opt, f.name, None) for f in dataclasses.fields(types.Option)}
  if hasattr(mjm.opt, "impratio"):
    opt_kwargs["impratio_invsqrt"] = 1.0 / np.sqrt(np.maximum(mjm.opt.impratio, mujoco.mjMINVAL))
  opt = types.Option(**opt_kwargs)

  # C MuJoCo tolerance was chosen for float64 architecture, but we default to float32 on GPU
  # adjust the tolerance for lower precision, to avoid the solver spending iterations needlessly
  # bouncing around the optimal solution
  opt.tolerance = max(opt.tolerance, 1e-6)

  # warp only fields
  opt.broadphase = types.BroadphaseType.NXN
  opt.broadphase_filter = types.BroadphaseFilter.PLANE | types.BroadphaseFilter.SPHERE | types.BroadphaseFilter.OBB
  opt.graph_conditional = True
  opt.run_collision_detection = True
  opt.warn_overflow = int(types.OverflowType.ALL)
  opt.run_rne_postconstraint = False
  contact_sensor_maxmatch_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_NUMERIC, "contact_sensor_maxmatch")
  if contact_sensor_maxmatch_id > -1:
    opt.contact_sensor_maxmatch = mjm.numeric_data[mjm.numeric_adr[contact_sensor_maxmatch_id]]
  else:
    opt.contact_sensor_maxmatch = 64

  # exact-contact (prox) solver: off by default; flattened schedule when enabled
  opt.prox_enable = 0
  opt.prox_rho = 1.0
  opt.prox_outer_iterations = 10
  opt.prox_inner_iterations = 1
  opt.prox_tol_z = 1e-4
  opt.prox_tol_sigma = 1e-4
  # Baumgarte bias on: the stabilization whose exact solution is the standard NCP; off is
  # the velocity-level target (kinetic-energy minimization). No solref/solimp either way.
  opt.prox_baumgarte = True
  # depenetration fraction: 0.1 is the only tested value with no failures on
  # impact-heavy dynamic scenes at low carried budgets (carry + outer=1 can explode at
  # 0.02); quiet knife-edge structures with carried anchors want 0.02 or 0 instead (0.1
  # excites chatter the one-step-delayed anchor amplifies)
  opt.prox_baumgarte_alpha = 0.1
  # the penetration the bias leaves alone: 0 pushes every one out to zero, where a
  # resting contact vanishes and comes back (Lodos sets 2e-4 m, lodos.cd.driver)
  opt.prox_baumgarte_slop = 0.0
  # the De Saxce correction on (the exact problem); MuJoCo's softness off (the exact rows)
  opt.prox_desaxce = True
  opt.prox_soft = False
  opt.prox_sigma_relax = 1.0
  opt.prox_z_carry = 0.0
  opt.prox_host_convergence = False
  # model features the prox solver does not support (checked at solve time if enabled)
  opt.prox_supportable = int(
    (mjm.geom_condim <= 3).all()
    and (mjm.pair_dim <= 3).all()
    and (mjm.npair == 0 or (mjm.pair_friction[:, 0] == mjm.pair_friction[:, 1]).all())
    and mjm.nflex == 0
  )
  # Newton's sparse Hessian summed per world in shared memory: off by default (solver.py)
  opt.shared_hessian = False

  # place opt on device
  for f in dataclasses.fields(types.Option):
    if is_array_spec(f.type):
      setattr(opt, f.name, _create_array(getattr(opt, f.name), f.type, {"*": 1}))
    else:
      setattr(opt, f.name, f.type(getattr(opt, f.name)))

  # create stat
  stat = types.Statistic(meaninertia=_create_array([mjm.stat.meaninertia], types.array("*", float), {"*": 1}))

  # create model
  m = types.Model(**{f.name: getattr(mjm, f.name, None) for f in dataclasses.fields(types.Model)})

  m.opt = opt
  m.stat = stat
  m.callback = types.Callback()

  m.nv_pad = _get_padded_sizes(
    mjm.nv,
    0,
    is_sparse(mjm),
    types.TILE_SIZE_JTDAJ_SPARSE if is_sparse(mjm) else types.TILE_SIZE_JTDAJ_DENSE,
    augment_cholesky=mjm.opt.solver == mujoco.mjtSolver.mjSOL_NEWTON and mjm.nv > 32,
  )[1]
  m.nacttrnbody = (mjm.actuator_trntype == mujoco.mjtTrn.mjTRN_BODY).sum()
  m.nsensortaxel = mjm.mesh_vertnum[mjm.sensor_objid[mjm.sensor_type == mujoco.mjtSensor.mjSENS_TACTILE]].sum()
  m.nsensorcontact = (mjm.sensor_type == mujoco.mjtSensor.mjSENS_CONTACT).sum()
  m.nrangefinder = (mjm.sensor_type == mujoco.mjtSensor.mjSENS_RANGEFINDER).sum()
  condim_arrays = [np.array([0]), mjm.geom_condim, mjm.pair_dim]
  if mjm.nflex > 0:
    condim_arrays.append(mjm.flex_condim)
    if (mjm.geom_type == mujoco.mjtGeom.mjGEOM_SDF).any():
      raise NotImplementedError("Flex-SDF collision is not implemented.")
    if (mjm.geom_type == mujoco.mjtGeom.mjGEOM_HFIELD).any():
      raise NotImplementedError("Flex-HField collision is not implemented.")
    if (mjm.flex_internal != 0).any():
      raise NotImplementedError("Flex internal collisions are not implemented.")
    if (mjm.flex_rigid != 0).any():
      raise NotImplementedError("Rigid flexes are not implemented.")
  m.nmaxcondim = np.concatenate(condim_arrays).max()
  m.nmaxpyramid = np.maximum(1, 2 * (m.nmaxcondim - 1))
  m.has_sdf_geom = (mjm.geom_type == mujoco.mjtGeom.mjGEOM_SDF).any()
  m.has_ellipsoid_geom = (mjm.geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID).any()
  m.has_plane_geom = bool(mjm.ngeom > 0 and (mjm.geom_type == mujoco.mjtGeom.mjGEOM_PLANE).any())
  m.has_flex_selfcollide = bool(
    mjm.nflex > 0 and np.any((mjm.flex_selfcollide != 0) & ((mjm.flex_contype & mjm.flex_conaffinity) != 0))
  )
  m.has_1d_flex = bool(mjm.nflex > 0 and np.any(mjm.flex_dim == 1))
  m.has_2d_flex = bool(mjm.nflex > 0 and np.any(mjm.flex_dim == 2))
  m.has_3d_flex = bool(mjm.nflex > 0 and np.any(mjm.flex_dim == 3))
  m.max_flex_dim = int(np.max(mjm.flex_dim)) if mjm.nflex > 0 else 0
  m.block_dim = types.BlockDim()
  # Derive CG solver block_dim from nv: clamp(round_up_to_32(nv), 32, 256)
  _nv_block = max(32, min(256, ((mjm.nv + 31) // 32) * 32))
  m.block_dim.update_gradient_grad = _nv_block
  m.block_dim.solve_beta_accumulate = _nv_block
  m.block_dim.solve_search_update_cg = _nv_block
  m.block_dim.solve_init_search_cg = _nv_block
  if mjm.nv > 500:
    m.block_dim.linesearch_iterative = 256
  m.is_sparse = is_sparse(mjm)
  m.has_fluid = bool(mjm.opt.wind.any() or mjm.opt.density > 0 or mjm.opt.viscosity > 0)
  m.flg_speculative = False  # lodos.sim.make_sim sets it where a Lodos pair is `toa`
  m.nflexintcell = _get_nflexintcell(mjm)

  # Precompute flex_cell_map
  flex_cell_map = []
  if mjm.nflex > 0:
    for fi in range(mjm.nflex):
      order = abs(int(mjm.flex_interp[fi]))
      if order == 0:
        continue
      if mjm.flex_edgeequality[fi] == 3:
        continue
      cx, cy, cz = mjm.flex_cellnum[fi]
      for ci in range(cx):
        for cj in range(cy):
          for ck in range(cz):
            flex_cell_map.append((fi, ci, cj, ck))
  if not flex_cell_map:
    m.flex_cell_map = np.zeros((0, 4), dtype=np.int32)
  else:
    m.flex_cell_map = np.array(flex_cell_map, dtype=np.int32)

  m.max_ten_J_rownnz = int(mjm.ten_J_rownnz.max()) if mjm.ntendon else 0

  # body ids grouped by tree level (depth-based traversal)
  bodies, body_depth = {}, np.zeros(mjm.nbody, dtype=int) - 1
  for i in range(mjm.nbody):
    body_depth[i] = body_depth[mjm.body_parentid[i]] + 1
    bodies.setdefault(body_depth[i], []).append(i)
  m.body_tree = tuple(wp.array(bodies[i], dtype=int) for i in sorted(bodies))

  # branch-based traversal data
  children_count = np.bincount(mjm.body_parentid[1:], minlength=mjm.nbody)
  ancestor_chain = lambda b: ancestor_chain(mjm.body_parentid[b]) + [b] if b else []
  branches = [ancestor_chain(l) for l in np.where(children_count[1:] == 0)[0] + 1]
  m.nbranch = len(branches)

  body_branches = []
  body_branch_start = []
  offset = 0

  for branch in branches:
    body_branches.extend(branch)
    body_branch_start.append(offset)
    offset += len(branch)
  body_branch_start.append(offset)

  m.body_branches = np.array(body_branches, dtype=int)
  m.body_branch_start = np.array(body_branch_start, dtype=int)

  m.mocap_bodyid = np.arange(mjm.nbody)[mjm.body_mocapid >= 0]
  m.mocap_bodyid = m.mocap_bodyid[mjm.body_mocapid[mjm.body_mocapid >= 0].argsort()]
  m.body_fluid_ellipsoid = np.zeros(mjm.nbody, dtype=bool)
  has_fluid = mjm.geom_fluid.reshape(mjm.ngeom, mujoco.mjNFLUID)[:, 0] > 0
  if np.any(has_fluid):
    fluid_bodyids = mjm.geom_bodyid[has_fluid]
    valid_mass = mjm.body_mass[fluid_bodyids] >= types.MINVAL
    m.body_fluid_ellipsoid[fluid_bodyids[valid_mass]] = True
  m.body_fluid_ellipsoid_adr = np.nonzero(m.body_fluid_ellipsoid)[0]
  body_is_free = np.zeros(mjm.nbody, dtype=bool)
  has_one_jnt = mjm.body_jntnum == 1
  if np.any(has_one_jnt):
    b_indices = np.nonzero(has_one_jnt)[0]
    jnt_adrs = mjm.body_jntadr[b_indices]
    is_free_jnt = mjm.jnt_type[jnt_adrs] == mujoco.mjtJoint.mjJNT_FREE
    if np.any(is_free_jnt):
      b_free = b_indices[is_free_jnt]
      jnt_free_adrs = jnt_adrs[is_free_jnt]
      dof_adrs = mjm.jnt_dofadr[jnt_free_adrs]
      tree_ids = mjm.dof_treeid[dof_adrs]
      tree_dof6 = mjm.tree_dofnum[tree_ids] == 6
      mass_match = mjm.body_subtreemass[b_free] == mjm.body_mass[b_free]
      body_is_free[b_free] = tree_dof6 & mass_match
  m.body_is_free = body_is_free
  m.body_freeadr = np.nonzero(m.body_is_free)[0]
  body_fluid_box = np.zeros(mjm.nbody, dtype=bool)
  for b in range(1, mjm.nbody):
    if not m.body_fluid_ellipsoid[b] and mjm.body_mass[b] >= types.MINVAL:
      body_fluid_box[b] = True
  m.body_fluid_box_adr = np.nonzero(body_fluid_box)[0]
  jnt_limited_slide_hinge = mjm.jnt_limited & np.isin(mjm.jnt_type, (mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_HINGE))
  m.jnt_limited_slide_hinge_adr = np.nonzero(jnt_limited_slide_hinge)[0]
  m.jnt_limited_ball_adr = np.nonzero(mjm.jnt_limited & (mjm.jnt_type == mujoco.mjtJoint.mjJNT_BALL))[0]
  m.dof_tri_row, m.dof_tri_col = np.triu_indices(mjm.nv)

  # precompute body_isdofancestor: which DOFs affect each body
  # TODO: Investigate alternative approach such as bitmap
  body_isdofancestor = np.zeros((mjm.nbody, m.nv_pad), dtype=np.int32)
  for bodyid in range(mjm.nbody):
    b = bodyid
    while b > 0 and mjm.body_dofnum[b] == 0:
      b = mjm.body_parentid[b]
    if mjm.body_dofnum[b] == 0:
      continue
    dofid = mjm.body_dofadr[b] + mjm.body_dofnum[b] - 1
    while dofid >= 0:
      body_isdofancestor[bodyid, dofid] = 1
      dofid = mjm.dof_parentid[dofid]
  m.body_isdofancestor = body_isdofancestor

  # precalculated geom pairs
  filterparent = not (mjm.opt.disableflags & types.DisableBit.FILTERPARENT)

  geom1, geom2 = np.triu_indices(mjm.ngeom, k=1)
  m.nxn_geom_pair = np.stack((geom1, geom2), axis=1)

  bodyid1 = mjm.geom_bodyid[geom1]
  bodyid2 = mjm.geom_bodyid[geom2]
  contype1 = mjm.geom_contype[geom1]
  contype2 = mjm.geom_contype[geom2]
  conaffinity1 = mjm.geom_conaffinity[geom1]
  conaffinity2 = mjm.geom_conaffinity[geom2]
  weldid1 = mjm.body_weldid[bodyid1]
  weldid2 = mjm.body_weldid[bodyid2]
  weld_parentid1 = mjm.body_weldid[mjm.body_parentid[weldid1]]
  weld_parentid2 = mjm.body_weldid[mjm.body_parentid[weldid2]]

  self_collision = weldid1 == weldid2
  parent_child_collision = (
    filterparent & (weldid1 != 0) & (weldid2 != 0) & ((weldid1 == weld_parentid2) | (weldid2 == weld_parentid1))
  )
  mask = np.array((contype1 & conaffinity2) | (contype2 & conaffinity1), dtype=bool)
  exclude = np.isin((bodyid1 << 16) + bodyid2, mjm.exclude_signature)

  nxn_pairid_contact = -1 * np.ones(len(geom1), dtype=int)
  nxn_pairid_contact[~(mask & ~self_collision & ~parent_child_collision & ~exclude)] = -2

  # contact pairs
  def upper_tri_index(n, i, j):
    i, j = (j, i) if j < i else (i, j)
    return (i * (2 * n - i - 3)) // 2 + j - 1

  for i in range(mjm.npair):
    nxn_pairid_contact[upper_tri_index(mjm.ngeom, mjm.pair_geom1[i], mjm.pair_geom2[i])] = i

  sensor_collision_adr = np.nonzero(is_collision_sensor)[0]
  collision_sensor_adr = np.full(mjm.nsensor, -1)
  collision_sensor_adr[sensor_collision_adr] = np.arange(len(sensor_collision_adr))

  nxn_pairid_collision = -1 * np.ones(len(geom1), dtype=int)
  pairids = []
  sensor_collision_start_adr = []
  for i in range(sensor_collision_adr.size):
    sensorid = sensor_collision_adr[i]
    objtype = mjm.sensor_objtype[sensorid]
    objid = mjm.sensor_objid[sensorid]
    reftype = mjm.sensor_reftype[sensorid]
    refid = mjm.sensor_refid[sensorid]

    # get lists of geoms to collide
    if objtype == types.ObjType.BODY:
      n1 = mjm.body_geomnum[objid]
      id1 = mjm.body_geomadr[objid]
    else:
      n1 = 1
      id1 = objid
    if reftype == types.ObjType.BODY:
      n2 = mjm.body_geomnum[refid]
      id2 = mjm.body_geomadr[refid]
    else:
      n2 = 1
      id2 = refid

    # collide all pairs
    for geom1id in range(id1, id1 + n1):
      for geom2id in range(id2, id2 + n2):
        pairid = upper_tri_index(mjm.ngeom, geom1id, geom2id)

        if pairid in pairids:
          sensor_collision_start_adr.append(nxn_pairid_collision[pairid])
        else:
          npairids = len(pairids)
          nxn_pairid_collision[pairid] = npairids
          sensor_collision_start_adr.append(npairids)
          pairids.append(pairid)

  m.nsensorcollision = (nxn_pairid_collision >= 0).sum()
  m.sensor_collision_start_adr = np.array(sensor_collision_start_adr)
  nxn_include = (nxn_pairid_contact > -2) | (nxn_pairid_collision >= 0)

  if nxn_include.sum() < 250_000:
    opt.broadphase = types.BroadphaseType.NXN
  elif mjm.ngeom < 1000:
    opt.broadphase = types.BroadphaseType.SAP_TILE
  else:
    opt.broadphase = types.BroadphaseType.SAP_SEGMENTED

  m.nxn_geom_pair_filtered = m.nxn_geom_pair[nxn_include]
  m.nxn_pairid = np.hstack([nxn_pairid_contact.reshape((-1, 1)), nxn_pairid_collision.reshape((-1, 1))])
  m.nxn_pairid_filtered = m.nxn_pairid[nxn_include]

  # count contact pair types
  def geom_trid_index(i, j):
    i, j = (j, i) if j < i else (i, j)
    return (i * (2 * len(types.GeomType) - i - 1)) // 2 + j

  m.geom_pair_type_count = tuple(
    np.bincount(
      [geom_trid_index(mjm.geom_type[geom1[i]], mjm.geom_type[geom2[i]]) for i in np.arange(len(geom1)) if nxn_include[i]],
      minlength=len(types.GeomType) * (len(types.GeomType) + 1) // 2,
    )
  )

  # check for unsupported margin + multicontact / box-box CCD combinations
  use_multiccd = (mjm.opt.disableflags & types.DisableBit.MULTICCD) == 0
  nativeccd_disabled = mjm.opt.disableflags & types.DisableBit.NATIVECCD
  BOX = int(mujoco.mjtGeom.mjGEOM_BOX)
  MESH = int(mujoco.mjtGeom.mjGEOM_MESH)

  # TODO(team): remove after implementing multicontact support for CCD pairs.
  if use_multiccd:
    unsupported_multiccd_pairs = []
    for (g1, g2), col_type in COLLISION_TABLE.items():
      if g1 == types.GeomType.BOX and g2 == types.GeomType.BOX and nativeccd_disabled:
        continue
      if col_type == types.CollisionType.CONVEX:
        if g1 in (types.GeomType.SPHERE, types.GeomType.ELLIPSOID) or g2 in (
          types.GeomType.SPHERE,
          types.GeomType.ELLIPSOID,
        ):
          continue
        if (g1, g2) not in (
          (types.GeomType.BOX, types.GeomType.BOX),
          (types.GeomType.BOX, types.GeomType.MESH),
          (types.GeomType.MESH, types.GeomType.MESH),
          (types.GeomType.CYLINDER, types.GeomType.CYLINDER),
          (types.GeomType.CYLINDER, types.GeomType.BOX),
          (types.GeomType.CYLINDER, types.GeomType.MESH),
        ):
          if m.geom_pair_type_count[geom_trid_index(int(g1), int(g2))] > 0:
            unsupported_multiccd_pairs.append((g1.name, g2.name))
    if unsupported_multiccd_pairs:
      warnings.warn(
        "MULTICCD is enabled, but the scene contains CCD pairs without multicontact support:"
        f" {unsupported_multiccd_pairs}. At most 1 contact will be generated for these pairs."
      )

  has_boxbox = m.geom_pair_type_count[geom_trid_index(BOX, BOX)] > 0
  has_multiccd_pairs = has_boxbox or (
    use_multiccd
    and (m.geom_pair_type_count[geom_trid_index(BOX, MESH)] > 0 or m.geom_pair_type_count[geom_trid_index(MESH, MESH)] > 0)
  )

  if has_multiccd_pairs:

    def _check_margin(name, t1, t2, margin):
      if use_multiccd:
        raise NotImplementedError(
          f"{name} has non-zero margin ({margin}) with MULTICCD enabled. Set margin to 0 or disable MULTICCD."
        )
      if t1 == BOX and t2 == BOX and not nativeccd_disabled:
        raise NotImplementedError(
          f"{name} has non-zero margin ({margin}) with NATIVECCD enabled. Set margin to 0 or disable NATIVECCD."
        )

    geom_name = lambda g: mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_GEOM, g) or str(g)

    for idx in np.nonzero(nxn_include & (nxn_pairid_contact == -1))[0]:
      g1, g2 = int(geom1[idx]), int(geom2[idx])
      t1, t2 = int(mjm.geom_type[g1]), int(mjm.geom_type[g2])
      m1, m2 = float(mjm.geom_margin[g1]), float(mjm.geom_margin[g2])
      if (m1 or m2) and t1 in (BOX, MESH) and t2 in (BOX, MESH):
        _check_margin(f"geom pair ({geom_name(g1)}, {geom_name(g2)})", t1, t2, (m1, m2))

    for pid in range(mjm.npair):
      g1, g2 = int(mjm.pair_geom1[pid]), int(mjm.pair_geom2[pid])
      t1, t2 = int(mjm.geom_type[g1]), int(mjm.geom_type[g2])
      pm = float(mjm.pair_margin[pid])
      if pm and t1 in (BOX, MESH) and t2 in (BOX, MESH):
        _check_margin(f"pair {pid} ({geom_name(g1)}, {geom_name(g2)})", t1, t2, pm)

  # filter plugins for only geom plugins, drop the rest
  m.plugin, m.plugin_attr = [], []
  m.geom_plugin_index = np.full_like(mjm.geom_type, -1)

  for i in range(len(mjm.geom_plugin)):
    if mjm.geom_plugin[i] == -1:
      continue
    p = mjm.geom_plugin[i]
    m.geom_plugin_index[i] = len(m.plugin)
    m.plugin.append(mjm.plugin[p])
    start = mjm.plugin_attradr[p]
    end = mjm.plugin_attradr[p + 1] if p + 1 < mjm.nplugin else len(mjm.plugin_attr)
    values = mjm.plugin_attr[start:end]
    attr_values = []
    current = []
    for v in values:
      if v == 0:
        if current:
          s = "".join(chr(int(x)) for x in current)
          attr_values.append(float(s))
          current = []
      else:
        current.append(v)
    if len(attr_values) > types._NPLUGINATTR:
      raise ValueError(f"Plugin has {len(attr_values)} attributes, which exceeds the maximum of {types._NPLUGINATTR}. ")
    # pad with zeros to _NPLUGINATTR
    attr_values += [0.0] * (types._NPLUGINATTR - len(attr_values))
    m.plugin_attr.append(attr_values[: types._NPLUGINATTR])

  # equality constraint addresses
  m.eq_connect_adr = np.nonzero(mjm.eq_type == types.EqType.CONNECT)[0]
  m.eq_wld_adr = np.nonzero(mjm.eq_type == types.EqType.WELD)[0]
  m.eq_jnt_adr = np.nonzero(mjm.eq_type == types.EqType.JOINT)[0]
  m.eq_ten_adr = np.nonzero(mjm.eq_type == types.EqType.TENDON)[0]
  m.eq_flex_adr = np.nonzero(mjm.eq_type == types.EqType.FLEX)[0]
  m.eq_flexstrain_adr = np.nonzero(mjm.eq_type == types.EqType.FLEXSTRAIN)[0]
  m.neq_flexstrain = m.eq_flexstrain_adr.size
  # Precompute flex strain Jacobian sparsity pattern
  flexstrain_J_rownnz = []
  flexstrain_J_colind = []

  if m.neq_flexstrain > 0:
    for eqstrainid, eqid in enumerate(m.eq_flexstrain_adr):
      f = int(mjm.eq_obj1id[eqid])
      order = int(mjm.flex_interp[f])
      order_abs = abs(order)
      ci = int(mjm.eq_data[eqid, 0])
      cj = int(mjm.eq_data[eqid, 1])
      ck = int(mjm.eq_data[eqid, 2])

      cellnum = mjm.flex_cellnum[f]
      cy = cellnum[1]
      cz = cellnum[2]
      nstart = mjm.flex_nodeadr[f]
      ny_g = cy * order_abs + 1
      nz_g = cz * order_abs + 1

      if order < 0:
        # Shell mode: 2D bilinear quad
        npc = (order_abs + 1) * (order_abs + 1)
        node_bodies = []
        for idx in range(npc):
          gidx = gather_face_node_index(int(cellnum[0]), int(cy), int(cz), ci, idx, order_abs)
          node_bodies.append(mjm.flex_nodebodyid[nstart + gidx])
      else:
        # Solid mode: 3D trilinear voxel
        node_bodies = [
          mjm.flex_nodebodyid[
            nstart + (ci * order_abs + li) * ny_g * nz_g + (cj * order_abs + lj) * nz_g + (ck * order_abs + lk)
          ]
          for li in range(order_abs + 1)
          for lj in range(order_abs + 1)
          for lk in range(order_abs + 1)
        ]

      active_dof_mask = np.any(body_isdofancestor[node_bodies, :] != 0, axis=0)
      sorted_dofs = np.nonzero(active_dof_mask)[0].tolist()
      flexstrain_J_rownnz.append(len(sorted_dofs))
      flexstrain_J_colind.extend(sorted_dofs)

    m.flexstrain_J_rownnz = np.array(flexstrain_J_rownnz, dtype=np.int32)
    m.flexstrain_J_colind = np.array(flexstrain_J_colind, dtype=np.int32)
    m.flexstrain_J_rowadr = np.cumsum([0] + flexstrain_J_rownnz[:-1], dtype=np.int32)
  else:
    m.flexstrain_J_rownnz = np.zeros((0,), dtype=np.int32)
    m.flexstrain_J_rowadr = np.zeros((0,), dtype=np.int32)
    m.flexstrain_J_colind = np.zeros((0,), dtype=np.int32)

  m.nJfs = m.flexstrain_J_colind.size

  # fixed tendon
  m.tendon_jnt_adr, m.wrap_jnt_adr = [], []
  for i in range(mjm.ntendon):
    adr = mjm.tendon_adr[i]
    if mjm.wrap_type[adr] == mujoco.mjtWrap.mjWRAP_JOINT:
      tendon_num = mjm.tendon_num[i]
      for j in range(tendon_num):
        m.tendon_jnt_adr.append(i)
        m.wrap_jnt_adr.append(adr + j)

  # spatial tendon
  m.tendon_site_pair_adr, m.tendon_geom_adr = [], []
  m.ten_wrapadr_site, m.ten_wrapnum_site = [0], []
  for i, tendon_num in enumerate(mjm.tendon_num):
    adr = mjm.tendon_adr[i]
    # sites
    if (mjm.wrap_type[adr : adr + tendon_num] == mujoco.mjtWrap.mjWRAP_SITE).all():
      if i < mjm.ntendon:
        m.ten_wrapadr_site.append(m.ten_wrapadr_site[-1] + tendon_num)
      m.ten_wrapnum_site.append(tendon_num)
    else:
      if i < mjm.ntendon:
        m.ten_wrapadr_site.append(m.ten_wrapadr_site[-1])
      m.ten_wrapnum_site.append(0)

    # geoms
    for j in range(tendon_num):
      wrap_type = mjm.wrap_type[adr + j]
      if j < tendon_num - 1:
        next_wrap_type = mjm.wrap_type[adr + j + 1]
        if wrap_type == mujoco.mjtWrap.mjWRAP_SITE and next_wrap_type == mujoco.mjtWrap.mjWRAP_SITE:
          m.tendon_site_pair_adr.append(i)
      if wrap_type == mujoco.mjtWrap.mjWRAP_SPHERE or wrap_type == mujoco.mjtWrap.mjWRAP_CYLINDER:
        m.tendon_geom_adr.append(i)

  m.tendon_limited_adr = np.nonzero(mjm.tendon_limited)[0]
  m.wrap_site_adr = np.nonzero(mjm.wrap_type == mujoco.mjtWrap.mjWRAP_SITE)[0]
  m.wrap_site_pair_adr = np.setdiff1d(m.wrap_site_adr[np.nonzero(np.diff(m.wrap_site_adr) == 1)[0]], mjm.tendon_adr[1:] - 1)
  m.wrap_geom_adr = np.nonzero(np.isin(mjm.wrap_type, [mujoco.mjtWrap.mjWRAP_SPHERE, mujoco.mjtWrap.mjWRAP_CYLINDER]))[0]

  # pulley scaling
  m.wrap_pulley_scale = np.ones(mjm.nwrap, dtype=float)
  pulley_adr = np.nonzero(mjm.wrap_type == mujoco.mjtWrap.mjWRAP_PULLEY)[0]
  for tadr, tnum in zip(mjm.tendon_adr, mjm.tendon_num):
    for padr in pulley_adr:
      if tadr <= padr < tadr + tnum:
        m.wrap_pulley_scale[padr : tadr + tnum] = 1.0 / mjm.wrap_prm[padr]

  m.actuator_trntype_body_adr = np.nonzero(mjm.actuator_trntype == mujoco.mjtTrn.mjTRN_BODY)[0]

  # sensor addresses
  m.sensor_pos_adr = np.nonzero(
    (mjm.sensor_needstage == mujoco.mjtStage.mjSTAGE_POS)
    & (mjm.sensor_type != mujoco.mjtSensor.mjSENS_JOINTLIMITPOS)
    & (mjm.sensor_type != mujoco.mjtSensor.mjSENS_TENDONLIMITPOS)
  )[0]
  m.sensor_limitpos_adr = np.nonzero(
    (mjm.sensor_type == mujoco.mjtSensor.mjSENS_JOINTLIMITPOS) | (mjm.sensor_type == mujoco.mjtSensor.mjSENS_TENDONLIMITPOS)
  )[0]
  m.sensor_vel_adr = np.nonzero(
    (mjm.sensor_needstage == mujoco.mjtStage.mjSTAGE_VEL)
    & (mjm.sensor_type != mujoco.mjtSensor.mjSENS_JOINTLIMITVEL)
    & (mjm.sensor_type != mujoco.mjtSensor.mjSENS_TENDONLIMITVEL)
  )[0]
  m.sensor_limitvel_adr = np.nonzero(
    (mjm.sensor_type == mujoco.mjtSensor.mjSENS_JOINTLIMITVEL) | (mjm.sensor_type == mujoco.mjtSensor.mjSENS_TENDONLIMITVEL)
  )[0]
  m.sensor_acc_adr = np.nonzero(
    (mjm.sensor_needstage == mujoco.mjtStage.mjSTAGE_ACC)
    & (
      (mjm.sensor_type != mujoco.mjtSensor.mjSENS_TOUCH)
      | (mjm.sensor_type != mujoco.mjtSensor.mjSENS_JOINTLIMITFRC)
      | (mjm.sensor_type != mujoco.mjtSensor.mjSENS_TENDONLIMITFRC)
      | (mjm.sensor_type != mujoco.mjtSensor.mjSENS_TENDONACTFRC)
    )
  )[0]
  m.sensor_rangefinder_adr = np.nonzero(mjm.sensor_type == mujoco.mjtSensor.mjSENS_RANGEFINDER)[0]
  m.rangefinder_sensor_adr = np.full(mjm.nsensor, -1)
  m.rangefinder_sensor_adr[m.sensor_rangefinder_adr] = np.arange(len(m.sensor_rangefinder_adr))
  m.collision_sensor_adr = np.full(mjm.nsensor, -1)
  m.collision_sensor_adr[sensor_collision_adr] = np.arange(len(sensor_collision_adr))
  m.sensor_touch_adr = np.nonzero(mjm.sensor_type == mujoco.mjtSensor.mjSENS_TOUCH)[0]
  limitfrc_sensors = (mujoco.mjtSensor.mjSENS_JOINTLIMITFRC, mujoco.mjtSensor.mjSENS_TENDONLIMITFRC)
  m.sensor_limitfrc_adr = np.nonzero(np.isin(mjm.sensor_type, limitfrc_sensors))[0]
  m.sensor_e_potential = (mjm.sensor_type == mujoco.mjtSensor.mjSENS_E_POTENTIAL).any()
  m.sensor_e_kinetic = (mjm.sensor_type == mujoco.mjtSensor.mjSENS_E_KINETIC).any()
  m.sensor_tendonactfrc_adr = np.nonzero(mjm.sensor_type == mujoco.mjtSensor.mjSENS_TENDONACTFRC)[0]
  subtreevel_sensors = (mujoco.mjtSensor.mjSENS_SUBTREELINVEL, mujoco.mjtSensor.mjSENS_SUBTREEANGMOM)
  m.sensor_subtree_vel = np.isin(mjm.sensor_type, subtreevel_sensors).any()
  m.sensor_contact_adr = np.nonzero(mjm.sensor_type == mujoco.mjtSensor.mjSENS_CONTACT)[0]
  m.sensor_adr_to_contact_adr = np.clip(np.cumsum(mjm.sensor_type == mujoco.mjtSensor.mjSENS_CONTACT) - 1, a_min=0, a_max=None)
  m.sensor_rne_postconstraint = np.isin(
    mjm.sensor_type,
    [
      mujoco.mjtSensor.mjSENS_ACCELEROMETER,
      mujoco.mjtSensor.mjSENS_FORCE,
      mujoco.mjtSensor.mjSENS_TORQUE,
      mujoco.mjtSensor.mjSENS_FRAMELINACC,
      mujoco.mjtSensor.mjSENS_FRAMEANGACC,
    ],
  ).any()
  m.sensor_rangefinder_bodyid = mjm.site_bodyid[mjm.sensor_objid[mjm.sensor_type == mujoco.mjtSensor.mjSENS_RANGEFINDER]]
  m.taxel_vertadr = [
    j + mjm.mesh_vertadr[mjm.sensor_objid[i]]
    for i in range(mjm.nsensor)
    if mjm.sensor_type[i] == mujoco.mjtSensor.mjSENS_TACTILE
    for j in range(mjm.mesh_vertnum[mjm.sensor_objid[i]])
  ]
  m.taxel_sensorid = [
    i
    for i in range(mjm.nsensor)
    if mjm.sensor_type[i] == mujoco.mjtSensor.mjSENS_TACTILE
    for j in range(mjm.mesh_vertnum[mjm.sensor_objid[i]])
  ]
  tactile_geomid = mjm.sensor_refid[mjm.sensor_type == mujoco.mjtSensor.mjSENS_TACTILE]
  tactile_weldid = mjm.body_weldid[mjm.geom_bodyid[tactile_geomid]]
  unique_tactile_welds = np.unique(tactile_weldid)
  m.ntactileweld = int(len(unique_tactile_welds))
  weld_tactile_id = np.full(mjm.nbody, -1, dtype=np.int32)
  for idx, weld in enumerate(unique_tactile_welds):
    weld_tactile_id[weld] = idx
  m.weld_tactile_id = weld_tactile_id

  # Per-block scalar/tile/sparse layout (see m_block_layout).
  _lay = m_block_layout(mjm)
  m.qLD_block_total = _lay["total"]  # packed dense region length / offset of the LDL region
  m.qLD_block_adr = _lay["dof_adr"]

  scalar_tiles = [
    types.TileSet(
      adr=wp.array(_lay["scalar_tiles"][size], dtype=int),
      size=size,
    )
    for size in sorted(_lay["scalar_tiles"])
  ]
  gather_tiles = [
    types.TileSet(adr=wp.array(_lay["gather_tiles"][size], dtype=int), size=size) for size in sorted(_lay["gather_tiles"])
  ]
  m.M_tiles = tuple(scalar_tiles + gather_tiles)

  # Group sparse LDL updates by tree depth. Block-path DOFs never touch the LDL region.
  sparse_updates, dof_depth = {}, np.zeros(mjm.nv, dtype=int) - 1

  for k in range(mjm.nv):
    # skip diagonal rows
    if mjm.M_rownnz[k] == 1:
      continue
    dof_depth[k] = dof_depth[mjm.dof_parentid[k]] + 1
    if _lay["dof_adr"][k] != types.Q_LD_BLOCK_SPARSE:
      continue
    i = mjm.dof_parentid[k]
    diag_k = mjm.M_rowadr[k] + mjm.M_rownnz[k] - 1
    Madr_ki = diag_k - 1
    while i > -1:
      sparse_updates.setdefault(dof_depth[i], []).append((i, k, Madr_ki))
      i = mjm.dof_parentid[i]
      Madr_ki -= 1
  m.qLD_updates = tuple(wp.array(sparse_updates[i], dtype=wp.vec3i) for i in sorted(sparse_updates))

  # Build concatenated updates for fused kernel
  all_updates_flat = []
  level_offsets = [0]
  for level in sorted(sparse_updates):
    all_updates_flat.extend(sparse_updates[level])
    level_offsets.append(len(all_updates_flat))
  m.qLD_all_updates = all_updates_flat if all_updates_flat else [(0, 0, 0)]
  m.qLD_level_offsets = level_offsets

  # Indices for sparse M_fullm (used in solver). M_fullm_i/j are built by
  # walking dof_parentid for each dof, so for joint types whose internal block
  # MuJoCo stores diagonal-only in the compact (M_rownnz, M_rowadr) layout
  # (e.g. free joints), the chain-aware layout here has more entries per row
  # than the compact layout.
  m.M_fullm_i, m.M_fullm_j = [], []
  for i in range(mjm.nv):
    j = i
    while j > -1:
      m.M_fullm_i.append(i)
      m.M_fullm_j.append(j)
      j = mjm.dof_parentid[j]
  # M_elemid maps (row, col) -> madr index in the native CSR M layout
  M_elemid = np.full((mjm.nv, mjm.nv), -1, dtype=np.int32)
  # M_hinit_i: row index of each CSR entry (its madr is the flat index). The dense Newton H-init
  # uses (M_hinit_i, M_colind) to scatter M's upper triangle into the dense H tile from CSR.
  M_hinit_i = np.zeros(mjm.nC, dtype=np.int32)
  for i in range(mjm.nv):
    rowadr = mjm.M_rowadr[i]
    rownnz = mjm.M_rownnz[i]
    for k in range(rownnz):
      madr = rowadr + k
      col = int(mjm.M_colind[madr])
      M_elemid[i, col] = madr
      M_hinit_i[madr] = i
  m.M_elemid = M_elemid
  m.M_hinit_i = M_hinit_i

  # Precompute per-block gather indices for the dense-block densify (tile_load_indexed). For each
  # dense block and flat slot (row, col), store the CSR address of M[max(i,j), min(i,j)], or nC
  # (out of bounds -> read as 0) for structurally absent pairs. Laid out [block, slot] so the kernel
  # reads slice (block_size^2,) at offset blk * block_size^2.
  for tile in gather_tiles:
    sz = tile.size
    starts = np.array(_lay["gather_tiles"][sz], dtype=np.int32)
    dofs = starts[:, None] + np.arange(sz)[None, :]  # (nblock, sz) global dof per block row
    gi = dofs[:, :, None]  # (nblock, sz, 1)
    gj = dofs[:, None, :]  # (nblock, 1, sz)
    elemid = M_elemid[np.maximum(gi, gj), np.minimum(gi, gj)]  # (nblock, sz, sz), -1 if absent
    elemid = np.where(elemid >= 0, elemid, mjm.nC)
    tile.elemid = wp.array(elemid.reshape(-1).astype(np.int32), dtype=int)

  upper_j, upper_i = np.triu_indices(mjm.nv)
  upper_elemid = M_elemid[upper_i, upper_j]
  valid_mask = upper_elemid != -1
  m.M_fullm_upper_i = upper_j[valid_mask].tolist()
  m.M_fullm_upper_j = upper_i[valid_mask].tolist()
  m.M_fullm_upper_elemid = upper_elemid[valid_mask].tolist()

  # indices for sparse qD_fullm (used in RNE derivatives)
  # D-structure is the full square sparsity pattern (both upper and lower triangle)
  m.qD_fullm_i, m.qD_fullm_j = [], []
  for i in range(mjm.nv):
    rowadr = mjm.D_rowadr[i]
    rownnz = mjm.D_rownnz[i]
    for k in range(rownnz):
      m.qD_fullm_i.append(i)
      m.qD_fullm_j.append(int(mjm.D_colind[rowadr + k]))
  m.nD = mjm.nD

  # Gather-based sparse mul_m: for each row, all (col, madr) including diagonal
  row_elements = [[] for _ in range(mjm.nv)]

  for i in range(mjm.nv):
    rowadr = mjm.M_rowadr[i]
    rownnz = mjm.M_rownnz[i]
    for k in range(rownnz):
      madr = rowadr + k
      col = int(mjm.M_colind[madr])
      row_elements[i].append((col, madr))  # row i gathers M[i,col] * vec[col]
      if i != col:
        row_elements[col].append((i, madr))  # row col gathers M[i,col] * vec[i]

  # Flatten into CSR-like arrays
  m.M_mulm_rowadr = [0]
  m.M_mulm_col = []
  m.M_mulm_madr = []
  for i in range(mjm.nv):
    for col, madr in row_elements[i]:
      m.M_mulm_col.append(col)
      m.M_mulm_madr.append(madr)
    m.M_mulm_rowadr.append(len(m.M_mulm_col))

  m.flexedge_J_rownnz = mjm.flexedge_J_rownnz
  m.flexedge_J_rowadr = mjm.flexedge_J_rowadr
  m.flexedge_J_colind = mjm.flexedge_J_colind.reshape(-1)

  # Populate lookup maps
  flex_elemflexid = np.zeros(mjm.nflexelem, dtype=np.int32)
  flex_shellflexid = np.zeros(mjm.nflexshelldata, dtype=np.int32)
  flex_vertflexid = np.zeros(mjm.nflexvert, dtype=np.int32)
  flex_edgeflexid = np.zeros(mjm.nflexedge, dtype=np.int32)
  flex_shelladr = np.zeros(mjm.nflex, dtype=np.int32)

  if mjm.nflex > 0:
    shell_offset = 0
    for fi in range(mjm.nflex):
      elem_start = mjm.flex_elemadr[fi]
      elem_num = mjm.flex_elemnum[fi]
      flex_elemflexid[elem_start : elem_start + elem_num] = fi

      edge_start = mjm.flex_edgeadr[fi]
      edge_num = mjm.flex_edgenum[fi]
      flex_edgeflexid[edge_start : edge_start + edge_num] = fi

      flex_shelladr[fi] = shell_offset
      shell_num = mjm.flex_shellnum[fi]
      flex_shellflexid[shell_offset : shell_offset + shell_num] = fi
      shell_offset += shell_num

      vert_start = mjm.flex_vertadr[fi]
      vert_num = mjm.flex_vertnum[fi]
      flex_vertflexid[vert_start : vert_start + vert_num] = fi

  m.flex_elemflexid = flex_elemflexid
  m.flex_shellflexid = flex_shellflexid
  m.flex_vertflexid = flex_vertflexid
  m.flex_edgeflexid = flex_edgeflexid
  m.flex_shelladr = flex_shelladr

  flex_bend_interp_map = []
  flex_face_map = []
  flex_faceadr = np.zeros(mjm.nflex, dtype=np.int32)
  if mjm.nflex > 0:
    face_offset = 0
    for fi in range(mjm.nflex):
      flex_faceadr[fi] = face_offset
      order = mjm.flex_interp[fi]
      if order >= 0:
        continue

      bendingadr = mjm.flex_bendingadr[fi]
      if bendingadr >= 0:
        nedge = int(mjm.flex_bending[bendingadr])
        for e in range(nedge):
          flex_bend_interp_map.append((fi, e))

      cx, cy, cz = mjm.flex_cellnum[fi]
      nfaces = 2 * (cy * cz + cx * cz + cx * cy)
      for face_idx in range(nfaces):
        flex_face_map.append((fi, face_idx))
      face_offset += nfaces

  if not flex_bend_interp_map:
    m.nflexbend_interp = 0
    m.flex_bend_interp_map = np.zeros((0, 2), dtype=np.int32)
  else:
    m.nflexbend_interp = len(flex_bend_interp_map)
    m.flex_bend_interp_map = np.array(flex_bend_interp_map, dtype=np.int32)

  if not flex_face_map:
    m.nflexface = 0
    m.flex_face_map = np.zeros((0, 2), dtype=np.int32)
  else:
    m.nflexface = len(flex_face_map)
    m.flex_face_map = np.array(flex_face_map, dtype=np.int32)
  m.flex_faceadr = flex_faceadr

  if m.nflexface > 0:
    flex_face = np.zeros((m.nflexface, 9), dtype=np.int32)
    for face_id, (fi, face_elem_idx) in enumerate(flex_face_map):
      order = mjm.flex_interp[fi]
      order_abs = -order
      cx, cy, cz = mjm.flex_cellnum[fi]
      nstart = mjm.flex_nodeadr[fi]
      npc = (order_abs + 1) * (order_abs + 1)

      for local_idx in range(9):
        if local_idx < npc:
          gidx = gather_face_node_index(int(cx), int(cy), int(cz), int(face_elem_idx), int(local_idx), int(order_abs))
          flex_face[face_id, local_idx] = nstart + gidx
        else:
          flex_face[face_id, local_idx] = -1
    m.flex_face = flex_face
  else:
    m.flex_face = np.zeros((0, 9), dtype=np.int32)

  sizes = {f.name: getattr(m, f.name) for f in dataclasses.fields(types.Model) if f.type is int}
  sizes.update(
    {
      "nbody_branches": len(m.body_branches),
      "nbranch_start": len(m.body_branch_start),
      "nbodyfree": sum(body_is_free),
      "nbody_fluid_ellipsoid": len(m.body_fluid_ellipsoid_adr),
      "nbody_fluid_box": len(m.body_fluid_box_adr),
      "njnt_limited_slide_hinge": len(m.jnt_limited_slide_hinge_adr),
      "njnt_limited_ball": len(m.jnt_limited_ball_adr),
      "ndof_tri": len(m.dof_tri_row),
      "nnxn_geom_pair": len(m.nxn_geom_pair),
      "nnxn_geom_pair_filtered": len(m.nxn_geom_pair_filtered),
      "neq_connect": len(m.eq_connect_adr),
      "neq_wld": len(m.eq_wld_adr),
      "neq_jnt": len(m.eq_jnt_adr),
      "neq_ten": len(m.eq_ten_adr),
      "neq_flex": len(m.eq_flex_adr),
      "ntendon_jnt": len(m.tendon_jnt_adr),
      "ntendon_site_pair": len(m.tendon_site_pair_adr),
      "ntendon_geom": len(m.tendon_geom_adr),
      "ntendon_limited": len(m.tendon_limited_adr),
      "nten_wrapadr_site": len(m.ten_wrapadr_site),
      "nwrap_jnt": len(m.wrap_jnt_adr),
      "nwrap_site": len(m.wrap_site_adr),
      "nwrap_site_pair": len(m.wrap_site_pair_adr),
      "nwrap_geom": len(m.wrap_geom_adr),
      "nsensor_pos": len(m.sensor_pos_adr),
      "nsensor_limitpos": len(m.sensor_limitpos_adr),
      "nsensor_vel": len(m.sensor_vel_adr),
      "nsensor_limitvel": len(m.sensor_limitvel_adr),
      "nsensor_acc": len(m.sensor_acc_adr),
      "nsensor_touch": len(m.sensor_touch_adr),
      "nsensor_limitfrc": len(m.sensor_limitfrc_adr),
      "nsensor_tendonactfrc": len(m.sensor_tendonactfrc_adr),
      "nsensor_collision_start_adr": len(m.sensor_collision_start_adr),
      "nqLD_all_updates": len(m.qLD_all_updates),
      "nqLD_level_offsets": len(m.qLD_level_offsets),
      "nM_fullm": len(m.M_fullm_i),
      "nM_fullm_upper": len(m.M_fullm_upper_i),
      "nqD_fullm": len(m.qD_fullm_i),
      "nv_plus_1": len(m.M_mulm_rowadr),
      "nM_mulm": len(m.M_mulm_col),
    }
  )
  for f in dataclasses.fields(types.Model):
    if is_array_spec(f.type):
      batch_size = batch_sizes.get(f.name, 1)
      setattr(m, f.name, _create_array(getattr(m, f.name), f.type, sizes, batch_size))

  mark_batched(m)
  return m


def _get_padded_sizes(nv: int, njmax: int, is_sparse: bool, tile_size: int, augment_cholesky: bool = False):
  def round_up(x, multiple):
    return ((x + multiple - 1) // multiple) * multiple

  njmax_padded = round_up(njmax, tile_size)
  nv_padded = round_up(nv + int(augment_cholesky), tile_size) if (is_sparse or nv > 32) else round_up(nv, 4)

  return njmax_padded, nv_padded


def _nvmax_pad(nvmax: int) -> int:
  """Reserve an augmented column and round nvmax up to the dense tile size."""
  t = types.TILE_SIZE_JTDAJ_DENSE
  return ((max(nvmax, 1) + t) // t) * t


def _default_nconmax(mjm: mujoco.MjModel, mjd: Optional[mujoco.MjData] = None) -> int:
  """Returns a default guess for an ideal nconmax given a Model and optional Data.

  This guess is based off a very simple heuristic, and may need to be manually raised if MJWarp
  reports ncon overflow, or lowered in order to get the very best performance.
  """
  valid_sizes = (2 + (np.arange(19) % 2)) * (2 ** (np.arange(19) // 2 + 3))  # 16, 24, 32, 48, ... 8192
  has_sdf = (mjm.geom_type == mujoco.mjtGeom.mjGEOM_SDF).any()
  has_flex = mjm.nflex > 0
  nconmax = max(mjm.nv * 0.35 * (mjm.nhfield > 0) * 10 + 45, 256 * has_flex, 64 * has_sdf, mjd.ncon if mjd else 0)
  if nconmax > valid_sizes[-1]:
    return int(nconmax)
  return int(valid_sizes[np.searchsorted(valid_sizes, nconmax)])


def _default_njmax(mjm: mujoco.MjModel, mjd: Optional[mujoco.MjData] = None) -> int:
  """Returns a default guess for an ideal njmax given a Model and optional Data.

  This guess is based off a very simple heuristic, and may need to be manually raised if MJWarp
  reports ncon overflow, or lowered in order to get the very best performance.
  """
  valid_sizes = (2 + (np.arange(19) % 2)) * (2 ** (np.arange(19) // 2 + 3))  # 16, 24, 32, 48, ... 8192
  has_sdf = (mjm.geom_type == mujoco.mjtGeom.mjGEOM_SDF).any()
  has_flex = mjm.nflex > 0
  njmax = max(mjm.nv * 2.26 * (mjm.nhfield > 0) * 18 + 53, 512 * has_flex, 256 * has_sdf, mjd.nefc if mjd else 0)
  if njmax > valid_sizes[-1]:
    return int(njmax)
  return int(valid_sizes[np.searchsorted(valid_sizes, njmax)])


def _body_pair_nnz(mjm: mujoco.MjModel, body1: int, body2: int) -> int:
  """Returns the number of unique DOFs in the kinematic tree union of two bodies."""
  body1 = mjm.body_weldid[body1]
  body2 = mjm.body_weldid[body2]
  da1 = mjm.body_dofadr[body1] + mjm.body_dofnum[body1] - 1
  da2 = mjm.body_dofadr[body2] + mjm.body_dofnum[body2] - 1
  nnz = 0
  while da1 >= 0 or da2 >= 0:
    da = max(da1, da2)
    if da1 == da:
      da1 = mjm.dof_parentid[da1]
    if da2 == da:
      da2 = mjm.dof_parentid[da2]
    nnz += 1
  return nnz


def _body_set_nnz(mjm: mujoco.MjModel, bodies) -> int:
  """Returns the number of unique DOFs in the kinematic tree union of a set of bodies."""
  active_dofs = set()
  for b in bodies:
    b = mjm.body_weldid[b]
    da = mjm.body_dofadr[b] + mjm.body_dofnum[b] - 1
    while da >= 0:
      active_dofs.add(da)
      da = mjm.dof_parentid[da]
  return len(active_dofs)


def _calculate_max_contact_nnz(mjm: mujoco.MjModel, body_pairs: Sequence[Tuple[int, int]] = ()) -> int:
  """Returns the maximum number of non-zeros for a single contact constraint.

  body_pairs: pairs of bodies that collide outside the model's own geom pairs (Lodos's surfaces, whose
    geoms make_sim takes out of the model's collision).
  """
  max_contact_nnz = 0

  for b1, b2 in body_pairs:
    max_contact_nnz = max(max_contact_nnz, _body_pair_nnz(mjm, int(b1), int(b2)))

  # contact pairs
  for i in range(mjm.npair):
    g1, g2 = mjm.pair_geom1[i], mjm.pair_geom2[i]
    b1, b2 = mjm.geom_bodyid[g1], mjm.geom_bodyid[g2]
    max_contact_nnz = max(max_contact_nnz, _body_pair_nnz(mjm, b1, b2))

  # filter geom-geom pairs (unique body pairs, filtered)
  body_pair_seen = set()
  for i in range(mjm.ngeom):
    bi = mjm.geom_bodyid[i]
    cti, cai = mjm.geom_contype[i], mjm.geom_conaffinity[i]
    for j in range(i + 1, mjm.ngeom):
      bj = mjm.geom_bodyid[j]
      if bi == bj:
        continue
      if mjm.body_weldid[bi] == 0 and mjm.body_weldid[bj] == 0:
        continue
      bp = (min(bi, bj), max(bi, bj))
      if bp in body_pair_seen:
        continue
      ctj, caj = mjm.geom_contype[j], mjm.geom_conaffinity[j]
      if not ((cti & caj) or (ctj & cai)):
        continue
      body_pair_seen.add(bp)
      max_contact_nnz = max(max_contact_nnz, _body_pair_nnz(mjm, bi, bj))

  if mjm.nflex == 0:
    return max_contact_nnz

  # Compute upper bound NNZ contribution for each flex individually
  flex_nnz = [0] * mjm.nflex
  for fi in range(mjm.nflex):
    if mjm.flex_interp[fi] == 0:
      vert_start = mjm.flex_vertadr[fi]
      dim = mjm.flex_dim[fi]
      elem_num = mjm.flex_elemnum[fi]
      elem_data_start = mjm.flex_elemdataadr[fi]
      if elem_num > 0:
        for e in range(elem_num):
          elem_bodies = {
            mjm.flex_vertbodyid[vert_start + mjm.flex_elem[elem_data_start + e * (dim + 1) + k]] for k in range(dim + 1)
          }
          flex_nnz[fi] = max(flex_nnz[fi], _body_set_nnz(mjm, elem_bodies))
      else:
        for v in range(mjm.flex_vertnum[fi]):
          flex_nnz[fi] = max(
            flex_nnz[fi],
            _body_set_nnz(mjm, {mjm.flex_vertbodyid[vert_start + v]}),
          )
    else:
      order = abs(mjm.flex_interp[fi])
      is_shell = mjm.flex_interp[fi] < 0
      cx, cy, cz = mjm.flex_cellnum[fi]
      nstart = mjm.flex_nodeadr[fi]
      dim = mjm.flex_dim[fi]
      nx = cx * order + 1
      ny = cy * order + 1 if dim > 1 else 1
      nz = cz * order + 1 if dim > 2 else 1

      ci, cj, ck = cx // 2, cy // 2, cz // 2
      cell_bodies = set()

      for li in range(order + 1):
        for lj in range(order + 1 if dim > 1 else 1):
          for lk in range(order + 1 if dim > 2 else 1):
            gi = ci + li
            gj = cj + lj
            gk = ck + lk

            is_interior = False
            if is_shell:
              is_interior = (
                (gi > 0 and gi < cx * order)
                and (gj > 0 and gj < cy * order if dim > 1 else True)
                and (gk > 0 and gk < cz * order if dim > 2 else True)
              )

            if is_interior:
              for bi in (0, gi, nx - 1):
                for bj in (0, gj, ny - 1 if dim > 1 else 0):
                  for bk in (0, gk, nz - 1 if dim > 2 else 0):
                    if (
                      bi == 0
                      or bi == nx - 1
                      or (dim > 1 and (bj == 0 or bj == ny - 1))
                      or (dim > 2 and (bk == 0 or bk == nz - 1))
                    ):
                      node_idx = bi * ny * nz + bj * nz + bk
                      cell_bodies.add(mjm.flex_nodebodyid[nstart + node_idx])
            else:
              node_idx = gi * ny * nz + gj * nz + gk
              cell_bodies.add(mjm.flex_nodebodyid[nstart + node_idx])

      flex_nnz[fi] = _body_set_nnz(mjm, cell_bodies)

  geom_nnz = [_body_set_nnz(mjm, {mjm.geom_bodyid[g]}) for g in range(mjm.ngeom)]

  for fi in range(mjm.nflex):
    fct = mjm.flex_contype[fi]
    fca = mjm.flex_conaffinity[fi]

    # flex-geom contacts
    for g in range(mjm.ngeom):
      ct, ca = mjm.geom_contype[g], mjm.geom_conaffinity[g]
      if (fct & ca) or (ct & fca):
        max_contact_nnz = max(max_contact_nnz, flex_nnz[fi] + geom_nnz[g])

    # flex self-collision
    if mjm.flex_selfcollide[fi] and (fct & fca):
      max_contact_nnz = max(max_contact_nnz, 2 * flex_nnz[fi])

    # flex-flex collision
    for fj in range(fi + 1, mjm.nflex):
      fct_j = mjm.flex_contype[fj]
      fca_j = mjm.flex_conaffinity[fj]
      if (fct & fca_j) or (fct_j & fca):
        max_contact_nnz = max(max_contact_nnz, flex_nnz[fi] + flex_nnz[fj])

  return max_contact_nnz


def _default_njmax_nnz(mjm: mujoco.MjModel, nconmax: int, njmax: int, body_pairs: Sequence[Tuple[int, int]] = ()) -> int:
  """Returns a heuristic estimate for the number of non-zeros in the sparse constraint Jacobian.

  Assumes all equality, friction, and limit constraints are active and computes
  their non-zeros. For contacts, assumes njmax contact rows at the maximum
  body-pair non-zeros from all enabled collision pairs.

  Args:
    mjm: The model containing kinematic and dynamic information (host).
    nconmax: Maximum number of contacts per world.
    njmax: Maximum number of constraint rows per world.
    body_pairs: pairs of bodies that also collide (`_calculate_max_contact_nnz`).

  Returns:
    Estimated number of non-zeros in the constraint Jacobian.
  """
  total_nnz = 0

  def _eq_bodies(i):
    """Returns body pair for equality constraint i."""
    obj1id, obj2id = mjm.eq_obj1id[i], mjm.eq_obj2id[i]
    if mjm.eq_objtype[i] == mujoco.mjtObj.mjOBJ_SITE:
      return mjm.site_bodyid[obj1id], mjm.site_bodyid[obj2id]
    return obj1id, obj2id

  # equality constraints (assume all active)
  for i in range(mjm.neq):
    eq_type = mjm.eq_type[i]

    if eq_type == mujoco.mjtEq.mjEQ_CONNECT:
      total_nnz += 3 * _body_pair_nnz(mjm, *_eq_bodies(i))

    elif eq_type == mujoco.mjtEq.mjEQ_WELD:
      total_nnz += 6 * _body_pair_nnz(mjm, *_eq_bodies(i))

    elif eq_type == mujoco.mjtEq.mjEQ_JOINT:
      total_nnz += 2 if mjm.eq_obj2id[i] >= 0 else 1

    elif eq_type == mujoco.mjtEq.mjEQ_TENDON:
      obj1id = mjm.eq_obj1id[i]
      obj2id = mjm.eq_obj2id[i]
      rownnz1 = mjm.ten_J_rownnz[obj1id] if obj1id < mjm.ntendon else 0
      if obj2id >= 0 and obj2id < mjm.ntendon:
        rowadr1 = mjm.ten_J_rowadr[obj1id]
        rowadr2 = mjm.ten_J_rowadr[obj2id]
        rownnz2 = mjm.ten_J_rownnz[obj2id]
        cols = set()
        for j in range(rownnz1):
          cols.add(mjm.ten_J_colind[rowadr1 + j])
        for j in range(rownnz2):
          cols.add(mjm.ten_J_colind[rowadr2 + j])
        total_nnz += len(cols)
      else:
        total_nnz += rownnz1

    elif eq_type == mujoco.mjtEq.mjEQ_FLEX:
      obj1id = mjm.eq_obj1id[i]
      if obj1id < mjm.nflex:
        edge_start = mjm.flex_edgeadr[obj1id]
        edge_count = mjm.flex_edgenum[obj1id]
        for e in range(edge_count):
          total_nnz += mjm.flexedge_J_rownnz[edge_start + e]

    elif eq_type == mujoco.mjtEq.mjEQ_FLEXSTRAIN:
      # strain constraints: each cell produces neig rows, each dense (nv)
      obj1id = mjm.eq_obj1id[i]
      if obj1id < mjm.nflex:
        # estimate neig from stiffness data
        adr = mjm.flex_stiffnessadr[obj1id]
        neig = int(mjm.flex_stiffness[adr])
        total_nnz += neig * mjm.nv

  # friction constraints
  total_nnz += (mjm.dof_frictionloss > 0).sum()
  for i in range(mjm.ntendon):
    if mjm.tendon_frictionloss[i] > 0:
      total_nnz += mjm.ten_J_rownnz[i]

  # limit constraints (assume all active)
  for i in range(mjm.njnt):
    if mjm.jnt_limited[i]:
      jnt_type = int(mjm.jnt_type[i])
      if jnt_type == mujoco.mjtJoint.mjJNT_BALL:
        total_nnz += 3
      elif jnt_type in (mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_HINGE):
        total_nnz += 1
  for i in range(mjm.ntendon):
    if mjm.tendon_limited[i]:
      total_nnz += mjm.ten_J_rownnz[i]

  # contact constraints: njmax rows at max contact non-zeros
  max_contact_nnz = _calculate_max_contact_nnz(mjm, body_pairs)
  total_nnz += njmax * max_contact_nnz

  return int(min(max(total_nnz, 1), njmax * mjm.nv))


def _resolve_batch_size(na: int | None, n: int | None, nworld: int, default: int) -> int:
  if na is not None:
    return na
  if n is not None:
    return n * nworld
  return default


def _allocate_island_arrays(
  mjm: mujoco.MjModel,
  d: types.Data,
  nworld: int,
  njmax: int,
  mjd: mujoco.MjData,
):
  ntree = mjm.ntree
  nv = mjm.nv
  nisland = int(mjd.nisland)
  nidof = int(mjd.nidof)
  nefc = int(mjd.nefc)

  d.nisland = wp.array(np.full(nworld, nisland, dtype=np.int32), dtype=int)
  d.nidof = wp.array(np.full(nworld, nidof, dtype=np.int32), dtype=int)

  tree_island = np.tile(mjd.tree_island[:ntree], (nworld, 1)) if nisland > 0 else np.full((nworld, ntree), -1, dtype=np.int32)
  d.tree_island = wp.array(tree_island, dtype=int)
  dof_island = np.tile(mjd.dof_island[:nv], (nworld, 1)) if nisland > 0 else np.full((nworld, nv), -1, dtype=np.int32)
  d.dof_island = wp.array(dof_island, dtype=int)

  # Island arrays sized by ntree
  for name in (
    "island_dofadr",
    "island_idofadr",
    "island_nv",
    "island_nefc",
    "island_ne",
    "island_nf",
    "island_iefcadr",
  ):
    arr = np.zeros((nworld, ntree), dtype=np.int32)
    if nisland > 0:
      arr[:, :nisland] = getattr(mjd, name)[:nisland]
    setattr(d, name, wp.array(arr, dtype=int))

  # DOF mapping arrays sized by nv
  map_dof2idof = np.tile(mjd.map_dof2idof[:nv], (nworld, 1))
  map_idof2dof = np.tile(mjd.map_idof2dof[:nv], (nworld, 1))
  dof_islandid = np.full((nworld, nv), -1, dtype=np.int32)
  if nisland > 0 and nidof > 0:
    dof_islandid[:, :nidof] = mjd.dof_island[mjd.map_idof2dof[:nidof]]

  d.map_dof2idof = wp.array(map_dof2idof, dtype=int)
  d.map_idof2dof = wp.array(map_idof2dof, dtype=int)
  d.dof_islandid = wp.array(dof_islandid, dtype=int)

  # Constraint mapping arrays sized by njmax
  map_efc2iefc = np.zeros((nworld, njmax), dtype=np.int32)
  map_iefc2efc = np.zeros((nworld, njmax), dtype=np.int32)
  efc_islandid = np.full((nworld, njmax), -1, dtype=np.int32)

  if nefc > 0:
    map_efc2iefc[:, :nefc] = mjd.map_efc2iefc[:nefc]
    map_iefc2efc[:, :nefc] = mjd.map_iefc2efc[:nefc]
    if nisland > 0:
      total_iefc = int(mjd.island_iefcadr[nisland - 1] + mjd.island_nefc[nisland - 1])
      efc_islandid[:, :total_iefc] = mjd.efc_island[mjd.map_iefc2efc[:total_iefc]]

  d.map_efc2iefc = wp.array(map_efc2iefc, dtype=int)
  d.map_iefc2efc = wp.array(map_iefc2efc, dtype=int)
  d.efc_islandid = wp.array(efc_islandid, dtype=int)


_COMPACT_DATA_FIELDS: tuple[str, ...] = (
  "ctol",
  "cls_tol",
  "cdof_tri_row",
  "cdof_tri_col",
  "cM",
  "cqLD",
  "crhs",
  "cx",
  "cJ",
  "cMa",
  "cqfrc_smooth",
  "cqacc_smooth",
  "cqacc_warmstart",
  "cqacc",
  "cqfrc_constraint",
)


def _allocate_compact_arrays(
  mjm: mujoco.MjModel,
  d: types.Data,
  nworld: int,
  nvmax_pad: int,
  njmax_pad: int,
  compact: bool,
):
  """Allocate workspace for the compacted dense factor/solve (when nvmax is requested).

  Mirrors the island-local ``i*`` Data fields with a ``c*`` (compact) prefix. The
  constant model-shadows (tolerances, dof-pair indices) are derived on the host since
  they depend only on nvmax_pad; the workspace shadows are sized by nvmax_pad so the
  blocked Cholesky never reads out of bounds on its partial tile. When the user does not
  request compaction (nvmax is None) everything is allocated empty.

  TODO(team): once the compact path replaces the island solver, the whole forward
  pipeline can run in compacted space and ``d.M`` / ``d.qacc`` etc. become nvmax-sized
  directly, collapsing these ``c*`` shadows into the primary Data fields.
  """
  nw = nworld if compact else 0
  nvp = nvmax_pad if compact else 0
  njp = njmax_pad if compact else 0

  if compact:
    # match the float32 tolerance clamp applied in put_model; rescale by nv/nvmax_pad so
    # the solver's nv-normalized convergence test matches the full-model baseline.
    scale = float(mjm.nv) / float(nvmax_pad)
    tol = max(float(mjm.opt.tolerance), 1e-6)
    ls_tol = float(mjm.opt.ls_tolerance)
    d.ctol = wp.array([tol * scale], dtype=float)
    d.cls_tol = wp.array([ls_tol * scale], dtype=float)
    # all (i, j) DOF pairs of the nvmax_pad-wide compacted Hessian (the global dof_tri,
    # triu over full nv, would index out of bounds).
    idx = np.arange(nvmax_pad, dtype=np.int32)
    d.cdof_tri_row = wp.array(np.repeat(idx, nvmax_pad), dtype=int)
    d.cdof_tri_col = wp.array(np.tile(idx, nvmax_pad), dtype=int)
  else:
    d.ctol = wp.empty(0, dtype=float)
    d.cls_tol = wp.empty(0, dtype=float)
    d.cdof_tri_row = wp.empty(0, dtype=int)
    d.cdof_tri_col = wp.empty(0, dtype=int)

  alloc_cJ = compact and not is_sparse(mjm)

  d.cM = wp.empty((nw, nvp, nvp), dtype=float)
  d.cqLD = wp.empty((nw, nvp, nvp), dtype=float)
  d.crhs = wp.empty((nw, nvp, 1), dtype=float)
  d.cx = wp.empty((nw, nvp, 1), dtype=float)
  d.cJ = wp.empty((nw, njp, nvp), dtype=float) if alloc_cJ else wp.empty((0, 0, 0), dtype=float)
  d.cMa = wp.empty((nw, nvp), dtype=float)
  d.cqfrc_smooth = wp.empty((nw, nvp), dtype=float)
  d.cqacc_smooth = wp.empty((nw, nvp), dtype=float)
  d.cqacc_warmstart = wp.empty((nw, nvp), dtype=float)
  d.cqacc = wp.empty((nw, nvp), dtype=float)
  d.cqfrc_constraint = wp.empty((nw, nvp), dtype=float)


def put_data(
  mjm: mujoco.MjModel,
  mjd: mujoco.MjData,
  nworld: int = 1,
  nconmax: Optional[int] = None,
  nccdmax: Optional[int] = None,
  njmax: Optional[int] = None,
  njmax_nnz: Optional[int] = None,
  naconmax: Optional[int] = None,
  naccdmax: Optional[int] = None,
  nvmax: Optional[int] = None,
  contact_body_pairs: Sequence[Tuple[int, int]] = (),
) -> types.Data:
  """Moves data from host to a device.

  Args:
    mjm: The model containing kinematic and dynamic information (host).
    mjd: The data object containing current state and output arrays (host).
    nworld: The number of worlds.
    nconmax: Number of contacts to allocate per world.  Contacts exist in large
             heterogeneous arrays: one world may have more than nconmax contacts.
    nccdmax: Number of CCD contacts to allocate per world. Same semantics as nconmax.
    njmax: Number of constraints to allocate per world.  Constraint arrays are
           batched by world: no world may have more than njmax constraints.
    njmax_nnz: Number of non-zeros in constraint Jacobian (sparse). Defaults to njmax * nv.
    naconmax: Number of contacts to allocate for all worlds. Overrides nconmax.
    naccdmax: Maximum number of CCD contacts. Defaults to naconmax.
    nvmax: Capacity for compacted active DOFs per world. Defaults to nv.
    contact_body_pairs: pairs of bodies that collide outside the model's own geom pairs (Lodos's
      surfaces), counted in the sparse Jacobian's default njmax_nnz.

  Returns:
    The data object containing the current state and output arrays (device).
  """
  # TODO(team): move nconmax and njmax to Model?
  # TODO(team): decide what to do about uninitialized warp-only fields created by put_data
  #             we need to ensure these are only workspace fields and don't carry state

  if nconmax is None:
    nconmax = _default_nconmax(mjm, mjd)

  if njmax is None:
    njmax = _default_njmax(mjm, mjd)

  sleep_enabled = bool(mjm.opt.enableflags & mujoco.mjtEnableBit.mjENBL_SLEEP) and not bool(
    mjm.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_ISLAND
  )
  compact_alloc = sleep_enabled or (nvmax is not None)

  if nvmax is None:
    nvmax = mjm.nv

  if nconmax < 0:
    raise ValueError("nconmax must be >= 0")

  if njmax < 0:
    raise ValueError("njmax must be >= 0")

  if nvmax < 0 or nvmax > mjm.nv:
    raise ValueError(f"nvmax ({nvmax}) must be in [0, nv ({mjm.nv})]")

  if nworld < 1:
    raise ValueError(f"nworld must be >= 1")

  naconmax_is_input = naconmax is not None
  naconmax = _resolve_batch_size(naconmax, nconmax, nworld, 0)
  if naconmax < 0:
    raise ValueError("naconmax must be >= 0")

  if not naconmax_is_input and mjd.ncon > nconmax:
    raise ValueError(f"nconmax overflow (nconmax must be >= {mjd.ncon})")
  elif naconmax < mjd.ncon * nworld:
    raise ValueError(f"naconmax overflow (naconmax must be >= {mjd.ncon * nworld})")

  naccdmax = _resolve_batch_size(naccdmax, nccdmax, nworld, naconmax)

  if naccdmax < 0:
    raise ValueError("naccdmax must be >= 0")
  elif naccdmax > naconmax:
    raise ValueError(f"naccdmax ({naccdmax}) must be <= naconmax ({naconmax})")

  if nccdmax is None:
    nccdmax = nconmax
  else:
    if nccdmax < 0:
      raise ValueError("nccdmax must be >= 0")
    elif nccdmax > nconmax:
      raise ValueError(f"nccdmax ({nccdmax}) must be <= nconmax ({nconmax})")

  if mjd.nefc > njmax:
    raise ValueError(f"njmax overflow (njmax must be >= {mjd.nefc})")

  nv_compact = nvmax < mjm.nv

  sizes = dict({"*": 1}, **{f.name: getattr(mjm, f.name, None) for f in dataclasses.fields(types.Model) if f.type is int})
  condim_arrays = [np.array([0]), mjm.geom_condim, mjm.pair_dim]
  if mjm.nflex > 0:
    condim_arrays.append(mjm.flex_condim)
  sizes["nmaxcondim"] = np.concatenate(condim_arrays).max()
  sizes["nmaxpyramid"] = np.maximum(1, 2 * (sizes["nmaxcondim"] - 1))
  tile_size = types.TILE_SIZE_JTDAJ_SPARSE if is_sparse(mjm) else types.TILE_SIZE_JTDAJ_DENSE
  sizes["njmax_pad"], sizes["nv_pad"] = _get_padded_sizes(
    mjm.nv,
    njmax,
    is_sparse(mjm),
    tile_size,
    augment_cholesky=mjm.opt.solver == mujoco.mjtSolver.mjSOL_NEWTON and mjm.nv > 32,
  )
  sizes["nworld"] = nworld
  sizes["naconmax"] = naconmax
  sizes["njmax"] = njmax
  sizes["nvmax"] = nvmax
  sizes["nvmax_pad"] = _nvmax_pad(nvmax)
  sizes["nvmax_pad_sq"] = sizes["nvmax_pad"] * sizes["nvmax_pad"]
  sizes["nflexface"] = _get_nflexface(mjm)
  sizes["ngroupcon"] = sizes["ngroupmember"] = 0  # contact groups: alloc_contact_groups

  if njmax_nnz is None:
    if is_sparse(mjm):
      njmax_nnz = _default_njmax_nnz(mjm, nconmax, njmax, contact_body_pairs)
    else:
      njmax_nnz = njmax * mjm.nv

  # Capture sleep state before mj_kinematics, which resets tree_asleep as a side effect.
  tree_asleep_init = mjd.tree_asleep.copy()
  body_awake_init = mjd.body_awake.copy()

  # Ensure kinematic state is populated. mujoco.MjData() does not call mj_kinematics, so a freshly
  # created mjd has zero geom positions. Static geoms are never updated by the physics loop
  # (see smooth.py geom_kinematics), so without this call they would remain at (0,0,0).
  mujoco.mj_kinematics(mjm, mjd)

  # create contact
  contact_kwargs = {"efc_address": None, "worldid": None, "type": None, "geomcollisionid": None}
  for f in dataclasses.fields(types.Contact):
    if f.name in contact_kwargs:
      continue
    if f.name in ["flex", "elem", "vert"] and mjm.nflex == 0:
      contact_kwargs[f.name] = wp.empty(0, dtype=wp.vec2i)
      continue
    val = getattr(mjd.contact, f.name)
    val = np.tile(val, (nworld,) + (1,) * (val.ndim - 1))
    width = ((0, naconmax - val.shape[0]),) + ((0, 0),) * (val.ndim - 1)
    val = np.pad(val, width)
    contact_kwargs[f.name] = _create_array(val, f.type, sizes)

  contact = types.Contact(**contact_kwargs)

  contact.efc_address = np.full((naconmax, sizes["nmaxpyramid"]), -1, dtype=int)
  for i in range(mjd.ncon):
    efc_address = mjd.contact.efc_address[i]
    if efc_address == -1:
      continue
    condim = mjd.contact.dim[i]
    ndim = max(1, 2 * (condim - 1)) if mjm.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL else condim
    for j in range(nworld):
      contact.efc_address[j * mjd.ncon + i, :ndim] = efc_address + np.arange(ndim)

  contact.efc_address = wp.array(contact.efc_address, dtype=int)
  contact.worldid = np.pad(np.repeat(np.arange(nworld), mjd.ncon), (0, naconmax - nworld * mjd.ncon))
  contact.worldid = wp.array(contact.worldid, dtype=int)
  contact.type = wp.ones((naconmax,), dtype=int)  # TODO(team): set values
  contact.geomcollisionid = wp.empty((naconmax,), dtype=int)  # TODO(team): set values

  # create efc
  efc = _create_constraint(mjm, nworld, njmax, sizes, mjd)

  # make_constraint builds the block list in-kernel; put_data does not run it, so build it here
  # -- otherwise solving a put_data state would assemble an empty J^T D J.
  if is_sparse(mjm) and mjm.opt.solver == mujoco.mjtSolver.mjSOL_NEWTON:
    jtdaj_adr, jtdaj_nrow = _jtdaj_groups(mjd)
    nblock = jtdaj_adr.shape[0]
    adr_row = np.zeros(njmax, dtype=int)
    nrow_row = np.zeros(njmax, dtype=int)
    adr_row[:nblock] = jtdaj_adr
    nrow_row[:nblock] = jtdaj_nrow
    efc.jtdaj_adr = wp.array(np.tile(adr_row, (nworld, 1)), dtype=int)
    efc.jtdaj_nrow = wp.array(np.tile(nrow_row, (nworld, 1)), dtype=int)
    efc.jtdaj_nblock = wp.array(np.full(nworld, nblock, dtype=int), dtype=int)

  if is_sparse(mjm):
    J_rownnz = np.zeros(njmax, dtype=np.int32)
    J_rowadr = np.zeros(njmax, dtype=np.int32)
    J_colind = np.zeros(njmax_nnz, dtype=np.int32)
    J = np.zeros(njmax_nnz, dtype=np.float64)
    if mjd.nefc:
      if mujoco.mj_isSparse(mjm):
        J_rownnz[: mjd.nefc] = mjd.efc_J_rownnz[: mjd.nefc]
        J_rowadr[: mjd.nefc] = mjd.efc_J_rowadr[: mjd.nefc]
        nnz = int(mjd.efc_J_rownnz[: mjd.nefc].sum())
        J_colind[:nnz] = mjd.efc_J_colind[:nnz]
        J[:nnz] = mjd.efc_J[:nnz]
      else:
        dense_J = mjd.efc_J.reshape((-1, mjm.nv))[: mjd.nefc]
        mujoco.mju_dense2sparse(
          J[: mjd.nefc * mjm.nv], dense_J, J_rownnz[: mjd.nefc], J_rowadr[: mjd.nefc], J_colind[: mjd.nefc * mjm.nv]
        )

    efc.J_rownnz = wp.array(np.tile(J_rownnz, (nworld, 1)), dtype=int)
    efc.J_rowadr = wp.array(np.tile(J_rowadr, (nworld, 1)), dtype=int)
    efc.J_colind = wp.array(np.tile(J_colind, (nworld, 1)).reshape((nworld, 1, -1)), dtype=int)
    efc.J = wp.array(np.tile(J, (nworld, 1)).reshape((nworld, 1, -1)), dtype=float)
  else:
    efc.J_rownnz = wp.zeros((nworld, 0), dtype=int)
    efc.J_rowadr = wp.zeros((nworld, 0), dtype=int)
    efc.J_colind = wp.zeros((nworld, 0, 0), dtype=int)

    mj_efc_J = np.zeros((mjd.nefc, mjm.nv))
    if mjd.nefc:
      if mujoco.mj_isSparse(mjm):
        mujoco.mju_sparse2dense(mj_efc_J, mjd.efc_J, mjd.efc_J_rownnz, mjd.efc_J_rowadr, mjd.efc_J_colind)
      else:
        mj_efc_J = mjd.efc_J.reshape((-1, mjm.nv))[: mjd.nefc]
    efc_J = np.zeros((nworld, sizes["njmax_pad"], sizes["nv_pad"]), dtype=float)
    efc_J[:, : mjd.nefc, : mjm.nv] = np.tile(mj_efc_J, (nworld, 1, 1))
    efc.J = wp.array(efc_J, dtype=float)

  # create data
  d_kwargs = {
    "contact": contact,
    "efc": efc,
    "nworld": nworld,
    "naconmax": naconmax,
    "naccdmax": naccdmax,
    "njmax": njmax,
    "nvmax": nvmax,
    "nvmax_pad": sizes["nvmax_pad"],
    "njmax_pad": sizes["njmax_pad"],
    "njmax_nnz": njmax_nnz,
    # fields set after initialization:
    "solver_niter": None,
    "qLD": None,
    "nacon": None,
    # island arrays
    "nisland": None,
    "tree_island": None,
    "dof_island": None,
    "island_dofadr": None,
    "island_idofadr": None,
    "island_nv": None,
    "island_nefc": None,
    "island_ne": None,
    "island_nf": None,
    "island_iefcadr": None,
    "nidof": None,
    "map_dof2idof": None,
    "map_idof2dof": None,
    "map_efc2iefc": None,
    "map_iefc2efc": None,
    "dof_islandid": None,
    "efc_islandid": None,
    # compact arrays (populated by _allocate_compact_arrays; skip eager allocation)
    **{name: None for name in _COMPACT_DATA_FIELDS},
    "tree_asleep": wp.array(np.tile(tree_asleep_init, (nworld, 1)), dtype=int),
    "tree_awake": wp.array(np.tile((tree_asleep_init < 0).astype(np.int32), (nworld, 1)), dtype=int),
    "body_awake": wp.array(np.tile(body_awake_init.astype(np.int32), (nworld, 1)), dtype=int),
  }
  for f in dataclasses.fields(types.Data):
    if f.name in d_kwargs:
      continue
    val = getattr(mjd, f.name, None)
    d_kwargs[f.name] = _create_array(val, f.type, sizes)

  d = types.Data(**d_kwargs)
  d.solver_niter = wp.full((nworld,), mjd.solver_niter[0], dtype=int)
  # radix_sort_pairs needs 2*count capacity in keys and values
  d.prox_zcache_key = wp.zeros((2 * naconmax,), dtype=wp.int64)
  d.prox_zcache_val = wp.zeros((2 * naconmax,), dtype=int)
  d.prox_zcache_n = wp.zeros((1,), dtype=int)
  d.contact_impratio_invsqrt = wp.zeros((naconmax,), dtype=float)

  # qLD = [packed block Cholesky | nC LDL region]. Block factors store their upper Cholesky
  # packed; the LDL region (present iff some block is sparse) holds MuJoCo's full L'DL factor (only
  # its sparse-block entries are read by the solve).
  lay = m_block_layout(mjm)
  qld_total = lay["total"] + (mjm.nC if lay["has_sparse"] else 0)
  qLD = np.zeros(qld_total, dtype=np.float32)
  if lay["total"]:
    Mfull = np.zeros((mjm.nv, mjm.nv))
    mujoco.mju_sym2dense(Mfull, mjd.M, mjm.M_rownnz, mjm.M_rowadr, mjm.M_colind)
    for size, starts in list(lay["scalar_tiles"].items()) + list(lay["gather_tiles"].items()):
      for start in starts:
        off = lay["dof_adr"][start]
        if off < 0:
          continue
        blk = Mfull[start : start + size, start : start + size]
        if blk.any():
          qLD[off : off + size * size] = np.linalg.cholesky(blk).T.reshape(-1)
  if lay["has_sparse"]:
    qLD[lay["total"] :] = mjd.qLD
  d.qLD = wp.array(np.full((nworld, qld_total), qLD), dtype=float)

  _allocate_island_arrays(mjm, d, nworld, njmax, mjd)
  _allocate_compact_arrays(mjm, d, nworld, sizes["nvmax_pad"], sizes["njmax_pad"], compact_alloc)
  d.ncdof.zero_()
  d.dof_cdof.fill_(-1)
  d.cdof_dof.fill_(-1)

  d.nacon = wp.array([mjd.ncon * nworld], dtype=int)

  mark_batched(d)
  return d


def override_model(model: types.Model | mujoco.MjModel, overrides: dict[str, Any] | Sequence[str]):
  """Overrides model parameters.

  Overrides are of the format:
    opt.iterations = 1
    opt.cone = pyramidal
    opt.disableflags = contact | spring
  """
  enum_fields = {
    "opt.broadphase": types.BroadphaseType,
    "opt.broadphase_filter": types.BroadphaseFilter,
    "opt.cone": types.ConeType,
    "opt.disableflags": types.DisableBit,
    "opt.enableflags": types.EnableBit,
    "opt.integrator": types.IntegratorType,
    "opt.solver": types.SolverType,
    "opt.warn_overflow": types.OverflowType,
  }
  # MuJoCo pybind11 enums don't support iteration, so we provide explicit mappings
  mj_enum_fields = {
    "opt.jacobian": {
      "DENSE": mujoco.mjtJacobian.mjJAC_DENSE,
      "SPARSE": mujoco.mjtJacobian.mjJAC_SPARSE,
      "AUTO": mujoco.mjtJacobian.mjJAC_AUTO,
    },
  }
  mjw_only_fields = {
    "opt.broadphase",
    "opt.broadphase_filter",
    "opt.graph_conditional",
    "opt.contact_sensor_maxmatch",
    "opt.warn_overflow",
    "opt.run_collision_detection",
    "opt.run_rne_postconstraint",
    "opt.prox_enable",
    "opt.prox_rho",
    "opt.prox_outer_iterations",
    "opt.prox_inner_iterations",
    "opt.prox_tol_z",
    "opt.prox_tol_sigma",
    "opt.prox_baumgarte",
    "opt.prox_baumgarte_alpha",
    "opt.prox_baumgarte_slop",
    "opt.prox_desaxce",
    "opt.prox_soft",
    "opt.prox_sigma_relax",
    "opt.prox_z_carry",
    "opt.prox_host_convergence",
    "opt.shared_hessian",
  }
  mj_only_fields = {"opt.jacobian", "vis.quality.offsamples"}

  if not isinstance(overrides, dict):
    overrides_dict = {}
    for override in overrides:
      if "=" not in override:
        raise ValueError(f"Invalid override format: {override}")
      k, v = override.split("=", 1)
      overrides_dict[k.strip()] = v.strip()
    overrides = overrides_dict

  for key, val in overrides.items():
    if key == "opt.ls_parallel":
      raise ValueError("ls_parallel was removed in MuJoCo Warp 3.9.1.")
    if key == "opt.ls_parallel_min_step":
      raise ValueError("ls_parallel_min_step was removed in MuJoCo Warp 3.9.1.")

    # skip overrides on MjModel for properties that are only on mjw.Model
    if key in mjw_only_fields and isinstance(model, mujoco.MjModel):
      continue
    if key in mj_only_fields and isinstance(model, types.Model):
      continue

    obj, attrs = model, key.split(".")
    for i, attr in enumerate(attrs):
      if not hasattr(obj, attr):
        raise ValueError(f"Unrecognized model field: {key}")
      if i < len(attrs) - 1:
        obj = getattr(obj, attr)
        continue

      typ = type(getattr(obj, attr))

      if key in mj_enum_fields and isinstance(val, str):
        enum_member = val.strip().upper()
        if enum_member not in mj_enum_fields[key]:
          raise ValueError(f"Unrecognized enum value for {key}: {enum_member}")
        val = mj_enum_fields[key][enum_member]
      elif key in enum_fields and isinstance(val, str):
        # special case: enum value
        enum_members = val.split("|")
        enum_cls = enum_fields[key]
        val = int(getattr(obj, attr)) if any(m.strip().startswith("~") for m in enum_members) else 0
        for enum_member in enum_members:
          enum_member = enum_member.strip().upper()
          is_negated = enum_member.startswith("~")
          name = enum_member[1:].strip() if is_negated else enum_member
          if name not in enum_cls.__members__:
            raise ValueError(f"Unrecognized enum value for {enum_cls.__name__}: {enum_member}")
          if is_negated:
            val &= ~int(enum_cls[name])
          else:
            val |= int(enum_cls[name])
      elif typ is bool and isinstance(val, str):
        # special case: "true", "TRUE", "false", "FALSE" etc.
        if val.upper() not in ("TRUE", "FALSE"):
          raise ValueError(f"Unrecognized value for field: {key}")
        val = val.upper() == "TRUE"
      elif typ is wp.array and isinstance(val, str):
        arr = getattr(obj, attr)
        floats = [float(p) for p in val.strip("[]").split()]
        val = wp.array([arr.dtype(*floats)], dtype=arr.dtype)
      elif typ is np.ndarray and isinstance(val, str):
        arr = getattr(obj, attr)
        val = np.array([float(p) for p in val.strip("[]").split()], dtype=arr.dtype)
      else:
        val = typ(val)

      setattr(obj, attr, val)


# From MuJoCo Warp's support.py.


# From mujoco_warp/_src/support.py.
@wp.func
def get_face_metadata(
  # In:
  cellnum_x: int,
  cellnum_y: int,
  cellnum_z: int,
  face_elem_idx: int,
  order_abs: int,
) -> Tuple[int, int, int, int, int, int]:
  size01 = cellnum_y * cellnum_z
  size23 = cellnum_x * cellnum_z
  size45 = cellnum_x * cellnum_y

  face_id = 0
  within_face = 0

  if face_elem_idx < size01:
    face_id = 0
    within_face = face_elem_idx
  elif face_elem_idx < 2 * size01:
    face_id = 1
    within_face = face_elem_idx - size01
  elif face_elem_idx < 2 * size01 + size23:
    face_id = 2
    within_face = face_elem_idx - 2 * size01
  elif face_elem_idx < 2 * size01 + 2 * size23:
    face_id = 3
    within_face = face_elem_idx - 2 * size01 - size23
  elif face_elem_idx < 2 * size01 + 2 * size23 + size45:
    face_id = 4
    within_face = face_elem_idx - 2 * size01 - 2 * size23
  else:
    face_id = 5
    within_face = face_elem_idx - 2 * size01 - 2 * size23 - size45

  normal_axis = face_id // 2

  c1 = 0
  if face_id == 0 or face_id == 1:
    c1 = cellnum_z
  elif face_id == 2 or face_id == 3:
    c1 = cellnum_x
  else:
    c1 = cellnum_y

  fixed_dim = 0
  if normal_axis == 0:
    fixed_dim = cellnum_x
  elif normal_axis == 1:
    fixed_dim = cellnum_y
  else:
    fixed_dim = cellnum_z
  g_fixed = (face_id % 2) * fixed_dim * order_abs

  q0 = within_face // c1
  q1 = within_face % c1

  ny_g = cellnum_y * order_abs + 1
  nz_g = cellnum_z * order_abs + 1

  return normal_axis, g_fixed, q0, q1, ny_g, nz_g


# From mujoco_warp/_src/support.py.
@wp.func
def gather_face_node_index_fast(
  # In:
  normal_axis: int,
  g_fixed: int,
  q0: int,
  q1: int,
  ny_g: int,
  nz_g: int,
  local_idx: int,
  order_abs: int,
) -> int:
  l0 = local_idx // (order_abs + 1)
  l1 = local_idx % (order_abs + 1)

  g = wp.vec3i(0, 0, 0)
  if normal_axis == 0:
    g = wp.vec3i(g_fixed, q0 * order_abs + l0, q1 * order_abs + l1)
  elif normal_axis == 1:
    g = wp.vec3i(q1 * order_abs + l1, g_fixed, q0 * order_abs + l0)
  else:
    g = wp.vec3i(q0 * order_abs + l0, q1 * order_abs + l1, g_fixed)

  return g[0] * ny_g * nz_g + g[1] * nz_g + g[2]


# From mujoco_warp/_src/support.py.
@wp.func
def gather_face_node_index(
  # In:
  cellnum_x: int,
  cellnum_y: int,
  cellnum_z: int,
  face_elem_idx: int,
  local_idx: int,
  order_abs: int,
) -> int:
  normal_axis, g_fixed, q0, q1, ny_g, nz_g = get_face_metadata(cellnum_x, cellnum_y, cellnum_z, face_elem_idx, order_abs)
  return gather_face_node_index_fast(normal_axis, g_fixed, q0, q1, ny_g, nz_g, local_idx, order_abs)
