"""Host side of the hybrid narrowphase: allocation, launches, graph capture, readback."""

from __future__ import annotations

import numpy as np
import warp as wp


from lodos.col.prim.collision_gjk import GJKResult
from lodos.col.prim.types import MAX_EPAFACES
from lodos.col.prim.types import MAX_EPAHORIZON
from lodos.mjcf.types import GeomType

from . import kernels as K
from . import meshdata
from .epa_warp import LANES
from .epa_warp import epa_warp_kernel

# the path counters of PipeState.stats, by index (kernels.count)
STAT_NAMES = ("pairs", "cache_found", "cert_contact", "cert_separated", "cold", "epa", "epa_fail",
              "feat_fail", "manifold_fallback", "cap_overflow", "epa_facet_decoded", "cold_certified", "seeded_epa", "gjk_unproven_to_epa",
              "gjk_unproven_far", "gjk_unproven_point")
# the pair types of MJWarp's GJK + EPA + multi-CCD clipping that are hulls (not CYLINDER)
HULL_TYPES = ((GeomType.BOX, GeomType.BOX), (GeomType.BOX, GeomType.MESH), (GeomType.MESH, GeomType.MESH))
MESHMESH = ((GeomType.MESH, GeomType.MESH),)  # the mesh-mesh pairs alone, as a `types`
FLOOR_TYPES = ((GeomType.PLANE, GeomType.MESH),)  # MJWarp's plane_convex (a primitive collider), on cell supports
ALL_TYPES = HULL_TYPES + FLOOR_TYPES  # every pair type the hybrid can take


class HybridNarrowphase:
  """Narrowphase for the pair types `types` (a subset of ALL_TYPES: pairs of hulls, and
  plane-mesh) on an MJWarp CollisionContext, into its own contact buffers
  (`lodos.col.prim.driver.PrimCollision.collide` appends them to MJWarp's contacts to drive the simulation).

  Built by `lodos.col.prim.driver.PrimCollision.__init__`; `PrimCollision.collide` launches it and appends its
  contacts to MJWarp's with `lodos.col.prim.driver.export_kernel`.
  """

  def __init__(self, mjm, m, d, ctx, tbits=10, use_cache=True, epa_mode="auto", types=ALL_TYPES, meshes=None):
    """Builds the hull tables and the kernels' model, and allocates every buffer of the pipeline. Raises for pair
    types outside ALL_TYPES, a model with no mesh or box, box sizes that differ between worlds, or an unknown
    epa_mode.

    Called by `lodos.col.prim.driver.PrimCollision.__init__`.
    In: mjm (mujoco.MjModel); m, the MJWarp-style Model: geom_type (ngeom,) int, geom_dataid (1 or nworld, ngeom)
    int, geom_size (1 or nworld, ngeom) wp.vec3, the mesh_* arrays; d, its Data: nworld, naconmax and naccdmax
    (int), ncollision (1,) int, geom_xpos (nworld, ngeom) wp.vec3, geom_xmat (nworld, ngeom) wp.mat33; ctx
    (CollisionContext): collision_pair (naconmax,) wp.vec2i, collision_worldid (naconmax,) int; tbits (int) log2 of
    each world's cache table size; use_cache (bool or int, `kernels.warm_kernel`'s); epa_mode ("auto", "thread" or
    "warp"); types, the pair types to take; meshes (set of int or None) the mesh ids anything can pair
    (`meshdata.build`). The m, d and ctx arrays are bound into self.mm and self.st, not copied.
    """
    self.nworld = d.nworld  # int
    self.naconmax = d.naconmax  # int: the broadphase's pair capacity (and MJWarp's contacts')
    self.cap_epa = d.naccdmax  # int: the EPA list's capacity
    self.cap_man = d.naccdmax  # int: the manifold queue's capacity
    self.cap_con = 4 * d.naccdmax  # int: the contact buffers' capacity
    self.use_cache = int(use_cache)  # int: 0 no cache, 1 the cache and seeded EPA, 2 the cache without seeded EPA
    self.types = tuple((GeomType(a), GeomType(b)) for a, b in types)  # tuple of (GeomType, GeomType): the pair types
    if not set(self.types) <= set(ALL_TYPES):
      raise ValueError(f"types must be a subset of {ALL_TYPES}")
    boxes = []  # list of int: the box geoms
    if any(GeomType.BOX in t for t in self.types):
      boxes = np.nonzero(mjm.geom_type == int(GeomType.BOX.value))[0].tolist()
      size = m.geom_size.numpy()  # (1 or nworld, ngeom, 3)
      if boxes and not np.all(size[:, boxes] == size[:1, boxes]):
        raise NotImplementedError("box sizes that differ between worlds")
    self.tables = meshdata.build(mjm, box_geoms=boxes, meshes=meshes)  # meshdata.HullTables: the host hull tables
    if not self.tables.report["nhull"]:
      raise ValueError("the model has no meshes or boxes")
    self.hull = self.tables.to_device()  # HullTablesDevice: the kernels' hull tables

    mm = K.MeshModel()  # MeshModel: the kernels' model
    mm.geom_type, mm.geom_dataid, mm.geom_size = m.geom_type, m.geom_dataid, m.geom_size
    for f in ("vertadr", "vertnum", "graphadr", "vert", "graph", "polynum", "polyadr", "polynormal",  # f: str
              "polyvertadr", "polyvertnum", "polyvert", "polymapadr", "polymapnum", "polymap"):
      setattr(mm, "mesh_" + f, getattr(m, "mesh_" + f))
    mm.geom_xpos, mm.geom_xmat = d.geom_xpos, d.geom_xmat
    mm.ccd_tolerance = float(mjm.opt.ccd_tolerance)
    mm.gjk_iterations = int(mjm.opt.ccd_iterations)
    mm.epa_iterations = int(mjm.opt.ccd_iterations)
    t = self.tables  # meshdata.HullTables
    mm.hull_vert = wp.array(t.vert.astype(np.float32), dtype=wp.vec3)
    mm.hull_vertadr, mm.hull_vertnum = wp.array(t.vertadr, dtype=int), wp.array(t.vertnum, dtype=int)
    mm.geom_hull = wp.array(t.geom_hull, dtype=int)
    ok = np.zeros((int(K.NTYPE), int(K.NTYPE)), dtype=np.int32)  # (NTYPE, NTYPE) int32: 1 for the pair types taken
    for a, b in self.types:  # a, b: GeomType
      ok[int(a), int(b)] = ok[int(b), int(a)] = 1
    mm.pair_ok = wp.array(ok.ravel(), dtype=int)
    self.mm = mm  # MeshModel: the kernels' model

    it = mm.epa_iterations  # int
    z = lambda n, dt=int: wp.zeros(n, dtype=dt)  # (n,) array of dt, zeroed
    # EPA scratch, one row per EPA pair. One thread per pair ("thread"): element-major (a
    # transposed view), so that the threads of a warp reading element i of their rows hit
    # adjacent words. One warp per pair ("warp", epa_warp.py): row-major, so that the lanes
    # reading consecutive elements of one row hit adjacent words too.
    # "auto" picks the warp version below ~10 worlds per SM, where EPA is latency-bound and the
    # warp version is the faster of the two; above it the thread version is.
    if epa_mode == "auto":
      epa_mode = "warp" if self.nworld < 10 * d.ncollision.device.sm_count else "thread"
    if epa_mode not in ("thread", "warp"):
      raise ValueError(f"epa_mode must be 'auto', 'thread' or 'warp', not {epa_mode!r}")
    self.epa_mode = epa_mode  # str: "thread" or "warp", the EPA kernel `launch` runs
    if epa_mode == "thread":
      rows = lambda n, length, dt: wp.empty((length, n), dtype=dt).transpose()  # (n, length) array of dt
    else:
      rows = lambda n, length, dt: wp.empty((n, length), dtype=dt)
    nt = self.nworld << tbits  # int: entries per cache half, 2^tbits per world
    self.ncache = nt  # int, entries per cache half
    shared = dict(  # dict of str -> wp.array: PipeState's arrays, by field
      cache_parity=z(1), cache_key=wp.full(2 * nt, -1, dtype=int), cache_feat=z(2 * nt),
      cache_dir=z(2 * nt, wp.vec3),
      ncollision=d.ncollision, collision_pair=ctx.collision_pair, collision_worldid=ctx.collision_worldid,
      cold_count=z(1), cold_slot=z(self.naconmax),
      epa_count=z(1), epa_slot=z(self.cap_epa), epa_gjk=wp.empty(self.cap_epa, dtype=GJKResult),
      epa_vert=rows(self.cap_epa, 10 + 2 * it, wp.vec3),
      epa_vert_index=rows(self.cap_epa, 10 + 2 * it, int),
      epa_face=rows(self.cap_epa, 6 + MAX_EPAFACES * it, int),
      epa_pr=rows(self.cap_epa, 6 + MAX_EPAFACES * it, wp.vec3),
      epa_norm2=rows(self.cap_epa, 6 + MAX_EPAFACES * it, float),
      epa_horizon=rows(self.cap_epa, MAX_EPAHORIZON, int),
      epa_del=rows(self.cap_epa, 6 + MAX_EPAFACES * it, int) if epa_mode == "warp" else wp.zeros((1, 1), dtype=int),
      overflow=z(max(self.nworld, 1)),
      man_count=z(1), man_slot=z(self.cap_man), man_feat=z(self.cap_man), man_n=z(self.cap_man, wp.vec3),
      man_depth=z(self.cap_man, float), man_incv=z(self.cap_man),
      con_count=z(1), con_slot=z(self.cap_con), con_k=z(self.cap_con), con_dist=z(self.cap_con, float),
      con_pos=z(self.cap_con, wp.vec3),
      con_normal=z(self.cap_con, wp.vec3), con_geom=z(self.cap_con, wp.vec2i), con_worldid=z(self.cap_con),
      stats=z(len(STAT_NAMES)),
      dbg_slot=wp.full(self.cap_epa, -1, dtype=int), dbg_feat=z(self.cap_epa), dbg_stat=z(self.cap_epa),
      dbg_depth=z(self.cap_epa, float), dbg_path=z(self.naconmax), dbg_man_min=z(self.cap_man, float),
    )
    st = K.PipeState()  # PipeState: the kernels' buffers
    for k, v in shared.items():  # k: str, a field; v: wp.array, its array
      setattr(st, k, v)
    st.tbits = tbits
    self.st = st  # PipeState: the kernels' buffers
    self.shared = shared  # dict of str -> wp.array: the same arrays, by name
    self.graph = None  # the graph `capture` records, None until then
    # int: an array's size in bytes
    nbytes = lambda a: int(np.prod(a.shape)) * wp.types.type_size_in_bytes(a.dtype)  # views: no .capacity
    # int: the device bytes of the pipeline's own buffers and of the hull tables' support cells
    self.bytes = sum(nbytes(a) for a in shared.values() if isinstance(a, wp.array) and a.ptr is not None
                     and a is not d.ncollision and a is not ctx.collision_pair and a is not ctx.collision_worldid)
    self.bytes += sum(nbytes(a) for a in (self.hull.cell, self.hull.cand_p, self.hull.cand_v))

  def launch(self):
    """Launches one narrowphase pass (capturable: the cache halves swap on the device).

    Called by `lodos.col.prim.driver.PrimCollision.collide`, `run` and `capture`.
    In: the arrays bound at construction: d.geom_xpos (nworld, ngeom) wp.vec3, d.geom_xmat (nworld, ngeom) wp.mat33,
    d.ncollision (1,) int, ctx.collision_pair (naconmax,) wp.vec2i, ctx.collision_worldid (naconmax,) int.
    Out: the contacts in self.st: con_count (1,) int, con_slot, con_k and con_worldid (4 * naccdmax,) int, con_dist
    (4 * naccdmax,) float, con_pos and con_normal (4 * naccdmax,) wp.vec3, con_geom (4 * naccdmax,) wp.vec2i; the
    cache for the next pass; st.stats (len(STAT_NAMES),) int.
    """
    st = self.st  # PipeState
    s = self.shared  # dict of str -> wp.array
    for k in ("cold_count", "epa_count", "man_count", "con_count", "stats", "overflow", "dbg_path"):  # k: str
      s[k].zero_()
    gap = 0.0  # float (m): the hybrid's geoms have zero margin and gap
    wp.launch(K.clear_cache_kernel, dim=self.ncache, inputs=[st])
    if any(t in HULL_TYPES for t in self.types):
      wp.launch(K.warm_kernel, block_dim=64, dim=self.naconmax, inputs=[self.mm, self.hull, st, gap, self.use_cache])
      wp.launch(K.gjk_kernel, block_dim=64, dim=self.naconmax, inputs=[self.mm, self.hull, st, gap])
      if self.epa_mode == "warp":
        wp.launch_tiled(epa_warp_kernel, dim=[self.cap_epa], inputs=[self.mm, self.hull, st, gap], block_dim=LANES)
      else:
        wp.launch(K.epa_kernel, block_dim=64, dim=self.cap_epa, inputs=[self.mm, self.hull, st, gap])
      wp.launch(K.manifold_kernel, block_dim=64, dim=self.cap_man, inputs=[self.mm, self.hull, st, gap])
    if any(t in FLOOR_TYPES for t in self.types):
      wp.launch(K.plane_kernel, block_dim=64, dim=self.naconmax, inputs=[self.mm, self.hull, st, gap])
    wp.launch(K.flip_cache_kernel, dim=1, inputs=[st])

  def run(self):
    """One narrowphase pass, from the captured graph once there is one.

    Not called in the repository.
    """
    if self.graph is None:
      self.launch()
    else:
      wp.capture_launch(self.graph)

  def capture(self):
    """Captures `launch` into a CUDA graph, which `run` replays from then on.

    Not called in the repository.
    """
    with wp.ScopedCapture() as cap:  # cap: wp.ScopedCapture
      self.launch()
    self.graph = cap.graph

  def contacts(self):
    """The pass's contacts, copied to the host (synchronizes).

    Not called in the repository.
    Out: dict of numpy arrays of n = min(con_count, 4 * naccdmax) rows: dist (n,) float32, pos (n, 3) float32, normal
    (n, 3) float32, geom (n, 2) int32, worldid (n,) int32.
    """
    s = self.shared  # dict of str -> wp.array
    n = min(int(s["con_count"].numpy()[0]), self.cap_con)  # int: the contacts kept
    return dict(dist=s["con_dist"].numpy()[:n], pos=s["con_pos"].numpy()[:n], normal=s["con_normal"].numpy()[:n],
                geom=s["con_geom"].numpy()[:n], worldid=s["con_worldid"].numpy()[:n])

  def stats(self):
    """The pass's path counters by name (STAT_NAMES), copied to the host (synchronizes).

    Nothing in Lodos calls it: it is for a caller's diagnostics.
    Out: dict of str -> int.
    """
    return dict(zip(STAT_NAMES, self.shared["stats"].numpy().tolist()))
