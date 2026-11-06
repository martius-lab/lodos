"""EPA with one warp per pair, for small batches.

At small batch sizes the EPA stage is latency-bound: its time is the slowest pair's chain of
iterations, and every iteration of MJWarp's EPA scans all faces the polytope has had (for the
closest face, then for visibility) out of global memory, one thread per pair. Here the 32 lanes
of a block share one pair: they scan disjoint strided subsets of the faces and vertices and
combine through tile reductions; the support, the bounds and the termination tests are computed
by every lane on identical inputs (cheaper than sending them); the horizon -- the edges of exactly
one deleted face, MJWarp's undirected-edge cancellation -- is found by the lanes over a compacted
list of the deleted faces; the new faces are attached one horizon edge per lane. Every branch
around a tile operation depends only on block-uniform values. The polytope layout, face flags,
initial polytope (built by lane 0) and the post-processing (`finish_epa`) are those of the
single-thread EPA, so the results decode the same way.
"""

from __future__ import annotations

import warp as wp


from lodos.col.prim.collision_gjk import FLOAT_MAX
from lodos.col.prim.collision_gjk import MIN_EPATOL
from lodos.col.prim.collision_gjk import Polytope
from lodos.col.prim.collision_gjk import _attach_face
from lodos.col.prim.collision_gjk import _delete_face
from lodos.col.prim.collision_gjk import _epa_witness
from lodos.col.prim.collision_gjk import _get_edge
from lodos.col.prim.collision_gjk import _get_face_verts
from lodos.col.prim.collision_gjk import _invalidate_face
from lodos.col.prim.collision_gjk import _is_face_deleted
from lodos.col.prim.collision_gjk import _is_invalid_face
from lodos.mjcf.types import OverflowType

from .ccd import cell_support
from .ccd import epa_init
from .kernels import MESH
from .kernels import MeshModel
from .kernels import PipeState
from .kernels import count
from .kernels import finish_epa
from .kernels import load_pair
from .kernels import hull_of
from .meshdata import HullTablesDevice

LANES = 32  # threads per block: one warp per EPA pair
PER_LANE = 8  # faces one lane may delete per iteration (strided subset of <= 8 * LANES faces)
vec_lane = wp.types.vector(length=PER_LANE, dtype=int)  # a lane's deleted faces in one iteration
vec_edges = wp.types.vector(length=3 * PER_LANE, dtype=int)  # a lane's horizon edges: three per deleted face


@wp.func
def has_edge(f: wp.vec3i, a: int, b: int) -> bool:
  """Face f has the undirected edge {a, b}.

  Called by `epa_warp_kernel`.
  In: f: wp.vec3i, a face's polytope vertex ids; a, b: int, polytope vertex ids. Out: bool.
  """
  return ((f[0] == a and f[1] == b) or (f[0] == b and f[1] == a) or (f[1] == a and f[2] == b) or (f[1] == b and f[2] == a)
          or (f[2] == a and f[0] == b) or (f[2] == b and f[0] == a))


@wp.kernel(enable_backward=False, module="unique")
def epa_warp_kernel(mm: MeshModel, hull: HullTablesDevice, st: PipeState, gap: float):
  """EPA for the pairs GJK found penetrating, the LANES lanes of a block on one pair (the module docstring), then
  lane 0 hands the result to `finish_epa`, as `kernels.epa_kernel` does.

  Launch: `wp.launch_tiled`, dim=(cap_epa,) with block_dim=LANES: a block per EPA slot, `wp.tid()` the slot and
  the lane, by `pipeline.HybridNarrowphase.launch` when its `epa_mode` is "warp" (cap_epa = d.naccdmax).
  Reads: st.epa_count (1,) int, st.epa_slot (cap_epa,) int, st.epa_gjk (cap_epa,) GJKResult, st.collision_pair
  (naconmax,) wp.vec2i and st.collision_worldid (naconmax,) int; mm's geom poses, hull vertices and
  epa_iterations; hull's cells and faces; gap: float, the contact gap (m; 0 from the pipeline).
  Writes: the slot's scratch rows of st.epa_vert and st.epa_vert_index (cap_epa, 10 + 2 * epa_iterations)
  wp.vec3 and int, of st.epa_face, st.epa_pr, st.epa_norm2 and st.epa_del (cap_epa, 6 + MAX_EPAFACES *
  epa_iterations) int, wp.vec3, float and int, and of st.epa_horizon (cap_epa, MAX_EPAHORIZON) int;
  st.overflow (max(nworld, 1),) int and st.stats; through `finish_epa`, the feature cache and the manifold queue.
  """
  tid, lane = wp.tid()  # int, int: the EPA slot, the lane
  if tid >= wp.min(st.epa_count[0], st.epa_slot.shape[0]):
    return
  slot = st.epa_slot[tid]  # int: the pair's broadphase slot
  geoms = st.collision_pair[slot]  # wp.vec2i: its geoms
  worldid = st.collision_worldid[slot]  # int
  g1, g2, origin = load_pair(mm, geoms, worldid)  # Geom, Geom, wp.vec3: the pair, in a frame at geom1's origin
  m1 = hull_of(mm, geoms[0], worldid)  # int: geom1's hull
  m2 = hull_of(mm, geoms[1], worldid)  # int: geom2's hull
  gr = st.epa_gjk[tid]  # GJKResult: the pair's GJK result
  face = st.epa_face[tid]  # (6 + MAX_EPAFACES * epa_iterations,) int: the slot's faces, packed vertex words
  face_pr = st.epa_pr[tid]  # (6 + MAX_EPAFACES * epa_iterations,) wp.vec3: the origin's projection on each face
  face_norm2 = st.epa_norm2[tid]  # (6 + MAX_EPAFACES * epa_iterations,) float: its squared length
  vert_index = st.epa_vert_index[tid]  # (10 + 2 * epa_iterations,) int: the vertices' hull-local ids
  dlist = st.epa_del[tid]  # (6 + MAX_EPAFACES * epa_iterations,) int: an iteration's deleted faces, compacted
  pt = Polytope()  # Polytope: on the slot's scratch rows
  pt.nface = 0
  pt.nvert = 0
  pt.nhorizon = 0
  pt.vert = st.epa_vert[tid]
  pt.vert_index = vert_index
  pt.face = face
  pt.face_pr = face_pr
  pt.face_norm2 = face_norm2
  pt.horizon = st.epa_horizon[tid]

  # the initial polytope, built by lane 0 and broadcast (the tile ops are block barriers)
  status = int(0)  # int: the initial polytope's status, 0: ready
  nvert = int(0)  # int: the polytope's vertex count
  nface = int(0)  # int: its face count
  center = wp.vec3(0.0)  # wp.vec3: its centre
  if lane == 0:
    count(st, 5)
    pt, gr = epa_init(pt, gr, hull, m1, m2, g1, g2)
    status = pt.status
    nvert = pt.nvert
    nface = pt.nface
    center = pt.center
  status = wp.tile_extract(wp.tile(status), 0)
  nvert = wp.tile_extract(wp.tile(nvert), 0)
  nface = wp.tile_extract(wp.tile(nface), 0)
  pt.center = wp.tile_extract(wp.tile(center, preserve_type=True), 0)

  upper = float(FLOAT_MAX)  # float: the upper bound on the penetration depth
  upper2 = float(FLOAT_MAX)  # float: upper^2
  idx = int(-1)  # int: the face closest to the origin
  pidx = int(-1)  # int: the previous iteration's
  nvalid = nface  # int: the faces that can still be expanded
  overflow = int(0)  # int: 1 when the deleted faces or the horizon overflowed
  cap_face = face.shape[0]  # int: the face row's length
  cap_horizon = pt.horizon.shape[0]  # int: the horizon row's length, MAX_EPAHORIZON
  if status == 0:
    for _ in range(wp.min(mm.epa_iterations, 1000)):  # _: int, unused
      pidx = idx

      # the face closest to the origin: per-lane minima over strided subsets, then argmin
      best = float(FLOAT_MAX)  # float: the lane's least squared face distance
      bi = int(-1)  # int: its face
      i = lane  # int: the lane's face, strided by LANES
      while i < nface:
        if not _is_invalid_face(face[i]) and face_norm2[i] < best:
          best = face_norm2[i]
          bi = i
        i += LANES
      tbest = wp.tile(best)  # tile (LANES,) float: every lane's best
      j = wp.tile_extract(wp.tile_argmin(tbest), 0)  # int: the lane with the closest face
      lower2 = wp.tile_extract(tbest, j)  # float: the closest face's squared distance, a lower bound on depth^2
      idx = wp.tile_extract(wp.tile(bi), j)
      if lower2 > upper2 or idx < 0:
        idx = pidx
        break
      if lower2 <= 0.0:
        break

      # support along the face normal (every lane; lane 0 stores the new vertex)
      lower = wp.sqrt(lower2)  # float
      wi = nvert  # int: the new vertex
      fpr = face_pr[idx]  # wp.vec3: the origin's projection onto the closest face's plane
      d = fpr / lower  # wp.vec3: the face's unit normal
      p1, v1 = cell_support(hull, m1, g1, d)  # wp.vec3, int: geom1's support and its vertex
      p2, v2 = cell_support(hull, m2, g2, -d)  # wp.vec3, int: geom2's
      if lane == 0:
        pt.vert[2 * wi] = p1
        pt.vert[2 * wi + 1] = p2
        vert_index[2 * wi] = v1
        vert_index[2 * wi + 1] = v2
      w = p1 - p2  # wp.vec3: the new Minkowski-difference point
      nvert += 1
      upper_k = wp.dot(fpr, w) / lower  # float: this iteration's upper bound
      if upper_k < upper:
        upper = upper_k
        upper2 = upper * upper
      if upper - lower < MIN_EPATOL:
        break

      # repeated support point
      rep = int(0)  # int: 1 when the lane finds the new vertex among the old
      i = lane
      while i < wi:
        if vert_index[2 * i] == v1 and vert_index[2 * i + 1] == v2:
          rep = 1
        i += LANES
      if wp.tile_extract(wp.tile_max(wp.tile(rep)), 0) > 0:
        break

      # delete the closest face and every face w sees; compact their vertex words into dlist
      dl = vec_lane()  # vec_lane: the lane's deleted faces
      ndel = int(0)  # int: their count
      nvdel = int(0)  # int: of them, those that were valid
      i = lane
      while i < nface:
        fc = face[i]  # int: the face's packed vertex word
        if not _is_face_deleted(fc):
          if i == idx or wp.dot(face_pr[i], w) - face_norm2[i] > 1.0e-10:
            if not _is_invalid_face(fc):
              nvdel += 1
            face[i] = _delete_face(fc)
            if ndel < PER_LANE:
              dl[ndel] = fc
            ndel += 1
        i += LANES
      off = wp.untile(wp.tile_scan_exclusive(wp.tile(ndel)))  # int: the lane's first slot in dlist
      for k in range(wp.min(ndel, PER_LANE)):  # k: int, a deleted face of the lane
        dlist[off + k] = dl[k]
      ndel_all = wp.tile_extract(wp.tile_sum(wp.tile(ndel)), 0)  # int: the faces deleted, over all lanes
      nvdel_all = wp.tile_extract(wp.tile_sum(wp.tile(nvdel)), 0)  # int: of them, those that were valid
      if wp.tile_extract(wp.tile_max(wp.tile(ndel)), 0) > PER_LANE or ndel_all > PER_LANE * LANES:
        overflow = 1
        idx = -1
        break

      # horizon: edges of exactly one deleted face
      he = vec_edges()  # vec_edges: the lane's horizon edges, packed (min << 10) | max
      nh = int(0)  # int: their count
      q = lane  # int: the lane's deleted face, strided by LANES
      while q < ndel_all:
        fv = _get_face_verts(dlist[q])  # wp.vec3i: its vertices
        for e in range(3):  # e: int, an edge of the face
          a = fv[e]  # int: the edge's first vertex
          b = fv[(e + 1) % 3]  # int: its second
          shared = bool(False)  # bool: another deleted face has the edge
          for r in range(ndel_all):  # r: int, a deleted face
            if r != q and has_edge(_get_face_verts(dlist[r]), a, b):
              shared = True
              break
          if not shared and nh < 3 * PER_LANE:
            he[nh] = (wp.min(a, b) << 10) | wp.max(a, b)
            nh += 1
        q += LANES
      hoff = wp.untile(wp.tile_scan_exclusive(wp.tile(nh)))  # int: the lane's first slot in pt.horizon
      for k in range(nh):
        if hoff + k < cap_horizon:
          pt.horizon[hoff + k] = he[k]
      nh_all = wp.tile_extract(wp.tile_sum(wp.tile(nh)), 0)  # int: the horizon edges, over all lanes
      if nh_all > cap_horizon or nface + nh_all > cap_face:
        overflow = wp.where(nh_all > cap_horizon, 1, 0)
        idx = -1
        break

      # attach the new faces, one horizon edge per lane
      nvadd = int(0)  # int: the lane's new faces that are valid
      bad = int(0)  # int: 1 when a new face of the lane is degenerate
      h = lane  # int: the lane's horizon edge, strided by LANES
      while h < nh_all:
        edge = _get_edge(pt.horizon[h])  # wp.vec2i: its vertices
        pt.nface = nface + h
        dist2 = _attach_face(pt, nface + h, wi, edge[0], edge[1])  # float: the new face's squared distance, 0: none
        if dist2 == 0.0:
          bad = 1
        elif dist2 >= lower2 and dist2 <= upper2:
          nvadd += 1
        else:
          face[nface + h] = _invalidate_face(face[nface + h])
        h += LANES
      nface += nh_all
      nvalid = nvalid - nvdel_all + wp.tile_extract(wp.tile_sum(wp.tile(nvadd)), 0)
      if wp.tile_extract(wp.tile_max(wp.tile(bad)), 0) > 0:
        idx = -1
        break
      if nvalid == 0:
        break

  if lane == 0:
    dist = float(FLOAT_MAX)  # float: the signed distance (m, negative in penetration)
    ncon = int(0)  # int: 1 for a contact
    x1 = wp.vec3()  # wp.vec3: geom1's witness
    x2 = wp.vec3()  # wp.vec3: geom2's
    res = int(-1)  # int: the minimizing face, -1: none
    if status != 0:
      dist = gr.dist
      ncon = 1
      x1 = gr.x1
      x2 = gr.x2
    elif idx > -1:
      pt.nface = nface
      x1, x2, dist = _epa_witness(pt, g1, g2, MESH, MESH, idx)
      ncon = 1
      res = idx
    if overflow != 0:
      wp.atomic_or(st.overflow, worldid, OverflowType.EPA_HORIZON)
    finish_epa(mm, hull, st, slot, geoms, worldid, g1, g2, m1, m2, dist, ncon, x1, x2, res, face, vert_index, gap)
