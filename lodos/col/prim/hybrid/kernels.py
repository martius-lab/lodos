"""Warp kernels of the hybrid convex narrowphase (pairs of hulls: meshes and boxes).

Stages, each a separate launch so that no stage pays for another's registers:

  warm      per broadphase pair: look up last step's minimizing-facet features, rebuild
            the facet at the new pose, and run the ball-containment certificate.
            Certified -> queue the contact for the manifold stage (and re-cache); a
            certified separation along the cached axis -> no contact (re-cache);
            anything else -> cold list.
  gjk       per cold pair: MJWarp's GJK (ccd.py), whose separating-plane exit at a 1 um margin
            is a certificate there; penetrating -> EPA list.
  epa       per EPA pair: MJWarp's EPA (ccd.py; one warp per pair at small batches, epa_warp.py),
            then the minimizing polytope triangle is decoded into hull features, whose exact
            depth and normal -> manifold queue + cache (finish_epa).
  plane     per broadphase plane-mesh pair (the floor): MuJoCo's plane_convex contacts -- the
            deepest vertex and up to three more among the vertices within 1 mm of it, spread
            out -- with the deepest vertex from one cell lookup and the others from scans of
            the hull's vertices, instead of MJWarp's hill climbs over the mesh graph.
  manifold  per queued contact pair: quadrangle clip (the incident face's largest quad through
            the support vertex, Sutherland-Hodgman against the side planes of the reference
            face's largest quad through its vertex nearest to that support), 4-point area
            reduction, or the edge-edge point. An edge-edge pair is a face patch instead when
            one of the four faces at its edges lies within MuJoCo's multi-CCD tolerance
            (FACE_TOL, 5.1 deg) of the normal and its depth within PATCH_TOL (25 um) of the
            exact depth: nearly parallel faces or an edge lying on a face, where an edge axis
            beats the face axis by microns (resting stacks); that face's clip gives the points,
            with depths below it. Edge contacts that win by more keep the single exact point, so
            every pair's deepest contact stays within PATCH_TOL of the exact depth.

Every support query, here and in GJK/EPA, is `ccd.cell_support`: the best of the few exact
candidates of the direction's Gauss-map cell, with no walk over the hull graph.

A geom's hull (`hull_of`) is its mesh for a mesh geom and its box hull for a box geom
(meshdata.build); every hull's vertices are in `MeshModel.hull_vert`, so the code below never
distinguishes the two. Which pair types run here is `MeshModel.pair_ok`.

All geometry of a pair is evaluated in a frame translated to geom1's origin (as MJWarp
does). Normals point from geom1 to geom2; contact positions are the midpoint of the two
witnesses; dist is negative in penetration -- MJWarp's conventions, so the contact sets
compare directly.

Certificate: D = P - Q, axis u (geom1 -> geom2), depth
d = h_D(u) = max_P u.x - min_Q u.y. The witnesses are the candidate facet F (reference
face loop minus the incident support vertex, or the edge-pair parallelogram) and
z- = s_D(-u). If every lateral plane of conv(F + z-) -- through an edge of F and z- --
and the base lie at distance >= d - EPS from the origin, the ball B(0, d - EPS) lies in D,
so the true depth is within EPS of d and u is a minimum-penetration axis.
"""

from __future__ import annotations

import warp as wp


from lodos.col.prim.collision_core import Geom
from lodos.col.prim.collision_core import geom_collision_pair_from_types
from lodos.col.prim.collision_gjk import FACE_TOL
from lodos.col.prim.collision_gjk import GJKResult
from lodos.col.prim.types import mat43
from lodos.mjcf.types import GeomType

from .ccd import cell_support
from .ccd import epa_phase
from .ccd import gjk
from .meshdata import HullTablesDevice

MESH = wp.constant(int(GeomType.MESH.value))  # GeomType.MESH, for kernels
PLANE = wp.constant(int(GeomType.PLANE.value))  # GeomType.PLANE, for kernels
NTYPE = wp.constant(16)  # row length of MeshModel.pair_ok (> every GeomType value)
PLANE_BAND = wp.constant(1.0e-3)  # MuJoCo's plane_convex: extra contacts within 1 mm of the deepest vertex
EPS = wp.constant(1.0e-5)  # certificate tolerance: certified depth is exact to EPS (m)
EPS_SUP = wp.constant(1.0e-5)  # "supporting" tolerance for facet features (m)
SEP_TOL = wp.constant(1.0e-6)  # GJK separation proofs: margin above float32 support error (~1e-7 m)
AREA_REL = wp.constant(1.0e-5)  # manifold reduction: a new corner must add AREA_REL * |p1 - p0|^2
MIN_SEP = wp.constant(1.0e-6)  # manifold reduction: corners closer than this (m) are redundant
PATCH_TOL = wp.constant(2.5e-5)  # edge -> face patch: face depth within PATCH_TOL (m) of the exact depth
FEAT_FACE1 = wp.constant(1)  # feature kind: a face of geom1, the reference, against geom2
FEAT_FACE2 = wp.constant(2)  # feature kind: a face of geom2, the reference, against geom1
FEAT_EDGE = wp.constant(3)  # feature kind: an edge of each geom
FEAT_DIR = wp.constant(-2)  # cache only: a separating direction, not a facet
mat83 = wp.types.matrix(shape=(8, 3), dtype=wp.float32)  # a quad clipped by a quad has <= 8 corners
vec8 = wp.types.vector(length=8, dtype=wp.float32)  # a value per corner of a mat83


@wp.struct
class MeshModel:
  """The MJWarp Model/Data fields the narrowphase reads, bundled for kernel signatures.

  Built by `pipeline.HybridNarrowphase.__init__`, which binds MJWarp's arrays rather than copying them, so the
  kernels read the current geom poses.
  """

  geom_type: wp.array(dtype=int)  # (ngeom,) GeomType
  geom_dataid: wp.array2d(dtype=int)  # (1 or nworld, ngeom) the geom's mesh id, -1 none
  geom_size: wp.array2d(dtype=wp.vec3)  # (1 or nworld, ngeom)
  mesh_vertadr: wp.array(dtype=int)  # (nmesh,)
  mesh_vertnum: wp.array(dtype=int)  # (nmesh,)
  mesh_graphadr: wp.array(dtype=int)  # (nmesh,) -1: no graph
  mesh_vert: wp.array(dtype=wp.vec3)  # (nmeshvert,) mesh frame
  mesh_graph: wp.array(dtype=int)  # (nmeshgraph,)
  mesh_polynum: wp.array(dtype=int)  # (nmesh,)
  mesh_polyadr: wp.array(dtype=int)  # (nmesh,)
  mesh_polynormal: wp.array(dtype=wp.vec3)  # (nmeshpoly,)
  mesh_polyvertadr: wp.array(dtype=int)  # (nmeshpoly,)
  mesh_polyvertnum: wp.array(dtype=int)  # (nmeshpoly,)
  mesh_polyvert: wp.array(dtype=int)  # (nmeshpolyvert,)
  mesh_polymapadr: wp.array(dtype=int)  # (nmeshvert,)
  mesh_polymapnum: wp.array(dtype=int)  # (nmeshvert,)
  mesh_polymap: wp.array(dtype=int)  # (nmeshpolymap,)
  geom_xpos: wp.array2d(dtype=wp.vec3)  # (nworld, ngeom) world position
  geom_xmat: wp.array2d(dtype=wp.mat33)  # (nworld, ngeom) world orientation
  ccd_tolerance: float  # GJK's distance tolerance (m), mjm.opt.ccd_tolerance
  gjk_iterations: int  # mjm.opt.ccd_iterations
  epa_iterations: int  # mjm.opt.ccd_iterations
  # the hybrid's own (meshdata.build)
  hull_vert: wp.array(dtype=wp.vec3)  # (nvert,) every hull's vertices, hull frame (meshes at their mesh_vertadr)
  hull_vertadr: wp.array(dtype=int)  # (nhull,)
  hull_vertnum: wp.array(dtype=int)  # (nhull,)
  geom_hull: wp.array(dtype=int)  # (ngeom,) box hull id; -1: mesh geom (hull = geom_dataid); -2: no hull
  pair_ok: wp.array(dtype=int)  # (NTYPE * NTYPE,) 1 where the pair type (type1, type2) runs here


@wp.struct
class PipeState:
  """Device buffers of the hybrid pipeline (counters are 1-element arrays).

  Built by `pipeline.HybridNarrowphase.__init__`, which binds MJWarp's broadphase arrays into it;
  `pipeline.HybridNarrowphase.launch` zeroes the counters, stats, overflow and dbg_path before every pass.
  """

  # broadphase input (MJWarp CollisionContext + d.ncollision)
  ncollision: wp.array(dtype=int)  # (1,) the broadphase's pair count
  collision_pair: wp.array(dtype=wp.vec2i)  # (naconmax,) the pairs' geoms
  collision_worldid: wp.array(dtype=int)  # (naconmax,) the pairs' worlds
  # feature cache: per-world open-addressing hash tables in the two halves of one array. The
  # half cache_parity selects holds last step's entries (read), the other receives this step's
  # (write); the parity flips at the end of every pass on the device, so one captured graph
  # serves every step
  tbits: int  # log2 of the per-world table size
  cache_parity: wp.array(dtype=int)  # (1,)
  cache_key: wp.array(dtype=int)  # (2 * nworld << tbits,) key g1 * 65536 + g2, -1 empty
  cache_feat: wp.array(dtype=int)  # (2 * nworld << tbits,) feature id (pack_feat) or FEAT_DIR
  cache_dir: wp.array(dtype=wp.vec3)  # (2 * nworld << tbits,) separating direction (geom1 frame) when feat == FEAT_DIR
  # cold path
  cold_count: wp.array(dtype=int)  # (1,)
  cold_slot: wp.array(dtype=int)  # (naconmax,) broadphase slots for GJK
  epa_count: wp.array(dtype=int)  # (1,)
  epa_slot: wp.array(dtype=int)  # (naccdmax,) broadphase slots for EPA
  epa_gjk: wp.array(dtype=GJKResult)  # (naccdmax,) their GJK results, or seed tetrahedra
  # EPA rows, one per EPA pair; per polytope vertex v, entries 2 v and 2 v + 1 are geom1's and geom2's
  epa_vert: wp.array2d(dtype=wp.vec3)  # (naccdmax, 10 + 2 * epa_iterations) points
  epa_vert_index: wp.array2d(dtype=int)  # (naccdmax, 10 + 2 * epa_iterations) hull-local vertex ids
  epa_face: wp.array2d(dtype=int)  # (naccdmax, 6 + MAX_EPAFACES * epa_iterations) three 10-bit vertex ids and flags
  epa_pr: wp.array2d(dtype=wp.vec3)  # (naccdmax, 6 + MAX_EPAFACES * epa_iterations) origin's projection per face
  epa_norm2: wp.array2d(dtype=float)  # (naccdmax, 6 + MAX_EPAFACES * epa_iterations) its squared norm
  epa_horizon: wp.array2d(dtype=int)  # (naccdmax, MAX_EPAHORIZON) the horizon's edges
  # (naccdmax, 6 + MAX_EPAFACES * epa_iterations); (1, 1), unused, with one thread per EPA pair
  epa_del: wp.array2d(dtype=int)  # warp-cooperative EPA only: the faces deleted in an iteration
  # (max(nworld, 1),) per world: EPA's OverflowType flags
  overflow: wp.array(dtype=int)
  # manifold queue
  man_count: wp.array(dtype=int)  # (1,)
  man_slot: wp.array(dtype=int)  # (naccdmax,) broadphase slot
  man_feat: wp.array(dtype=int)  # (naccdmax,) feature id, 0 none
  man_n: wp.array(dtype=wp.vec3)  # (naccdmax,) normal geom1 -> geom2
  man_depth: wp.array(dtype=float)  # (naccdmax,) exact depth (m)
  man_incv: wp.array(dtype=int)  # (naccdmax,) incident support vertex (FACE*), -1 none
  # contacts out
  con_count: wp.array(dtype=int)  # (1,)
  con_slot: wp.array(dtype=int)  # (4 * naccdmax,) broadphase slot of the pair
  con_k: wp.array(dtype=int)  # (4 * naccdmax,) index of the contact within its pair
  con_dist: wp.array(dtype=float)  # (4 * naccdmax,) negative in penetration (m)
  con_pos: wp.array(dtype=wp.vec3)  # (4 * naccdmax,) world position, the witnesses' midpoint
  con_normal: wp.array(dtype=wp.vec3)  # (4 * naccdmax,) geom1 -> geom2
  con_geom: wp.array(dtype=wp.vec2i)  # (4 * naccdmax,)
  con_worldid: wp.array(dtype=int)  # (4 * naccdmax,)
  # statistics, counters named by pipeline.STAT_NAMES
  stats: wp.array(dtype=int)  # (len(STAT_NAMES),)
  # debug: per EPA pair (slot, feature, certificate status, EPA depth)
  dbg_slot: wp.array(dtype=int)  # (naccdmax,)
  dbg_feat: wp.array(dtype=int)  # (naccdmax,)
  dbg_stat: wp.array(dtype=int)  # (naccdmax,)
  dbg_depth: wp.array(dtype=float)  # (naccdmax,)
  dbg_path: wp.array(dtype=int)  # (naconmax,) per broadphase slot (see warm_kernel, gjk_kernel and finish_epa)
  dbg_man_min: wp.array(dtype=float)  # (naccdmax,) per manifold item: deepest dist emitted


#######################################################################################
# Small helpers
#######################################################################################


@wp.func
def count(st: PipeState, i: int):
  """Path counter i (pipeline.STAT_NAMES). Same-address atomics, but aggregated per warp by the
  compiler: together they cost nothing measurable.

  Called by `warm_kernel`, `gjk_kernel`, `finish_epa` and 3 others.
  Writes: st.stats (len(STAT_NAMES),) int, entry i added to.
  """
  wp.atomic_add(st.stats, i, 1)


@wp.func
def pair_key(g1: int, g2: int) -> int:
  """The cache key of the geom pair (g1, g2).

  Called by `warm_kernel` and `finish_epa`.
  In: g1, g2 (int) the geoms. Out: int, g1 * 65536 + g2.
  """
  return g1 * 65536 + g2


@wp.func
def hslot(key: int, tbits: int) -> int:
  """The home entry of `key` in a table of 2^tbits entries: the top tbits bits of the 32-bit product
  key * 2654435761 (multiplicative hashing).

  Called by `cache_lookup` and `push_cache`.
  In: key (int) a `pair_key`; tbits (int) log2 of the table size. Out: int, 0 to 2^tbits - 1.
  """
  return int((wp.uint32(key) * wp.uint32(2654435761)) >> wp.uint32(32 - tbits))


@wp.func
def table_base(st: PipeState, worldid: int, write: bool) -> int:
  """First entry of a world's table in the read (last step's) or write (this step's) half.

  Called by `cache_lookup` and `push_cache`.
  Reads: st.cache_parity (1,) int, st.tbits, the length of st.cache_key (2 * nworld << tbits,).
  In: worldid (int); write (bool) the write half. Out: int, an index into st.cache_key, cache_feat and cache_dir.
  """
  p = st.cache_parity[0]  # int: the read half, 0 or 1
  return wp.where(write, 1 - p, p) * (st.cache_key.shape[0] // 2) + worldid * (1 << st.tbits)


@wp.func
def cache_lookup(st: PipeState, worldid: int, key: int) -> int:
  """Index into the read half holding `key`, or -1.

  Called by `warm_kernel`.
  Reads: st.cache_key (2 * nworld << tbits,) int, st.cache_parity (1,) int, st.tbits.
  In: worldid (int); key (int) a `pair_key`. Out: int.
  """
  size = 1 << st.tbits  # int: entries per world's table
  base = table_base(st, worldid, False)  # int: the world's table in the read half
  h = hslot(key, st.tbits)  # int: the entry probed, from base
  for _ in range(size):  # _: int, a probe
    k = st.cache_key[base + h]  # int: its key, -1 empty
    if k == key:
      return base + h
    if k == -1:
      return -1
    h = (h + 1) & (size - 1)
  return -1


@wp.func
def pack_feat(kind: int, a: int, b: int) -> int:
  """A feature id: kind (FEAT_FACE1, FEAT_FACE2 or FEAT_EDGE) in bits 29 and up, a (the face, or geom1's edge) in
  bits 15 to 28, b (geom2's edge, 0 for a face) in bits 0 to 14.

  Called by `decode_epa_facet`.
  In: kind, a, b (int). Out: int.
  """
  return kind * 536870912 + a * 32768 + b  # kind << 29 | a << 15 | b; a < 16384, b < 32768


@wp.func
def feat_kind(f: int) -> int:
  """The kind of feature id f (`pack_feat`): FEAT_FACE1, FEAT_FACE2, FEAT_EDGE, or 0 for no feature.

  Called by `warm_kernel`, `finish_epa` and `manifold_kernel`.
  In: f (int). Out: int.
  """
  return f // 536870912


@wp.func
def feat_a(f: int) -> int:
  """Field a of feature id f (`pack_feat`): the face, or geom1's edge.

  Called by `warm_kernel`, `finish_epa` and `manifold_kernel`.
  In: f (int). Out: int, a hull-local face or edge id.
  """
  return (f // 32768) % 16384


@wp.func
def feat_b(f: int) -> int:
  """Field b of feature id f (`pack_feat`): geom2's edge, 0 for a face.

  Called by `warm_kernel`, `finish_epa` and `manifold_kernel`.
  In: f (int). Out: int, a hull-local edge id.
  """
  return f % 32768


@wp.func
def vworld(geom: Geom, k: int) -> wp.vec3:
  """World (pair-frame) position of mesh-local vertex k.

  Called by `loop_vertex`, `certify_face`, `certify_edge` and 2 others.
  Reads: geom.pos (wp.vec3), geom.rot (wp.mat33), geom.vertadr (int), geom.vert (nvert,) wp.vec3.
  In: k (int). Out: wp.vec3.
  """
  return geom.rot @ geom.vert[geom.vertadr + k] + geom.pos


@wp.func
def face_world(geom: Geom, hull: HullTablesDevice, mesh: int, f: int):
  """(outward normal, support-plane offset) of local face f of `geom`, pair frame.

  Called by `certify_face`, `finish_epa` and `manifold_kernel`.
  Reads: geom.pos, geom.rot; hull.faceadr (nhull,) int, hull.face_n (nface,) wp.vec3, hull.face_d (nface,) float.
  In: mesh (int) the geom's hull id; f (int). Out: (wp.vec3, float).
  """
  fi = hull.faceadr[mesh] + f  # int: the face's global index
  n = geom.rot @ hull.face_n[fi]  # wp.vec3: its outward normal, pair frame
  return n, hull.face_d[fi] + wp.dot(n, geom.pos)


@wp.func
def face_n_local(hull: HullTablesDevice, mesh: int, f: int) -> wp.vec3:
  """Outward unit normal of local face f of hull `mesh`, hull frame.

  Called by `best_face_around`.
  Reads: hull.faceadr (nhull,) int, hull.face_n (nface,) wp.vec3. In: mesh, f (int). Out: wp.vec3.
  """
  return hull.face_n[hull.faceadr[mesh] + f]


@wp.func
def best_face_around(geom: Geom, hull: HullTablesDevice, mesh: int, v: int, d: wp.vec3):
  """(local face incident to local vertex v whose outward normal best aligns with d, position
  of v in that face's loop).

  Called by `certify_face` and `manifold_kernel`.
  Reads: geom.vertadr, geom.rot; hull.vf_adr and hull.vf_num (nvert,) int, hull.vf and hull.vf_pos (nloop,) int,
  the face normals (`face_n_local`). In: mesh (int) the hull id; v (int); d (wp.vec3) pair frame.
  Out: (int, int), (-1, -1) for a vertex on no face.
  """
  gv = geom.vertadr + v  # int: v's global vertex id
  best = float(-2.0)  # float: the best alignment so far
  bestf = int(-1)  # int: its face
  bestk = int(-1)  # int: v's position in that face's loop
  for j in range(hull.vf_num[gv]):  # j: int, an incidence of v
    f = hull.vf[hull.vf_adr[gv] + j]  # int: a face at v
    s = wp.dot(geom.rot @ face_n_local(hull, mesh, f), d)  # float: its normal's alignment with d
    if s > best:
      best = s
      bestf = f
      bestk = hull.vf_pos[hull.vf_adr[gv] + j]
  return bestf, bestk


@wp.func
def loop_vertex(geom: Geom, hull: HullTablesDevice, mesh: int, f: int, k: int) -> wp.vec3:
  """Pair-frame position of corner k of local face f's loop.

  Called by `inside_margin`, `certify_face` and `manifold_kernel`.
  Reads: hull.faceadr (nhull,) int, hull.face_vadr (nface,) int, hull.face_v (nloop,) int; geom as `vworld`.
  In: mesh (int) the hull id; f (int); k (int) a loop position. Out: wp.vec3.
  """
  fi = hull.faceadr[mesh] + f  # int: the face's global index
  return vworld(geom, hull.face_v[hull.face_vadr[fi] + k])


@wp.func
def loop_len(hull: HullTablesDevice, mesh: int, f: int) -> int:
  """Number of corners in local face f's loop.

  Called by `inside_margin` and `certify_face`.
  Reads: hull.faceadr (nhull,) int, hull.face_vnum (nface,) int. In: mesh, f (int). Out: int.
  """
  return hull.face_vnum[hull.faceadr[mesh] + f]


@wp.func
def inside_margin(geom: Geom, hull: HullTablesDevice, mesh: int, f: int, nr: wp.vec3, p: wp.vec3) -> float:
  """Min over the face's edges of the inward distance of p's projection (negative = outside).

  Called by `certify_face`.
  Reads: the face's loop (`loop_len`, `loop_vertex`). In: mesh (int) the hull id; f (int); nr (wp.vec3) the face's
  outward normal; p (wp.vec3), pair frame. Out: float (m), 1e30 when no edge has a nonzero length.
  """
  nv = loop_len(hull, mesh, f)  # int: the loop's corners
  m = float(1.0e30)  # float: the least inward distance so far
  a = loop_vertex(geom, hull, mesh, f, nv - 1)  # wp.vec3: the edge's first corner
  for k in range(nv):  # k: int, a loop position
    b = loop_vertex(geom, hull, mesh, f, k)  # wp.vec3: the edge's second corner
    s = wp.cross(b - a, nr)  # wp.vec3, outward for a CCW loop about nr
    ls = wp.length(s)  # float
    if ls > 1.0e-12:
      m = wp.min(m, wp.dot(s / ls, a - p))
    a = b
  return m


@wp.func
def lateral_ok(a: wp.vec3, b: wp.vec3, zm: wp.vec3, c: wp.vec3, depth: float) -> bool:
  """Plane through a, b and z-: origin at distance >= depth - EPS on the inner side.

  Called by `certify_face` and `certify_edge`.
  In: a, b (wp.vec3) an edge of the facet; zm (wp.vec3) z-; c (wp.vec3) a point inside the pyramid, on the inner
  side; all points of D, pair frame; depth (float) the candidate depth (m). Out: bool.
  """
  m = wp.cross(b - a, zm - a)  # wp.vec3: the plane's normal
  lm = wp.length(m)  # float
  if lm < 1.0e-14:
    return True  # degenerate (zero-area) lateral face: no constraint
  m = m / lm
  if wp.dot(m, c - a) > 0.0:
    m = -m
  return wp.dot(m, a) >= depth - EPS


@wp.struct
class Cert:
  """The certificate of a candidate facet, and the EPA seed when its pyramid test fails.

  Returned by `certify_face` and `certify_edge`; read by `warm_kernel`.
  """

  status: int  # 0 fail -> cold, 1 certified contact, 2 certified separated
  n: wp.vec3  # axis geom1 -> geom2
  depth: float  # the facet's depth along n (m), negative when separated
  incv: int  # local vertex on the incident side (FACE*), unused for EDGE
  # EPA seed when the pyramid fails: base triangle of the facet + z-, D = geom1 - geom2
  seed: int  # 1 when s1, s2, i1 and i2 hold a tetrahedron containing the origin, else 0
  s1: mat43  # geom1 points
  s2: mat43  # geom2 points
  i1: wp.vec4i  # their mesh-local vertex ids
  i2: wp.vec4i  # s2's, as i1 holds s1's


#######################################################################################
# Certificate
#######################################################################################


@wp.func
def tri_contains(a: wp.vec3, b: wp.vec3, c: wp.vec3, p: wp.vec3, n: wp.vec3) -> bool:
  """p (in the plane of the triangle, normal n) lies inside a, b, c (either winding).

  Called by `certify_face` and `certify_edge`.
  In: a, b, c, p, n (wp.vec3). Out: bool, True on the boundary too.
  """
  s0 = wp.dot(wp.cross(b - a, p - a), n)  # float: p's side of edge ab
  s1 = wp.dot(wp.cross(c - b, p - b), n)  # float: of edge bc
  s2 = wp.dot(wp.cross(a - c, p - c), n)  # float: of edge ca
  return (s0 >= 0.0 and s1 >= 0.0 and s2 >= 0.0) or (s0 <= 0.0 and s1 <= 0.0 and s2 <= 0.0)


@wp.func
def origin_in_tetra(w0: wp.vec3, w1: wp.vec3, w2: wp.vec3, w3: wp.vec3) -> bool:
  """Origin strictly inside conv(w0..w3), with an absolute margin (EPA must not see it on a face).

  Called by `set_seed`.
  In: w0, w1, w2, w3 (wp.vec3) points of D, pair frame. Out: bool, False too when the origin lies within 1e-6 m of
  a face's plane or a face is degenerate.
  """
  ok = bool(True)  # bool
  for j in range(4):  # j: int, the face opposite corner j
    a = wp.where(j == 0, w1, w0)  # wp.vec3: the face's corners a, b, c
    b = wp.where(j <= 1, w2, w1)  # wp.vec3
    c = wp.where(j <= 2, w3, w2)  # wp.vec3
    o = wp.where(j == 0, w0, wp.where(j == 1, w1, wp.where(j == 2, w2, w3)))  # wp.vec3: the opposite corner
    m = wp.cross(b - a, c - a)  # wp.vec3: the face's normal
    lm = wp.length(m)  # float
    if lm < 1.0e-14:
      return False
    m = m / lm
    so = wp.dot(m, o - a)  # float: the opposite corner's side of the face
    sz = wp.dot(m, -a)  # float: the origin's side, its distance to the face's plane (m)
    if so * sz <= 0.0 or wp.abs(sz) < 1.0e-6:
      ok = False
  return ok


@wp.func
def set_seed(c: Cert, a1: wp.vec3, a2: wp.vec3, ia1: int, ia2: int, b1: wp.vec3, b2: wp.vec3, ib1: int, ib2: int,
             d1: wp.vec3, d2: wp.vec3, id1: int, id2: int, z1: wp.vec3, z2: wp.vec3, iz1: int, iz2: int) -> Cert:
  """Fill the EPA seed (points given as geom1/geom2 pairs) if the origin is inside it.

  Called by `certify_face` and `certify_edge`.
  In: c (Cert); a1, b1, d1, z1 (wp.vec3) geom1's points and a2, b2, d2, z2 geom2's, pair frame, so that the seed's
  corners are a1 - a2, b1 - b2, d1 - d2, z1 - z2; ia1, ia2, ..., iz2 (int) their hull-local vertex ids.
  Out: Cert, c with seed 1 and s1, s2, i1, i2 set, or with seed 0.
  """
  if not origin_in_tetra(a1 - a2, b1 - b2, d1 - d2, z1 - z2):
    c.seed = 0
    return c
  c.s1 = mat43(a1[0], a1[1], a1[2], b1[0], b1[1], b1[2], d1[0], d1[1], d1[2], z1[0], z1[1], z1[2])
  c.s2 = mat43(a2[0], a2[1], a2[2], b2[0], b2[1], b2[2], d2[0], d2[1], d2[2], z2[0], z2[1], z2[2])
  c.i1 = wp.vec4i(ia1, ib1, id1, iz1)
  c.i2 = wp.vec4i(ia2, ib2, id2, iz2)
  c.seed = 1
  return c


@wp.func
def certify_face(
  ref: Geom, inc: Geom, hull: HullTablesDevice, refmesh: int, incmesh: int, f: int, ref_is_1: bool, gap: float,
) -> Cert:
  """Face f of `ref` against `inc`. Works for both FACE1 (ref = geom1) and FACE2.

  nr is the reference face's outward normal; the pair axis is u = nr for FACE1 and
  u = -nr for FACE2. Writing everything about nr: depth = off_ref - min_inc nr.y, the
  incident support y is inc's support along -nr, and D's facet is the reference loop
  minus y (FACE1) or y minus the loop (FACE2) -- the same set up to the sign of D, so
  the pyramid test is done once in the "ref minus inc" orientation with axis nr.

  Called by `warm_kernel`.
  In: ref, inc (Geom) the reference and incident geoms, pair frame; hull (HullTablesDevice): faceadr (nhull,) int,
  face_vadr (nface,) int, face_v (nloop,) int, and the faces, support cells and vertex-face incidence its helpers
  read; refmesh, incmesh (int) their hull ids; f (int) a local face of ref; ref_is_1 (bool) ref is geom1; gap
  (float, m).
  Out: Cert: status 2 when separated by more than gap + EPS, 1 certified, 0 failed, with an EPA seed when one holds
  the origin.
  """
  c = Cert()  # Cert: the result
  nr, off = face_world(ref, hull, refmesh, f)  # wp.vec3, float: the face's outward normal and support offset
  y, yv = cell_support(hull, incmesh, inc, -nr)  # wp.vec3, int: inc's support along -nr and its vertex
  depth = off - wp.dot(nr, y)  # float: the face axis's depth (m)
  c.n = wp.where(ref_is_1, nr, -nr)
  c.depth = depth
  c.incv = yv
  if depth < -gap - EPS:
    c.status = 2
    return c
  nv = loop_len(hull, refmesh, f)  # int: the face's corners
  # face-face: among the incident face's near-supporting vertices take the one whose
  # projection lies deepest inside the reference face
  ybest = y  # wp.vec3: the chosen incident vertex
  yv_best = yv  # int: its local id
  margin = inside_margin(ref, hull, refmesh, f, nr, y)  # float: its projection's inward margin (m)
  g, gk = best_face_around(inc, hull, incmesh, yv, -nr)  # int, int: inc's face at yv facing ref, yv's loop position
  if g >= 0:
    gn = loop_len(hull, incmesh, g)  # int: that face's corners
    for k in range(gn):  # k: int, a loop position
      gi = hull.faceadr[incmesh] + g  # int: the face's global index
      vk = hull.face_v[hull.face_vadr[gi] + k]  # int: the corner's local vertex
      pk = vworld(inc, vk)  # wp.vec3: its position
      if wp.dot(nr, pk) <= wp.dot(nr, y) + EPS_SUP:
        mk = inside_margin(ref, hull, refmesh, f, nr, pk)  # float: its projection's inward margin (m)
        if mk > margin:
          margin = mk
          ybest = pk
          yv_best = vk
  c.incv = yv_best
  # base: the facet points (loop minus the chosen incident vertex) must sit at distance
  # >= depth - EPS along nr, i.e. the loop lies on the support plane and ybest supports inc
  for k in range(nv):
    if wp.dot(nr, loop_vertex(ref, hull, refmesh, f, k) - ybest) < depth - EPS:
      c.status = 0
      return c
  # far witness z- = s_D(-nr) in the ref-minus-inc orientation
  xr, xrv = cell_support(hull, refmesh, ref, -nr)  # wp.vec3, int: ref's support along -nr
  yi, yiv = cell_support(hull, incmesh, inc, nr)  # wp.vec3, int: inc's support along nr
  zm = xr - yi  # wp.vec3: z-
  # interior point of the pyramid: midpoint of the base centroid and the apex
  cen = wp.vec3(0.0)  # wp.vec3: the base centroid, a point of D
  for k in range(nv):
    cen += loop_vertex(ref, hull, refmesh, f, k)
  cen = cen / float(nv) - ybest
  cin = 0.5 * (cen + zm)  # wp.vec3: the interior point
  ok = bool(True)  # bool: every lateral plane passes
  a = loop_vertex(ref, hull, refmesh, f, nv - 1) - ybest  # wp.vec3: a base edge's first corner, a point of D
  for k in range(nv):
    b = loop_vertex(ref, hull, refmesh, f, k) - ybest  # wp.vec3: its second corner
    if not lateral_ok(a, b, zm, cin, depth):
      ok = False
    a = b
  c.status = wp.where(ok, 1, 0)
  if ok:
    return c
  # seed EPA with the fan triangle of the reference face that holds ybest's projection
  p = ybest + (off - wp.dot(nr, ybest)) * nr  # wp.vec3: ybest projected onto the face's plane
  fi = hull.faceadr[refmesh] + f  # int: the face's global index
  vadr = hull.face_vadr[fi]  # int: its loop's start in face_v
  r0v = hull.face_v[vadr]  # int: the fan's apex, a local vertex
  r0 = vworld(ref, r0v)  # wp.vec3
  for k in range(1, nv - 1):
    rkv = hull.face_v[vadr + k]  # int: the fan triangle's second corner
    rlv = hull.face_v[vadr + k + 1]  # int: its third corner
    rk = vworld(ref, rkv)  # wp.vec3
    rl = vworld(ref, rlv)  # wp.vec3
    if tri_contains(r0, rk, rl, p, nr):
      if ref_is_1:
        c = set_seed(c, r0, ybest, r0v, yv_best, rk, ybest, rkv, yv_best, rl, ybest, rlv, yv_best,
                     xr, yi, xrv, yiv)
      else:
        c = set_seed(c, ybest, r0, yv_best, r0v, ybest, rk, yv_best, rkv, ybest, rl, yv_best, rlv,
                     yi, xr, yiv, xrv)
      return c
  return c


@wp.func
def certify_edge(
  g1: Geom, g2: Geom, hull: HullTablesDevice, mesh1: int, mesh2: int, e1: int, e2: int, gap: float,
) -> Cert:
  """Edge e1 of geom1 against edge e2 of geom2: the axis u along e1 x e2, oriented away from geom1's origin, its
  depth h_D(u), both edges supporting along u, and the pyramid test on the edges' parallelogram in D with
  z- = s_D(-u).

  Called by `warm_kernel`.
  In: g1, g2 (Geom) pair frame; hull (HullTablesDevice): edgeadr (nhull,) int, edge_v (nedge,) wp.vec2i, and the
  support cells; mesh1, mesh2 (int) the hull ids; e1, e2 (int) hull-local edge ids; gap (float, m).
  Out: Cert: status 2 when separated by more than gap + EPS, 1 certified, 0 failed (parallel edges too), with an
  EPA seed when one holds the origin; incv -1.
  """
  c = Cert()  # Cert: the result
  ev1 = hull.edge_v[hull.edgeadr[mesh1] + e1]  # wp.vec2i: e1's local vertices
  ev2 = hull.edge_v[hull.edgeadr[mesh2] + e2]  # wp.vec2i: e2's
  p0 = vworld(g1, ev1[0])  # wp.vec3: e1's ends p0, p1
  p1 = vworld(g1, ev1[1])  # wp.vec3
  q0 = vworld(g2, ev2[0])  # wp.vec3: e2's ends q0, q1
  q1 = vworld(g2, ev2[1])  # wp.vec3
  u = wp.cross(p1 - p0, q1 - q0)  # wp.vec3: the edge-pair axis
  lu = wp.length(u)  # float
  if lu < 1.0e-9 * wp.length(p1 - p0) * wp.length(q1 - q0):
    c.status = 0
    return c
  u = u / lu
  if wp.dot(u, p0 - g1.pos) < 0.0:
    u = -u
  x1, x1v = cell_support(hull, mesh1, g1, u)  # wp.vec3, int: geom1's support along u
  y2, y2v = cell_support(hull, mesh2, g2, -u)  # wp.vec3, int: geom2's along -u
  h1 = wp.dot(u, x1)  # float: max over geom1 of u.x
  m2 = wp.dot(u, y2)  # float: min over geom2 of u.y
  depth = h1 - m2  # float: h_D(u) (m)
  c.n = u
  c.depth = depth
  c.incv = -1
  if depth < -gap - EPS:
    c.status = 2
    return c
  # both edges must be the supporting features along u
  if wp.dot(u, p0) < h1 - EPS_SUP or wp.dot(u, p1) < h1 - EPS_SUP:
    c.status = 0
    return c
  if wp.dot(u, q0) > m2 + EPS_SUP or wp.dot(u, q1) > m2 + EPS_SUP:
    c.status = 0
    return c
  xr, xrv = cell_support(hull, mesh1, g1, -u)  # wp.vec3, int: geom1's support along -u
  yi, yiv = cell_support(hull, mesh2, g2, u)  # wp.vec3, int: geom2's along u
  zm = xr - yi  # wp.vec3: z- = s_D(-u)
  w0 = p0 - q0  # wp.vec3: the parallelogram's corners w0 to w3, points of D
  w1 = p1 - q0  # wp.vec3
  w2 = p1 - q1  # wp.vec3
  w3 = p0 - q1  # wp.vec3
  # base plane at distance depth (the parallelogram lies on it when both edges support)
  ok = wp.dot(u, w0) >= depth - EPS and wp.dot(u, w1) >= depth - EPS  # bool: the base, then every lateral plane
  ok = ok and wp.dot(u, w2) >= depth - EPS and wp.dot(u, w3) >= depth - EPS
  cin = 0.5 * (0.25 * (w0 + w1 + w2 + w3) + zm)  # wp.vec3: a point inside the pyramid
  ok = ok and lateral_ok(w0, w1, zm, cin, depth) and lateral_ok(w1, w2, zm, cin, depth)
  ok = ok and lateral_ok(w2, w3, zm, cin, depth) and lateral_ok(w3, w0, zm, cin, depth)
  c.status = wp.where(ok, 1, 0)
  if ok:
    return c
  foot = depth * u  # wp.vec3: the origin's projection onto the base plane
  if tri_contains(w0, w1, w2, foot, u):
    c = set_seed(c, p0, q0, ev1[0], ev2[0], p1, q0, ev1[1], ev2[0], p1, q1, ev1[1], ev2[1], xr, yi, xrv, yiv)
  elif tri_contains(w0, w2, w3, foot, u):
    c = set_seed(c, p0, q0, ev1[0], ev2[0], p1, q1, ev1[1], ev2[1], p0, q1, ev1[0], ev2[1], xr, yi, xrv, yiv)
  return c


#######################################################################################
# Pair setup shared by the kernels
#######################################################################################


@wp.func
def hull_of(mm: MeshModel, g: int, worldid: int) -> int:
  """Hull id of geom g in this world: its mesh, its box hull, or -1 (no hull).

  Called by `load_geom`, `warm_kernel`, `gjk_kernel` and 4 others.
  Reads: mm.geom_hull (ngeom,) int, mm.geom_dataid (1 or nworld, ngeom) int. In: g, worldid (int). Out: int.
  """
  h = mm.geom_hull[g]  # int: the box hull id, -1 a mesh geom, -2 none
  if h == -1:
    return mm.geom_dataid[worldid % mm.geom_dataid.shape[0], g]
  return wp.max(h, -1)


@wp.func
def load_geom(mm: MeshModel, g: int, worldid: int) -> Geom:
  """The Geom fields the hybrid reads: pose, size, and the hull's vertices in hull_vert.

  Called by `load_pair` and `plane_kernel`.
  Reads: mm.geom_xpos (nworld, ngeom) wp.vec3, mm.geom_xmat (nworld, ngeom) wp.mat33, mm.geom_size (1 or nworld,
  ngeom) wp.vec3, mm.hull_vert (nvert,) wp.vec3, mm.hull_vertadr and mm.hull_vertnum (nhull,) int, and `hull_of`'s.
  In: g, worldid (int). Out: Geom, world frame; vertadr and vertnum -1 without a hull.
  """
  geom = Geom()  # Geom: the result
  geom.pos = mm.geom_xpos[worldid, g]
  geom.rot = mm.geom_xmat[worldid, g]
  geom.size = mm.geom_size[worldid % mm.geom_size.shape[0], g]
  geom.normal = wp.vec3(geom.rot[0, 2], geom.rot[1, 2], geom.rot[2, 2])
  h = hull_of(mm, g, worldid)  # int: the hull id, -1 none
  geom.vert = mm.hull_vert
  geom.vertadr = wp.where(h >= 0, mm.hull_vertadr[wp.max(h, 0)], -1)
  geom.vertnum = wp.where(h >= 0, mm.hull_vertnum[wp.max(h, 0)], -1)
  geom.graphadr = -1
  geom.margin = 0.0
  geom.index = -1
  return geom


@wp.func
def load_pair(mm: MeshModel, geoms: wp.vec2i, worldid: int):
  """Both geoms of a pair in a frame translated to geom1's origin (MJWarp's convention).

  Called by `warm_kernel`, `gjk_kernel`, `epa_kernel` and 2 others.
  Reads: as `load_geom`. In: geoms (wp.vec2i) geom1, geom2; worldid (int).
  Out: (Geom, Geom, wp.vec3): geom1, geom2, and the frame's origin, geom1's world position.
  """
  g1 = load_geom(mm, geoms[0], worldid)  # Geom
  g2 = load_geom(mm, geoms[1], worldid)  # Geom
  origin = g1.pos  # wp.vec3: geom1's world position
  g1.pos = wp.vec3(0.0)
  g2.pos = g2.pos - origin
  return g1, g2, origin


@wp.func
def load_pair_mjwarp(mm: MeshModel, geoms: wp.vec2i, worldid: int):
  """MJWarp's own Geom setup of a mesh-mesh pair (with the mesh graph and polygons its support,
  GJK and multicontact read), same frame, for comparing the hybrid with MJWarp's own
  narrowphase and for debugging.

  Not called in the repository.
  Reads: mm.geom_dataid, mm.geom_size, the mm.mesh_* arrays, mm.geom_xpos and mm.geom_xmat (see `MeshModel`).
  In: geoms (wp.vec2i); worldid (int). Out: (Geom, Geom, wp.vec3) as `load_pair`, margins 0.
  """
  g1, g2 = geom_collision_pair_from_types(  # Geom, Geom
    mm.geom_dataid, mm.geom_size, mm.mesh_vertadr, mm.mesh_vertnum, mm.mesh_graphadr, mm.mesh_vert,
    mm.mesh_graph, mm.mesh_polynum, mm.mesh_polyadr, mm.mesh_polynormal, mm.mesh_polyvertadr,
    mm.mesh_polyvertnum, mm.mesh_polyvert, mm.mesh_polymapadr, mm.mesh_polymapnum, mm.mesh_polymap,
    mm.geom_xpos, mm.geom_xmat, MESH, MESH, geoms, worldid,
  )
  origin = g1.pos  # wp.vec3: geom1's world position
  g1.pos = wp.vec3(0.0)
  g2.pos = g2.pos - origin
  g1.margin = 0.0
  g2.margin = 0.0
  return g1, g2, origin


@wp.func
def push_cache(st: PipeState, worldid: int, key: int, feat: int, dir: wp.vec3):
  """Stores (key, feat, dir) in the world's table in the write half: linear probing from `hslot`, an empty entry
  claimed by compare-and-swap or the entry already holding key overwritten; a full table counts cap_overflow.

  Called by `warm_kernel` and `finish_epa`.
  In: worldid (int); key (int) a `pair_key`; feat (int) a feature id or FEAT_DIR; dir (wp.vec3) the separating
  direction, geom1 frame (zero with a feature).
  Reads: st.cache_parity (1,) int, st.tbits. Writes: st.cache_key and st.cache_feat (2 * nworld << tbits,) int,
  st.cache_dir (2 * nworld << tbits,) wp.vec3; st.stats[9] when full.
  """
  size = 1 << st.tbits  # int: entries per world's table
  base = table_base(st, worldid, True)  # int: the world's table in the write half
  h = hslot(key, st.tbits)  # int: the entry probed, from base
  for _ in range(size):  # _: int, a probe
    old = wp.atomic_cas(st.cache_key, base + h, -1, key)  # int: the entry's key before the swap
    if old == -1 or old == key:
      st.cache_feat[base + h] = feat
      st.cache_dir[base + h] = dir
      return
    h = (h + 1) & (size - 1)
  wp.atomic_add(st.stats, 9, 1)  # table full


@wp.func
def push_manifold(st: PipeState, slot: int, feat: int, n: wp.vec3, depth: float, incv: int):
  """Queues a pair for `manifold_kernel`; a full queue counts cap_overflow.

  Called by `warm_kernel` and `finish_epa`.
  In: slot (int) the broadphase slot; feat (int) the feature id, 0 for none; n (wp.vec3) the normal geom1 -> geom2;
  depth (float) the exact depth (m); incv (int) the incident support vertex (FACE*), -1 for none.
  Writes: st.man_count (1,) int, added to; st.man_slot, st.man_feat and st.man_incv (naccdmax,) int, st.man_n
  (naccdmax,) wp.vec3, st.man_depth (naccdmax,) float; st.stats[9] when full.
  """
  i = wp.atomic_add(st.man_count, 0, 1)  # int: the queue entry
  if i < st.man_slot.shape[0]:
    st.man_slot[i] = slot
    st.man_feat[i] = feat
    st.man_n[i] = n
    st.man_depth[i] = depth
    st.man_incv[i] = incv
  else:
    wp.atomic_add(st.stats, 9, 1)


#######################################################################################
# Kernels
#######################################################################################


@wp.kernel(enable_backward=False, module="unique")
def clear_cache_kernel(st: PipeState):
  """Empties the write half of the cache (launched over one half).

  Launch: dim=nworld << tbits, a thread per entry of a half, by `pipeline.HybridNarrowphase.launch`.
  Reads: st.cache_parity (1,) int. Writes: st.cache_key (2 * nworld << tbits,) int, the write half to -1.
  """
  i = wp.tid()  # int: the entry, within the half
  st.cache_key[(1 - st.cache_parity[0]) * (st.cache_key.shape[0] // 2) + i] = -1


@wp.kernel(enable_backward=False, module="unique")
def flip_cache_kernel(st: PipeState):
  """This step's entries become next step's read half (launched with one thread).

  Launch: dim=1, by `pipeline.HybridNarrowphase.launch`, after every stage.
  Writes: st.cache_parity (1,) int.
  """
  st.cache_parity[0] = 1 - st.cache_parity[0]


@wp.kernel(enable_backward=False, module="unique", launch_bounds=(64, 8))
def warm_kernel(mm: MeshModel, hull: HullTablesDevice, st: PipeState, gap: float, use_cache: int):
  """The warm stage, per broadphase pair of hulls: the cached feature's certificate at the new pose, or a separation
  test along the cached direction (or the centre axis, with no entry). A certified contact goes to the manifold
  queue and the cache, a certified separation to the cache, a failed certificate with a seed (use_cache 1) to the
  EPA list, anything else to the cold list.

  Launch: dim=naconmax, block_dim=64, a thread per broadphase slot, by `pipeline.HybridNarrowphase.launch`.
  In: gap (float, m); use_cache (int) 0 no cache, 1 the cache and seeded EPA, 2 the cache without seeded EPA.
  Reads: st.ncollision (1,) int, st.collision_pair (naconmax,) wp.vec2i, st.collision_worldid (naconmax,) int, the
  read half of the cache; mm's poses and hulls (`load_pair`); hull.
  Writes: the write half of the cache (`push_cache`); st.cold_count (1,) int, st.cold_slot (naconmax,) int;
  st.epa_count (1,) int, st.epa_slot (naccdmax,) int, st.epa_gjk (naccdmax,) GJKResult; the manifold queue
  (`push_manifold`); st.dbg_path (naconmax,) int; st.stats.
  """
  slot = wp.tid()  # int: the broadphase slot
  if slot >= st.ncollision[0] or slot >= st.collision_pair.shape[0]:
    return
  geoms = st.collision_pair[slot]  # wp.vec2i: geom1, geom2
  if mm.pair_ok[mm.geom_type[geoms[0]] * NTYPE + mm.geom_type[geoms[1]]] == 0:
    return
  worldid = st.collision_worldid[slot]  # int
  m1 = hull_of(mm, geoms[0], worldid)  # int: geom1's hull
  m2 = hull_of(mm, geoms[1], worldid)  # int: geom2's hull
  if m1 < 0 or m2 < 0:
    return
  count(st, 0)
  key = pair_key(geoms[0], geoms[1])  # int: the cache key
  idx = int(-1)  # int: the pair's entry in the read half, -1 none
  if use_cache != 0:
    idx = cache_lookup(st, worldid, key)
  g1, g2, origin = load_pair(mm, geoms, worldid)  # Geom, Geom, wp.vec3: the pair frame's geoms and origin
  feat = int(0)  # int: the cached feature id
  if idx >= 0:
    count(st, 1)
    feat = st.cache_feat[idx]
  if idx < 0 or feat == FEAT_DIR:
    # separation test along the cached direction, or the centre axis (= GJK's first test)
    u = wp.vec3(0.0)  # wp.vec3: the test axis, geom1 -> geom2
    if idx >= 0:
      u = g1.rot @ st.cache_dir[idx]
    else:
      u = g2.pos - g1.pos
      lu = wp.length(u)  # float
      if lu > 1.0e-12:
        u = u / lu
    x, xv = cell_support(hull, m1, g1, u)  # wp.vec3, int: geom1's support along u
    y, yv = cell_support(hull, m2, g2, -u)  # wp.vec3, int: geom2's along -u
    if wp.dot(u, y) - wp.dot(u, x) > gap + EPS:
      st.dbg_path[slot] = 1
      count(st, 3)
      if use_cache != 0:
        push_cache(st, worldid, key, FEAT_DIR, wp.transpose(g1.rot) @ u)
      return
    j = wp.atomic_add(st.cold_count, 0, 1)  # int: the cold-list entry (later: the EPA-list entry)
    st.dbg_path[slot] = 5
    if j < st.cold_slot.shape[0]:
      st.cold_slot[j] = slot
    return
  kind = feat_kind(feat)  # int: FEAT_FACE1, FEAT_FACE2 or FEAT_EDGE
  c = Cert()  # Cert: the feature's certificate
  if kind == FEAT_FACE1:
    c = certify_face(g1, g2, hull, m1, m2, feat_a(feat), True, gap)
  elif kind == FEAT_FACE2:
    c = certify_face(g2, g1, hull, m2, m1, feat_a(feat), False, gap)
  else:
    c = certify_edge(g1, g2, hull, m1, m2, feat_a(feat), feat_b(feat), gap)
  if c.status == 2:
    st.dbg_path[slot] = 2
    count(st, 3)
    push_cache(st, worldid, key, feat, wp.vec3(0.0))
    return
  if c.status == 1 and c.depth <= EPS:
    c.status = 0  # within EPS of touching: let GJK/EPA decide, as MJWarp does
    c.seed = 0
  if c.status == 1:
    st.dbg_path[slot] = 3
    count(st, 2)
    push_cache(st, worldid, key, feat, wp.vec3(0.0))
    push_manifold(st, slot, feat, c.n, c.depth, c.incv)
    return
  if c.seed == 1 and use_cache == 1:  # use_cache = 2: cache without seeded EPA
    # certify-or-improve: EPA from the cached facet's tetrahedron, no GJK
    j = wp.atomic_add(st.epa_count, 0, 1)
    if j < st.epa_slot.shape[0]:
      count(st, 12)
      st.dbg_path[slot] = 4
      gr = GJKResult()  # GJKResult: the seed, as a GJK simplex for EPA
      gr.separated = False
      gr.dist = 0.0
      gr.dim = 4
      gr.simplex1 = c.s1
      gr.simplex2 = c.s2
      gr.simplex = c.s1 - c.s2
      gr.simplex_index1 = c.i1
      gr.simplex_index2 = c.i2
      gr.x1 = wp.vec3(c.s1[0, 0], c.s1[0, 1], c.s1[0, 2])
      gr.x2 = wp.vec3(c.s2[0, 0], c.s2[0, 1], c.s2[0, 2])
      gr.index1 = -1
      gr.index2 = -1
      st.epa_slot[j] = slot
      st.epa_gjk[j] = gr
      return
  j = wp.atomic_add(st.cold_count, 0, 1)
  st.dbg_path[slot] = 5
  if j < st.cold_slot.shape[0]:
    st.cold_slot[j] = slot


@wp.func
def simplex_direction(r: GJKResult) -> wp.vec3:
  """Unit direction geom1 -> geom2 at GJK's final simplex: minus its closest point to the origin.

  Solved in float64. The simplex vertices can be centimetres from the origin while the distance
  is microns, so the float32 closest point (or the witness difference x2 - x1) loses its
  direction to cancellation, and the separation measured along it can be off by more than the
  distance itself.

  Called by `gjk_kernel`.
  In: r (GJKResult): simplex (mat43) the simplex's points of D, pair frame; dim (int) how many; x1, x2 (wp.vec3) the
  witnesses. Out: wp.vec3, unit; normalize(x2 - x1) when the closest point is the origin.
  """
  w0 = wp.vec3d(wp.float64(r.simplex[0, 0]), wp.float64(r.simplex[0, 1]), wp.float64(r.simplex[0, 2]))  # wp.vec3d
  w1 = wp.vec3d(wp.float64(r.simplex[1, 0]), wp.float64(r.simplex[1, 1]), wp.float64(r.simplex[1, 2]))  # wp.vec3d
  w2 = wp.vec3d(wp.float64(r.simplex[2, 0]), wp.float64(r.simplex[2, 1]), wp.float64(r.simplex[2, 2]))  # wp.vec3d
  c = w0  # wp.vec3d: the closest point
  if r.dim == 2:
    e = w1 - w0  # wp.vec3d: the segment
    ee = wp.dot(e, e)  # float64
    if ee > wp.float64(0.0):
      c = w0 - (wp.dot(w0, e) / ee) * e
  elif r.dim == 3:
    e1 = w1 - w0  # wp.vec3d: the triangle's edges e1, e2
    e2 = w2 - w0  # wp.vec3d
    a11 = wp.dot(e1, e1)  # float64: the normal equations' matrix a11, a12, a22
    a12 = wp.dot(e1, e2)  # float64
    a22 = wp.dot(e2, e2)  # float64
    b1 = -wp.dot(w0, e1)  # float64: and right-hand side b1, b2
    b2 = -wp.dot(w0, e2)  # float64
    det = a11 * a22 - a12 * a12  # float64
    if det > wp.float64(0.0):
      c = w0 + ((b1 * a22 - b2 * a12) / det) * e1 + ((a11 * b2 - a12 * b1) / det) * e2
  lc = wp.length(c)  # float64
  if lc > wp.float64(0.0):
    return -wp.vec3(float(c[0] / lc), float(c[1] / lc), float(c[2] / lc))
  return wp.normalize(r.x2 - r.x1)


@wp.kernel(enable_backward=False, module="unique", launch_bounds=(64, 8))
def gjk_kernel(mm: MeshModel, hull: HullTablesDevice, st: PipeState, gap: float):
  """The GJK stage, per cold pair: MJWarp's GJK on cell supports, with its separating-plane exit at gap + SEP_TOL.
  A separation the exit proves, or exact supports prove along the closest-point direction, ends the pair, as does an
  unproven one-point simplex; anything else goes to the EPA list.

  Launch: dim=naconmax, block_dim=64, a thread per cold-list entry, by `pipeline.HybridNarrowphase.launch`.
  In: gap (float, m).
  Reads: st.cold_count (1,) int, st.cold_slot (naconmax,) int, st.collision_pair (naconmax,) wp.vec2i,
  st.collision_worldid (naconmax,) int; mm's poses and hulls (`load_pair`); hull.
  Writes: st.epa_count (1,) int, st.epa_slot (naccdmax,) int, st.epa_gjk (naccdmax,) GJKResult; st.dbg_path
  (naconmax,) int; st.stats, cap_overflow counted for each pair a full EPA list drops.
  """
  tid = wp.tid()  # int: the cold-list entry
  if tid >= wp.min(st.cold_count[0], st.cold_slot.shape[0]):
    return
  count(st, 4)
  slot = st.cold_slot[tid]  # int: the broadphase slot
  geoms = st.collision_pair[slot]  # wp.vec2i
  worldid = st.collision_worldid[slot]  # int
  g1, g2, origin = load_pair(mm, geoms, worldid)  # Geom, Geom, wp.vec3
  m1 = hull_of(mm, geoms[0], worldid)  # int: geom1's hull
  m2 = hull_of(mm, geoms[1], worldid)  # int: geom2's hull
  # GJK with its separating-plane exit at a margin: ccd.py's GJK tests along the direction the
  # support was taken in, so an exit is a proof that all of D lies more than
  # gap + SEP_TOL beyond the origin along that direction (MJWarp's own exit is not, near touching)
  gres = gjk(mm.gjk_iterations, hull, m1, m2, g1, g2, g1.pos, g2.pos, gap + SEP_TOL)  # GJKResult
  if gres.separated and gres.dim == 0:
    st.dbg_path[slot] = 6
    return  # certified separated by the exit
  needs_epa = not (gres.dist > mm.ccd_tolerance or gres.dim < 2 or gres.separated)  # bool
  if not needs_epa:
    # GJK stopped on a distance: prove the separation along its closest-point direction with
    # exact supports. Unproven -> the distance is not trusted and EPA decides: its initial
    # polytope must contain the origin (_polytope3's test beyond 1e-5), which proves the
    # penetration, so a separated pair cannot come back as a contact. A one-point simplex
    # cannot seed EPA; it stays separated, counted.
    u = simplex_direction(gres)  # wp.vec3: unit, geom1 -> geom2
    x, xv = cell_support(hull, m1, g1, u)  # wp.vec3, int: geom1's support along u
    y, yv = cell_support(hull, m2, g2, -u)  # wp.vec3, int: geom2's along -u
    if wp.dot(u, y) - wp.dot(u, x) <= gap + SEP_TOL:
      if gres.dim >= 2:
        count(st, wp.where(gres.dist <= EPS, 13, 14))
        needs_epa = True
      else:
        count(st, 15)
    if not needs_epa:
      st.dbg_path[slot] = 6
      return  # certified separated (no contact at gap = 0)
  j = wp.atomic_add(st.epa_count, 0, 1)  # int: the EPA-list entry
  if j >= st.epa_slot.shape[0]:
    wp.atomic_add(st.stats, 9, 1)  # EPA list full
    return
  st.epa_slot[j] = slot
  st.epa_gjk[j] = gres
  st.dbg_path[slot] = 7


@wp.func
def on_face(geom: Geom, hull: HullTablesDevice, v: int, f: int) -> bool:
  """Local vertex v lies on local face f, by v's incidence list (a face loop can be long).

  Called by `face_with`.
  Reads: geom.vertadr; hull.vf_adr and hull.vf_num (nvert,) int, hull.vf (nloop,) int. In: v, f (int). Out: bool.
  """
  gv = geom.vertadr + v  # int: v's global vertex id
  for j in range(hull.vf_num[gv]):  # j: int, an incidence of v
    if hull.vf[hull.vf_adr[gv] + j] == f:
      return True
  return False


@wp.func
def face_with(geom: Geom, hull: HullTablesDevice, a: int, b: int, c: int) -> int:
  """Local face of the mesh whose loop holds vertices a, b and c, or -1.

  Called by `decode_epa_facet`.
  Reads: geom.vertadr; hull.vf_adr and hull.vf_num (nvert,) int, hull.vf (nloop,) int. In: a, b, c (int) hull-local
  vertex ids. Out: int.
  """
  gv = geom.vertadr + a  # int: a's global vertex id
  for j in range(hull.vf_num[gv]):  # j: int, an incidence of a
    f = hull.vf[hull.vf_adr[gv] + j]  # int: a face at a
    if on_face(geom, hull, b, f) and on_face(geom, hull, c, f):
      return f
  return -1


@wp.func
def edge_with(geom: Geom, hull: HullTablesDevice, mesh: int, a: int, b: int) -> int:
  """Local edge of hull `mesh` between local vertices a and b, or -1.

  Called by `decode_epa_facet`.
  Reads: geom.vertadr; hull.ve_adr and hull.ve_num (nvert,) int, hull.ve (2 * nedge,) int, hull.edgeadr (nhull,)
  int, hull.edge_v (nedge,) wp.vec2i. In: mesh (int) the hull id; a, b (int). Out: int.
  """
  gv = geom.vertadr + a  # int: a's global vertex id
  for j in range(hull.ve_num[gv]):  # j: int, an incidence of a
    e = hull.ve[hull.ve_adr[gv] + j]  # int: an edge at a
    ev = hull.edge_v[hull.edgeadr[mesh] + e]  # wp.vec2i: its vertices
    if (ev[0] == a and ev[1] == b) or (ev[0] == b and ev[1] == a):
      return e
  return -1


@wp.func
def distinct3(a: int, b: int, c: int) -> int:
  """Number of distinct values among a, b and c.

  Called by `decode_epa_facet`.
  In: a, b, c (int). Out: int, 1 to 3.
  """
  n = int(1)  # int: the count
  if b != a:
    n += 1
  if c != a and c != b:
    n += 1
  return n


@wp.func
def other_of(a: int, b: int, c: int, x: int) -> int:
  """Some value of (a, b, c) different from x (or x itself if all equal).

  Called by `decode_epa_facet`.
  In: a, b, c, x (int). Out: int.
  """
  return wp.where(a != x, a, wp.where(b != x, b, c))


@wp.func
def decode_epa_facet(g1: Geom, g2: Geom, hull: HullTablesDevice, m1: int, m2: int, face: int,
                     vert_index: wp.array(dtype=int)) -> int:
  """Feature of the hulls realizing EPA's minimizing polytope triangle, or 0.

  Called by `finish_epa`.
  In: g1, g2 (Geom) pair frame; m1, m2 (int) their hull ids; face (int) the triangle, three 10-bit polytope vertex
  ids; vert_index (10 + 2 * epa_iterations,) int, the pair's EPA row: per polytope vertex, geom1's then geom2's
  hull-local vertex id. Reads: hull's incidence and edges (`face_with`, `edge_with`). Out: int, a feature id.
  """
  v0 = face & 1023  # int: the triangle's polytope vertices v0, v1, v2
  v1 = (face >> 10) & 1023  # int
  v2 = (face >> 20) & 1023  # int
  a0 = vert_index[2 * v0]  # int: their geom1 vertices a0, a1, a2
  a1 = vert_index[2 * v1]  # int
  a2 = vert_index[2 * v2]  # int
  b0 = vert_index[2 * v0 + 1]  # int: their geom2 vertices b0, b1, b2
  b1 = vert_index[2 * v1 + 1]  # int
  b2 = vert_index[2 * v2 + 1]  # int
  na = distinct3(a0, a1, a2)  # int: distinct geom1 vertices
  nb = distinct3(b0, b1, b2)  # int: distinct geom2 vertices
  if na == 3:
    f = face_with(g1, hull, a0, a1, a2)  # int: the face through them, -1 none
    if f >= 0:
      return pack_feat(FEAT_FACE1, f, 0)
  if nb == 3:
    f = face_with(g2, hull, b0, b1, b2)
    if f >= 0:
      return pack_feat(FEAT_FACE2, f, 0)
  if na >= 2 and nb >= 2:
    ea = edge_with(g1, hull, m1, a0, other_of(a0, a1, a2, a0))  # int: geom1's edge, -1 none
    eb = edge_with(g2, hull, m2, b0, other_of(b0, b1, b2, b0))  # int: geom2's edge, -1 none
    if ea >= 0 and eb >= 0:
      return pack_feat(FEAT_EDGE, ea, eb)
  return 0


@wp.func
def finish_epa(
  mm: MeshModel, hull: HullTablesDevice, st: PipeState, slot: int, geoms: wp.vec2i, worldid: int, g1: Geom, g2: Geom,
  m1: int, m2: int, dist: float, ncon: int, w1: wp.vec3, w2: wp.vec3, idx: int, face_row: wp.array(dtype=int),
  vert_index_row: wp.array(dtype=int), gap: float,
):
  """EPA's result -> hull features of its minimizing triangle, exact depth and normal -> cache and
  manifold queue (or one EPA contact without features).

  Called by `epa_kernel` and `epa_warp.epa_warp_kernel`.
  In: slot (int) the broadphase slot; geoms (wp.vec2i); worldid (int); g1, g2 (Geom) pair frame; m1, m2 (int) their
  hull ids; dist (float) EPA's distance, negative in penetration (m); ncon (int) 0 when EPA failed; w1, w2
  (wp.vec3) its witnesses; idx (int) the minimizing face, -1 none; face_row (6 + MAX_EPAFACES * epa_iterations,) int
  and vert_index_row (10 + 2 * epa_iterations,) int, the pair's EPA rows; gap (float, m).
  Writes: the cache (`push_cache`), the manifold queue (`push_manifold`), st.dbg_path (naconmax,) int, st.stats.
  """
  if ncon == 0 or dist >= gap:
    st.dbg_path[slot] = wp.where(ncon == 0, 10, 9)
    if ncon == 0:
      count(st, 6)
    return  # EPA failure (MJWarp drops these too) or no penetration
  n = w1 - w2  # wp.vec3, overlapping: witness1 - witness2 points geom1 -> geom2
  ln = wp.length(n)  # float
  if ln < 1.0e-12:
    return
  n = n / ln
  depth_epa = -dist  # float: EPA's depth (m)
  key = pair_key(geoms[0], geoms[1])  # int: the cache key
  feat = int(0)  # int: the facet's feature, 0 none
  depth = depth_epa  # float: the depth queued (m)
  incv = int(-1)  # int: the incident support vertex, -1 none
  if idx >= 0:
    f = decode_epa_facet(g1, g2, hull, m1, m2, face_row[idx], vert_index_row)  # int: the decoded feature id
    kind = feat_kind(f)  # int
    # exact depth and normal: a face's from its plane and the other hull's exact support (one support query), an
    # edge pair's from the facet's own vertices (no support queries)
    if kind == FEAT_FACE1:
      nr, off = face_world(g1, hull, m1, feat_a(f))  # wp.vec3, float: the face's normal and support offset
      y, yv = cell_support(hull, m2, g2, -nr)  # wp.vec3, int, exact incident support: the deepest vertex
      dd = off - wp.dot(nr, y)  # float: the face axis's depth (m)
      if wp.abs(dd - depth_epa) < 1.0e-5:
        feat = f
        depth = dd
        n = nr
        incv = yv
    elif kind == FEAT_FACE2:
      nr, off = face_world(g2, hull, m2, feat_a(f))
      x, xv = cell_support(hull, m1, g1, -nr)  # wp.vec3, int: the exact incident support
      dd = off - wp.dot(nr, x)
      if wp.abs(dd - depth_epa) < 1.0e-5:
        feat = f
        depth = dd
        n = -nr
        incv = xv
    elif kind == FEAT_EDGE:
      ev1 = hull.edge_v[hull.edgeadr[m1] + feat_a(f)]  # wp.vec2i: geom1's edge's vertices
      ev2 = hull.edge_v[hull.edgeadr[m2] + feat_b(f)]  # wp.vec2i: geom2's
      p0 = vworld(g1, ev1[0])  # wp.vec3
      q0 = vworld(g2, ev2[0])  # wp.vec3
      u = wp.cross(vworld(g1, ev1[1]) - p0, vworld(g2, ev2[1]) - q0)  # wp.vec3: the edge-pair axis
      lu = wp.length(u)  # float
      if lu > 1.0e-12:
        u = u / lu
        if wp.dot(u, n) < 0.0:
          u = -u
        dd = wp.dot(u, p0 - q0)
        if wp.abs(dd - depth_epa) < 1.0e-5:
          feat = f
          depth = dd
          n = u
  if feat == 0:
    count(st, 7)  # no clean facet: one EPA contact, nothing cached
    push_manifold(st, slot, 0, n, depth_epa, -1)
    return
  count(st, 10)
  st.dbg_path[slot] = st.dbg_path[slot] * 100 + 8  # x08: contact via EPA (x = 4 seeded, 7 cold)
  push_cache(st, worldid, key, feat, wp.vec3(0.0))
  push_manifold(st, slot, feat, n, depth, incv)


@wp.kernel(enable_backward=False, module="unique", launch_bounds=(64, 8))
def epa_kernel(mm: MeshModel, hull: HullTablesDevice, st: PipeState, gap: float):
  """The EPA stage with a thread per pair: MJWarp's EPA on cell supports from the pair's GJK result (or seed
  tetrahedron), then `finish_epa`.

  Launch: dim=naccdmax, block_dim=64, a thread per EPA-list entry, by `pipeline.HybridNarrowphase.launch` when
  epa_mode is "thread".
  In: gap (float, m).
  Reads: st.epa_count (1,) int, st.epa_slot (naccdmax,) int, st.epa_gjk (naccdmax,) GJKResult, the broadphase
  pair; mm's poses and hulls (`load_pair`); hull.
  Writes: the pair's EPA rows: st.epa_vert (naccdmax, 10 + 2 * epa_iterations) wp.vec3, st.epa_vert_index (same)
  int, st.epa_face, st.epa_pr and st.epa_norm2 (naccdmax, 6 + MAX_EPAFACES * epa_iterations) int, wp.vec3 and
  float, st.epa_horizon (naccdmax, MAX_EPAHORIZON) int; st.overflow (max(nworld, 1),) int; and `finish_epa`'s.
  """
  tid = wp.tid()  # int: the EPA-list entry
  if tid >= wp.min(st.epa_count[0], st.epa_slot.shape[0]):
    return
  count(st, 5)
  slot = st.epa_slot[tid]  # int: the broadphase slot
  geoms = st.collision_pair[slot]  # wp.vec2i
  worldid = st.collision_worldid[slot]  # int
  g1, g2, origin = load_pair(mm, geoms, worldid)  # Geom, Geom, wp.vec3
  m1 = hull_of(mm, geoms[0], worldid)  # int: geom1's hull
  m2 = hull_of(mm, geoms[1], worldid)  # int: geom2's hull
  gr = st.epa_gjk[tid]  # GJKResult: the pair's GJK result or seed
  dist, ncon, w1, w2, idx = epa_phase(  # float, int, wp.vec3, wp.vec3, int: as `ccd.epa_phase` returns them
    mm.epa_iterations, gr, hull, m1, m2, g1, g2, st.epa_vert[tid], st.epa_vert_index[tid], st.epa_face[tid],
    st.epa_pr[tid], st.epa_norm2[tid], st.epa_horizon[tid], worldid, st.overflow,
  )
  finish_epa(mm, hull, st, slot, geoms, worldid, g1, g2, m1, m2, dist, ncon, w1, w2, idx, st.epa_face[tid],
             st.epa_vert_index[tid], gap)


@wp.func
def write_contact(st: PipeState, slot: int, k: int, geoms: wp.vec2i, worldid: int, pos: wp.vec3, n: wp.vec3,
                  dist: float):
  """Appends a contact to the hybrid's contact buffers; past their capacity it is dropped, still counted.

  Called by `manifold_kernel` and `plane_kernel`.
  In: slot (int) the broadphase slot; k (int) the contact's index within its pair; geoms (wp.vec2i); worldid (int);
  pos (wp.vec3) world position; n (wp.vec3) the normal geom1 -> geom2; dist (float) negative in penetration (m).
  Writes: st.con_count (1,) int, added to; st.con_slot, st.con_k and st.con_worldid (4 * naccdmax,) int, st.con_dist
  (4 * naccdmax,) float, st.con_pos and st.con_normal (4 * naccdmax,) wp.vec3, st.con_geom (4 * naccdmax,) wp.vec2i.
  """
  i = wp.atomic_add(st.con_count, 0, 1)  # int: the contact's entry
  if i < st.con_dist.shape[0]:
    st.con_slot[i] = slot
    st.con_k[i] = k
    st.con_dist[i] = dist
    st.con_pos[i] = pos
    st.con_normal[i] = n
    st.con_geom[i] = geoms
    st.con_worldid[i] = worldid


@wp.func
def seg_closest(p0: wp.vec3, p1: wp.vec3, q0: wp.vec3, q1: wp.vec3):
  """Closest points of the segments p0-p1 and q0-q1 (Ericson's ClosestPtSegmentSegment).

  Called by `manifold_kernel`.
  In: p0, p1, q0, q1 (wp.vec3). Out: (wp.vec3, wp.vec3), the point on p0-p1 and the point on q0-q1.
  """
  d1 = p1 - p0  # wp.vec3
  d2 = q1 - q0  # wp.vec3
  r = p0 - q0  # wp.vec3
  a = wp.dot(d1, d1)  # float
  e = wp.dot(d2, d2)  # float
  f = wp.dot(d2, r)  # float
  c = wp.dot(d1, r)  # float
  b = wp.dot(d1, d2)  # float
  den = a * e - b * b  # float
  s = float(0.0)  # float: the parameter on p0-p1, 0 to 1
  if den > 1.0e-20:
    s = wp.clamp((b * f - c * e) / den, 0.0, 1.0)
  t = (b * s + f) / wp.max(e, 1.0e-20)  # float: the parameter on q0-q1
  if t < 0.0:
    t = 0.0
    s = wp.clamp(-c / wp.max(a, 1.0e-20), 0.0, 1.0)
  elif t > 1.0:
    t = 1.0
    s = wp.clamp((b - c) / wp.max(a, 1.0e-20), 0.0, 1.0)
  return p0 + s * d1, q0 + t * d2


@wp.kernel(enable_backward=False, module="unique", launch_bounds=(64, 8))
def manifold_kernel(mm: MeshModel, hull: HullTablesDevice, st: PipeState, gap: float):
  """The manifold stage, per queued pair: the contacts of its feature. A face (or an edge pair that is a face patch):
  the quadrangle clip, its penetrating points reduced to at most 4 by area, plus the support vertex when the clip
  loses the deepest point; an edge pair: one point between the edges' closest points; no feature: one point between
  the supports along n.

  Launch: dim=naccdmax, block_dim=64, a thread per manifold-queue entry, by `pipeline.HybridNarrowphase.launch`.
  In: gap (float, m).
  Reads: st.man_count (1,) int; st.man_slot, st.man_feat and st.man_incv (naccdmax,) int, st.man_n (naccdmax,)
  wp.vec3, st.man_depth (naccdmax,) float; the broadphase pair; mm's poses and hulls (`load_pair`); hull's faces,
  loops, quads (face_q) and edges.
  Writes: the contacts (`write_contact`); st.dbg_man_min (naccdmax,) float; st.stats.
  """
  tid = wp.tid()  # int: the manifold-queue entry
  if tid >= wp.min(st.man_count[0], st.man_slot.shape[0]):
    return
  st.dbg_man_min[tid] = 1.0e30
  slot = st.man_slot[tid]  # int: the broadphase slot
  feat = st.man_feat[tid]  # int: the feature id, 0 none
  n = st.man_n[tid]  # wp.vec3: the normal geom1 -> geom2
  depth = st.man_depth[tid]  # float: the exact depth (m)
  geoms = st.collision_pair[slot]  # wp.vec2i
  worldid = st.collision_worldid[slot]  # int
  g1, g2, origin = load_pair(mm, geoms, worldid)  # Geom, Geom, wp.vec3: the pair frame's geoms and origin
  m1 = hull_of(mm, geoms[0], worldid)  # int: geom1's hull
  m2 = hull_of(mm, geoms[1], worldid)  # int: geom2's hull
  kind = feat_kind(feat)  # int: FEAT_FACE1, FEAT_FACE2, FEAT_EDGE or 0
  f = feat_a(feat)  # int: the reference face (or geom1's edge)
  incv = st.man_incv[tid]  # int: the incident support vertex, -1 none
  if kind == FEAT_EDGE:
    # a face at either edge within FACE_TOL of the normal whose depth nearly attains the exact
    # one: a face patch, clipped as for a face feature (the shallowest such face)
    ef1 = hull.edge_f[hull.edgeadr[m1] + feat_a(feat)]  # wp.vec2i: the faces at geom1's edge
    ef2 = hull.edge_f[hull.edgeadr[m2] + feat_b(feat)]  # wp.vec2i: at geom2's
    fdepth = depth + PATCH_TOL  # float: the depth a patch face must not exceed, then its depth (m)
    fkind = int(0)  # int: the patch face's kind, 0 none
    for j in range(4):  # j: int, geom1's two faces, then geom2's
      ong1 = j < 2  # bool: a face of geom1
      fj = wp.where(ong1, ef1[j % 2], ef2[j % 2])  # int: the face
      nr = wp.vec3(0.0)  # wp.vec3: its outward normal
      off = float(0.0)  # float: its support offset
      if ong1:
        nr, off = face_world(g1, hull, m1, fj)
      else:
        nr, off = face_world(g2, hull, m2, fj)
      if wp.dot(nr, wp.where(ong1, n, -n)) > FACE_TOL:
        y = wp.vec3(0.0)  # wp.vec3: the other geom's support along -nr
        yv = int(-1)  # int: its vertex
        if ong1:
          y, yv = cell_support(hull, m2, g2, -nr)
        else:
          y, yv = cell_support(hull, m1, g1, -nr)
        dj = off - wp.dot(nr, y)  # float: the face axis's depth (m)
        if dj <= fdepth:
          fdepth = dj
          fkind = wp.where(ong1, FEAT_FACE1, FEAT_FACE2)
          f = fj
          incv = yv
    if fkind == 0:
      ev1 = hull.edge_v[hull.edgeadr[m1] + feat_a(feat)]  # wp.vec2i: geom1's edge's vertices
      ev2 = hull.edge_v[hull.edgeadr[m2] + feat_b(feat)]  # wp.vec2i: geom2's
      # wp.vec3, wp.vec3: the edges' closest points
      c1, c2 = seg_closest(vworld(g1, ev1[0]), vworld(g1, ev1[1]), vworld(g2, ev2[0]), vworld(g2, ev2[1]))
      write_contact(st, slot, 0, geoms, worldid, 0.5 * (c1 + c2) + origin, n, -depth)
      return
    kind = fkind
    depth = fdepth
    if kind == FEAT_FACE1:
      nf1, of1 = face_world(g1, hull, m1, f)  # wp.vec3, float
      n = nf1
    else:
      nf2, of2 = face_world(g2, hull, m2, f)  # wp.vec3, float
      n = -nf2
  if kind == 0:
    # EPA fallback without features: one contact at the support points along n
    x, xv = cell_support(hull, m1, g1, n)  # wp.vec3, int: geom1's support along n
    y, yv = cell_support(hull, m2, g2, -n)
    write_contact(st, slot, 0, geoms, worldid, 0.5 * (x + y) + origin, n, -depth)
    return
  # face reference: ref/inc geoms and meshes by kind
  ref = g1  # Geom: the reference geom
  inc = g2  # Geom: the incident geom
  refmesh = m1  # int: its hull
  incmesh = m2  # int: its hull
  if kind == FEAT_FACE2:
    ref = g2
    inc = g1
    refmesh = m2
    incmesh = m1
  nr, off = face_world(ref, hull, refmesh, f)
  y = vworld(inc, incv)
  # quadrangles through the witness vertices (hull.face_q): the incident face's largest quad
  # through the exact support vertex y, clipped against the side planes of the reference face's
  # largest quad through its vertex nearest to y; faces with <= 4 vertices are their own quad
  g, gk = best_face_around(inc, hull, incmesh, incv, -nr)  # int, int: the incident face, y's position in its loop
  ncon = int(0)  # int: the contacts written
  if g >= 0:
    fr = hull.faceadr[refmesh] + f  # int: the reference face's global index
    kr = int(0)  # int: its loop position nearest to y
    dmin = float(1.0e30)  # float: that corner's squared distance to y
    for k in range(hull.face_vnum[fr]):  # k: int, a loop position
      dk = wp.length_sq(loop_vertex(ref, hull, refmesh, f, k) - y)  # float
      if dk < dmin:
        dmin = dk
        kr = k
    rq = hull.face_q[hull.face_vadr[fr] + kr]  # wp.vec3i: the reference quad's other corners
    r = mat43()  # mat43: the reference quad's corners, pair frame
    r[0] = loop_vertex(ref, hull, refmesh, f, kr)
    for k in range(3):
      r[k + 1] = vworld(ref, rq[k])
    iq = hull.face_q[hull.face_vadr[hull.faceadr[incmesh] + g] + gk]  # wp.vec3i: the incident quad's other corners
    buf = mat83()  # mat83: the clipped polygon's corners
    out = mat83()  # mat83: a clip's output, then the points kept
    buf[0] = y
    for k in range(3):
      buf[k + 1] = vworld(inc, iq[k])
    npt = int(4)  # int: corners in buf
    for k in range(4):
      a = r[(k + 3) % 4]  # wp.vec3: the clipping edge's first corner
      b = r[k]  # wp.vec3: its second corner
      s = wp.cross(b - a, nr)  # wp.vec3: the side plane's outward normal
      ls = wp.length(s)  # float
      if ls > 1.0e-12 and npt > 0:  # zero-length edges (a triangle's repeated corner) do not clip
        s = s / ls
        h = wp.dot(s, a)  # float: the side plane's offset
        nout = int(0)  # int: corners in out
        prev = buf[npt - 1]  # wp.vec3: the previous corner
        dprev = wp.dot(s, prev) - h  # float: its distance outside the plane
        for j in range(npt):
          cur = buf[j]  # wp.vec3: the current corner
          dcur = wp.dot(s, cur) - h  # float: its distance outside the plane
          # a convex quad gains at most one corner per plane (4 -> 8); the bound guards the
          # buffer when float noise makes an edge-on incident quad non-convex in projection
          if (dprev <= 0.0) != (dcur <= 0.0) and nout < 8:
            out[nout] = prev + (dprev / (dprev - dcur)) * (cur - prev)
            nout += 1
          if dcur <= 0.0 and nout < 8:
            out[nout] = cur
            nout += 1
          prev = cur
          dprev = dcur
        for j in range(nout):
          buf[j] = out[j]
        npt = nout
    # keep penetrating points (dist < gap), then reduce to at most 4 by area
    dep = vec8()  # vec8: the kept points' distances (m)
    nkeep = int(0)  # int: the points kept
    for j in range(npt):
      p = buf[j]  # wp.vec3: a clipped corner
      dj = wp.dot(nr, p) - off
      if dj < gap:
        out[nkeep] = p
        dep[nkeep] = dj
        nkeep += 1
    if nkeep > 0:
      i0 = int(0)  # int: the deepest kept point
      for j in range(nkeep):
        if dep[j] < dep[i0]:
          i0 = j
      p0 = out[i0]  # wp.vec3
      i1 = i0  # int: the kept point farthest from p0
      best = float(MIN_SEP * MIN_SEP)  # float: the best score so far
      for j in range(nkeep):
        dd = wp.length_sq(out[j] - p0)  # float
        if dd > best:
          best = dd
          i1 = j
      p1 = out[i1]  # wp.vec3
      # corners 2-4 must lie MIN_SEP from the chosen ones (closer ones are redundant contacts:
      # tiny patches clip into corners nanometres apart), and corners 3 and 4 must add (twice the)
      # area beyond AREA_REL * |p1 - p0|^2: in float32 with FMA contraction cross(v, v) is a
      # rounding residue of ~1e-7 |v|^2, not 0, and the same corner could otherwise be picked twice
      amin = wp.max(AREA_REL * wp.length_sq(p1 - p0), MIN_SEP * wp.length(p1 - p0))  # float: the least (twice) area
      i2 = int(-1)  # int: the third corner, -1 none
      best = float(amin)
      for j in range(nkeep):
        ar = wp.dot(wp.cross(p1 - p0, out[j] - p0), nr)  # float: twice the signed area of (p0, p1, point j)
        if wp.abs(ar) > best and j != i0 and j != i1:
          best = wp.abs(ar)
          i2 = j
      i3 = int(-1)  # int: the fourth corner, -1 none
      if i2 >= 0:
        p2 = out[i2]  # wp.vec3
        if wp.dot(wp.cross(p1 - p0, p2 - p0), nr) < 0.0:
          tmp = i1  # int
          i1 = i2
          i2 = tmp
          p1 = out[i1]
          p2 = out[i2]
        best = -amin
        for j in range(nkeep):
          q = out[j]  # wp.vec3: a candidate fourth corner
          a01 = wp.dot(wp.cross(p1 - p0, q - p0), nr)  # float: twice q's signed areas with edges 01, 12, 20
          a12 = wp.dot(wp.cross(p2 - p1, q - p1), nr)  # float
          a20 = wp.dot(wp.cross(p0 - p2, q - p2), nr)  # float
          worst = wp.min(a01, wp.min(a12, a20))  # float: the most negative, q's reach outside the triangle
          if worst < best and j != i0 and j != i1 and j != i2:
            best = worst
            i3 = j
      for k in range(4):
        ii = wp.where(k == 0, i0, wp.where(k == 1, i1, wp.where(k == 2, i2, i3)))  # int: the k-th corner, -1 none
        if ii >= 0 and (k == 0 or ii != i0):
          p = out[ii]
          dj = dep[ii]
          write_contact(st, slot, ncon, geoms, worldid, p - 0.5 * dj * nr + origin, n, dj)
          st.dbg_man_min[tid] = wp.min(st.dbg_man_min[tid], dj)
          ncon += 1
  if ncon == 0 or st.dbg_man_min[tid] > -depth + 1.0e-5:
    # the clip lost the deepest point (the support set is an edge crossing the reference face's
    # boundary, or y lies outside the reference quad): keep the exact support vertex as a
    # contact so the depth stays exact
    count(st, 8)
    dj = wp.dot(nr, y) - off
    if dj < gap:
      write_contact(st, slot, ncon, geoms, worldid, y - 0.5 * dj * nr + origin, n, dj)
      st.dbg_man_min[tid] = wp.min(st.dbg_man_min[tid], dj)


BAND_LEN = 32  # plane contacts: band vertices gathered by the flood fill before falling back to scans
BAND_MAX = wp.constant(BAND_LEN)  # BAND_LEN, for kernels
vec_band = wp.types.vector(length=BAND_LEN, dtype=int)  # the band's hull-local vertex ids


@wp.func
def plane_score(mode: int, v: wp.vec3, a: wp.vec3, b: wp.vec3, nl: wp.vec3, ab: wp.vec3, ac: wp.vec3, bc: wp.vec3) -> float:
  """MuJoCo's plane_convex picks: 1 distance from a, 2 from the line ab, 3 from the edges ac and bc.

  Called by `plane_pick`.
  In: mode (int) 1, 2 or 3; v (wp.vec3) a candidate vertex; a, b (wp.vec3) vertices picked before; nl (wp.vec3) the
  plane's normal; ab, ac, bc (wp.vec3) nl x (a - b), nl x (a - c), nl x (b - c); all hull frame.
  Out: float, the larger the better.
  """
  if mode == 1:
    return wp.length_sq(a - v)
  if mode == 2:
    return wp.abs(wp.dot(a - v, ab))
  return wp.abs(wp.dot(a - v, ac)) + wp.abs(wp.dot(b - v, bc))


@wp.func
def plane_band(g: Geom, hull: HullTablesDevice, mesh: int, pl: wp.vec3, nl: wp.vec3, thr: float, ia: int):
  """(band, count) -- the hull vertices penetrating deeper than thr, by a flood fill over hull edges
  from the deepest vertex ia (the vertices above a plane are a connected subgraph of a convex
  polytope's graph); count -1 when more than BAND_MAX.

  Called by `plane_kernel`.
  In: g (Geom): vertadr, vert (nvert,) wp.vec3; mesh (int) the hull id; pl, nl (wp.vec3) a point of the plane and its
  normal, hull frame; thr (float) the penetration threshold (m); ia (int) a hull-local vertex.
  Reads: hull.ve_adr and hull.ve_num (nvert,) int, hull.ve (2 * nedge,) int, hull.edgeadr (nhull,) int, hull.edge_v
  (nedge,) wp.vec2i. Out: (vec_band, int), the band's hull-local vertex ids and their count.
  """
  band = vec_band()  # vec_band: the band's vertices
  band[0] = ia
  nb = int(1)  # int: vertices in the band
  head = int(0)  # int: the next band vertex to expand
  while head < nb:
    v = band[head]  # int: the vertex expanded
    head += 1
    gv = g.vertadr + v  # int: its global vertex id
    for j in range(hull.ve_num[gv]):  # j: int, an edge at v
      ev = hull.edge_v[hull.edgeadr[mesh] + hull.ve[hull.ve_adr[gv] + j]]  # wp.vec2i: its vertices
      u = wp.where(ev[0] == v, ev[1], ev[0])  # int: the neighbour
      if wp.dot(pl - g.vert[g.vertadr + u], nl) > thr:
        seen = bool(False)  # bool: u is in the band already
        for k in range(nb):  # k: int, a band entry
          if band[k] == u:
            seen = True
        if not seen:
          if nb >= BAND_MAX:
            return band, -1
          band[nb] = u
          nb += 1
  return band, nb


@wp.func
def plane_pick(g: Geom, hull: HullTablesDevice, band: vec_band, nb: int, pl: wp.vec3, nl: wp.vec3, thr: float,
               mode: int, ia: int, a: wp.vec3, b: wp.vec3, c: wp.vec3) -> int:
  """The band vertex with the best plane_score (over the band, or a full scan of the hull's vertices
  when the band overflowed, nb < 0); ia if none qualifies.

  Called by `plane_kernel`.
  In: g (Geom): vertadr, vertnum, vert (nvert,) wp.vec3; band (vec_band) and nb (int) from `plane_band`; pl, nl
  (wp.vec3) the plane's point and normal, hull frame; thr (float, m); mode (int) as `plane_score`; ia (int) the
  deepest vertex; a, b, c (wp.vec3) vertices picked before. Reads: hull.vf_num (nvert,) int. Out: int, hull-local.
  """
  ab = wp.cross(nl, a - b)  # wp.vec3
  ac = wp.cross(nl, a - c)  # wp.vec3
  bc = wp.cross(nl, b - c)  # wp.vec3
  best = float(-1.0)  # float: the best score so far
  bi = ia  # int: its vertex
  if nb >= 0:
    for k in range(nb):  # k: int, a band entry, or a vertex in the scan
      s = plane_score(mode, g.vert[g.vertadr + band[k]], a, b, nl, ab, ac, bc)  # float
      if s > best:
        best = s
        bi = band[k]
  else:
    for k in range(g.vertnum):
      if hull.vf_num[g.vertadr + k] > 0:  # a hull vertex (MJWarp walks the hull graph)
        v = g.vert[g.vertadr + k]  # wp.vec3: hull frame
        if wp.dot(pl - v, nl) > thr:
          s = plane_score(mode, v, a, b, nl, ab, ac, bc)
          if s > best:
            best = s
            bi = k
  return bi


@wp.kernel(enable_backward=False, module="unique", launch_bounds=(64, 8))
def plane_kernel(mm: MeshModel, hull: HullTablesDevice, st: PipeState, gap: float):
  """Plane-mesh pairs: the contacts of MJWarp's plane_convex (same points, depths and positions).
  The deepest vertex is the exact support along -n (one cell lookup), and a pair whose deepest
  vertex is not below the plane ends there; otherwise b, c and d are picked among the vertices
  within PLANE_BAND of the deepest (and below the plane), gathered by a flood fill from it, and
  the distinct ones are written in MJWarp's order, each at the vertex moved half its distance
  onto the plane.

  Launch: dim=naconmax, block_dim=64, a thread per broadphase slot, by `pipeline.HybridNarrowphase.launch`.
  In: gap (float, m).
  Reads: st.ncollision (1,) int, st.collision_pair (naconmax,) wp.vec2i, st.collision_worldid (naconmax,) int;
  mm.geom_xpos (nworld, ngeom) wp.vec3 and mm.geom_xmat (nworld, ngeom) wp.mat33, the mesh's hull (`load_geom`);
  hull. Writes: the contacts (`write_contact`).
  """
  slot = wp.tid()  # int: the broadphase slot
  if slot >= st.ncollision[0] or slot >= st.collision_pair.shape[0]:
    return
  geoms = st.collision_pair[slot]  # wp.vec2i: the plane, the mesh
  if mm.geom_type[geoms[0]] != PLANE or mm.pair_ok[PLANE * NTYPE + mm.geom_type[geoms[1]]] == 0:
    return
  worldid = st.collision_worldid[slot]  # int
  h = hull_of(mm, geoms[1], worldid)  # int: the mesh's hull
  if h < 0:
    return
  prot = mm.geom_xmat[worldid, geoms[0]]  # wp.mat33: the plane's orientation
  n = wp.vec3(prot[0, 2], prot[1, 2], prot[2, 2])  # wp.vec3: the plane's normal, world frame
  p0 = mm.geom_xpos[worldid, geoms[0]]  # wp.vec3: the plane's position
  g = load_geom(mm, geoms[1], worldid)  # Geom: the mesh, world frame
  x, ia = cell_support(hull, h, g, -n)  # wp.vec3, int: the deepest vertex
  # MJWarp's frame: the hull's own (plane point and normal moved into it); every penetration below,
  # the deepest one included, is computed there as in plane_convex, so a vertex's test is the same
  # number wherever it is evaluated (a world-frame depth could differ by rounding at touching)
  nl = wp.transpose(g.rot) @ n  # wp.vec3: the normal, hull frame
  pl = wp.transpose(g.rot) @ (p0 - g.pos)  # wp.vec3: the plane's position, hull frame
  a = g.vert[g.vertadr + ia]  # wp.vec3: the deepest vertex, hull frame
  smax = wp.dot(pl - a, nl)  # float, penetration of the deepest vertex
  if -smax >= gap:
    return
  thr = wp.max(0.0, smax - PLANE_BAND)  # float: the band's penetration threshold (m)
  band, nb = plane_band(g, hull, h, pl, nl, thr, ia)  # vec_band, int
  ib = plane_pick(g, hull, band, nb, pl, nl, thr, 1, ia, a, a, a)  # int: vertex b
  b = g.vert[g.vertadr + ib]  # wp.vec3
  ic = plane_pick(g, hull, band, nb, pl, nl, thr, 2, ia, a, b, b)  # int: vertex c
  c = g.vert[g.vertadr + ic]  # wp.vec3
  i_d = plane_pick(g, hull, band, nb, pl, nl, thr, 3, ia, a, b, c)  # int: vertex d
  idx = wp.vec4i(ia, ib, ic, i_d)  # wp.vec4i: the picks
  ncon = int(0)  # int: the contacts written
  for i in range(3, -1, -1):  # i: int, a pick, the last first
    first = bool(True)  # bool: idx[i] is no earlier pick
    for j in range(i):  # j: int, an earlier pick
      if idx[j] == idx[i]:
        first = False
    if first:
      v = g.vert[g.vertadr + idx[i]]  # wp.vec3: hull frame
      dist = -wp.dot(pl - v, nl)  # float: negative in penetration (m)
      if dist < gap:
        write_contact(st, slot, ncon, geoms, worldid, g.rot @ v + g.pos - 0.5 * dist * n, n, dist)
        ncon += 1
