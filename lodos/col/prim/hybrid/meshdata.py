"""Host-side convex-hull topology for the hybrid narrowphase.

Hulls 0 .. nmesh - 1 are MuJoCo's meshes (hull id = mesh id, so a mesh geom's hull is its
`geom_dataid`), built from the hull MuJoCo already computes for every collision mesh -- the
triangles stored in `mesh_graph` (the same hull the hill-climbing `support` walks) -- and NOT
from `mesh_poly*`: on CoACD pieces MuJoCo's merged polygons need not be faces of the hull
(a merged polygon can lie far off its plane, and V - E + F need not be 2), so they cannot
serve as reference faces or exact face axes. Hulls nmesh .. are boxes, one per distinct size
among the box geoms given to `build` (8 vertices, two triangles per face, so the same merge
yields the 6 quad faces). Every hull's vertices live in one array `vert`, the meshes' at
their `mesh_vertadr` offsets and the boxes' after them.

Per hull:

  faces      hull triangles merged where adjacent normals agree to `coplanar_cos`;
             loops wound counter-clockwise about the outward normal
  face_d     the TRUE support offset of each face normal, max over all vertices -- a
             merged face's own vertices sit up to its non-planarity below it, and the
             face query is only exact when the plane supports the whole hull
  edges      hull edges between faces of different normal, with both adjacent faces
  vert->face and vert->edge incidence, for incident-face and edge-candidate searches
  face_q     per face corner, the other three corners of the largest quadrangle of the face
             through that corner (see `largest_quads`): the manifold clips quad against quad
  cells      exact support candidates per cube-map cell of directions (see `support_cells`):
             the support along any direction in a cell is the best of the cell's few
             candidate vertices, so no support query walks the hull graph

Vertex indices are local to a hull (vertex k of hull i is vert[vertadr[i] + k]); face and
edge indices are local to the hull and offset by `faceadr`/`edgeadr` into the packed arrays;
the incidence lists are indexed by GLOBAL vertex id.
"""

from __future__ import annotations

import dataclasses
import hashlib

import mujoco

import numpy as np
import warp as wp


@wp.struct
class HullTablesDevice:
  """The hull tables the kernels read, on the device: `HullTables`' arrays, positions and normals in float32.

  Built by `HullTables.to_device`; read by `ccd.cell_support` and the kernels of `kernels`.
  """

  faceadr: wp.array(dtype=int)  # (nhull,)
  facenum: wp.array(dtype=int)  # (nhull,)
  face_n: wp.array(dtype=wp.vec3)  # (nface,) outward unit normal, mesh frame
  face_d: wp.array(dtype=float)  # (nface,) support offset over the whole hull
  face_vadr: wp.array(dtype=int)  # (nface,)
  face_vnum: wp.array(dtype=int)  # (nface,)
  face_v: wp.array(dtype=int)  # (nloop,) local vertex ids, CCW
  face_q: wp.array(dtype=wp.vec3i)  # (nloop,) per corner: the other corners of its largest quad, CCW
  edgeadr: wp.array(dtype=int)  # (nhull,)
  edgenum: wp.array(dtype=int)  # (nhull,)
  edge_v: wp.array(dtype=wp.vec2i)  # (nedge,) local vertex ids
  edge_f: wp.array(dtype=wp.vec2i)  # (nedge,) local face ids
  vf_adr: wp.array(dtype=int)  # (nvert,) per global vertex
  vf_num: wp.array(dtype=int)  # (nvert,) the faces at each global vertex
  vf: wp.array(dtype=int)  # (nloop,) local face ids
  vf_pos: wp.array(dtype=int)  # (nloop,) position of the vertex in that face's loop
  ve_adr: wp.array(dtype=int)  # (nvert,) per global vertex
  ve_num: wp.array(dtype=int)  # (nvert,) the edges at each global vertex
  ve: wp.array(dtype=int)  # (2 * nedge,) local edge ids
  ncell: int  # cube-map cells per face side
  cell: wp.array(dtype=wp.vec2i)  # (nhull * 6 * ncell^2,) candidate adr, num
  cand_p: wp.array(dtype=wp.vec3)  # (ncand,) candidate position, mesh frame
  cand_v: wp.array(dtype=int)  # (ncand,) candidate local vertex id


@dataclasses.dataclass
class HullTables:
  """The hull tables on the host (the module docstring), as numpy arrays, and the build's report.

  Built by `_build`, for `build`; `pipeline.HybridNarrowphase.__init__` copies them to the device (`to_device`).
  """

  faceadr: np.ndarray  # (nhull,) int64: each hull's first face
  facenum: np.ndarray  # (nhull,) int64: its face count
  face_n: np.ndarray  # (nface, 3) float64: outward unit normals, hull frame
  face_d: np.ndarray  # (nface,) float64: support offsets over the whole hull
  face_vadr: np.ndarray  # (nface,) int64: each face's first entry in face_v
  face_vnum: np.ndarray  # (nface,) int64: its loop length
  face_v: np.ndarray  # (nloop,) int64: hull-local vertex ids, CCW
  face_q: np.ndarray  # (nloop, 3) int64: per corner, the other corners of its largest quad, CCW
  edgeadr: np.ndarray  # (nhull,) int64: each hull's first edge
  edgenum: np.ndarray  # (nhull,) int64: its edge count
  edge_v: np.ndarray  # (nedge, 2) int64: hull-local vertex ids
  edge_f: np.ndarray  # (nedge, 2) int64: hull-local face ids
  vf_adr: np.ndarray  # (nvert,) int64: per global vertex, its first entry in vf
  vf_num: np.ndarray  # (nvert,) int64: its face count
  vf: np.ndarray  # (nloop,) int64: hull-local face ids
  vf_pos: np.ndarray  # (nloop,) int64: the vertex's position in that face's loop
  ve_adr: np.ndarray  # (nvert,) int64: per global vertex, its first entry in ve
  ve_num: np.ndarray  # (nvert,) int64: its edge count
  ve: np.ndarray  # (2 * nedge,) int64: hull-local edge ids
  ncell: int  # cube-map cells per face side
  cell: np.ndarray  # (nhull * 6 * ncell^2, 2) int64: per cell, its first candidate and their count
  cand_p: np.ndarray  # (ncand, 3) float64: candidate positions, hull frame
  cand_v: np.ndarray  # (ncand,) int64: candidate hull-local vertex ids
  nfacemax: int  # largest face loop
  report: dict  # the build's counts and checks: nhull, nbox, nface, nedge, worst non-planarity, ...
  nmesh: int  # hulls 0 .. nmesh - 1 are the meshes, the rest boxes
  vert: np.ndarray  # (nvert, 3) every hull's vertices, hull frame
  vertadr: np.ndarray  # (nhull,)
  vertnum: np.ndarray  # (nhull,) int32
  tris: list  # per hull, (nf, 3) outward CCW triangles, hull-local vertex ids
  geom_hull: np.ndarray  # (ngeom,) box hull id, -1 for mesh geoms (hull = geom_dataid), -2 for none

  def to_device(self) -> HullTablesDevice:
    """The device copy of the tables the kernels read: float32 positions and normals, int32 ids.

    Called by `pipeline.HybridNarrowphase.__init__`.
    Out: HullTablesDevice, each field the array of the same name, of the same length.
    """
    def arr(x, dt):
      """`x` as a contiguous Warp array of dtype `dt`.

      Called by `to_device`.
      In: x: np.ndarray, (n,) or, for a vector `dt` of length k, (n, k); dt: a Warp dtype. Out: (n,) wp.array.
      """
      return wp.array(np.ascontiguousarray(x), dtype=dt)

    s = HullTablesDevice()  # HullTablesDevice
    s.faceadr, s.facenum = arr(self.faceadr, int), arr(self.facenum, int)
    s.face_n, s.face_d = arr(self.face_n.astype(np.float32), wp.vec3), arr(self.face_d.astype(np.float32), float)
    s.face_vadr, s.face_vnum, s.face_v = arr(self.face_vadr, int), arr(self.face_vnum, int), arr(self.face_v, int)
    s.face_q = arr(self.face_q.astype(np.int32), wp.vec3i)
    s.edgeadr, s.edgenum = arr(self.edgeadr, int), arr(self.edgenum, int)
    s.edge_v, s.edge_f = arr(self.edge_v.astype(np.int32), wp.vec2i), arr(self.edge_f.astype(np.int32), wp.vec2i)
    s.vf_adr, s.vf_num, s.vf = arr(self.vf_adr, int), arr(self.vf_num, int), arr(self.vf, int)
    s.vf_pos = arr(self.vf_pos, int)
    s.ve_adr, s.ve_num, s.ve = arr(self.ve_adr, int), arr(self.ve_num, int), arr(self.ve, int)
    s.ncell = self.ncell
    s.cell = arr(self.cell.astype(np.int32), wp.vec2i)
    s.cand_p, s.cand_v = arr(self.cand_p.astype(np.float32), wp.vec3), arr(self.cand_v, int)
    return s


def hull_triangles(mjm: mujoco.MjModel, i: int) -> np.ndarray:
  """(nf, 3) outward CCW hull triangles of mesh i, mesh-local vertex ids, from mesh_graph.

  Called by `_build`.
  In: mjm: mujoco.MjModel, its mesh_graph and mesh_graphadr; i: int, the mesh. Out: (nf, 3) int64.
  """
  g = mjm.mesh_graph[mjm.mesh_graphadr[i]:]  # (nmeshgraph - adr,) int32: mesh_graph from mesh i's graph on
  nv, nf = int(g[0]), int(g[1])  # int, int: the hull's vertex and face counts
  start = 2 + 3 * nv + 3 * nf  # int, after numvert, numface, vert_edgeadr, vert_globalid, edge_localid
  return g[start: start + 3 * nf].reshape(nf, 3).astype(int)


def box_hull(size) -> tuple[np.ndarray, np.ndarray]:
  """(V (8, 3), T (12, 3)) of a box of half sizes `size`: vertex 4 ix + 2 iy + iz at
  ((2 ix - 1) sx, (2 iy - 1) sy, (2 iz - 1) sz), two outward CCW triangles per face.

  Called by `_build`.
  In: size: 3 floats (m). Out: V (8, 3) float64, T (12, 3) int64.
  """
  s = np.asarray(size, dtype=np.float64)  # (3,) float64
  V = np.array([[(2 * ix - 1) * s[0], (2 * iy - 1) * s[1], (2 * iz - 1) * s[2]]  # (8, 3) float64
                for ix in (0, 1) for iy in (0, 1) for iz in (0, 1)])
  quads = ((4, 6, 7, 5), (0, 1, 3, 2), (2, 3, 7, 6), (0, 4, 5, 1), (1, 5, 7, 3), (0, 2, 6, 4))  # +x -x +y -y +z -z
  return V, np.array([t for a, b, c, e in quads for t in ((a, b, c), (a, c, e))])


def _merge(V, T, coplanar_cos):
  """Union-find over edge-adjacent triangles with agreeing normals -> group id per triangle.

  Called by `_build`.
  In: V: (nv, 3) float64, the hull's vertices; T: (nf, 3) int, its outward CCW triangles; coplanar_cos: float,
  the cosine above which adjacent normals agree. Out: group (nf,) int64, each triangle's merged face, numbered
  from 0; n (nf, 3) float64, the unit triangle normals; area (nf,) float64, twice the triangles' areas.
  """
  n = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])  # (nf, 3) float64: the normals, unit below
  area = np.linalg.norm(n, axis=1)  # (nf,) float64: twice the triangles' areas
  n = n / area[:, None]
  parent = np.arange(len(T))  # (nf,) int64: the union-find forest

  def find(a):
    """The union-find root of triangle `a`, halving the path on the way.

    Called by `_merge`.
    In: a: int, a triangle. Out: int, its root.
    """
    while parent[a] != a:
      parent[a] = parent[parent[a]]
      a = parent[a]
    return a

  e2t = {}  # {(a, b) with a < b: [triangles]}: the triangles at each undirected edge
  for t, (a, b, c) in enumerate(T):  # t: int, a triangle; a, b, c: its vertices
    for x, y in ((a, b), (b, c), (c, a)):  # x, y: an edge's vertices
      e2t.setdefault((min(x, y), max(x, y)), []).append(t)
  for ts in e2t.values():  # ts: list of int, the triangles at an edge
    if len(ts) == 2 and n[ts[0]] @ n[ts[1]] > coplanar_cos:
      ra, rb = find(ts[0]), find(ts[1])  # int, int: their roots
      if ra != rb:
        parent[ra] = rb
  roots = np.array([find(t) for t in range(len(T))])  # (nf,) int64: each triangle's root
  _, group = np.unique(roots, return_inverse=True)  # group: (nf,) int64, each triangle's merged face
  return group, n, area


def largest_quads(P, n):
  """Per corner k of the CCW loop P (m, 3) about n: loop positions of the other three corners
  (CCW after k) of the largest-area quadrangle with corners among the loop through corner k.

  A triangle gives its other two corners and the last one again (a zero-length edge). For a
  fixed corner v and opposite corner c, the best remaining corners are the loop vertices
  farthest from the diagonal v-c on either side, so each corner costs O(m^2).

  Called by `_build`.
  In: P: (m, 3) float64, the loop's vertices; n: (3,) float64, the face's unit normal.
  Out: a list of m sequences of 3 ints, loop positions.
  """
  m = len(P)  # int: the loop's length
  if m <= 4:
    q = list(range(m)) + [m - 1] * (4 - m)  # list of 4 ints: the loop, its last corner repeated to 4
    return [(q[k:] + q[:k])[1:] for k in range(m)]
  D = P[None, :, :] - P[:, None, :]  # (m, m, 3) float64: D[a, b] = P[b] - P[a]
  S = np.cross(D[:, :, None, :], D[:, None, :, :]) @ n  # (m, m, m) float64, S[a, b, c]: twice the area of (a, b, c)
  out = []  # list of m 3-tuples of int: the result
  for v in range(m):  # v: int, the corner
    best, quad = -1.0, None  # float, 3-tuple of int: the largest twice-area so far, its other corners
    for dvc in range(2, m - 1):  # dvc: int, the opposite corner's offset from v
      c = (v + dvc) % m  # int: the opposite corner
      arc1 = (v + 1 + np.arange(dvc - 1)) % m  # (dvc - 1,) int64: strictly between v and c
      arc2 = (c + 1 + np.arange(m - dvc - 1)) % m  # (m - dvc - 1,) int64: strictly between c and v
      s1, s2 = S[v, arc1, c], S[v, c, arc2]  # (dvc - 1,), (m - dvc - 1,) float64: twice the areas on either side
      if s1.max() + s2.max() > best:
        best, quad = s1.max() + s2.max(), (int(arc1[np.argmax(s1)]), c, int(arc2[np.argmax(s2)]))
    out.append(quad)
  return out


def cube_cells(ncell: int):
  """Cells of an ncell x ncell cube map of directions: (centres (k, 3), planes (k, 4, 3)).

  Cell id = face * ncell^2 + iv * ncell + iu, faces +x, -x, +y, -y, +z, -z; on face (ax, s)
  the coordinates are u = d[(ax + 1) % 3] / |d[ax]| and v = d[(ax + 2) % 3] / |d[ax]|, and
  the cell is the cone {d : m . d >= 0} of its four bounding planes (unit inward normals m):
  lines on a cube face are great arcs on the sphere.

  Called by `support_cells`.
  In: ncell: int. Out: centres (k, 3) float64, unit, and planes (k, 4, 3) float64, the unit inward normals, of the
  k = 6 ncell^2 cells.
  """
  e = np.eye(3)  # (3, 3) float64: the axes
  cen, planes = [], []  # lists of (3,) and (4, 3) float64, a cell each
  for face in range(6):  # face: int, the cube face
    ax, s = face // 2, 1.0 - 2.0 * (face % 2)  # int, float: its axis and sign
    i, j = (ax + 1) % 3, (ax + 2) % 3  # int, int: the u and v axes
    for iv in range(ncell):  # iv: int, the cell's row
      for iu in range(ncell):  # iu: int, its column
        u0, u1 = -1.0 + 2.0 * iu / ncell, -1.0 + 2.0 * (iu + 1) / ncell  # float, float: the cell's u range
        v0, v1 = -1.0 + 2.0 * iv / ncell, -1.0 + 2.0 * (iv + 1) / ncell  # float, float: its v range
        c = s * e[ax] + 0.5 * (u0 + u1) * e[i] + 0.5 * (v0 + v1) * e[j]  # (3,) float64: the centre, on the cube
        cen.append(c / np.linalg.norm(c))
        # (4, 3) float64: the bounding planes' normals, inward
        P = np.array([e[i] - u0 * s * e[ax], u1 * s * e[ax] - e[i], e[j] - v0 * s * e[ax], v1 * s * e[ax] - e[j]])
        planes.append(P / np.linalg.norm(P, axis=1, keepdims=True))
  return np.array(cen), np.array(planes)


def support_cells(V, T, ncell, delta=1e-4, nsample=16, seed=0):
  """Per cube-map cell, the hull vertices that are the support of some direction in the cell.

  The support along d is a vertex whose normal cone contains d. A cone meets a cell iff the
  cell lies inside it (its vertex is the support of the cell centre) or one of its boundary
  arcs meets the cell, and the boundary arcs are the Gauss images of the hull edges (between
  the normals of the two triangles at the edge). So a cell's candidates are the support of its
  centre plus both endpoints of every edge whose arc meets the cell; the arc (1 - t) n1 + t n2
  meets the cell iff four linear intervals in t intersect. The planes are inflated by `delta`
  (about delta rad) so that a direction rounded into a neighbouring cell is still covered.
  Every cell is then checked against full scans at its corners, centre and `nsample` random
  directions, and any missed maximizer is added (counted in the returned `added`).

  Called by `_build`.
  In: V: (nv, 3) float64, the hull's vertices; T: (nf, 3) int, its outward CCW triangles; ncell: int; delta:
  float (rad); nsample: int; seed: int, of the random directions. Out: a list of k = 6 ncell^2 sorted (ncand_c,)
  int64 arrays, each cell's hull-local candidate vertex ids, in `cube_cells`' cell order; added: int.
  """
  rng = np.random.default_rng(seed)  # np.random.Generator
  hv = np.unique(T)  # (nhv,) int64: the hull's vertices
  tn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])  # (nf, 3) float64: the normals, unit below
  tn /= np.linalg.norm(tn, axis=1, keepdims=True)
  e2t = {}  # {(a, b) with a < b: [triangles]}: the triangles at each undirected edge
  for t, (a, b, c) in enumerate(T):  # t: int, a triangle; a, b, c: its vertices
    for x, y in ((a, b), (b, c), (c, a)):  # x, y: an edge's vertices
      e2t.setdefault((min(x, y), max(x, y)), []).append(t)
  E = np.array([e for e, ts in e2t.items() if len(ts) == 2])  # (ne, 2) int64: the edges between two triangles
  n1 = tn[[ts[0] for ts in e2t.values() if len(ts) == 2]]  # (ne, 3) float64: each edge's first triangle's normal
  n2 = tn[[ts[1] for ts in e2t.values() if len(ts) == 2]]  # (ne, 3) float64: its second's
  cen, planes = cube_cells(ncell)  # (k, 3), (k, 4, 3) float64: see cube_cells
  centre_sup = hv[np.argmax(V[hv] @ cen.T, axis=0)]  # (k,) int64: the support of each cell's centre
  lists = []  # k sets of int: each cell's candidates
  for c0 in range(0, len(cen), 128):  # c0: int, the first cell of a chunk of up to 128
    P = planes[c0:c0 + 128]  # (kc, 4, 3) float64: the chunk's planes
    a = np.einsum("ckx,ax->cak", P, n1) + delta  # (kc, ne, 4) float64: n1's inflated distance to each plane
    b = np.einsum("ckx,ax->cak", P, n2) + delta  # (kc, ne, 4) float64: n2's
    cross = a * b < 0.0  # (kc, ne, 4) bool: the edge's arc crosses the plane
    t = np.where(cross, a / np.where(cross, a - b, 1.0), 0.0)  # (kc, ne, 4) float64: where, along the arc
    lo = np.where(cross & (a < 0.0), t, 0.0)  # (kc, ne, 4) float64: where the arc enters the plane's inner side
    hi = np.where(cross & (a >= 0.0), t, 1.0)  # (kc, ne, 4) float64: where it leaves it
    lo = np.where((a < 0.0) & (b < 0.0), 2.0, lo)
    meet = lo.max(2) <= hi.min(2)  # (kc, ne) bool: the edge's arc meets the cell
    for k in range(len(P)):  # k: int, a cell of the chunk
      lists.append(set(E[meet[k]].ravel().tolist()) | {int(centre_sup[c0 + k])})
  added = 0  # int: the maximizers the check added
  e = np.eye(3)  # (3, 3) float64: the axes
  for face in range(6):  # face: int, the cube face
    ax, s = face // 2, 1.0 - 2.0 * (face % 2)  # int, float: its axis and sign
    i, j = (ax + 1) % 3, (ax + 2) % 3  # int, int: the u and v axes
    for iv in range(ncell):  # iv: int, the cell's row
      for iu in range(ncell):  # iu: int, its column
        c = face * ncell * ncell + iv * ncell + iu  # int: the cell
        u0, v0 = -1.0 + 2.0 * iu / ncell, -1.0 + 2.0 * iv / ncell  # float, float: the cell's lower corner
        h = 2.0 / ncell  # float: its side
        # (5 + nsample,) float64: the check directions' u, at the corners, the centre and random
        u = np.r_[u0, u0 + h, u0, u0 + h, u0 + 0.5 * h, rng.uniform(u0, u0 + h, nsample)]
        v = np.r_[v0, v0, v0 + h, v0 + h, v0 + 0.5 * h, rng.uniform(v0, v0 + h, nsample)]  # (5 + nsample,) float64
        D = s * e[ax][None] + u[:, None] * e[i][None] + v[:, None] * e[j][None]  # (5 + nsample, 3) float64
        L = np.array(sorted(lists[c]))  # (ncand_c,) int64: the cell's candidates
        full = V[hv] @ D.T  # (nhv, 5 + nsample) float64: every hull vertex's support value
        miss = full[np.searchsorted(hv, L)].max(0) < full.max(0)  # (5 + nsample,) bool, same rows: no rounding mismatch
        for k in np.nonzero(miss)[0]:
          lists[c].add(int(hv[np.argmax(full[:, k])]))
          added += 1
  return [np.array(sorted(L)) for L in lists], added


# Tables already built, by what they are built from. Building them is host-side numpy over
# every hull -- slow for a scene of many hulls -- and a scene is built again for every solver
# it is switched to, every capacity the live viewer grows and every configuration a tuning
# tool times. Callers only read the tables.
_CACHE = {}


def _key(mjm, coplanar_cos, ncell, box_geoms, meshes):
  """`build`'s `_CACHE` key: a SHA-256 digest of the meshes' arrays, the geom types and the boxes' float32 sizes,
  with the other arguments.

  Called by `build`.
  In: as `build`. Out: a tuple (str, float, int, tuple of int, tuple of int or None).
  """
  h = hashlib.sha256()  # hashlib's SHA-256 object
  # a: np.ndarray, an array the tables are built from
  for a in (mjm.mesh_vert, mjm.mesh_vertadr, mjm.mesh_vertnum, mjm.mesh_graph, mjm.mesh_graphadr, mjm.geom_type,
            np.float32(mjm.geom_size[list(box_geoms)]) if len(box_geoms) else np.zeros(0)):
    a = np.ascontiguousarray(a)
    h.update(str(a.shape).encode() + a.tobytes())
  return (h.hexdigest(), float(coplanar_cos), int(ncell), tuple(int(g) for g in box_geoms),
          None if meshes is None else tuple(sorted(int(i) for i in meshes)))


def build(mjm: mujoco.MjModel, coplanar_cos: float = 1.0 - 1e-7, ncell: int = 16, box_geoms=(),
          meshes=None) -> HullTables:
  """Tables of every mesh of `mjm`, then of one box hull per distinct size among `box_geoms`
  (sizes rounded to float32, as the device holds them).

  `meshes`, when given, is the set of mesh ids anything can pair: every other mesh keeps its
  hull id (hull id = mesh id) but gets a placeholder -- a 1 mm box no kernel ever reads --
  instead of its own tables. A mesh nothing collides with may have no hull at all: MuJoCo
  builds `mesh_graph` only for meshes of colliding geoms.

  Built once per process for each distinct input (`_CACHE`); the result is shared, and read only.

  Called by `pipeline.HybridNarrowphase.__init__`.
  In: mjm: mujoco.MjModel; coplanar_cos: float, the cosine above which adjacent triangles' normals merge them into
  one face; ncell: int, cube-map cells per face side; box_geoms: a sequence of box geom ids; meshes: a set of mesh
  ids, or None for every mesh. Out: HullTables.
  """
  key = _key(mjm, coplanar_cos, ncell, box_geoms, meshes)  # tuple: the cache key
  if key not in _CACHE:
    _CACHE[key] = _build(mjm, coplanar_cos, ncell, box_geoms, meshes)
  return _CACHE[key]


def _build(mjm, coplanar_cos, ncell, box_geoms, meshes):
  """The tables `build` returns, built anew: per hull, its merged faces, the edges between them, the vertices'
  incident faces and edges, the faces' largest quads and the cells' support candidates, packed.

  Called by `build`.
  In: as `build`. Out: HullTables. Raises RuntimeError when a hull triangle is not outward CCW or a merged face's
  boundary is not a single cycle.
  """
  # {name: list}: the packed per-hull, per-face and per-edge fields
  out = {k: [] for k in ("faceadr", "facenum", "face_n", "face_d", "face_vadr", "face_vnum", "face_v",
                         "edgeadr", "edgenum", "edge_v", "edge_f")}
  nvm = int(mjm.mesh_vertadr[-1] + mjm.mesh_vertnum[-1]) if mjm.nmesh else 0  # int: MuJoCo's mesh vertices
  hulls = []  # list of (vertadr, V, T), a hull each: V (nv, 3) float64, T (nf, 3) int
  own = []  # list of int: hulls whose vertices live in the hybrid's array rather than in mesh_vert
  nvt = nvm  # int: the vertices so far, of all hulls
  for i in range(mjm.nmesh):  # i: int, a mesh
    if meshes is not None and i not in meshes:
      V, T = box_hull((1e-3, 1e-3, 1e-3))  # (8, 3) float64, (12, 3) int64: the placeholder
      own.append(len(hulls))
      hulls.append((nvt, V, T))
      nvt += len(V)
      continue
    va, vn = int(mjm.mesh_vertadr[i]), int(mjm.mesh_vertnum[i])  # int, int: the mesh's first vertex, its count
    hulls.append((va, mjm.mesh_vert[va:va + vn].astype(np.float64), hull_triangles(mjm, i)))
  # (ngeom,) int32: -1 for mesh geoms, -2 for the others, box hull ids set below
  geom_hull = np.where(mjm.geom_type == int(mujoco.mjtGeom.mjGEOM_MESH), -1, -2).astype(np.int32)
  box_of_size = {}  # {size: hull id}: the box hulls
  for g in box_geoms:  # g: int, a box geom
    size = tuple(np.float32(mjm.geom_size[g]).astype(np.float64).tolist())  # 3 floats: the half sizes
    if size not in box_of_size:
      V, T = box_hull(size)
      box_of_size[size] = len(hulls)
      own.append(len(hulls))
      hulls.append((nvt, V, T))
      nvt += len(V)
    geom_hull[g] = box_of_size[size]
  vf_lists = [[] for _ in range(nvt)]  # nvt lists of int: the faces at each global vertex
  vf_pos_lists = [[] for _ in range(nvt)]  # nvt lists of int: the vertex's position in each of those faces' loops
  face_q = []  # list of (3,) int64: per face corner, the other corners of its largest quad
  ve_lists = [[] for _ in range(nvt)]  # nvt lists of int: the edges at each global vertex
  worst_planar, worst_support, nfacemax, euler = 0.0, 0.0, 0, set()  # float, float, int, set: for the report
  cell, cand_p, cand_v, cells_added = [], [], [], 0  # lists of cells (adr, num), candidate positions and ids; int
  for i, (va, V, T) in enumerate(hulls):
    tri_n = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])  # (nf, 3) float64: unnormalized normals
    if np.any(np.einsum("ij,ij->i", tri_n, V[T].mean(1) - V[np.unique(T)].mean(0)) <= 0.0):
      raise RuntimeError(f"hull {i}: a triangle is not outward CCW")
    group, tn, area = _merge(V, T, coplanar_cos)  # (nf,) int64, (nf, 3) float64, (nf,) float64: see _merge
    nfg = int(group.max()) + 1  # int: the hull's merged faces
    out["faceadr"].append(len(out["face_n"]))
    out["facenum"].append(nfg)
    hull_v = np.unique(T)  # (nhv,) int64: the hull's vertices
    for f in range(nfg):  # f: int, a merged face
      ts = np.where(group == f)[0]  # (nt,) int64: its triangles
      nrm = (tn[ts] * area[ts, None]).sum(0)  # (3,) float64: its area-weighted normal, unit below
      nrm /= np.linalg.norm(nrm)
      nxt = {}  # {a: b}, boundary of the group: directed edges whose reverse is not in the group
      directed = set()  # set of (a, b): the directed edges of the face's triangles
      for t in ts:  # t: int64, a triangle of the face
        a, b, c = T[t]  # int64: its vertices
        directed |= {(a, b), (b, c), (c, a)}
      for (a, b) in directed:
        if (b, a) not in directed:
          nxt[a] = b
      start = next(iter(nxt))  # int64: a boundary vertex
      loop = [start]  # list of int64: the boundary's vertices, in order (an array below)
      while nxt[loop[-1]] != start:
        loop.append(nxt[loop[-1]])
        if len(loop) > len(nxt):
          raise RuntimeError(f"hull {i} face {f}: boundary is not a single cycle")
      loop = np.array(loop)
      d = float((V[hull_v] @ nrm).max())  # float: the normal's support offset over the whole hull
      worst_planar = max(worst_planar, d - float((V[loop] @ nrm).min()))
      out["face_n"].append(nrm)
      out["face_d"].append(d)
      out["face_vadr"].append(len(out["face_v"]))
      out["face_vnum"].append(len(loop))
      out["face_v"].extend(loop.tolist())
      nfacemax = max(nfacemax, len(loop))
      face_q.extend([loop[list(q)] for q in largest_quads(V[loop], nrm)])
      for k, v in enumerate(loop):  # k: int, a loop position; v: int64, its vertex
        vf_lists[va + v].append(f)
        vf_pos_lists[va + v].append(k)
    # hull edges between different faces
    e2g = {}  # {(a, b) with a < b: set of int}: the merged faces at each undirected edge
    for t, (a, b, c) in enumerate(T):
      for x, y in ((a, b), (b, c), (c, a)):  # x, y: an edge's vertices
        e2g.setdefault((min(x, y), max(x, y)), set()).add(int(group[t]))
    out["edgeadr"].append(len(out["edge_v"]))
    ne = 0  # int: the hull's edges so far
    for (a, b), gs in sorted(e2g.items()):  # gs: set of int, the faces at edge (a, b)
      if len(gs) != 2:
        continue  # a seam inside a merged face
      ga, gb = sorted(gs)  # int, int: its two faces
      ve_lists[va + a].append(ne)
      ve_lists[va + b].append(ne)
      out["edge_v"].append((a, b))
      out["edge_f"].append((ga, gb))
      ne += 1
    out["edgenum"].append(ne)
    euler.add(len(hull_v) - ne + nfg)
    worst_support = max(worst_support, float(((V[hull_v] @ np.array(out["face_n"][-nfg:]).T).max(0)
                                              - np.array(out["face_d"][-nfg:])).max()))
    lists, added = support_cells(V, T, ncell)  # a (ncand_c,) int64 array per cell; int
    cells_added += added
    for L in lists:  # L: (ncand_c,) int64, a cell's candidates
      cell.append((len(cand_v), len(L)))
      cand_v.extend(L.tolist())
      cand_p.extend(V[L].tolist())
  vf_adr = np.cumsum([0] + [len(x) for x in vf_lists])[:-1]  # (nvt,) int64
  ve_adr = np.cumsum([0] + [len(x) for x in ve_lists])[:-1]  # (nvt,) int64
  cell = np.array(cell).reshape(-1, 2)
  vert = np.zeros((nvt, 3))  # (nvt, 3) float64: every hull's vertices
  vert[:nvm] = mjm.mesh_vert[:nvm]
  for h in own:  # h: int, a hull of the hybrid's own
    va, V, _ = hulls[h]  # _: T, unused
    vert[va:va + len(V)] = V
  # dict: the build's counts and checks
  report = dict(nhull=len(hulls), nbox=len(box_of_size), nface=len(out["face_n"]), nedge=len(out["edge_v"]),
                nfacemax=nfacemax, worst_nonplanarity=worst_planar, euler=sorted(euler), support_check=worst_support,
                cell_candidates_mean=float(cell[:, 1].mean()) if len(cell) else 0.0,
                cell_candidates_max=int(cell[:, 1].max()) if len(cell) else 0, cell_candidates_added_by_check=cells_added)
  return HullTables(
    faceadr=np.array(out["faceadr"]), facenum=np.array(out["facenum"]), face_n=np.array(out["face_n"]),
    face_d=np.array(out["face_d"]), face_vadr=np.array(out["face_vadr"]), face_vnum=np.array(out["face_vnum"]),
    face_v=np.array(out["face_v"]), face_q=np.array(face_q), edgeadr=np.array(out["edgeadr"]),
    edgenum=np.array(out["edgenum"]),
    edge_v=np.array(out["edge_v"]), edge_f=np.array(out["edge_f"]),
    vf_adr=vf_adr, vf_num=np.array([len(x) for x in vf_lists]), vf=np.array(sum(vf_lists, [])),
    vf_pos=np.array(sum(vf_pos_lists, [])),
    ve_adr=ve_adr, ve_num=np.array([len(x) for x in ve_lists]), ve=np.array(sum(ve_lists, [])),
    ncell=ncell, cell=cell, cand_p=np.array(cand_p), cand_v=np.array(cand_v),
    nfacemax=nfacemax, report=report, nmesh=mjm.nmesh, vert=vert,
    vertadr=np.array([h[0] for h in hulls], dtype=np.int32), vertnum=np.array([len(h[1]) for h in hulls], dtype=np.int32),
    tris=[h[2] for h in hulls], geom_hull=geom_hull,
  )
