"""Float64 ground truth for the penetration depth of two convex hulls.

depth = min over the facets of D = P - Q of h_D(axis), where the facet axes are the hull
triangle normals of P, the negated triangle normals of Q, and the cross products of edge
pairs that pass Gregorius' Gauss-map arc test (exactly the edge-edge facets of D). Supports
are full scans over the hull vertices in float64. Used to adjudicate disagreements between
the hybrid narrowphase and MJWarp's GJK/EPA on specific pairs.
"""

from __future__ import annotations

import numpy as np

class Hull:
  """A hull in float64 for the exact depth: its vertices, unit triangle normals, and the edges between triangles
  that are not coplanar, with the two triangles at each.

  Built by `hulls_from_tables`.
  """

  def __init__(self, V, T):
    """V: (n, 3) hull-frame vertices, T: (nf, 3) outward triangles (meshdata.build's `tris`).

    Called by `hulls_from_tables`.
    Out: self.V (n, 3) float64, self.hv (nhv,) int, self.tn (nf, 3) float64, self.E and self.EF (ne, 2) int.
    """
    self.V = np.asarray(V, dtype=np.float64)  # (n, 3) float64
    self.hv = np.unique(T)  # (nhv,) int: the vertices on the hull
    n = np.cross(self.V[T[:, 1]] - self.V[T[:, 0]], self.V[T[:, 2]] - self.V[T[:, 0]])  # (nf, 3) float64
    self.tn = n / np.linalg.norm(n, axis=1, keepdims=True)  # (nf, 3) float64: unit triangle normals
    e2t = {}  # {(a, b): [triangle ids]} per undirected edge, a < b
    for t, (a, b, c) in enumerate(T):  # t: int, a triangle; a, b, c: int, its vertices
      for x, y in ((a, b), (b, c), (c, a)):  # x, y: int, one of its edges
        e2t.setdefault((min(x, y), max(x, y)), []).append(t)
    E, EF = [], []  # lists of (int, int): the kept edges, the two triangles at each
    for (a, b), ts in e2t.items():  # ts: list of int, the edge's triangles
      if len(ts) == 2 and self.tn[ts[0]] @ self.tn[ts[1]] < 1.0 - 1e-12:
        E.append((a, b))
        EF.append(ts)
    self.E, self.EF = np.array(E), np.array(EF)  # (ne, 2) int: the edges; (ne, 2) int: their triangles


def exact_depth(PV, Ptn, PE, PEF, QV, Qtn, QE, QEF):
  """(depth, axis P -> Q) of world-frame hulls (vertices, triangle normals, edges, edge faces).

  Not called in the repository.
  In: PV (nP, 3) float64, Ptn (nfP, 3) float64, PE (EP, 2) int, PEF (EP, 2) int: a `Hull`'s V, tn, E and EF in the
  world frame; QV, Qtn, QE, QEF the same of Q.
  Out: (float, (3,) float64): the depth (m), negative when separated, and the unit axis.
  """
  def hD(U):  # (k, 3) unit axes P -> Q
    """h_D(u) = max over P of u.x - min over Q of u.y, for each row u of U: (k,) float64.

    Called by `exact_depth`.
    """
    return (PV @ U.T).max(0) - (QV @ U.T).min(0)

  axes = [Ptn, -Qtn]  # list of (k, 3) float64: the candidate axes
  a, b = Ptn[PEF[:, 0]], Ptn[PEF[:, 1]]  # (EP, 3), (EP, 3): the normals of the two triangles at P's edges
  c, d = -Qtn[QEF[:, 0]], -Qtn[QEF[:, 1]]  # (EQ, 3), (EQ, 3): the negated normals at Q's edges
  bxa, dxc = np.cross(b, a), np.cross(d, c)  # (EP, 3), (EQ, 3)
  cba, dba = c @ bxa.T, d @ bxa.T  # (EQ, EP)
  adc, bdc = (a @ dxc.T).T, (b @ dxc.T).T  # (EQ, EP), (EQ, EP)
  ok = (cba * dba < 0) & (adc * bdc < 0) & (cba * bdc > 0)  # (EQ, EP) bool: the edges' Gauss-map arcs cross
  jq, ip = np.nonzero(ok)  # (k,) int, (k,) int: Q's and P's edge of each crossing pair
  if len(ip):
    eP = PV[PE[ip, 1]] - PV[PE[ip, 0]]  # (k, 3) float64: P's edge vectors
    eQ = QV[QE[jq, 1]] - QV[QE[jq, 0]]  # (k, 3) float64: Q's
    u = np.cross(eP, eQ)  # (k, 3) float64: the edge-pair axes
    ln = np.linalg.norm(u, axis=1)  # (k,) float64
    keep = ln > 1e-12 * np.linalg.norm(eP, axis=1) * np.linalg.norm(eQ, axis=1)  # (k,) bool: edges not parallel
    u = u[keep] / ln[keep, None]
    flip = np.einsum("ij,ij->i", u, PV[PE[ip[keep], 0]] - PV.mean(0)) < 0  # (nkeep,) bool: axes pointing into P
    u[flip] *= -1
    axes.append(u)
  U = np.concatenate(axes)  # (nfP + nfQ + nkeep, 3) float64: every candidate axis
  h = hD(U)  # (nfP + nfQ + nkeep,) float64
  k = int(np.argmin(h))  # int: the minimizing axis
  return float(h[k]), U[k]


def hulls_from_tables(tables):
  """{hull id: Hull} for every hull of meshdata.build's tables (meshes and boxes).

  Not called in the repository.
  In: tables (meshdata.HullTables): vert (nvert, 3) float64, vertadr and vertnum (nhull,) int32, tris, per hull
  (nf, 3) int. Out: dict of int -> Hull.
  """
  return {i: Hull(tables.vert[a:a + n], T) for i, (a, n, T) in enumerate(zip(tables.vertadr, tables.vertnum, tables.tris))}


def pair_truth(mjm, d_numpy, world, g1, g2, hulls, tables):
  """Exact depth for broadphase pair (world, g1, g2), positions shifted to geom1's origin. A plane
  geom1 (the floor) gives the deepest hull vertex below it, with the plane normal as the axis.

  Not called in the repository.
  In: mjm (mujoco.MjModel); d_numpy (geom_xpos (nworld, ngeom, 3), geom_xmat (nworld, ngeom, 3, 3)), numpy arrays;
  world, g1, g2 (int); hulls ({hull id: Hull}, `hulls_from_tables`); tables (meshdata.HullTables).
  Out: (float, (3,) float64): the depth (m) and the unit axis geom1 -> geom2.
  """
  xpos, xmat = d_numpy  # (nworld, ngeom, 3), (nworld, ngeom, 3, 3)
  o = xpos[world, g1].astype(np.float64)  # (3,) float64: geom1's position, the pair frame's origin
  # Hull: geom g's, its mesh's or its box hull
  hull_of = lambda g: hulls[int(mjm.geom_dataid[g]) if tables.geom_hull[g] == -1 else int(tables.geom_hull[g])]
  if mjm.geom_type[g1] == 0:  # mjGEOM_PLANE
    n = xmat[world, g1].astype(np.float64)[:, 2]  # (3,) float64: the plane's normal
    h = hull_of(g2)  # Hull
    # (nhv, 3) float64: geom2's hull vertices, pair frame
    V = h.V[h.hv] @ xmat[world, g2].astype(np.float64).T + (xpos[world, g2].astype(np.float64) - o)
    return float((-V @ n).max()), n
  hp, hq = hull_of(g1), hull_of(g2)  # Hull, Hull: P and Q
  # edges index the full vertex array: world (pair-frame) vertices for all ids
  RP, RQ = xmat[world, g1].astype(np.float64), xmat[world, g2].astype(np.float64)  # (3, 3) float64 each
  PVa = hp.V @ RP.T  # (nP, 3) float64: every vertex of P, pair frame
  QVa = hq.V @ RQ.T + (xpos[world, g2].astype(np.float64) - o)  # (nQ, 3) float64: of Q
  return _exact_full(PVa, hp, QVa, hq, RP, RQ)


def _exact_full(PVa, hp, QVa, hq, RP, RQ):
  """The exact depth of hulls hp (P) and hq (Q) posed in the pair frame: `exact_depth` from every vertex of each
  (PVa, QVa, which the edges index) and their rotations (RP, RQ, which turn the triangle normals).

  Called by `pair_truth`.
  In: PVa (nP, 3) float64, QVa (nQ, 3) float64; hp, hq (Hull); RP, RQ (3, 3) float64.
  Out: (float, (3,) float64): the depth (m), negative when separated, and the unit axis P -> Q.
  """
  PV, QV = PVa[hp.hv], QVa[hq.hv]  # (nhvP, 3), (nhvQ, 3) float64: the hull vertices
  Ptn, Qtn = hp.tn @ RP.T, hq.tn @ RQ.T  # (nfP, 3), (nfQ, 3) float64: unit triangle normals, pair frame

  def hD(U):
    """h_D(u) = max over P of u.x - min over Q of u.y, for each row u of U (k, 3): (k,) float64.

    Called by `_exact_full`.
    """
    return (PV @ U.T).max(0) - (QV @ U.T).min(0)

  axes = [Ptn, -Qtn]  # list of (k, 3) float64: the candidate axes
  a, b = Ptn[hp.EF[:, 0]], Ptn[hp.EF[:, 1]]  # (EP, 3), (EP, 3): the normals of the two triangles at P's edges
  c, d = -Qtn[hq.EF[:, 0]], -Qtn[hq.EF[:, 1]]  # (EQ, 3), (EQ, 3): the negated normals at Q's edges
  bxa, dxc = np.cross(b, a), np.cross(d, c)  # (EP, 3), (EQ, 3)
  cba, dba = c @ bxa.T, d @ bxa.T  # (EQ, EP), (EQ, EP)
  adc, bdc = (a @ dxc.T).T, (b @ dxc.T).T  # (EQ, EP), (EQ, EP)
  ok = (cba * dba < 0) & (adc * bdc < 0) & (cba * bdc > 0)  # (EQ, EP) bool: the edges' Gauss-map arcs cross
  jq, ip = np.nonzero(ok)  # (k,) int, (k,) int: Q's and P's edge of each crossing pair
  if len(ip):
    eP = PVa[hp.E[ip, 1]] - PVa[hp.E[ip, 0]]  # (k, 3) float64: P's edge vectors
    eQ = QVa[hq.E[jq, 1]] - QVa[hq.E[jq, 0]]  # (k, 3) float64: Q's
    u = np.cross(eP, eQ)  # (k, 3) float64: the edge-pair axes
    ln = np.linalg.norm(u, axis=1)  # (k,) float64
    keep = ln > 1e-12 * np.linalg.norm(eP, axis=1) * np.linalg.norm(eQ, axis=1)  # (k,) bool: edges not parallel
    u = u[keep] / ln[keep, None]
    flip = np.einsum("ij,ij->i", u, PVa[hp.E[ip[keep], 0]] - PV.mean(0)) < 0  # (nkeep,) bool: axes pointing into P
    u[flip] *= -1
    axes.append(u)
  U = np.concatenate(axes)  # (nfP + nfQ + nkeep, 3) float64: every candidate axis
  h = hD(U)  # (nfP + nfQ + nkeep,) float64
  k = int(np.argmin(h))  # int: the minimizing axis
  return float(h[k]), U[k]
