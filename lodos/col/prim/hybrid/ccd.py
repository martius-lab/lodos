"""MJWarp's mesh-mesh GJK and EPA on exact Gauss-map cell supports instead of hill climbs.

`gjk`, `_gjk_support`, `_epa_support`, `_polytope2`, `_polytope3`, `_epa` and `epa_phase` are
adapted from MuJoCo Warp 3.14.0's mujoco_warp/_src/collision_gjk.py and reduced to MESH-MESH
pairs without margins (MJWarp's `is_discrete` case). The algorithms are unchanged except for the
support query and one fix in GJK (below). MJWarp's `support` hill-climbs the hull graph from the
previous support (`geom.index`); `cell_support` looks up the cube-map cell of the direction in
the mesh frame and returns the best of the cell's precomputed candidates
(`meshdata.support_cells`), exact for every direction of the cell. All other helpers are
MJWarp's own.

The GJK fix: below |x_k| = 1e-4, MJWarp's `_gjk_support` replaces the search direction x_k by a
normal of the simplex (x_k is noisy there), but GJK's two tests -- the separating-plane exit
`x_k . w > 0` and the duality gap `x_k . (x_k - w) < 0` -- still dot x_k with the support w taken
along the tuned direction, which is not the minimizer of x_k . z over D. Both can then fire
falsely: the exit reports "separated" at tens of microns of overlap, the gap test stops GJK at
a false distance of microns on an overlapping pair (whose triangle EPA then cannot expand). Here
both tests use the direction the support was taken in, which makes them valid again; without
tuning they are MJWarp's tests exactly.
"""

from __future__ import annotations

import warp as wp


from lodos.col.prim.collision_core import Geom
from lodos.col.prim.collision_gjk import FLOAT_MAX
from lodos.col.prim.collision_gjk import MIN_DIST2
from lodos.col.prim.collision_gjk import MIN_DIST3
from lodos.col.prim.collision_gjk import MIN_EPATOL
from lodos.col.prim.collision_gjk import MINVAL
from lodos.col.prim.collision_gjk import MINVAL2
from lodos.col.prim.collision_gjk import GJKResult
from lodos.col.prim.collision_gjk import Polytope
from lodos.col.prim.collision_gjk import _add_edge
from lodos.col.prim.collision_gjk import _attach_face
from lodos.col.prim.collision_gjk import _delete_face
from lodos.col.prim.collision_gjk import _epa_witness
from lodos.col.prim.collision_gjk import _get_edge
from lodos.col.prim.collision_gjk import _get_face_verts
from lodos.col.prim.collision_gjk import _invalidate_face
from lodos.col.prim.collision_gjk import _is_face_deleted
from lodos.col.prim.collision_gjk import _is_invalid_face
from lodos.col.prim.collision_gjk import _linear_combine
from lodos.col.prim.collision_gjk import _polytope4
from lodos.col.prim.collision_gjk import _ray_triangle
from lodos.col.prim.collision_gjk import _replace_simplex3
from lodos.col.prim.collision_gjk import _rotmat
from lodos.col.prim.collision_gjk import _subdistance
from lodos.col.prim.collision_gjk import _test_tetra
from lodos.col.prim.collision_gjk import _tri_point_intersect
from lodos.col.prim.types import mat43
from lodos.mjcf.types import GeomType
from lodos.mjcf.types import OverflowType

from .meshdata import HullTablesDevice

MESH = wp.constant(int(GeomType.MESH.value))  # the mesh geom type, for `_epa_witness`


@wp.func
def cell_support(hull: HullTablesDevice, mesh: int, geom: Geom, d: wp.vec3):
  """Exact hull support along d: (point, mesh-local vertex id).

  The cube-map cell of the mesh-frame direction (as in meshdata.cube_cells) holds every vertex
  that is the support of some direction in the cell; the support is the best of them.

  Called by `_gjk_support`, `_epa_support`, `epa_warp.epa_warp_kernel` and 8 others in `ccd` and `kernels`.
  In: hull.ncell, hull.cell (nhull * 6 * ncell^2,) wp.vec2i, hull.cand_p (ncand,) wp.vec3 and hull.cand_v (ncand,)
  int; mesh: int, the hull id; geom.rot wp.mat33 and geom.pos wp.vec3, the hull's pose; d: wp.vec3, of any length.
  Out: the support point, wp.vec3 in the frame of geom.pos, and its hull-local vertex id, int.
  """
  l = wp.transpose(geom.rot) @ d  # wp.vec3: d in the hull frame
  ax = int(0)  # int: the axis of l's largest component, the cube face's
  if wp.abs(l[1]) > wp.abs(l[ax]):
    ax = 1
  if wp.abs(l[2]) > wp.abs(l[ax]):
    ax = 2
  m = wp.max(wp.abs(l[ax]), 1.0e-30)  # float: |l[ax]|, kept from zero
  n = hull.ncell  # int: cells per cube-face side
  half = 0.5 * float(n)  # float
  iu = wp.clamp(int((l[(ax + 1) % 3] / m + 1.0) * half), 0, n - 1)  # int: the cell's column
  iv = wp.clamp(int((l[(ax + 2) % 3] / m + 1.0) * half), 0, n - 1)  # int: the cell's row
  face = 2 * ax + wp.where(l[ax] < 0.0, 1, 0)  # int: the cube face, 0 to 5: +x, -x, +y, -y, +z, -z
  an = hull.cell[(mesh * 6 + face) * n * n + iv * n + iu]  # wp.vec2i: the cell's first candidate, candidate count
  k = an[0]  # int: the best candidate so far
  best = wp.dot(hull.cand_p[k], l)  # float: its support value along l
  for c in range(an[0] + 1, an[0] + an[1]):  # c: int, a candidate
    s = wp.dot(hull.cand_p[c], l)  # float
    if s > best:
      best = s
      k = c
  return geom.rot @ hull.cand_p[k] + geom.pos, hull.cand_v[k]


#######################################################################################
# GJK
#######################################################################################


@wp.func
def _closest64(n: int, simplex: mat43, lmbda: wp.vec4):
  """(barycentric coordinates, point) of the origin's projection onto the affine hull of the first
  n simplex points, solved in float64; float32's coordinates when the projection leaves them.

  GJK's closest point x_k = sum lambda_i w_i loses its small components to float32 cancellation
  when the simplex vertices are metres from the origin and x_k is sub-millimetre (a large geometry
  against a small one): its direction carries noise, the supports along it jump between the ends
  of a tied face, GJK sees no progress and stops on a false distance -- a penetrating pair
  reported apart. The projection in float64 (as kernels.simplex_direction) keeps x_k exact to its
  own float32 rounding.

  Called by `gjk`.
  In: n: int, 1 to 3; simplex: mat43, the simplex points in its rows; lmbda: wp.vec4, `_subdistance`'s
  coordinates. Out: wp.vec4, the coordinates; wp.vec3, the point x_k.
  """
  w0 = wp.vec3d(wp.float64(simplex[0, 0]), wp.float64(simplex[0, 1]), wp.float64(simplex[0, 2]))  # wp.vec3d
  w1 = wp.vec3d(wp.float64(simplex[1, 0]), wp.float64(simplex[1, 1]), wp.float64(simplex[1, 2]))  # wp.vec3d
  w2 = wp.vec3d(wp.float64(simplex[2, 0]), wp.float64(simplex[2, 1]), wp.float64(simplex[2, 2]))  # wp.vec3d
  zero = wp.float64(0.0)  # wp.float64
  one = wp.float64(1.0)  # wp.float64
  l0 = wp.float64(lmbda[0])  # wp.float64: w0's coordinate
  l1 = wp.float64(lmbda[1])  # wp.float64: w1's
  l2 = wp.float64(lmbda[2])  # wp.float64: w2's
  if n == 2:
    e = w1 - w0  # wp.vec3d: the segment
    ee = wp.dot(e, e)  # wp.float64: |e|^2
    if ee > zero:
      t = -wp.dot(w0, e) / ee  # wp.float64: the projection's parameter along e
      if t >= zero and t <= one:
        l0 = one - t
        l1 = t
  elif n == 3:
    e1 = w1 - w0  # wp.vec3d: the triangle's edges from w0
    e2 = w2 - w0  # wp.vec3d
    a11 = wp.dot(e1, e1)  # wp.float64: the Gram matrix of e1, e2
    a12 = wp.dot(e1, e2)  # wp.float64
    a22 = wp.dot(e2, e2)  # wp.float64
    b1 = -wp.dot(w0, e1)  # wp.float64: the normal equations' right-hand side
    b2 = -wp.dot(w0, e2)  # wp.float64
    det = a11 * a22 - a12 * a12  # wp.float64: the Gram determinant
    if det > zero:
      s = (b1 * a22 - b2 * a12) / det  # wp.float64: w1's coordinate (t: w2's)
      t = (a11 * b2 - a12 * b1) / det
      if s >= zero and t >= zero and s + t <= one:
        l0 = one - s - t
        l1 = s
        l2 = t
  x = w0  # wp.vec3d: the projection
  if n == 2:
    x = l0 * w0 + l1 * w1
  elif n == 3:
    x = l0 * w0 + l1 * w1 + l2 * w2
  lm = lmbda  # wp.vec4: the coordinates returned
  if n == 2 or n == 3:
    lm = wp.vec4(float(l0), float(l1), wp.where(n == 3, float(l2), 0.0), 0.0)
  return lm, wp.vec3(float(x[0]), float(x[1]), float(x[2]))


# Adapted from MuJoCo Warp 3.14.0's mujoco_warp/_src/collision_gjk.py, `_gjk_support`.
@wp.func
def _gjk_support(
  hull: HullTablesDevice, mesh1: int, mesh2: int, geom1: Geom, geom2: Geom, x_k: wp.vec3, x_norm: float,
  simplex: mat43, n: int,
):
  dir_neg = x_k / x_norm

  # tuning for discrete geoms when direction is noisy
  if x_norm < 1e-4:
    if n == 2:
      edge = simplex[1] - simplex[0]
      edge_norm2 = wp.dot(edge, edge)
      if edge_norm2 > MINVAL2:
        proj = wp.dot(dir_neg, edge) / edge_norm2
        dir_neg = dir_neg - proj * edge
        dir_norm = wp.length(dir_neg)
        if dir_norm > MINVAL:
          dir_neg = dir_neg / dir_norm
    elif n == 3:
      e1 = simplex[1] - simplex[0]
      e2 = simplex[2] - simplex[0]
      normal = wp.cross(e1, e2)
      normal_norm = wp.length(normal)
      if normal_norm > MINVAL:
        dir_neg = wp.sign(wp.dot(dir_neg, normal)) * normal / normal_norm

  p1, v1 = cell_support(hull, mesh1, geom1, -dir_neg)
  p2, v2 = cell_support(hull, mesh2, geom2, dir_neg)
  return p1, v1, p2, v2, dir_neg


# Adapted from MuJoCo Warp 3.14.0's mujoco_warp/_src/collision_gjk.py, `gjk`.
@wp.func
def gjk(
  gjk_iterations: int, hull: HullTablesDevice, mesh1: int, mesh2: int, geom1: Geom, geom2: Geom,
  x1_0: wp.vec3, x2_0: wp.vec3, cutoff: float,
) -> GJKResult:
  """Find distance within a tolerance between two meshes (MJWarp's gjk, discrete geoms)."""
  simplex = mat43()
  simplex1 = mat43()
  simplex2 = mat43()
  simplex_index1 = wp.vec4i()
  simplex_index2 = wp.vec4i()
  n = int(0)
  lmbda = wp.vec4(1.0, 0.0, 0.0, 0.0)  # barycentric coordinates

  # for discrete geoms GJK is guaranteed to converge in a finite number of iterations
  epsilon = 0.0
  min_norm = MINVAL

  # set initial guess
  x_k = x1_0 - x2_0
  xnorm = wp.sqrt(wp.dot(x_k, x_k))
  xnorm_prev = float(0.0)

  for _ in range(gjk_iterations):
    if xnorm < min_norm or wp.abs(xnorm_prev - xnorm) < MINVAL:
      break

    # compute the support point with direction tuning; x_d is x_k along the direction used
    p1, v1, p2, v2, dir_neg = _gjk_support(hull, mesh1, mesh2, geom1, geom2, x_k, xnorm, simplex, n)
    x_d = xnorm * dir_neg
    simplex1[n] = p1
    simplex_index1[n] = v1
    simplex2[n] = p2
    simplex_index2[n] = v2

    # compute the kth support point
    simplex[n] = simplex1[n] - simplex2[n]

    # stopping criteria using the Frank-Wolfe duality gap given by
    #  |f(x_k) - f(x_min)|^2 <= < grad f(x_k), (x_k - simplex[n]) >
    if wp.dot(x_d, x_k - simplex[n]) < epsilon:
      break

    # the lower bound on distance between the two geoms is (lower / x_norm)
    # if lower > 0, then the geoms are separated
    lower = wp.dot(x_d, simplex[n])
    if cutoff == 0.0:
      if lower > 0.0:
        result = GJKResult()
        result.separated = True
        result.dim = 0
        result.dist = FLOAT_MAX
        result.index1 = -1
        result.index2 = -1
        return result
    elif cutoff < FLOAT_MAX:
      if lower > 0.0 and lower >= cutoff * xnorm:
        result = GJKResult()
        result.separated = True
        result.dim = 0
        result.dist = FLOAT_MAX
        result.index1 = -1
        result.index2 = -1
        return result

    # run the distance subalgorithm to compute the barycentric coordinates
    # of the closest point to the origin in the simplex
    lmbda = _subdistance(n + 1, simplex)

    # remove vertices from the simplex no longer needed
    n = int(0)
    for i in range(4):
      if lmbda[i] == 0.0:
        continue

      simplex[n] = simplex[i]
      simplex1[n] = simplex1[i]
      simplex2[n] = simplex2[i]
      simplex_index1[n] = simplex_index1[i]
      simplex_index2[n] = simplex_index2[i]
      lmbda[n] = lmbda[i]
      n += int(1)

    # SHOULD NOT OCCUR
    if n < 1:
      break

    # get the next iteration of x_k, projected in float64 (see _closest64)
    if n < 4:
      lmbda, x_k = _closest64(n, simplex, lmbda)
    else:
      x_k = _linear_combine(n, lmbda, simplex)
    xnorm_prev = xnorm
    xnorm = wp.sqrt(wp.dot(x_k, x_k))

    # we have a tetrahedron containing the origin so return early
    if n == 4:
      break

  result = GJKResult()
  result.separated = False

  # compute the approximate witness points
  # if n is zero, then there was an immediate return meaning the initial points
  # are the witness points
  result.x1 = wp.where(n == 0, x1_0, _linear_combine(n, lmbda, simplex1))
  result.x2 = wp.where(n == 0, x2_0, _linear_combine(n, lmbda, simplex2))

  if xnorm > 0.0:
    dir = x_k / xnorm
    p1, v1 = cell_support(hull, mesh1, geom1, -dir)
    p2, v2 = cell_support(hull, mesh2, geom2, dir)
    result.separated = wp.dot(x_k, p1 - p2) > 0.0

  # if 3-simplex and not separated, then the origin is contained in the simplex
  result.dist = wp.where(n == 4 and not result.separated, 0.0, xnorm)

  result.dim = n
  result.simplex1 = simplex1
  result.simplex2 = simplex2
  result.simplex_index1 = simplex_index1
  result.simplex_index2 = simplex_index2
  result.simplex = simplex
  result.index1 = -1
  result.index2 = -1
  return result


#######################################################################################
# EPA
#######################################################################################


# Adapted from MuJoCo Warp 3.14.0's mujoco_warp/_src/collision_gjk.py, `_epa_support`.
@wp.func
def _epa_support(
  pt: Polytope, idx: int, hull: HullTablesDevice, mesh1: int, mesh2: int, geom1: Geom, geom2: Geom, dir: wp.vec3
):
  p1, v1 = cell_support(hull, mesh1, geom1, dir)
  pt.vert[2 * idx] = p1
  pt.vert_index[2 * idx] = v1

  p2, v2 = cell_support(hull, mesh2, geom2, -dir)
  pt.vert[2 * idx + 1] = p2
  pt.vert_index[2 * idx + 1] = v2


# Adapted from MuJoCo Warp 3.14.0's mujoco_warp/_src/collision_gjk.py, `_polytope2`.
@wp.func
def _polytope2(
  pt: Polytope, simplex: mat43, simplex1: mat43, simplex2: mat43, simplex_index1: wp.vec4i,
  simplex_index2: wp.vec4i, hull: HullTablesDevice, mesh1: int, mesh2: int, geom1: Geom, geom2: Geom,
):
  """Create polytope for EPA given a 1-simplex from GJK."""
  diff = simplex[1] - simplex[0]

  # set the polytope center
  pt.center = 0.5 * (simplex[0] + simplex[1])

  # find component with smallest magnitude (so cross product is largest)
  value = FLOAT_MAX
  index = 0
  for i in range(3):
    if wp.abs(diff[i]) < value:
      value = wp.abs(diff[i])
      index = i

  # cross product with best coordinate axis
  e = wp.vec3(0.0, 0.0, 0.0)
  e[index] = 1.0
  d1 = wp.cross(e, diff)

  # rotate around the line segment to get three more points spaced 120 degrees apart
  R = _rotmat(diff)
  d2 = R @ d1
  d3 = R @ d2

  # save vertices and get indices for each one
  pt.vert[0] = simplex1[0]
  pt.vert[1] = simplex2[0]
  pt.vert[2] = simplex1[1]
  pt.vert[3] = simplex2[1]

  pt.vert_index[0] = simplex_index1[0]
  pt.vert_index[1] = simplex_index2[0]
  pt.vert_index[2] = simplex_index1[1]
  pt.vert_index[3] = simplex_index2[1]

  _epa_support(pt, 2, hull, mesh1, mesh2, geom1, geom2, d1 / wp.norm_l2(d1))
  _epa_support(pt, 3, hull, mesh1, mesh2, geom1, geom2, d2 / wp.norm_l2(d2))
  _epa_support(pt, 4, hull, mesh1, mesh2, geom1, geom2, d3 / wp.norm_l2(d3))

  # build hexahedron
  if _attach_face(pt, 0, 0, 2, 3) < MIN_DIST2:
    pt.status = -1
    return pt, _replace_simplex3(pt, 0, 2, 3)

  if _attach_face(pt, 1, 0, 4, 2) < MIN_DIST2:
    pt.status = -1
    return pt, _replace_simplex3(pt, 0, 4, 2)

  if _attach_face(pt, 2, 0, 3, 4) < MIN_DIST2:
    pt.status = -1
    return pt, _replace_simplex3(pt, 0, 3, 4)

  if _attach_face(pt, 3, 1, 3, 2) < MIN_DIST2:
    pt.status = -1
    return pt, _replace_simplex3(pt, 1, 3, 2)

  if _attach_face(pt, 4, 1, 2, 4) < MIN_DIST2:
    pt.status = -1
    return pt, _replace_simplex3(pt, 1, 2, 4)

  if _attach_face(pt, 5, 1, 4, 3) < MIN_DIST2:
    pt.status = -1
    return pt, _replace_simplex3(pt, 1, 4, 3)

  # check hexahedron is convex
  v2 = pt.vert[4] - pt.vert[5]
  v3 = pt.vert[6] - pt.vert[7]
  v4 = pt.vert[8] - pt.vert[9]
  if not _ray_triangle(simplex[0], simplex[1], v2, v3, v4):
    pt.status = 1
    return pt, GJKResult()

  # set polytope counts
  pt.nvert = 5
  pt.nface = 6
  pt.status = 0
  return pt, GJKResult()


# Adapted from MuJoCo Warp 3.14.0's mujoco_warp/_src/collision_gjk.py, `_polytope3`.
@wp.func
def _polytope3(
  pt: Polytope, dist: float, simplex: mat43, simplex1: mat43, simplex2: mat43, simplex_index1: wp.vec4i,
  simplex_index2: wp.vec4i, hull: HullTablesDevice, mesh1: int, mesh2: int, geom1: Geom, geom2: Geom,
) -> Polytope:
  """Create polytope for EPA given a 2-simplex from GJK."""
  # set the polytope center
  pt.center = (simplex[0] + simplex[1] + simplex[2]) * wp.static(1.0 / 3.0)

  # get normals in both directions
  n = wp.cross(simplex[1] - simplex[0], simplex[2] - simplex[0])
  norm = wp.norm_l2(n)
  if norm < MINVAL:
    pt.status = 2
    return pt
  n = n / norm

  pt.vert[0] = simplex1[0]
  pt.vert[1] = simplex2[0]
  pt.vert[2] = simplex1[1]
  pt.vert[3] = simplex2[1]
  pt.vert[4] = simplex1[2]
  pt.vert[5] = simplex2[2]

  pt.vert_index[0] = simplex_index1[0]
  pt.vert_index[1] = simplex_index2[0]
  pt.vert_index[2] = simplex_index1[1]
  pt.vert_index[3] = simplex_index2[1]
  pt.vert_index[4] = simplex_index1[2]
  pt.vert_index[5] = simplex_index2[2]

  _epa_support(pt, 3, hull, mesh1, mesh2, geom1, geom2, -n)
  _epa_support(pt, 4, hull, mesh1, mesh2, geom1, geom2, n)

  v1 = simplex[0]
  v2 = simplex[1]
  v3 = simplex[2]
  v4 = pt.vert[6] - pt.vert[7]
  v5 = pt.vert[8] - pt.vert[9]

  # check that v4 is not contained in the 2-simplex
  if _tri_point_intersect(v1, v2, v3, v4):
    pt.status = 3
    return pt

  # check that v5 is not contained in the 2-simplex
  if _tri_point_intersect(v1, v2, v3, v5):
    pt.status = 4
    return pt

  # if origin does not lie on simplex then we need to check that the hexahedron contains the
  # origin
  if dist > 1e-5 and not _test_tetra(v1, v2, v3, v4) and not _test_tetra(v1, v2, v3, v5):
    pt.status = 5
    return pt

  # create hexahedron for EPA
  if _attach_face(pt, 0, 4, 0, 1) < MIN_DIST3:
    pt.status = 6
    return pt
  if _attach_face(pt, 1, 4, 2, 0) < MIN_DIST3:
    pt.status = 7
    return pt
  if _attach_face(pt, 2, 4, 1, 2) < MIN_DIST3:
    pt.status = 8
    return pt
  if _attach_face(pt, 3, 3, 1, 0) < MIN_DIST3:
    pt.status = 9
    return pt
  if _attach_face(pt, 4, 3, 0, 2) < MIN_DIST3:
    pt.status = 10
    return pt
  if _attach_face(pt, 5, 3, 2, 1) < MIN_DIST3:
    pt.status = 11
    return pt

  # set polytope counts
  pt.nvert = 5
  pt.nface = 6
  pt.status = 0
  return pt


# Adapted from MuJoCo Warp 3.14.0's mujoco_warp/_src/collision_gjk.py, `_epa`.
@wp.func
def _epa(
  epa_iterations: int, pt: Polytope, hull: HullTablesDevice, mesh1: int, mesh2: int, geom1: Geom, geom2: Geom,
  worldid: int, overflow_out: wp.array(dtype=int),
):
  """Recover penetration data from two meshes in contact given an initial polytope."""
  upper = FLOAT_MAX
  upper2 = FLOAT_MAX
  idx = int(-1)
  pidx = int(-1)
  epsilon = MIN_EPATOL
  nvalid = pt.nface  # number of potential faces for expanding the polytope

  # the face vertices are encoded in 10-bits that index the vertex array,
  # so iterations must be cap to limit the number of generated vertices
  # (one new vertex per iteration)
  epa_iterations = wp.min(epa_iterations, 1000)
  for _ in range(epa_iterations):
    pidx = idx
    idx = int(-1)
    lower2 = float(FLOAT_MAX)

    # find the face closest to the origin (lower bound for penetration depth)
    for i in range(pt.nface):
      if not _is_invalid_face(pt.face[i]) and pt.face_norm2[i] < lower2:
        idx = i
        lower2 = pt.face_norm2[i]

    # face not valid, return previous face
    if lower2 > upper2 or idx < 0:
      idx = pidx
      break

    # check if lower bound is 0
    if lower2 <= 0.0:
      break

    # compute support point w from the closest face's normal
    lower = wp.sqrt(lower2)
    wi = pt.nvert
    face_pr = pt.face_pr[idx]
    _epa_support(pt, wi, hull, mesh1, mesh2, geom1, geom2, face_pr / lower)
    w = pt.vert[2 * wi] - pt.vert[2 * wi + 1]
    pt.nvert += 1

    # upper bound for kth iteration (dot product before normalizing for better precision)
    upper_k = wp.dot(face_pr, w) / lower
    if upper_k < upper:
      upper = upper_k
      upper2 = upper * upper

    if upper - lower < epsilon:
      break

    # check if vertex wi is a repeated support point
    found_repeated = bool(False)
    for i in range(pt.nvert - 1):
      if pt.vert_index[2 * i] == pt.vert_index[2 * wi] and pt.vert_index[2 * i + 1] == pt.vert_index[2 * wi + 1]:
        found_repeated = True
        break
    if found_repeated:
      break

    nvalid -= 1
    pt.face[idx] = _delete_face(pt.face[idx])
    face = _get_face_verts(pt.face[idx])
    pt.nhorizon = _add_edge(pt, face[0], face[1])
    pt.nhorizon = _add_edge(pt, face[1], face[2])
    pt.nhorizon = _add_edge(pt, face[2], face[0])
    if pt.nhorizon == -1:
      wp.atomic_or(overflow_out, worldid, OverflowType.EPA_HORIZON)
      idx = -1
      break

    # compute horizon for w
    for i in range(pt.nface):
      if _is_face_deleted(pt.face[i]):
        continue

      if wp.dot(pt.face_pr[i], w) - pt.face_norm2[i] > 1e-10:
        nvalid = wp.where(_is_invalid_face(pt.face[i]), nvalid, nvalid - 1)
        pt.face[i] = _delete_face(pt.face[i])
        face = _get_face_verts(pt.face[i])
        pt.nhorizon = _add_edge(pt, face[0], face[1])
        pt.nhorizon = _add_edge(pt, face[1], face[2])
        pt.nhorizon = _add_edge(pt, face[2], face[0])
        if pt.nhorizon == -1:
          wp.atomic_or(overflow_out, worldid, OverflowType.EPA_HORIZON)
          idx = -1
          break

    # insert w as new vertex and attach faces along the horizon
    for i in range(pt.nhorizon):
      edge = _get_edge(pt.horizon[i])
      dist2 = _attach_face(pt, pt.nface, wi, edge[0], edge[1])
      if dist2 == 0.0:
        idx = -1
        break

      pt.nface += 1

      if dist2 >= lower2 and dist2 <= upper2:
        nvalid += 1
      else:
        pt.face[pt.nface - 1] = _invalidate_face(pt.face[pt.nface - 1])

    # no face candidates left
    if nvalid == 0 or idx == -1:
      break

    # clear horizon
    pt.nhorizon = 0

  # return from valid face
  if idx > -1:
    x1, x2, dist = _epa_witness(pt, geom1, geom2, MESH, MESH, idx)
    return dist, x1, x2, idx
  return 0.0, wp.vec3(), wp.vec3(), -1


@wp.func
def epa_init(pt: Polytope, result: GJKResult, hull: HullTablesDevice, mesh1: int, mesh2: int, geom1: Geom,
             geom2: Geom):
  """The initial polytope from GJK's simplex (pt.status 0: ready for EPA). Returns (pt, result).

  Called by `epa_phase` and `epa_warp.epa_warp_kernel`.
  In: pt: Polytope, on its scratch arrays, empty; result: GJKResult, GJK's (dim, dist and the simplex); hull,
  mesh1, mesh2, geom1, geom2: as `_epa_support`. Out: pt, its vertices, faces, centre, counts and status set;
  result, its simplex replaced by a triangle (dim 3) when the segment's or the tetrahedron's polytope had a face
  nearly through the origin.
  """
  if result.dim == 2:
    pt, new_result = _polytope2(  # new_result: GJKResult, the triangle to fall back on when pt.status is -1
      pt, result.simplex, result.simplex1, result.simplex2, result.simplex_index1, result.simplex_index2, hull,
      mesh1, mesh2, geom1, geom2,
    )
    if pt.status == -1:
      result.simplex = new_result.simplex
      result.simplex1 = new_result.simplex1
      result.simplex2 = new_result.simplex2
      result.simplex_index1 = new_result.simplex_index1
      result.simplex_index2 = new_result.simplex_index2
      result.dim = 3
  elif result.dim == 4:
    pt, new_result = _polytope4(
      pt, result.simplex, result.simplex1, result.simplex2, result.simplex_index1, result.simplex_index2
    )
    if pt.status == -1:
      result.simplex = new_result.simplex
      result.simplex1 = new_result.simplex1
      result.simplex2 = new_result.simplex2
      result.simplex_index1 = new_result.simplex_index1
      result.simplex_index2 = new_result.simplex_index2
      result.dim = 3

  if result.dim == 3:
    pt = _polytope3(
      pt, result.dist, result.simplex, result.simplex1, result.simplex2, result.simplex_index1,
      result.simplex_index2, hull, mesh1, mesh2, geom1, geom2,
    )
  return pt, result


# Adapted from MuJoCo Warp 3.14.0's mujoco_warp/_src/collision_gjk.py, `epa_phase`.
@wp.func
def epa_phase(
  epa_iterations: int, result: GJKResult, hull: HullTablesDevice, mesh1: int, mesh2: int, geom1: Geom,
  geom2: Geom, vert: wp.array(dtype=wp.vec3), vert_index: wp.array(dtype=int), face: wp.array(dtype=int),
  face_pr: wp.array(dtype=wp.vec3), face_norm2: wp.array(dtype=float), horizon: wp.array(dtype=int),
  worldid: int, overflow_out: wp.array(dtype=int),
):
  """Run EPA given a GJK result. Returns (dist, ncontact, x1, x2, index of the minimizing face)."""
  pt = Polytope()
  pt.nface = 0
  pt.nvert = 0
  pt.nhorizon = 0
  pt.vert = vert
  pt.vert_index = vert_index
  pt.face = face
  pt.face_pr = face_pr
  pt.face_norm2 = face_norm2
  pt.horizon = horizon

  pt, result = epa_init(pt, result, hull, mesh1, mesh2, geom1, geom2)
  if pt.status:
    return result.dist, 1, result.x1, result.x2, -1

  dist, x1, x2, idx = _epa(epa_iterations, pt, hull, mesh1, mesh2, geom1, geom2, worldid, overflow_out)
  if idx == -1:
    return FLOAT_MAX, 0, wp.vec3(), wp.vec3(), -1
  return dist, 1, x1, x2, idx
