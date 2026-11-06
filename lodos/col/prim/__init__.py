"""
MJWarp's collision pipeline, with the exact hull narrowphase.

  driver   geometry poses from Lodos's kinematics, then MJWarp's broadphase and
           narrowphases, with the hybrid for box-box, box-mesh, mesh-mesh and
           plane-mesh; contacts land in MJWarp's `d.contact`
  hybrid   the exact narrowphase: Gauss-map cell supports, the feature cache and
           ball certificate, GJK/EPA with float64 projections, the quadrangle
           manifold, box hulls and the floor kernel
  collision_driver, collision_core, collision_primitive, collision_convex, collision_gjk
           MJWarp's broadphases and narrowphases (Apache-2.0), adapted from MuJoCo Warp
           3.14.0's mujoco_warp/_src; each module's header names its source files
  types    the types and constants these modules and the hybrid use, adapted
           likewise from MuJoCo Warp's types.py

This half collides the MJCF's own geometries -- polytopes and MuJoCo's primitives --
with each other. Lodos's surfaces collide with each other through `lodos.col.sdf`;
the two never meet: a polytope that supplements a Lodos surface for
edge-edge contacts collides with other polytopes, not with surfaces.
"""

from . import driver, hybrid
