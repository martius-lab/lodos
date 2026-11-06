"""
What a live scene looks like: both pipelines' geometry as one `VisScene`.

`viewer.draw.make_scene` draws Lodos surfaces only. A fused scene also has MJCF
geoms colliding through MJWarp's pipeline -- primitives, and polytopes -- and geoms
that collide with nothing, such as the mesh a surface's body takes its inertia from.
`LiveScene` lists all of them, one drawn geometry each, in four kinds:

    KIND_LODOS     a Lodos surface, drawn from its baked vertices in its body's frame
    KIND_PRIM      an MJCF geom in the prim pipeline's collision (`lodos.col.prim`)
    KIND_PLANE     an MJCF plane, whichever pipelines it is in
    KIND_VISUAL    an MJCF geom in no collision (`contype = conaffinity = 0` on
                   `make_sim`'s model, which is how a surface's own geom ends up)

The last are hidden at first, and so are the MJCF geoms of a body that also has a
Lodos surface: in the mixed scenes those are the 34 convex pieces that supplement an
armadillo's surface, which would otherwise be drawn into it.

Every drawn geometry is a body plus a fixed pose in that body's frame, so a frame's
poses are the bodies' (`geom_xform`): a Lodos surface sits at its body's pose, an
MJCF geom at `geom_pos`/`geom_quat` from it -- which for a mesh includes the offset
MuJoCo moved its vertices by, since `mesh_vert` is stored in the geom's frame.

Primitives are triangulated once per distinct size and instanced, and so are meshes,
per MJCF mesh: twenty armadillos of 34 pieces are 34 prototypes, not 680.
"""

from __future__ import annotations

import dataclasses

import mujoco
import numpy as np

from lodos.col.sdf import types as sdf_types
from viewer.draw import PALETTE, VisScene, plane_mesh, sphere_mesh

KIND_LODOS, KIND_PRIM, KIND_PLANE, KIND_VISUAL = 0, 1, 2, 3
KIND_NAMES = ("Lodos surfaces", "MJCF geoms", "planes", "non-colliding geoms")

# MuJoCo's default geom colour. A geom that kept it is coloured by its body here; one
# the MJCF coloured keeps its colour, and its alpha becomes its transparency.
MJ_DEFAULT_RGBA = (0.5, 0.5, 0.5, 1.0)
WORLD_COLOR = (0.62, 0.64, 0.68)
# MuJoCo's group for collision geometry, which its viewer hides: models tint what they
# put there (the menagerie's hulls are green at a quarter opacity). A colliding geom in
# it is drawn opaque in its body's colour instead, as the body's surface would be.
COLLISION_GROUP = 3
# MuJoCo's checkered ground, `builtin="checker" rgb1=".2 .3 .4"`: the shader darkens every
# other square (`viewer.draw.enable_transparency`).
GROUND_RGB = (0.2, 0.3, 0.4)

# Meshes with fewer faces than this are drawn flat-shaded, each face its own three
# vertices: a convex piece or a box with smoothed normals reads as a blob.
FLAT_FACES = 4000


@dataclasses.dataclass
class LiveScene(VisScene):
    """A `VisScene` over both pipelines' geometry, with what the live viewer adds."""

    geom_body: np.ndarray = None  # (nG,) body each drawn geometry moves with
    geom_id: np.ndarray = None  # (nG,) its MJCF geom, -1 for a Lodos surface
    geom_local: np.ndarray = None  # (nG, 7) pose in that body's frame, [pos, quat (x, y, z, w)]
    geom_kind: np.ndarray = None  # (nG,) KIND_*
    geom_alpha: np.ndarray = None  # (nG,) the MJCF's alpha, 1 where it gave none
    geom_show0: np.ndarray = None  # (nG,) whether it is drawn at first
    body_name: list = None  # (nbody,)
    body_color: np.ndarray = None  # (nbody, 3)
    lodos_bodies: np.ndarray = None  # bodies that carry a Lodos surface
    checker: float = 0.0  # the ground's square (m), `checker_size` of the largest body


def checker_size(span):
    """
    The ground's square under a scene whose largest body spans `span` (m): half of it,
    rounded down to 1, 2 or 5 x 10^k. The largest body rather than the median one: a
    hand's links are 3 cm, and squares under a whole hand want its palm's size.
    """
    if not span > 0.0:
        return 0.0
    half = 0.5 * float(span)
    e = 10.0 ** np.floor(np.log10(half))
    return float(max(m for m in (1.0, 2.0, 5.0) if m * e <= half * (1.0 + 1e-9)) * e)


#######################################################################################
# Primitive meshes, in the geom's frame
#######################################################################################


def box_mesh(h):
    """A box of half-extents `h`, four vertices per face so its edges stay sharp."""
    v, f = [], []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            u, w = [a for a in range(3) if a != axis]
            quad = []
            for su, sw in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = np.zeros(3)
                p[axis], p[u], p[w] = sign * h[axis], su * h[u], sw * h[w]
                quad.append(p)
            k = len(v)
            v += quad
            # Outward winding: the quad's (u, w) order is right-handed about +axis
            # for (axis, u, w) a cyclic permutation, and flips otherwise.
            right = (axis, u, w) in ((0, 1, 2), (1, 2, 0), (2, 0, 1))
            out = right == (sign > 0)
            f += [[k, k + 1, k + 2], [k, k + 2, k + 3]] if out else [[k, k + 2, k + 1], [k, k + 3, k + 2]]
    return np.asarray(v, np.float32), np.asarray(f, np.int32)


def _revolve(profile, nu=24):
    """A surface of revolution about z from `(r, z)` profile points, bottom to top."""
    a = np.linspace(0.0, 2.0 * np.pi, nu, endpoint=False)
    ring = np.stack([np.cos(a), np.sin(a)], -1)
    v = np.concatenate([np.column_stack([r * ring, np.full(nu, z)]) for r, z in profile])
    f = []
    for i in range(len(profile) - 1):
        for j in range(nu):
            a0, a1 = i * nu + j, i * nu + (j + 1) % nu
            b0, b1 = a0 + nu, a1 + nu
            f += [[a0, a1, b1], [a0, b1, b0]]
    return v.astype(np.float32), np.asarray(f, np.int32)


def capsule_mesh(r, h, nv=8):
    """A capsule: radius `r`, cylinder half-length `h` along z."""
    th = np.linspace(-np.pi / 2, np.pi / 2, 2 * nv + 1)
    prof = [(r * np.cos(t), (h if t >= 0 else -h) + r * np.sin(t)) for t in th]
    return _revolve(prof)


def cylinder_mesh(r, h):
    """A capped cylinder: radius `r`, half-height `h` along z."""
    return _revolve([(0.0, -h), (r, -h), (r, h), (0.0, h)])


def ellipsoid_mesh(radii):
    v, f = sphere_mesh(np.zeros((1, 3)), np.ones(1), nu=24, nv=13)
    return (v * np.asarray(radii, np.float32)).astype(np.float32), f


def flat(v, f):
    """Each face its own vertices, so the renderer's per-vertex normals are the face's."""
    f = np.asarray(f, np.int64).reshape(-1, 3)
    return np.asarray(v, np.float32)[f.reshape(-1)], np.arange(3 * len(f), dtype=np.int32).reshape(-1, 3)


def geom_mesh(mjm, g, plane_size):
    """`(key, (v, f))` for MJCF geom `g`: the key identifies identical prototypes."""
    t = int(mjm.geom_type[g])
    s = np.asarray(mjm.geom_size[g], np.float64)
    T = mujoco.mjtGeom
    if t == T.mjGEOM_PLANE:
        hx = s[0] if s[0] > 0 else plane_size
        hy = s[1] if s[1] > 0 else plane_size
        v, f = plane_mesh(1.0)
        return ("plane", hx, hy), ((v * np.array([hx, hy, 1.0], np.float32)).astype(np.float32), f)
    if t == T.mjGEOM_SPHERE:
        return ("sphere", s[0]), sphere_mesh(np.zeros((1, 3)), np.array([s[0]]), nu=24, nv=13)
    if t == T.mjGEOM_CAPSULE:
        return ("capsule", s[0], s[1]), capsule_mesh(s[0], s[1])
    if t == T.mjGEOM_CYLINDER:
        return ("cylinder", s[0], s[1]), flat(*cylinder_mesh(s[0], s[1]))
    if t == T.mjGEOM_ELLIPSOID:
        return ("ellipsoid", *s), ellipsoid_mesh(s)
    if t == T.mjGEOM_BOX:
        return ("box", *s), box_mesh(s)
    if t == T.mjGEOM_MESH:
        m = int(mjm.geom_dataid[g])
        va, vn = int(mjm.mesh_vertadr[m]), int(mjm.mesh_vertnum[m])
        fa, fn = int(mjm.mesh_faceadr[m]), int(mjm.mesh_facenum[m])
        v = np.asarray(mjm.mesh_vert[va:va + vn], np.float32)
        f = np.asarray(mjm.mesh_face[fa:fa + fn], np.int32)
        return ("mesh", m), (flat(v, f) if fn < FLAT_FACES else (v, f))
    return None, None  # heightfields and SDF geoms are not drawn


#######################################################################################
# The scene
#######################################################################################


def _xyzw(q_wxyz):
    q = np.asarray(q_wxyz, np.float64).reshape(-1, 4)
    return np.concatenate([q[:, 1:], q[:, :1]], axis=1)


def _rgba(mjm, g):
    m = int(mjm.geom_matid[g])
    rgba = np.asarray(mjm.mat_rgba[m] if m >= 0 else mjm.geom_rgba[g], np.float64)
    return rgba


def make_live_scene(sim, plane_size=None):
    """
    The drawable geometry of a `lodos.sim.Sim`: its Lodos surfaces and every MJCF geom
    of `sim.mjm` (the model `make_sim` built, where a surface's geom has left MJWarp's
    collision).
    """
    mjm = sim.mjm
    nbody, ngeom = int(mjm.nbody), int(mjm.ngeom)
    geom_body = np.asarray(mjm.geom_bodyid, int)

    surfaces = sim.surfaces
    lodos = []  # (surface, body)
    if surfaces is not None and sim.mcol is not None:
        for s, g in zip(surfaces.surfaces, surfaces.geoms):
            if not isinstance(s, sdf_types.PlaneSurface):
                lodos.append((s, int(geom_body[int(g)])))
    lodos_bodies = np.array(sorted({b for _, b in lodos}), dtype=int)

    # Each body's extent, from everything it carries, sets the scene's length scale:
    # the median body, not the median piece, which for polytopes is a fifth of one.
    # Bounding boxes of each geom's own vertices, placed in the body's frame.
    contype, conaff = np.asarray(mjm.geom_contype), np.asarray(mjm.geom_conaffinity)
    pts = {}
    for s, b in lodos:
        pts.setdefault(b, []).append(np.asarray(s.v, np.float64))
    prims = {}
    for g in range(ngeom):
        key, mesh = geom_mesh(mjm, g, 1.0)
        prims[g] = (key, mesh)
        if key is None or int(mjm.geom_type[g]) == mujoco.mjtGeom.mjGEOM_PLANE or geom_body[g] == 0:
            continue
        v = np.asarray(mesh[0], np.float64)
        lo, hi = v.min(0), v.max(0)
        box = np.array([[hi[0] if i & 4 else lo[0], hi[1] if i & 2 else lo[1], hi[2] if i & 1 else lo[2]]
                        for i in range(8)])
        q = _xyzw(mjm.geom_quat[g])
        box = rotate_xyzw(np.repeat(q, 8, 0), box) + np.asarray(mjm.geom_pos[g], np.float64)
        pts.setdefault(int(geom_body[g]), []).append(box)
    spans = [float(np.ptp(np.concatenate(p), axis=0).max()) for p in pts.values()]
    scale = float(np.median(spans)) if spans else 1.0
    plane_size = 20.0 * scale if plane_size is None else float(plane_size)

    body_color = np.zeros((nbody, 3), np.float32)
    k = 0
    for b in range(nbody):
        if b == 0:
            body_color[b] = WORLD_COLOR
        elif b in pts:
            body_color[b] = PALETTE[k % len(PALETTE)]
            k += 1
        else:
            body_color[b] = PALETTE[k % len(PALETTE)]
    body_name = [mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_BODY, b) or f"body {b}" for b in range(nbody)]

    meshes, index = [], {}
    rows = []  # (mesh, body, local7, kind, color, alpha, label, geom)
    ident = np.array([0, 0, 0, 0, 0, 0, 1], np.float64)
    for s, b in lodos:
        key = ("lodos", id(s))
        if key not in index:
            index[key] = len(meshes)
            meshes.append((np.asarray(s.v, np.float32), np.asarray(s.f, np.int32)))
        rows.append((index[key], b, ident, KIND_LODOS, body_color[b], 1.0, f"{body_name[b]}: Lodos surface", -1))

    for g in range(ngeom):
        key, mesh = prims[g]
        if key is not None and key[0] == "plane":
            key, mesh = geom_mesh(mjm, g, plane_size)
        if key is None:
            continue
        if key not in index:
            index[key] = len(meshes)
            meshes.append(mesh)
        b = int(geom_body[g])
        t = int(mjm.geom_type[g])
        if t == mujoco.mjtGeom.mjGEOM_PLANE:
            kind = KIND_PLANE
        elif contype[g] or conaff[g]:
            kind = KIND_PRIM
        else:
            kind = KIND_VISUAL
        rgba = _rgba(mjm, g)
        alpha = float(rgba[3])
        if kind == KIND_PLANE:
            color = GROUND_RGB
        elif np.allclose(rgba, MJ_DEFAULT_RGBA):
            color = body_color[b]
        elif kind == KIND_PRIM and int(mjm.geom_group[g]) == COLLISION_GROUP:
            color, alpha = body_color[b], 1.0
        else:
            color = rgba[:3]
        local = np.concatenate([np.asarray(mjm.geom_pos[g], np.float64), _xyzw(mjm.geom_quat[g])[0]])
        name = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_GEOM, g)
        tname = mujoco.mjtGeom(t).name.removeprefix("mjGEOM_").lower()
        label = f"{body_name[b]}: {name or f'geom {g}'} ({tname})"
        rows.append((index[key], b, local, kind, color, alpha, label, g))

    n = len(rows)
    kind = np.array([r[3] for r in rows], int)
    body = np.array([r[1] for r in rows], int)
    show0 = (kind != KIND_VISUAL) & ~((kind == KIND_PRIM) & np.isin(body, lodos_bodies))
    jnt_type = np.asarray(mjm.jnt_type).astype(int)
    return LiveScene(
        meshes=meshes,
        geom_mesh=np.array([r[0] for r in rows], int),
        geom_color=np.array([r[4] for r in rows], np.float32).reshape(n, 3),
        geom_label=np.array([r[6] for r in rows], dtype="U96"),
        jnt_axis_ok=np.isin(jnt_type, (mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_HINGE)),
        scale=scale,
        geom_body=body,
        geom_id=np.array([r[7] for r in rows], int),
        geom_local=np.array([r[2] for r in rows], np.float64).reshape(n, 7),
        geom_kind=kind,
        geom_alpha=np.array([r[5] for r in rows], np.float64),
        geom_show0=show0,
        body_name=body_name,
        body_color=body_color,
        lodos_bodies=lodos_bodies,
        checker=checker_size(max(spans) if spans else scale),
    )


#######################################################################################
# Poses
#######################################################################################


def quat_mul_xyzw(a, b):
    """Hamilton products of `(..., 4)` quaternions in (x, y, z, w) order."""
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], axis=-1)


def rotate_xyzw(q, v):
    """Rotate `(..., 3)` vectors by `(..., 4)` unit quaternions in (x, y, z, w) order."""
    u, w = q[..., :3], q[..., 3:4]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def geom_xforms(scene, xpos, xquat):
    """
    `(nG, 7)` world poses of the drawn geometries, in Warp's (x, y, z, w) order, from
    the bodies' `xpos` `(nbody, 3)` and MuJoCo-order `xquat` `(nbody, 4)`.
    """
    bq = _xyzw(xquat)[scene.geom_body]
    bp = np.asarray(xpos, np.float64)[scene.geom_body]
    lp, lq = scene.geom_local[:, :3], scene.geom_local[:, 3:]
    out = np.empty((len(scene.geom_body), 7), np.float64)
    out[:, :3] = bp + rotate_xyzw(bq, lp)
    q = quat_mul_xyzw(bq, lq)
    out[:, 3:] = q / np.linalg.norm(q, axis=1, keepdims=True)
    return out
