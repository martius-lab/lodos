"""
MJWarp's collision pipeline, run on Lodos's rigid-body state.

The geometries here are the MJCF's own: planes, spheres, capsules, cylinders, boxes,
meshes. MJWarp's broadphase pairs them, MJWarp's narrowphases handle every pair type
the exact hull narrowphase (`lodos.col.prim.hybrid`) does not, and the hybrid handles
box-box, box-mesh, mesh-mesh and plane-mesh. Everything lands in MJWarp's contact
arrays, `d.contact` and `d.nacon`, where the contact solve reads them, beside Lodos's
contacts, which `lodos.cd.driver` appends.

Two things differ from MJWarp's own `collision()`:

**Geometry poses come from Lodos's kinematics.** `lodos.rbd` computes body poses
only, so `geom_kinematics` derives `geom_xpos`/`geom_xmat` for the geometries this
pipeline can pair -- the ones MJWarp's broadphase can return -- and nothing else.

**The hybrid runs beside MJWarp's narrowphases, on a persistent context.** The
collision context is created once and persists, which the hybrid's feature cache
needs across steps, and the step calls `collide` directly.

The hybrid works at zero margin and gap on its geoms; `PrimCollision` checks it.
"""

from __future__ import annotations

import mujoco
import numpy as np
import warp as wp


from lodos.col.prim import collision_driver as prim_collision_driver
from lodos.col.prim.collision_convex import convex_narrowphase
from lodos.col.prim.collision_core import contact_params
from lodos.col.prim.collision_core import create_collision_context
from lodos.col.prim.collision_core import write_contact
from lodos.col.prim.collision_primitive import primitive_narrowphase
from lodos.mjcf.types import COLLISION_TABLE
from lodos.mjcf.types import BroadphaseType
from lodos.mjcf.types import CollisionType
from lodos.mjcf.types import DisableBit
from lodos.mjcf.types import vec5
from lodos.utils.math import make_frame

from .hybrid.kernels import PipeState
from .hybrid.pipeline import ALL_TYPES
from .hybrid.pipeline import HybridNarrowphase

wp.set_module_options({"enable_backward": False})


#######################################################################################
# Geometry poses
#######################################################################################


@wp.kernel
def geom_kinematics(
    geom_ids: wp.array(dtype=int),  # (n,) the geometries to place # type: ignore
    geom_bodyid: wp.array(dtype=int),  # (ngeom,) # type: ignore
    geom_pos: wp.array(dtype=wp.vec3),  # (ngeom,) offset in the body frame # type: ignore
    geom_rot: wp.array(dtype=wp.mat33),  # (ngeom,) orientation in the body frame # type: ignore
    xpos: wp.array(dtype=wp.vec3, ndim=2),  # (B, nbody) # type: ignore
    xmat: wp.array(dtype=wp.mat33, ndim=2),  # (B, nbody) # type: ignore
    geom_xpos: wp.array(dtype=wp.vec3, ndim=2),  # (B, ngeom) out # type: ignore
    geom_xmat: wp.array(dtype=wp.mat33, ndim=2),  # (B, ngeom) out # type: ignore
):
    """
    World pose of each pairable geometry, from its body's: MuJoCo's `mj_kinematics` tail.

    Launch: dim=(nworld, n), nworld = B, a thread per world and pairable geometry, by `PrimCollision.poses`.
    In: xpos (B, nbody) wp.vec3 and xmat (B, nbody) wp.mat33, Lodos's body poses; the geometries' bodies and
    body-frame poses. Out: geom_xpos (B, ngeom) wp.vec3 and geom_xmat (B, ngeom) wp.mat33, MJWarp's
    `d.geom_xpos`/`d.geom_xmat`, at the n geometries of `geom_ids`.
    """
    b, k = wp.tid()  # int, int: the world, the entry of geom_ids
    g = geom_ids[k]  # int: the geometry
    body = geom_bodyid[g]  # int: its body
    r = xmat[b, body]  # wp.mat33: the body's orientation
    geom_xpos[b, g] = xpos[b, body] + r @ geom_pos[g]
    geom_xmat[b, g] = r @ geom_rot[g]


#######################################################################################
# The hybrid's contacts, into MJWarp's arrays
#######################################################################################


@wp.struct
class ContactModel:
    """
    Model fields `contact_params` reads, plus the broadphase pair ids.

    Built by `PrimCollision.__init__` from MJWarp's model and collision context; read by `export_kernel`.
    """

    geom_condim: wp.array(dtype=int)  # (ngeom,)
    geom_priority: wp.array(dtype=int)  # (ngeom,)
    geom_solmix: wp.array2d(dtype=float)  # (1 or nworld, ngeom)
    geom_solref: wp.array2d(dtype=wp.vec2)  # (1 or nworld, ngeom)
    geom_solimp: wp.array2d(dtype=vec5)  # (1 or nworld, ngeom)
    geom_friction: wp.array2d(dtype=wp.vec3)  # (1 or nworld, ngeom)
    geom_margin: wp.array2d(dtype=float)  # (1 or nworld, ngeom)
    geom_gap: wp.array2d(dtype=float)  # (1 or nworld, ngeom)
    geom_adhesion: wp.array2d(dtype=float)  # (1 or nworld, ngeom)
    pair_dim: wp.array(dtype=int)  # (npair,)
    pair_solref: wp.array2d(dtype=wp.vec2)  # (1 or nworld, npair)
    pair_solreffriction: wp.array2d(dtype=wp.vec2)  # (1 or nworld, npair)
    pair_solimp: wp.array2d(dtype=vec5)  # (1 or nworld, npair)
    pair_margin: wp.array2d(dtype=float)  # (1 or nworld, npair)
    pair_gap: wp.array2d(dtype=float)  # (1 or nworld, npair)
    pair_adhesion: wp.array2d(dtype=float)  # (1 or nworld, npair)
    pair_friction: wp.array2d(dtype=vec5)  # (1 or nworld, npair)
    collision_pairid: wp.array(dtype=wp.vec2i)  # (naconmax,) per broadphase slot: its explicit pair and sensor ids


@wp.struct
class ContactOut:
    """
    MJWarp's contact arrays (Data.contact) and counter.

    Built by `contact_out`; written by `export_kernel`, `lodos.cd.driver.export_lodos` and
    `lodos.cd.groups.export_groups`.
    """

    naconmax: int  # the contact capacity, over all worlds
    dist: wp.array(dtype=float)  # (naconmax,)
    pos: wp.array(dtype=wp.vec3)  # (naconmax,)
    frame: wp.array(dtype=wp.mat33)  # (naconmax,) the normal in row 0
    includemargin: wp.array(dtype=float)  # (naconmax,)
    friction: wp.array(dtype=vec5)  # (naconmax,)
    solref: wp.array(dtype=wp.vec2)  # (naconmax,)
    solreffriction: wp.array(dtype=wp.vec2)  # (naconmax,)
    solimp: wp.array(dtype=vec5)  # (naconmax,)
    dim: wp.array(dtype=int)  # (naconmax,)
    geom: wp.array(dtype=wp.vec2i)  # (naconmax,)
    efc_address: wp.array2d(dtype=int)  # (naconmax, nmaxpyramid)
    worldid: wp.array(dtype=int)  # (naconmax,)
    type: wp.array(dtype=int)  # (naconmax,) ContactType bits
    geomcollisionid: wp.array(dtype=int)  # (naconmax,)
    adhesion: wp.array(dtype=float)  # (naconmax,)
    nacon: wp.array(dtype=int)  # (1,) the contacts written, over all worlds


def contact_out(d) -> ContactOut:
    """
    `d`'s contact arrays and counter as a `ContactOut`: the same arrays, not copies.

    Called by `PrimCollision.__init__` and `lodos.cd.driver.SolverDriver.__init__`.
    In: d: MJWarp's Data, its naconmax, the (naconmax,) arrays of d.contact and d.nacon (1,) int. Out: ContactOut.
    """
    co = ContactOut()  # ContactOut
    co.naconmax = d.naconmax
    # f: str, a field of d.contact
    for f in ("dist", "pos", "frame", "includemargin", "friction", "solref", "solreffriction", "solimp", "dim",
              "geom", "efc_address", "worldid", "type", "geomcollisionid", "adhesion"):
        setattr(co, f, getattr(d.contact, f))
    co.nacon = d.nacon
    return co


@wp.kernel
def export_kernel(cm: ContactModel, st: PipeState, co: ContactOut):  # type: ignore
    """
    Appends the hybrid's contacts to MJWarp's contact arrays (one thread per contact).

    Launch: dim=cap_con (4 * naccdmax), a thread per slot of the hybrid's contact buffer, by `PrimCollision.collide`.
    Reads: st.con_count (1,) int; st.con_slot and st.con_k (cap_con,) int, st.con_dist (cap_con,) float, st.con_pos
    and st.con_normal (cap_con,) wp.vec3; st.collision_pair (naconmax,) wp.vec2i and st.collision_worldid
    (naconmax,) int; cm's geom and pair parameters. Writes: co's arrays (naconmax,), at the slots co.nacon (1,) int
    hands out.
    """
    j = wp.tid()  # int: the hybrid's contact
    if j >= wp.min(st.con_count[0], st.con_dist.shape[0]):
        return
    slot = st.con_slot[j]  # int: its pair's broadphase slot
    worldid = st.collision_worldid[slot]  # int
    # wp.vec2i, float, float, int, vec5, wp.vec2, wp.vec2, vec5, float: the pair's contact parameters
    geoms, margin, gap, condim, friction, solref, solreffriction, solimp, adhesion = contact_params(
        cm.geom_condim, cm.geom_priority, cm.geom_solmix, cm.geom_solref, cm.geom_solimp, cm.geom_friction,
        cm.geom_margin, cm.geom_gap, cm.geom_adhesion, cm.pair_dim, cm.pair_solref, cm.pair_solreffriction,
        cm.pair_solimp, cm.pair_margin, cm.pair_gap, cm.pair_adhesion, cm.pair_friction, st.collision_pair,
        cm.collision_pairid, slot, worldid,
    )
    write_contact(
        co.naconmax, st.con_k[j], st.con_dist[j], st.con_pos[j], make_frame(st.con_normal[j]), margin, gap, condim,
        friction, solref, solreffriction, solimp, adhesion, geoms, cm.collision_pairid[slot], worldid, co.dist, co.pos,
        co.frame, co.includemargin, co.friction, co.solref, co.solreffriction, co.solimp, co.dim, co.geom,
        co.efc_address, co.worldid, co.type, co.geomcollisionid, co.adhesion, co.nacon,
    )


#######################################################################################
# The pipeline
#######################################################################################


def pairable_geoms(mjm: mujoco.MjModel, m):
    """
    Geometries MJWarp's broadphase can pair, and how many pairs it can return.

    Read off MJWarp's own precomputed candidate list (`nxn_geom_pair_filtered`: the
    contype/conaffinity filter, no self or parent-child pairs, `<exclude>`s and
    explicit `<pair>`s applied), keeping the pairs whose type combination has a
    collider. The count decides whether the step runs this pipeline at all: a scene
    whose only MJCF collider left is its floor has nothing to pair.

    Called by `PrimCollision.__init__`.
    In: mjm: mujoco.MjModel, its geom_type; m: MJWarp's Model, its nxn_geom_pair_filtered (nnxn_geom_pair_filtered,)
    wp.vec2i. Out: (n,) int32, the pairable geometries, sorted; int, the pairs with a collider.
    """
    # (nnxn_geom_pair_filtered, 2) int32: MJWarp's candidate pairs
    pairs = m.nxn_geom_pair_filtered.numpy().reshape(-1, 2) if m.nxn_geom_pair_filtered.shape[0] else np.zeros((0, 2), int)
    table = COLLISION_TABLE  # {(GeomType, GeomType): CollisionType}: the pair types with a collider
    gt = np.asarray(mjm.geom_type)  # (ngeom,) int32
    keep = []  # list of (g1, g2): the candidate pairs with a collider
    for g1, g2 in pairs:  # g1, g2: int32, a candidate pair's geometries
        t1, t2 = sorted((int(gt[g1]), int(gt[g2])))  # int, int: their types, in order
        if any(int(k[0]) == t1 and int(k[1]) == t2 for k in table):
            keep.append((g1, g2))
    keep = np.asarray(keep, dtype=np.int32).reshape(-1, 2)
    return np.unique(keep).astype(np.int32), len(keep)


class PrimCollision:
    """
    MJWarp's broadphase and narrowphases with the exact hull narrowphase, on a
    persistent collision context, writing `d.contact`/`d.nacon`.

    hybrid: the pair types the hybrid takes over (a subset of
        `hybrid.pipeline.ALL_TYPES`), or None/() to run MJWarp's own narrowphases
        for every pair type.

    Built by `lodos.sim.make_sim`, as `sim.prim`; `lodos.step.forward` calls `poses` and `collide` every step.
    """

    def __init__(self, mjm, m, d, hybrid=ALL_TYPES, **hybrid_kwargs):
        """
        Finds the pairable geometries, puts their body-frame poses on the device, and builds the hybrid for those
        of the `hybrid` pair types the scene has.

        Called by `lodos.sim.make_sim`.
        In: mjm: mujoco.MjModel; m, d: MJWarp's Model and Data (d.naconmax, the contact capacity); hybrid: the
        hybrid's pair types; hybrid_kwargs: `hybrid.pipeline.HybridNarrowphase`'s options. Raises ValueError for a
        nonzero margin or gap on a hybrid geom or an explicit pair, NotImplementedError for NATIVECCD off, SDF geoms
        or flex, when the hybrid runs.
        """
        self.m, self.d = m, d  # MJWarp's Model and Data
        self.geom_ids_np, self.npairs = pairable_geoms(mjm, m)  # (n,) int32: the pairable geometries; int
        self.active = self.npairs > 0 and d.naconmax > 0  # bool: whether the step runs this pipeline
        if m.opt.disableflags & (DisableBit.CONSTRAINT | DisableBit.CONTACT):
            self.active = False

        # Static body-frame offsets of every geometry, for `geom_kinematics`.
        rot = np.zeros((mjm.ngeom, 9))  # (ngeom, 9) float64: the geometries' rotation matrices, row-major
        for g in range(mjm.ngeom):  # g: int, a geometry
            mujoco.mju_quat2Mat(rot[g], mjm.geom_quat[g])
        # (max(n, 1),) int: the pairable geometries, or one placeholder
        self.geom_ids = wp.array(self.geom_ids_np if len(self.geom_ids_np) else np.zeros(1, np.int32), dtype=int)
        self.geom_bodyid = wp.array(np.asarray(mjm.geom_bodyid, dtype=np.int32), dtype=int)  # (ngeom,) int
        self.geom_pos = wp.array(np.asarray(mjm.geom_pos, dtype=np.float32), dtype=wp.vec3)  # (ngeom,) wp.vec3
        self.geom_rot = wp.array(rot.reshape(-1, 3, 3).astype(np.float32), dtype=wp.mat33)  # (ngeom,) wp.mat33

        self.hy = None  # hybrid.pipeline.HybridNarrowphase, or None
        self.ctx = create_collision_context(d.naconmax)  # CollisionContext: the broadphase's pairs, (naconmax,)
        types = tuple(hybrid or ())  # tuple of (GeomType, GeomType): the pair types for the hybrid
        if self.active and types:
            used = np.zeros(mjm.ngeom, dtype=bool)  # (ngeom,) bool: the pairable geometries
            used[self.geom_ids_np] = True
            # (ngeom,) bool: the pairable geometries of the hybrid's geom types
            in_types = np.isin(mjm.geom_type, [int(t) for pair in types for t in pair]) & used
            if np.any(mjm.geom_margin[in_types] != 0.0) or np.any(mjm.geom_gap[in_types] != 0.0) or mjm.npair:
                raise ValueError("the hybrid narrowphase supports zero margin and gap on its geoms, and no explicit pairs")
            if m.opt.disableflags & DisableBit.NATIVECCD or m.has_sdf_geom or m.nflex > 0:
                raise NotImplementedError("NATIVECCD off, SDF geoms and flex are not supported with the hybrid narrowphase")
            # Only pair types that actually occur get the hybrid; a scene with no hull
            # or mesh geometries runs MJWarp's narrowphases alone.
            # set of int: the pairable geometries' types
            present = {int(t) for t in np.asarray(mjm.geom_type)[self.geom_ids_np]}
            types = tuple(t for t in types if int(t[0]) in present and int(t[1]) in present)
        if self.active and types:
            # Hull tables for the meshes something can pair; the rest -- a surface's
            # inertia mesh, say -- get placeholders.
            meshes = {int(mjm.geom_dataid[g]) for g in self.geom_ids_np  # set of int: the meshes something can pair
                      if int(mjm.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_MESH)}
            self.hy = HybridNarrowphase(mjm, m, d, self.ctx, types=types, meshes=meshes, **hybrid_kwargs)
            cm = ContactModel()  # ContactModel
            # f: str, a model field
            for f in ("geom_condim", "geom_priority", "geom_solmix", "geom_solref", "geom_solimp", "geom_friction",
                      "geom_margin", "geom_gap", "geom_adhesion", "pair_dim", "pair_solref", "pair_solreffriction",
                      "pair_solimp", "pair_margin", "pair_gap", "pair_adhesion", "pair_friction"):
                setattr(cm, f, getattr(m, f))
            cm.collision_pairid = self.ctx.collision_pairid
            self.cm, self.co = cm, contact_out(d)  # ContactModel, ContactOut: export_kernel's
        table = COLLISION_TABLE  # {(GeomType, GeomType): CollisionType}
        taken = self.hy.types if self.hy is not None else ()  # tuple of (GeomType, GeomType): the hybrid's types
        # list of (GeomType, GeomType): the pair types for MJWarp's convex narrowphase
        self.convex_pairs = [k for k, v in table.items() if v == CollisionType.CONVEX and k not in taken]
        # list of (GeomType, GeomType): those for its primitive narrowphase
        self.primitive_pairs = [k for k, v in table.items() if v == CollisionType.PRIMITIVE and k not in taken]

    def poses(self, xpos, xmat):
        """
        `geom_xpos`/`geom_xmat` of the pairable geometries, from body poses.

        Called by `lodos.step.forward` and `tools.performance_tuning.profile_scene.stages_of`.
        In: xpos (B, nbody) wp.vec3 and xmat (B, nbody) wp.mat33, Lodos's body poses (`drbd.xpos`, `drbd.xmat`).
        Out: d.geom_xpos (nworld, ngeom) wp.vec3 and d.geom_xmat (nworld, ngeom) wp.mat33, at the pairable
        geometries (nworld = B).
        """
        if not self.active:
            return
        d = self.d  # MJWarp's Data
        wp.launch(
            geom_kinematics,
            dim=(d.nworld, len(self.geom_ids_np)),
            inputs=(self.geom_ids, self.geom_bodyid, self.geom_pos, self.geom_rot, xpos, xmat),
            outputs=(d.geom_xpos, d.geom_xmat),
        )

    def collide(self):
        """
        MJWarp's `collision()`, with the hybrid for its pair types. Zeroes `nacon` first.

        Called by `lodos.step.forward` and `tools.performance_tuning.profile_scene.stages_of`.
        Reads: d.geom_xpos (nworld, ngeom) wp.vec3 and d.geom_xmat (nworld, ngeom) wp.mat33; the model.
        Writes: d.ncollision (1,) int and the broadphase's pairs in `ctx`, (naconmax,) each; d.contact's arrays
        (naconmax,) and d.nacon (1,) int; the hybrid's buffers.
        """
        m, d = self.m, self.d  # MJWarp's Model and Data
        d.ncollision.zero_()
        d.nacon.zero_()
        if not self.active:
            return
        if m.opt.broadphase == BroadphaseType.NXN:
            prim_collision_driver.nxn_broadphase(m, d, self.ctx)
        else:
            prim_collision_driver.sap_broadphase(m, d, self.ctx)
        convex_narrowphase(m, d, self.ctx, self.convex_pairs)
        primitive_narrowphase(m, d, self.ctx, self.primitive_pairs)
        if self.hy is not None:
            self.hy.launch()
            wp.launch(export_kernel, dim=self.hy.cap_con, inputs=[self.cm, self.hy.st, self.co])
        if m.callback.contactfilter:
            m.callback.contactfilter(m, d)
