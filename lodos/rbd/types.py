"""
Data structures for unconstrained rigid-body dynamics in Warp.

This is a port of the algorithms `mujoco_warp` implements, cut down to exactly
the feature set `_SUPPORTED` names. It exists so that the whole step --
unconstrained dynamics, collision detection, contact dynamics and the integrator
-- lives on the device and can be captured into a single CUDA graph.

Three conventions carry over from the rest of the Warp code and are load-bearing:

**Quaternions are (w, x, y, z)**, MuJoCo's order, stored as `wp.vec4` rather than
`wp.quat` (which is (x, y, z, w)). `lodos.col.sdf.broadphase.compute_T_w` is the
one place the convention changes, and it expects this order.

**State arrays are `(B, n)`**, batch first. Every kernel here launches with the
batch as the *slowest*-varying tid, so consecutive threads walk consecutive `n`
and reads coalesce.

**The arrays are shared, not copied.** `make_collision_context` and the
contact solver (`lodos.cd.driver.bind_state`) bind the very same `wp.array` objects this module
allocates, so `xpos` written by `kinematics` is the `xpos` collision reads. There
is no per-step transfer anywhere.

Tree passes are organized by *level*: `body_level_id[body_level_beg[l] :
body_level_beg[l+1]]` lists the bodies whose parent chain to the world is `l`
long, so a forward pass launches one kernel per level and every body in it reads
a parent written by the previous launch. Reverse passes read *children* through
`body_child_id`/`body_child_beg` rather than having children accumulate into
their parent, which keeps them free of atomics and therefore bitwise
reproducible.
"""

import numpy as np
import warp as wp

import mujoco

from lodos.utils.math import MINVAL  # MuJoCo's mjMINVAL; here also the mass below which a subtree is massless

# com-based body inertia: [Ixx, Iyy, Izz, Ixy, Ixz, Iyz, mass*offset(3), mass].
# Defined here: it is a rigid-body quantity, and collision and contact dynamics
# both depend on this package, not the reverse.
vec10 = wp.types.vector(length=10, dtype=wp.float32)

# Joint types, MuJoCo's numbering (mjtJoint).
FREE = 0
BALL = 1
SLIDE = 2
HINGE = 3

# Actuator transmission types, MuJoCo's numbering (mjtTrn).
TRN_JOINT = 0
TRN_JOINTINPARENT = 1
TRN_SITE = 4

# Actuator dynamics types (mjtDyn).
DYN_NONE = 0
DYN_INTEGRATOR = 1
DYN_FILTER = 2
DYN_FILTEREXACT = 3

# Actuator gain and bias types (mjtGain, mjtBias). Both are 0 for the constant
# case -- `mjGAIN_FIXED` and `mjBIAS_NONE` -- and 1 for AFFINE, which is the only
# value a kernel has to test.
AFFINE = 1


#######################################################################################
# Model
#######################################################################################


@wp.struct
class RbdModel:
    """
    Everything that depends only on the model. Not batched -- one model serves every
    world, and nothing here may vary with `b`.

    Built by `make_rbd_context`; a `lodos.sim.Sim` holds it as `mrbd`.
    """

    B: wp.int32  # batch size
    nq: wp.int32  # generalized coordinates
    nv: wp.int32  # dofs
    nbody: wp.int32  # bodies, the world included
    njnt: wp.int32  # joints
    nsite: wp.int32  # sites
    nu: wp.int32  # actuators
    na: wp.int32  # actuator activation states
    nlevel: wp.int32  # depth of the body tree, world included
    ntree: wp.int32  # independent kinematic trees; M is block diagonal across them
    nblk: wp.int32  # total entries across all per-tree mass-matrix blocks
    ngravcomp: wp.int32  # bodies with gravity compensation

    dt: wp.float32  # m.opt.timestep
    gravity: wp.vec3  # m.opt.gravity (m/s^2)
    # `mujoco.mjtDisableBit`, split out so a kernel tests an int rather than a bit.
    # Decoded from the live enum -- see `_disable_flags`.
    disable_spring: wp.int32  # 1: mjDSBL_SPRING set
    disable_damper: wp.int32  # 1: mjDSBL_DAMPER set
    disable_gravity: wp.int32  # 1: mjDSBL_GRAVITY set
    disable_clampctrl: wp.int32  # 1: mjDSBL_CLAMPCTRL set
    disable_actuation: wp.int32  # 1: mjDSBL_ACTUATION set
    # The integrator: 0 semi-implicit Euler, 1 MuJoCo's `implicitfast` (`lodos.rbd.implicit`).
    # `implicit_constraints` picks the matrix a constraint solve sees under it: 0 the mass
    # matrix, as MuJoCo has it, the implicit one only integrating the result; 1 `M - dt D`
    # throughout.
    implicit: wp.int32  # 0 or 1, as above
    implicit_constraints: wp.int32  # 0 or 1, as above

    # --------------------------------------------------------------------------------
    # Body tree. `body_level_*` drives the forward passes, `body_child_*` the
    # reverse ones; see the module docstring for why the two directions differ.
    body_parentid: wp.array(dtype=int)  # (nbody,) # type: ignore
    body_rootid: wp.array(dtype=int)  # (nbody,) top of this body's kinematic tree # type: ignore
    body_level_beg: wp.array(dtype=int)  # (nlevel+1,) # type: ignore
    body_level_id: wp.array(dtype=int)  # (nbody,) bodies sorted by level # type: ignore
    body_child_beg: wp.array(dtype=int)  # (nbody+1,) # type: ignore
    body_child_id: wp.array(dtype=int)  # (nbody-1,) children, grouped by parent # type: ignore

    body_jntadr: wp.array(dtype=int)  # (nbody,) first joint, -1 if none # type: ignore
    body_jntnum: wp.array(dtype=int)  # (nbody,) # type: ignore
    body_dofadr: wp.array(dtype=int)  # (nbody,) first dof, -1 if none # type: ignore
    body_dofnum: wp.array(dtype=int)  # (nbody,) # type: ignore
    body_pos: wp.array(dtype=wp.vec3)  # (nbody,) frame offset from parent # type: ignore
    body_quat: wp.array(dtype=wp.vec4)  # (nbody,) # type: ignore
    body_ipos: wp.array(dtype=wp.vec3)  # (nbody,) inertial frame in body frame # type: ignore
    body_iquat: wp.array(dtype=wp.vec4)  # (nbody,) # type: ignore
    body_mass: wp.array(dtype=float)  # (nbody,) # type: ignore
    body_inertia: wp.array(dtype=wp.vec3)  # (nbody,) diagonal, in the inertial frame # type: ignore
    body_gravcomp: wp.array(dtype=float)  # (nbody,) fraction of weight to cancel # type: ignore
    # Bodies of each subtree, flattened -- body i's own subtree is
    # `body_subtree_id[body_subtree_beg[i] : body_subtree_beg[i+1]]`, itself
    # included. Only gravity compensation needs it, and it is empty without.
    body_subtree_beg: wp.array(dtype=int)  # (nbody+1,) # type: ignore
    body_subtree_id: wp.array(dtype=int)  # (sum of subtree sizes,) # type: ignore

    # --------------------------------------------------------------------------------
    # Joints.
    jnt_type: wp.array(dtype=int)  # (njnt,) # type: ignore
    jnt_bodyid: wp.array(dtype=int)  # (njnt,) # type: ignore
    jnt_qposadr: wp.array(dtype=int)  # (njnt,) # type: ignore
    jnt_dofadr: wp.array(dtype=int)  # (njnt,) # type: ignore
    jnt_pos: wp.array(dtype=wp.vec3)  # (njnt,) anchor in body frame # type: ignore
    jnt_axis: wp.array(dtype=wp.vec3)  # (njnt,) axis in body frame # type: ignore
    jnt_stiffness: wp.array(dtype=float)  # (njnt,) # type: ignore
    jnt_actgravcomp: wp.array(dtype=int)  # (njnt,) 1: gravcomp via actuators, not passive # type: ignore
    jnt_actfrclimited: wp.array(dtype=int)  # (njnt,) # type: ignore
    jnt_actfrcrange: wp.array(dtype=wp.vec2)  # (njnt,) # type: ignore
    qpos0: wp.array(dtype=float)  # (nq,) reference pose of the hinge/slide joints # type: ignore
    qpos_spring: wp.array(dtype=float)  # (nq,) rest pose of the joint springs # type: ignore

    # --------------------------------------------------------------------------------
    # Dofs, and the kinematic-tree partition of them. MuJoCo numbers dofs so that
    # each tree is contiguous, and a tree starts wherever `dof_parentid` is -1.
    dof_jntid: wp.array(dtype=int)  # (nv,) joint owning each dof # type: ignore
    dof_bodyid: wp.array(dtype=int)  # (nv,) # type: ignore
    dof_parentid: wp.array(dtype=int)  # (nv,) parent dof, -1 at a tree root # type: ignore
    dof_armature: wp.array(dtype=float)  # (nv,) # type: ignore
    dof_damping: wp.array(dtype=float)  # (nv,) # type: ignore
    tree_beg: wp.array(dtype=int)  # (ntree+1,) dof range of each tree # type: ignore
    blk_beg: wp.array(dtype=int)  # (ntree+1,) offset of each tree's block # type: ignore
    dof_tree: wp.array(dtype=int)  # (nv,) the tree of each dof # type: ignore
    # Free bodies standing alone: a tree of their own 6 dofs with nothing massive below.
    # Under `implicitfast` each solves its own 6x6 step, its gyroscopic force's derivative
    # included, as MuJoCo's `mjd_freeMhat` has it.
    nfree: wp.int32  # free bodies standing alone
    free_body: wp.array(dtype=int)  # (nfree,) # type: ignore
    # The entries of `D` that can be nonzero under `implicitfast` (`implicit_pairs_np`):
    # pairs (i, j), j = i or an ancestor of i, and for each the actuators whose moment
    # reaches both, `dpair_act[dpair_beg[p] : dpair_beg[p+1]]`.
    npair: wp.int32  # entries (i, j) of D's pattern
    dpair_i: wp.array(dtype=int)  # (npair,) # type: ignore
    dpair_j: wp.array(dtype=int)  # (npair,) # type: ignore
    dpair_beg: wp.array(dtype=int)  # (npair+1,) # type: ignore
    dpair_act: wp.array(dtype=int)  # (total,) # type: ignore

    # --------------------------------------------------------------------------------
    # Large trees. A tree of at least `tile_min` dofs is built one dof per thread and
    # factored with a tile Cholesky, one block of threads per (world, tree), instead
    # of one thread running the whole factorization, which is latency-bound at any
    # batch size. `big_np` is the tile size, every large tree padded to it with an
    # identity block. Small trees -- a free body is 6x6 -- keep the serial kernels.
    nbig: wp.int32  # number of large trees
    big_np: wp.int32  # tile size the large trees are padded to
    nbig_dof: wp.int32  # dofs in large trees
    tree_big: wp.array(dtype=int)  # (ntree,) index among the large trees, -1 if small # type: ignore
    big_tree: wp.array(dtype=int)  # (nbig,) tree id of each large tree # type: ignore
    big_dof: wp.array(dtype=int)  # (nbig_dof,) the dofs of the large trees # type: ignore
    big_dof_tree: wp.array(dtype=int)  # (nbig_dof,) the tree each of them is in # type: ignore

    # --------------------------------------------------------------------------------
    # Actuators. Empty when `nu` is 0 -- the transmission and actuation kernels are
    # then never launched.
    actuator_trntype: wp.array(dtype=int)  # (nu,) # type: ignore
    actuator_dyntype: wp.array(dtype=int)  # (nu,) # type: ignore
    actuator_gaintype: wp.array(dtype=int)  # (nu,) # type: ignore
    actuator_biastype: wp.array(dtype=int)  # (nu,) # type: ignore
    actuator_trnid: wp.array(dtype=int, ndim=2)  # (nu, 2) joint or site, and refsite # type: ignore
    actuator_actadr: wp.array(dtype=int)  # (nu,) # type: ignore
    actuator_actnum: wp.array(dtype=int)  # (nu,) # type: ignore
    actuator_ctrllimited: wp.array(dtype=int)  # (nu,) # type: ignore
    actuator_forcelimited: wp.array(dtype=int)  # (nu,) # type: ignore
    actuator_actlimited: wp.array(dtype=int)  # (nu,) # type: ignore
    actuator_dynprm: wp.array(dtype=float, ndim=2)  # (nu, mjNDYN) # type: ignore
    actuator_gainprm: wp.array(dtype=float, ndim=2)  # (nu, mjNGAIN) # type: ignore
    actuator_biasprm: wp.array(dtype=float, ndim=2)  # (nu, mjNBIAS) # type: ignore
    actuator_ctrlrange: wp.array(dtype=wp.vec2)  # (nu,) # type: ignore
    actuator_forcerange: wp.array(dtype=wp.vec2)  # (nu,) # type: ignore
    actuator_actrange: wp.array(dtype=wp.vec2)  # (nu,) # type: ignore
    actuator_gear: wp.array(dtype=float, ndim=2)  # (nu, 6) # type: ignore
    # (nu, nv), and only for site transmissions: zeroes the columns of the dof
    # chain the actuator's site and refsite share, so an actuator cannot pull on
    # the dofs that move both ends of itself.
    site_dof_mask: wp.array(dtype=float, ndim=2)  # (nu, nv) # type: ignore

    # --------------------------------------------------------------------------------
    # Sites. Only their world poses are needed, and only by actuator transmission.
    site_bodyid: wp.array(dtype=int)  # (nsite,) # type: ignore
    site_pos: wp.array(dtype=wp.vec3)  # (nsite,) # type: ignore
    site_quat: wp.array(dtype=wp.vec4)  # (nsite,) # type: ignore


#######################################################################################
# Data
#######################################################################################


@wp.struct
class RbdData:
    """
    State and everything derived from it. Batched `(B, ...)`.

    `qpos`/`qvel` are the state proper; every other field is a function of them
    and is overwritten each step.

    Built by `make_rbd_context`; a `lodos.sim.Sim` holds it as `drbd`.
    """

    # --------------------------------------------------------------------------------
    # State.
    qpos: wp.array(dtype=float, ndim=2)  # (B, nq) # type: ignore
    qvel: wp.array(dtype=float, ndim=2)  # (B, nv) # type: ignore

    # --------------------------------------------------------------------------------
    # Kinematics. `xquat` is (w, x, y, z); `xmat` is the same rotation as a matrix,
    # kept because `cdof` reads its columns and `transmission` its rows.
    xpos: wp.array(dtype=wp.vec3, ndim=2)  # (B, nbody) # type: ignore
    xquat: wp.array(dtype=wp.vec4, ndim=2)  # (B, nbody) # type: ignore
    xmat: wp.array(dtype=wp.mat33, ndim=2)  # (B, nbody) # type: ignore
    xipos: wp.array(dtype=wp.vec3, ndim=2)  # (B, nbody) inertial frame origin # type: ignore
    ximat: wp.array(dtype=wp.mat33, ndim=2)  # (B, nbody) # type: ignore
    xanchor: wp.array(dtype=wp.vec3, ndim=2)  # (B, njnt) # type: ignore
    xaxis: wp.array(dtype=wp.vec3, ndim=2)  # (B, njnt) # type: ignore
    site_xpos: wp.array(dtype=wp.vec3, ndim=2)  # (B, nsite) # type: ignore
    site_xmat: wp.array(dtype=wp.mat33, ndim=2)  # (B, nsite) # type: ignore

    # --------------------------------------------------------------------------------
    # Frames centered at the subtree center of mass.
    subtree_com: wp.array(dtype=wp.vec3, ndim=2)  # (B, nbody) # type: ignore
    subtree_mass: wp.array(dtype=float, ndim=2)  # (B, nbody) scratch for the above # type: ignore
    cinert: wp.array(dtype=vec10, ndim=2)  # (B, nbody) # type: ignore
    cdof: wp.array(dtype=wp.spatial_vector, ndim=2)  # (B, nv) [angular, linear] # type: ignore

    # --------------------------------------------------------------------------------
    # Mass matrix, one dense n_t x n_t block per kinematic tree rather than a dense
    # (nv, nv) per world. Blocks are indexed `[blk_beg[t] + r * n_t + c, b]`, batch
    # *last* so that a warp's 32 threads read 32 adjacent floats -- `_build_M` and
    # `_factor_M` launch (ntree, B), so `b` varies fastest, the opposite of
    # everything above.
    crb: wp.array(dtype=vec10, ndim=2)  # (B, nbody) subtree inertia sums # type: ignore
    M_blk: wp.array(dtype=float, ndim=2)  # (nblk, B) # type: ignore
    L_blk: wp.array(dtype=float, ndim=2)  # (nblk, B) lower Cholesky factor, M = L L^T # type: ignore
    # The large trees' blocks again, batch first and padded for the tile kernels:
    # entry `[b * nbig + k]` is large tree k of world b. `L_big` is the same factor
    # `L_blk` holds, which every consumer reads from there.
    M_big: wp.array(dtype=float, ndim=3)  # (B * nbig, big_np, big_np) # type: ignore
    L_big: wp.array(dtype=float, ndim=3)  # (B * nbig, big_np, big_np) # type: ignore
    y_big: wp.array(dtype=float, ndim=2)  # (B * nbig, big_np) solve scratch # type: ignore

    # --------------------------------------------------------------------------------
    # Velocity-dependent quantities.
    cvel: wp.array(dtype=wp.spatial_vector, ndim=2)  # (B, nbody) # type: ignore
    cdof_dot: wp.array(dtype=wp.spatial_vector, ndim=2)  # (B, nv) # type: ignore
    cacc: wp.array(dtype=wp.spatial_vector, ndim=2)  # (B, nbody) rne scratch # type: ignore
    cfrc: wp.array(dtype=wp.spatial_vector, ndim=2)  # (B, nbody) rne scratch # type: ignore
    qfrc_bias: wp.array(dtype=float, ndim=2)  # (B, nv) Coriolis, centrifugal, gravity # type: ignore
    qfrc_passive: wp.array(dtype=float, ndim=2)  # (B, nv) springs, dampers, gravcomp # type: ignore
    qfrc_gravcomp: wp.array(dtype=float, ndim=2)  # (B, nv) # type: ignore

    # --------------------------------------------------------------------------------
    # Actuation. `ctrl` and `act` are inputs; the rest is derived.
    ctrl: wp.array(dtype=float, ndim=2)  # (B, nu) # type: ignore
    act: wp.array(dtype=float, ndim=2)  # (B, na) # type: ignore
    act_dot: wp.array(dtype=float, ndim=2)  # (B, na) # type: ignore
    actuator_length: wp.array(dtype=float, ndim=2)  # (B, nu) # type: ignore
    actuator_moment: wp.array(dtype=float, ndim=3)  # (B, nu, nv) # type: ignore
    actuator_velocity: wp.array(dtype=float, ndim=2)  # (B, nu) # type: ignore
    actuator_force: wp.array(dtype=float, ndim=2)  # (B, nu) # type: ignore
    qfrc_actuator: wp.array(dtype=float, ndim=2)  # (B, nv) # type: ignore

    # --------------------------------------------------------------------------------
    # The unconstrained totals. `qfrc_applied` is an input -- the one way to push on
    # the system from outside; there is no `xfrc_applied`.
    qfrc_applied: wp.array(dtype=float, ndim=2)  # (B, nv) # type: ignore
    qfrc_smooth: wp.array(dtype=float, ndim=2)  # (B, nv) # type: ignore
    qacc_smooth: wp.array(dtype=float, ndim=2)  # (B, nv) = M^-1 qfrc_smooth # type: ignore
    # What the integrator consumes: `qacc_smooth` plus whatever the contact solver
    # added. MJWarp's solve writes it, `lodos.cd.driver.bind_state` binding this array
    # into its data.
    qacc: wp.array(dtype=float, ndim=2)  # (B, nv) # type: ignore
    time: wp.array(dtype=float)  # (B,) # type: ignore

    # --------------------------------------------------------------------------------
    # `implicitfast` (`lodos.rbd.implicit`): each actuator's force derivative in its own
    # velocity; under MuJoCo's form, `M - dt D` beside M and its factor, `M qacc`, and the
    # acceleration the step integrates. One-element dummies where unused.
    actuator_dfdv: wp.array(dtype=float, ndim=2)  # (B, nu) # type: ignore
    Mi_blk: wp.array(dtype=float, ndim=2)  # (nblk, B) # type: ignore
    Li_blk: wp.array(dtype=float, ndim=2)  # (nblk, B) # type: ignore
    qacc_ma: wp.array(dtype=float, ndim=2)  # (B, nv) # type: ignore
    qacc_int: wp.array(dtype=float, ndim=2)  # (B, nv) # type: ignore


#######################################################################################
# Model conversion
#######################################################################################


# Everything `lodos/rbd` models. A scene that asks for anything else is rejected
# rather than silently simulated without it -- see `_validate`.
_SUPPORTED = """\
joints (free, ball, slide, hinge), joint springs, dof damping, dof armature,
gravity compensation, actuators with joint/jointinparent/site transmission, and
semi-implicit Euler or implicitfast integration"""


def _disable_flags(mjm: mujoco.MjModel):
    """
    The `mjtDisableBit` flags this port honours, read from the live enum.

    Not written down as numbers: MuJoCo 3.6 split `mjDSBL_PASSIVE`
    (bit 32) into `mjDSBL_SPRING` (32) and `mjDSBL_DAMPER` (64) and shifted
    everything after it, so which bit a flag is depends on the MuJoCo version.

    Called by `make_rbd_context`.
    In: mjm, the MuJoCo model. Out: (spring, damper, gravity, clampctrl, actuation), int each, 1 where the flag is set.
    """
    b = mujoco.mjtDisableBit  # the enum class
    flags = int(mjm.opt.disableflags)  # int: the model's disable bits

    return tuple(
        int(bool(flags & int(bit)))
        for bit in (b.mjDSBL_SPRING, b.mjDSBL_DAMPER, b.mjDSBL_GRAVITY,
                    b.mjDSBL_CLAMPCTRL, b.mjDSBL_ACTUATION)
    )


def _validate(mjm: mujoco.MjModel, constraints: bool = False):
    """
    Rejects any model feature `lodos/rbd` does not implement.

    The failure mode this guards against is silence: MuJoCo will happily hand over
    a model with tendons or joint limits, and a port that ignores those fields
    produces a plausible trajectory that is simply not the model's. Every check
    here corresponds to something `lodos/rbd` never reads.

    `constraints`: joint limits and dof friction loss are solved elsewhere -- by
    MJWarp's constraint solve -- so they are allowed.

    Called by `make_rbd_context`.
    In: mjm, the MuJoCo model; constraints bool, as above. Raises NotImplementedError naming every unsupported feature.
    """
    bad = []  # list of str: the unsupported features found

    if mjm.ntendon:
        bad.append(f"{mjm.ntendon} tendon(s)")
    if mjm.nflex:
        bad.append(f"{mjm.nflex} flex(es)")
    if mjm.neq:
        bad.append(f"{mjm.neq} equality constraint(s)")
    if mjm.nmocap:
        bad.append(f"{mjm.nmocap} mocap body(s)")
    if np.any(mjm.jnt_limited) and not constraints:
        bad.append("joint limits")
    if np.any(mjm.dof_frictionloss) and not constraints:
        bad.append("dof friction loss")
    if mjm.opt.integrator not in (mujoco.mjtIntegrator.mjINT_EULER, mujoco.mjtIntegrator.mjINT_IMPLICITFAST):
        bad.append(f"integrator {mujoco.mjtIntegrator(mjm.opt.integrator).name}")
    if mjm.opt.density or mjm.opt.viscosity or np.any(mjm.opt.wind):
        bad.append("fluid forces")

    if mjm.nu:
        trn_ok = (  # tuple of mujoco.mjtTrn: the transmissions supported
            mujoco.mjtTrn.mjTRN_JOINT,
            mujoco.mjtTrn.mjTRN_JOINTINPARENT,
            mujoco.mjtTrn.mjTRN_SITE,
        )
        for t in set(mjm.actuator_trntype) - {int(x) for x in trn_ok}:  # t: int, an unsupported transmission type
            bad.append(f"actuator transmission {mujoco.mjtTrn(t).name}")
        for t in set(mjm.actuator_dyntype) - {0, 1, 2, 3}:
            bad.append(f"actuator dynamics {mujoco.mjtDyn(t).name}")
        for t in set(mjm.actuator_gaintype) - {0, 1}:
            bad.append(f"actuator gain {mujoco.mjtGain(t).name}")
        for t in set(mjm.actuator_biastype) - {0, 1}:
            bad.append(f"actuator bias {mujoco.mjtBias(t).name}")
        if np.any(mjm.actuator_actearly):
            bad.append("actuator actearly")
        if np.any(mjm.actuator_actnum > 1):
            bad.append("actuators with more than one activation state")

    if bad:
        raise NotImplementedError(
            "lodos.rbd models "
            + _SUPPORTED
            + ".\nThis model also uses: "
            + ", ".join(sorted(set(bad)))
            + "."
        )


def body_levels_np(body_parentid):
    """
    Bodies grouped by depth in the tree, world (depth 0) first.

    MuJoCo numbers a parent before its children, so one forward sweep assigns
    every depth. Returns `(level_id, level_beg)` with the same meaning as
    `mujoco_warp`'s `body_treeadr`.

    Called by `make_rbd_context`.
    In: body_parentid (nbody,) int. Out: level_id (nbody,) int64, the bodies sorted by depth; level_beg (nlevel + 1,)
    int64, `level_id[level_beg[l] : level_beg[l+1]]` the bodies at depth l.
    """
    parent = np.asarray(body_parentid, dtype=int)  # (nbody,) int64
    depth = np.zeros(len(parent), dtype=int)  # (nbody,) int64: each body's depth, the world's 0
    for i in range(1, len(parent)):  # i: int, a body, parents first
        depth[i] = depth[parent[i]] + 1

    level_id = np.argsort(depth, kind="stable").astype(int)  # (nbody,) int64
    level_beg = np.searchsorted(depth[level_id], np.arange(depth.max() + 2))  # (nlevel + 1,) int64

    return level_id, level_beg.astype(int)


def body_children_np(body_parentid):
    """
    Child lists, flattened and grouped by parent.

    This is what lets a reverse tree pass be a *pull* -- each body sums over its
    own children -- instead of a push with `atomic_add`. The pull is race-free and
    so gives the same answer bit for bit on every run, which an atomic sum over an
    unspecified order does not.

    Called by `make_rbd_context`.
    In: body_parentid (nbody,) int. Out: order (nbody - 1,) int64, the bodies but the world, grouped by parent; beg
    (nbody + 1,) int64, `order[beg[i] : beg[i+1]]` the children of body i.
    """
    parent = np.asarray(body_parentid, dtype=int)  # (nbody,) int64
    nbody = len(parent)  # int

    order = np.argsort(parent[1:], kind="stable").astype(int) + 1  # (nbody - 1,) int64: bodies 1.., sorted by parent
    counts = np.bincount(parent[1:], minlength=nbody)  # (nbody,) int64: each body's child count
    beg = np.concatenate([[0], np.cumsum(counts)]).astype(int)  # (nbody + 1,) int64

    return order, beg


def body_subtrees_np(body_parentid):
    """
    Every body's subtree, flattened and grouped by root, itself included.

    This is the mask of the dofs that move a body: dof `v` moves body `i` exactly
    when `i` lies in the subtree of `v`'s body. Only gravity compensation needs it.

    Called by `make_rbd_context`, for a model with gravity compensation.
    In: body_parentid (nbody,) int. Out: ids (sum of subtree sizes,) int64, each subtree's bodies sorted, grouped by
    root; beg (nbody + 1,) int64, `ids[beg[i] : beg[i+1]]` the subtree of body i.
    """
    parent = np.asarray(body_parentid, dtype=int)  # (nbody,) int64
    nbody = len(parent)  # int

    subtree = [[i] for i in range(nbody)]  # nbody lists of int: each body's subtree, filled below
    for i in range(nbody - 1, 0, -1):  # i: int, a body, children before parents
        subtree[parent[i]].extend(subtree[i])

    beg = np.concatenate([[0], np.cumsum([len(x) for x in subtree])]).astype(int)  # (nbody + 1,) int64
    return np.concatenate([sorted(x) for x in subtree]).astype(int), beg


def dof_trees_np(mjm):
    """
    Contiguous dof range of each kinematic tree, and each tree's block offset.

    A tree starts at every dof with no dof parent, which is how `mujoco_warp`
    derives its `qM_tiles` block-diagonal decomposition of the mass matrix.

    Called by `make_rbd_context`.
    In: mjm, the MuJoCo model. Out: tree_beg (ntree + 1,) int64, `tree_beg[t] : tree_beg[t+1]` the dofs of tree t;
    blk_beg (ntree + 1,) int64, each tree's offset into the blocks, `blk_beg[-1]` their total `nblk`.
    """
    nv = int(mjm.nv)  # int
    parent = np.asarray(mjm.dof_parentid)  # (nv,) int32
    tree_beg = np.array([i for i in range(nv) if int(parent[i]) == -1] + [nv], dtype=int)  # (ntree + 1,) int64

    sizes = np.diff(tree_beg)  # (ntree,) int64: each tree's dof count
    blk_beg = np.concatenate([[0], np.cumsum(sizes * sizes)]).astype(int)  # (ntree + 1,) int64
    return tree_beg, blk_beg


def _dof_chain(mjm, body):
    """
    The dofs that move `body`: its weld body's and every ancestor's, walking `dof_parentid` up.

    Called by `implicit_pairs_np`.
    In: mjm, the MuJoCo model; body int. Out: set of int, the dofs.
    """
    b = mjm.body_weldid[body]  # int: the body `body` is welded to, itself if it has joints
    a = mjm.body_dofadr[b] + mjm.body_dofnum[b] - 1 if mjm.body_dofnum[b] else -1  # int: its last dof, -1 if none
    out = set()  # set of int: the dofs
    while a >= 0:
        out.add(int(a))
        a = mjm.dof_parentid[a]
    return out


def implicit_pairs_np(mjm):
    """
    The entries of `implicitfast`'s `D` that can be nonzero (`lodos.rbd.implicit`): pairs
    (i, j), j = i or an ancestor of i -- the mass matrix's pattern -- that some actuator's
    moment reaches both of, and the diagonal of every damped dof. An actuator reaches a
    joint transmission's dofs, or the dofs that move a site transmission's site and
    reference site. Returns `(pair_i, pair_j, pair_beg, pair_act)`, `pair_act[pair_beg[p]
    : pair_beg[p+1]]` the actuators of pair p.

    Called by `make_rbd_context`, under `implicitfast`.
    In: mjm, the MuJoCo model. Out: pair_i and pair_j (npair,) int64, pair_beg (npair + 1,) int64, pair_act (total,)
    int64.
    """
    # dict int -> int: the dof count of a free and of a ball joint; every other joint has 1
    width = {int(mujoco.mjtJoint.mjJNT_FREE): 6, int(mujoco.mjtJoint.mjJNT_BALL): 3}
    support = []  # nu sets of int: the dofs each actuator's moment reaches
    for u in range(mjm.nu):  # u: int, an actuator
        if mjm.actuator_trntype[u] in (TRN_JOINT, TRN_JOINTINPARENT):
            j = mjm.actuator_trnid[u, 0]  # int: the actuator's joint
            a = int(mjm.jnt_dofadr[j])  # int: the joint's first dof
            support.append(set(range(a, a + width.get(int(mjm.jnt_type[j]), 1))))
        else:
            site, ref = mjm.actuator_trnid[u]  # int, int: the site, the reference site (-1 if none)
            dofs = _dof_chain(mjm, mjm.site_bodyid[site])  # set of int: the dofs that move either site
            if ref >= 0:
                dofs |= _dof_chain(mjm, mjm.site_bodyid[ref])
            support.append(dofs)
    pair_i, pair_j, pair_beg, pair_act = [], [], [0], []  # lists of int, returned as arrays
    for i in range(mjm.nv):  # i: int, a dof
        j = i  # int: i, then each of its ancestor dofs
        while j >= 0:
            # list of int: the actuators whose moment reaches both i and j
            acts = [u for u in range(mjm.nu) if i in support[u] and j in support[u]]
            if acts or (i == j and mjm.dof_damping[i] != 0.0):
                pair_i.append(i)
                pair_j.append(int(j))
                pair_act.extend(acts)
                pair_beg.append(len(pair_act))
            j = mjm.dof_parentid[j]
    return (np.asarray(pair_i, int), np.asarray(pair_j, int), np.asarray(pair_beg, int),
            np.asarray(pair_act, int))


def site_dof_mask_np(mjm):
    """
    Which dofs a site-transmission actuator may act on.

    An actuator spanning a site and a reference site must not pull on the dofs
    that move *both* ends of it: those columns are cleared, walking up from each
    end until the two dof chains meet.

    Called by `make_rbd_context`.
    In: mjm, the MuJoCo model. Out: (max(nu, 1), max(nv, 1)) float32, 1 where actuator u may act on dof v, else 0.
    """
    mask = np.ones((max(int(mjm.nu), 1), max(int(mjm.nv), 1)), dtype=np.float32)  # (max(nu, 1), max(nv, 1)) float32
    if not mjm.nu:
        return mask

    for i in np.nonzero(mjm.actuator_trnid[:, 1] != -1)[0]:  # i: int, an actuator with a reference site
        id_, refid = mjm.actuator_trnid[i]  # int, int: the site, the reference site
        b0 = mjm.body_weldid[mjm.site_bodyid[id_]]  # int: the site's weld body
        b1 = mjm.body_weldid[mjm.site_bodyid[refid]]  # int: the reference site's weld body
        a0 = mjm.body_dofadr[b0] + mjm.body_dofnum[b0] - 1  # int: b0's last dof, walked up below
        a1 = mjm.body_dofadr[b1] + mjm.body_dofnum[b1] - 1  # int: b1's last dof, walked up below

        while a0 != a1:
            if a0 < a1:
                a1 = mjm.dof_parentid[a1]
            else:
                a0 = mjm.dof_parentid[a0]
            if a0 == -1 or a1 == -1:
                break

        da = a0 if a0 == a1 else -1  # int: the dof where the two chains meet, -1 if they do not
        while da >= 0:
            mask[i, da] = 0.0
            da = mjm.dof_parentid[da]

    return mask


# Trees at least this many dofs wide take the tiled path; see `RbdModel.nbig`.
TILE_MIN = 13
TILE_SIZES = (16, 32, 48, 64, 96, 128)  # the padded tile sizes: `big_np` is the smallest fitting the widest large tree


def make_rbd_context(mjm: mujoco.MjModel, B: int, device=None, tile_min: int = TILE_MIN, constraints=False,
                     implicit_constraints=False):
    """
    Builds the Warp rigid-body model and state from a `mujoco.MjModel`.

    `qpos` starts at `qpos0` and `qvel` at zero.

    `constraints`: a constraint solver outside this package handles the model's joint
    limits and dof friction loss (MJWarp's), so a model with
    them is accepted rather than refused. This package still reads neither.

    `implicit_constraints`: under `implicitfast` (`mjm.opt.integrator`), the constraint
    solve sees `M - dt D` rather than M (`lodos.rbd.implicit`).

    Called by `lodos.sim.make_sim`.
    In: mjm, the MuJoCo model; B int, the number of worlds; device, the Warp device (None for the default); tile_min
    int, the dofs from which a tree takes the tiled path; constraints and implicit_constraints bool, as above.
    Out: (m, d), the `RbdModel` and the `RbdData`, every array allocated (shapes in the structs' field comments).
    """
    _validate(mjm, constraints)

    m = RbdModel()  # RbdModel
    m.B = int(B)
    m.nq = int(mjm.nq)
    m.nv = int(mjm.nv)
    m.nbody = int(mjm.nbody)
    m.njnt = int(mjm.njnt)
    m.nsite = int(mjm.nsite)
    m.nu = int(mjm.nu)
    m.na = int(mjm.na)
    m.ngravcomp = int(np.count_nonzero(mjm.body_gravcomp))
    m.dt = float(mjm.opt.timestep)
    m.gravity = wp.vec3(*[float(g) for g in mjm.opt.gravity])

    spring, damper, gravity, clampctrl, actuation = _disable_flags(mjm)  # int each, 1 where the flag is set
    m.disable_spring = spring
    m.disable_damper = damper
    m.disable_gravity = gravity
    m.disable_clampctrl = clampctrl
    m.disable_actuation = actuation
    m.implicit = int(mjm.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
    m.implicit_constraints = int(bool(implicit_constraints))

    level_id, level_beg = body_levels_np(mjm.body_parentid)  # (nbody,), (nlevel + 1,) int64
    child_id, child_beg = body_children_np(mjm.body_parentid)  # (nbody - 1,), (nbody + 1,) int64
    tree_beg, blk_beg = dof_trees_np(mjm)  # (ntree + 1,), (ntree + 1,) int64
    m.nlevel = int(len(level_beg) - 1)
    m.ntree = int(len(tree_beg) - 1)
    m.nblk = int(blk_beg[-1])

    # An empty model array (no sites, no actuators) becomes a one-element dummy:
    # a Warp struct field must be a real array, and the kernels that would read it
    # are never launched.
    def _arr(x, dtype, np_dtype, width=0):
        """
        `x` as a device array of `dtype`, flat or in rows of `width`; a one-element dummy if `x` is empty.

        Called by `_i`, `_f`, `_v2` and 4 others, the typed helpers below.
        In: x, array-like; dtype, the Warp dtype; np_dtype, the numpy dtype `x` is cast to first; width int, the row
        width, 0 for flat. Out: wp.array on `device`, (n,) of `dtype` when flat or `dtype` is a vector of `width`,
        else (n, width).
        """
        a = np.asarray(x, dtype=np_dtype)  # numpy, x's shape: the field
        if not a.size:
            a = np.zeros((1, width) if width else (1,), np_dtype)
        else:
            a = a.reshape(-1, width) if width else a.reshape(-1)
        return wp.array(np.ascontiguousarray(a), dtype=dtype, device=device)

    def _i(x):
        """
        `x` as an `(n,)` int32 device array.

        Called by `make_rbd_context`.
        """
        return _arr(x, int, np.int32)

    def _f(x):
        """
        `x` as an `(n,)` float32 device array.

        Called by `make_rbd_context`.
        """
        return _arr(x, float, np.float32)

    def _v2(x):
        """
        `x`, `(n, 2)`, as an `(n,)` wp.vec2 device array.

        Called by `make_rbd_context`.
        """
        return _arr(x, wp.vec2, np.float32, 2)

    def _v3(x):
        """
        `x`, `(n, 3)`, as an `(n,)` wp.vec3 device array.

        Called by `make_rbd_context`.
        """
        return _arr(x, wp.vec3, np.float32, 3)

    def _v4(x):
        """
        `x`, `(n, 4)`, as an `(n,)` wp.vec4 device array.

        Called by `make_rbd_context`.
        """
        return _arr(x, wp.vec4, np.float32, 4)

    def _i2(x, w):
        """
        `x` as an `(n, w)` int32 device array.

        Called by `make_rbd_context`.
        """
        return _arr(x, int, np.int32, w)

    def _f2(x, w):
        """
        `x` as an `(n, w)` float32 device array.

        Called by `make_rbd_context`.
        """
        return _arr(x, float, np.float32, w)

    m.body_parentid = _i(mjm.body_parentid)
    m.body_rootid = _i(mjm.body_rootid)
    m.body_level_beg = _i(level_beg)
    m.body_level_id = _i(level_id)
    m.body_child_beg = _i(child_beg)
    m.body_child_id = _i(child_id)
    m.body_jntadr = _i(mjm.body_jntadr)
    m.body_jntnum = _i(mjm.body_jntnum)
    m.body_dofadr = _i(mjm.body_dofadr)
    m.body_dofnum = _i(mjm.body_dofnum)
    m.body_pos = _v3(mjm.body_pos)
    m.body_quat = _v4(mjm.body_quat)
    m.body_ipos = _v3(mjm.body_ipos)
    m.body_iquat = _v4(mjm.body_iquat)
    m.body_mass = _f(mjm.body_mass)
    m.body_inertia = _v3(mjm.body_inertia)
    m.body_gravcomp = _f(mjm.body_gravcomp)

    if m.ngravcomp:
        subtree_id, subtree_beg = body_subtrees_np(mjm.body_parentid)  # (sum of subtree sizes,), (nbody + 1,) int64
    else:
        subtree_id, subtree_beg = np.zeros(0, int), np.zeros(m.nbody + 1, int)
    m.body_subtree_id = _i(subtree_id)
    m.body_subtree_beg = _i(subtree_beg)

    m.jnt_type = _i(mjm.jnt_type)
    m.jnt_bodyid = _i(mjm.jnt_bodyid)
    m.jnt_qposadr = _i(mjm.jnt_qposadr)
    m.jnt_dofadr = _i(mjm.jnt_dofadr)
    m.jnt_pos = _v3(mjm.jnt_pos)
    m.jnt_axis = _v3(mjm.jnt_axis)
    m.jnt_stiffness = _f(mjm.jnt_stiffness)
    m.jnt_actgravcomp = _i(mjm.jnt_actgravcomp)
    m.jnt_actfrclimited = _i(mjm.jnt_actfrclimited)
    m.jnt_actfrcrange = _v2(mjm.jnt_actfrcrange)
    m.qpos0 = _f(mjm.qpos0)
    m.qpos_spring = _f(mjm.qpos_spring)

    m.dof_jntid = _i(mjm.dof_jntid)
    m.dof_bodyid = _i(mjm.dof_bodyid)
    m.dof_parentid = _i(mjm.dof_parentid)
    m.dof_armature = _f(mjm.dof_armature)
    m.dof_damping = _f(mjm.dof_damping)
    m.tree_beg = _i(tree_beg)
    m.blk_beg = _i(blk_beg)

    sizes = np.diff(tree_beg)  # (ntree,) int64: each tree's dof count
    dof_tree = np.repeat(np.arange(len(sizes)), sizes)  # (nv,) int64: each dof's tree
    m.dof_tree = _i(dof_tree)
    # MJWarp's rule (`io.put_model`'s `body_is_free`): one joint, a free one, alone in its
    # tree, and the subtree's mass the body's own.
    free = [k for k in range(1, mjm.nbody)  # list of int: the free bodies standing alone
            if mjm.body_jntnum[k] == 1 and mjm.jnt_type[mjm.body_jntadr[k]] == mujoco.mjtJoint.mjJNT_FREE
            and sizes[dof_tree[mjm.body_dofadr[k]]] == 6 and mjm.body_subtreemass[k] == mjm.body_mass[k]]
    m.nfree = len(free)
    m.free_body = _i(free)
    # (npair,), (npair,), (npair + 1,), (total,) int64: `implicit_pairs_np`'s, no pairs unless `implicitfast`
    pair_i, pair_j, pair_beg, pair_act = (implicit_pairs_np(mjm) if m.implicit
                                          else (np.zeros(0, int), np.zeros(0, int), np.zeros(1, int), np.zeros(0, int)))
    m.npair = len(pair_i)
    m.dpair_i, m.dpair_j, m.dpair_beg, m.dpair_act = _i(pair_i), _i(pair_j), _i(pair_beg), _i(pair_act)
    big = [t for t in range(m.ntree) if sizes[t] >= tile_min]  # list of int: the large trees
    m.nbig = len(big)
    nmax = int(max(sizes[big])) if big else 0  # int: the widest large tree's dof count
    if nmax > TILE_SIZES[-1]:
        raise NotImplementedError(f"a kinematic tree of {nmax} dofs is wider than the largest tile, {TILE_SIZES[-1]}")
    m.big_np = next((n for n in TILE_SIZES if n >= nmax), TILE_SIZES[0]) if big else 0
    tree_big = -np.ones(max(m.ntree, 1), dtype=int)  # (max(ntree, 1),) int64: index among the large trees, -1 if small
    for k, t in enumerate(big):  # k: int, a large tree's index among them; t: int, its tree
        tree_big[t] = k
    big_dof = [i for t in big for i in range(tree_beg[t], tree_beg[t + 1])]  # nbig_dof ints: the large trees' dofs
    big_dof_tree = [t for t in big for _ in range(tree_beg[t], tree_beg[t + 1])]  # nbig_dof ints: the tree of each
    m.nbig_dof = len(big_dof)
    m.big_dof_tree = _i(big_dof_tree)
    m.tree_big = _i(tree_big)
    m.big_tree = _i(big)
    m.big_dof = _i(big_dof)

    m.actuator_trntype = _i(mjm.actuator_trntype)
    m.actuator_dyntype = _i(mjm.actuator_dyntype)
    m.actuator_gaintype = _i(mjm.actuator_gaintype)
    m.actuator_biastype = _i(mjm.actuator_biastype)
    m.actuator_trnid = _i2(mjm.actuator_trnid, 2)
    m.actuator_actadr = _i(mjm.actuator_actadr)
    m.actuator_actnum = _i(mjm.actuator_actnum)
    m.actuator_ctrllimited = _i(mjm.actuator_ctrllimited)
    m.actuator_forcelimited = _i(mjm.actuator_forcelimited)
    m.actuator_actlimited = _i(mjm.actuator_actlimited)
    m.actuator_dynprm = _f2(mjm.actuator_dynprm, mjm.actuator_dynprm.shape[1])
    m.actuator_gainprm = _f2(mjm.actuator_gainprm, mjm.actuator_gainprm.shape[1])
    m.actuator_biasprm = _f2(mjm.actuator_biasprm, mjm.actuator_biasprm.shape[1])
    m.actuator_ctrlrange = _v2(mjm.actuator_ctrlrange)
    m.actuator_forcerange = _v2(mjm.actuator_forcerange)
    m.actuator_actrange = _v2(mjm.actuator_actrange)
    m.actuator_gear = _f2(mjm.actuator_gear, 6)
    m.site_dof_mask = wp.array(site_dof_mask_np(mjm), dtype=float, device=device)

    m.site_bodyid = _i(mjm.site_bodyid)
    m.site_pos = _v3(mjm.site_pos)
    m.site_quat = _v4(mjm.site_quat)

    d = RbdData()  # RbdData
    nv, nu, na = max(m.nv, 1), max(m.nu, 1), max(m.na, 1)  # int each: the dimensions, at least 1 for the dummies

    d.qpos = wp.array(
        np.tile(np.asarray(mjm.qpos0, dtype=np.float32), (B, 1)),
        dtype=float,
        device=device,
    )
    d.qvel = wp.zeros((B, nv), dtype=float, device=device)

    d.xpos = wp.zeros((B, m.nbody), dtype=wp.vec3, device=device)
    d.xquat = wp.zeros((B, m.nbody), dtype=wp.vec4, device=device)
    d.xmat = wp.zeros((B, m.nbody), dtype=wp.mat33, device=device)
    d.xipos = wp.zeros((B, m.nbody), dtype=wp.vec3, device=device)
    d.ximat = wp.zeros((B, m.nbody), dtype=wp.mat33, device=device)
    d.xanchor = wp.zeros((B, max(m.njnt, 1)), dtype=wp.vec3, device=device)
    d.xaxis = wp.zeros((B, max(m.njnt, 1)), dtype=wp.vec3, device=device)
    d.site_xpos = wp.zeros((B, max(m.nsite, 1)), dtype=wp.vec3, device=device)
    d.site_xmat = wp.zeros((B, max(m.nsite, 1)), dtype=wp.mat33, device=device)

    d.subtree_com = wp.zeros((B, m.nbody), dtype=wp.vec3, device=device)
    d.subtree_mass = wp.zeros((B, m.nbody), dtype=float, device=device)
    d.cinert = wp.zeros((B, m.nbody), dtype=vec10, device=device)
    d.cdof = wp.zeros((B, nv), dtype=wp.spatial_vector, device=device)

    d.crb = wp.zeros((B, m.nbody), dtype=vec10, device=device)
    d.M_blk = wp.zeros((max(m.nblk, 1), B), dtype=float, device=device)
    d.L_blk = wp.zeros((max(m.nblk, 1), B), dtype=float, device=device)
    nb, npad = max(B * m.nbig, 1), max(m.big_np, 1)  # int, int: B * nbig and big_np, at least 1 for the dummies
    d.M_big = wp.zeros((nb, npad, npad), dtype=float, device=device)
    d.L_big = wp.zeros((nb, npad, npad), dtype=float, device=device)
    d.y_big = wp.zeros((nb, npad), dtype=float, device=device)

    d.cvel = wp.zeros((B, m.nbody), dtype=wp.spatial_vector, device=device)
    d.cdof_dot = wp.zeros((B, nv), dtype=wp.spatial_vector, device=device)
    d.cacc = wp.zeros((B, m.nbody), dtype=wp.spatial_vector, device=device)
    d.cfrc = wp.zeros((B, m.nbody), dtype=wp.spatial_vector, device=device)
    d.qfrc_bias = wp.zeros((B, nv), dtype=float, device=device)
    d.qfrc_passive = wp.zeros((B, nv), dtype=float, device=device)
    d.qfrc_gravcomp = wp.zeros((B, nv), dtype=float, device=device)

    d.ctrl = wp.zeros((B, nu), dtype=float, device=device)
    d.act = wp.zeros((B, na), dtype=float, device=device)
    d.act_dot = wp.zeros((B, na), dtype=float, device=device)
    d.actuator_length = wp.zeros((B, nu), dtype=float, device=device)
    d.actuator_moment = wp.zeros((B, nu, nv), dtype=float, device=device)
    d.actuator_velocity = wp.zeros((B, nu), dtype=float, device=device)
    d.actuator_force = wp.zeros((B, nu), dtype=float, device=device)
    d.qfrc_actuator = wp.zeros((B, nv), dtype=float, device=device)

    d.qfrc_applied = wp.zeros((B, nv), dtype=float, device=device)
    d.qfrc_smooth = wp.zeros((B, nv), dtype=float, device=device)
    d.qacc_smooth = wp.zeros((B, nv), dtype=float, device=device)
    d.qacc = wp.zeros((B, nv), dtype=float, device=device)
    d.time = wp.zeros(B, dtype=float, device=device)

    d.actuator_dfdv = wp.zeros((B, nu), dtype=float, device=device)
    mujoco_form = m.implicit and not m.implicit_constraints  # int or bool, true under MuJoCo's form of `implicitfast`
    # int, int, int: the dimensions of Mi_blk, Li_blk, qacc_ma and qacc_int, 1 where unused
    nblk_i, B_i, nv_i = (max(m.nblk, 1), B, nv) if mujoco_form else (1, 1, 1)
    d.Mi_blk = wp.zeros((nblk_i, B_i), dtype=float, device=device)
    d.Li_blk = wp.zeros((nblk_i, B_i), dtype=float, device=device)
    d.qacc_ma = wp.zeros((B_i, nv_i), dtype=float, device=device)
    d.qacc_int = wp.zeros((B_i, nv_i), dtype=float, device=device)

    return m, d
