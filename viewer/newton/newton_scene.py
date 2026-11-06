"""
One of Lodos's scenes in Newton: its model, its state in MuJoCo's layout and back,
and a step with Newton's own collision pipeline under Newton's MuJoCo Warp solver or Kamino.

Nothing here imports lodos. A scene arrives as a *package* (`viewer.newton_link`
writes them): the scene's MJCF, which `ModelBuilder.add_mjcf` imports as it is -- its
primitives and convex hulls collide as Newton collides them -- and, for a scene with Lodos
surfaces, each surface's triangle mesh, which goes on its body as a mesh shape with a mesh
SDF at the voxel size of the surface's own bake. The MJCF's geoms keyed to those surfaces
collide with nothing in the MJCF, so in Newton too the bodies collide through the meshes
alone.

Under MuJoCo Warp the MJCF's MuJoCo settings come through as well (`builder_for`): its
options, the geoms' and joints' contact parameters, its actuators. The masses, centres of
mass and inertias are MuJoCo's, copied from the compiled MJCF onto Newton's bodies: the two
compute a mesh's inertia differently, and a comparison of solvers should not be one of
inertias.

**States.** Every state crosses in MuJoCo's layout (`qpos`, `qvel`), which the viewer, the
state sets and the tuning tools all use. Newton's coordinates differ for a free joint: its
position is the joint's relative transform, `xyzw`, its velocity the child's centre-of-mass
twist in the parent frame; MuJoCo's is the body's world pose, `wxyz`, the origin's world
velocity and the angular velocity in the body frame. `Mapping` converts both ways exactly as
`SolverMuJoCo`'s own `convert_*_coords_*` kernels do, per world, for every joint type the
scenes use: free, ball, and hinges and slides, also several on one body.
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path

import mujoco
import numpy as np
import warp as wp

import newton

SOLVERS = ("mjwarp", "kamino")
# The viewer's Kamino options, flat in `options`: these of PADMM's (`cfg.padmm`), and `preconditioning`
# (`cfg.dynamics`) and `gamma`, the contacts' Baumgarte gain (`cfg.constraints`) beside them.
KAMINO_PADMM = ("primal_tolerance", "dual_tolerance", "compl_tolerance", "max_iterations", "rho_0", "use_acceleration",
                "warmstart_mode", "contact_warmstart_method")
SDF_CACHE = Path.home() / ".cache" / "lodos_newton_sdf"


#######################################################################################
# Quaternions and transforms, xyzw as Warp and Newton store them
#######################################################################################


def qmul(a, b):
    x1, y1, z1, w1 = np.moveaxis(a, -1, 0)
    x2, y2, z2, w2 = np.moveaxis(b, -1, 0)
    return np.stack([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                     w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2], -1)


def qinv(q):
    return q * np.array([-1.0, -1.0, -1.0, 1.0])


def qrot(q, v):
    """Rotate `v` by the unit quaternion `q` (xyzw)."""
    u, w = q[..., :3], q[..., 3:]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def tmul(a, b):
    """Compose transforms `(p, q)`: `a * b`."""
    return a[0] + qrot(a[1], b[0]), qmul(a[1], b[1])


def tinv(a):
    qi = qinv(a[1])
    return -qrot(qi, a[0]), qi


def wxyz_to_xyzw(q):
    return np.concatenate([q[..., 1:], q[..., :1]], -1)


def xyzw_to_wxyz(q):
    return np.concatenate([q[..., 3:], q[..., :3]], -1)


#######################################################################################
# The model
#######################################################################################


def name_of(label):
    """An MJCF name from a Newton label: its last path component."""
    return str(label).rsplit("/", 1)[-1]


def newton_name(name):
    """An MJCF name as Newton's importer labels it (`sanitize_name`): every "-" a "_"."""
    return name.replace("-", "_")


def mj_ids(mjm, objtype):
    """
    `{name: id}` of MuJoCo's bodies, joints or geoms, by the names Newton labels them with
    (`newton_name`): the Dexonomy scenes' (`hand-rh_palm`, ...) are not Newton's.
    """
    n = {mujoco.mjtObj.mjOBJ_BODY: mjm.nbody, mujoco.mjtObj.mjOBJ_JOINT: mjm.njnt,
         mujoco.mjtObj.mjOBJ_GEOM: mjm.ngeom}[objtype]
    names = (mujoco.mj_id2name(mjm, objtype, i) for i in range(n))
    return {newton_name(s): i for i, s in enumerate(names) if s is not None}


def length_scale(mjm):
    """The scene's typical body size (m): the median over moving bodies of their geoms' bounding diameter."""
    spans = [2.0 * float(mjm.geom_rbound[mjm.geom_bodyid == k].max()) for k in range(1, mjm.nbody)
             if (mjm.geom_bodyid == k).any() and int(mjm.body_rootid[k]) > 0 and mjm.body_jntnum[mjm.body_rootid[k]] > 0]
    return float(np.median(spans)) if spans else 1.0


def gap_of(mjm, options=None):
    """
    The contact gap every colliding shape gets but one whose geom has a gap of its own in the
    MJCF (`world_builder`): `options["gap"]`, else 1 % of the length scale. Newton's default,
    0.1 m, hands contacts to every pair of shapes within 10 cm: 5,185 within one SHARPA hand,
    where MuJoCo (margin 0) has a handful.
    """
    o = options or {}
    return float(o["gap"]) if o.get("gap") is not None else 0.01 * length_scale(mjm)


def builder_for(solver):
    """
    A `ModelBuilder` with the solver's custom attributes. SolverMuJoCo's carry the MJCF's
    MuJoCo settings through `add_mjcf`: its `<option>`s (cone, impratio, integrator,
    iterations, tolerance), each geom's solref, solimp, condim and priority, the joints'
    armature, damping and limits' softness, and its actuators. Without them Newton runs
    its own defaults (pyramidal cones, implicitfast, impratio 1, default softness).
    """
    b = newton.ModelBuilder()
    (newton.solvers.SolverKamino if solver == "kamino" else newton.solvers.SolverMuJoCo).register_custom_attributes(b)
    return b


def world_builder(pkg, solver, options=None):
    """
    One world of the scene, as a `ModelBuilder`: MuJoCo's inertias, the hulls as convex
    hulls, every colliding shape's gap, the surfaces' SDF meshes, and under Kamino the
    joint servos as Newton's joint drives (under MuJoCo Warp the MJCF's actuators come
    through). Returns the builder, MuJoCo's model of the scene, and the drives within the
    world, `[(dof, actuator)]` (`_drives`).
    """
    b = builder_for(solver)
    b.add_mjcf(pkg["xml"])
    mjm = mujoco.MjModel.from_xml_string(pkg["xml"])
    # MuJoCo's gravity, which the importer leaves at Newton's default: none where the MJCF
    # disables it (the Dexonomy grasps, whose hand and toy are weightless). Each world takes it.
    off = int(mjm.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_GRAVITY)
    b.gravity = wp.vec3(0.0, 0.0, 0.0) if off else wp.vec3(*map(float, mjm.opt.gravity))
    gap = gap_of(mjm, options)
    geoms, bodies = mj_ids(mjm, mujoco.mjtObj.mjOBJ_GEOM), mj_ids(mjm, mujoco.mjtObj.mjOBJ_BODY)
    # A geom's own gap in the MJCF, where it has one (the NIST demo's, `nist.newton_xml`), else `gap`.
    gap_at = lambda g: float(mjm.geom_gap[g]) if g >= 0 and mjm.geom_gap[g] > 0 else gap  # noqa: E731
    # MuJoCo collides a mesh geom as its convex hull, and the polytope scenes' pieces are
    # convex: Newton's importer makes them triangle meshes, which collide as such (tens of
    # thousands of contacts on two hands, and a run that diverges). They become its convex hulls.
    collide = int(newton.ShapeFlags.COLLIDE_SHAPES)
    for i in range(b.shape_count):
        if int(b.shape_flags[i]) & collide:
            b.shape_gap[i] = gap_at(geoms.get(name_of(b.shape_label[i]), -1))
            if int(b.shape_type[i]) == int(newton.GeoType.MESH):
                b.shape_type[i] = newton.GeoType.CONVEX_MESH
    _static_planes_to_world(b, mjm)
    targets = _drives(b, mjm, drives=solver == "kamino")
    geom_of_shape, contact = {}, {}
    for i, label in enumerate(b.shape_label):
        g = geoms.get(name_of(label), -1)
        if g >= 0:
            geom_of_shape[i] = g
            if g in pkg.get("planes", {}):  # a Lodos plane's contact parameters, its pairs' (`scene_package`)
                contact[i] = pkg["planes"][g]
                b.shape_material_mu[i] = float(contact[i]["mu"])
    # MuJoCo's inertias on Newton's bodies (`body_inertia` is about the centre of mass, in
    # the body frame; MuJoCo's principal axes are `body_iquat`).
    for i, label in enumerate(b.body_label):
        k = bodies.get(name_of(label), -1)
        if k < 0:
            continue
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, mjm.body_iquat[k])
        R = R.reshape(3, 3)
        mass, inertia = float(mjm.body_mass[k]), R @ np.diag(mjm.body_inertia[k]) @ R.T
        if solver == "kamino" and mass <= 0.0 and int(mjm.body_jntnum[k]) > 0:
            # Kamino refuses a moving body without mass, which MuJoCo takes in a chain (the
            # grasp's wrist): a gram, and the inertia of a centimetre.
            mass, inertia = 1e-3, np.eye(3) * 1e-8
        b.body_mass[i] = mass
        b.body_inv_mass[i] = 1.0 / mass if mass > 0 else 0.0
        b.body_com[i] = wp.vec3(*map(float, mjm.body_ipos[k]))
        b.body_inertia[i] = wp.mat33(*map(float, inertia.reshape(-1)))
        b.body_inv_inertia[i] = wp.mat33(*map(float, np.linalg.inv(inertia).reshape(-1))) if mass > 0 else wp.mat33()
    # The Lodos surfaces' meshes, each with a mesh SDF: at its bake's voxel size, or at the
    # settings the scene gives Newton for it (`sdf`: Isaac Lab's for the NIST parts).
    names = [name_of(x) for x in b.body_label]
    imported, keyed = b.shape_count, set()
    for s in pkg.get("sdf_shapes", []):
        body = names.index(newton_name(s["body"]))
        v = np.asarray(s["vertices"], np.float32)
        f = np.asarray(s["faces"], np.int32).reshape(-1)
        g = int(s["geom_id"]) if s.get("geom_id") is not None else -1
        material = {k: float(s[k]) for k in ("ke", "kd") if k in s}  # Newton's own contact stiffness and damping
        cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=float(s["mu"]), gap=gap_at(g), **material)
        mesh = newton.Mesh(v, f, compute_inertia=False)
        if "sdf" not in s:
            band = max(16.0 * float(s["voxel"]), 0.02 * float(np.ptp(v, axis=0).max()))
            mesh.build_sdf(target_voxel_size=float(s["voxel"]), narrow_band_range=(-band, band), cache_dir=str(SDF_CACHE))
        b.add_shape_mesh(body, mesh=mesh, cfg=cfg, label=f"{b.body_label[body]}/lodos_surface")
        if "sdf" in s:
            # Newton's own settings, as its USD importer takes Isaac Lab's parts (`newton:sdfMaxResolution`): written
            # on the shape for `finalize` to build, not built on the mesh (`Mesh.build_sdf`), which collides apart.
            band, i = float(s["sdf"]["band"]), b.shape_count - 1
            b.shape_sdf_max_resolution[i] = int(s["sdf"]["max_resolution"])
            b.shape_sdf_narrow_band_range[i] = (-band, band)
        if s.get("geom_id") is not None:
            geom_of_shape[b.shape_count - 1] = int(s["geom_id"])  # the geom the surface is keyed to
            keyed.add(int(s["geom_id"]))
        contact[b.shape_count - 1] = s
    # The geoms those meshes stand for collide no more, as `lodos.sim.make_sim` switches them off for
    # MuJoCo Warp: a nut's mesh would otherwise collide as its convex hull too, its hole filled.
    for i in range(imported):
        if geom_of_shape.get(i) in keyed:
            b.shape_flags[i] = int(b.shape_flags[i]) & ~collide
    _mujoco_params(b, mjm, geom_of_shape, contact, solref=(options or {}).get("solref"))
    return b, mjm, targets


def _static_planes_to_world(b, mjm):
    """
    Planes on bodies MuJoCo welds to the world (no joint between them and it: the walls,
    the incline's slope) onto the world itself, where SolverMuJoCo takes a plane only.
    Newton's importer keeps such a body as a body of its own on a fixed joint; this moves
    its planes as Newton's `collapse_fixed_joints` would, the body's pose composed into
    each plane's.
    """
    names = [name_of(x) for x in b.body_label]
    bodies = mj_ids(mjm, mujoco.mjtObj.mjOBJ_BODY)
    for i in range(b.shape_count):
        k = int(b.shape_body[i])
        if k < 0 or int(b.shape_type[i]) != int(newton.GeoType.PLANE):
            continue
        m = bodies.get(names[k], -1)
        if m < 0 or int(mjm.body_weldid[m]) != 0:
            continue
        b.shape_transform[i] = wp.transform(*b.body_q[k]) * wp.transform(*b.shape_transform[i])
        b.shape_body[i] = -1
        b.body_shapes[k].remove(i)
        b.body_shapes[-1].append(i)


SOLREF_RAW = 1  # Newton's solref mode: handed to MuJoCo as it is
UNLIMITED = 1e10  # Newton's joint limit for none (`MAXVAL`)


def _joint_axis(b, mjm, j):
    """Newton's joint and axis for MuJoCo's hinge or slide joint `j`: by the joint's name, else by its body and axis."""
    names = [name_of(x) for x in b.joint_label]
    jname = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, j)
    if jname is not None and newton_name(jname) in names:
        return names.index(newton_name(jname)), 0
    child = [name_of(b.body_label[c]) for c in b.joint_child]
    bname = newton_name(mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_BODY, int(mjm.jnt_bodyid[j])))
    return child.index(bname), j - int(mjm.body_jntadr[mjm.jnt_bodyid[j]])


def _dof_of(b, mjm, j):
    """Newton's DOF for MuJoCo's hinge or slide joint `j`."""
    k, axis = _joint_axis(b, mjm, j)
    return int(b.joint_qd_start[k]) + axis


def _target_of(b, mjm, j):
    """
    The entry of `Control.joint_target_q` for MuJoCo's hinge or slide joint `j`: its coordinate
    where Newton lays targets out by coordinate (`newton.use_coord_layout_targets`), else its DOF.
    The two differ for every joint after a free joint, which has 7 coordinates and 6 DOFs.
    """
    k, axis = _joint_axis(b, mjm, j)
    start = b.joint_q_start if getattr(newton, "use_coord_layout_targets", False) else b.joint_qd_start
    return int(start[k]) + axis


def _mujoco_params(b, mjm, geom_of_shape, contact=None, solref=None):
    """
    The MJCF's contact parameters and joint limits as MuJoCo compiles them, over what Newton
    1.7's importer read. It fills `solref="0.01"` to (0.01, 0), where MuJoCo fills the
    missing damping ratio with 1: the SHARPA fingers' contacts would be undamped. And it limits
    a joint with a `range` and `limited="false"`: every joint of the SHARPA hands without
    their limits (`sharpa.scene_xml(jlimit=False)`). The contact parameters are SolverMuJoCo's attributes, for the geoms in
    `geom_of_shape` (Newton's shape: MuJoCo's geom), a Lodos surface's solref and solimp its
    pairs' where `contact` has them (`newton_link.scene_package`); the limits are Newton's,
    for both solvers. `solref`, where given (`options["solref"]`), every shape's.
    """
    ca, contact, every = b.custom_attributes, contact or {}, solref
    if "mujoco:solref" in ca:
        for i, g in geom_of_shape.items():
            own = contact.get(i, {})
            solref, solimp = own.get("solref", mjm.geom_solref[g]), own.get("solimp", mjm.geom_solimp[g])
            solref = solref if every is None else every
            ca["mujoco:solref"].values[i] = ca["mujoco:solref"].dtype(*map(float, solref))
            ca["mujoco:solref_mode"].values[i] = SOLREF_RAW
            ca["mujoco:geom_solimp"].values[i] = ca["mujoco:geom_solimp"].dtype(*map(float, solimp))
            ca["mujoco:condim"].values[i] = int(mjm.geom_condim[g])
            ca["mujoco:geom_priority"].values[i] = int(mjm.geom_priority[g])
            ca["mujoco:geom_solmix"].values[i] = float(mjm.geom_solmix[g])
    hinge_slide = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
    for j in range(int(mjm.njnt)):
        if int(mjm.jnt_type[j]) in hinge_slide and not mjm.jnt_limited[j]:
            dof = _dof_of(b, mjm, j)
            b.joint_limit_lower[dof], b.joint_limit_upper[dof] = -UNLIMITED, UNLIMITED


def _drives(b, mjm, drives=True):
    """
    The MJCF's position servos on joints, as `[(target, actuator)]`: the actuator's control is the
    joint's target, `Control.joint_target_q[target]` (`_target_of`, `Sim.set_ctrl`). With
    `drives` (Kamino), each becomes Newton's position drive with MuJoCo's compiled gains:
    stiffness `kp` (gainprm[0]), damping the compiled `kv` (-biasprm[2]; the SHARPA hands'
    `0.9` there is a damping ratio until MuJoCo compiles it). Newton's importer makes drives
    of `<position>` actuators only, and the hands' are `<general>`; under MuJoCo Warp those
    come through as MuJoCo's actuators, and the targets reach the `<position>` ones.
    """
    out = []
    for a in range(int(mjm.nu)):
        if int(mjm.actuator_trntype[a]) != int(mujoco.mjtTrn.mjTRN_JOINT):
            continue
        kp, bias = float(mjm.actuator_gainprm[a, 0]), mjm.actuator_biasprm[a]
        affine = (int(mjm.actuator_gaintype[a]) == int(mujoco.mjtGain.mjGAIN_FIXED)
                  and int(mjm.actuator_biastype[a]) == int(mujoco.mjtBias.mjBIAS_AFFINE))
        if not affine or abs(float(bias[1]) + kp) > 1e-9 * max(kp, 1.0):
            continue  # not a position servo: MuJoCo reads `biasprm` only for an affine bias
        j = int(mjm.actuator_trnid[a, 0])
        if drives:
            dof = _dof_of(b, mjm, j)
            b.joint_target_ke[dof] = kp
            b.joint_target_kd[dof] = -float(bias[2])
            b.joint_target_mode[dof] = int(newton.JointTargetMode.POSITION)
        out.append((_target_of(b, mjm, j), a))
    return out


def site_forces(b, mjm):
    """
    The MJCF's forces on sites, which Kamino has no actuators for (under MuJoCo Warp they are
    MuJoCo's own): `[(body, force, torque, lever, actuator)]`, Newton's body within the world, the
    world force and torque a unit of the actuator's force applies -- its gear, in the frame of a
    reference site on the world body -- and the site's offset from the body's centre of mass, in the
    body's frame. Each must be a plain force (the control times the gain, no bias, no dynamics) in a
    frame fixed in the world: any other site transmission is refused.
    """
    out = []
    bodies = {name_of(x): i for i, x in enumerate(b.body_label)}
    for a in range(int(mjm.nu)):
        if int(mjm.actuator_trntype[a]) != int(mujoco.mjtTrn.mjTRN_SITE):
            continue
        site, ref = (int(x) for x in mjm.actuator_trnid[a])
        plain = (int(mjm.actuator_gaintype[a]) == int(mujoco.mjtGain.mjGAIN_FIXED)
                 and int(mjm.actuator_biastype[a]) == int(mujoco.mjtBias.mjBIAS_NONE)
                 and int(mjm.actuator_dyntype[a]) == int(mujoco.mjtDyn.mjDYN_NONE))
        if not plain or ref < 0 or int(mjm.site_bodyid[ref]) != 0:
            raise ValueError(f"actuator {mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_ACTUATOR, a)!r}: Kamino takes a "
                             "site's actuator only as a plain force in a frame fixed in the world (a reference site "
                             "on the world body)")
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, mjm.site_quat[ref])
        R, gear = R.reshape(3, 3), np.asarray(mjm.actuator_gear[a], np.float64)
        k = int(mjm.site_bodyid[site])
        body = bodies[newton_name(mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_BODY, k))]
        out.append((body, R @ gear[:3], R @ gear[3:], np.asarray(mjm.site_pos[site]) - mjm.body_ipos[k], a))
    return out


def gravity_compensation(b, mjm):
    """
    The MJCF's gravity compensation (`gravcomp`), which Kamino does not model either (MuJoCo Warp
    does): `[(body, force)]`, Newton's body within the world and the force at its centre of mass
    that cancels `gravcomp` of its weight, as MuJoCo's passive force does. Compensation through
    actuators (`actuatorgravcomp`), which their force ranges limit, is refused.
    """
    if np.any(mjm.jnt_actgravcomp):
        raise ValueError("Kamino takes gravity compensation only as a passive force, not through actuators")
    if int(mjm.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_GRAVITY):
        return []
    bodies = {name_of(x): i for i, x in enumerate(b.body_label)}
    g = np.asarray(mjm.opt.gravity, np.float64)
    return [(bodies[newton_name(mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_BODY, k))],
             -float(mjm.body_gravcomp[k]) * float(mjm.body_mass[k]) * g)
            for k in range(1, int(mjm.nbody)) if mjm.body_gravcomp[k] and mjm.body_mass[k] > 0]


def actuator_force(mjm, a, u):
    """A plain force actuator's force at control `u`: within its control range, times its gain, within its force range."""
    if mjm.actuator_ctrllimited[a]:
        u = np.clip(u, *mjm.actuator_ctrlrange[a])
    f = float(mjm.actuator_gainprm[a, 0]) * float(u)
    return float(np.clip(f, *mjm.actuator_forcerange[a])) if mjm.actuator_forcelimited[a] else f


@wp.kernel
def _track(contacts: wp.array(dtype=wp.int32), tri: wp.array(dtype=wp.int32), nefc: wp.array(dtype=wp.int32),  # type: ignore
           red_count: wp.array(dtype=wp.int32), red_failed: wp.array(dtype=wp.int32),  # type: ignore
           red_dropped: wp.array(dtype=wp.int32), pairs: wp.array(dtype=wp.int32),  # type: ignore
           peak: wp.array(dtype=wp.int32)):  # type: ignore
    w = wp.tid()
    if w == 0:
        wp.atomic_max(peak, 0, contacts[0])
        wp.atomic_max(peak, 1, tri[0])
        wp.atomic_max(peak, 3, red_count[0] + red_dropped[0])
        wp.atomic_max(peak, 4, red_failed[0] + red_dropped[0])
        wp.atomic_max(peak, 5, pairs[0])
    wp.atomic_max(peak, 2, nefc[w])


@wp.kernel
def _count_by_world(count: wp.array(dtype=wp.int32), shape0: wp.array(dtype=wp.int32),  # type: ignore
                    shape1: wp.array(dtype=wp.int32), shape_world: wp.array(dtype=wp.int32),
                    n: wp.array(dtype=wp.int32)):  # type: ignore
    i = wp.tid()
    if i < count[0] and shape0[i] >= 0:
        w = shape_world[shape0[i]]
        if w < 0:
            w = shape_world[shape1[i]]
        if w >= 0:
            wp.atomic_add(n, w, 1)


@wp.kernel
def _track_world(n: wp.array(dtype=wp.int32), peak: wp.array(dtype=wp.int32)):  # type: ignore
    w = wp.tid()
    wp.atomic_max(peak, 0, n[w])
    n[w] = 0


@wp.kernel
def _point_forces(body_q: wp.array(dtype=wp.transform), body: wp.array(dtype=int), force: wp.array(dtype=wp.vec3),
                  torque: wp.array(dtype=wp.vec3), lever: wp.array(dtype=wp.vec3), mag: wp.array2d(dtype=float),
                  per_world: int, body_f: wp.array(dtype=wp.spatial_vector)):
    """
    Forces at points of bodies, `mag` times `force` and `torque`, as Newton's wrenches on them (`state.body_f`):
    in the world, about the centre of mass, from which `lever` places the point in the body's frame.
    """
    w, k = wp.tid()
    i = body[k] + w * per_world
    f = force[k] * mag[w, k]
    r = wp.quat_rotate(wp.transform_get_rotation(body_q[i]), lever[k])
    wp.atomic_add(body_f, i, wp.spatial_vector(f, torque[k] * mag[w, k] + wp.cross(r, f)))


#######################################################################################
# MuJoCo's layout and Newton's
#######################################################################################


@dataclasses.dataclass
class Mapping:
    """Per joint of world 0: what `SolverMuJoCo`'s coordinate kernels read, and where MuJoCo keeps it."""

    jtype: np.ndarray
    q_start: np.ndarray
    qd_start: np.ndarray
    naxes: np.ndarray
    X_p: tuple
    X_c: tuple
    com: np.ndarray  # the child's centre of mass, in its body frame
    mj_q: np.ndarray  # -1: no MuJoCo coordinates (a fixed joint)
    mj_qd: np.ndarray
    nq: int  # Newton's per world
    nqd: int
    mj_nq: int
    mj_nv: int

    @classmethod
    def of(cls, model, mjm):
        n = model.joint_count // model.world_count
        jt = model.joint_type.numpy()[:n]
        qs = model.joint_q_start.numpy()
        qds = model.joint_qd_start.numpy()
        dim = model.joint_dof_dim.numpy()[:n]
        Xp = model.joint_X_p.numpy()[:n]
        Xc = model.joint_X_c.numpy()[:n]
        child = model.joint_child.numpy()[:n]
        com = model.body_com.numpy()[child]
        labels = model.body_label
        bodies = mj_ids(mjm, mujoco.mjtObj.mjOBJ_BODY)
        mj_q = np.full(n, -1)
        mj_qd = np.full(n, -1)
        for j in range(n):
            k = bodies.get(name_of(labels[child[j]]), -1)
            if k >= 0 and int(mjm.body_jntnum[k]) > 0:
                jid = int(mjm.body_jntadr[k])
                mj_q[j], mj_qd[j] = int(mjm.jnt_qposadr[jid]), int(mjm.jnt_dofadr[jid])
        q_per = int(qs[n]) if len(qs) > n else int(model.joint_coord_count // model.world_count)
        qd_per = int(qds[n]) if len(qds) > n else int(model.joint_dof_count // model.world_count)
        return cls(jt, qs[:n], qds[:n], dim[:, 0] + dim[:, 1], (Xp[:, :3], Xp[:, 3:]), (Xc[:, :3], Xc[:, 3:]), com,
                   mj_q, mj_qd, q_per, qd_per, int(mjm.nq), int(mjm.nv))

    def to_newton(self, qpos, qvel):
        """`(B, nq)`, `(B, nv)` in MuJoCo's layout -> Newton's `(B * nq_n,)`, `(B * nqd_n,)`."""
        qpos, qvel = np.atleast_2d(qpos).astype(np.float64), np.atleast_2d(qvel).astype(np.float64)
        B = qpos.shape[0]
        jq = np.zeros((B, self.nq))
        jqd = np.zeros((B, self.nqd))
        free, ball = int(newton.JointType.FREE), int(newton.JointType.BALL)
        for j in range(len(self.jtype)):
            qi, di = self.mj_q[j], self.mj_qd[j]
            if qi < 0:
                continue
            wq, wd = self.q_start[j], self.qd_start[j]
            if self.jtype[j] == free:
                pos, rot = qpos[:, qi:qi + 3], wxyz_to_xyzw(qpos[:, qi + 3:qi + 7])
                Xp = (self.X_p[0][j], self.X_p[1][j])
                Xc = (self.X_c[0][j], self.X_c[1][j])
                p, q = tmul(tmul(tinv(Xp), (pos, rot)), Xc)
                jq[:, wq:wq + 3], jq[:, wq + 3:wq + 7] = p, q
                w_world = qrot(rot, qvel[:, di + 3:di + 6])
                v_com = qvel[:, di:di + 3] + np.cross(w_world, qrot(rot, self.com[j]))
                jqd[:, wd:wd + 3] = qrot(qinv(Xp[1]), v_com)
                jqd[:, wd + 3:wd + 6] = qrot(qinv(Xp[1]), w_world)
            elif self.jtype[j] == ball:
                q_cj = self.X_c[1][j]
                q_mj = wxyz_to_xyzw(qpos[:, qi:qi + 4])
                to_anchor = qmul(qinv(q_cj), q_mj)
                jq[:, wq:wq + 4] = qmul(to_anchor, q_cj)
                jqd[:, wd:wd + 3] = qrot(to_anchor, qvel[:, di:di + 3])
            else:
                a = int(self.naxes[j])
                jq[:, wq:wq + a] = qpos[:, qi:qi + a]
                jqd[:, wd:wd + a] = qvel[:, di:di + a]
        return jq.reshape(-1), jqd.reshape(-1)

    def to_mujoco(self, joint_q, joint_qd, B):
        """Newton's coordinates for `B` worlds -> `(B, nq)`, `(B, nv)` in MuJoCo's layout."""
        jq = np.asarray(joint_q, np.float64).reshape(B, self.nq)
        jqd = np.asarray(joint_qd, np.float64).reshape(B, self.nqd)
        qpos = np.zeros((B, self.mj_nq))
        qvel = np.zeros((B, self.mj_nv))
        free, ball = int(newton.JointType.FREE), int(newton.JointType.BALL)
        for j in range(len(self.jtype)):
            qi, di = self.mj_q[j], self.mj_qd[j]
            if qi < 0:
                continue
            wq, wd = self.q_start[j], self.qd_start[j]
            if self.jtype[j] == free:
                Xp = (self.X_p[0][j], self.X_p[1][j])
                Xc = (self.X_c[0][j], self.X_c[1][j])
                pos, rot = tmul(tmul(Xp, (jq[:, wq:wq + 3], jq[:, wq + 3:wq + 7])), tinv(Xc))
                qpos[:, qi:qi + 3], qpos[:, qi + 3:qi + 7] = pos, xyzw_to_wxyz(rot)
                v_com = qrot(Xp[1], jqd[:, wd:wd + 3])
                w_world = qrot(Xp[1], jqd[:, wd + 3:wd + 6])
                qvel[:, di:di + 3] = v_com - np.cross(w_world, qrot(rot, self.com[j]))
                qvel[:, di + 3:di + 6] = qrot(qinv(rot), w_world)
            elif self.jtype[j] == ball:
                q_cj = self.X_c[1][j]
                r = jq[:, wq:wq + 4]
                qpos[:, qi:qi + 4] = xyzw_to_wxyz(qmul(qmul(q_cj, r), qinv(q_cj)))
                qvel[:, di:di + 3] = qrot(qmul(q_cj, qinv(r)), jqd[:, wd:wd + 3])
            else:
                a = int(self.naxes[j])
                qpos[:, qi:qi + a] = jq[:, wq:wq + a]
                qvel[:, di:di + a] = jqd[:, wd:wd + a]
        return qpos, qvel


#######################################################################################
# A simulation
#######################################################################################


@dataclasses.dataclass
class Caps:
    """
    Newton's capacities. `contacts_per_world` sizes the collision pipeline's buffer (times
    the worlds) and Kamino's (`max_contacts_per_world`), and MuJoCo Warp's per world
    (`nconmax`); `None` measures it from the states the simulation starts from. Kamino's own
    estimate, from the geometry alone, overflows 32-bit sizes on the hand piles.
    `triangle_pairs_per_world` sizes the pipeline's mesh triangle pairs and the table that
    reduces their contacts (`triangle_pairs`): Newton's default is a batch total, a million,
    which twenty armadillos fill to 99 % at 1,024 worlds; past it, failed inserts probe the
    whole table and a step takes minutes. `njmax`, MuJoCo Warp's constraint rows per world,
    defaults to what the contacts need (`constraint_rows`). `shape_pairs_per_world`, under a
    broad phase that finds its pairs (`options["broad_phase"]`, "sap" or "nxn"), caps their
    candidate shape pairs (`shape_pairs`), which size the narrow phase's work buffers too;
    None takes Newton's bound, every pair of a world's shapes. Newton's default broad phase,
    the explicit pairs, sizes those buffers for every pair the collision filters allow, in
    every world: 9.8 GB at 4,096 worlds of five hands.
    """

    contacts_per_world: int | None = None
    triangle_pairs_per_world: int = 4096
    njmax: int | None = None
    shape_pairs_per_world: int | None = None

    def triangle_pairs(self, B):
        """The pipeline's `max_triangle_pairs` for `B` worlds: at least Newton's default."""
        return max(1_000_000, int(self.triangle_pairs_per_world) * int(B))

    def shape_pairs(self, B):
        """The pipeline's `shape_pairs_max` for `B` worlds; None for Newton's bound."""
        return None if self.shape_pairs_per_world is None else int(self.shape_pairs_per_world) * int(B)


PROBE_WORLDS = 64


def constraint_rows(mjm, contacts):
    """
    MuJoCo Warp's constraint rows per world (`njmax`) for `contacts` contacts: a pyramid's
    2 (condim - 1) rows for each (an elliptic cone needs fewer), a row for every joint and
    tendon limit and friction loss, and 6 for every equality. Newton's default sizes the rows
    from its own estimate of the contacts rather than the measured count: 23 GB of constraint
    Jacobian at a thousand worlds of the five armadillos.
    """
    condim = max(3, int(mjm.geom_condim.max(initial=3)), int(mjm.pair_dim.max(initial=3)))
    other = (int(mjm.jnt_limited.sum()) + int((mjm.dof_frictionloss > 0).sum()) + 6 * int(mjm.neq)
             + int(mjm.tendon_limited.sum()) + int((mjm.tendon_frictionloss > 0).sum()))
    return int(contacts) * 2 * (condim - 1) + other


def _peak(model, mapping, qpos, qvel, max_triangle_pairs):
    """The most contacts one world of `model` has over the states, colliding them as many at a time as it has worlds."""
    B = int(model.world_count)
    state = model.state()
    pipe = newton.CollisionPipeline(model, max_triangle_pairs=int(max_triangle_pairs))
    con = pipe.contacts()
    shape_world = model.shape_world.numpy()
    peak = 0
    for i in range(0, len(qpos), B):
        q, v = qpos[i:i + B], qvel[i:i + B]
        if len(q) < B:
            q = np.concatenate([q, np.repeat(q[-1:], B - len(q), 0)])
            v = np.concatenate([v, np.repeat(v[-1:], B - len(v), 0)])
        jq, jqd = mapping.to_newton(q, v)
        state.joint_q.assign(jq.astype(np.float32))
        state.joint_qd.assign(jqd.astype(np.float32))
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        pipe.collide(state, con)
        n = min(int(con.rigid_contact_count.numpy()[0]), int(con.rigid_contact_max))
        if n:
            w0 = shape_world[con.rigid_contact_shape0.numpy()[:n]]
            w1 = shape_world[con.rigid_contact_shape1.numpy()[:n]]
            w = np.where(w0 >= 0, w0, w1)
            peak = max(peak, int(np.bincount(w[w >= 0], minlength=B).max()))
    return peak


def peak_contacts(pkg, solver, qpos, qvel, options=None, worlds=PROBE_WORLDS, max_triangle_pairs=1_000_000):
    """
    The most contacts one world has over the states `(qpos, qvel)` (MuJoCo's layout), by
    Newton's pipeline at its own capacity, in a model of `worlds` worlds of its own. That
    capacity grows with the worlds: at a thousand worlds of the twenty armadillos it alone
    needs 3.5 GB.
    """
    world, mjm, _ = world_builder(pkg, solver, options)
    builder = builder_for(solver)
    builder.replicate(world, max(1, min(int(worlds), len(qpos))))
    model = builder.finalize()
    return _peak(model, Mapping.of(model, mjm), qpos, qvel, max_triangle_pairs)


def mjwarp_settings(solver):
    """The viewer's MuJoCo Warp options as `SolverMuJoCo` `solver`'s model holds them, by name."""
    o = solver.mjw_model.opt

    def scalar(v):  # a device array of one, or a number
        return float(np.asarray(v.numpy() if hasattr(v, "numpy") else v).reshape(-1)[0])

    def name(v, names):
        return names.get(int(v), int(v))

    return dict(iterations=int(o.iterations), ls_iterations=int(o.ls_iterations), tolerance=scalar(o.tolerance),
                ls_tolerance=scalar(o.ls_tolerance), solver=name(o.solver, {1: "cg", 2: "newton"}),
                integrator=name(o.integrator, {0: "euler", 3: "implicitfast"}),
                cone=name(o.cone, {0: "pyramidal", 1: "elliptic"}))


def kamino_settings(cfg):
    """The viewer's Kamino options as `SolverKamino.Config` `cfg` holds them."""
    out = {k: getattr(cfg.padmm, k) for k in KAMINO_PADMM}
    out.update(preconditioning=bool(cfg.dynamics.preconditioning), gamma=float(cfg.constraints.gamma))
    return out


class Sim:
    """`B` worlds of a package under one of Newton's solvers, stepped with Newton's collision pipeline."""

    def __init__(self, pkg, solver, B=1, caps=None, options=None, capture=True, device=None, probe=None):
        """
        probe: `(qpos, qvel)` states in MuJoCo's layout to measure the contacts per world on,
            when `caps.contacts_per_world` is not given (default: the package's states, else
            the scene's start, else the MJCF's).
        """
        if solver not in SOLVERS:
            raise ValueError(f"solver {solver!r}: one of {SOLVERS}")
        self.pkg, self.solver_name, self.B = pkg, solver, int(B)
        self.caps = dataclasses.replace(caps) if caps is not None else Caps()
        self.options = dict(options or {})
        t0 = time.perf_counter()
        world, self.mjm, self.targets = world_builder(pkg, solver, self.options)
        # What Kamino does not model is applied as forces on bodies in each step (`_step`): the MJCF's forces
        # on sites, their size each actuator's force (`set_ctrl`), and its gravity compensation.
        self.site_forces = site_forces(world, self.mjm) if solver == "kamino" else []
        self.gravcomp = gravity_compensation(world, self.mjm) if solver == "kamino" else []
        applied = [s[:4] for s in self.site_forces] + [(k, f, np.zeros(3), np.zeros(3)) for k, f in self.gravcomp]
        self._applied = None
        if applied:
            body, force, torque, lever = zip(*applied)
            self._applied = dict(body=wp.array(body, dtype=int), force=wp.array(force, dtype=wp.vec3),
                                 torque=wp.array(torque, dtype=wp.vec3), lever=wp.array(lever, dtype=wp.vec3),
                                 mag=wp.zeros((self.B, len(body)), dtype=float), per_world=int(world.body_count))
        builder = builder_for(solver)
        builder.replicate(world, self.B)
        self.model = builder.finalize()
        self.map = Mapping.of(self.model, self.mjm)
        self.dt = float(self.options.get("timestep") or pkg.get("dt") or self.mjm.opt.timestep)
        self.solver_defaults, self.solver_config = {}, {}  # Kamino's: its options before and after `options`
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        if self.caps.contacts_per_world is None:
            self.caps.contacts_per_world = self.probe_contacts(probe)
        # The broad phase: Newton's, the explicit pairs the filters allow, or `options["broad_phase"]`, which
        # finds its pairs, at most `caps.shape_pairs` of them
        bp = self.options.get("broad_phase")
        kw = {} if bp is None else dict(broad_phase=str(bp), shape_pairs_max=self.caps.shape_pairs(self.B))
        self.pipeline = newton.CollisionPipeline(self.model,
                                                 rigid_contact_max=int(self.caps.contacts_per_world) * self.B,
                                                 max_triangle_pairs=self.caps.triangle_pairs(self.B), **kw)
        self.contacts = self.pipeline.contacts()
        self.solver = self._make_solver()
        # The shapes' contact parameters as built, which the viewer's Collision pairs folder starts from
        self.shape_params0 = self.shape_params()
        self.build_seconds = time.perf_counter() - t0
        self.graph = None
        self.want_capture = capture and wp.get_device(device).is_cuda
        self.set_ctrl()

    def probe_contacts(self, probe=None, margin=2.0, least=256):
        """
        Contacts per world to allocate: twice the most any world has over the probe states
        (or `least`), measured by a pipeline at Newton's own estimate (`peak_contacts`; on
        this model when it has few enough worlds).
        """
        if probe is None:
            st = self.pkg.get("states")
            if st is not None and len(st["qpos"]):
                k = np.random.default_rng(0).choice(len(st["qpos"]), size=min(len(st["qpos"]), 4 * self.B),
                                                    replace=False)
                probe = (st["qpos"][k], st["qvel"][k])
            elif self.pkg.get("qpos0") is not None:
                # The scene's start: a pile's MJCF has its bodies on top of each other (the five hands' some 8,000
                # contacts, sizing Kamino's dense Delassus matrix past 2^31 entries), its sampler sets them apart
                probe = (np.asarray(self.pkg["qpos0"])[None], np.asarray(self.pkg["qvel0"])[None])
            else:
                probe = (self.mjm.qpos0[None], np.zeros((1, self.mjm.nv)))
        qpos, qvel = probe
        if self.B <= PROBE_WORLDS:
            peak = _peak(self.model, self.map, qpos, qvel, self.caps.triangle_pairs(self.B))
        else:
            peak = peak_contacts(self.pkg, self.solver_name, qpos, qvel, self.options)
        return int(max(least, margin * peak))

    def _make_solver(self):
        o = self.options
        if self.solver_name == "mjwarp":
            # The MJCF's options as MuJoCo compiles them: where an MJCF leaves one out,
            # Newton's default is its own (implicitfast, pyramidal) rather than MuJoCo's.
            mo = self.mjm.opt
            kw = dict(use_mujoco_contacts=False, separate_worlds=True, nconmax=int(self.caps.contacts_per_world),
                      solver=int(mo.solver), integrator=int(mo.integrator), cone=int(mo.cone),
                      impratio=float(mo.impratio), iterations=int(mo.iterations), ls_iterations=int(mo.ls_iterations),
                      tolerance=float(mo.tolerance), ls_tolerance=float(mo.ls_tolerance))
            for k in ("iterations", "ls_iterations", "tolerance", "ls_tolerance", "cone", "solver", "integrator",
                      "impratio"):
                if o.get(k) is not None:
                    kw[k] = o[k]  # the viewer's choices by name, which Newton parses
            kw["njmax"] = int(self.caps.njmax or constraint_rows(self.mjm, self.caps.contacts_per_world))
            solver = newton.solvers.SolverMuJoCo(self.model, **kw)
            self.solver_config = mjwarp_settings(solver)
            self.friction_mix = None  # MuJoCo's: the larger, but by priority (`pair_info`)
            return solver
        cfg = newton.solvers.SolverKamino.Config.from_model(self.model)
        self.solver_defaults = kamino_settings(cfg)
        cfg.use_collision_detector = False  # Newton's collision pipeline, as for MuJoCo Warp
        cfg.collision_detector.max_contacts_per_world = int(self.caps.contacts_per_world)
        if o.get("dynamics_solver"):
            cfg.dynamics_solver = o["dynamics_solver"]
        for k, v in (o.get("padmm") or {}).items():
            setattr(cfg.padmm, k, v)
        for k in KAMINO_PADMM:  # the viewer's, each as the field's type (its switches come as 0/1)
            if o.get(k) is not None:
                setattr(cfg.padmm, k, type(getattr(cfg.padmm, k))(o[k]))
        if o.get("preconditioning") is not None:
            cfg.dynamics.preconditioning = bool(o["preconditioning"])
        if o.get("gamma") is not None:
            cfg.constraints.gamma = float(o["gamma"])
        # Execution options, the same solve: the Jacobians stored sparse; the dense Delassus matrix factored as
        # `LLTB` (the default) or after a reverse Cuthill-McKee reordering, `LLTBRCM`
        if o.get("sparse_jacobian") is not None:
            cfg.sparse_jacobian = bool(o["sparse_jacobian"])
        if o.get("linear_solver"):
            cfg.dynamics.linear_solver_type = str(o["linear_solver"])
        self.solver_config = kamino_settings(cfg)
        self.friction_mix = str(cfg.materials.friction_mix_mode)  # how a contact's friction mixes its shapes'
        return newton.solvers.SolverKamino(self.model, config=cfg)

    def set_ctrl(self, ctrl=None):
        """
        The controls `ctrl`, `(nu,)` in the MJCF's order (default the package's constant one), in
        every world. Under MuJoCo Warp the MJCF's actuators read MuJoCo's `ctrl`
        (`control.mujoco.ctrl`), except its `<position>` servos, which Newton makes joint drives;
        under Kamino every servo is a joint drive. A drive's target is its actuator's control
        (`world_builder`), and Kamino applies the MJCF's forces on sites itself (`site_forces`, `_step`).
        Written into the arrays the step reads, so a captured step takes them.
        """
        given = self.pkg.get("ctrl") if ctrl is None else ctrl
        u = np.zeros(int(self.mjm.nu)) if given is None else np.asarray(given, np.float64).reshape(-1)
        if self._applied is not None:
            mag = [actuator_force(self.mjm, a, u[a]) for *_, a in self.site_forces] + [1.0] * len(self.gravcomp)
            self._applied["mag"].assign(np.tile(np.asarray(mag, np.float32), (self.B, 1)))
        if self.solver_name == "mjwarp" and given is not None:
            arr = getattr(getattr(self.control, "mujoco", None), "ctrl", None)
            if arr is not None and arr.shape[0]:
                arr.assign(np.tile(u.astype(np.float32), self.B))
        if not self.targets or getattr(self.control, "joint_target_q", None) is None:
            return
        jt = self.control.joint_target_q.numpy().reshape(self.B, -1)
        for target, a in self.targets:
            jt[:, target] = u[a]
        self.control.joint_target_q.assign(jt.reshape(-1))

    def solver_status(self):
        """
        The first world's solve in the last step: Kamino's PADMM iterations, whether it converged,
        and its primal, dual and complementarity residuals (absolute, in physical units,
        `SolverKamino.status`); MuJoCo Warp's Newton iterations, and whether it stopped short of
        its cap.
        """
        if self.solver_name == "mjwarp":  # MuJoCo Warp records its iterations; at its cap it did not converge
            n = int(self.solver.mjw_data.solver_niter.numpy()[0])
            return dict(iterations=n, converged=n < int(self.solver.mjw_model.opt.iterations))
        st = self.solver.status.numpy()[0]
        return dict(iterations=int(st["iterations"]), converged=bool(st["converged"]), r_p=float(st["r_p"]),
                    r_d=float(st["r_d"]), r_c=float(st["r_c"]))

    def overflows(self):
        """
        Which capacities the last collision pass or solve overflowed: `contacts`, more than the contact buffer
        holds; `triangle_pairs`, more mesh triangle pairs than their buffer, or inserts that failed in the table
        reducing the mesh contacts; `rows`, MuJoCo Warp's constraint rows a world.
        """
        out = []
        if int(self.contacts.rigid_contact_count.numpy()[0]) > int(self.contacts.rigid_contact_max):
            out.append("contacts")
        nph = self.pipeline.narrow_phase
        tri = getattr(nph, "triangle_pairs_count", None)
        red = getattr(nph, "global_contact_reducer", None)
        if (tri is not None and int(tri.numpy()[0]) > self.caps.triangle_pairs(self.B)) or (
                red is not None and (int(red.ht_insert_failures.numpy()[0]) > 0
                                     or int(red.buffer_overflows.numpy()[0]) > 0)):
            out.append("triangle_pairs")
        if self.solver_name == "mjwarp" and \
                int(self.solver.mjw_data.nefc.numpy().max()) >= int(self.solver.mjw_data.njmax):
            out.append("rows")
        return out

    def overflowed(self):
        """Whether the last collision pass or solve dropped work (`overflows`)."""
        return bool(self.overflows())

    # -----------------------------------------------------------------------------------
    # For the viewer: the capacities' demand, and the shape pairs' contact parameters

    def counts(self):
        """
        What the last collision pass asked of the capacities -- its contacts and mesh triangle pairs, raw, past what
        they hold -- and the last solve: MuJoCo Warp's constraint rows a world.
        """
        tri = getattr(self.pipeline.narrow_phase, "triangle_pairs_count", None)
        return dict(contacts=self.contact_count(), triangle_pairs=int(tri.numpy()[0]) if tri is not None else 0,
                    rows=int(self.solver.mjw_data.nefc.numpy().max()) if self.solver_name == "mjwarp" else 0)

    def capacity(self):
        """What the capacities hold: contacts a world, mesh triangle pairs (at least Newton's million), rows a world."""
        out = dict(contacts=int(self.caps.contacts_per_world), triangle_pairs=self.caps.triangle_pairs(self.B))
        if self.solver_name == "mjwarp":
            out["rows"] = int(self.solver.mjw_data.njmax)
        return out

    def _geom_of_shape(self):
        """World 0's MuJoCo Warp geom of each Newton shape, -1 for none: SolverMuJoCo's map, inverted."""
        m = self.solver.mjc_geom_to_newton_shape.numpy()
        g2s = m[0] if len(m) else np.zeros(0, int)
        out = np.full(int(self.model.shape_count), -1, int)
        ok = g2s >= 0
        out[g2s[ok]] = np.nonzero(ok)[0]
        return out

    def shape_params(self):
        """
        Each shape's contact parameters as MuJoCo Warp holds them, its geom's: `friction` (slide, torsion, roll),
        `solref`, `solimp`, `solmix`, `priority`; zeros for a shape without a geom (`geom` -1). Under Kamino its
        friction alone, the shape's own, which Kamino mixes a contact's from (`pair_info`).
        """
        if self.solver_name == "kamino":
            n = int(self.model.shape_count)
            fr = np.zeros((n, 3))
            fr[:, 0] = self.model.shape_material_mu.numpy()
            return dict(geom=np.arange(n), friction=fr, solref=np.zeros((n, 2)), solimp=np.zeros((n, 5)),
                        solmix=np.zeros(n), priority=np.zeros(n, np.int32))
        mm, geom = self.solver.mjw_model, self._geom_of_shape()
        has = geom >= 0  # a scene can have shapes MuJoCo Warp makes no geoms of, or none at all
        out = dict(geom=geom)
        for k, a, w in (("friction", mm.geom_friction, (3,)), ("solref", mm.geom_solref, (2,)),
                        ("solimp", mm.geom_solimp, (5,)), ("solmix", mm.geom_solmix, ())):
            v = np.zeros((len(geom),) + w)
            if has.any():
                v[has] = a.numpy()[0][geom[has]]
            out[k] = v
        out["priority"] = np.zeros(len(geom), np.int32)
        if has.any():
            out["priority"][has] = mm.geom_priority.numpy()[geom[has]]
        return out

    def pair_info(self):
        """
        What the viewer's Collision pairs folder lists: the shape pairs Newton's pipeline collides
        (`Model.shape_contact_pairs`) -- under MuJoCo Warp those between shapes it has geoms for -- each shape's
        name -- its MJCF geom's, or `body/surface` for a Lodos surface's mesh -- its contact parameters as built
        (`shape_params`), and `mix`, how Kamino mixes a contact's friction from its two shapes' (None: MuJoCo's).
        """
        if self.B != 1:
            raise ValueError("the shapes' contact parameters are read at one world")
        p = self.shape_params0
        m = self.model
        pairs = m.shape_contact_pairs.numpy().reshape(-1, 2).astype(int) if m.shape_contact_pair_count \
            else np.zeros((0, 2), int)
        pairs = pairs[(p["geom"][pairs[:, 0]] >= 0) & (p["geom"][pairs[:, 1]] >= 0)]
        names = []
        for label in m.shape_label:
            parts = str(label).split("/")
            names.append(f"{parts[-2]}/surface" if parts[-1] == "lodos_surface" and len(parts) > 1 else parts[-1])
        return dict(pairs=pairs.astype(np.int32), names=names, mix=self.friction_mix,
                    **{k: p[k] for k in ("friction", "solref", "solimp", "solmix", "priority")})

    def set_shape_params(self, mu, solref, solimp):
        """
        Every shape's slide friction, solref and solimp -- `(nshape,)`, `(nshape, 2)`, `(nshape, 5)` -- on the model
        as SolverMuJoCo reads them, the solref raw, as MuJoCo takes it; MuJoCo Warp's geoms take them
        (`notify_model_changed`), and the step is captured again, its contacts' parameters packed anew. Kamino
        takes the friction alone, which it reads per shape at every step's contacts.
        """
        if self.B != 1:
            raise ValueError("the shapes' contact parameters are set at one world")
        m = self.model
        has = self.shape_params0["geom"] >= 0

        def put(arr, values, dtype=np.float32):
            a = arr.numpy()
            a[has] = np.asarray(values, dtype)[has].reshape(a[has].shape)
            arr.assign(a)

        put(m.shape_material_mu, mu)
        if self.solver_name == "kamino":
            return
        ca = m.mujoco
        put(ca.solref, solref)
        put(ca.solref_mode, np.full(len(has), SOLREF_RAW), np.int32)
        put(ca.geom_solimp, solimp)
        self.solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
        self.graph = None

    # -----------------------------------------------------------------------------------

    def set_state(self, qpos, qvel):
        """`(B, nq)`, `(B, nv)` (or one row for every world) in MuJoCo's layout."""
        qpos, qvel = np.atleast_2d(qpos), np.atleast_2d(qvel)
        if qpos.shape[0] == 1 and self.B > 1:
            qpos, qvel = np.repeat(qpos, self.B, 0), np.repeat(qvel, self.B, 0)
        jq, jqd = self.map.to_newton(qpos, qvel)
        for st in (self.state_0, self.state_1):
            st.joint_q.assign(jq.astype(np.float32))
            st.joint_qd.assign(jqd.astype(np.float32))
            newton.eval_fk(self.model, st.joint_q, st.joint_qd, st)
        if self.solver_name == "kamino":
            # The bodies are where the joints put them: keep them, clear Kamino's internals.
            self.solver.reset(self.state_0, config=newton.solvers.SolverKamino.ResetConfig.preserve())
        self.graph = None if self.graph is not None and self.solver_name == "kamino" else self.graph

    def get_state(self):
        return self.map.to_mujoco(self.state_0.joint_q.numpy(), self.state_0.joint_qd.numpy(), self.B)

    def track(self, per_world=False):
        """
        From now on, every step keeps on the device the most contacts and mesh triangle pairs of a collision pass
        (batch totals, raw), the most constraint rows of a world (MuJoCo Warp's), and of the table that reduces the
        mesh contacts (sized by the triangle pairs, a quarter of them its hash table) the most contacts it was given
        and the most it lost, to a full table or buffer, and the most candidate shape pairs of the broad phase, past
        what it holds: `peaks`, read once at the end, so a timed run is not stopped to read them. `per_world`: the
        most contacts of one world too, counted from their shapes, which Kamino holds per world (`contacts_per_world`
        each). The step is captured again, the tracking in it.
        """
        self.peak = wp.zeros(6, dtype=wp.int32)
        c = self.contacts
        self._world_args = (c.rigid_contact_count, c.rigid_contact_shape0, c.rigid_contact_shape1,
                            self.model.shape_world) if per_world else None
        self._world_n, self.world_peak = wp.zeros(max(1, self.B), dtype=wp.int32), wp.zeros(1, dtype=wp.int32)
        z = wp.zeros(max(1, self.B), dtype=wp.int32)
        tri = getattr(self.pipeline.narrow_phase, "triangle_pairs_count", None)
        nefc = self.solver.mjw_data.nefc if self.solver_name == "mjwarp" else z
        red = getattr(self.pipeline.narrow_phase, "global_contact_reducer", None)
        reds = (red.contact_count, red.ht_insert_failures, red.buffer_overflows) if red is not None else (z[:1],) * 3
        self._track_args = (self.contacts.rigid_contact_count, tri if tri is not None else z[:1], nefc, *reds,
                            self.pipeline.broad_phase_pair_count)
        self.graph = None

    def peaks(self):
        """
        `dict(contacts, triangle_pairs, rows, reduced, reduction_lost, shape_pairs)`, and `world_contacts` where
        `track` counted them per world: the largest demands since `track` (or `reset_peaks`).
        """
        p = self.peak.numpy()
        out = dict(contacts=int(p[0]), triangle_pairs=int(p[1]), rows=int(p[2]), reduced=int(p[3]),
                   reduction_lost=int(p[4]), shape_pairs=int(p[5]))
        if self._world_args is not None:
            out["world_contacts"] = int(self.world_peak.numpy()[0])
        return out

    def reset_peaks(self):
        self.peak.zero_()
        self.world_peak.zero_()

    def _step(self):
        self.state_0.clear_forces()
        if self._applied is not None:
            s = self._applied
            wp.launch(_point_forces, dim=s["mag"].shape,
                      inputs=[self.state_0.body_q, s["body"], s["force"], s["torque"], s["lever"], s["mag"], s["per_world"]],
                      outputs=[self.state_0.body_f])
        self.pipeline.collide(self.state_0, self.contacts)
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.dt)
        self.state_0.assign(self.state_1)
        if getattr(self, "_track_args", None) is not None:
            wp.launch(_track, dim=max(1, self.B), inputs=list(self._track_args), outputs=[self.peak])
            if self._world_args is not None:
                wp.launch(_count_by_world, dim=int(self.contacts.rigid_contact_max), inputs=list(self._world_args),
                          outputs=[self._world_n])
                wp.launch(_track_world, dim=max(1, self.B), inputs=[self._world_n], outputs=[self.world_peak])

    def step(self, n=1):
        if self.want_capture and self.graph is None:
            self._step()  # loads the modules outside the capture
            with wp.ScopedCapture() as c:
                self._step()  # recorded, not run
            self.graph = c.graph
            n -= 1
        for _ in range(max(n, 0)):
            if self.graph is not None:
                wp.capture_launch(self.graph)
            else:
                self._step()

    def collide(self):
        self.pipeline.collide(self.state_0, self.contacts)

    def contact_count(self):
        return int(self.contacts.rigid_contact_count.numpy()[0])

    def contacts_world(self, world=0):
        """World 0's contacts: points, normals, signed distances, forces, MuJoCo body pairs."""
        n = min(self.contact_count(), int(self.contacts.rigid_contact_max))
        z3 = np.zeros((0, 3))
        if n == 0:
            return z3, z3, np.zeros(0), z3, np.zeros((0, 2), int)
        c = self.contacts
        s0, s1 = c.rigid_contact_shape0.numpy()[:n], c.rigid_contact_shape1.numpy()[:n]
        sb = self.model.shape_body.numpy()
        bq = self.state_0.body_q.numpy()
        b0, b1 = sb[s0], sb[s1]
        nw = self.model.body_count // self.model.world_count
        mine = ((b0 < 0) | (b0 // max(nw, 1) == world)) & ((b1 < 0) | (b1 // max(nw, 1) == world))
        p0 = c.rigid_contact_point0.numpy()[:n]
        p1 = c.rigid_contact_point1.numpy()[:n]

        def world_point(b, p):
            out = p.astype(np.float64).copy()
            on = b >= 0
            out[on] = bq[b[on], :3] + qrot(bq[b[on], 3:].astype(np.float64), p[on].astype(np.float64))
            return out

        w0, w1 = world_point(b0, p0), world_point(b1, p1)
        normal = c.rigid_contact_normal.numpy()[:n].astype(np.float64)
        dist = np.einsum("ij,ij->i", w1 - w0, normal)
        force = c.rigid_contact_force.numpy()[:n, :3].astype(np.float64) if c.rigid_contact_force is not None \
            else np.zeros((n, 3))
        mjb = self.mj_body_ids()
        bodies = np.stack([np.where(b0 >= 0, mjb[np.maximum(b0, 0) % max(nw, 1)], 0),
                           np.where(b1 >= 0, mjb[np.maximum(b1, 0) % max(nw, 1)], 0)], 1)
        k = mine
        return 0.5 * (w0 + w1)[k], normal[k], dist[k], force[k], bodies[k]

    def mj_body_ids(self):
        """MuJoCo's body id of each of world 0's Newton bodies."""
        if not hasattr(self, "_mjb"):
            nw = self.model.body_count // self.model.world_count
            bodies = mj_ids(self.mjm, mujoco.mjtObj.mjOBJ_BODY)
            self._mjb = np.array([max(bodies.get(name_of(x), -1), 0) for x in self.model.body_label[:nw]], int)
        return self._mjb
