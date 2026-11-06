"""
A complete simulation: one scene, both collision pipelines, one contact solver.

A scene is an MJCF model plus, optionally, a list of Lodos surfaces:

- **Lodos surfaces** (`lodos.col.sdf`) are remeshes, vertex-and-SDF geometries, each
  keyed to an MJCF geom -- the geom MuJoCo takes the body's inertia from, typically a
  mesh of the same surface. They collide with each other through Lodos's
  directional pairs. A keyed geom leaves MJWarp's collision, except a plane: a
  `PlaneSurface` keyed to the MJCF floor makes one floor for both pipelines.
- **MJCF geoms** that are not keyed to a surface collide with each other through
  MJWarp's pipeline (`lodos.col.prim`) -- MuJoCo's primitives, and polytopes through
  the exact hull narrowphase. The two kinds never collide with each other: a
  polytope that supplements a surface where edge-edge contact matters is there to
  collide with other polytopes.

The contact solve is MJWarp's constraint solver over both kinds of contacts
(`lodos.cd.driver`): its Newton solver on MuJoCo's rows, or inside the prox solver's loop,
under the exact mode or MuJoCo's (`mode="mj"`). Lodos's contacts reach it one
constraint per contact, or gathered into contact groups (`lodos.cd.groups`).

Unconstrained dynamics and integration are `lodos.rbd`'s. Joint limits and dof friction
loss are constraints of MJWarp's solve; `make_sim`'s `joint_constraints=False` takes them
out.

Pair parameters for Lodos pairs are one table of thirteen columns (`lodos.cd.params`):
`[mu, stiffness, sharpness, dissipation, stiction, solref(2), solimp(5), w]`. The solve
reads `mu`, and under mj `solref` and `solimp`; contact groups weigh their members by the
spring-damper columns. MJWarp's contacts take MuJoCo's parameters from the MJCF.

The exact mode -- the default -- takes none of MuJoCo's impedance: no solref, solimp
or impratio reaches its contact solve (`make_sim`).
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any

import mujoco
import numpy as np
import warp as wp


from lodos.cd import driver as cd_driver
from lodos.cd import params as cd_params
from lodos.col.prim.driver import PrimCollision
from lodos.col.prim.hybrid.pipeline import ALL_TYPES
from lodos.col.sdf import _is_plane
from lodos.col.sdf import types as sdf_types
from lodos.mjcf import io as mjcf_io
from lodos.mjcf.io import override_model
from lodos.rbd import io as rbd_io
from lodos.rbd import smooth
from lodos.rbd import types as rbd_types

# `make_sim`'s `integrator`: MuJoCo's integrators `lodos.rbd` has.
INTEGRATORS = {"euler": mujoco.mjtIntegrator.mjINT_EULER, "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST}


def exact_mode(solver_kwargs=None):
    """
    Whether `solver_kwargs` pose the exact contact problem, which no MuJoCo impedance reaches.

    Called by `make_sim`.
    In: solver_kwargs, a dict of the solver's options or None. Out: bool, True unless `mode="mj"`.
    """
    return (solver_kwargs or {}).get("mode", "exact") == "exact"


_MJ_COLS = [0, 5, 6, 7, 8, 9, 10, 11]  # mu, solref, solimp


def default_pair_params(mu, nCPR):
    """
    The thirteen-column table's defaults with friction `mu` (`lodos.cd.params`).

    Called by `make_sim`, the viewer's scenes (`viewer.scenes.armadillo.build`, `viewer.scenes.nist.surfaces` and
    13 others) and three of the notebooks.
    In: mu float; nCPR int, the Lodos pairs. Out: (nCPR, 13) float32 numpy array.
    """
    return cd_params.make_default_contact_params(mu, nCPR)


@dataclasses.dataclass
class Surfaces:
    """
    The Lodos half of a scene.

    surfaces: baked surfaces (`lodos.col.sdf.load_surface_file`) or `PlaneSurface()`s.
    geoms: the MJCF geom id each surface is keyed to.
    pairs: directional `[vertex surface, SDF surface]` index pairs.
    params: `(len(pairs), 13)` pair parameters, or None for the defaults with `mu`.
    mu: friction for the default parameters.
    col_kwargs: `make_collision_context` options (capacities, tiles, contact point).

    Built by the viewer's scenes (`viewer.scenes.nist.surfaces`, `viewer.scenes.sharpa.surfaces` and 11 others) and
    three of the notebooks; read by `make_sim`.
    """

    surfaces: list  # nB surfaces
    geoms: list  # (nB,) int: each surface's MJCF geom
    pairs: list  # (nCPR,) [int, int] pairs of surface indices
    params: Any = None  # (nCPR, 13) float pair parameters, or None
    mu: float = 0.8  # the default parameters' friction
    col_kwargs: dict = dataclasses.field(default_factory=dict)  # `make_collision_context`'s keyword arguments


class Sim:
    """
    Every context one scene needs, built by `make_sim`. Step it with `lodos.step`.

    Its attributes, set by `make_sim`: surfaces, the `Surfaces` or None; mrbd, drbd, Lodos's rigid-body model and
    data (`lodos.rbd.types`); mcol, dcol, Lodos's collision model and data (`lodos.col.sdf.types`), None without
    Lodos pairs; params, the (nCPR, 13) float32 pair parameters or None; options, the solver's resolved options
    (`lodos.cd.driver.resolve_options`); m, d, MJWarp's model and data; prim, MJWarp's collision
    (`lodos.col.prim.driver.PrimCollision`); geom_bodyid, (ngeom,) int32 Warp array, each geom's body; solver, the
    contact solve (`lodos.cd.driver.SolverDriver`).
    """

    mjm: mujoco.MjModel  # make_sim's copy of the scene: keyed geoms out of MJWarp's collision, the options applied
    B: int  # worlds

    def __repr__(self):
        """
        The worlds and each pipeline's pairs, as `Sim(B=8, lodos pairs=3, prim pairs=2)`.

        Called by `repr`.
        """
        parts = [f"B={self.B}"]  # list of str
        if self.mcol is not None:
            parts.append(f"lodos pairs={int(self.mcol.nCPR)}")
        if self.prim is not None and self.prim.active:
            parts.append(f"prim pairs={self.prim.npairs}")
        return f"Sim({', '.join(parts)})"


def make_sim(
    mjm: mujoco.MjModel,
    B: int,
    *,
    surfaces: Surfaces | None = None,
    qpos=None,
    solver_kwargs: dict | None = None,
    overrides=(),
    nconmax: int | None = None,
    njmax: int | None = None,
    nccdmax: int | None = None,
    naconmax: int | None = None,
    njmax_nnz: int | None = None,
    hybrid=ALL_TYPES,
    condim: int = 3,
    contact_reduction: bool | None = None,
    joint_constraints: bool = True,
    integrator: str | None = None,
    implicit_constraints: bool = False,
    tuning=None,
):
    """
    Builds a simulation of `B` worlds of `mjm`.

    surfaces: the Lodos half of the scene, or None for an MJCF-only one.
    qpos: `(B, nq)` initial positions, or None for `qpos0`.
    solver_kwargs: the solver's options (`lodos.cd.driver.resolve_options`: `mode`, the
        proximal loop's, `baumgarte`, `de_saxce`, `refresh_de_saxce_per_prox`, `warm_start`,
        `fused_newton`, `groups` and their bins, `row_space`, `shared_hessian`). `warm_start=False` starts each
        step cold -- MuJoCo's warmstart disable bit, and no carried multipliers either.
    overrides: model overrides ("opt.iterations=50", ...), after the ones the options come to
        (`lodos.cd.driver.overrides`). They are applied to the copy of `mjm` before Lodos's
        rigid-body model and MJWarp's are built from it, so both step with them.
    nconmax, njmax, nccdmax, naconmax, njmax_nnz: MJWarp's capacities, per world unless
        `naconmax`. They must hold MJWarp's contacts and Lodos's. `njmax_nnz` bounds a
        sparse Jacobian's nonzeros per world (a model of more than 32 dofs; a contact row
        between two free bodies has 12). Left out, it is `lodos.mjcf.io`'s estimate, which counts
        Lodos's pairs beside the MJCF's own.
    hybrid: the pair types the exact hull narrowphase takes, or () for MJWarp's own
        narrowphases throughout.
    condim: contact dimension of Lodos's contacts.
    contact_reduction: hand MJWarp the at most 240 contacts per Lodos pair segment that
        `lodos.cd.reduce` keeps -- Newton's reduction -- rather than every one; a grouped
        pair's groups are made of those. None: the tuning file's, else False.
    joint_constraints: whether the joints' constraints take part -- their limits and dof
        friction loss, the only non-contact constraints `lodos.rbd` admits (it refuses
        tendons, equality constraints and flexes). True (the default): the solve has them
        as MuJoCo's soft constraint rows. False: the model is the copy with both removed: a
        limited joint turns freely and a joint with friction loss slides without it.
    integrator: "euler" or "implicitfast", in place of the MJCF's own `opt.integrator`;
        None keeps the MJCF's. `implicitfast` (`lodos.rbd.implicit`) is MuJoCo's: implicit
        in the actuators' and the dofs' damping, and for a free body standing alone in its
        gyroscopic force.
    implicit_constraints: under `implicitfast`, the matrix the constraint solve sees.
        False (the default): M, as MuJoCo has it; the implicit matrix `M - dt D` only
        integrates the solver's result, and a step factors both. True: `M - dt D`
        throughout -- the mass solves and the solve's M -- so the contact impulse sees the
        damping that acts in the same step; one factorization, without the free bodies'
        gyroscopic term, which is not symmetric.
    tuning: a tuning file (`lodos.utils.tuning`: a path or a loaded dict) whose settings fill
        whatever is not given here -- the capacities above, `contact_reduction`,
        `solver_kwargs` keys, and the collision options `surfaces.col_kwargs` does not name.
        Below the file's batch size its batch totals are left out; a file measured on a
        different scene is used with a warning (`lodos.utils.tuning.stale`).

    The exact mode (`exact_mode`: unless `mode="mj"`) takes none of MuJoCo's
    impedance, and runs its copy of the model at `impratio` 1. It takes `baumgarte` (default
    True), the Baumgarte bias in the normal target. The solve writes its own weight
    `D = dt / rho_i` on all three rows of every contact and its own reference in place of
    MuJoCo's `aref` (`opt.prox_baumgarte`). What `impratio` would still reach is MJWarp's
    elliptic cone. It is written in residual space with the coefficient `mu sqrt(R_t / R_n)`,
    MuJoCo's own ratio of frictional to normal regularization, which MJWarp takes as
    `mu / sqrt(impratio)`. That is the true cone only under MuJoCo's weights,
    `D_t = impratio D_n`; under the exact rows' equal ones it must be `mu`. At `impratio` 10
    a sticking contact holds at most a tenth of `mu f_n`, the cost jumps at the stick-slide
    boundary, and the solve creeps (a box down a slope it should hold) or diverges.
    `lodos.cd.solver.nlsp` takes its De Saxce coefficient as `mu / sqrt(impratio)`
    too. The joint-limit and friction-loss rows stay MuJoCo's soft rows under either
    mode: the prox solver solves contacts only.

    Called by the viewer's scene builders (`viewer.scenes.armadillo.build`, `viewer.scenes.nist.build`,
    `viewer.scenes.sharpa.build` and 10 others), `docs/manual/make_data.py` and three of the notebooks.
    In: mjm, the scene, copied (the caller's is left as it is); B int, the worlds. Out: a `Sim`, its state `qpos`
    (or `qpos0`) at rest, its poses computed (`lodos.rbd.smooth.kinematics`). Raises ValueError where the surfaces
    and their geoms differ in number, on groups without Lodos surfaces, and on what the contexts it builds refuse.
    """
    if tuning is not None:
        from lodos.utils import tuning as ltuning

        # dict: MJWarp's capacities, None where not given
        caps = dict(nconmax=nconmax, njmax=njmax, nccdmax=nccdmax, naconmax=naconmax, njmax_nnz=njmax_nnz)
        # extra: dict, the tuning file's contact_reduction
        surfaces, solver_kwargs, caps, extra = ltuning.merge_into_make_sim(
            tuning, B, surfaces, solver_kwargs, caps, dict(contact_reduction=contact_reduction), mjm=mjm)
        nconmax, njmax, nccdmax, naconmax, njmax_nnz = (caps[k] for k in ("nconmax", "njmax", "nccdmax", "naconmax",
                                                                          "njmax_nnz"))
        contact_reduction = extra["contact_reduction"]
    contact_reduction = bool(contact_reduction)
    solver_kwargs = dict(solver_kwargs or {})
    mjm = copy.copy(mjm)  # keyed geoms are switched off for MJWarp below
    if integrator is not None:
        mjm.opt.integrator = INTEGRATORS[integrator]
    joint_rows = bool(joint_constraints)  # bool: the joints' limits and friction loss are rows of the solve
    if not joint_rows:
        mjm.jnt_limited[:] = 0
        mjm.dof_frictionloss[:] = 0.0
    if exact_mode(solver_kwargs):
        mjm.opt.impratio = 1.0

    sim = Sim()  # Sim
    sim.B = int(B)
    sim.surfaces = surfaces

    # ------------------------------------------------------------------------------
    # Lodos surfaces: their geoms leave MJWarp's collision, planes excepted.
    g_geom = None  # (nB,) int32: each surface's MJCF geom; None without Lodos pairs
    if surfaces is not None and len(surfaces.pairs):
        g_geom = np.asarray(surfaces.geoms, dtype=np.int32)
        if len(g_geom) != len(surfaces.surfaces):
            raise ValueError("every surface needs the MJCF geom it is keyed to")
        for s, g in zip(surfaces.surfaces, g_geom):  # s: a surface; g: int32, its geom
            if not _is_plane(s):
                mjm.geom_contype[g] = 0
                mjm.geom_conaffinity[g] = 0
    sim.mjm = mjm

    # ------------------------------------------------------------------------------
    # The model overrides, the options' and the caller's: applied to `mjm` before either
    # model is built from it, so the rigid-body pass and the solve step with one timestep,
    # gravity, set of disable flags and integrator (`lodos.cd.driver.check_models`).
    sim.options = cd_driver.resolve_options(**solver_kwargs)
    ov = cd_driver.overrides(sim.options) + tuple(overrides)  # tuple of str: the model overrides
    if not sim.options["warm_start"]:
        # Or-ed into the model's own flags rather than an override, which would replace
        # them: a scene may disable other things (`eulerdamp`, say).
        mjm.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_WARMSTART)
    override_model(mjm, ov)

    # ------------------------------------------------------------------------------
    # Rigid-body dynamics.
    # RbdModel, RbdData: B worlds' rigid bodies
    mrbd, drbd = rbd_types.make_rbd_context(mjm, B, constraints=joint_rows, implicit_constraints=implicit_constraints)
    if qpos is not None:
        rbd_io.set_state(drbd, qpos=np.asarray(qpos).reshape(B, -1))
    # Poses for the first step, and -- because `kinematics` writes the normalized
    # quaternions back into `qpos` -- the initial state, normalized. Normalizing in
    # float32 is not idempotent to the last bit, so a rollout started without this
    # differs from one started with it by rounding.
    smooth.kinematics(mrbd, drbd)
    sim.mrbd, sim.drbd = mrbd, drbd

    # ------------------------------------------------------------------------------
    # Lodos collision.
    sim.mcol = sim.dcol = None
    params13 = None  # (nCPR, 13) float32: the pair parameters; None without Lodos pairs
    if g_geom is not None:
        body_ids = [int(mjm.geom_bodyid[g]) for g in g_geom]  # (nB,) list of int: each surface's body
        sim.mcol, sim.dcol = sdf_types.make_collision_context(
            mjm, drbd, surfaces.pairs, surfaces.surfaces, body_ids, **surfaces.col_kwargs
        )
        nCPR = int(sim.mcol.nCPR)  # int: the Lodos pairs
        params13 = (default_pair_params(surfaces.mu, nCPR) if surfaces.params is None
                    else np.asarray(surfaces.params, dtype=np.float32).reshape(nCPR, 13))
    sim.params = params13

    # ------------------------------------------------------------------------------
    # MJWarp: model, data and collision.
    m = mjcf_io.put_model(mjm)  # lodos.mjcf.types.Model: MJWarp's model
    override_model(m, ov)  # again, for the fields only MJWarp's model has (`opt.prox_*`, `opt.shared_hessian`)
    if sim.options["shared_hessian"] is None and not any(str(x).split("=")[0].strip() == "opt.shared_hessian"
                                                         for x in overrides):
        # The default: MJWarp's Hessian summed in shared memory wherever it can be built, from 64 padded dofs
        m.opt.shared_hessian = cd_driver.shared_hessian_default(m, condim if g_geom is not None else None,
                                                           sim.options["fused_newton"], sim.options["row_space"])
    sim.options["shared_hessian"] = bool(m.opt.shared_hessian)
    caps = dict(nconmax=nconmax, njmax=njmax, nccdmax=nccdmax, naconmax=naconmax, njmax_nnz=njmax_nnz)
    # Lodos's pairs collide outside the MJCF's own geom pairs; a sparse Jacobian's nonzeros must hold their rows too
    body_pairs = ()  # (nCPR,) (int, int): each Lodos pair's bodies; () without Lodos pairs
    if g_geom is not None:
        body = np.asarray(mjm.geom_bodyid)[g_geom]  # (nB,) int32: each surface's body
        body_pairs = [(int(body[a]), int(body[b])) for a, b in surfaces.pairs]
    # lodos.mjcf.types.Data: MJWarp's data, B worlds
    d = mjcf_io.put_data(mjm, mujoco.MjData(mjm), nworld=B, contact_body_pairs=body_pairs,
                         **{k: v for k, v in caps.items() if v is not None})
    sim.m, sim.d = m, d
    sim.prim = PrimCollision(mjm, m, d, hybrid=hybrid)
    sim.geom_bodyid = wp.array(np.asarray(mjm.geom_bodyid, dtype=np.int32), dtype=int)

    # ------------------------------------------------------------------------------
    # The solver.
    lodos = None  # (mcol, dcol, (nB,) int32, (nCPR, 8) float32, int): the solver's Lodos half, or None
    if g_geom is not None:
        lodos = (sim.mcol, sim.dcol, g_geom, params13[:, _MJ_COLS], condim)
    groups = None  # dict: `lodos.cd.groups.Groups`' keyword arguments, or None
    o = sim.options  # dict: the resolved solver options
    if o["groups"] is not None:
        if g_geom is None:
            raise ValueError("groups gather Lodos's contacts: the scene has no Lodos surfaces")
        from lodos.cd import groups as cd_groups

        groups = dict(kinds=cd_groups.resolve_kinds(o["groups"], int(sim.mcol.nCPR)), params13=params13,
                      nnorm=o["normal_bins"], nmom=o["moment_bins"])
    sim.solver = cd_driver.SolverDriver(mjm, m, d, mrbd, drbd, lodos=lodos, reduce=contact_reduction,
                               fused=o["fused_newton"], groups=groups, row_space=o["row_space"])
    return sim
