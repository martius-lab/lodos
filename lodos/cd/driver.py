"""
The constraint solvers -- MJWarp's Newton solver and Lodos's prox solver around it -- on
Lodos's rigid-body state.

MJWarp's solvers work on constraint rows assembled from `d.contact`, against the
mass matrix in `d.M`, starting from `qacc_smooth`. All three are made available here
from Lodos's side:

**State is shared, not copied.** `d.qpos`, `d.qvel`, `d.cdof`, `d.subtree_com`, `d.xpos`,
`d.xmat`, `d.qfrc_smooth`, `d.qacc_smooth` and `d.qacc` are rebound to the arrays
`lodos.rbd` owns; the layouts agree (batch first, MuJoCo's quaternion order,
`cdof` as [angular, linear]). The solver therefore writes the constrained
acceleration straight into the array Lodos's integrator reads, and
`fwd_acceleration_con` has nothing to do.

**The mass matrix is gathered.** `lodos.rbd` keeps one dense block per kinematic
tree, batch last; MJWarp reads MuJoCo's CSR `M` (each row's ancestor dofs, then
the diagonal). `gather_M` copies the one into the other, entry for entry.

**Lodos's contacts join MJWarp's.** MJWarp's collision has already written its own
contacts when this runs; `export_lodos` appends every Lodos contact with the
parameters of its Lodos pair -- friction, solref, solimp, condim -- and the MJCF
geoms its two surfaces are keyed to, so the Jacobian is formed on the right bodies.
The exact mode uses the friction and condim alone: it rewrites every contact
row's weight and reference, so no solref or solimp reaches its solve (`lodos.sim.make_sim`).
With `contact_reduction` it appends only the ones `lodos.cd.reduce` keeps, Newton's
reduction of a pair's contacts to at most 240 that hold it the same way.
They are typed `NOCARRY`: prox's cross-step multiplier cache leaves them out.
The geom of the SDF side goes first: MJWarp's normal points from `geom[0]` to
`geom[1]`, and Lodos's points out of the SDF surface, towards the vertex one.
Symmetric pairs export both directions: the rows are constraints, and the solve
distributes force among them.

This is Lodos's contact solver, the NLSP solver -- a nonsmooth nonlinear least-squares solver with
proximal minimization -- (`resolve_options`): under `mode="mj"` with no proximal loop and no de Saxce
term, MJWarp's Newton solver on MuJoCo's problem; otherwise the same solver inside the prox solver's loop
(`opt.prox_enable`, `lodos/cd/solver/nlsp.py`), on the exact problem's rows or on
MuJoCo's soft ones (`opt.prox_soft`). At its defaults the exact mode runs ten outer
proximal iterations, re-taking the de Saxce term after each. It warm starts by default, from
the previous step's acceleration and, around a proximal loop, its carried multipliers;
`warm_start=False` starts every step cold. With `fused_newton` the solve is `lodos.cd.fused`'s:
MJWarp's Newton and the proximal rounds around it as one kernel per world, for small models.

With `groups`, the contacts of the pairs chosen are not exported one by one: `lodos.cd.groups`
gathers each segment's contacts in a normal bin and moment sector into one MJWarp contact typed `GROUP`,
whose rows, weights, references and prox terms MJWarp builds from its members. With `row_space`, MJWarp's
Newton takes its search direction from `lodos.cd.rowspace`, which solves the Newton system through the
rows rather than forming the nv x nv Hessian, which lets groups' fewer rows pay on models of many dofs.
Wherever it can on a model of 48 dofs or more (`shared_hessian`), MJWarp forms that Hessian, on its sparse
path, per world in shared memory (`opt.shared_hessian`) rather than in global memory every iteration.
"""

from __future__ import annotations

import numpy as np
import warp as wp


from lodos.cd import reduce as cd_reduce
from lodos.cd.solver import constraint
from lodos.cd.solver import nlsp
from lodos.col.prim.driver import ContactOut, contact_out
from lodos.mjcf.types import ConeType
from lodos.mjcf.types import ContactType
from lodos.mjcf.types import DisableBit
from lodos.mjcf.types import IntegratorType
from lodos.mjcf.types import vec5
from lodos.utils.math import make_frame

wp.set_module_options({"enable_backward": False})

MODES = ("exact", "mj")  # the solver's modes (`resolve_options`)
GROUP_KINDS = ("contact", "ray", "cone")  # `lodos.cd.groups.KINDS`
# The solver's options, `lodos.sim.make_sim`'s `solver_kwargs` (`resolve_options`).
OPTIONS = ("mode", "prox_rho", "prox_iters_per_qp", "prox_tol_z", "prox_tol_sigma", "baumgarte", "baumgarte_alpha",
           "baumgarte_slop", "de_saxce", "refresh_de_saxce_per_prox", "warm_start", "fused_newton", "groups",
              "normal_bins", "moment_bins", "row_space", "shared_hessian")
# The penetration the exact mode's Baumgarte target leaves alone (m) (`baumgarte_slop`).
# A contact exists only while it penetrates: pushed out to zero, a resting one vanishes, its
# body drops a step and it comes back, so bodies at rest vibrate.
BAUMGARTE_SLOP = 2e-4


def resolve_options(mode="exact", prox_rho=1.0, prox_iters_per_qp=None, prox_tol_z=1e-3, prox_tol_sigma=1e-3,
                       baumgarte=None, baumgarte_alpha=None, baumgarte_slop=None, de_saxce=None,
                       refresh_de_saxce_per_prox=None, warm_start=True, fused_newton=False, groups=None,
                       normal_bins=None, moment_bins=None, row_space=False, shared_hessian=None):
    """
    The solver's options, checked and with every default filled in.

    mode: "exact", the exact problem with no MuJoCo impedance -- every contact row's
        weight is the proximal one, `D = 1 / (prox_rho invweight)`, and its reference
        Baumgarte's (`baumgarte`, `baumgarte_alpha`, and `baumgarte_slop`, the penetration it
        leaves alone, `BAUMGARTE_SLOP` by default) -- or "mj", MuJoCo's soft
        rows from solref/solimp, which refuses those three.
    prox_rho, prox_iters_per_qp: the proximal loop around the solve -- `rho_hat` relative to
        a contact's translational invweight, and its outer iterations (None: 10 under exact, 0
        under mj). The exact mode's rows have no softness of their own, so it needs the loop.
        Under mj the loop keeps MuJoCo's softness beside its weight, `D = 1 / (R + rho_i)`, and
        converges on MuJoCo's soft solution.
    prox_tol_z, prox_tol_sigma: the loop's relative tolerances (`opt.prox_tol_z`,
        `opt.prox_tol_sigma`), 1e-3 each: a world stops in the first round in which its anchor
        moved by less than `prox_tol_z` and, if the round re-took the de Saxce term, the term
        by less than `prox_tol_sigma`, each relative to its size.
    de_saxce: the de Saxce term `mu |sigma_t|` in each contact's normal target; None is on
        under exact and off under mj, where it shifts MuJoCo's reference through the row's own
        damping, `-b Gamma`.
    refresh_de_saxce_per_prox: the term re-taken after every outer iteration at the velocity
        that iterate's force leaves; else taken once, at the warm start's. None: wherever
        there is a term and a loop to refresh it in.
    warm_start: MuJoCo's warm start of the acceleration, and with a loop the proximal anchor
        carried across steps.
    fused_newton: the solve -- MJWarp's Newton, and under the exact mode the proximal
        rounds around it -- as one kernel per world (`lodos.cd.fused`), for small models (nv <= 16);
        off by default. It solves the same problem to the same iterates up to rounding. Under
        mj it takes MuJoCo's problem as it stands: no proximal loop, no de Saxce term.
    groups: contact groups (`lodos.cd.groups`), off by default: None or "contact", every
        Lodos pair per contact; "cone" or "ray", every pair grouped so; or one of "contact", "ray",
        "cone" per Lodos pair. A group is a pair segment's contacts in one normal bin (`normal_bins`)
        and moment sector (`moment_bins`, `lodos.cd.bins`), 1 and 1 by default, solved as one super-contact
        under either mode, by MJWarp's solve or `fused_newton`'s. With `contact_reduction` a group
        is made of the contacts the reduction keeps.
    row_space: the Newton system solved through the constraint rows (`lodos.cd.rowspace`) rather than
        as MJWarp's dense nv x nv Hessian, off by default: the same iterates, at a cost set by a world's
        active rows (with contact groups, a few dozen) instead of its dofs. Its memory grows with the
        row capacity `njmax` (rows x nv, and rows^2 up to 1024 rows), so it wants a tuned one at large B.
        Not with `fused_newton`.
    shared_hessian: MJWarp's Newton Hessian on its sparse path -- a model of more than 32 dofs -- summed per
        world in shared memory and written once, rather than M re-densified and the rows added in global
        memory every iteration (`opt.shared_hessian`): the same Hessian up to the order of sums. None, the
        default, turns it on wherever MJWarp can build it on a model of 48 dofs or more, 64 padded
        (`make_sim`, `shared_hessian_default`): below, it measured slower; True asks for it, and `make_sim`
        refuses a model it cannot build; False turns it off. It takes up to 144 padded dofs and condim 3 under the
        elliptic cone; not with `fused_newton` or `row_space`, which form no such Hessian.

    Called by `lodos.sim.make_sim` and `viewer.session.mode_options`. In: the options above: Python scalars,
    strings or None, and for `groups` a sequence of kinds. Out: dict keyed by `OPTIONS`, every default filled in: the
    three Baumgarte options None under mj, `groups` None where every pair is per contact, `shared_hessian` None left
    for `make_sim` to decide.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}, expected one of {MODES}")
    mj = mode == "mj"  # bool
    if mj and (baumgarte is not None or baumgarte_alpha is not None or baumgarte_slop is not None):
        raise ValueError("the mj mode takes MuJoCo's solref/solimp: no baumgarte, baumgarte_alpha or "
                         "baumgarte_slop")
    if baumgarte_slop is not None and float(baumgarte_slop) < 0.0:
        raise ValueError(f"baumgarte_slop must be >= 0, got {baumgarte_slop}")
    iters = (0 if mj else 10) if prox_iters_per_qp is None else int(prox_iters_per_qp)  # int: the loop's iterations
    if iters < 0:
        raise ValueError(f"prox_iters_per_qp must be >= 0, got {iters}")
    loop = iters > 0 and float(prox_rho) > 0.0  # bool: a proximal loop runs
    if not mj and not loop:
        raise ValueError("the exact mode has rows with no softness of their own: it needs the proximal loop "
                         "(prox_iters_per_qp > 0, prox_rho > 0)")
    ds = (not mj) if de_saxce is None else bool(de_saxce)  # bool: the de Saxce term
    if refresh_de_saxce_per_prox and not ds:
        raise ValueError("refresh_de_saxce_per_prox re-takes the de Saxce term: it needs de_saxce")
    if refresh_de_saxce_per_prox and not loop:
        raise ValueError("refresh_de_saxce_per_prox re-takes the term between outer proximal iterations: it needs a "
                         "proximal loop (prox_iters_per_qp > 0, prox_rho > 0)")
    refresh = (ds and loop) if refresh_de_saxce_per_prox is None else bool(refresh_de_saxce_per_prox)  # bool
    if groups is not None and not isinstance(groups, str):
        groups = tuple(groups)
    kinds = (groups,) if isinstance(groups, str) else (groups or ())  # tuple of str: the kinds asked for
    for k in kinds:  # k: str, a kind
        if k not in GROUP_KINDS:
            raise ValueError(f"groups: unknown kind {k!r}, expected one of {GROUP_KINDS}")
    if all(k == "contact" for k in kinds):
        groups = None
    if groups is None and (normal_bins is not None or moment_bins is not None):
        raise ValueError("normal_bins and moment_bins bin contact groups: they need groups")
    nbins = 1 if normal_bins is None else int(normal_bins)  # int
    mbins = 1 if moment_bins is None else int(moment_bins)  # int
    if groups is not None:
        from lodos.cd import groups as cd_groups

        cd_groups.check_bins(nbins, mbins)
    if row_space and fused_newton:
        raise ValueError("row_space and fused_newton are two solves of the Newton system: one at a time")
    if shared_hessian and fused_newton:
        raise ValueError("fused_newton sums its own Hessian in its kernel: shared_hessian builds MJWarp's sparse one")
    if shared_hessian and row_space:
        raise ValueError("row_space solves the Newton system through the rows and forms no Hessian for "
                         "shared_hessian to build")
    if fused_newton and mj and (loop or ds):
        raise ValueError("fused_newton runs the mj mode as MuJoCo's problem stands: no proximal loop "
                         "(prox_iters_per_qp 0) and no de Saxce term")
    return dict(mode=mode, prox_rho=float(prox_rho), prox_iters_per_qp=iters, prox_tol_z=float(prox_tol_z),
                prox_tol_sigma=float(prox_tol_sigma),
                baumgarte=None if mj else (True if baumgarte is None else bool(baumgarte)),
                baumgarte_alpha=None if mj else (0.1 if baumgarte_alpha is None else float(baumgarte_alpha)),
                baumgarte_slop=None if mj else (BAUMGARTE_SLOP if baumgarte_slop is None else float(baumgarte_slop)),
                de_saxce=ds, refresh_de_saxce_per_prox=refresh, warm_start=bool(warm_start),
                fused_newton=bool(fused_newton), groups=groups, normal_bins=nbins, moment_bins=mbins,
                row_space=bool(row_space), shared_hessian=None if shared_hessian is None else bool(shared_hessian))


def prox_path(options):
    """
    Whether resolved options run the prox solver: anything but MuJoCo's problem as it stands.

    Called by `overrides`. In: options, the dict `resolve_options` returns. Out: bool.
    """
    o = options  # dict
    return o["mode"] == "exact" or (o["prox_iters_per_qp"] > 0 and o["prox_rho"] > 0.0) or o["de_saxce"]


def overrides(options):
    """
    The MJWarp model overrides resolved options come to (`opt.prox_*`).

    None for MuJoCo's own problem, where the prox path does not run: MJWarp's Newton solver on
    the model's rows and cone. Otherwise the elliptic cone the prox solver
    needs, whatever the model's -- under mj a loop or the term moves a pyramidal model's problem
    onto it -- and the loop as `outer x inner` rounds: `refresh_de_saxce_per_prox` re-takes the term
    at every round's end, `prox_iters_per_qp x 1`; without it the term is taken once, from the
    warm start, `1 x prox_iters_per_qp`, and every round tests the anchor alone. Under mj
    without a loop, one round with no proximal term (`prox_rho` 0) carries the de Saxce term
    alone. At its defaults the exact mode comes to `10 x 1` rounds, the anchor carried from step
    to step, with the Baumgarte slop. `shared_hessian=True` comes to `opt.shared_hessian`, under
    either mode; its default is decided on MJWarp's model (`make_sim`, `shared_hessian_default`).

    Called by `lodos.sim.make_sim`. In: options, the dict `resolve_options` returns.
    Out: tuple of str, each `"opt.<field>=<value>"`.
    """
    o = options  # dict
    shared = ("opt.shared_hessian=true",) if o["shared_hessian"] else ()  # tuple of str
    if not prox_path(o):
        return shared
    mj = o["mode"] == "mj"  # bool
    loop = o["prox_iters_per_qp"] > 0 and o["prox_rho"] > 0.0  # bool: a proximal loop runs
    n = o["prox_iters_per_qp"] if loop else 1  # int: the rounds
    outer, inner = (n, 1) if o["refresh_de_saxce_per_prox"] else (1, n)  # int, int: the rounds as outer x inner
    ov = ["opt.cone=elliptic", "opt.prox_enable=1", f"opt.prox_rho={o['prox_rho'] if loop else 0.0}",  # list of str
          f"opt.prox_outer_iterations={outer}", f"opt.prox_inner_iterations={inner}",
          f"opt.prox_z_carry={1.0 if o['warm_start'] and loop else 0.0}",
          f"opt.prox_tol_z={o['prox_tol_z']}", f"opt.prox_tol_sigma={o['prox_tol_sigma']}",
          f"opt.prox_desaxce={str(o['de_saxce']).lower()}", f"opt.prox_soft={str(mj).lower()}"]
    if not mj:
        ov += [f"opt.prox_baumgarte={str(o['baumgarte']).lower()}", f"opt.prox_baumgarte_alpha={o['baumgarte_alpha']}",
               f"opt.prox_baumgarte_slop={o['baumgarte_slop']}"]
    return tuple(ov) + shared


def shared_hessian_refusal(m, condim=None, fused_newton=False, row_space=False):
    """
    Why MJWarp's Hessian cannot be summed in shared memory (`opt.shared_hessian`) on its model `m`;
    empty if it can. MJWarp's reasons (`lodos.cd.solver.nlsp.shared_hessian_refusal`: CG, the
    dense path, elliptic contacts above condim 3, more than 144 padded dofs, sleeping), then
    Lodos's: its own contacts above condim 3 under the elliptic cone (`condim`; None without Lodos
    surfaces), which MJWarp's model does not count, and the solves that form no such Hessian.

    Called by `shared_hessian_default` and `SolverDriver.__init__`. In: m, MJWarp's `Model` (its `opt.cone` and what
    `nlsp.shared_hessian_refusal` reads); condim int or None; fused_newton, row_space bool. Out: str.
    """
    reason = nlsp.shared_hessian_refusal(m)  # str: empty if it can
    if not reason and condim is not None and int(condim) > 3 and m.opt.cone == ConeType.ELLIPTIC:
        reason = f"it takes the elliptic cone's curvature for condim <= 3: Lodos's contacts have condim {int(condim)}"
    if not reason and fused_newton:
        reason = "fused_newton sums its own Hessian in its kernel"
    if not reason and row_space:
        reason = "row_space solves the Newton system through the rows and forms no Hessian"
    return reason


# The default turns the shared Hessian on from this many padded dofs (MJWarp's `nv_pad`, a sparse model's dofs plus
# one rounded up to 16: 48 dofs or more). Below, it measured slower, and from 64 faster.
SHARED_HESSIAN_MIN_NV_PAD = 64


def shared_hessian_default(m, condim=None, fused_newton=False, row_space=False):
    """
    What `shared_hessian=None` chooses on MJWarp's model `m`: on where the Hessian can be summed in shared memory
    (`shared_hessian_refusal`, which takes the same arguments) on a model of `SHARED_HESSIAN_MIN_NV_PAD` padded
    dofs or more, off elsewhere.

    Called by `lodos.sim.make_sim`. In: m, MJWarp's `Model` (`nv_pad`); the rest as `shared_hessian_refusal`'s.
    Out: bool.
    """
    return int(m.nv_pad) >= SHARED_HESSIAN_MIN_NV_PAD and not shared_hessian_refusal(m, condim, fused_newton, row_space)

# MuJoCo's defaults for the friction coefficients a Lodos pair does not carry.
TORSIONAL = 0.005
ROLLING = 0.0001


@wp.struct
class LodosContacts:
    """
    What `export_lodos` reads: Lodos's contact list and its pairs' parameters.

    Built by `SolverDriver.__init__`.
    """

    ct: wp.array(dtype=int)  # (5,) `dcol.ct`, [nCa1, nCa2, nCa3, nTa, nTb]: ct[2] the contacts
    b_nCa3: wp.array(dtype=int)  # (nCa3_max,) a contact's world
    seg_nCa3: wp.array(dtype=int)  # (nCa3_max,) its segment
    p_nCa3: wp.array(dtype=wp.vec3)  # (nCa3_max,) its world position
    n_nCa3: wp.array(dtype=wp.vec3)  # (nCa3_max,) its normal, out of the SDF surface
    sdf_nCa3: wp.array(dtype=float)  # (nCa3_max,) its depth, < 0
    seg_cpr: wp.array(dtype=int)  # (nSEG,) a segment's pair
    seg_g_v: wp.array(dtype=int)  # (nSEG,) its geometry supplying the vertices
    seg_g_s: wp.array(dtype=int)  # (nSEG,) its geometry supplying the SDF
    g_geom: wp.array(dtype=int)  # (nB,) MJCF geom each Lodos geometry is keyed to
    cpr_friction: wp.array(dtype=vec5)  # (nCPR,) a pair's MJWarp friction: mu, mu, torsional, rolling, rolling
    cpr_solref: wp.array(dtype=wp.vec2)  # (nCPR,) its solref
    cpr_solimp: wp.array(dtype=vec5)  # (nCPR,) its solimp
    condim: int  # every Lodos contact's condim
    reduced: int  # 1: export only the contacts `keep` marks
    keep: wp.array(dtype=wp.uint8)  # (nCa3_max,) from `lodos.cd.reduce`
    grouped: int  # 1: skip the contacts `lodos.cd.groups` keyed (key >= 0): their groups export them
    key: wp.array(dtype=int)  # (nCa3_max,) `GroupData.key_nCa3`


# A Lodos contact is a constraint, and has no identity the prox solver could
# carry its multipliers by from step to step: a pair holds dozens of vertex contacts
# under one geom pair, and the cache's nearest-position match within a geom pair would
# hand one cached force to many of them. They start each step cold.
LODOS_CONTACT = int(ContactType.CONSTRAINT | ContactType.NOCARRY)
# A `toa` pair's contact at a positive gap: rows of its own (`export_lodos_toa`).
LODOS_SPECULATIVE = int(LODOS_CONTACT | ContactType.SPECULATIVE)


@wp.kernel
def export_lodos(lc: LodosContacts, co: ContactOut):  # type: ignore
    """
    Appends Lodos's contacts to MJWarp's arrays: `collision_core.write_contact`'s
    writes, for contacts that are all active (`sdf < 0`, zero margin and gap) and
    typed `LODOS_CONTACT`.

    Launch: dim=nCa3_max, a thread per Lodos contact slot, by `SolverDriver.contact_dynamics`.
    Reads: lc, `LodosContacts`: the contacts (nCa3_max,), their segments' pairs and geometries (nSEG,), g_geom (nB,),
    the pairs' friction, solref and solimp (nCPR,), keep and key (nCa3_max,). Writes: co, `ContactOut`: co.nacon (1,)
    int32, added to, and the slots it hands out in co's (naconmax,) contact arrays and (naconmax, nmaxpyramid)
    efc_address.
    """
    i = wp.tid()  # int: the Lodos contact
    if i >= lc.ct[2]:
        return
    if lc.reduced != 0:
        if lc.keep[i] == wp.uint8(0):
            return
    if lc.grouped != 0:
        if lc.key[i] >= 0:
            return
    sg = lc.seg_nCa3[i]  # int: its segment
    j = lc.seg_cpr[sg]  # int: its pair
    cid = wp.atomic_add(co.nacon, 0, 1)  # int: its MJWarp contact
    if cid >= co.naconmax:
        return
    co.dist[cid] = lc.sdf_nCa3[i]
    co.pos[cid] = lc.p_nCa3[i]
    co.frame[cid] = make_frame(lc.n_nCa3[i])
    co.geom[cid] = wp.vec2i(lc.g_geom[lc.seg_g_s[sg]], lc.g_geom[lc.seg_g_v[sg]])
    co.worldid[cid] = lc.b_nCa3[i]
    co.includemargin[cid] = 0.0
    co.dim[cid] = lc.condim
    co.friction[cid] = lc.cpr_friction[j]
    co.solref[cid] = lc.cpr_solref[j]
    co.solreffriction[cid] = wp.vec2(0.0, 0.0)
    co.solimp[cid] = lc.cpr_solimp[j]
    co.adhesion[cid] = 0.0
    co.type[cid] = LODOS_CONTACT
    co.geomcollisionid[cid] = 0
    for k in range(co.efc_address.shape[1]):  # k: int, an efc_address column
        co.efc_address[cid, k] = -1


@wp.kernel
def export_lodos_toa(lc: LodosContacts, co: ContactOut):  # type: ignore
    """
    `export_lodos` for a context with a `toa` pair: a contact at a positive gap, a speculative one
    (`lodos.col.sdf.narrowphase`), is typed `SPECULATIVE` as well, so that the solve gives it rows
    (`constraint._efc_contact_init` under the model's `flg_speculative`).

    Launch, Reads, Writes: as `export_lodos`, by `SolverDriver.contact_dynamics`.
    """
    i = wp.tid()  # int: the Lodos contact
    if i >= lc.ct[2]:
        return
    if lc.reduced != 0:
        if lc.keep[i] == wp.uint8(0):
            return
    if lc.grouped != 0:
        if lc.key[i] >= 0:
            return
    sg = lc.seg_nCa3[i]  # int: its segment
    j = lc.seg_cpr[sg]  # int: its pair
    cid = wp.atomic_add(co.nacon, 0, 1)  # int: its MJWarp contact
    if cid >= co.naconmax:
        return
    co.dist[cid] = lc.sdf_nCa3[i]
    co.pos[cid] = lc.p_nCa3[i]
    co.frame[cid] = make_frame(lc.n_nCa3[i])
    co.geom[cid] = wp.vec2i(lc.g_geom[lc.seg_g_s[sg]], lc.g_geom[lc.seg_g_v[sg]])
    co.worldid[cid] = lc.b_nCa3[i]
    co.includemargin[cid] = 0.0
    co.dim[cid] = lc.condim
    co.friction[cid] = lc.cpr_friction[j]
    co.solref[cid] = lc.cpr_solref[j]
    co.solreffriction[cid] = wp.vec2(0.0, 0.0)
    co.solimp[cid] = lc.cpr_solimp[j]
    co.adhesion[cid] = 0.0
    co.type[cid] = LODOS_CONTACT
    if lc.sdf_nCa3[i] >= 0.0:  # no gap left to close is a row too: `pos < 0` alone would drop it
        co.type[cid] = LODOS_SPECULATIVE
    co.geomcollisionid[cid] = 0
    for k in range(co.efc_address.shape[1]):  # k: int, an efc_address column
        co.efc_address[cid, k] = -1


@wp.kernel
def gather_M(
    M_map: wp.array(dtype=int),  # (nM,) row of M_blk holding each CSR entry # type: ignore
    M_blk: wp.array(dtype=float, ndim=2),  # (nblk, B) # type: ignore
    M: wp.array(dtype=float, ndim=2),  # (B, nM) out # type: ignore
):
    """
    MuJoCo's CSR `M` from Lodos's per-tree blocks, entry for entry.

    Launch: dim=(B, nM), a thread per world and CSR entry, by `SolverDriver.contact_dynamics`.
    Reads: M_map (nM,) int32, M_blk (nblk, B) float32. Writes: M (B, nM) float32.
    """
    b, e = wp.tid()  # int, int: the world, the CSR entry
    M[b, e] = M_blk[M_map[e], b]


def csr_map(mjm, mrbd):
    """
    For each entry of MuJoCo's CSR `M`, its row in Lodos's per-tree `M_blk`.

    Called by `SolverDriver.__init__`. In: mjm, MuJoCo's `MjModel` (`nv`, `nC`, `M_rowadr`,
    `M_rownnz`, `M_colind`); mrbd, `RbdModel` (`tree_beg` and `blk_beg`, (ntree + 1,) int32). Out: (nC,) int32;
    MuJoCo's `nC`, the CSR's entries, equals `nM`.
    """
    tree_beg = mrbd.tree_beg.numpy()  # (ntree + 1,) int32: each tree's first dof
    blk_beg = mrbd.blk_beg.numpy()  # (ntree + 1,) int32: each tree's first entry in M_blk
    nv = int(mjm.nv)  # int
    dof_tree = np.zeros(nv, dtype=int)  # (nv,) int64: each dof's tree
    for t in range(len(tree_beg) - 1):  # t: int, a tree
        dof_tree[tree_beg[t]:tree_beg[t + 1]] = t
    out = np.zeros(int(mjm.nC), dtype=np.int32)  # (nC,) int32
    for i in range(nv):  # i: int, a dof: a row of M
        t = dof_tree[i]
        n = tree_beg[t + 1] - tree_beg[t]  # int: the tree's dofs, its block's side
        for e in range(mjm.M_rowadr[i], mjm.M_rowadr[i] + mjm.M_rownnz[i]):  # e: int, a CSR entry of row i
            j = int(mjm.M_colind[e])  # int: its column
            assert dof_tree[j] == t, "a CSR entry couples two kinematic trees"
            out[e] = blk_beg[t] + (i - tree_beg[t]) * n + (j - tree_beg[t])
    return out


def bind_state(d, drbd):
    """
    Points MJWarp's state fields at Lodos's arrays. See the module docstring.

    Called by `SolverDriver.__init__`. In: d, MJWarp's `Data`; drbd, `RbdData`. Out: d.qpos (B, nq), d.qvel,
    d.qfrc_smooth, d.qacc_smooth and d.qacc (B, nv) float32, d.cdof (B, nv) wp.spatial_vector, d.subtree_com and
    d.xpos (B, nbody) wp.vec3, d.xmat (B, nbody) wp.mat33, rebound to drbd's arrays.
    """
    # name: str, a state field both sides hold
    for name in ("qpos", "qvel", "cdof", "subtree_com", "xpos", "xmat", "qfrc_smooth", "qacc_smooth", "qacc"):
        mine, theirs = getattr(d, name), getattr(drbd, name)  # wp.array, wp.array: MJWarp's field, Lodos's
        if tuple(mine.shape) != tuple(theirs.shape):
            raise ValueError(f"{name}: MJWarp's shape {mine.shape} is not Lodos's {theirs.shape}")
        if wp.types.type_size_in_bytes(mine.dtype) != wp.types.type_size_in_bytes(theirs.dtype):
            raise ValueError(f"{name}: MJWarp's {mine.dtype} is not Lodos's {theirs.dtype}")
        setattr(d, name, theirs)


def check_models(m, mrbd):
    """
    Refuses an MJWarp model and a rigid-body model that disagree on what both step with: the timestep, gravity,
    the disable flags `lodos.rbd` honours, and the integrator. `lodos.sim.make_sim` builds both from one
    `MjModel` with its overrides applied, so they disagree only where one was built or changed apart.

    Called by `SolverDriver.__init__`. In: m, MJWarp's `Model`; mrbd, `RbdModel`. Raises ValueError, naming each
    disagreement.
    """
    bad = []  # list of str: each disagreement, MJWarp's value then Lodos's
    dt = m.opt.timestep.numpy()  # (1 or nworld,) float32: MJWarp's timestep
    if not np.all(dt == np.float32(mrbd.dt)):
        bad.append(f"timestep {dt.tolist()} against {mrbd.dt}")
    g = m.opt.gravity.numpy()  # (1 or nworld, 3) float32: MJWarp's gravity
    g_rbd = np.array([float(x) for x in mrbd.gravity], dtype=np.float32)  # (3,) float32: Lodos's
    if not np.all(g == g_rbd):
        bad.append(f"gravity {g.tolist()} against {g_rbd.tolist()}")
    for name in ("spring", "damper", "gravity", "clampctrl", "actuation"):  # name: str, a flag lodos.rbd honours
        mine = bool(int(m.opt.disableflags) & int(DisableBit[name.upper()]))  # bool: MJWarp's
        theirs = bool(getattr(mrbd, "disable_" + name))  # bool: Lodos's
        if mine != theirs:
            bad.append(f"disable {name} {mine} against {theirs}")
    integrator = IntegratorType(int(m.opt.integrator)).name.lower()  # str: MJWarp's integrator
    integrator_rbd = "implicitfast" if mrbd.implicit else "euler"  # str: Lodos's
    if integrator != integrator_rbd:
        bad.append(f"integrator {integrator} against {integrator_rbd}")
    if bad:
        raise ValueError("MJWarp's model and Lodos's rigid-body model disagree (MJWarp's value first): "
                         + "; ".join(bad))


class SolverDriver:
    """
    MJWarp's Newton solve, or the prox solve around it, as one Lodos contact-dynamics step.

    m, d: MJWarp's model and data, shared with `lodos.col.prim.driver.PrimCollision`.
    lodos: None, or `(mcol, dcol, g_geom, params, condim)` -- Lodos's collision
        context, the MJCF geom each of its geometries is keyed to, and per-pair
        `[mu, solref(2), solimp(5)]` rows.
    reduce: export Lodos's contacts through `lodos.cd.reduce` rather than all of them.
    fused: solve with `lodos.cd.fused.FusedNewton`, one kernel per world, rather than
        MJWarp's solve (`lodos.cd.solver.nlsp.solve`).
    groups: None, or `lodos.cd.groups.Groups`' keyword arguments (`kinds` one per pair,
        `params13` the pairs' thirteen columns, `nnorm` and `nmom` the bins).
    row_space: MJWarp's Newton takes its search direction from `lodos.cd.rowspace.RowSpaceNewton`.

    Built by `lodos.sim.make_sim`.
    """

    def __init__(self, mjm, m, d, mrbd, drbd, lodos=None, reduce=False, fused=False, groups=None, row_space=False):
        """
        Binds MJWarp's state to Lodos's, builds the export of Lodos's contacts and the solve chosen, and refuses the
        combinations they do not take, and models that disagree (`check_models`).

        Called by `lodos.sim.make_sim`. In: mjm, MuJoCo's `MjModel`; m, d, MJWarp's `Model` and `Data`; mrbd, drbd,
        Lodos's `RbdModel` and `RbdData`; the rest as the class docstring has them.
        """
        check_models(m, mrbd)
        self.m, self.d, self.mrbd, self.drbd = m, d, mrbd, drbd  # MJWarp's model and data, Lodos's rigid-body ones
        bind_state(d, drbd)
        self.M_map = wp.array(csr_map(mjm, mrbd), dtype=int)  # (nC,) int32: each CSR entry's row of M_blk
        self.co = contact_out(d)  # ContactOut: MJWarp's contact arrays
        self.lc = None  # LodosContacts, or None without Lodos surfaces
        # bool: some Lodos pair is `toa`, whose speculative contacts the export types as such
        self.toa = lodos is not None and int(lodos[0].has_toa) == 1
        self.rm = self.rd = None  # ReduceModel and ReduceData under the reduction, else None
        self.groups = None  # lodos.cd.groups.Groups where some pair is grouped, else None
        if lodos is not None:
            # CollisionModel, CollisionData, (nB,) int32, (nCPR, 8) float32, int
            mcol, dcol, g_geom, params, condim = lodos
            params = np.asarray(params, dtype=np.float32).reshape(int(mcol.nCPR), 8)
            lc = LodosContacts()  # LodosContacts: what export_lodos reads
            for f in ("ct", "b_nCa3", "seg_nCa3", "p_nCa3", "n_nCa3", "sdf_nCa3"):  # f: str, a field
                setattr(lc, f, getattr(dcol, f))
            for f in ("seg_cpr", "seg_g_v", "seg_g_s"):
                setattr(lc, f, getattr(mcol, f))
            lc.g_geom = wp.array(np.asarray(g_geom, dtype=np.int32), dtype=int)
            mu = params[:, 0]  # (nCPR,) float32: the pairs' friction
            # (nCPR, 5) float32: MJWarp's friction coefficients, sliding (2), torsional and rolling (2)
            fr = np.stack([mu, mu, np.full_like(mu, TORSIONAL), np.full_like(mu, ROLLING), np.full_like(mu, ROLLING)], 1)
            lc.cpr_friction = wp.array(fr, dtype=vec5)
            lc.cpr_solref = wp.array(params[:, 1:3], dtype=wp.vec2)
            lc.cpr_solimp = wp.array(params[:, 3:8], dtype=vec5)
            lc.condim = int(condim)
            lc.reduced = int(bool(reduce))
            lc.grouped = 0
            lc.key = wp.zeros(1, dtype=int)
            if reduce:
                self.rm, self.rd = cd_reduce.make_reduce_context(mcol, dcol)
                lc.keep = self.rd.keep
            else:
                lc.keep = wp.zeros(1, dtype=wp.uint8)
            if groups is not None:
                from lodos.cd import groups as cd_groups

                # The reduction keeps the grouped pairs' contacts too: their groups are made of those it keeps.
                gr = cd_groups.Groups(mjm, m, d, drbd, mcol, dcol, g_geom, friction=lc.cpr_friction,  # Groups
                                    solref=lc.cpr_solref, solimp=lc.cpr_solimp,
                                    keep=self.rd.keep if reduce else None, **groups)
                if gr.active:
                    self.groups = gr
                    lc.grouped = 1
                    lc.key = gr.gd.key_nCa3
            self.lc = lc
            self.dcol = dcol  # CollisionData: Lodos's contacts
            self.nCa3_max = int(dcol.nCa3_max)  # int: Lodos's contact capacity, export_lodos's launch
            if fused and int(condim) > 3:
                raise ValueError("fused_newton takes contacts of condim 3 at most")
        self.fused = None  # lodos.cd.fused.FusedNewton under fused_newton, else None
        if fused:
            from lodos.cd import fused as cd_fused

            self.fused = cd_fused.FusedNewton(mjm, m, d)
        self.rowspace = None  # lodos.cd.rowspace.RowSpaceNewton under row_space, else None
        if row_space:
            if fused:
                raise ValueError("row_space and fused_newton are two solves of the Newton system: one at a time")
            from lodos.cd import rowspace as cd_rowspace

            self.rowspace = cd_rowspace.RowSpaceNewton(mjm, m, d, mrbd, drbd)
        if m.opt.shared_hessian:
            reason = shared_hessian_refusal(m, None if lodos is None else lodos[4], fused, row_space)  # str
            if reason:
                raise ValueError(f"shared_hessian: {reason}")

    def contact_dynamics(self):
        """
        Lodos's contacts in, M gathered, rows assembled, solved. Capturable.

        Called by `lodos.step.forward`, every step, and `tools.performance_tuning.profile_scene.stages_of`.
        Reads: Lodos's contacts (`dcol`), drbd.M_blk (nblk, B) float32 and the state MJWarp's rows are built from.
        Writes: Lodos's contacts appended to d.contact and d.nacon, d.M (nworld, nC) float32, d.efc, d.qacc (nworld,
        nv) float32 (Lodos's own `drbd.qacc`) and d.qacc_warmstart (nworld, nv) float32.
        """
        m, d = self.m, self.d  # MJWarp's model and data
        if self.lc is not None:
            if self.rm is not None:
                cd_reduce.reduce(self.rm, self.rd, self.dcol)
            if self.groups is not None:
                self.groups.before_export()
            wp.launch(export_lodos_toa if self.toa else export_lodos, dim=self.nCa3_max,
                      inputs=(self.lc, self.co))
            if self.groups is not None:
                self.groups.export(self.co)
        wp.launch(gather_M, dim=(d.nworld, self.M_map.shape[0]), inputs=(self.M_map, self.drbd.M_blk), outputs=(d.M,))
        constraint.make_constraint(m, d)
        if self.fused is not None:
            self.fused.solve(m, d)
        elif self.rowspace is not None:
            self.rowspace.prepare(d)
            nlsp.solve(m, d, newton_solve=self.rowspace)
        else:
            nlsp.solve(m, d)
        wp.copy(d.qacc_warmstart, d.qacc)

    def fwd_acceleration_con(self):
        """
        Nothing to do: the solve wrote `qacc`, which is Lodos's own array.

        Not called in the repository.
        """
