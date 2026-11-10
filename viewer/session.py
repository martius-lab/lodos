"""
A live session: one scene, the sims it has run under, and the takes it has made.

Nothing here draws. `Session` is what the panel drives, so everything a user can do
in the window -- load, edit the state, switch sims, change their options and
parameters, play, pause, scrub -- is a method here.

## Sims

A scene runs under one of three sims (`SIMS`): Lodos, and for comparison Newton's
MuJoCo Warp and Kamino, where `viewer/newton/`'s environment is installed
(`viewer.newton_link`).

## Runs

A `Runner` is the scene built for one configuration at one world, with three
CUDA graphs captured from it: `forward` (everything but the integrator), `integrate`
and the whole `step`. A frame is collected between the first two -- see
`viewer.collect` -- so a recorded step is `forward`, collect, `integrate`, and
a step that is not recorded is one `step` launch. Runners are kept per configuration,
so switching back to one costs nothing; the first switch builds it. Where the GPU has no
room for a build beside them, the others are freed (`Session.free_runners`) -- each keeping
its capacities and the demand it has seen -- and built again when switched back to.

Capturing never advances the state: the warm-up step that loads the kernels is undone
before the capture. A parameter that only changes a Python-side constant of the step
(MJWarp's iteration counts) is applied by recapturing; one that lives in a device
array (the pairs' friction, solref and solimp, MJWarp's tolerance) is written in place,
which a captured graph sees; one that changes what is allocated (the contact groups,
the proximal loop's rounds, a Lodos pair made two-sided or given its time of arrival) is part
of the configuration's key, so it is a different runner. The pairs' parameters are `viewer.pairs`'.

## Capacities

MJWarp's capacities start from the scene's, Lodos's collision arenas at their worst case.
A runner keeps the largest demand on each over everything it has simulated (`Runner.seen`,
from `lodos.utils.calibrate.Tracker`, captured in its step), and `Session.tune_capacities` sizes
them all to it, times one plus an overhead. A step that overflows one grows it -- to 1.5
times the demand, at least double -- the run rolling back to the last recorded frame and
stepping again, so a live session never shows a step with contacts dropped.

`Session.export_config` writes everything the panel sets on the configuration looked at --
its options, its capacities, every pair's contact parameters -- to a JSON file.

## Takes

A `Take` is a run from one state under one configuration: its frames, every
`stride`-th step, and the state after its last step (`tail`), from which it continues.

- Playing a take from a frame before its end replays what was recorded; at its end it
  continues simulating, appending frames.
- Editing the state, switching sim or changing an option does not touch the take
  being looked at. It makes a *draft*: a take that starts at the frame being looked
  at, changed by the edit, holding just that one frame (the edited state, forwarded,
  so its contacts and forces show). Further edits change the draft. Playing commits
  it as a new take; looking at another take discards it.
- Earlier takes are kept, so one state can be run under two sims or two configurations
  and the two takes scrubbed side by side.
- A take saves whole to an `.npz` (`save_take`): its frames, the state after its last step
  and its configuration. Loaded into a session of the same scene (`Session.load_take`), it
  is a take like the others, scrubbed and continued from its end.
"""

from __future__ import annotations

import copy
import dataclasses
import gc
import inspect
import json
import math
import time as _time
from pathlib import Path

import numpy as np
import warp as wp

from lodos.utils import calibrate as lcalibrate
from lodos import sim as lsim
from lodos import step as lstep
from lodos.cd import bins as cd_bins
from lodos.cd import driver as cd_driver
from lodos.cd.driver import BAUMGARTE_SLOP
from lodos.rbd import integrate as rbd_integrate
from lodos.rbd import io as rbd_io
from viewer.collect import Collector, LiveFrame
from viewer import newton_link as NL
from viewer import pairs as VP
from viewer.geometry import make_live_scene
from viewer.scenes.catalog import MODEL_OPTIONS, spline_of

# Lodos, and Newton's own sims, for comparison, where `viewer/newton/`'s environment is installed.
SIMS = ("lodos",) + (NL.SIMS if NL.available() else ())
SIM_NAMES = {"lodos": "Lodos", "mjwarp": "MJWarp", "kamino": "Kamino"}
SIM_INFO = {
    "lodos": "MJWarp's constraint solve on Lodos's and MJWarp's contacts: the exact problem by proximal point, "
             "or MuJoCo's by its Newton solver (mode)",
}
SIM_INFO.update(NL.INFO)

# Each sim's options in the panel: (name, kind, lo, hi), or (name, "choice", values,
# None) for one picked from a list. `REBUILD_OPTIONS` are part of a runner's key; the
# others are applied to a built runner.
_ON = (0, 1)
_AUTO_ON = ("auto", 0, 1)  # "auto": the solver's default, decided on the model
_BINS = (("normal_bins", "choice", cd_bins.NORMAL_BINS, None), ("moment_bins", "choice", cd_bins.MOMENT_BINS, None))
# The normal problem's mode: the exact one, or MuJoCo's soft one.
_MODE = (("mode", "choice", tuple(cd_driver.MODES), None),)
# The proximal loop's weight and iterations, and the exact mode's Baumgarte bias
# (`baumgarte`; 0 is the velocity-level target), its gain and the penetration it leaves
# alone (m) -- `EXACT_OPTIONS`, which a configuration under the mj mode leaves out.
_PROX = (("prox_rho", float, 1e-3, 100.0), ("baumgarte", "choice", _ON, None), ("baumgarte_alpha", float, 0.0, 1.0),
         ("baumgarte_slop", float, 0.0, 0.01), ("prox_iters_per_qp", int, 0, 200),
         ("prox_tol_z", float, 1e-12, 1.0), ("prox_tol_sigma", float, 1e-12, 1.0))
EXACT_OPTIONS = ("baumgarte", "baumgarte_alpha", "baumgarte_slop")
# The de Saxce term in the normal target (`de_saxce`): on by default under the exact mode,
# off under mj. Its refresh after every outer proximal iteration (`refresh_de_saxce_per_prox`)
# needs the term and a loop (`refresh_applies`); where it applies it is on by default.
_DE_SAXCE = (("de_saxce", "choice", _ON, None), ("refresh_de_saxce_per_prox", "choice", _ON, None))
# The timestep (s, the MJCF's by default), the integrator (`make_sim`'s `integrator`, the MJCF's by
# default), and under implicitfast the matrix the contact solve sees: M as MuJoCo has it (0), or
# `M - dt D` throughout (1).
_INTEGRATOR = (("timestep", float, 1e-6, 0.1), ("integrator", "choice", tuple(lsim.INTEGRATORS), None),
               ("implicit_constraints", "choice", _ON, None))
OPTIONS = {
    # `contact_reduction` (0/1): Lodos's contacts reduced as Newton does, `lodos.cd.reduce`.
    # `warm_start` (0/1): MJWarp's warm start, and around a proximal loop the carry across steps.
    # `fused_newton` (0/1): the solve as one kernel per world (`lodos.cd.fused`), small models only.
    # `groups`: every Lodos pair per contact, or its contacts gathered into ray or cone groups
    # (`lodos.cd.groups`), binned by `normal_bins` x `moment_bins`.
    # `row_space` (0/1): the Newton system solved through the rows (`lodos.cd.rowspace`).
    # `shared_hessian` (auto/0/1): MJWarp's sparse Newton Hessian summed per world in shared memory; "auto",
    # the solver's default, wherever MJWarp can build it on a model of 48 dofs or more (64 padded, up to 144).
    "lodos": _MODE + (("iterations", int, 1, 500), ("tolerance", float, NL.MJWARP_MIN_TOLERANCE, 1e-2),
                      ("ls_iterations", int, 1, 1000),
                      ("ls_tolerance", float, 1e-12, 1.0), ("cone", "choice", ("elliptic", "pyramidal"), None))
             + _PROX + _DE_SAXCE
             + (("spline_type", "choice", ("cubic", "linear"), None),
                ("contact_reduction", "choice", _ON, None), ("warm_start", "choice", _ON, None),
                ("fused_newton", "choice", _ON, None), ("groups", "choice", cd_driver.GROUP_KINDS, None)) + _BINS
             + (("row_space", "choice", _ON, None), ("shared_hessian", "choice", _AUTO_ON, None)) + _INTEGRATOR,
}
OPTIONS.update(NL.OPTIONS)
REBUILD_OPTIONS = ("mode", "cone", "spline_type", "de_saxce", "refresh_de_saxce_per_prox", "normal_bins",
                   "moment_bins", "contact_reduction", "warm_start", "integrator", "implicit_constraints", "timestep")
RECAPTURE_OPTIONS = ("iterations", "ls_iterations", "prox_rho", "baumgarte", "baumgarte_alpha", "baumgarte_slop",
                     "prox_iters_per_qp", "prox_tol_z", "prox_tol_sigma", "ls_tolerance")
# Lodos's, which decide whether MJWarp's prox path runs at all and with how many rounds --
# the path allocates its own state -- so under Lodos they are rebuild options too; the fused
# solve, a solver of its own; the contact groups, which allocate theirs; and the shared
# Hessian, a model option the step is captured with.
LODOS_REBUILD_OPTIONS = ("prox_iters_per_qp", "prox_rho", "fused_newton", "groups", "row_space", "shared_hessian")
# The options a mode decides (`mode_options`), which start over when it changes.
MODE_OPTIONS = ("prox_rho", "baumgarte", "baumgarte_alpha", "baumgarte_slop", "prox_iters_per_qp", "prox_tol_z",
                "prox_tol_sigma", "cone")
# The convergence options' defaults, for the take labels.
CONVERGENCE = ("prox_tol_z", "prox_tol_sigma")
CONVERGENCE_DEFAULTS = {k: inspect.signature(cd_driver.resolve_options).parameters[k].default for k in CONVERGENCE}

# The way each option makes a sim faster: "^" larger, "v" smaller, "<>" an optimum between, "-> x" the value x.
# `FASTER` holds Lodos's options, and any other sim's of the same name; `NEWTON_FASTER` Newton's own, over it: the
# contact gap (a larger one finds contacts sooner, so more of them), MuJoCo Warp's solver, and Kamino's PADMM --
# its tolerances, its cap, and what trades fewer iterations for dearer ones: the penalty, the acceleration, the
# warm starts and the preconditioning. A physics choice is no speed knob: the friction cone (elliptic everywhere),
# Kamino's Baumgarte gain `gamma`, Lodos's mode and Baumgarte bias.
FASTER = {
    "timestep": "^", "integrator": "<>", "iterations": "v", "tolerance": "^", "ls_iterations": "v",
    "ls_tolerance": "^", "prox_iters_per_qp": "v", "prox_tol_z": "^", "prox_tol_sigma": "^", "prox_rho": "<>",
    "refresh_de_saxce_per_prox": "-> 0", "warm_start": "-> 1", "contact_reduction": "-> 1", "groups": "<>",
    "normal_bins": "<>", "moment_bins": "<>", "spline_type": "-> linear", "shared_hessian": "<>",
    "fused_newton": "<>", "row_space": "<>",
}
NEWTON_FASTER = {
    "mjwarp": {"gap": "v", "solver": "<>"},
    "kamino": {"gap": "v", "primal_tolerance": "^", "dual_tolerance": "^", "compl_tolerance": "^",
               "max_iterations": "v", "rho_0": "<>", "use_acceleration": "<>", "warmstart_mode": "<>",
               "contact_warmstart_method": "<>", "preconditioning": "<>"},
}


def speed_arrow(sim, option):
    """Which way `option` of `sim` makes it faster (`NEWTON_FASTER`, else `FASTER`); "" for no speed knob."""
    own = NEWTON_FASTER.get(sim, {})
    if option in own:
        return own[option]
    return FASTER.get(option, "")


def arrow_tip(arrow):
    """What a `speed_arrow` says, for its tooltip."""
    return {"^": "faster when larger", "v": "faster when smaller",
            "<>": "fastest somewhere between: the best value depends on the scene"}.get(arrow, f"faster at {arrow[3:]}")


# A runner's capacities (`Runner.capacities`): Lodos's collision stages, which `lodos.utils.calibrate` names, and the
# fraction of a world's worst case that sizes each (`make_collision_context`; `lodos.utils.calibrate.grow`'s pairing).
STAGES = ("nCa1", "nCa2", "nCa3", "nTa", "nTb")
STAGE_NAMES = {"nTa": "Lodos tiles", "nTb": "Lodos compacted tiles", "nCa1": "Lodos stage 1",
               "nCa2": "Lodos stage 2", "nCa3": "Lodos contacts"}
FRACTIONS = ("capacity_tile", "capacity2", "capacity1")
FRACTION_OF = {"nCa1": "capacity_tile", "nTa": "capacity_tile", "nCa2": "capacity2", "nCa3": "capacity2",
               "nTb": "capacity1"}
# `Session.tune_capacities`' default: each capacity 1.5 times its peak, as the tuning tools size them. A capacity
# that overflows grows to `GROW` times the demand, at least double.
CAPACITY_OVERHEAD = 0.5
GROW = 1.5
CONFIG_FORMAT = "lodos-viewer-config/2"


def normalize_option(name, value):
    """An option's value as the solver takes it: the bin counts snap to ones it has."""
    if name == "normal_bins":
        return min(cd_bins.NORMAL_BINS, key=lambda n: (abs(n - int(value)), n))
    if name == "moment_bins":
        return min(cd_bins.MOMENT_BINS, key=lambda n: (abs(n - int(value)), n))
    return value

# How long one `tick` may spend stepping: the window stays responsive at about 30 fps
# whatever the scene costs, and a scene too slow for the requested rate runs slower
# than it rather than freezing the panel.
TICK_BUDGET = 0.030

# What the recorded frames may hold before playing pauses, so a take left running cannot
# take the machine's memory with it. The twenty-body scenes record 5-27 MB per 1000
# frames (two simulated seconds at every step), so 4 GB is five to twenty-five minutes
# of them; recording every n-th step stretches it n times.
MEMORY_BUDGET = 4e9


#######################################################################################
# State
#######################################################################################


@dataclasses.dataclass
class State:
    """What a run continues from: positions, velocities, time and MJWarp's warm start."""

    qpos: np.ndarray
    qvel: np.ndarray
    time: float = 0.0
    warm: np.ndarray | None = None

    def copy(self):
        return State(self.qpos.copy(), self.qvel.copy(), float(self.time),
                     None if self.warm is None else self.warm.copy())

    @classmethod
    def of(cls, frame):
        return cls(frame.qpos.copy(), frame.qvel.copy(), float(frame.time),
                   None if frame.warm is None else frame.warm.copy())


def refresh_applies(opt):
    """Whether `refresh_de_saxce_per_prox` can be on: a de Saxce term, and a proximal loop to re-take it in."""
    return bool(opt.get("de_saxce")) and int(opt.get("prox_iters_per_qp", 0)) > 0 and float(opt.get("prox_rho", 0.0)) > 0.0


def mode_options(mode, source):
    """
    The options `mode` decides -- the proximal loop's, the de Saxce term's and, under the
    exact mode, the Baumgarte bias's: the scene's (`SceneSource.defaults`), else
    `lodos.cd.driver.resolve_options`'s defaults.
    """
    own = {k: v for k, v in source.defaults(mode).items() if k in cd_driver.OPTIONS}
    r = cd_driver.resolve_options(**dict(own, mode=mode))
    opt = {"prox_rho": r["prox_rho"], "prox_iters_per_qp": r["prox_iters_per_qp"],
           "prox_tol_z": float(r["prox_tol_z"]), "prox_tol_sigma": float(r["prox_tol_sigma"]),
           "de_saxce": int(r["de_saxce"]), "refresh_de_saxce_per_prox": int(r["refresh_de_saxce_per_prox"])}
    if mode != "mj":
        opt["baumgarte"] = int(r["baumgarte"])
        opt["baumgarte_alpha"] = float(r["baumgarte_alpha"])
        opt["baumgarte_slop"] = float(r["baumgarte_slop"])
    else:  # the friction cone, the MJCF's: the exact mode's proximal loop runs the elliptic one whatever it is
        opt["cone"] = NL.MJWARP_CONES[int(source.mjm.opt.cone)]
    return opt


def default_config(source, sim):
    """A sim's configuration for a scene: its options at the scene's values, no parameter overrides."""
    mjm = source.mjm
    opt = {}
    if sim in NL.SIMS:
        opt = NL.default_options(source, sim)
    else:
        own = source.defaults()
        opt["mode"] = str(own.get("mode", "exact"))
        # MuJoCo's own solver options: the scene's for its mode (`MODEL_OPTIONS`), else the model's.
        opt["iterations"] = int(own.get("iterations", mjm.opt.iterations))
        # MuJoCo Warp's floor, which make_sim's put_model applies: the viewer runs what the benches run
        opt["tolerance"] = max(float(own.get("tolerance", mjm.opt.tolerance)), NL.MJWARP_MIN_TOLERANCE)
        opt["ls_iterations"] = int(own.get("ls_iterations", mjm.opt.ls_iterations))
        opt["ls_tolerance"] = float(own.get("ls_tolerance", mjm.opt.ls_tolerance))
        if source.surfaces is not None:
            opt["contact_reduction"] = int(bool(source.sim_kwargs.get("contact_reduction", False)))
            # The SDF representation the Lodos surfaces are evaluated with: the tricubic B-spline ("cubic",
            # c2) or trilinear interpolation ("linear", c0), the scene's own to start with
            spline = spline_of(source.surfaces)
            if spline is not None:
                opt["spline_type"] = spline
        opt.update(mode_options(opt["mode"], source))
        opt["warm_start"] = int(bool(own.get("warm_start", True)))
        opt["fused_newton"] = int(bool(own.get("fused_newton", False)))
        opt["row_space"] = int(bool(own.get("row_space", False)))
        sh = own.get("shared_hessian")
        opt["shared_hessian"] = "auto" if sh is None else int(bool(sh))
        if source.surfaces is not None:
            # Contact groups, off by default; their bins the scene's (`SceneSource.group_bins`), else 1 x 1.
            opt["groups"] = str(own.get("groups") or "contact")
            gb = source.group_bins or {}
            opt["normal_bins"] = int(own.get("normal_bins", gb.get("normal_bins", 1)))
            opt["moment_bins"] = int(own.get("moment_bins", gb.get("moment_bins", 1)))
    if sim not in NL.SIMS:
        opt["timestep"] = float(mjm.opt.timestep)
        opt["integrator"] = next((k for k, v in lsim.INTEGRATORS.items() if v == mjm.opt.integrator), "euler")
        opt["implicit_constraints"] = int(bool(source.sim_kwargs.get("implicit_constraints", False)))
    return dict(sim=sim, options=opt, pairs=VP.no_overrides())


def edit_config(source, cfg, name, value):
    """
    `cfg` with option `name` at `value`, and what follows from it -- the configuration
    `Session.set_option` makes, and the panel builds for. `cfg` itself when nothing changes.
    """
    value = normalize_option(name, value)
    if cfg["options"].get(name) == value:
        return cfg
    cfg = copy.deepcopy(cfg)
    o = cfg["options"]
    applied = refresh_applies(o)
    o[name] = value
    if name == "mode":
        # The proximal loop's and the target's options are the mode's: they start
        # over at its defaults, the scene's, and mj takes no Baumgarte bias.
        for k in MODE_OPTIONS:
            o.pop(k, None)
        o.update(mode_options(value, source))
        # MuJoCo's own options follow the mode where the scene sets them for one (`mode_defaults`).
        per = {k for opts in source.mode_defaults.values() for k in opts}
        own = source.defaults(value)
        for k in MODEL_OPTIONS:
            if k in per:
                o[k] = type(getattr(source.mjm.opt, k))(own.get(k, getattr(source.mjm.opt, k)))
        if "tolerance" in o:
            o["tolerance"] = max(float(o["tolerance"]), NL.MJWARP_MIN_TOLERANCE)
    if name != "refresh_de_saxce_per_prox" and "refresh_de_saxce_per_prox" in o:
        # The solver refuses the refresh without the term and a loop: it goes off with
        # either, and when an option brings both back it returns to the scene's default,
        # else the solver's -- on, as `resolve_options` has it.
        if not refresh_applies(o):
            o["refresh_de_saxce_per_prox"] = 0
        elif not applied and name != "mode":
            o["refresh_de_saxce_per_prox"] = int(bool(source.defaults().get("refresh_de_saxce_per_prox", True)))
    if name in TOA_OPTIONS:
        why = toa_conflict(source, cfg)
        if why:
            raise ValueError(why)
    return cfg


# The options a Lodos pair's time of arrival needs at a value (`toa_conflict`).
TOA_OPTIONS = ("mode", "baumgarte", "groups")


def toa_conflict(source, cfg):
    """
    Why `cfg` cannot run its Lodos pairs' time of arrival, or None: a pair with `toa` on needs the exact mode
    with its Baumgarte bias, whose positive-gap term is what lets a speculative contact close, and no contact
    groups, whose bias was not made for a gap -- `lodos.sim.make_sim` refuses the others, and the Baumgarte bias,
    which the viewer sets on a built runner, it does not see.
    """
    if cfg["sim"] != "lodos" or source.surfaces is None:
        return None
    n = int(np.count_nonzero(VP.lodos_values(source.surfaces, cfg["pairs"]["lodos"])["toa"]))
    if not n:
        return None
    o = cfg["options"]
    why = ("the mj mode" if o.get("mode") == "mj" else "no Baumgarte bias" if not o.get("baumgarte", 1)
           else f"{o['groups']} groups" if o.get("groups", "contact") != "contact" else None)
    if why is None:
        return None
    return (f"{n} Lodos pairs have toa on, which needs the exact mode with its Baumgarte bias and no contact "
            f"groups, not {why}: turn their toa off first")


def config_label(cfg):
    o = cfg["options"]
    if cfg["sim"] in NL.SIMS:
        parts = [SIM_NAMES[cfg["sim"]], f"gap {1e3 * o['gap']:.3g} mm"]
        parts += [f"{k.replace('_', ' ')} {o[k]:.3g}" for k in ("iterations", "tolerance", "ls_iterations",
                                                                 "ls_tolerance") if k in o]
        parts += [v for k, v in (("solver", "cg"), ("integrator", "implicitfast"), ("cone", "pyramidal"))
                  if o.get(k) == v]
        if "timestep" in o:
            parts.append(f"dt {1e3 * o['timestep']:.4g} ms")
        # Kamino's solver options, where they are not Newton's defaults
        parts += [f"{k.replace('_', ' ')} {v:.3g}" if isinstance(v, float) else f"{k.replace('_', ' ')} {v}"
                  for k, v in o.items() if k in NL.KAMINO_DEFAULTS and v != NL.KAMINO_DEFAULTS[k]]
        parts += VP.label(cfg["pairs"], cfg["sim"] == "kamino", ("newton",))  # Kamino: friction alone
        return ", ".join(parts)
    parts = [f"Lodos {o['mode']}"]
    if "iterations" in o:
        parts.append(f"{o['iterations']} it" + (f", ls {o['ls_iterations']} it" if "ls_iterations" in o else ""))
    if o.get("cone") == "pyramidal":
        parts.append("pyramidal")
    if "prox_rho" in o:
        bits = [f"rho {o['prox_rho']:.3g}"]
        if "baumgarte" in o:
            bits.append(f"alpha {o['baumgarte_alpha']:.3g}" if o["baumgarte"] else "no Baumgarte")
            slop = BAUMGARTE_SLOP
            if o["baumgarte"] and o.get("baumgarte_slop", slop) != slop:
                bits.append(f"slop {1e3 * o['baumgarte_slop']:.3g} mm")
        parts.append(", ".join(bits + [f"{o['prox_iters_per_qp']} prox it"]))
    elif not o.get("baumgarte", 1):
        parts.append("no Baumgarte")
    if o.get("groups", "contact") != "contact":
        parts.append(f"{o['groups']} groups")
    if o.get("row_space"):
        parts.append("row space")
    if o.get("shared_hessian", "auto") != "auto":
        parts.append("shared Hessian" if o["shared_hessian"] else "no shared Hessian")
    if o.get("fused_newton"):
        parts.append("fused Newton")
    if "normal_bins" in o and o.get("groups", "contact") != "contact":
        parts.append(f"{o['normal_bins']}x{o.get('moment_bins', 1)} bins")
    if "de_saxce" in o and o["de_saxce"] != int(o.get("mode") != "mj"):
        parts.append("de Saxce" if o["de_saxce"] else "no de Saxce")
    # The solver refreshes the term wherever it applies: the label names the other choice.
    refresh = refresh_applies(o)
    if "refresh_de_saxce_per_prox" in o and bool(o["refresh_de_saxce_per_prox"]) != refresh:
        parts.append("de Saxce refreshed" if o["refresh_de_saxce_per_prox"] else "de Saxce not refreshed")
    if "warm_start" in o and o["warm_start"] != 1:
        parts.append("warm start" if o["warm_start"] else "cold start")
    if o.get("contact_reduction"):
        parts.append("reduced contacts")
    if o.get("spline_type") == "linear":
        parts.append("trilinear")
    if o.get("integrator") == "implicitfast":
        parts.append("implicitfast, M - dt D throughout" if o.get("implicit_constraints") else "implicitfast")
    if "timestep" in o:
        parts.append(f"dt {1e3 * o['timestep']:.4g} ms")
    # The solves' convergence, where it is not the family's default.
    defaults = CONVERGENCE_DEFAULTS
    for k in CONVERGENCE:
        if k in o and k in defaults and o[k] != defaults[k]:
            parts.append(f"{k.replace('_', ' ')} {o[k]:.3g}")
    parts += VP.label(cfg["pairs"], o.get("mode") != "mj")
    return ", ".join(parts)


def edit_pairs(source, cfg, kind, indices, field, value):
    """
    `cfg` with `field` at `value` on the pairs `indices` of `kind` (`viewer.pairs.KINDS`), or back
    at the scene's where `value` is None -- the configuration `Session.set_pairs` makes. `cfg` itself when
    nothing changes. Friction is a float, solref two, solimp five; Lodos's alone, `sided` 1 or 2, a pair with a
    plane only 1, and its time of arrival (`viewer.pairs.TOA_FIELDS`, as `viewer.pairs.check` holds them), refused
    where the configuration cannot run it (`toa_conflict`). `indices` None is every pair of the kind: one value
    for them all, which clears their own; `toa_horizon` takes only that, one horizon for every Lodos pair.
    """
    if kind not in VP.KINDS:
        raise ValueError(f"pairs are {VP.KINDS}, not {kind!r}")
    fields = VP.kind_fields(kind)
    if field not in fields:
        raise ValueError(f"{kind} pairs take {fields}, not {field!r}")
    if field == "toa_horizon" and indices is not None:
        raise ValueError("the toa horizon is one value for every Lodos pair: Lodos takes one per collision context")
    v = None if value is None else VP.check(field, value)
    new = copy.deepcopy(cfg)
    ov = new["pairs"][kind]
    d = ov[field]
    rebuild = kind == "lodos" and field in VP.REBUILD_FIELDS  # an override changes the runner's key
    if indices is None:
        d.clear()
        ov["all"].pop(field, None)
        if v is not None:
            ov["all"][field] = v
            if rebuild and field not in VP.rebuild_changes(source.surfaces, ov):
                ov["all"].pop(field)  # the scene's own already: no override, so the scene's runner
        return _edited(source, cfg, new)
    idx = [int(i) for i in np.atleast_1d(np.asarray(indices, dtype=int))]
    if field == "sided":
        planes = VP.plane_pairs(source.surfaces)
        if v == 2 and any(planes[i] for i in idx):
            raise ValueError(f"pair {next(i for i in idx if planes[i])} has a plane, which has no vertices: "
                             "it is one-sided")
    if rebuild:
        for i in idx:
            d.pop(i, None)
        base = VP.lodos_values(source.surfaces, ov)[field]  # without their own: what they would be
        for i in idx:  # a pair at that already is no override, so the same key and runner
            if v is not None and v != base[i]:
                d[i] = v
    else:
        for i in idx:
            if v is None:
                d.pop(i, None)
            else:
                d[i] = v
    return _edited(source, cfg, new)


def _edited(source, cfg, new):
    """`new`, a pair edit of `cfg` -- `cfg` itself where it changes nothing -- refused where `toa_conflict` says."""
    if new["pairs"] == cfg["pairs"]:
        return cfg
    why = toa_conflict(source, new)
    if why:
        raise ValueError(why)
    return new


def edit_all_pairs(source, cfg, kinds, field, value):
    """
    `cfg` with `field` at `value` on every pair of each of `kinds` (`edit_pairs`), as one edit; "sided" and the
    time of arrival are Lodos's.
    """
    for kind in kinds:
        if field in VP.kind_fields(kind):
            cfg = edit_pairs(source, cfg, kind, None, field, value)
    return cfg


#######################################################################################
# A runner: one configuration's simulation
#######################################################################################


class Runner:
    """
    One Lodos configuration of a scene at one world, with its step captured.

    source: the `SceneSource`.
    cfg: a configuration from `default_config`; only its rebuild options are fixed here
        -- `apply` sets the rest.
    scene: the `LiveScene` to collect poses for; built from this runner's model when None.
    """

    def __init__(self, source, cfg, scene=None, kept=None):
        self.source = source
        self.key = runner_key(cfg)
        self.caps = dict(source.caps)
        self.collision = {}  # Lodos's arenas over the scene's (`FRACTIONS`), once tuned or grown
        # The largest demand on every capacity over everything this runner simulated, its builds at other
        # capacities included (`overflow` keeps it), and how many forwards that was.
        self.seen = dict.fromkeys(lcalibrate.TRACKED + STAGES, 0)
        self.steps_seen = 0
        if kept is not None:  # its configuration's, from a runner freed for room (`kept`, `Session.free_runners`)
            self.caps, self.collision = dict(kept["caps"]), dict(kept["collision"])
            self.seen, self.steps_seen = dict(kept["seen"]), int(kept["steps_seen"])
        self.applied = None
        self.scene = scene
        self.controls_playing = True  # the session's (`Session.play_controls`): off, the drives keep their targets
        self.build()

    # -------------------------------------------------------------------------------

    def build(self):
        kw = {k: self._key_options()[k] for k in self._key_options()}
        if kw.get("groups", "contact") == "contact":
            # The bins bin contact groups: with every pair per contact there is nothing to bin
            for k in ("groups", "normal_bins", "moment_bins"):
                kw.pop(k, None)
        if kw.get("shared_hessian") == "auto":
            kw.pop("shared_hessian")  # the solver's default: wherever MJWarp can build it, from 48 dofs
        ov = (f"opt.cone={kw.pop('cone')}",) if "cone" in kw else ()  # under mj: MuJoCo's own option
        # Lodos's pairs made one- or two-sided, or given their time of arrival, against the scene's own: the key's
        pov = VP.no_overrides()["lodos"]
        for f, every, own in self.key[2]:
            if every is not None:
                pov["all"][f] = every
            pov[f] = dict(own)
        toa = {} if self.source.surfaces is None else VP.rebuild_changes(self.source.surfaces, pov)
        sided = toa.pop("sided", {})
        source = self.source
        if self.collision and source.surfaces is not None:
            source = dataclasses.replace(source, surfaces=dataclasses.replace(
                source.surfaces, col_kwargs={**source.surfaces.col_kwargs, **self.collision}))
        self.sim = source.make_sim(solver_kwargs=kw, caps=self.caps, overrides=ov, sided=sided, toa=toa)
        self.tracker = lcalibrate.Tracker(self.sim)
        self._tables = None
        if self.scene is None:
            self.scene = make_live_scene(self.sim)
        self.collector = Collector(self.sim, self.scene)
        self.graphs = None
        self.forwarded = False
        self._warm_before = None
        self.owner = None  # (take id, steps) whose state the runner holds
        self.controls_due = True  # a scene's controls before the next step, whatever its time
        self.defaults = self._read_defaults()
        if self.applied is not None:
            cfg, self.applied = self.applied, None
            self.apply(cfg)

    def _key_options(self):
        return {k: v for k, v in dict(self.key[1]).items()}

    def _read_defaults(self):
        """The scene's values of every live parameter, to restore an override from."""
        sim, m = self.sim, self.sim.m
        out = {"opt_tolerance": m.opt.tolerance.numpy().copy()}
        for k in ("geom_friction", "geom_solref", "geom_solimp") + (
                ("pair_friction", "pair_solref", "pair_solimp") if int(sim.mjm.npair) else ()):
            out[k] = getattr(m, k).numpy().copy()
        if sim.solver.lc is not None:
            for k in ("cpr_friction", "cpr_solref", "cpr_solimp"):
                out[k] = getattr(sim.solver.lc, k).numpy().copy()
        return out

    @property
    def pair_counts(self):
        """`{kind: pairs}` for the kinds of pair the scene has, without making their tables."""
        out = {}
        if "cpr_friction" in self.defaults:
            out["lodos"] = len(self.defaults["cpr_friction"])
        n = int(self.sim.m.nxn_geom_pair_filtered.shape[0])
        if n:
            out["prim"] = n
        return out

    @property
    def pair_tables(self):
        """`{kind: viewer.pairs.PairTable}` for the kinds of pair the scene has, made on first use."""
        if self._tables is None:
            t = {}
            if "lodos" in self.pair_counts:
                t["lodos"] = VP.PairTable.lodos(self.sim, self.defaults, self.source.surfaces)
            if "prim" in self.pair_counts:
                t["prim"] = VP.PairTable.prim(self.sim, self.defaults)
            self._tables = t
        return self._tables

    # -------------------------------------------------------------------------------

    def apply(self, cfg):
        """Set `cfg`'s options and contact parameters on this runner's arrays."""
        if runner_key(cfg) != self.key:
            raise ValueError("a configuration with other rebuild options needs its own runner")
        sim, dfl = self.sim, self.defaults
        old = self.applied
        recapture = False
        opt = cfg["options"]
        m = sim.m
        if "iterations" in opt and int(m.opt.iterations) != int(opt["iterations"]):
            m.opt.iterations = int(opt["iterations"])
            recapture = True
        if "ls_iterations" in opt and int(m.opt.ls_iterations) != int(opt["ls_iterations"]):
            m.opt.ls_iterations = int(opt["ls_iterations"])  # a launch bound: recaptured
            recapture = True
        # The Baumgarte bias is compiled into MJWarp's prox init kernel: a recapture rebuilds it.
        # Its gain is an array the captured step reads.
        if "baumgarte" in opt and bool(m.opt.prox_baumgarte) != bool(opt["baumgarte"]):
            m.opt.prox_baumgarte = bool(opt["baumgarte"])
            recapture = True
        if "baumgarte_alpha" in opt and bool(m.opt.prox_enable):
            m.opt.prox_baumgarte_alpha.fill_(float(opt["baumgarte_alpha"]))
        if "baumgarte_slop" in opt and bool(m.opt.prox_enable):
            m.opt.prox_baumgarte_slop.fill_(float(opt["baumgarte_slop"]))
        if "tolerance" in opt:
            m.opt.tolerance.fill_(max(float(opt["tolerance"]), NL.MJWARP_MIN_TOLERANCE))  # put_model's floor
        if "ls_tolerance" in opt:
            m.opt.ls_tolerance.fill_(float(opt["ls_tolerance"]))
        # The proximal loop's stops, on its anchor and on the de Saxce term: arrays the
        # captured step reads.
        if "prox_tol_z" in opt and bool(m.opt.prox_enable):
            m.opt.prox_tol_z.fill_(float(opt["prox_tol_z"]))
        if "prox_tol_sigma" in opt and bool(m.opt.prox_enable):
            m.opt.prox_tol_sigma.fill_(float(opt["prox_tol_sigma"]))

        # The pairs' contact parameters (`viewer.pairs`): the exact mode takes their friction alone
        # Without overrides the scene's arrays, and no table made: a million pairs' would cost a build seconds.
        exact = opt.get("mode") != "mj"
        pairs = cfg["pairs"]
        if "lodos" in self.pair_counts:
            lc = sim.solver.lc
            if VP.has_overrides(pairs["lodos"], exact):
                v = self.pair_tables["lodos"].values(pairs, exact)
                fr = dfl["cpr_friction"].copy()
                fr[:, :2] = v["mu"][:, None]
                arrays = {"cpr_friction": fr, "cpr_solref": v["solref"], "cpr_solimp": v["solimp"]}
            else:
                arrays = {k: dfl[k] for k in ("cpr_friction", "cpr_solref", "cpr_solimp")}
            for k, a in arrays.items():
                getattr(lc, k).assign(np.asarray(a, np.float32))
        if "prim" in self.pair_counts:
            g = self.pair_tables["prim"].geom_arrays(pairs, exact) if VP.has_overrides(pairs["prim"], exact) else None
            for k in ("friction", "solref", "solimp"):
                for f, src in ((f"geom_{k}", None if g is None else g[k]),
                               (f"pair_{k}", None if g is None or g["pair"] is None else g["pair"][k])):
                    if f in dfl:
                        a = dfl[f].copy()
                        if src is not None:
                            a[:] = src
                        getattr(m, f).assign(a.astype(np.float32))

        self.applied = copy.deepcopy(cfg)
        if recapture or old is None:
            self.graphs = None

    # -------------------------------------------------------------------------------

    def state(self):
        """The current `State`. Synchronizes."""
        d = self.sim.drbd
        warm = self.sim.d.qacc_warmstart.numpy()[0].astype(np.float64)
        return State(d.qpos.numpy()[0].astype(np.float64), d.qvel.numpy()[0].astype(np.float64),
                     float(d.time.numpy()[0]), warm)

    def set_state(self, st, cold=True):
        """
        Load `st` into the simulation. `cold` empties prox's cross-step cache, whose
        entries belong to whatever state the runner held before.
        """
        d = self.sim.drbd
        rbd_io.set_state(d, qpos=np.asarray(st.qpos)[None], qvel=np.asarray(st.qvel)[None])
        d.time.fill_(float(st.time))
        self.controls_due = True
        dd = self.sim.d
        if st.warm is not None:
            dd.qacc_warmstart.assign(np.asarray(st.warm, np.float32)[None])
        else:
            dd.qacc_warmstart.zero_()
        if cold and dd.prox_zcache_n.shape[0]:
            dd.prox_zcache_n.zero_()
        if cold and self.sim.solver.groups is not None:
            self.sim.solver.groups.forget()  # the contact groups' frames carried from the state before
        self.forwarded = False

    def capture(self):
        """Capture `forward`, `integrate` and `step`, leaving the state as it was. A forward tracks its demand."""
        sim = self.sim
        st = self.state()
        lstep.forward(sim)
        self.tracker.launch()
        rbd_integrate.integrate(sim.mrbd, sim.drbd)
        wp.synchronize()
        self.set_state(st, cold=False)
        dev = wp.get_device()
        with wp.ScopedCapture(dev) as c:
            lstep.forward(sim)
            self.tracker.launch()
        g_fwd = c.graph
        with wp.ScopedCapture(dev) as c:
            rbd_integrate.integrate(sim.mrbd, sim.drbd)
        g_int = c.graph
        with wp.ScopedCapture(dev) as c:
            lstep.step_sim(sim)
            self.tracker.launch()  # the forward's counts, which integrating leaves alone
        g_step = c.graph
        self.graphs = (g_fwd, g_int, g_step)
        self.forwarded = False

    def _graphs(self):
        if self.graphs is None:
            self.capture()
        return self.graphs

    def _warm(self):
        """MJWarp's warm start as the next solve will read it, or None."""
        if self.sim.solver is None:
            return None
        return self.sim.d.qacc_warmstart.numpy()[0].astype(np.float64)

    def forward(self):
        """Recompute the current state's contacts and forces, and collect it."""
        g_fwd, _, _ = self._graphs()
        warm = self._warm()
        wp.capture_launch(g_fwd)
        self.forwarded = True
        self._warm_before = warm
        frame = self.collector.collect()
        # The warm start the solve *started* from, so that restoring this frame's state
        # repeats its step exactly (the solve overwrites it with its own result).
        frame.warm = warm
        return frame

    def _control(self):
        """
        A scene's controls (`SceneSource.controls`): written before a step that starts one of
        their ticks, and before the first step from a state just loaded. The forward pass is
        then redone, since the actuators' forces read the control. Synchronizes.
        """
        c, d = self.source.controls, self.sim.drbd
        t = float(d.time.numpy()[0])
        if not self.controls_due and abs(t - round(t / c.period) * c.period) >= 0.5 * float(self.sim.mjm.opt.timestep):
            return
        u = np.asarray(c(t, d.qpos.numpy()[0].astype(np.float64)), dtype=np.float32)
        rbd_io.set_state(d, ctrl=u[None])
        self.controls_due = False
        self.forwarded = False

    def step(self, record):
        """One step; returns the frame of the state it started from when `record`."""
        g_fwd, g_int, g_step = self._graphs()
        if self.source.controls is not None and self.controls_playing:
            self._control()
        frame = None
        if record:
            if not self.forwarded:
                warm = self._warm()
                wp.capture_launch(g_fwd)
            else:
                warm = self._warm_before
            frame = self.collector.collect()
            frame.warm = warm
            wp.capture_launch(g_int)
        elif self.forwarded:
            wp.capture_launch(g_int)
        else:
            wp.capture_launch(g_step)
        self.forwarded = False
        return frame

    # -------------------------------------------------------------------------------

    def overflow(self):
        """
        Which capacities the last forward overflowed, a set of names: MJWarp's `nconmax`, `njmax` and `nccdmax`, and
        Lodos's arenas by the fraction that sizes them (`FRACTION_OF`). Keeps every demand in `seen`.
        """
        sim, d = self.sim, self.sim.d
        out = set()
        p = dict(zip(lcalibrate.TRACKED, self.tracker.peak.numpy().tolist()))  # the largest demands since the build
        if sim.dcol is not None:
            p.update(zip(STAGES, sim.dcol.ct_peak.numpy().tolist()))
            out |= {FRACTION_OF[k] for k, of in zip(STAGES, sim.dcol.overflow.numpy().tolist()) if of}
        for k, v in p.items():
            self.seen[k] = max(self.seen[k], int(v))
        self.steps_seen += 1
        if sim.prim.active or sim.solver is not None:
            if max(p["nacon"], p["ncollision"]) > int(d.naconmax):
                out.add("nconmax")
        if sim.solver is not None:
            if p["nefc_world"] >= int(d.njmax):
                out.add("njmax")
            rs = sim.solver.rowspace
            if rs is not None and int(rs.overflow.numpy()[0]):
                rs.overflow.zero_()
                out.add("njmax")  # the row space's capacity is the row capacity's
        hy = self.tracker.hy
        if hy is not None and (p["epa"] > hy.cap_epa or p["manifold"] > hy.cap_man or p["hybrid_contacts"] > hy.cap_con):
            out.add("nccdmax")
        return out

    def _need(self):
        """Each capacity's largest demand in `seen`, in its own units: MJWarp's per world, Lodos's as a fraction."""
        s = self.seen
        out = dict(nconmax=max(s["nacon"], s["ncollision"]), njmax=s["nefc_world"],
                   nccdmax=max(s["epa"], s["manifold"], math.ceil(s["hybrid_contacts"] / 4)))
        col = self.sim.dcol
        if col is not None:
            worst = max(1, int(self.sim.mcol.n_tile_slots))
            out.update(capacity_tile=s["nTa"] / worst, capacity2=s["nCa2"] / max(1, int(col.nCa1_max)),
                       capacity1=s["nTb"] / worst)
        return out

    def grow(self, which):
        """
        Raise the named capacities -- to `GROW` times their demand, at least double; a fraction at most 1, the worst
        case -- and rebuild, keeping the state and configuration.
        """
        st, old = self.state(), (dict(self.caps), dict(self.collision))
        need, frac = self._need(), self.fractions()
        for cap in which:
            if cap in FRACTIONS:
                self.collision[cap] = min(1.0, max(2.0 * frac[cap], GROW * need[cap]))
                continue
            now = int(self.caps.get(cap) or 0) or (int(self.sim.d.naccdmax) if cap == "nccdmax" else 0)
            self.caps[cap] = max(2 * now or 1024, math.ceil(GROW * need[cap]))
            if cap == "njmax" and self.caps.get("njmax_nnz"):
                self.caps["njmax_nnz"] = 2 * int(self.caps["njmax_nnz"])
            if cap == "nconmax" and self.caps.get("nccdmax"):
                self.caps["nccdmax"] = min(2 * int(self.caps["nccdmax"]), self.caps["nconmax"])
            if cap == "nccdmax" and self.caps.get("nconmax"):
                self.caps["nccdmax"] = min(self.caps["nccdmax"], int(self.caps["nconmax"]))
        self._rebuild(st, old)

    def _rebuild(self, st, old):
        """
        Build again at the capacities now set and load `st`. The old sim goes first: a scene whose sim and the other
        configurations' runners fill the GPU has no room for a second. A build that fails anyway puts back the
        capacities `old` (`caps`, `collision`) it ran at, and raises.
        """
        self.release()
        try:
            self.build()
        except Exception:
            self.caps, self.collision = old
            self.build()
            self.set_state(st)
            raise
        self.set_state(st)

    def release(self):
        """Drop the sim and what is made of it -- graphs, collector, tracker, pair tables -- and its GPU memory."""
        self.sim = self.collector = self.tracker = self.graphs = self._tables = None
        gc.collect()  # the sim's arrays, in reference cycles, freed now rather than at the next collection

    def kept(self):
        """What a freed runner hands its configuration's next build: its capacities and the demand it has seen."""
        return dict(caps=dict(self.caps), collision=dict(self.collision), seen=dict(self.seen),
                    steps_seen=self.steps_seen)

    def fractions(self):
        """Lodos's collision arenas as fractions of a world's worst case (`FRACTIONS`); {} without its surfaces."""
        if self.source.surfaces is None:
            return {}
        kw = self.source.surfaces.col_kwargs
        return {k: float(self.collision.get(k, kw.get(k, 1.0))) for k in FRACTIONS}

    def capacities(self):
        """
        `[(name, allocated, peak)]`: every capacity of the built sim, for its one world, and the largest demand on it
        in `seen` -- MJWarp's contacts (Lodos's among them), constraint rows and CCD slots, and Lodos's arenas.
        """
        d, s = self.sim.d, self.seen
        need = self._need()
        out = [("contacts (nconmax)", int(d.naconmax), need["nconmax"]),
               ("constraint rows (njmax)", int(d.njmax), need["njmax"])]
        if self.tracker.hy is not None:
            out.append(("CCD slots (nccdmax)", int(d.naccdmax), need["nccdmax"]))
        col = self.sim.dcol
        if col is not None:
            alloc = dict(nTa=col.nTa_max, nTb=col.nTb_max, nCa1=col.nCa1_max, nCa2=col.nCa2_max, nCa3=col.nCa3_max)
            for k in ("nTa", "nTb", "nCa1", "nCa2", "nCa3"):
                if k != "nTb" or int(self.sim.mcol.seg_compact):  # the compacted launch, where the context compacts
                    out.append((f"{STAGE_NAMES[k]} ({k})", int(alloc[k]), s[k]))
        return out

    def demand(self):
        """The built sim's `lodos.utils.calibrate.Tracker` report, its peaks those of `seen`: the demand over every run."""
        rep = self.tracker.report()
        s, m = self.seen, rep["runtime"]
        m.update(nacon=s["nacon"], ncollision=s["ncollision"], contacts_per_world=s["runtime_contacts_world"],
                 nefc_per_world=s["nefc_world"])
        if "ccd" in m:
            m["ccd"].update(epa=s["epa"], manifold=s["manifold"], contacts=s["hybrid_contacts"])
        if "lodos_col" in rep:
            rep["lodos_col"].update(peak={k: s[k] for k in STAGES}, contacts_per_world=s["lodos_contacts_world"])
        # Every step that overflowed was grown and taken again (`Session._rollback`), so the peaks are demands
        rep["overflowed"] = False
        return rep

    def tune_capacities(self, overhead):
        """
        Every capacity at its demand in `seen` times 1 + `overhead`, as a tuning file sizes them
        (`lodos.utils.calibrate.settings`), and rebuild, keeping the state. Returns that section.
        """
        sizes = lcalibrate.settings(self.demand(), safety=1.0 + float(overhead))
        st, old = self.state(), (dict(self.caps), dict(self.collision))
        mw = sizes["runtime"]
        self.caps.update({k: int(mw[k]) for k in ("nconmax", "njmax", "nccdmax") if k in mw})
        self.collision = {k: float(v) for k, v in sizes["collision"].items()}
        self._rebuild(st, old)
        return sizes


def _out_of_memory(exc):
    """Whether `exc` is an allocation the GPU had no room for: Warp's, or CUDA's."""
    s = str(exc)
    return "Failed to allocate" in s or "out of memory" in s.lower()


def runner_key(cfg):
    """
    What a runner is built for: the sim, the options that change allocation and, under Lodos, the pairs
    made one- or two-sided or given their time of arrival against the scene's own (`viewer.pairs.REBUILD_FIELDS`:
    each as `(field, the value for all, the pairs' own)`). A Newton sim has one runner, whose options -- its
    timestep among them -- rebuild it in its own process.
    """
    lodos = cfg["sim"] == "lodos"
    keys = REBUILD_OPTIONS + LODOS_REBUILD_OPTIONS if lodos else ()
    ov = cfg["pairs"]["lodos"]
    pairs = tuple((f, ov["all"].get(f), tuple(sorted(ov[f].items()))) for f in VP.REBUILD_FIELDS
                  if f in ov["all"] or ov[f]) if lodos else ()
    return (cfg["sim"], tuple(sorted((k, v) for k, v in cfg["options"].items() if k in keys)), pairs)


#######################################################################################
# Takes
#######################################################################################


@dataclasses.dataclass
class Take:
    """A run: frames every `stride` steps from one state, under one configuration."""

    id: int
    config: dict
    parent: tuple | None  # (take id, frame index) it branched from
    frames: list
    dt: float  # simulation timestep
    stride: int = 1
    steps: int = 0  # steps taken since the first frame
    tail: State | None = None  # the state after the last step; None before the first
    note: str = ""

    @property
    def start(self):
        """The state the take starts from: its first frame's."""
        return State.of(self.frames[0])

    @property
    def frame_dt(self):
        return self.dt * self.stride

    def label(self):
        src = "initial state" if self.parent is None else f"take {self.parent[0]} frame {self.parent[1]}"
        name = "draft" if self.id < 0 else f"take {self.id}"
        extra = f", {self.note}" if self.note else ""
        return f"{name}: {config_label(self.config)} from {src}{extra}"

    def nbytes(self):
        return sum(frame_bytes(f) for f in self.frames)


# What a saved take holds of each frame (`LiveFrame`), a row a frame, and of each contact, the frames'
# contacts one after another (`con_off`); `TAKE_FORMAT` changes when either does.
TAKE_FRAME = ("qpos", "qvel", "geom_xform", "body_xform", "jnt_anchor", "jnt_axis", "body_v", "body_w")
TAKE_CONTACT = ("con_p", "con_n", "con_v", "con_f", "con_depth", "con_seg", "con_src", "con_body")
TAKE_FORMAT = 2


def save_take(take, source, path):
    """
    Write `take` -- every frame, the state after its last step and its configuration -- to an `.npz` that
    `Session.load_take` adds back to a session of the same scene, to scrub and to continue. Returns the
    path written (`.npz` appended where it has none).
    """
    path = Path(path)
    path = path if path.suffix == ".npz" else path.with_name(path.name + ".npz")
    fr, mjm = take.frames, source.mjm
    meta = dict(format=TAKE_FORMAT, scene=source.name, path=source.path, nq=int(mjm.nq), nv=int(mjm.nv),
                nbody=int(mjm.nbody), take=take.id, label=take.label(), config=_config_to_json(take.config),
                dt=take.dt, stride=take.stride, steps=take.steps, stats=[f.stats for f in fr])
    out = {k: np.stack([getattr(f, k) for f in fr]) for k in TAKE_FRAME}
    out["time"] = np.array([f.time for f in fr], dtype=np.float64)
    out["has_warm"] = np.array([f.warm is not None for f in fr])
    out["warm"] = np.stack([np.zeros(int(mjm.nv)) if f.warm is None else f.warm for f in fr])
    out["con_off"] = np.concatenate([[0], np.cumsum([f.ncon for f in fr])]).astype(np.int64)
    for k in TAKE_CONTACT:
        out[k] = np.concatenate([getattr(f, k) for f in fr])
    if take.tail is not None:
        out.update(tail_qpos=take.tail.qpos, tail_qvel=take.tail.qvel, tail_time=np.float64(take.tail.time))
        if take.tail.warm is not None:
            out["tail_warm"] = take.tail.warm
    np.savez_compressed(path, meta=np.array(json.dumps(meta, default=lambda x: x.item())), **out)
    return path


def take_meta(path):
    """A saved take's description (`save_take`): its scene, model sizes, label and configuration."""
    with np.load(path) as z:
        meta = json.loads(str(z["meta"]))
    if meta.get("format") != TAKE_FORMAT:
        old = " (an older format; migrate it: python tools/performance_tuning/migrate_formats.py PATH)" \
            if meta.get("format") == 1 else ""
        raise ValueError(f"{path} is not a take this viewer saved (format {meta.get('format')!r}){old}")
    return meta


def read_take(path):
    """What `save_take` wrote: `(meta, take)`, the take under the id it was saved with."""
    meta = take_meta(path)
    with np.load(path) as z:
        a = {k: z[k] for k in z.files if k != "meta"}
    off = a["con_off"]
    frames = []
    for i in range(len(a["time"])):
        lo, hi = int(off[i]), int(off[i + 1])
        frames.append(LiveFrame(time=float(a["time"][i]), **{k: a[k][lo:hi] for k in TAKE_CONTACT},
                                **{k: a[k][i] for k in TAKE_FRAME},
                                warm=a["warm"][i] if a["has_warm"][i] else None, stats=meta["stats"][i]))
    tail = None
    if "tail_qpos" in a:
        tail = State(a["tail_qpos"], a["tail_qvel"], float(a["tail_time"]), a.get("tail_warm"))
    take = Take(int(meta["take"]), _config_from_json(meta["config"]), None, frames, float(meta["dt"]),
                int(meta["stride"]), int(meta["steps"]), tail, note=f"loaded from {Path(path).name}")
    return meta, take


def _config_to_json(cfg):
    p = cfg["pairs"]  # by field for all a kind's pairs, else by pair
    pairs = {kind: {f: dict(d) if f == "all" else {str(i): v for i, v in d.items()} for f, d in p[kind].items()}
             for kind in VP.KINDS}
    return dict(sim=cfg["sim"], options=cfg["options"], pairs=dict(pairs, mode=p["mode"]))


def _config_from_json(c):
    pairs = VP.no_overrides()
    for kind in VP.KINDS:
        for f, d in c["pairs"].get(kind, {}).items():  # a file from before a kind has none of it
            if f == "all":
                pairs[kind]["all"] = {g: VP.check(g, v) for g, v in d.items()}
            else:
                pairs[kind][f] = {int(i): VP.check(f, v) for i, v in d.items()}
    pairs["mode"] = c["pairs"]["mode"]
    return dict(sim=c["sim"], options=c["options"], pairs=pairs)


def _pairs_json(table, cfg, exact):
    """
    A kind of pair's contact parameters for `Session.export_config`, as set: the values most of its pairs share
    (`common`, with how many), every pair whose values differ by its name (`others`), and the fields the solve
    applies -- under the exact mode friction, sidedness and time of arrival alone, which no other mode takes. A
    Lodos pair's time of arrival in words: `toa`, `toa_drift` and `toa_side` true or false, `toa_trace` its mode,
    `toa_horizon` in seconds or null for the timestep.
    """
    fields = VP.kind_fields(table.kind)
    v = table.values(cfg["pairs"], exact=False)
    rows = np.concatenate([np.asarray(v[f], np.float64).reshape(len(table), -1) for f in fields], axis=1)
    uniq, inv, cnt = np.unique(rows, axis=0, return_inverse=True, return_counts=True)
    top = int(np.argmax(cnt))

    def record(row):
        out, i = {}, 0
        for f in fields:
            w = VP.WIDTH.get(f, 1)
            x = [float(str(np.float32(y))) for y in row[i:i + w]]  # the solver's float32, in its shortest decimals
            if f in ("sided", "toa_trace_iters"):
                out[f] = int(x[0])
            elif f in ("toa", "toa_drift", "toa_side"):
                out[f] = bool(x[0])
            elif f == "toa_trace":
                out[f] = VP.TRACE_MODES[int(x[0])]
            elif f == "toa_horizon":
                out[f] = x[0] or None
            else:
                out[f] = x[0] if w == 1 else x
            i += w
        return out

    applies = [f for f in fields if (exact if f in VP.TOA_ALL else not exact or f in ("mu", "sided"))]
    return dict(count=len(table), applies=applies,
                common=dict(record(uniq[top]), pairs=int(cnt[top])),
                others=[dict(pair=table.label(int(i)), **record(rows[i])) for i in np.nonzero(inv.ravel() != top)[0]])


def frame_bytes(f):
    """What one frame holds in memory, near enough."""
    return sum(getattr(f, k).nbytes for k in ("con_p", "con_n", "con_v", "con_f", "con_depth", "con_seg", "con_src",
                                               "con_body", "geom_xform", "body_xform", "qpos", "qvel"))


class TrajectoryView:
    """
    A take as the `viewer.trajectory.Trajectory` the drawing code reads: indexable,
    `dt` between frames, and `peak` for the arrow scales.
    """

    def __init__(self, take):
        self.take = take

    @property
    def frames(self):
        return self.take.frames

    @property
    def dt(self):
        return self.take.frame_dt

    def __len__(self):
        return len(self.take.frames)

    def __getitem__(self, t):
        return self.take.frames[t]

    def __iter__(self):
        return iter(self.take.frames)

    def peak(self, field, percentile=99.0):
        mags = [np.linalg.norm(getattr(f, field), axis=-1) for f in self.take.frames if f.ncon]
        if not mags:
            return 0.0
        return float(np.percentile(np.concatenate(mags), percentile))


#######################################################################################
# The session
#######################################################################################


class Session:
    """
    Everything the live viewer drives. See the module docstring for runs and takes.

    stride: record every `stride`-th step.
    """

    def __init__(self, stride=1):
        self.source = None
        self.scene = None
        self.runners = {}
        self.takes = []
        self.take = None  # the take being looked at, a draft included
        self.draft = None
        self.frame_index = 0
        self.playing = False
        self.rate = 1.0
        self.max_speed = False
        self.stride = int(stride)
        self.status = ""
        self.memory_budget = MEMORY_BUDGET
        self._bytes = 0  # what the committed takes' frames hold
        self.rtf = 0.0
        self.grown = []  # (time, capacities) every time a capacity grew
        self.kept = {}  # a freed Lodos runner's capacities and demand, by configuration key (`free_runners`)
        self._next_id = 1
        self._acc = 0.0
        self._last = None
        self.on_build = None  # called with a message before a runner is built
        self.controls_playing = True  # whether a scene's controls play (`play_controls`)

    # -------------------------------------------------------------------------------
    # Loading

    def load(self, source, sim="lodos"):
        """Start over on `source`, at its initial state under `sim`."""
        self.source = source
        self.free_runners()  # the scene's GPU memory, before the next one's
        self.kept = {}
        self.takes = []
        self.scene = None
        self.draft = self.take = None
        self.playing = False
        self.controls_playing = True
        self._next_id = 1
        self._bytes = 0
        self.grown = []
        cfg = default_config(source, sim)
        runner = self.runner(cfg)
        self.scene = runner.scene
        st = State(source.qpos0.copy(), source.qvel0.copy(), 0.0)
        self._make_draft(cfg, st, parent=None)
        return self

    def load_take(self, path):
        """
        Add the take `save_take` wrote at `path`, and look at it: its frames to scrub, its end to continue
        from under its configuration. A take of another scene (its name and model sizes) is refused.
        """
        meta, take = read_take(path)
        m = self.source.mjm
        mine = dict(scene=self.source.name, nq=int(m.nq), nv=int(m.nv), nbody=int(m.nbody))
        if any(meta[k] != v for k, v in mine.items()):
            raise ValueError(f"{Path(path).name} was recorded in {meta['scene']} (nq {meta['nq']}, nv {meta['nv']}); "
                             f"the session has {mine['scene']} (nq {mine['nq']}, nv {mine['nv']})")
        take.id = self._next_id
        self._next_id += 1
        self.takes.append(take)
        self._bytes += take.nbytes()
        self.draft = None
        self.take = take
        self.frame_index = 0
        self.playing = False
        return take

    @property
    def loaded(self):
        return self.source is not None

    @property
    def dt(self):
        """The timestep of the take being looked at (its configuration's), else the scene's."""
        return self.take.dt if self.take is not None else float(self.source.mjm.opt.timestep)

    def runner(self, cfg):
        """The runner for `cfg`, built the first time (with room made: `_room`), with `cfg` applied."""
        key = runner_key(cfg)
        r = self.runners.get(key)
        if r is None:
            if self.on_build is not None:
                self.on_build(f"building {config_label(cfg)}")
            t0 = _time.perf_counter()
            kept = self.kept.get(key)  # a retry for room starts from it too
            kind = NL.NewtonRunner if cfg["sim"] in NL.SIMS else Runner
            r = self._room(lambda: kind(self.source, cfg, scene=self.scene, kept=kept))
            self.kept.pop(key, None)
            r.controls_playing = self.controls_playing
            self.runners[key] = r
            self.status = f"built {SIM_NAMES[cfg['sim']]} in {_time.perf_counter() - t0:.1f} s"
        if r.applied != cfg:
            r.apply(cfg)
        return r

    def free_runners(self, keep=None):
        """
        Free every runner but `keep`, and the GPU memory it holds -- a Newton one's process ends -- each handing its
        capacities and the demand it has seen to its configuration's next build (`kept`). Each is built again when
        its configuration is looked at.
        """
        for key, r in list(self.runners.items()):
            if r is keep:
                continue
            self.kept[key] = r.kept()
            if isinstance(r, Runner):
                r.release()
            else:
                r.close()  # its process, and the GPU memory it held
            del self.runners[key]
        gc.collect()

    def _room(self, build, keep=None):
        """
        `build()`; where the GPU has no room for it, every runner but `keep` is freed (`free_runners`) and it is
        tried once more.
        """
        try:
            return build()
        except RuntimeError as exc:
            if not _out_of_memory(exc) or not any(r is not keep for r in self.runners.values()):
                raise
        self.free_runners(keep=keep)
        return build()

    def play_controls(self, on=True):
        """
        Play the scene's controls (`SceneSource.controls`), or, with `on` False, hold the
        drives at the targets they last had while the scene runs: at its start, the scene's
        `ctrl`, a recording's first row. Played again, the controls resume at the scene's
        time, from the tick that holds it, at once.
        """
        self.controls_playing = bool(on)
        for r in self.runners.values():
            r.controls_playing = self.controls_playing
            if self.controls_playing:
                r.controls_due = True

    def pair_tables(self):
        """The collision pairs of the configuration looked at (`Runner.pair_tables`); none under a Newton sim."""
        cfg = self.config
        return self.runner(cfg).pair_tables

    def _toa_runner(self):
        """The runner of the configuration looked at, where it exists and runs a Lodos pair on time of arrival."""
        r = self.runners.get(runner_key(self.config)) if self.loaded and self.take is not None else None
        mcol = getattr(getattr(r, "sim", None), "mcol", None)
        return r if mcol is not None and int(mcol.has_toa) else None

    def toa_checks(self):
        """
        What the time-of-arrival checks report on the configuration looked at -- `lodos.col.sdf.collide`'s
        `check_toa_band` (a speculative gap read outside a baked grid) and `check_toa_bend` (a rotating body's arc
        bent off its straight line by more than half a voxel in a step) -- over everything its runner simulated since
        it was built or `clear_toa_checks`: their messages, none where neither reports anything. Synchronizes.
        """
        r = self._toa_runner()
        if r is None:
            return []
        from lodos.col.sdf import collide as sdf_collide

        m, d = r.sim.mcol, r.sim.dcol
        return [x for x in (sdf_collide.check_toa_band(m, d, warn=False), sdf_collide.check_toa_bend(m, d, warn=False))
                if x]

    def clear_toa_checks(self):
        """Forget what `toa_checks` reports: its runner's sticky flags and maxima, from the next step on."""
        r = self._toa_runner()
        if r is not None:
            r.sim.dcol.toa_band.zero_()
            r.sim.dcol.toa_bend.zero_()

    # -------------------------------------------------------------------------------
    # What is being looked at

    @property
    def config(self):
        return self.take.config

    @property
    def frame(self):
        t = self.take
        if t is None or not t.frames:
            return None
        return t.frames[min(self.frame_index, len(t.frames) - 1)]

    def select(self, i):
        """Look at take `i` (an index into `takes`), discarding a draft."""
        self.draft = None
        self.take = self.takes[i]
        self.frame_index = min(self.frame_index, len(self.take.frames) - 1)
        self.playing = False

    def seek(self, k):
        """Look at frame `k` of the current take. Leaves a draft, which has one frame."""
        if self.take is None:
            return
        self.frame_index = int(np.clip(k, 0, len(self.take.frames) - 1))

    def delete(self, i):
        """Forget take `i`."""
        t = self.takes.pop(i)
        self._bytes -= t.nbytes()
        if self.take is t:
            self.take = self.takes[min(i, len(self.takes) - 1)] if self.takes else None
            self.frame_index = 0
            if self.take is None:
                st = State(self.source.qpos0.copy(), self.source.qvel0.copy(), 0.0)
                self._make_draft(default_config(self.source, t.config["sim"]), st, None)

    # -------------------------------------------------------------------------------
    # Edits: each makes (or changes) a draft

    def _base(self):
        """`(config, state, parent)` an edit starts from: the frame being looked at."""
        if self.draft is not None and self.take is self.draft:
            return self.draft.config, State.of(self.draft.frames[0]), self.draft.parent
        fr = self.frame
        return self.take.config, State.of(fr), (self.take.id, self.frame_index)

    def _make_draft(self, cfg, st, parent):
        runner = self.runner(cfg)
        frame = self._forward(runner, st)
        dt = float(cfg["options"].get("timestep", self.source.mjm.opt.timestep))
        self.draft = Take(-1, copy.deepcopy(cfg), parent, [frame], dt, self.stride)
        self.take = self.draft
        self.frame_index = 0
        self.playing = False
        return self.draft

    def _forward(self, runner, st):
        """The state forwarded, with capacities grown until nothing overflows."""
        for _ in range(8):
            runner.set_state(st)
            frame = runner.forward()
            over = runner.overflow()
            if not over:
                runner.owner = ("forward", id(frame))
                return frame
            self._grow(runner, over)
        raise RuntimeError("the capacities kept overflowing")

    def _grow(self, runner, over):
        self._room(lambda: runner.grow(over), keep=runner)
        self.grown.append((self.frame.time if self.frame is not None else 0.0, sorted(over)))
        now = {**runner.caps, **getattr(runner, "collision", {})}
        self.status = f"grew {', '.join(sorted(over))} to " + ", ".join(f"{k}={now[k]:.4g}" for k in sorted(over))

    def tune_capacities(self, overhead=CAPACITY_OVERHEAD):
        """
        Size the capacities of the configuration looked at to what it has simulated: each at its largest demand over
        every step its runner took (`Runner.seen`) times 1 + `overhead`, rebuilt keeping the state. A step that
        outgrows them later grows them, as any overflow does. Under Newton's sims, Newton's capacities
        (`newton_link.NewtonRunner`).
        """
        overhead = float(overhead)
        if not overhead >= 0.0:
            raise ValueError(f"the overhead is a fraction of each peak, at least 0, not {overhead}")
        r = self.runner(self.config)
        if not r.steps_seen:
            raise ValueError("nothing has been simulated under this configuration yet: play it first")
        sizes = self._room(lambda: r.tune_capacities(overhead), keep=r)
        parts = [f"{name} {a:,}" for name, a, _ in r.capacities()]
        parts += [f"{k} {v:.3g}" for k, v in r.fractions().items()]
        self.status = f"capacities at {1.0 + overhead:.3g}x the peaks of {r.steps_seen} steps: " + ", ".join(parts)
        return sizes

    def export_config(self, path):
        """
        Write everything the panel sets on the configuration looked at to the JSON file `path`: the sim and every
        option it shows; its capacities (per world, with the demand seen: Lodos's, or under Newton's sims Newton's);
        the contact parameters of every pair it applies -- Lodos's and MJWarp's, or under Newton's MuJoCo Warp
        Newton's: the values most pairs share and each pair that differs, as set (the exact mode applies friction,
        sidedness and Lodos's time of arrival alone) -- and the overrides as the take files hold them
        (`pairs.overrides`). Returns the path written.
        """
        cfg, src = self.config, self.source
        out = dict(format=CONFIG_FORMAT, scene=src.name, path=src.path, sim=cfg["sim"], options=dict(cfg["options"]))
        r = self.runner(cfg)
        rows = r.capacities()
        per_world = {k: int(v) for k, v in r.caps.items() if (k in ("nconmax", "njmax", "nccdmax") and v)
                     or cfg["sim"] != "lodos"}
        out["capacities"] = dict(per_world=per_world, collision=r.fractions(),
                                 allocated={name: a for name, a, _ in rows}, peaks={name: p for name, _, p in rows},
                                 steps=r.steps_seen)
        alone = cfg["sim"] == "kamino" or (cfg["sim"] == "lodos" and cfg["options"].get("mode") != "mj")
        out["pairs"] = {kind: _pairs_json(t, cfg, alone) for kind, t in r.pair_tables.items()}
        if out["pairs"]:
            out["pairs"].update(prim_geom_mode=cfg["pairs"]["mode"], overrides=_config_to_json(cfg)["pairs"])
        path = Path(path)
        path.write_text(json.dumps(out, indent=1, default=lambda x: x.item()))
        return path

    def edit_state(self, qpos=None, qvel=None):
        """Change the state being looked at; a draft from it."""
        cfg, st, parent = self._base()
        if qpos is not None:
            st.qpos = np.asarray(qpos, np.float64).copy()
        if qvel is not None:
            st.qvel = np.asarray(qvel, np.float64).copy()
        self._make_draft(cfg, st, parent)

    def set_sim(self, sim):
        cfg, st, parent = self._base()
        if sim == cfg["sim"]:
            return
        new = default_config(self.source, sim)
        new["pairs"] = copy.deepcopy(cfg["pairs"])
        why = toa_conflict(self.source, new)
        if why:
            raise ValueError(why)
        self._make_draft(new, st, parent)

    def set_option(self, name, value):
        cfg, st, parent = self._base()
        new = edit_config(self.source, cfg, name, value)
        if new is not cfg:
            self._make_draft(new, st, parent)

    def set_pairs(self, kind, indices, field, value):
        """
        Set `field` -- "mu", "solref", "solimp", or Lodos's "sided" and time of arrival (`viewer.pairs.TOA_ALL`:
        "toa", "toa_trace", "toa_trace_iters", "toa_drift", "toa_side", and "toa_horizon" on every pair alone) --
        to `value` on the pairs `indices` of `kind` (`viewer.pairs.KINDS`), or with `value` None restore the
        scene's (`edit_pairs`). A pair made one- or two-sided, or given its time of arrival, is a rebuild.
        """
        cfg, st, parent = self._base()
        n = self.runner(cfg).pair_counts.get(kind, 0)
        if n == 0:
            raise ValueError(f"{SIM_NAMES[cfg['sim']]} has no {kind} pairs in this scene")
        if indices is not None and any(not 0 <= int(i) < n for i in np.atleast_1d(indices)):
            raise ValueError(f"the scene has {n} {kind} pairs")
        new = edit_pairs(self.source, cfg, kind, indices, field, value)
        if new is not cfg:
            self._make_draft(new, st, parent)

    def set_all_pairs(self, field, value):
        """
        `field` at `value` on every pair of every kind the scene has (Lodos's "sided" on its pairs without a plane),
        or with `value` None back at the scene's: one edit (`edit_all_pairs`).
        """
        cfg, st, parent = self._base()
        kinds = list(self.runner(cfg).pair_counts)
        if not kinds:
            raise ValueError(f"{SIM_NAMES[cfg['sim']]} takes the scene's own contact parameters")
        new = edit_all_pairs(self.source, cfg, kinds, field, value)
        if new is not cfg:
            self._make_draft(new, st, parent)

    def set_geom_mode(self, mode):
        """How a geom takes the values set on MJWarp's pairs it is in: their "min", "max" or "mean"."""
        if mode not in VP.GEOM_MODES:
            raise ValueError(f"the modes are {VP.GEOM_MODES}, not {mode!r}")
        cfg, st, parent = self._base()
        if cfg["pairs"]["mode"] != mode:
            new = copy.deepcopy(cfg)
            new["pairs"]["mode"] = mode
            self._make_draft(new, st, parent)

    def reset_state(self, key=None):
        """A draft from the scene's initial state, or keyframe `key`, under the current configuration."""
        cfg = self.take.config if self.take is not None else default_config(self.source, "lodos")
        if key is None:
            st = State(self.source.qpos0.copy(), self.source.qvel0.copy(), 0.0)
        else:
            _, q, v = self.source.keyframes[key]
            st = State(np.array(q, np.float64), np.array(v, np.float64), 0.0)
        self._make_draft(cfg, st, None)

    # -------------------------------------------------------------------------------
    # Playing

    def play(self):
        if self.take is None:
            return
        if self.draft is not None and self.take is self.draft:
            self._commit()
        self.playing = True
        self._last = None

    def pause(self):
        self.playing = False

    def toggle(self):
        if self.playing:
            self.pause()
        else:
            self.play()

    def _commit(self):
        t = self.draft
        self._bytes += t.nbytes()
        t.id = self._next_id
        self._next_id += 1
        t.tail = None
        self.takes.append(t)
        self.draft = None
        # The runner still holds the draft's state, forwarded: the take's first step
        # then only integrates it.
        r = self.runner(t.config)
        if r.owner == ("forward", id(t.frames[0])):
            r.owner = (t.id, 0)

    def _runner_for(self, take):
        """The take's runner, holding the state the take continues from."""
        r = self.runner(take.config)
        if r.owner != (take.id, take.steps):
            r.set_state(take.tail if take.tail is not None else take.start)
            r.owner = (take.id, take.steps)
        return r

    def simulate(self, n, deadline=None):
        """
        Step the current take `n` times, or until `perf_counter()` passes `deadline`,
        recording every `stride`-th state. Returns the steps taken. A frame is recorded
        when the step count is a multiple of the stride and that frame does not exist
        yet -- which is also what makes a rollback re-record the frame it dropped.
        """
        take = self.take
        if take is None or take.id < 0:
            return 0
        r = self._runner_for(take)
        done = retries = 0
        while done < n and (deadline is None or _time.perf_counter() < deadline):
            record = take.steps % take.stride == 0 and len(take.frames) == take.steps // take.stride
            fr = r.step(record)
            if fr is not None:
                take.frames.append(fr)
                self._bytes += frame_bytes(fr)
            over = r.overflow()
            if over:
                retries += 1  # in a row: a step that no growth makes fit
                if retries > 8:
                    raise RuntimeError("the capacities kept overflowing")
                self._rollback(take, r, over)
                continue
            take.steps += 1
            done += 1
            retries = 0
        take.tail = r.state()
        r.owner = (take.id, take.steps)
        return done

    def _rollback(self, take, r, over):
        """
        A step overflowed one of MJWarp's capacities, so its contacts were incomplete:
        grow them, drop the last recorded frame and step again from the state it holds.
        Every step since that frame is redone, recorded or not.
        """
        self._grow(r, over)
        j = len(take.frames) - 1
        fr = take.frames.pop() if j > 0 else take.frames[0]
        if j > 0:
            self._bytes -= frame_bytes(fr)
        take.steps = j * take.stride
        st = State.of(fr)
        take.tail = st if take.steps > 0 else None
        r.set_state(st)
        r.owner = (take.id, take.steps)
        if j == 0:
            # The first frame stays (it is what the take starts from); its step is
            # redone unrecorded, from the same state.
            old = take.frames[0]
            take.frames[0] = self._forward(r, st)
            self._bytes += frame_bytes(take.frames[0]) - frame_bytes(old)
            r.owner = (take.id, 0)

    def tick(self, now=None):
        """
        Advance playback by the wall time since the last tick: through recorded frames
        while there are any ahead, then by simulating. Paced to `rate` times real time,
        or as fast as `TICK_BUDGET` allows with `max_speed`.
        """
        now = _time.perf_counter() if now is None else now
        elapsed = 0.0 if self._last is None else min(now - self._last, 0.25)
        self._last = now
        if not self.playing or self.take is None:
            return
        take = self.take
        if self.frame_index < len(take.frames) - 1:
            self._acc += elapsed * self.rate / take.frame_dt
            n, self._acc = divmod(self._acc, 1.0)
            self.frame_index = min(self.frame_index + int(n), len(take.frames) - 1)
            return
        if self.status.startswith("paused") and self._bytes <= self.memory_budget:
            self.status = ""
        if self._bytes > self.memory_budget:
            self.playing = False
            self.status = (f"paused: the recording holds {self._bytes / 1e9:.1f} GB, the budget; delete takes or "
                           "record every n-th step")
            return
        t0 = _time.perf_counter()
        want = 1 << 30 if self.max_speed else self._acc + elapsed * self.rate / self.dt
        done = self.simulate(int(want), deadline=t0 + TICK_BUDGET) if want >= 1 else 0
        # Behind (the budget ran out first): drop the backlog rather than owe it, so a
        # scene slower than the rate runs as fast as it can instead of lurching.
        self._acc = want - done if done == int(want) and not self.max_speed else 0.0
        if elapsed > 0:
            rtf = done * self.dt / elapsed
            self.rtf = rtf if self.rtf == 0.0 else 0.8 * self.rtf + 0.2 * rtf
        self.frame_index = len(take.frames) - 1

    def step_once(self):
        """One recorded frame forward: the next one if recorded, else simulate up to it."""
        if self.take is None:
            return
        if self.draft is not None and self.take is self.draft:
            self._commit()
        take = self.take
        if self.frame_index < len(take.frames) - 1:
            self.frame_index += 1
            return
        n0 = len(take.frames)
        for _ in range(64 * take.stride):
            self.simulate(1)
            if len(take.frames) > n0:
                break
        self.frame_index = len(take.frames) - 1

    def memory(self):
        """Bytes the recorded frames hold."""
        return self._bytes + (self.draft.nbytes() if self.draft is not None else 0)
