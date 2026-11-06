"""
Where a live session's scene comes from: a built-in scene, or an MJCF file.

A `SceneSource` is a scene before it is a simulation -- the MJCF model, the Lodos
half if there is one, the initial state and the capacities -- because a live session
builds it more than once: once per configuration that allocates differently, and again
whenever a capacity turns out to be too small. Every build is one world (`B = 1`).

## Built-in scenes

`BUILTIN` names the scenes `viewer.scenes` builds, at the seed the session asks for. The
viewer's panel lists `PANEL`, in this order: the FBF benchmarks; cubes,
armadillos, SHARPA and Shadow hands dropped on a floor and in a walled box, five and twenty;
the SHARPA hand holding an armadillo and the Shadow hand's grasp of the Kong toy; the NIST
threading, the nut on its bolt at each size and remesh, the driven nut and the board. The
rest load by name only: the mixed piles that hold both kinds of pair (`arm_box_*`),
Newton's nut-bolt (`nutbolt_*`) and two SHARPA hands (`sharpa*_2_*`). A scene that exists
in both collision representations carries its representation as a suffix: `_sdf` collides
as Lodos surfaces, `_polytope` as MJCF convex pieces or primitives through MJWarp's
pipeline, from the same bodies and the same stacks. Scenes whose assets are not on this
machine are left out.

## MJCF files

Any MJCF loads, and by default everything in it collides through MJWarp's pipeline. A
geom becomes a Lodos surface by a custom text field named after it:

    <custom>
      <text name="lodos:arm0" data="armadillo"/>   <!-- geom "arm0" is the shipped surface -->
      <text name="lodos:shell" data="bake/surface_data.npz"/>   <!-- a bake of your own -->
      <text name="lodos:floor" data="plane"/>      <!-- a plane, for both pipelines -->
      <numeric name="lodos:mu" data="0.8"/>        <!-- optional pair parameters -->
      <numeric name="lodos:stiffness" data="5000"/>
      <numeric name="lodos:dissipation" data="0.15"/>
      <text name="lodos:controls" data="controls.npz"/>   <!-- pre-recorded controls -->
    </custom>

The data is a shipped surface's name (`viewer/assets/surfaces/<name>`), a path to a
`surface_data.npz` (relative to the MJCF file), or `plane` for a plane geom. A
surface's vertices are in its body's frame, as `lodos.col.sdf` takes them, so the geom
it is keyed to -- the one MuJoCo takes the body's inertia from -- should be the same
mesh placed with no offset, as `viewer.scenes.assets.surface_obj` writes it. The pairs are
every ordered pair of surfaces on different bodies and every surface against every
Lodos plane, as in `viewer.scenes.armadillo`.

The initial state is the model's first keyframe if it has one, else `qpos0` at rest.
`lodos:controls` names no geom: it is a recording of the actuators' controls, an `.npz`
relative to the MJCF file (`RecordedControls`), which the viewer plays from the initial
state on.

A file that names no friction cone (`<option cone>`, in it or in a file it includes) gets
the elliptic one, as every built-in scene has, rather than MuJoCo's default, the pyramidal
one; a file that names a cone keeps it.
"""

from __future__ import annotations

import copy
import dataclasses
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from lodos import sim as lsim
from viewer.scenes.assets import SURF_DIR, load_surface
from lodos.cd import driver as cd_driver
from lodos.col import sdf
from lodos.col.sdf import types as sdf_types
from lodos.rbd import io as rbd_io

# MJWarp's capacities at one world when a scene does not name its own. They grow when
# a step overflows them (`viewer.session`), so these only set where that starts.
DEFAULT_CAPS = dict(nconmax=1024, njmax=4096)

# The options only the exact mode takes, its Baumgarte bias's: a scene's defaults for
# them stay out of a build under mj, which the solver refuses them under.
EXACT_OPTIONS = ("baumgarte", "baumgarte_alpha", "baumgarte_slop")

# MuJoCo's own solver options, which the solver reads from the model rather than taking as
# options: a scene's defaults may set them, and its builds write them on their copy of the model.
MODEL_OPTIONS = ("iterations", "tolerance", "ls_iterations", "ls_tolerance")

# What a scene's defaults may set: the solver's options (`lodos.cd.driver.OPTIONS`) and MuJoCo's.
SCENE_OPTIONS = tuple(cd_driver.OPTIONS) + MODEL_OPTIONS


@dataclasses.dataclass(eq=False)
class RecordedControls:
    """
    Pre-recorded controls, a `SceneSource.controls`: row k of `ctrl`, `(T, nu)`, over the k-th
    `period` (s) of the scene's time from 0, and the last row after the recording ends. `load`
    reads one from an `.npz` holding `ctrl` and `period`, as `save` writes it; an MJCF file names
    its own by its `lodos:controls` annotation, and `python -m viewer SCENE --controls
    FILE.npz` plays one in a scene with as many actuators (`with_controls`).
    """

    ctrl: np.ndarray
    period: float

    def __post_init__(self):
        self.ctrl = np.asarray(self.ctrl, dtype=np.float64)
        if self.ctrl.ndim != 2 or len(self.ctrl) == 0:
            raise ValueError(f"recorded controls are (T, nu) with T >= 1, not {self.ctrl.shape}")
        if not np.isfinite(self.ctrl).all():
            raise ValueError("recorded controls must be finite")
        self.period = float(self.period)
        if not self.period > 0.0:
            raise ValueError(f"recorded controls' period must be > 0 s, not {self.period}")

    @property
    def nu(self):
        return self.ctrl.shape[1]

    @property
    def duration(self):
        """The recording's length, s."""
        return len(self.ctrl) * self.period

    def tick(self, t):
        """The row that holds at time `t`: counted a hundredth of a period early, so that the drift
        of time accumulated in float32 cannot hold a tick back."""
        return min(max(int(np.floor(float(t) / self.period + 0.01)), 0), len(self.ctrl) - 1)

    def __call__(self, t, qpos=None):
        return self.ctrl[self.tick(t)]

    @classmethod
    def load(cls, path):
        with np.load(path) as z:
            missing = sorted({"ctrl", "period"} - set(z.files))
            if missing:
                raise ValueError(f"{path}: recorded controls need {missing}")
            return cls(z["ctrl"], float(z["period"]))

    def save(self, path):
        np.savez(path, ctrl=self.ctrl, period=self.period)


def with_controls(source, controls):
    """`source` playing `controls` from its initial state, holding their first row where nothing plays them."""
    return dataclasses.replace(source, controls=controls, ctrl=np.array(controls(0.0, source.qpos0), dtype=np.float64))



def spline_of(surfaces):
    """The SDF representation `surfaces`' Lodos surfaces are evaluated with ("cubic" or "linear"; a
    context's surfaces agree on one), or None where there are none but half-spaces."""
    for sf in surfaces.surfaces:
        if not isinstance(sf, sdf.PlaneSurface):
            return str(np.asarray(getattr(sf, "spline_type", sdf.DEFAULT_SPLINE_TYPE)).item())
    return None


def respline(surfaces, spline_type):
    """
    `surfaces` with every Lodos surface evaluated as `spline_type` (`lodos.col.sdf.with_spline_type`):
    each distinct surface converted once, as a pile shares one among its bodies; half-spaces and
    surfaces already of that type as they are.
    """
    done = {}
    out = []
    for sf in surfaces.surfaces:
        if not isinstance(sf, sdf.PlaneSurface) and str(np.asarray(sf.spline_type).item()) != spline_type:
            if id(sf) not in done:
                done[id(sf)] = sdf.with_spline_type(sf, spline_type)
            sf = done[id(sf)]
        out.append(sf)
    return dataclasses.replace(surfaces, surfaces=out)

@dataclasses.dataclass
class SceneSource:
    """
    A scene, ready to be built as a one-world `lodos.sim.Sim`.

    name: what the panel calls it.
    mjm: the MJCF model as loaded. `make_sim` copies it before switching surfaces'
        geoms off, so this one is never changed.
    surfaces: the Lodos half, or None.
    qpos0, qvel0: the initial state, `(nq,)` and `(nv,)`.
    caps: MJWarp's capacities per world (`nconmax`, `njmax`, `nccdmax`, `njmax_nnz`).
    sim_kwargs: everything else `make_sim` takes from the scene (`contact_reduction`, ...).
    solver_defaults: the solver's options at the scene's values (`defaults`), any it takes
        (`lodos.cd.driver.OPTIONS`), and MuJoCo's `MODEL_OPTIONS`, which its builds write on the
        model.
    group_bins: the binning a scene's contacts take when the viewer groups them,
        `dict(normal_bins, moment_bins)`; None for 1 x 1.
    path: the MJCF file, for one loaded from disk.
    xml: the MJCF text, for a scene generated in code; what `export_mjcf` writes.
    keyframes: `(name, qpos, qvel)` for each of the model's keyframes.
    ctrl: the control the scene holds, `(nu,)`, applied to every world it builds -- a
        grasp's closed fingers; None leaves the model's, zero.
    controls: in the viewer, a control that changes as the scene plays, or None to hold
        `ctrl`: an object with a `period` (s) and a call `controls(t, qpos) -> (nu,)`, which
        the viewer makes at every multiple of `period` of the scene's time, before that step
        (pre-recorded controls, `RecordedControls`; an open-loop replay of recorded actions,
        `viewer.scenes.nist.ActionReplay`). The tuning tools hold `ctrl`.
    sampler: `sampler(n, seed)` -> `(n, nq)` starts drawn the way the scene draws its own
        (a drop's stacks, the hands' orientations), for the tuning tools' exploration; None
        for a scene with one start.
    mode_defaults: the solver's options under one mode, `{mode: options}`, over
        `solver_defaults` when the solver runs under that mode (`defaults`): 2 outer
        proximal iterations under exact and MuJoCo's tolerance 1e-5 under mj, say. Not the
        mode itself, and none of `EXACT_OPTIONS` under mj.
    newton_shapes: what Newton's sims collide for a Lodos surface, by the id of the geom it
        is keyed to (`newton_link.scene_package`): `dict(vertices, faces, sdf=dict(
        max_resolution, band))`, a mesh in the body's frame with an SDF at Newton's own
        settings, or None for the MJCF's geom, which Newton collides as the MJCF has it. A
        surface not named, or every one where this is None, is its own mesh, its SDF at its
        bake's voxel size.
    newton_xml: the MJCF Newton's sims import, where it is not `xml`, or one for each,
        `{sim: xml}`: the same bodies, joints and actuators, as Newton should simulate them
        (`nist.newton_xml`).
    newton_controls: the controls Newton's sims play, where they are not `controls`
        (`nist.newton_controls`, for the drives of `newton_xml`).
    newton_options: Newton's options at the scene's values (`newton_link.default_options`).
    """

    name: str
    mjm: mujoco.MjModel
    surfaces: lsim.Surfaces | None = None
    qpos0: np.ndarray | None = None
    qvel0: np.ndarray | None = None
    caps: dict = dataclasses.field(default_factory=dict)
    sim_kwargs: dict = dataclasses.field(default_factory=dict)
    solver_defaults: dict = dataclasses.field(default_factory=dict)
    path: str | None = None
    xml: str | None = None
    keyframes: list = dataclasses.field(default_factory=list)
    ctrl: np.ndarray | None = None
    sampler: object = None
    controls: object = None
    mode_defaults: dict = dataclasses.field(default_factory=dict)
    group_bins: dict | None = None
    newton_shapes: dict | None = None
    newton_xml: str | dict | None = None
    newton_controls: object = None
    newton_options: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if self.qpos0 is None:
            self.qpos0 = np.array(self.mjm.qpos0, dtype=np.float64)
        if self.qvel0 is None:
            self.qvel0 = np.zeros(self.mjm.nv)
        self.qpos0 = np.asarray(self.qpos0, dtype=np.float64).reshape(self.mjm.nq)
        self.qvel0 = np.asarray(self.qvel0, dtype=np.float64).reshape(self.mjm.nv)
        caps = dict(DEFAULT_CAPS)
        caps.update(self.caps)
        self.caps = caps
        unknown = sorted(set(self.solver_defaults) - set(SCENE_OPTIONS))
        if unknown:
            raise ValueError(f"{self.name}: the solver takes no {unknown}")
        self.solver_defaults = dict(self.solver_defaults)
        for mode, options in self.mode_defaults.items():
            if mode not in cd_driver.MODES:
                raise ValueError(f"{self.name}: defaults under {mode!r}, not one of {cd_driver.MODES}")
            unknown = sorted(set(options) - (set(SCENE_OPTIONS) - {"mode"}))
            if unknown:
                raise ValueError(f"{self.name}: the solver under {mode} takes no {unknown}")
            if mode == "mj" and set(options) & set(EXACT_OPTIONS):
                raise ValueError(f"{self.name}: the mj mode takes none of {EXACT_OPTIONS}")
        self.mode_defaults = {mode: dict(options) for mode, options in self.mode_defaults.items()}
        if isinstance(self.controls, RecordedControls) and self.controls.nu != self.mjm.nu:
            raise ValueError(f"{self.name}: recorded controls for {self.controls.nu} actuators, the model has {self.mjm.nu}")

    def defaults(self, mode=None):
        """
        The solver's options at the scene's values, then its `mode_defaults` under `mode`
        (None: the scene's own mode, else exact); under mj none of `EXACT_OPTIONS`.
        """
        kw = dict(self.solver_defaults)
        mode = mode or kw.get("mode", "exact")
        kw.update(self.mode_defaults.get(mode, {}))
        return {k: v for k, v in kw.items() if mode != "mj" or k not in EXACT_OPTIONS}

    def make_sim(self, solver_kwargs=None, overrides=(), caps=None, B=1, qpos=None, qvel=None,
                 tuning=None, sim_options=None, sided=None):
        """
        `B` worlds of the scene, each at the initial state unless `qpos`.

        solver_kwargs: the solver's options over the scene's (`defaults`), and
            `contact_reduction` over the scene's.
        overrides: MJWarp model overrides, after the scene's own.
        caps: capacities over `self.caps`.
        qpos, qvel: `(B, nq)`, `(B, nv)` starts, or None for the scene's own in every world.
        tuning: a tuning file (`lodos.utils.tuning`). Its settings replace the scene's own
            defaults -- `caps`, `solver_defaults`, `sim_kwargs` -- where it has them at
            this `B`; what is passed here explicitly still wins over it.
        sim_options: `make_sim`'s other options over the scene's `sim_kwargs`
            (`integrator`, `implicit_constraints`, ...), and `timestep`, which a build writes on
            its copy of the model, and `spline_type`, the SDF representation its Lodos surfaces
            are evaluated with ("cubic" or "linear", `respline`). The viewer passes those four
            with the solver's options, which take them out to here.
        sided: `{pair index: 1 or 2}`, Lodos pairs made one-sided (the first surface's vertices
            against the second's SDF) or two-sided (both ways, `cpr_is_symmetric`) over the scene's
            own. A pair with a plane is one-sided: a plane has no vertices.
        """
        solver_kwargs = dict(solver_kwargs or {})
        sim_options = dict(sim_options or {})
        for k in ("integrator", "implicit_constraints", "timestep", "spline_type"):
            if k in solver_kwargs:
                sim_options[k] = solver_kwargs.pop(k)
        spline = sim_options.pop("spline_type", None)
        surfaces = self.surfaces if spline is None or self.surfaces is None else respline(self.surfaces, spline)
        if sided:
            if surfaces is None:
                raise ValueError(f"{self.name} has no Lodos pairs to make one- or two-sided")
            n = len(surfaces.pairs)
            sym = [bool(x) for x in (surfaces.col_kwargs.get("cpr_is_symmetric") or [False] * n)]
            for j, s in dict(sided).items():
                if not 0 <= int(j) < n or s not in (1, 2):
                    raise ValueError(f"pair {j} of {n} cannot be {s}-sided")
                a, b = surfaces.pairs[int(j)]
                if s == 2 and any(isinstance(surfaces.surfaces[g], sdf_types.PlaneSurface) for g in (a, b)):
                    raise ValueError(f"pair {j} has a plane, which has no vertices: it is one-sided")
                sym[int(j)] = s == 2
            surfaces = dataclasses.replace(surfaces, col_kwargs=dict(surfaces.col_kwargs, cpr_is_symmetric=sym))
        sim_kwargs = dict(self.sim_kwargs)
        scene_caps = dict(self.caps)
        given = {}
        if tuning is not None:
            from lodos.utils import tuning as ltuning

            a = ltuning.apply(tuning, B, quiet=True)
            for k in a["runtime"]:
                scene_caps.pop(k, None)
            if "naconmax" in a["runtime"]:
                scene_caps.pop("nconmax", None)
            for k in a["sim"]:
                sim_kwargs.pop(k, None)
            given.update(a["solver_kwargs"])
        if "contact_reduction" in solver_kwargs:
            sim_kwargs["contact_reduction"] = bool(solver_kwargs["contact_reduction"])
        # The scene's defaults, a tuning file's over them and what is passed here over both,
        # of which the solver keeps those it takes.
        given.update(solver_kwargs)
        mode = given.get("mode", self.defaults().get("mode"))
        merged = dict(self.defaults(mode), **given)
        kw = {k: v for k, v in merged.items() if k in cd_driver.OPTIONS}
        mjm = self.mjm
        model = {k: merged[k] for k in MODEL_OPTIONS if k in merged}
        if model:  # MuJoCo's own solver options, on the copy of the model the solver's MJWarp is made from
            mjm = copy.copy(self.mjm)
            for k, v in model.items():
                setattr(mjm.opt, k, type(getattr(mjm.opt, k))(v))
        timestep = sim_options.pop("timestep", None)
        if timestep is not None and float(timestep) != float(mjm.opt.timestep):  # the step, on a copy too
            mjm = copy.copy(self.mjm) if mjm is self.mjm else mjm
            mjm.opt.timestep = float(timestep)
        c = dict(scene_caps)
        c.update(caps or {})
        c.setdefault("njmax_nnz", None)
        B = int(B)
        q = np.tile(self.qpos0, (B, 1)) if qpos is None else np.asarray(qpos, dtype=np.float64).reshape(B, -1)
        sim = lsim.make_sim(
            mjm, B, surfaces=surfaces, qpos=q,
            solver_kwargs=kw,
            overrides=tuple(overrides), tuning=tuning, **c, **dict(sim_kwargs, **(sim_options or {})),
        )
        v = np.tile(self.qvel0, (B, 1)) if qvel is None else np.asarray(qvel, dtype=np.float64).reshape(B, -1)
        if np.any(v):
            rbd_io.set_state(sim.drbd, qvel=v)
        if self.ctrl is not None:
            rbd_io.set_state(sim.drbd, ctrl=np.tile(np.asarray(self.ctrl, dtype=np.float64).reshape(1, -1), (B, 1)))
        return sim


#######################################################################################
# Built-in scenes
#######################################################################################


# A scene in both collision representations takes its representation as a suffix:
# `_sdf` collides as Lodos surfaces, `_polytope` as MJCF convex pieces or primitives
# through MJWarp's pipeline (`viewer.scenes.representations`' "surfaces" and "polytopes").
GEOMETRIES = {"sdf": "surfaces", "polytope": "polytopes"}


def _surface_available(name):

    return (SURF_DIR / name / "surface_data.npz").exists()


def _drop(name, kind, n_bodies, walls, geometry, seed):
    """`viewer.scenes.representations`' drop of armadillos or cubes, as `geometry`."""
    from viewer.scenes import drop as D
    from viewer.scenes import representations as R

    rep = GEOMETRIES[geometry]
    xml, q0 = R.scene_and_q0(n_bodies, R.default_columns(n_bodies), rep, 1, seed, kind, walls)
    mjm = mujoco.MjModel.from_xml_string(xml)
    surfaces = R.surfaces(mjm, n_bodies, kind=kind, walls=walls) if rep == "surfaces" else None
    scale = max(1, n_bodies // 5)
    if rep == "surfaces":
        # Lodos's contacts are MJWarp contacts too, a few thousand of them in a
        # pile of five.
        caps = dict(nconmax=4096 * scale, njmax=12288 * scale)
    elif kind == "cube":
        caps = dict(D.CAPS[f"cube{n_bodies}"])
    else:
        caps = dict(nconmax=2048 * scale, njmax=4096 * scale)
    cols = R.default_columns(n_bodies)
    return SceneSource(name, mjm, surfaces, q0[0], caps=caps, xml=xml,
                       sampler=lambda n, seed: R.scene_and_q0(n_bodies, cols, rep, n, seed, kind, walls)[1])


def _mixed(name, scene, seed):
    from viewer.scenes import mixed as M

    xml, q0 = M.scene_and_q0(scene, 1, seed)
    mjm = mujoco.MjModel.from_xml_string(xml)
    surfaces = M.surfaces(mjm, M.SCENES[scene]["n_bodies"])
    return SceneSource(name, mjm, surfaces, q0[0], caps=dict(M.CAPS[scene]), xml=xml,
                       sampler=lambda n, seed: M.scene_and_q0(scene, n, seed)[1])


def _nutbolt(name, nut, spline, seed):
    from viewer.scenes import nutbolt as NB

    xml = NB.scene_xml()
    mjm = mujoco.MjModel.from_xml_string(xml)
    surfaces = NB.surfaces(mjm, nut=nut, spline=spline)
    # Every Lodos contact is an MJWarp contact unless reduced: ~2300 a world mid-thread,
    # ~4400 seated. The session doubles these on overflow.
    caps = dict(nconmax=8192, njmax=24576)
    # Seed 0 is Newton's start; seed k > 0 is world k of a batch of jittered starts
    # (`initial_qpos(jitter=0.05)`), so that a world that fails to seat, or
    # one that rattles on its seat, can be watched.
    q0 = NB.initial_qpos(seed + 1, jitter=0.05 if seed else 0.0)[seed]
    return SceneSource(name + (f", start {seed}" if seed else ""), mjm, surfaces, q0, caps=caps, xml=xml,
                       group_bins=NB.GROUP_BINS, sim_kwargs=dict(contact_reduction=True),
                       sampler=lambda n, seed: NB.initial_qpos(n, jitter=0.05, seed=seed))


def _nist_nutbolt(name, size, nut, seed):
    from viewer.scenes import nist as NI

    z, turn = NI.start_z(size), NI.thread_phase(size, nut)[0]
    xml = NI.scene_xml(size, z, turn)
    mjm = mujoco.MjModel.from_xml_string(xml)
    surfaces = NI.surfaces(mjm, size, nut)
    # Every Lodos contact is an MJWarp contact unless reduced, as on the M20 nut-bolt.
    caps = dict(nconmax=8192, njmax=24576)
    q0 = NI.initial_qpos(seed + 1, size, z, turn, jitter=0.05 if seed else 0.0)[seed]
    return SceneSource(name + (f", start {seed}" if seed else ""), mjm, surfaces, q0, caps=caps, xml=xml,
                       solver_defaults=NI.OPTIONS, group_bins=NI.GROUP_BINS,
                       sim_kwargs=dict(contact_reduction=True), newton_shapes=NI.newton_shapes(mjm, surfaces),
                       sampler=lambda n, seed: NI.initial_qpos(n, size, z, turn, jitter=0.05, seed=seed))


def _nist_driven(name, size):
    """The lone pair with the nut turned on at a turn a second, as the arm would, at the demo's friction."""
    from viewer.scenes import nist as NI

    nut = NI.default_nut(size)
    z, turn = NI.start_z(size), NI.thread_phase(size, nut)[0]
    xml = NI.driven_xml(size, z, turn)
    mjm = mujoco.MjModel.from_xml_string(xml)
    surfaces = NI.surfaces(mjm, size, nut, mu=NI.MU_BOARD)
    return SceneSource(name, mjm, surfaces, np.zeros(4), caps=dict(nconmax=8192, njmax=24576), xml=xml,
                       solver_defaults=NI.OPTIONS, group_bins=NI.GROUP_BINS,
                       sim_kwargs=dict(contact_reduction=True), newton_shapes=NI.newton_shapes(mjm, surfaces),
                       ctrl=np.array([-2 * np.pi]), sampler=lambda n, seed: np.zeros((n, 4)))


def _nist_board(name, seed):
    from viewer.scenes import nist as NI

    xml = NI.board_xml()
    mjm = mujoco.MjModel.from_xml_string(xml)
    surfaces = NI.board_surfaces(mjm)
    caps = dict(nconmax=16384, njmax=49152)  # three nuts' contacts, as for one pair
    q0 = NI.board_qpos(seed + 1, jitter=0.05 if seed else 0.0)[seed]
    return SceneSource(name + (f", start {seed}" if seed else ""), mjm, surfaces, q0, caps=caps, xml=xml,
                       solver_defaults=NI.OPTIONS, group_bins=NI.GROUP_BINS,
                       sim_kwargs=dict(contact_reduction=True), newton_shapes=NI.newton_shapes(mjm, surfaces),
                       sampler=lambda n, seed: NI.board_qpos(n, jitter=0.05, seed=seed))


def _nist_threading(name, start=None):
    """
    Isaac Lab's NIST policy threading the M8 nut onto its bolt, replayed (`viewer.scenes.nist.threading`):
    the Franka from the recorded state of the policy's decision 280, or the state in `start`
    (`nist.THREADING_EASY`), its drives' targets the recording's (`RecordedControls`, a row a 10 ms tick
    for 6.2 s), its fingers on stiffer drives that follow their recorded path. The exact mode runs at
    2 outer proximal iterations, and mj at MuJoCo's tolerance 1e-5 (`nist.THREADING_OPTIONS`).
    """
    from viewer.scenes import nist as NI

    mjm, xml, q0, v0, ctrl, period = NI.threading(start=start)
    controls = RecordedControls(ctrl, period)
    # Newton's sims run it as Isaac Lab's demo runs on Newton (`nist.newton_xml`): its parts' gaps their own,
    # the gripper's colliders none.
    newton = dict(newton_xml={s: NI.newton_xml(xml, s) for s in ("mjwarp", "kamino")},
                  newton_controls=RecordedControls(*NI.newton_controls()), newton_options=dict(gap=0.0))
    # Capacities at 1.5x the peaks over the whole replay (137 contacts, 450 rows under the exact mode and 600
    # under mj), which the fused_newton solve fits in shared memory; the session doubles them if a step ever
    # overflows.
    caps = dict(nconmax=137, njmax=600)
    surfaces = NI.franka_surfaces(mjm, "m8")
    return SceneSource(name, mjm, surfaces, q0, v0, caps=caps, xml=xml,
                       solver_defaults=NI.OPTIONS, group_bins=NI.GROUP_BINS,
                       mode_defaults=NI.THREADING_OPTIONS, newton_shapes=NI.newton_shapes(mjm, surfaces), **newton,
                       sim_kwargs=dict(contact_reduction=True), ctrl=controls(0.0), controls=controls)


# Newton's nut-bolt, offered only where its surfaces are (`viewer.scenes.nutbolt.ASSET_DIR`).
def _nutbolt_available(nut):
    from viewer.scenes import nutbolt as NB

    return NB.assets_present() and (NB.ASSET_DIR / "surfaces" / nut / "surface_data.npz").exists()


def _fbf(name, scene, geometry):
    """One of `viewer.scenes.fbf`'s exactness scenes, as `geometry`. They have no seed."""
    from viewer.scenes import fbf as F

    rep = GEOMETRIES[geometry]
    xml, qvel0 = F.scene_xml(scene, rep)
    mjm = mujoco.MjModel.from_xml_string(xml)
    surfaces = F.surfaces(mjm, scene) if rep == "surfaces" else None
    return SceneSource(name, mjm, surfaces, None, qvel0, caps=dict(F.CAPS[rep]), xml=xml)


def _fbf_available(scene):
    from viewer.scenes import fbf as F

    names = [F.stone_surface(i) for i in range(1, F.ARCH_STONES + 1)] if scene == "arch" else [F.SURFACES[scene]]
    return all(_surface_available(n) for n in names)


def _sharpa(name, n_hands, walls, geometry, seed):
    """`viewer.scenes.sharpa`'s hands, stacked from `seed`, as `geometry`, with their joint limits and friction loss."""
    from viewer.scenes import sharpa as S

    rep = GEOMETRIES[geometry]
    xml = S.scene_xml(n_hands, rep, walls, jlimit=True)
    mjm = mujoco.MjModel.from_xml_string(xml)
    q0 = S.initial_qpos(mjm, n_hands, seed)
    surfaces = S.surfaces(mjm, n_hands, walls) if rep == "surfaces" else None
    caps = dict(nconmax=4096 * n_hands, njmax=12288 * n_hands) if rep == "surfaces" else \
        dict(nconmax=512 * n_hands, njmax=2048 * n_hands)
    return SceneSource(name, mjm, surfaces, q0, caps=caps, xml=xml,
                       sampler=lambda n, seed: np.stack([S.initial_qpos(mjm, n_hands, seed + i) for i in range(n)]))


def _sharpa_armadillo(name, geometry):
    """`viewer.scenes.sharpa_armadillo`: the hand holding the armadillo at its home keyframe, with its control."""
    from viewer.scenes import sharpa_armadillo as SA

    rep = GEOMETRIES[geometry]
    xml = SA.scene_xml(rep)
    mjm = mujoco.MjModel.from_xml_string(xml)
    surfaces = SA.surfaces(mjm) if rep == "surfaces" else None
    caps = dict(nconmax=4096, njmax=12288) if rep == "surfaces" else dict(nconmax=512, njmax=2048)
    return SceneSource(name, mjm, surfaces, SA.KEY_QPOS, caps=caps, xml=xml, ctrl=SA.KEY_CTRL,
                       keyframes=[("home", SA.KEY_QPOS.copy(), np.zeros(mjm.nv))])


def _shadow(name, n_hands, walls, geometry, seed):
    """`viewer.scenes.shadow`'s hands, stacked from `seed`, as `geometry`."""
    from viewer.scenes import shadow as SH

    rep = GEOMETRIES[geometry]
    xml = SH.scene_xml(n_hands, rep, walls)
    mjm = mujoco.MjModel.from_xml_string(xml)
    q0 = SH.initial_qpos(mjm, n_hands, seed)
    surfaces = SH.surfaces(mjm, n_hands, walls) if rep == "surfaces" else None
    caps = dict(nconmax=4096 * n_hands, njmax=12288 * n_hands) if rep == "surfaces" else \
        dict(nconmax=512 * n_hands, njmax=2048 * n_hands)
    return SceneSource(name, mjm, surfaces, q0, caps=caps, xml=xml,
                       sampler=lambda n, seed: np.stack([SH.initial_qpos(mjm, n_hands, seed + i) for i in range(n)]))


def _shadow_grasp(name, geometry):
    """
    `viewer.scenes.shadow_kong`: the Shadow hand's grasp of the Kong toy, lifted 100 mm, replayed from its
    recording (`RecordedControls`, a row a 4 ms tick for 1.7 s, the last held after it): the wrist's and the
    fingers' targets, and the toy's weight from the lift on; as `geometry`.
    """
    from viewer.scenes import shadow_kong as SK

    rep = GEOMETRIES[geometry]
    xml = SK.scene_xml(rep=rep)
    mjm = mujoco.MjModel.from_xml_string(xml)
    q0, v0, ctrl, period = SK.lift(mjm)
    controls = RecordedControls(ctrl, period)
    surfaces = SK.surfaces(mjm) if rep == "surfaces" else None
    return SceneSource(name, mjm, surfaces, q0, v0, xml=xml, ctrl=controls(0.0), controls=controls)


FBF = {  # viewer name: (viewer.scenes.fbf scene, what it is, its polytope version)
    "incline": ("incline", "a box sliding down a slope of atan(1/2) at mu 0.45 (FBF)", "as MJCF boxes"),
    "backspin_ball": ("backspin", "a ball launched at 4 m/s with 200 rad/s of backspin, mu 0.5 (FBF)",
                      "as an MJCF sphere"),
    "painleve_box": ("painleve", "a tall box sliding at 4 m/s at the tipping threshold, mu 0.5 (FBF)",
                     "as an MJCF box"),
    "house_of_cards": ("cards", "a house of 40 cards in five levels, mu 0.8 (FBF)", "as MJCF boxes"),
    "masonry_arch": ("arch", "a masonry arch of 25 stones, the end ones welded, mu 0.8 (FBF)",
                     "as 25 convex polytopes (GJK/EPA)"),
}


def _builtin():
    """
    `(panel, by_name)`, each name -> (description, factory(seed)): the scenes the panel lists, in its
    order, and those that load by name only.
    """
    panel, by_name = {}, {}
    for name in ("incline", "painleve_box", "backspin_ball", "masonry_arch", "house_of_cards"):
        scene, what, poly = FBF[name]
        for g in GEOMETRIES:
            if g == "sdf" and not _fbf_available(scene):
                continue
            how = "as Lodos surfaces (SDF)" if g == "sdf" else poly
            panel[f"{name}_{g}"] = (f"{what}, {how}", lambda s, n=f"{name}_{g}", sc=scene, g=g: _fbf(n, sc, g))
    how = {"sdf": "as Lodos surfaces (SDF)", "polytope": "as 34-piece convex polytopes (GJK/EPA)"}
    for kind, what in (("cube", "cubes"), ("arm", "armadillos")):
        for n, many, layout in ((5, "five", ""), (20, "twenty", " in four stacks")):
            for walls, where in ((False, "on a floor"), (True, "in a walled box")):
                for g in GEOMETRIES:
                    if kind == "cube" and g == "sdf" and not _surface_available("cube_0.8"):
                        continue
                    name = f"{kind}{'_walled' if walls else ''}_{n}_{g}"
                    as_ = ("as MJCF boxes" if g == "polytope" else how[g]) if kind == "cube" else how[g]
                    panel[name] = (f"{many} {what} dropped{layout} {where}, {as_}",
                                   lambda s, name=name, kind=kind, n=n, walls=walls, g=g:
                                   _drop(name, "cube" if kind == "cube" else "armadillo", n, walls, g, s))
    from viewer.scenes import shadow as SH
    from viewer.scenes import sharpa as S

    hands = (("sharpa", "SHARPA", S.assets_present(), _sharpa, "as 87 convex hulls each (GJK/EPA)",
              "; joint limits and friction loss"),
             ("shadow", "Shadow", SH.assets_present(), _shadow, "as 70 convex hulls each (GJK/EPA)",
              "; joint limits and friction loss"))
    for prefix, who, present, factory, hulls, joints in hands:
        if not present:
            continue
        for n, many, layout in ((5, "five", ""), (20, "twenty", " in four stacks")):
            for walls, where in ((False, "on a floor"), (True, "in a walled box")):
                for g in GEOMETRIES:
                    name = f"{prefix}{'_walled' if walls else ''}_{n}_{g}"
                    as_ = "as Lodos surfaces (SDF)" if g == "sdf" else hulls
                    panel[name] = (f"{many} {who} hands dropped{layout} {where}, {as_}{joints}",
                                   lambda s, name=name, n=n, walls=walls, g=g, factory=factory:
                                   factory(name, n, walls, g, s))
    if S.assets_present():
        for g in GEOMETRIES:
            how_ = ("as Lodos surfaces (SDF), its own pairs as MuJoCo's" if g == "sdf"
                    else "as 87 + 34 convex hulls (GJK/EPA)")
            panel[f"sharpa_armadillo_{g}"] = (f"the SHARPA hand holding a 70 mm armadillo, palm up, {how_}",
                                              lambda s, g=g: _sharpa_armadillo(f"sharpa_armadillo_{g}", g))
    from viewer.scenes import shadow_kong as SK

    for g in GEOMETRIES:
        if SK.assets_present(rep=GEOMETRIES[g]):
            how_ = "as Lodos surfaces (SDF)" if g == "sdf" else "as 70 + 13 convex hulls (GJK/EPA)"
            panel[f"shadow_grasp_{g}"] = ("the Shadow hand lifting an 81 g Kong toy 100 mm in Dexonomy's four-finger "
                                          "grasp, replayed from its recording (1.7 s, then held), its wrist servoed; "
                                          f"mu 0.6, {how_}", lambda s, g=g: _shadow_grasp(f"shadow_grasp_{g}", g))
    from viewer.scenes import nist as NI

    if NI.franka_assets_present("m8"):
        panel["nist_threading_m8"] = ("Isaac Lab's NIST policy threading the M8 nut onto its bolt, replayed from its "
                                      "recording (6.2 s, the robot on its recorded path); mu 0.75",
                                      lambda s: _nist_threading("nist_threading_m8"))
        panel["nist_threading_m8_easy"] = ("the same from an easier start: the nut at rest 3.4 pitches onto the "
                                           "thread, the robot as recorded",
                                           lambda s: _nist_threading("nist_threading_m8_easy", NI.THREADING_EASY))
    for size in NI.SIZES:
        for nut in NI.NUTS[size]:
            if not NI.assets_present(size, nut):  # viewer/assets/nist, made by tools/mesh_processing/nist
                continue
            # the default nut is the scene's plain name; a denser one carries its vertex spacing
            name = f"nist_nutbolt_{size}" + ("" if nut == NI.default_nut(size) else "_" + nut.rsplit("_", 1)[1])
            panel[name] = (f"Isaac Lab's NIST {size.upper()} nut, vertices {nut.rsplit('_s', 1)[1]} mm apart, started "
                           f"on its bolt's thread at mu 0.01 (at 0.75 it self-locks); seed k: jittered start k",
                           lambda s, name=name, size=size, nut=nut: _nist_nutbolt(name, size, nut, s))
    for size in NI.SIZES:
        if NI.assets_present(size):
            panel[f"nist_driven_{size}"] = (f"the NIST {size.upper()} nut turned on at a turn a second, as the arm "
                                            f"would, at the demo's friction (0.75)",
                                            lambda s, size=size: _nist_driven(f"nist_driven_{size}", size))
    if NI.board_assets_present():
        panel["nist_board"] = ("Isaac Lab's NIST board, fixed, with its M8, M12 and M16 bolts, each nut started on "
                               "its thread at mu 0.01; seed k: jittered start k", lambda s: _nist_board("nist_board", s))

    by_name["arm_box_6"] = ("armadillos (surfaces + pieces) and cubes in a walled box: both pipelines",
                            lambda s: _mixed("arm_box_6", "mix6", s))
    by_name["arm_box_20"] = ("twenty of them in four stacks", lambda s: _mixed("arm_box_20", "mix20", s))
    for nut, mm in (("nut_s0.5", "05"), ("nut_s1.0", "1")):
        if not _nutbolt_available(nut):
            continue
        size = "0.5 mm" if mm == "05" else "1 mm"
        for spline, sdf in ((None, "tricubic SDF"), ("linear", "trilinear SDF")):
            name = f"nutbolt_{mm}" + ("_trilinear" if spline else "")
            by_name[name] = (f"Newton's nut-bolt example (M20 loose), the {size} nut, {sdf}; seed k: jittered start k",
                             lambda s, name=name, nut=nut, spline=spline: _nutbolt(name, nut, spline, s))
    if S.assets_present():
        for walls, where in ((False, "on a floor"), (True, "in a walled box")):
            for g in GEOMETRIES:
                name = f"sharpa{'_walled' if walls else ''}_2_{g}"
                as_ = "as Lodos surfaces (SDF)" if g == "sdf" else "as 87 convex hulls each (GJK/EPA)"
                by_name[name] = (f"two SHARPA hands dropped {where}, {as_}; joint limits and friction loss",
                                 lambda s, name=name, walls=walls, g=g: _sharpa(name, 2, walls, g, s))
    return panel, by_name


_PANEL, _BY_NAME = _builtin()
BUILTIN = {**_PANEL, **_BY_NAME}
PANEL = tuple(_PANEL)  # what the viewer's panel lists, in this order; the rest of `BUILTIN` loads by name


def builtin(name, seed=0):
    """The built-in scene `name` (a key of `BUILTIN`), its stacks drawn from `seed`."""
    if name not in BUILTIN:
        raise KeyError(f"no built-in scene {name!r}; there are {sorted(BUILTIN)}")
    return BUILTIN[name][1](int(seed))


#######################################################################################
# MJCF files
#######################################################################################

PREFIX = "lodos:"
CONTROLS = PREFIX + "controls"  # a recording of the controls (`RecordedControls`), not a geom's surface


def _texts(mjm):
    """The model's custom text fields, as `{name: data}`."""
    out = {}
    adr, size = np.asarray(mjm.text_adr), np.asarray(mjm.text_size)
    data = bytes(np.asarray(mjm.text_data))
    for i in range(int(mjm.ntext)):
        name = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_TEXT, i)
        out[name] = data[adr[i] : adr[i] + size[i]].rstrip(b"\x00").decode()
    return out


def _numeric(mjm, name, default):
    try:
        return float(mjm.numeric(name).data[0])
    except KeyError:
        return default


def _names_cone(path, main_dir=None):
    """
    Whether an MJCF file, or a file it includes, names the friction cone (`<option cone>`).
    An include's path is relative to the main file's directory, as MuJoCo reads it.
    """
    main_dir = Path(path).parent if main_dir is None else main_dir
    root = ET.parse(path).getroot()
    return any("cone" in o.attrib for o in root.iter("option")) or any(
        _names_cone(main_dir / i.get("file"), main_dir) for i in root.iter("include"))


def _surface(spec, base_dir):
    """A surface from an annotation's data: a shipped name or a bake's path."""
    if spec.endswith(".npz") or os.sep in spec:
        path = Path(spec)
        if not path.is_absolute():
            path = Path(base_dir) / path
        surf = sdf.load_surface_data(str(path))
        if not hasattr(surf, "bbox"):
            lo, hi = np.asarray(surf.v).min(0), np.asarray(surf.v).max(0)
            surf.bbox = np.array([[hi[0] if i & 4 else lo[0], hi[1] if i & 2 else lo[1], hi[2] if i & 1 else lo[2]]
                                  for i in range(8)], dtype=np.float64)
        surf.sph_c = np.zeros((0, 3))
        surf.sph_r = np.zeros(0)
        return surf
    return load_surface(spec)


def lodos_surfaces(mjm, base_dir="."):
    """
    The `lodos.sim.Surfaces` an MJCF model's `lodos:` annotations declare, or None.

    Surfaces of one shipped name are loaded once and shared, so instancing sees one
    mesh. Raises ValueError for an annotation naming a geom the model lacks, or
    `plane` on a geom that is not one.
    """
    texts = {k[len(PREFIX):]: v for k, v in _texts(mjm).items() if k.startswith(PREFIX) and k != CONTROLS}
    if not texts:
        return None
    cache, surfs, geoms, planes = {}, [], [], []
    for geom_name, spec in texts.items():
        g = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if g < 0:
            raise ValueError(f"{PREFIX}{geom_name}: the model has no geom named {geom_name!r}")
        spec = spec.strip()
        if spec == "plane":
            if int(mjm.geom_type[g]) != mujoco.mjtGeom.mjGEOM_PLANE:
                raise ValueError(f"{PREFIX}{geom_name}: 'plane' on a geom that is not a plane")
            surfs.append(sdf_types.PlaneSurface())
            planes.append(len(geoms))
        else:
            if spec not in cache:
                cache[spec] = _surface(spec, base_dir)
            surfs.append(cache[spec])
        geoms.append(g)

    body = np.asarray(mjm.geom_bodyid)[geoms]
    solid = [i for i in range(len(geoms)) if i not in planes]
    pairs = [[a, b] for a in solid for b in solid if a != b and body[a] != body[b]]
    pairs += [[a, p] for a in solid for p in planes]
    if not pairs:
        return None
    mu = _numeric(mjm, PREFIX + "mu", 0.8)
    params = lsim.default_pair_params(mu, len(pairs))
    params[:, 1] = _numeric(mjm, PREFIX + "stiffness", params[0, 1])
    params[:, 3] = _numeric(mjm, PREFIX + "dissipation", params[0, 3])
    return lsim.Surfaces(surfaces=surfs, geoms=geoms, pairs=pairs, params=params, mu=mu)


def keyframes(mjm):
    """`(name, qpos, qvel)` for each of the model's keyframes."""
    out = []
    for k in range(int(mjm.nkey)):
        name = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_KEY, k) or f"key {k}"
        out.append((name, np.array(mjm.key_qpos[k]), np.array(mjm.key_qvel[k])))
    return out


def from_mjcf(path):
    """An MJCF file as a `SceneSource`; see the module docstring for the annotations and the cone."""
    path = Path(path).expanduser().resolve()
    mjm = mujoco.MjModel.from_xml_path(str(path))
    if not _names_cone(path):
        mjm.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC  # every built-in scene's cone, not MuJoCo's default
    surfaces = lodos_surfaces(mjm, path.parent)
    keys = keyframes(mjm)
    q0, v0 = (keys[0][1], keys[0][2]) if keys else (None, None)
    src = SceneSource(path.stem, mjm, surfaces, q0, v0, path=str(path), keyframes=keys)
    rec = _texts(mjm).get(CONTROLS)
    return with_controls(src, RecordedControls.load(path.parent / rec.strip())) if rec else src


def export_mjcf(source, path, qpos=None, qvel=None, time=0.0):
    """
    Write a generated scene as an MJCF file the session can load back: its MJCF, its
    Lodos half as `lodos:` annotations, and the state given (default the initial
    one) as a keyframe. Asset paths stay absolute, so the file is for this machine.
    Recorded controls (`RecordedControls`) go beside it, `<file>.controls.npz`, from the
    tick of the state's `time` on: a state between two ticks is refused, since the file
    would replay that tick whole. Any other controls cannot be written, and are refused.
    """
    if source.xml is None:
        raise ValueError("only a scene generated in code has MJCF text to export")
    rec = source.controls
    if rec is not None and not isinstance(rec, RecordedControls):
        raise ValueError("only recorded controls (RecordedControls) can be exported with a scene")
    if rec is not None:
        k = rec.tick(time)
        if abs(float(time) - k * rec.period) > 0.01 * rec.period:
            raise ValueError(f"{time:g} s is between two of the recording's ticks ({rec.period:g} s): export a state at one")
    qpos = source.qpos0 if qpos is None else np.asarray(qpos)
    qvel = source.qvel0 if qvel is None else np.asarray(qvel)
    lines = []
    if rec is not None:
        out = Path(path).with_name(Path(path).stem + ".controls.npz")
        RecordedControls(rec.ctrl[k:], rec.period).save(out)
        lines.append(f'    <text name="{CONTROLS}" data="{out.name}"/>')
    if source.surfaces is not None:
        names = {}
        for s, g in zip(source.surfaces.surfaces, source.surfaces.geoms):
            gname = mujoco.mj_id2name(source.mjm, mujoco.mjtObj.mjOBJ_GEOM, int(g))
            if gname is None:
                raise ValueError(f"geom {g} carries a Lodos surface but has no name to annotate it by")
            names[gname] = "plane" if isinstance(s, sdf_types.PlaneSurface) else _shipped_name(s)
        lines += [f'    <text name="{PREFIX}{k}" data="{v}"/>' for k, v in names.items()]
        p = np.asarray(source.surfaces.params if source.surfaces.params is not None
                       else lsim.default_pair_params(source.surfaces.mu, 1))
        lines += [f'    <numeric name="{PREFIX}mu" data="{p[0, 0]:.6g}"/>',
                  f'    <numeric name="{PREFIX}stiffness" data="{p[0, 1]:.6g}"/>',
                  f'    <numeric name="{PREFIX}dissipation" data="{p[0, 3]:.6g}"/>']
    block = ""
    if lines:
        block += "  <custom>\n" + "\n".join(lines) + "\n  </custom>\n"
    fmt = lambda a: " ".join(f"{x:.9g}" for x in a)
    block += f'  <keyframe>\n    <key name="initial" qpos="{fmt(qpos)}" qvel="{fmt(qvel)}"/>\n  </keyframe>\n'
    xml = source.xml.rstrip()
    end = xml.rfind("</mujoco>")
    text = xml[:end] + block + xml[end:] + "\n"
    Path(path).write_text(text)
    return path


def _shipped_name(surf):
    """The shipped surface a loaded one is, by vertex count, for the export."""
    root = SURF_DIR
    for d in sorted(root.iterdir()):
        try:
            if len(load_surface(d.name).v) == len(surf.v):
                return d.name
        except Exception:  # noqa: BLE001 -- a directory that is not a bake
            continue
    raise ValueError("a surface that is not one of the shipped ones cannot be exported by name")


#######################################################################################
# Finding MJCF files
#######################################################################################

# Directories the search never descends into: version control, environments, caches.
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", ".scratch", ".cache", "site-packages",
             ".mypy_cache", ".pytest_cache", "build", "dist"}


def is_mjcf(path, head=4096):
    """Whether a file's first `head` bytes open a `<mujoco` element."""
    try:
        with open(path, "rb") as f:
            return b"<mujoco" in f.read(head)
    except OSError:
        return False


def find_mjcf(roots, max_depth=6, limit=20000):
    """
    Every MJCF file under `roots`, as sorted absolute paths.

    A walk with the usual directories pruned and a depth limit, then a check of each
    `.xml`'s first few kilobytes for `<mujoco`, so a home directory is a second's scan
    rather than a minute's, and an `.xml` that is not a model does not show up.
    """
    found = set()
    for root in roots:
        root = Path(root).expanduser()
        if not root.is_dir():
            continue
        base = len(root.parts)
        for dirpath, dirnames, filenames in os.walk(root):
            depth = len(Path(dirpath).parts) - base
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
            if depth >= max_depth:
                dirnames[:] = []
            for fn in filenames:
                if fn.lower().endswith(".xml"):
                    p = os.path.join(dirpath, fn)
                    if is_mjcf(p):
                        found.add(os.path.abspath(p))
                        if len(found) >= limit:
                            return sorted(found)
    return sorted(found)


def search(paths, query, limit=200):
    """
    The paths matching `query`, best first: every whitespace-separated term must
    appear in the path (case-insensitive), and a term in the file name ranks above one
    only in the directories.
    """
    terms = query.lower().split()
    if not terms:
        return list(paths[:limit])
    scored = []
    for p in paths:
        low = p.lower()
        if all(t in low for t in terms):
            name = os.path.basename(low)
            scored.append((-sum(t in name for t in terms), len(p), p))
    scored.sort()
    return [p for _, _, p in scored[:limit]]
