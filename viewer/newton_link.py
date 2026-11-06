"""
The live viewer's link to Newton's own simulators, for comparing them with Lodos.

Two sims, `mjwarp` (Newton's `SolverMuJoCo`, MuJoCo Warp) and `kamino` (`SolverKamino`),
run in a process of their own, in `viewer/newton/`'s environment: a
pinned Newton release with its own upstream MuJoCo Warp, where nothing of Lodos is imported
-- Newton pins MuJoCo 3.15 and its own MuJoCo Warp, where Lodos pins MuJoCo 3.14. Both
take their contacts from Newton's `CollisionPipeline`: a scene's primitives and convex hulls
as Newton collides them, and each Lodos surface as its triangle mesh with Newton's own mesh
SDF, at the voxel size of the surface's bake (`viewer/newton/newton_scene.py`).

This module is the Lodos side: a scene as a *package* -- its MJCF, its surfaces' meshes,
its time step and control -- and `NewtonRunner`, which stands in for `session.Runner`: it
starts the Newton process, hands it states in MuJoCo's layout, steps it and turns what comes
back into the viewer's frames.
"""

from __future__ import annotations

import copy
import math
import os
import pickle
import struct
import subprocess
from pathlib import Path

import mujoco
import numpy as np

from viewer import pairs as VP

ROOT = Path(__file__).resolve().parents[1]
NEWTON_DIR = ROOT / "viewer" / "newton"
SIMS = ("mjwarp", "kamino")


def newton_python():
    """The Newton environment's interpreter (`$LODOS_NEWTON_PYTHON`, else `viewer/newton/.venv`), or None."""
    p = Path(os.environ.get("LODOS_NEWTON_PYTHON") or NEWTON_DIR / ".venv" / "bin" / "python")
    return p if p.exists() else None


def available():
    """Whether the Newton environment is installed (`cd viewer/newton && uv sync`)."""
    return newton_python() is not None


#######################################################################################
# A scene as a package
#######################################################################################


def _newton_contact(surfaces):
    """
    Each Lodos surface's contact parameters for Newton, which gives each shape one set and a contact
    its two shapes' combined (MuJoCo's rules): the larger friction, the solmix-weighted mean of the
    solref and solimp. The friction: each surface the smallest of its pairs', so that every pair gets
    its own -- the nut-bolt scenes' thread (0.01) and their floor (0.5) both, where giving each surface
    its largest would make the thread the floor's and lock it; a pair table that one friction a surface
    cannot express is refused. The solref and solimp: a surface's pairs', where they all have the same;
    where they differ (the threading's), none, and the scene's MJCF geom's stand.
    """
    from lodos.cd import params as cd_params

    n = len(surfaces.pairs)
    p = (cd_params.make_default_contact_params(surfaces.mu, n) if surfaces.params is None
         else np.asarray(surfaces.params, np.float32).reshape(n, 13))
    out = []
    for k in range(len(surfaces.surfaces)):
        rows = p[[i for i, pair in enumerate(surfaces.pairs) if k in pair]]
        c = dict(mu=float(rows[:, 0].min()) if len(rows) else float(surfaces.mu))
        if len(rows) and (rows[:, 5:12] == rows[0, 5:12]).all():
            c.update(solref=tuple(map(float, rows[0, 5:7])), solimp=tuple(map(float, rows[0, 7:12])))
        out.append(c)
    for i, (a, b) in enumerate(surfaces.pairs):
        if not np.isclose(max(out[a]["mu"], out[b]["mu"]), float(p[i, 0])):
            raise ValueError(f"Newton takes one friction a shape and a contact the larger of its two: surfaces {a} "
                             f"and {b} cannot have {float(p[i, 0]):g} together with their other pairs'")
    return out


def scene_package(source, sim="mjwarp"):
    """
    What Newton's `sim` needs of a `scenes.SceneSource`: its MJCF (`newton_xml`, that sim's, where it
    has one); each Lodos surface (not a plane) as its body's name, the geom it is keyed to, a triangle
    mesh in the body's frame and its SDF's settings, and its contact parameters (`_newton_contact`) --
    its own mesh and its bake's voxel size, or the shape the scene names for Newton
    (`SceneSource.newton_shapes`), or nothing where Newton collides the MJCF's geom; each Lodos plane's
    contact parameters, for its geom (`planes`); the time step; the constant control, if any (the first
    row of `newton_controls` where the scene has them); the scene's start (`qpos0`, `qvel0`), which
    Newton's contacts are probed at (`newton_scene.Sim.probe_contacts`): a pile's MJCF has its bodies on
    top of each other.
    """
    from lodos.col.sdf import types as sdf_types

    mjm = source.mjm
    xml = source.newton_xml.get(sim) if isinstance(source.newton_xml, dict) else source.newton_xml
    ctrl = source.newton_controls(0.0, source.qpos0) if source.newton_controls is not None else source.ctrl
    pkg = dict(name=source.name, xml=xml or source.xml, dt=float(mjm.opt.timestep), sdf_shapes=[], planes={},
               ctrl=None if ctrl is None else np.asarray(ctrl, np.float64),
               qpos0=np.asarray(source.qpos0, np.float64), qvel0=np.asarray(source.qvel0, np.float64))
    surf, own = source.surfaces, source.newton_shapes or {}
    if surf is not None:
        contact = _newton_contact(surf)
        for k, (s, g) in enumerate(zip(surf.surfaces, surf.geoms)):
            if isinstance(s, sdf_types.PlaneSurface):
                pkg["planes"][int(g)] = contact[k]
                continue
            if int(g) in own and own[int(g)] is None:
                continue  # Newton collides the MJCF's geom
            mesh = own.get(int(g)) or dict(vertices=np.asarray(s.v, np.float32), faces=np.asarray(s.f, np.int32),
                                           voxel=float(np.min(np.asarray(s.h))))
            body = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_BODY, int(mjm.geom_bodyid[int(g)]))
            pkg["sdf_shapes"].append(dict(body=body, geom_id=int(g), **contact[k], **mesh))
    return pkg


#######################################################################################
# The Newton process
#######################################################################################


class NewtonProcess:
    """`viewer/newton/runner.py` in the Newton environment: length-prefixed pickles both ways."""

    def __init__(self):
        py = newton_python()
        if py is None:
            raise RuntimeError("Newton's environment is not installed: cd viewer/newton && uv sync")
        env = dict(os.environ)
        env.pop("VIRTUAL_ENV", None)
        env.pop("PYTHONPATH", None)
        self.proc = subprocess.Popen([str(py), "-u", str(NEWTON_DIR / "runner.py")], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, cwd=str(NEWTON_DIR), env=env)

    def call(self, cmd, **kw):
        data = pickle.dumps(dict(cmd=cmd, **kw), protocol=5)
        self.proc.stdin.write(struct.pack("<Q", len(data)) + data)
        self.proc.stdin.flush()
        head = self.proc.stdout.read(8)
        if len(head) < 8:
            raise RuntimeError(f"Newton's process ended (exit {self.proc.poll()}); see its output above")
        (n,) = struct.unpack("<Q", head)
        out = pickle.loads(self.proc.stdout.read(n))
        if not out.get("ok"):
            raise RuntimeError(f"Newton: {out.get('error')}")
        return out

    def close(self):
        if self.proc.poll() is None:
            try:
                self.call("quit")
            except Exception:  # noqa: BLE001
                pass
            self.proc.kill()

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass


#######################################################################################
# The viewer's runner
#######################################################################################


def length_scale(mjm):
    """`viewer/newton/newton_scene.length_scale`, which the default gap is 1 % of."""
    spans = [2.0 * float(mjm.geom_rbound[mjm.geom_bodyid == k].max()) for k in range(1, mjm.nbody)
             if (mjm.geom_bodyid == k).any() and int(mjm.body_rootid[k]) > 0 and mjm.body_jntnum[mjm.body_rootid[k]] > 0]
    return float(np.median(spans)) if spans else 1.0


# Kamino's options in the panel, at Newton 1.7's defaults (`SolverKamino.Config`), which `newton_scene` reports
# as `solver_defaults`: PADMM's tolerances on the primal, dual and complementarity residuals, its iteration cap,
# the ALM penalty's initial value, its acceleration and its two warm starts; the dynamics' preconditioning; and
# `gamma`, the contacts' Baumgarte gain. Under Kamino the timestep is an option too, the MJCF's by default.
KAMINO_DEFAULTS = dict(primal_tolerance=1e-6, dual_tolerance=1e-6, compl_tolerance=1e-6, max_iterations=200,
                       rho_0=1.0, use_acceleration=1, warmstart_mode="containers",
                       contact_warmstart_method="key_and_position", preconditioning=1, gamma=0.01)
WARMSTART_MODES = ("none", "internal", "containers")
CONTACT_WARMSTART_METHODS = ("key_and_position", "geom_pair_net_force", "geom_pair_net_wrench",
                             "key_and_position_with_net_force_backup", "key_and_position_with_net_wrench_backup")


# MuJoCo Warp's choices in the panel, by the names Newton's `SolverMuJoCo` takes: the solver, the
# integrator (the two Lodos takes too) and the friction cone.
MJWARP_SOLVERS = {int(mujoco.mjtSolver.mjSOL_NEWTON): "newton", int(mujoco.mjtSolver.mjSOL_CG): "cg"}
MJWARP_INTEGRATORS = {int(mujoco.mjtIntegrator.mjINT_EULER): "euler",
                      int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST): "implicitfast"}
MJWARP_CONES = {int(mujoco.mjtCone.mjCONE_ELLIPTIC): "elliptic", int(mujoco.mjtCone.mjCONE_PYRAMIDAL): "pyramidal"}
MJWARP_MIN_TOLERANCE = 1e-6  # MuJoCo Warp's put_model raises a smaller opt.tolerance to this


def default_options(source, sim):
    """
    A Newton sim's panel options: the contact gap (m), for the shapes without a gap of their own in the
    MJCF (`newton_scene.world_builder`); MuJoCo Warp's solver options, the MJCF's, and its timestep;
    Kamino's timestep and solver options (`KAMINO_DEFAULTS`). The scene's own values
    (`SceneSource.newton_options`) over all of these.
    """
    opt = dict(gap=0.01 * length_scale(source.mjm))
    mo = source.mjm.opt
    if sim == "mjwarp":
        # MuJoCo Warp's tolerance is at least 1e-6 (float32: its put_model raises a smaller one)
        opt.update(iterations=int(mo.iterations), tolerance=max(float(mo.tolerance), MJWARP_MIN_TOLERANCE),
                   ls_iterations=int(mo.ls_iterations),
                   ls_tolerance=float(mo.ls_tolerance), solver=MJWARP_SOLVERS[int(mo.solver)],
                   integrator=MJWARP_INTEGRATORS[int(mo.integrator)], cone=MJWARP_CONES[int(mo.cone)],
                   timestep=float(mo.timestep))
    if sim == "kamino":
        opt.update(KAMINO_DEFAULTS, timestep=float(source.mjm.opt.timestep))
    opt.update({k: v for k, v in source.newton_options.items() if k in opt})
    return opt


OPTIONS = {
    "mjwarp": (("gap", float, 0.0, 1.0), ("timestep", float, 1e-6, 0.1),
               ("integrator", "choice", tuple(MJWARP_INTEGRATORS.values()), None),
               ("solver", "choice", tuple(MJWARP_SOLVERS.values()), None),
               ("cone", "choice", tuple(MJWARP_CONES.values()), None), ("iterations", int, 1, 1000),
               ("tolerance", float, MJWARP_MIN_TOLERANCE, 1e-2), ("ls_iterations", int, 1, 1000),
               ("ls_tolerance", float, 1e-12, 1.0)),
    "kamino": (("gap", float, 0.0, 1.0), ("timestep", float, 1e-6, 0.1),
               ("primal_tolerance", float, 1e-12, 1.0), ("dual_tolerance", float, 1e-12, 1.0),
               ("compl_tolerance", float, 1e-12, 1.0), ("max_iterations", int, 1, 10000),
               ("rho_0", float, 1e-6, 1e6), ("use_acceleration", "choice", (0, 1), None),
               ("warmstart_mode", "choice", WARMSTART_MODES, None),
               ("contact_warmstart_method", "choice", CONTACT_WARMSTART_METHODS, None),
               ("preconditioning", "choice", (0, 1), None), ("gamma", float, 0.0, 1.0)),
}
INFO = {
    "mjwarp": "Newton 1.7's SolverMuJoCo (upstream MuJoCo Warp) on Newton's collision pipeline; nothing of Lodos",
    "kamino": "Newton 1.7's SolverKamino (proximal ADMM) on Newton's collision pipeline; nothing of Lodos",
}


class NewtonRunner:
    """
    `session.Runner`'s part for one of Newton's sims: the scene in the Newton process,
    one world, its frames built here from the state it returns -- poses and velocities by
    MuJoCo's own kinematics on the viewer's model, contacts as Newton reports them. A
    scene's controls (`SceneSource.controls`) play as under Lodos: each tick's row goes to
    the Newton process with the step that starts it (`_control`).

    Its capacities are Newton's (`newton_scene.Caps`): contacts and mesh triangle pairs a world, and under MuJoCo Warp
    its constraint rows; it keeps the largest demand on each (`seen`), tunes them to it and grows the one that
    overflows, as `session.Runner` does Lodos's. Its pairs are Newton's shape pairs (`pair_tables`), their contact
    parameters set on the running model: MuJoCo Warp's friction, solref and solimp, Kamino's friction.
    """

    def __init__(self, source, cfg, scene=None, kept=None):
        from types import SimpleNamespace

        from viewer import session as S
        from viewer.collect import ancestor_matrix
        from viewer.geometry import make_live_scene

        self.source, self.sim_name = source, cfg["sim"]
        self.key = S.runner_key(cfg)
        self.caps, self.defaults = {}, {}
        self.applied, self.owner, self.sim, self.graphs = None, None, None, None
        self.mjm = source.mjm
        self.scene = scene if scene is not None else make_live_scene(
            SimpleNamespace(mjm=source.mjm, surfaces=source.surfaces, mcol=True if source.surfaces is not None else None))
        self.data = mujoco.MjData(self.mjm)
        self.A = ancestor_matrix(self.mjm)
        self.rootid = np.asarray(self.mjm.body_rootid, int)
        self.proc = NewtonProcess()
        self.pkg = scene_package(source, self.sim_name)
        opts = {k: cfg["options"][k] for k, *_ in OPTIONS[self.sim_name]}
        # The largest demand on each capacity over everything simulated, its builds at other capacities included,
        # and how many forwards that was; a freed runner's, from `kept` (`session.Session.free_runners`)
        self.seen, self.steps_seen, caps = dict(contacts=0, triangle_pairs=0, rows=0), 0, None
        if kept is not None:
            self.seen, self.steps_seen = dict(kept["seen"]), int(kept["steps_seen"])
            caps = self._newton_caps(kept["caps"])
        self._tables, self._over = None, set()
        self.info = self.proc.call("build", pkg=self.pkg, solver=self.sim_name, options=opts, caps=caps)
        self._built()
        self._opts = opts
        self._state = S.State(np.asarray(source.qpos0, np.float64).copy(), np.asarray(source.qvel0, np.float64).copy(),
                              0.0)
        self.controls_playing = True  # the session's (`Session.play_controls`): off, the drives keep their targets
        self.controls_due = True  # a scene's controls before the next step, whatever its time
        self.last_ms = float("nan")

    def _built(self):
        """After a build in the Newton process: its capacities, and its shapes at the parameters built with."""
        self.caps = {k: int(v) for k, v in self.info["caps"].items()}
        self._pairs_sent = None

    @staticmethod
    def _newton_caps(caps):
        """`newton_scene.Caps`, as a dict, from the viewer's names of them: at one world a total is a world's."""
        out = dict(contacts_per_world=int(caps["contacts"]), triangle_pairs_per_world=int(caps["triangle_pairs"]))
        if "rows" in caps:
            out["njmax"] = int(caps["rows"])
        return out

    def apply(self, cfg):
        """Its options, by a rebuild in the Newton process at the same state; the pairs' parameters (`_send_pairs`)."""
        opts = {k: cfg["options"][k] for k, *_ in OPTIONS[self.sim_name]}
        if opts != self._opts:
            self.info = self.proc.call("options", options=opts)
            self._built()
            self._opts = opts
            self.controls_due = True  # the rebuild took the scene's constant control
        self._send_pairs(cfg)
        self.applied = copy.deepcopy(cfg)

    # -------------------------------------------------------------------------------
    # The pairs: Newton's shape pairs

    @property
    def pair_tables(self):
        """`{"newton": viewer.pairs.PairTable}`, Newton's shape pairs, made on first use; {} where it collides none."""
        if self._tables is None:
            t = VP.PairTable.newton(self.proc.call("pairs"))
            self._tables = {"newton": t} if len(t) else {}
        return self._tables

    @property
    def pair_counts(self):
        return {k: len(t) for k, t in self.pair_tables.items()}

    def _send_pairs(self, cfg):
        """
        The configuration's overrides of Newton's pairs on its shapes, each shape the `mode` of what is set on its
        pairs (`PairTable.geom_arrays`), on the running model -- Kamino's friction alone; nothing while a model as
        built has none.
        """
        alone = self.sim_name == "kamino"  # friction alone
        snap = copy.deepcopy((cfg["pairs"]["newton"], cfg["pairs"]["mode"]))
        if snap == self._pairs_sent:
            return
        if self._pairs_sent is not None or VP.has_overrides(snap[0], alone):
            g = self.pair_tables["newton"].geom_arrays(cfg["pairs"], alone)
            self.proc.call("shape_params", mu=g["friction"][:, 0], solref=g["solref"], solimp=g["solimp"])
        self._pairs_sent = snap

    # -------------------------------------------------------------------------------
    # The capacities

    def capacities(self):
        """
        `[(name, allocated, peak)]`, as `session.Runner.capacities`: Newton's contacts and mesh triangle pairs a world
        (the pairs Newton's million at least), and under MuJoCo Warp its constraint rows a world.
        """
        names = dict(contacts="Newton contacts", triangle_pairs="Newton triangle pairs (1M at least)",
                     rows="MuJoCo Warp rows (njmax)")
        return [(names[k], int(v), int(self.seen[k])) for k, v in self.caps.items()]

    def fractions(self):
        return {}  # Lodos's arenas: none here

    def kept(self):
        """What a freed runner hands its configuration's next build: its capacities and the demand it has seen."""
        return dict(caps=dict(self.caps), seen=dict(self.seen), steps_seen=self.steps_seen)

    def _rebuild(self, caps):
        """Rebuilt in the Newton process at `caps` (the viewer's names), at the same state and parameters."""
        self.info = self.proc.call("rebuild", caps=self._newton_caps(caps))
        self._built()
        self.set_state(self._state)
        if self.applied is not None:
            self._send_pairs(self.applied)

    def tune_capacities(self, overhead):
        """Every capacity at its demand in `seen` times 1 + `overhead`, rebuilt at the same state; returns them."""
        f = 1.0 + float(overhead)
        caps = {k: max(1, math.ceil(f * self.seen[k])) for k in self.caps}
        if "rows" in caps:
            caps["rows"] = max(16, caps["rows"])  # as a tuning file's njmax (`lodos.utils.calibrate.settings`)
        self._rebuild(caps)
        return dict(self.caps)

    def set_state(self, st, cold=True):
        from viewer import session as S

        self.proc.call("set_state", qpos=np.asarray(st.qpos, np.float64), qvel=np.asarray(st.qvel, np.float64),
                       time=float(st.time))
        self._state = S.State(np.asarray(st.qpos, np.float64).copy(), np.asarray(st.qvel, np.float64).copy(),
                              float(st.time))
        self.controls_due = True

    def state(self):
        from viewer import session as S

        st = self._state
        return S.State(st.qpos.copy(), st.qvel.copy(), st.time)

    def forward(self):
        r = self.proc.call("forward")
        self._see(r)  # this state's overflows, once the capacities have grown for it
        return self._frame(r)

    def _see(self, r):
        """A forward's or a step's answer: which capacities it overflowed, and its demand on each, kept in `seen`."""
        self._over = set(r["overflow"])
        for k, v in r["counts"].items():
            self.seen[k] = max(self.seen.get(k, 0), int(v))
        self.steps_seen += 1

    def _control(self):
        """
        The row of a scene's controls (`SceneSource.newton_controls` where it has them, else
        `controls`) to write before the next step, as `session.Runner._control` writes Lodos's: at a
        step that starts one of their ticks, and the first from a state just loaded; else None, and the
        Newton process keeps the targets it has.
        """
        c = self.source.newton_controls if self.source.newton_controls is not None else self.source.controls
        if c is None or not self.controls_playing:
            return None
        t = float(self._state.time)
        if not self.controls_due and abs(t - round(t / c.period) * c.period) >= 0.5 * float(self.info["dt"]):
            return None
        self.controls_due = False
        return np.asarray(c(t, np.asarray(self._state.qpos, np.float64)), dtype=np.float64)

    def step(self, record):
        r = self.proc.call("step", record=bool(record), ctrl=self._control())
        frame = self._frame(r["pre"], r.get("solver")) if record else None
        from viewer import session as S

        self._state = S.State(np.asarray(r["qpos"], np.float64), np.asarray(r["qvel"], np.float64), float(r["time"]))
        self._see(r)
        self.last_ms = float(r["ms"])
        return frame

    def overflow(self):
        """Which capacities the last forward or step overflowed: `contacts`, `triangle_pairs`, `rows`."""
        return set(self._over)

    def grow(self, which):
        """Raise the named capacities -- to 1.5 times their demand, at least double -- rebuilt at the same state."""
        from viewer import session as S

        caps = dict(self.caps)
        for k in which:
            caps[k] = max(2 * int(caps[k]), math.ceil(S.GROW * self.seen[k]))
        self._rebuild(caps)

    def _frame(self, r, status=None):
        """
        A `LiveFrame` of the state `r` holds, by MuJoCo's kinematics, with Newton's contacts; `status`, the
        solve of the step taken from that state (`newton_scene.Sim.solver_status`), gives its iterations
        (`niter`), whether it converged, and under Kamino its primal, dual and complementarity residuals.
        """
        from viewer.collect import SRC_PRIM, LiveFrame, energy_reference
        from viewer.geometry import geom_xforms

        m, d = self.mjm, self.data
        qpos, qvel = np.asarray(r["qpos"], np.float64), np.asarray(r["qvel"], np.float64)
        d.qpos[:], d.qvel[:] = qpos, qvel
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        xpos, xquat = d.xpos.copy(), d.xquat.copy()
        cdof, com = d.cdof.copy(), d.subtree_com.copy()
        w = self.A @ (cdof[:, :3] * qvel[:, None])
        vc = self.A @ (cdof[:, 3:] * qvel[:, None])
        c = com[self.rootid]
        v0 = vc + np.cross(w, xpos - c)
        con = r["contacts"]
        p, n, dist, f = (np.asarray(con[k], np.float64) for k in ("p", "n", "dist", "f"))
        bodies = np.asarray(con["bodies"], int).reshape(-1, 2)
        v = (vc[bodies[:, 1]] + np.cross(w[bodies[:, 1]], p - c[bodies[:, 1]])
             - vc[bodies[:, 0]] - np.cross(w[bodies[:, 0]], p - c[bodies[:, 0]])) if len(p) else np.zeros((0, 3))
        src = np.full(len(p), SRC_PRIM)
        sv = status or {}
        stats = dict(ncon_lodos=0, ncon_prim=int(len(p)), depth=float(dist.min()) if len(dist) else 0.0,
                     ke=float(energy_reference(m, qpos, qvel)), niter=float(sv.get("iterations", float("nan"))),
                     ms=self.last_ms)
        if status is not None:
            stats["converged"] = bool(sv["converged"])
            stats.update({k: float(sv[k]) for k in ("r_p", "r_d", "r_c") if k in sv})  # Kamino's residuals
        body_xform = np.concatenate([xpos, xquat[:, 1:], xquat[:, :1]], axis=1)
        return LiveFrame(
            time=float(r["time"]),
            con_p=p.astype(np.float32), con_n=n.astype(np.float32), con_v=v.astype(np.float32),
            con_f=f.astype(np.float32), con_depth=dist.astype(np.float32), con_seg=np.full(len(p), -1, np.int32),
            geom_xform=geom_xforms(self.scene, xpos, xquat).astype(np.float32), body_xform=body_xform.astype(np.float32),
            jnt_anchor=d.xanchor.astype(np.float32), jnt_axis=d.xaxis.astype(np.float32),
            qpos=qpos, qvel=qvel, con_src=src.astype(np.int8), con_body=bodies.astype(np.int32),
            body_v=v0.astype(np.float32), body_w=w.astype(np.float32), warm=None, stats=stats,
        )

    def close(self):
        self.proc.close()
