"""
State sets: the starting configurations every tuning tool runs from.

Every capacity in a `lodos.sim.Sim` is sized to a demand, and the demand is a property of
the states the scene visits: a capacity fitted to one rollout fits that rollout. A state
set is the list of states the calibration, the timings, the tile sweeps and the accuracy
checks all start from -- the same list for every one of them, so that the capacities, the
timings and the accuracy they report are about the same thing.

A state is `qpos`, `qvel`, the control it holds and the actuators' activations; a set adds
a weight per state, an optional control sequence per state, where it came from, and the
model it belongs to:

    S = states.from_keyframes(mjm)                 # the MJCF's keyframes
    S = states.from_qpos(mjm, qpos, source="...")  # any (N, nq) array
    S.save("scene.states.npz")
    S = states.load("scene.states.npz", mjm)       # refuses a set made for another model

## Layouts: which world starts from which state

Lodos's collision capacities are *batch totals* -- fractions of `B` times what one world
could need, shared by the batch -- while MJWarp's `njmax` is *per world*. So the batch total a calibration sees depends on how the `B` worlds of a batch
are assigned to states, and a tuning file is valid for the assignment it was measured
under:

    independent   each world its own state, the batch drawing `B` of them (RL-style: many
                  independent episodes). A batch's totals are about `B` times the *mean*
                  load, and the larger the batch the closer to it.
    shared        every world of a batch from one state, each with its own controls
                  (MPC-style: one plant, many candidate futures). A batch's totals are
                  about `B` times *that state's* load, so the heaviest state sets them.
    explicit      an index array, one state per world.

## Controls during a rollout

A rollout runs `horizon` steps from each start. What the actuators do meanwhile:

    hold          each world keeps its state's control (the default)
    none          zero control
    random        a smooth random walk inside `ctrlrange`, mean-reverting to the held
                  control: `Controls.random(sigma, theta)`
    sequence      the set's own `(N, H, nu)` sequences, world `b` playing its state's

`rollout` drives a `Sim` through all of it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

FORMAT = "lodos-states/1"  # the format every saved state set names; `load` refuses any other
LAYOUTS = ("independent", "shared", "explicit")  # the ways `batches` assigns a batch's worlds to states


def _sha(*arrays):
    """
    A 16-hex-digit SHA-256 of `arrays`: each one's dtype, shape and bytes.

    Called by `model_signature`, `StateSet.digest`, `lodos.utils.tuning.surface_fingerprint` and 2 others.
    In: arrays, numpy arrays or anything `np.ascontiguousarray` takes. Out: str.
    """
    h = hashlib.sha256()  # hashlib's SHA-256 state
    for a in arrays:  # a: an array, made contiguous below
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode() + str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:16]


def model_signature(mjm: mujoco.MjModel) -> dict:
    """
    What a state must agree with to belong to `mjm`: the sizes of `qpos`, `qvel`, `ctrl`
    and `act`, and a hash of the kinematic tree and the actuators' transmissions. Geometry,
    masses and gains are not in it -- a state stays a valid state of a scene whose meshes
    were re-baked (`lodos.utils.tuning.scene_fingerprint` is the stricter check).

    Called by `StateSet.check`, `from_qpos`, `lodos.utils.tuning.scene_fingerprint` and 1 other.
    In: mjm mujoco.MjModel. Out: dict: "nq", "nv", "nu" and "na" int, "structure" str (16 hex digits).
    """
    structure = _sha(  # str: a hash of the kinematic tree and the transmissions
        np.asarray(mjm.body_parentid), np.asarray(mjm.jnt_type), np.asarray(mjm.jnt_bodyid),
        np.asarray(mjm.actuator_trntype), np.asarray(mjm.actuator_trnid),
    )
    return dict(nq=int(mjm.nq), nv=int(mjm.nv), nu=int(mjm.nu), na=int(mjm.na), structure=structure)


@dataclasses.dataclass
class StateSet:
    """
    `N` states of one model.

    qpos, qvel, ctrl, act: `(N, nq)`, `(N, nv)`, `(N, nu)`, `(N, na)`.
    weights: `(N,)`, how often `batches` draws each state under the independent layout.
    controls: `(N, H, nu)` control sequences, one per state, or None.
    labels: a short name per state ("keyframe home", "scatter 12 t=0.40", ...).
    signature: `model_signature` of the model the states belong to.
    source: where the set came from -- a dict that is saved with it.

    Built by `load`, `from_qpos`, `from_keyframes` and 2 others; `subset` copies one.
    """

    qpos: np.ndarray  # (N, nq) float64
    qvel: np.ndarray  # (N, nv) float64
    ctrl: np.ndarray  # (N, nu) float64
    act: np.ndarray  # (N, na) float64
    weights: np.ndarray  # (N,) float64
    signature: dict  # `model_signature` of the model
    labels: list = dataclasses.field(default_factory=list)  # N str
    controls: np.ndarray | None = None  # (N, H, nu) float64, or None
    source: dict = dataclasses.field(default_factory=dict)  # where the set came from

    def __post_init__(self):
        """
        Makes every array float64 of its shape, fills in the labels, and checks `qpos` against the signature.

        Called by the dataclass's `__init__`, once the fields are set.
        Out: qpos (N, nq), qvel (N, nv), ctrl (N, nu), act (N, na), weights (N,) and controls (N, H, nu), float64.
        """
        n = len(self.qpos)  # int: N
        self.qpos = np.asarray(self.qpos, dtype=np.float64).reshape(n, -1)
        for name, width in (("qvel", "nv"), ("ctrl", "nu"), ("act", "na")):  # str, str: a field, its width's key
            setattr(self, name, np.asarray(getattr(self, name), dtype=np.float64).reshape(n, self.signature[width]))
        self.weights = np.asarray(self.weights, dtype=np.float64).reshape(n)
        if not self.labels:
            self.labels = [f"state {i}" for i in range(n)]
        if self.controls is not None:
            self.controls = np.asarray(self.controls, dtype=np.float64).reshape(n, -1, self.signature["nu"])
        if self.qpos.shape[1] != self.signature["nq"]:
            raise ValueError(f"qpos has {self.qpos.shape[1]} columns, the model nq={self.signature['nq']}")

    def __len__(self):
        """
        `N`, the number of states.

        Called through `len()`, by `lodos.utils.calibrate.measure`, `concat`, `viewer.performance.Performance.ui` and
        12 others. Out: int.
        """
        return len(self.qpos)

    def subset(self, idx):
        """
        The states `idx`, as a new set; the source is copied.

        Called by `tools.performance_tuning.explore.main` and `tools.performance_tuning.tune.make_states`.
        In: idx, k indices (any int array-like). Out: StateSet of k states.
        """
        idx = np.asarray(idx, dtype=np.int64).reshape(-1)
        return dataclasses.replace(
            self, qpos=self.qpos[idx], qvel=self.qvel[idx], ctrl=self.ctrl[idx], act=self.act[idx],
            weights=self.weights[idx], labels=[self.labels[i] for i in idx],
            controls=None if self.controls is None else self.controls[idx], source=dict(self.source),
        )

    def check(self, mjm):
        """
        Raises unless the set belongs to `mjm`.

        Called by `load`. In: mjm mujoco.MjModel.
        """
        theirs = model_signature(mjm)  # dict: mjm's signature
        if theirs != self.signature:
            raise ValueError(f"this state set is for another model: {self.signature} against {theirs}")

    def save(self, path):
        """
        An `.npz` with the arrays and a JSON header (signature, source, labels).

        Called by `tools.performance_tuning.explore.main`, `tools.performance_tuning.tune.make_states` and
        `viewer.performance.Performance.save_states`.
        In: path str or Path. Out: Path, the file written.
        """
        path = Path(path)
        if path.suffix != ".npz":  # numpy appends it, so the file written is that path
            path = path.with_name(path.name + ".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = dict(format=FORMAT, signature=self.signature, source=self.source, labels=self.labels)  # dict: the header
        # dict: field -> its (N, ...) float64 array
        arrays = dict(qpos=self.qpos, qvel=self.qvel, ctrl=self.ctrl, act=self.act, weights=self.weights)
        if self.controls is not None:
            arrays["controls"] = self.controls
        np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)
        return path

    def digest(self):
        """
        A hash of the states themselves, which a tuning file records as its provenance.

        Called by `tools.performance_tuning.calibrate.main`, `tools.performance_tuning.explore.main` and
        `tools.performance_tuning.tune.main`. Out: str, 16 hex digits.
        """
        parts = [self.qpos, self.qvel, self.ctrl, self.act, self.weights]  # list of the fields' arrays, float64
        if self.controls is not None:
            parts.append(self.controls)
        return _sha(*parts)


def load(path, mjm=None) -> StateSet:
    """
    A saved set; checked against `mjm` when given.

    Called by `tools.performance_tuning.calibrate.main`, `tools.performance_tuning.tune.main`,
    `viewer.performance.Performance.load_states` and 5 others.
    In: path str or Path, an `.npz` `StateSet.save` wrote; mjm mujoco.MjModel or None. Out: StateSet.
    """
    z = np.load(path, allow_pickle=False)  # NpzFile: the arrays and the header
    meta = json.loads(str(z["meta"]))  # dict: the header
    if meta.get("format") != FORMAT:
        raise ValueError(f"{path}: not a state set ({meta.get('format')!r})")
    s = StateSet(  # StateSet
        qpos=z["qpos"], qvel=z["qvel"], ctrl=z["ctrl"], act=z["act"], weights=z["weights"],
        signature=meta["signature"], labels=meta["labels"],
        controls=z["controls"] if "controls" in z else None, source=meta["source"],
    )
    if mjm is not None:
        s.check(mjm)
    return s


def from_qpos(mjm, qpos, qvel=None, ctrl=None, act=None, weights=None, labels=None, controls=None, source=None):
    """
    A set from arrays; anything left out is zero (`ctrl`: the model's `qpos0` has none either).

    Called by `tools.performance_tuning.explore.main`, `tools.performance_tuning.explore._rollout_samples`,
    `viewer.performance.Performance._from_frames` and 3 others.
    In: mjm mujoco.MjModel; qpos (N, nq) or (nq,); qvel (N, nv), ctrl (nu,) or (N, nu), act (N, na) and weights (N,),
    each or None; labels, N str or None; controls (N, H, nu) or None; source dict or None. Out: StateSet.
    """
    qpos = np.atleast_2d(np.asarray(qpos, dtype=np.float64))
    n = len(qpos)  # int: N
    zeros = lambda w: np.zeros((n, w))  # noqa: E731  # callable: (n, w) float64 zeros
    return StateSet(
        qpos=qpos,
        qvel=zeros(mjm.nv) if qvel is None else np.atleast_2d(qvel),
        ctrl=zeros(mjm.nu) if ctrl is None else np.broadcast_to(np.asarray(ctrl, dtype=np.float64), (n, mjm.nu)),
        act=zeros(mjm.na) if act is None else np.atleast_2d(act),
        weights=np.ones(n) if weights is None else weights,
        signature=model_signature(mjm), labels=list(labels or []), controls=controls,
        source=dict(source or {"kind": "arrays"}),
    )


def from_keyframes(mjm, names=None):
    """
    The model's keyframes (all, or those named), each with its own `ctrl` and `act`.

    Called by `tools.performance_tuning.explore.main`.
    In: mjm mujoco.MjModel; names, keyframe names (a collection of str) or None. Out: StateSet.
    """
    keys = [k for k in range(mjm.nkey) if names is None or mjm.key(k).name in names]  # list of int: the keyframes
    if not keys:
        raise ValueError("the model has no keyframes" if names is None else f"no keyframes named {names}")
    return StateSet(
        qpos=mjm.key_qpos[keys], qvel=mjm.key_qvel[keys], ctrl=mjm.key_ctrl[keys], act=mjm.key_act[keys],
        weights=np.ones(len(keys)), signature=model_signature(mjm),
        labels=[f"keyframe {mjm.key(k).name or k}" for k in keys], source=dict(kind="keyframes"),
    )


def concat(*sets):
    """
    One set from several of the same model; their sources are kept as a list.

    Called by `sample_rollout`, `tools.performance_tuning.explore.main`, `tools.performance_tuning.tune.make_states`
    and 2 others.
    In: sets, StateSets or None (skipped, as are empty sets). Out: StateSet of their states in order, the control
    sequences cut to the shortest.
    """
    sets = [s for s in sets if s is not None and len(s)]
    if not sets:
        raise ValueError("nothing to concatenate")
    sig = sets[0].signature  # dict: the model's signature
    if any(s.signature != sig for s in sets):
        raise ValueError("the sets are for different models")
    has = [s.controls is not None for s in sets]  # list of bool: which sets have control sequences
    controls = None  # (sum of N, H, nu) float64 or None
    if all(has):
        H = min(s.controls.shape[1] for s in sets)  # int: the shortest sequence
        controls = np.concatenate([s.controls[:, :H] for s in sets])
    elif any(has):
        raise ValueError("either every set has control sequences or none does")
    return StateSet(
        qpos=np.concatenate([s.qpos for s in sets]), qvel=np.concatenate([s.qvel for s in sets]),
        ctrl=np.concatenate([s.ctrl for s in sets]), act=np.concatenate([s.act for s in sets]),
        weights=np.concatenate([s.weights for s in sets]), signature=sig,
        labels=sum((list(s.labels) for s in sets), []), controls=controls,
        source=dict(kind="concat", parts=[s.source for s in sets]),
    )


#######################################################################################
# Layouts
#######################################################################################


def batches(n_states, B, layout="independent", assign=None, weights=None, rng=None, cover=True):
    """
    The batches that visit a set: a list of `(B,)` state-index arrays, one per batch.

    independent: `cover=True` (the default) visits every state at least once, in
        `ceil(N / B)` batches, the last filled out by weighted draws; `cover=False` draws
        one batch of `B` by `weights`.
    shared: one batch per state, every world of it from that state.
    explicit: `assign`, `(B,)` or `(nbatch, B)` indices.

    Called by `lodos.utils.calibrate.measure`, `tools.performance_tuning.profile_scene.profile` and
    `tools.performance_tuning.tune.timing_batch`.
    In: n_states int, N; B int; layout str, one of `LAYOUTS`; assign as above, or None; weights (N,) or None;
    rng, a numpy Generator or None (one seeded 0); cover bool. Out: list of (B,) int64 arrays.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    p = None  # (N,) float64 or None: each state's probability of being drawn
    if weights is not None:
        p = np.asarray(weights, dtype=np.float64)
        p = p / p.sum()
    if layout == "independent":
        if not cover:
            return [rng.choice(n_states, size=B, p=p)]
        order = rng.permutation(n_states)  # (N,) int64: the states, shuffled
        out = []  # list of (B,) int64
        for start in range(0, n_states, B):  # start: int, the batch's first place in `order`
            chunk = order[start:start + B]  # (<= B,) int64: the batch's states
            if len(chunk) < B:
                chunk = np.concatenate([chunk, rng.choice(n_states, size=B - len(chunk), p=p)])
            out.append(chunk)
        return out
    if layout == "shared":
        return [np.full(B, i, dtype=np.int64) for i in range(n_states)]
    if layout == "explicit":
        a = np.asarray(assign, dtype=np.int64)  # (B,) or (nbatch, B) int64
        a = a.reshape(-1, B)
        if a.min() < 0 or a.max() >= n_states:
            raise ValueError("an explicit assignment names a state the set does not have")
        return list(a)
    raise ValueError(f"layout must be one of {LAYOUTS}, not {layout!r}")


#######################################################################################
# Controls
#######################################################################################


@dataclasses.dataclass
class Controls:
    """
    What the actuators do during a rollout: `kind` is "hold", "none", "random" or
    "sequence" (see the module docstring).

    random: an Ornstein-Uhlenbeck walk per world and actuator, `du = theta (u0 - u) dt +
        sigma w sqrt(dt) N(0, 1)`, clipped to `ctrlrange`; `w` is the range's width (1
        for an unlimited actuator) and `u0` the held control.

    Built by `random`, `rollout`, `lodos.utils.calibrate.measure` and 1 other.
    """

    kind: str = "hold"  # "hold", "none", "random" or "sequence"
    sigma: float = 0.5  # random: the noise, in ranges' widths per sqrt(s)
    theta: float = 2.0  # random: the rate of reversion to the held control, 1/s
    seed: int = 0  # random: the noise's seed

    @classmethod
    def random(cls, sigma=0.5, theta=2.0, seed=0):
        """
        The random walk's `Controls`.

        Called by `tools.performance_tuning.calibrate.controls_of`, `tools.performance_tuning.explore.main` and
        `tools.performance_tuning.tune.make_states`.
        In: sigma and theta float, seed int, as the fields. Out: Controls.
        """
        return cls("random", sigma, theta, seed)

    def describe(self):
        """
        The controls as a tuning file records them: their kind, and the walk's parameters when random.

        Called by `lodos.utils.calibrate.measure` and `tools.performance_tuning.calibrate.main`. Out: dict.
        """
        return dict(kind=self.kind, sigma=self.sigma, theta=self.theta, seed=self.seed) if self.kind == "random" \
            else dict(kind=self.kind)


class _ControlStream:
    """
    The per-step `(B, nu)` controls of one batch.

    Built by `rollout`.
    """

    def __init__(self, mjm, sset, idx, controls):
        """
        Takes the batch's held controls and the ranges a random walk is clipped to; raises on an unknown kind, or on
        "sequence" for a set without sequences.

        Called when a `_ControlStream` is built (see the class).
        In: mjm mujoco.MjModel; sset StateSet; idx (B,) int, the worlds' states; controls, a `Controls`.
        """
        self.kind = controls.kind  # str: "hold", "none", "random" or "sequence"
        self.u0 = sset.ctrl[idx]  # (B, nu) float64: the held controls
        self.u = self.u0.copy()  # (B, nu) float64: the controls now
        self.seq = None  # (B, H, nu) float64 or None: the sequences played
        if self.kind == "sequence":
            if sset.controls is None:
                raise ValueError("the set has no control sequences")
            self.seq = sset.controls[idx]
        elif self.kind == "none":
            self.u[:] = 0.0
        elif self.kind not in ("hold", "random"):
            raise ValueError(f"unknown controls {self.kind!r}")
        lo, hi = mjm.actuator_ctrlrange[:, 0], mjm.actuator_ctrlrange[:, 1]  # (nu,), (nu,) float64: the ranges
        limited = np.asarray(mjm.actuator_ctrllimited, dtype=bool) & (hi > lo)  # (nu,) bool: a range applies
        self.lo = np.where(limited, lo, -np.inf)  # (nu,) float64: the lower bounds, -inf where unlimited
        self.hi = np.where(limited, hi, np.inf)  # (nu,) float64: the upper bounds, inf where unlimited
        self.width = np.where(limited, hi - lo, 1.0)  # (nu,) float64: the ranges' widths, 1 where unlimited
        self.dt = float(mjm.opt.timestep)  # float: the timestep, s
        self.c = controls  # Controls
        self.rng = np.random.default_rng(controls.seed)  # numpy Generator: the walk's noise

    @property
    def constant(self):
        """
        Whether the controls stay as `rollout` first writes them: "hold" and "none".

        Called by `rollout`. Out: bool.
        """
        return self.kind in ("hold", "none")

    def at(self, t):
        """
        The `(B, nu)` controls of step `t`: a sequence's step `t` (its last past the end), the walk advanced one step
        on every call with `t > 0`, or the held (or zero) controls.

        Called by `rollout`.
        In: t int, the step. Out: (B, nu) float64; for "hold", "none" and "random", the stream's own array.
        """
        if self.kind == "sequence":
            return self.seq[:, min(t, self.seq.shape[1] - 1)]
        if self.kind == "random" and t > 0:
            c, dt = self.c, self.dt  # Controls, float (s)
            noise = self.rng.standard_normal(self.u.shape)  # (B, nu) float64: N(0, 1) draws
            self.u += c.theta * (self.u0 - self.u) * dt + c.sigma * self.width * np.sqrt(dt) * noise
            np.clip(self.u, self.lo, self.hi, out=self.u)
        return self.u


#######################################################################################
# Rollouts
#######################################################################################


def install(sim, sset, idx):
    """
    Writes the states `idx` (`(B,)` indices) into `sim`'s worlds and clears the solver's carried state.

    Called by `rollout`, `tools.performance_tuning.common.throughput`, `tools.performance_tuning.common.artifacts` and
    3 others.
    In: sim, a `lodos.sim.Sim`; sset StateSet; idx (B,) int. Writes: drbd.qpos (B, nq) and qvel (B, nv), and ctrl
    (B, nu) and act (B, na) where the model has them, float32; the poses `lodos.rbd.smooth.kinematics` derives;
    d.qacc_warmstart (nworld, nv) float32, zeroed.
    """
    from lodos.rbd import io as rbd_io
    from lodos.rbd import smooth

    rbd_io.set_state(sim.drbd, qpos=sset.qpos[idx], qvel=sset.qvel[idx],
                   ctrl=sset.ctrl[idx] if sset.ctrl.shape[1] else None,
                   act=sset.act[idx] if sset.act.shape[1] else None)
    smooth.kinematics(sim.mrbd, sim.drbd)
    d = getattr(sim, "d", None)  # lodos.mjcf.types.Data or None: MJWarp's data
    if d is not None:
        d.qacc_warmstart.zero_()


def rollout(sim, sset, idx, horizon, controls=None, replay=None, on_step=None, every=1):
    """
    Runs the worlds of `sim` from the states `idx` for `horizon` steps.

    replay: a captured step (`lodos.step.make_step_graph(sim)`), or None to step without
        a graph. Capture before installing: capturing runs a step.
    on_step(t): called after step `t` when `(t + 1) % every == 0` and after the last.

    Called by `sample_rollout` and `lodos.utils.calibrate.measure`.
    In: sim, a `lodos.sim.Sim`; sset StateSet; idx (B,) int; horizon int; controls, a `Controls` or None (hold);
    every int. Writes: sim's state, advanced `horizon` steps, and drbd.ctrl (B, nu) float32 unless held.
    """
    from lodos import step as lstep
    from lodos.rbd import io as rbd_io

    controls = controls or Controls()
    stream = _ControlStream(sim.mjm, sset, idx, controls)  # _ControlStream: the batch's controls
    install(sim, sset, idx)
    if sim.mjm.nu and controls.kind != "hold":
        rbd_io.set_state(sim.drbd, ctrl=stream.at(0))
    step = replay or (lambda: lstep.fwd_step(sim))  # callable: one step
    for t in range(horizon):  # t: int, the step
        if sim.mjm.nu and not stream.constant and t > 0:
            rbd_io.set_state(sim.drbd, ctrl=stream.at(t))
        step()
        if on_step is not None and ((t + 1) % every == 0 or t == horizon - 1):
            on_step(t)


def sample_rollout(sim, sset, idx, horizon, stride, controls=None, replay=None, label="rollout"):
    """
    A new set: the worlds' states every `stride` steps of a rollout from `idx`, with the
    control each world was holding. What a still frame misses -- the impact, the motion
    -- is in a rollout, which is why calibrating on still frames undersizes.

    Called by `tools.performance_tuning.explore._rollout_samples`.
    In: sim, a `lodos.sim.Sim`; sset StateSet; idx (B,) int; horizon and stride int; controls and replay as `rollout`
    takes them; label str, the new states' label and source kind. Out: StateSet of B * ceil(horizon / stride) states.
    """
    from lodos.rbd import io as rbd_io

    out = []  # list of (t, qpos (B, nq), qvel (B, nv), ctrl (B, nu), act (B, na)): the samples, float64

    nu, na = sset.signature["nu"], sset.signature["na"]  # int, int

    def grab(t):
        """
        Records the worlds' states after step `t` in `out`, with the controls they held.

        Called by `rollout`, as its `on_step`. In: t int, the step.
        """
        # `ctrl` and `act` are one column wide on the device when the model has none
        st = rbd_io.get_state(sim.drbd)  # dict: qpos (B, nq), qvel (B, nv), act (B, max(na, 1)) and time (B,) float32
        out.append((t, st["qpos"].astype(np.float64), st["qvel"].astype(np.float64),
                    sim.drbd.ctrl.numpy()[:, :nu].astype(np.float64), st["act"][:, :na].astype(np.float64)))

    rollout(sim, sset, idx, horizon, controls, replay, grab, every=stride)
    sets = []  # list of StateSet: a sample each
    for t, q, v, u, a in out:  # t: int; q (B, nq), v (B, nv), u (B, nu), a (B, na) float64
        sets.append(StateSet(q, v, u, a, np.ones(len(q)), sset.signature,
                             labels=[f"{label} {sset.labels[i]} t={t + 1}" for i in idx],
                             source=dict(kind=label)))
    return concat(*sets)
