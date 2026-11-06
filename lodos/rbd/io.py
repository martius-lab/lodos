"""
Putting a state into the Warp context, and taking one back out.

Nothing in the step path calls any of this. `set_state` is how a scene starts
somewhere other than `qpos0` -- a stack of dropped bodies, a grasp to replay --
and `save_state`/`restore_state` branch a trajectory without a host round trip.

`sync_state` and `load_state` are the same thing from a namespace of batched
arrays, for comparing this port with another implementation: driving both sides
from one state is what makes "the two agree" a statement about the same state
rather than about two states that merely started the same way. They take anything
supporting `__dlpack__`.
"""

import numpy as np
import warp as wp

from lodos.rbd import forward
from lodos.rbd.types import RbdData, RbdModel


def sync_state(d, drbd: RbdData):  # type: ignore
    """
    Copies the state -- `qpos`, `qvel`, and `ctrl`/`act` if the model has them --
    from a namespace of batched arrays into the Warp context.

    The state and nothing else: every other field of `RbdData` is derived, and
    deriving it is what the port is for. Copied rather than rebound, so pointers
    captured in a CUDA graph stay valid -- the same discipline the rest of the
    Warp code follows.

    Called by `load_state`.
    In: d, with `qpos` (B, nq) and `qvel` (B, nv) float32 arrays supporting `__dlpack__`, and optionally `ctrl`
    (B, nu) and `act` (B, na). Out: drbd.qpos, drbd.qvel, drbd.ctrl and drbd.act, copied into.
    """
    wp.copy(drbd.qpos, wp.from_dlpack(d.qpos, dtype=wp.float32))
    wp.copy(drbd.qvel, wp.from_dlpack(d.qvel, dtype=wp.float32))
    if drbd.ctrl.shape[1] and hasattr(d, "ctrl") and d.ctrl.shape[-1]:
        wp.copy(drbd.ctrl, wp.from_dlpack(d.ctrl, dtype=wp.float32))
    if drbd.act.shape[1] and hasattr(d, "act") and d.act.shape[-1]:
        wp.copy(drbd.act, wp.from_dlpack(d.act, dtype=wp.float32))


def set_state(drbd: RbdData, qpos=None, qvel=None, ctrl=None, act=None):  # type: ignore
    """
    Writes host arrays into the state fields, leaving the rest of `drbd` alone.

    Each argument is `(B, n)` -- the batch included, so different worlds can start
    differently -- and anything left out keeps what it had. Nothing is derived
    here: call `forward.fwd_rbd` afterwards, or `load_state`'s numpy twin
    `set_state` + `fwd_rbd`, before reading poses or colliding.

    Copied into the existing arrays rather than rebound, so pointers already
    captured in a CUDA graph stay valid.

    Called by `lodos.sim.make_sim`, `lodos.utils.states.install`, `viewer.session.Runner.set_state` and 7 others.
    In: qpos (B, nq), qvel (B, nv), ctrl (B, nu), act (B, na), host arrays cast to float32 and reshaped to the field's
    shape, or None. Out: drbd.qpos, drbd.qvel, drbd.ctrl and drbd.act (B, n) float32, the ones given copied into.
    """
    # name: str, the field; host: the host array given for it, or None
    for name, host in (("qpos", qpos), ("qvel", qvel), ("ctrl", ctrl), ("act", act)):
        if host is None:
            continue
        dst = getattr(drbd, name)  # wp.array (B, n) float32: the field
        src = np.ascontiguousarray(host, dtype=np.float32).reshape(dst.shape)  # (B, n) float32
        wp.copy(dst, wp.array(src, dtype=wp.float32))


def load_state(d, mrbd: RbdModel, drbd: RbdData):  # type: ignore
    """
    Loads a state's arrays and derives everything the Warp pipeline reads.

    It copies the state (`sync_state`) and derives the rest here (`forward.fwd_rbd`).

    Not called in the repository.
    In: d, as `sync_state` takes it; mrbd, drbd, B worlds' model and state. Out: drbd's state, and every field
    `forward.fwd_rbd` derives from it.
    """
    sync_state(d, drbd)
    forward.fwd_rbd(mrbd, drbd)


def get_state(drbd: RbdData):  # type: ignore
    """
    `qpos`, `qvel`, `act` and `time` as numpy arrays.

    Synchronizes, so this is for recording a trajectory or handing a frame to a
    renderer, not for the step loop.

    Called by `lodos.utils.states.sample_rollout.grab`.
    Out: dict of float32 numpy arrays: "qpos" (B, nq), "qvel" (B, nv), "act" (B, max(na, 1)), "time" (B,).
    """
    return {
        "qpos": drbd.qpos.numpy(),
        "qvel": drbd.qvel.numpy(),
        "act": drbd.act.numpy(),
        "time": drbd.time.numpy(),
    }


def save_state(drbd: RbdData):  # type: ignore
    """
    A device-side copy of the state, for restoring later.

    On the device rather than through numpy, so branching a trajectory -- run to
    a configuration, then continue from it under two different settings -- costs
    no host round trip and reproduces the branch point exactly.

    Not called in the repository.
    Out: dict of float32 wp.array clones: "qpos" (B, nq), "qvel" (B, nv), "act" (B, max(na, 1)), "time" (B,).
    """
    return {f: wp.clone(getattr(drbd, f)) for f in ("qpos", "qvel", "act", "time")}


def restore_state(drbd: RbdData, snapshot):  # type: ignore
    """
    Puts a `save_state` snapshot back.

    Not called in the repository.
    In: snapshot, `save_state`'s dict. Out: drbd.qpos, drbd.qvel, drbd.act and drbd.time, copied into.
    """
    for f, a in snapshot.items():  # f: str, a field; a: wp.array, its saved copy
        wp.copy(getattr(drbd, f), a)
