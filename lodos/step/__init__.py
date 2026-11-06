"""
The complete simulation step, on the device.

Unconstrained rigid-body dynamics, collision detection, contact dynamics and the
integrator, in that order, with no host round trip anywhere between them -- so
`make_step_graph` can capture the whole thing into a single CUDA graph. That is
what this module exists for; the packages it calls also work on their own.
"""

import warp as wp

from lodos.col.sdf import collide as sdf_collide
from lodos.rbd import forward as rbd_forward
from lodos.rbd import integrate as rbd_integrate


def forward(sim):
    """
    Everything `step_sim` does before the integrator, like MuJoCo's `mj_forward`: the
    poses, contacts, constraint forces and `qacc` of the current state, which is left
    as it was.

    Called by `step_sim`, `viewer.session.Runner.capture`, `viewer.scenes.nist.thread_phase` and 2 others.
    In: sim (`lodos.sim.Sim`), its state drbd.qpos (B, nq) and drbd.qvel (B, nv) float32. Out: drbd's kinematics
    and smooth dynamics; the contacts, MJWarp's (d.contact) and Lodos's (dcol); the constraint rows and forces
    (d.efc); drbd.qacc (B, nv) float32 and d.qacc_warmstart (B, nv) float32.
    """
    rbd_forward.fwd_rbd(sim.mrbd, sim.drbd)
    if sim.prim.active:
        sim.prim.poses(sim.drbd.xpos, sim.drbd.xmat)
    sim.prim.collide()
    if sim.mcol is not None:
        sdf_collide.collide(sim.mcol, sim.dcol)
    sim.solver.contact_dynamics()


def step_sim(sim):
    """
    One step of a `lodos.sim.Sim`, in place.

        rigid-body pass         lodos.rbd: kinematics .. qacc_smooth, M and its factor
        MJWarp collision        geom poses from body poses, broadphase, narrowphases
                                and the hybrid, into d.contact (if it has any pairs)
        Lodos collision         lodos.col.sdf, into dcol (if there are Lodos pairs)
        contact dynamics        MJWarp's constraint solve over d.contact, Lodos's
                                contacts added (`lodos.cd.driver`)
        integration             lodos.rbd's semi-implicit Euler, or implicitfast
                                (`lodos.rbd.implicit`)

    Called by `fwd_step` and `viewer.session.Runner.capture`.
    In: sim (`lodos.sim.Sim`). Out: its state, drbd.qpos (B, nq) and drbd.qvel (B, nv) float32 and drbd.time (B,)
    float32, advanced one step in place; everything `forward` writes.
    """
    forward(sim)
    rbd_integrate.integrate(sim.mrbd, sim.drbd)


def fwd_step(sim):
    """
    One step: state in `drbd.qpos`/`drbd.qvel`, state out in the same arrays (`step_sim`).

    Called by `make_step_graph`, `lodos.utils.calibrate.Tracker.capture`, `lodos.utils.states.rollout` and 5 others.
    In and out: sim (`lodos.sim.Sim`), as `step_sim`.
    """
    step_sim(sim)


def make_step_graph(sim, device=None):
    """
    Captures `fwd_step` into a CUDA graph and returns a callable that replays it.

    Valid only while the buffers keep their addresses, which is the discipline
    every context here follows: nothing is rebound after construction, and state
    is copied into the preallocated arrays rather than swapped for new ones.

    Each replay advances the state one step: call it `n` times to run `n` steps.

    Called by `lodos.utils.calibrate.peaks`, `tools.performance_tuning.common.throughput`
    and others (the tuning tools, `docs/manual/make_data.py` and the notebooks).
    In: sim (`lodos.sim.Sim`); device, a Warp CUDA device or None for the current one. Out: `replay()`, which takes
    one step a call. Building it takes one step, the warm-up: the captured launches are recorded, not run.
    """
    device = device or wp.get_device()

    # Warm up so kernel modules are loaded before capture.
    fwd_step(sim)
    wp.synchronize()

    with wp.ScopedCapture(device) as capture:  # capture: wp.ScopedCapture
        fwd_step(sim)

    graph = capture.graph  # wp.Graph: the captured step

    def replay():
        """
        One step: the captured graph, launched.

        Called by whoever `make_step_graph` returned it to.
        """
        wp.capture_launch(graph)

    return replay
