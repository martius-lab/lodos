"""
The Newton side of the live viewer's link (`viewer.newton_link.NewtonRunner`): one
scene, one world, under one of Newton's solvers, driven by length-prefixed pickles on stdin
and answered the same way on a private copy of stdout. Whatever Newton, Warp or MuJoCo
print goes to stderr, so it cannot corrupt the messages.

Commands (each a dict with `cmd`; each answer has `ok`, and `error` when it is False):

    build      pkg, solver, options, caps   the scene, its solver, Newton's pipeline; `caps` a dict
                                         of `newton_scene.Caps`, else measured
    set_state  qpos, qvel, time          MuJoCo's layout
    forward                              the current state's contacts, which capacities overflowed,
                                         and what it asked of them (`counts`), no step
    step       record, ctrl              one step, after writing `ctrl` (the actuators' controls,
                                         in the MJCF's order) where given; with record, the
                                         state it started from and that state's contacts, as
                                         a frame wants them; its overflows and counts as forward's
    options    options                   rebuilt with these, at the same state
    rebuild    options, caps             rebuilt with these (`caps` a dict), at the same state
    grow                                 twice the contacts and triangle pairs per world, rebuilt
    pairs                                the shape pairs and their contact parameters as built
                                         (`newton_scene.Sim.pair_info`), MuJoCo Warp's
    shape_params mu, solref, solimp      every shape's, on the running model (`set_shape_params`)
    params                               every shape's contact parameters now (`shape_params`)
    quit
"""

import os
import pickle
import struct
import sys
import time
import traceback

OUT = os.fdopen(os.dup(1), "wb")
os.dup2(2, 1)  # anything printed from here on goes to stderr
sys.stdout = sys.stderr

import numpy as np  # noqa: E402
import warp as wp  # noqa: E402

import newton_scene as NS  # noqa: E402

wp.config.quiet = True


def read():
    head = sys.stdin.buffer.read(8)
    if len(head) < 8:
        return None
    (n,) = struct.unpack("<Q", head)
    return pickle.loads(sys.stdin.buffer.read(n))


def write(obj):
    data = pickle.dumps(obj, protocol=5)
    OUT.write(struct.pack("<Q", len(data)) + data)
    OUT.flush()


class Server:
    def __init__(self):
        self.sim = None
        self.time = 0.0
        self.args = None

    def build(self, pkg, solver, options=None, caps=None, state=None):
        self.args = dict(pkg=pkg, solver=solver, options=dict(options or {}))
        caps = NS.Caps(**caps) if isinstance(caps, dict) else caps
        self.sim = NS.Sim(pkg, solver, B=1, options=options, caps=caps)
        if state is not None:
            self.sim.set_state(state[0][None], state[1][None])
        return dict(build_seconds=self.sim.build_seconds, contacts_per_world=self.sim.caps.contacts_per_world,
                    gap=NS.gap_of(self.sim.mjm, options), nq=int(self.sim.mjm.nq), nv=int(self.sim.mjm.nv),
                    dt=self.sim.dt, solver_defaults=self.sim.solver_defaults, solver_config=self.sim.solver_config,
                    caps=self.sim.capacity())

    def state(self):
        q, v = self.sim.get_state()
        return dict(qpos=q[0], qvel=v[0], time=self.time)

    def contacts(self):
        p, n, dist, f, bodies = self.sim.contacts_world()
        return dict(p=p, n=n, dist=dist, f=f, bodies=bodies, count=self.sim.contact_count())

    def set_state(self, qpos, qvel, time=0.0):
        self.sim.set_state(np.asarray(qpos)[None], np.asarray(qvel)[None])
        self.time = float(time)
        return {}

    def forward(self):
        self.sim.collide()
        return dict(contacts=self.contacts(), overflow=self.sim.overflows(), counts=self.sim.counts(), **self.state())

    def step(self, record=False, ctrl=None):
        out = {}
        if ctrl is not None:
            self.sim.set_ctrl(ctrl)
        if record:
            self.sim.collide()
            out["pre"] = dict(contacts=self.contacts(), **self.state())
        wp.synchronize()
        t0 = time.perf_counter()
        self.sim.step(1)
        wp.synchronize()
        out["ms"] = (time.perf_counter() - t0) * 1e3
        out["solver"] = self.sim.solver_status()
        self.time += self.sim.dt
        out["overflow"] = self.sim.overflows()
        out["counts"] = self.sim.counts()
        out.update(self.state())
        return out

    def rebuild(self, options=None, caps=None):
        st = self.sim.get_state()
        args = dict(self.args)
        if options is not None:
            args["options"] = dict(options)
        caps = NS.Caps(**caps) if isinstance(caps, dict) else caps if caps is not None else self.sim.caps
        t = self.time
        out = self.build(args["pkg"], args["solver"], args["options"], caps=caps, state=(st[0][0], st[1][0]))
        self.time = t
        return out

    def options(self, options):
        return self.rebuild(options=options)

    def grow(self):
        c = self.sim.caps
        caps = NS.dataclasses.replace(c, contacts_per_world=2 * int(c.contacts_per_world),
                                      triangle_pairs_per_world=2 * int(c.triangle_pairs_per_world))
        return self.rebuild(caps=caps)

    def pairs(self):
        return self.sim.pair_info()

    def shape_params(self, mu, solref, solimp):
        self.sim.set_shape_params(mu, solref, solimp)
        return {}

    def params(self):
        return self.sim.shape_params()


def main():
    srv = Server()
    while True:
        msg = read()
        if msg is None or msg.get("cmd") == "quit":
            write(dict(ok=True))
            return
        try:
            cmd = msg.pop("cmd")
            out = getattr(srv, cmd)(**msg)
            write(dict(ok=True, **out))
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            write(dict(ok=False, error=f"{type(e).__name__}: {e}"))


if __name__ == "__main__":
    main()
