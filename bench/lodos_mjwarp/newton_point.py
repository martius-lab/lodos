"""
One measurement of Newton's MuJoCo Warp on a scene, in a process of its own in viewer/newton's environment
(`run.py` drives it):

    viewer/newton/.venv/bin/python bench/lodos_mjwarp/newton_point.py measure PKG OPTIONS STARTS CONTROLS OUT
    viewer/newton/.venv/bin/python bench/lodos_mjwarp/newton_point.py point PKG OPTIONS STARTS CONTROLS OUT \
        --sizes SIZES --B B

The scene comes as `viewer.newton_link.scene_package` writes it, its options (Lodos's solver settings, the
implicitfast integrator, the elliptic cone, every contact's solref) as JSON, its starts and recorded controls as
`.npz` (CONTROLS "-" for none). `measure`: every capacity's largest demand of one world -- contacts and mesh
triangle pairs a collision pass, MuJoCo Warp's rows -- over the starts, each simulated alone for the scene's whole
run, and a world's capacities from them, each 1.5 times its peak. `point`: B worlds from their own starts for the
whole run, timed, the demand tracked in the step (`newton_scene.Sim.track`), the lost worlds counted.
`--broad-phase sap`: Newton's sweep-and-prune broad phase in place of its explicit pairs, its candidate shape
pairs a capacity like the others (`newton_scene.Caps.shape_pairs_per_world`), measured and tracked.
"""

from __future__ import annotations

import argparse
import math
import pickle
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "viewer" / "newton"))
sys.path.insert(0, str(ROOT))

import newton_scene as NS  # noqa: E402

from bench.lodos_mjwarp import common as C  # noqa: E402

wp.config.quiet = True


@wp.kernel
def _row(table: wp.array2d(dtype=wp.float32), k: int, n: int, out: wp.array(dtype=wp.float32)):  # type: ignore
    i = wp.tid()
    out[i] = table[k, i % n]


@wp.kernel
def _world_contacts(count: wp.array(dtype=wp.int32), shape0: wp.array(dtype=wp.int32),  # type: ignore
                    shape1: wp.array(dtype=wp.int32), shape_world: wp.array(dtype=wp.int32),
                    n: wp.array(dtype=wp.int32)):
    i = wp.tid()
    if i < count[0] and shape0[i] >= 0:
        w = shape_world[shape0[i]]
        if w < 0:
            w = shape_world[shape1[i]]
        if w >= 0:
            wp.atomic_add(n, w, 1)


@wp.kernel
def _world_peaks(n: wp.array(dtype=wp.int32), nefc: wp.array(dtype=wp.int32),  # type: ignore
                 peak: wp.array2d(dtype=wp.int32)):
    w = wp.tid()
    peak[0, w] = wp.max(peak[0, w], n[w])
    peak[1, w] = wp.max(peak[1, w], nefc[w])
    n[w] = 0


class Ticks:
    """
    Recorded controls on the device: for each array `Sim.set_ctrl` writes (MuJoCo's `ctrl`, the drives' targets,
    under Kamino the forces it applies on sites), its first world's part at every tick, which a kernel writes to
    every world at the tick.
    """

    def __init__(self, sim, rows, period):
        self.period, self.k, self.arrs = period, None, []
        if rows is None:
            return
        cand = []
        cm = getattr(getattr(sim.control, "mujoco", None), "ctrl", None)
        if cm is not None and cm.shape[0]:
            cand.append(cm)
        jt = getattr(sim.control, "joint_target_q", None)
        if sim.targets and jt is not None:
            cand.append(jt)
        applied = getattr(sim, "_applied", None)  # Kamino's forces on sites, each its actuator's (B, n): flat, a view
        if applied is not None:
            cand.append(applied["mag"].reshape(-1))
        per = [a.shape[0] // sim.B for a in cand]
        tabs = [np.zeros((len(rows), n), np.float32) for n in per]
        for k, r in enumerate(rows):
            sim.set_ctrl(r)
            for a, tab, n in zip(cand, tabs, per):
                tab[k] = a.numpy()[:n]
        self.arrs = [(a, wp.array(t, dtype=wp.float32), n) for a, t, n in zip(cand, tabs, per)]

    def at(self, k):
        if self.arrs and k != self.k:
            for a, t, n in self.arrs:
                wp.launch(_row, dim=a.shape[0], inputs=[t, int(min(k, t.shape[0] - 1)), int(n)], outputs=[a])
            self.k = k

    def step(self, i, dt):
        if self.arrs:
            self.at(C.tick(i, dt, self.period))


def load(args):
    pkg = pickle.loads(Path(args.pkg).read_bytes())
    opts = C.read_json(args.options)
    z = np.load(args.starts)
    ctl = None if args.controls == "-" else np.load(args.controls)
    return pkg, opts, z["qpos"], z["qvel"], (None, None) if ctl is None else (ctl["rows"], float(ctl["period"]))


def overflows(sim, peaks):
    """
    Which capacities the tracked run overflowed: `contacts`, `triangle_pairs` (or the table reducing the mesh
    contacts, which they size, in any step), `rows`, and the broad phase's capped `shape_pairs`.
    """
    out = []
    if peaks["contacts"] > int(sim.contacts.rigid_contact_max):
        out.append("contacts")
    if peaks["triangle_pairs"] > sim.caps.triangle_pairs(sim.B) or peaks["reduction_lost"] > 0:
        out.append("triangle_pairs")
    if peaks["rows"] >= int(sim.solver.mjw_data.njmax):
        out.append("rows")
    if sim.caps.shape_pairs_per_world is not None and peaks["shape_pairs"] > sim.caps.shape_pairs(sim.B):
        out.append("shape_pairs")
    return out


def start(sim, q, v):
    sim.set_state(q, v)
    sim.solver.mjw_data.qacc_warmstart.zero_()  # as Lodos's start does its own


def measure(pkg, opts, q, v, rows, period):
    """The starts each alone for the whole run: the peaks, and a world's capacities, each 1.5 times its peak."""
    caps = NS.Caps(contacts_per_world=8192)  # generous at one world: the piles start apart, a probe there sees none
    t0 = time.perf_counter()
    for attempt in range(6):
        sim = NS.Sim(pkg, "mjwarp", B=1, caps=caps, options=opts)
        sim.track()
        dt, n = sim.dt, C.steps(pkg["name"], sim.dt)
        ticks = Ticks(sim, rows, period)
        peaks, over = dict(contacts=0, triangle_pairs=0, rows=0, reduced=0, shape_pairs=0), set()
        for s in range(len(q)):
            start(sim, q[s:s + 1], v[s:s + 1])
            sim.reset_peaks()
            ticks.k = None
            for i in range(n):
                ticks.step(i, dt)
                sim.step(1)
            p = sim.peaks()
            over |= set(overflows(sim, p))
            peaks = {k: max(peaks[k], p[k]) for k in peaks}
            print(f"start {s}: {time.perf_counter() - t0:.0f} s, {p}", flush=True)
        if not over:
            break
        caps = grown(sim, over)
        del sim
    else:
        raise RuntimeError(f"{pkg['name']}: the capacities kept overflowing")
    return sized(sim, peaks, n, dt, t0, attempt)


def measure_batched(pkg, opts, q, v, rows, period):
    """
    `measure` with the starts at once, each a world of its own: a world's contacts (counted from its shapes) and
    rows tracked in every step are those of its start alone, the worlds being independent; its mesh triangle pairs
    the batch's (an upper bound). For a scene one world of which steps too slowly to take the starts one by one.
    """
    caps = NS.Caps(contacts_per_world=8192)
    B = len(q)
    t0 = time.perf_counter()
    for attempt in range(6):
        sim = NS.Sim(pkg, "mjwarp", B=B, caps=caps, options=opts)
        sim.track()
        dt, n = sim.dt, C.steps(pkg["name"], sim.dt)
        ticks = Ticks(sim, rows, period)
        con, cnt, per = sim.contacts, wp.zeros(B, dtype=wp.int32), wp.zeros((2, B), dtype=wp.int32)
        start(sim, q, v)
        sim.reset_peaks()
        t1 = time.perf_counter()
        for i in range(n):
            ticks.step(i, dt)
            sim.step(1)
            wp.launch(_world_contacts, dim=int(con.rigid_contact_max), inputs=[
                con.rigid_contact_count, con.rigid_contact_shape0, con.rigid_contact_shape1, sim.model.shape_world],
                outputs=[cnt])
            wp.launch(_world_peaks, dim=B, inputs=[cnt, sim.solver.mjw_data.nefc], outputs=[per])
            if (i + 1) % 100 == 0:
                print(f"step {i + 1}: {time.perf_counter() - t1:.0f} s, {sim.peaks()}", flush=True)
        p = sim.peaks()
        over = set(overflows(sim, p))
        if not over:
            break
        caps = grown(sim, over)
        del sim
    else:
        raise RuntimeError(f"{pkg['name']}: the capacities kept overflowing")
    pw = per.numpy()
    peaks = dict(contacts=int(pw[0].max()), triangle_pairs=p["triangle_pairs"], rows=int(pw[1].max()),
                 reduced=p["reduced"], shape_pairs=p["shape_pairs"])
    return dict(sized(sim, peaks, n, dt, t0, attempt), batched=True, world_peaks=pw.tolist())


def grown(sim, over):
    """`sim`'s capacities with each that overflowed doubled."""
    c = sim.caps
    return NS.Caps(contacts_per_world=int(c.contacts_per_world) * (2 if "contacts" in over else 1),
                   triangle_pairs_per_world=C.doubled_triangle_pairs(c.triangle_pairs_per_world, sim.B)
                   if "triangle_pairs" in over else c.triangle_pairs_per_world,
                   njmax=int(sim.solver.mjw_data.njmax) * (2 if "rows" in over else 1),
                   shape_pairs_per_world=None if c.shape_pairs_per_world is None else
                   int(c.shape_pairs_per_world) * (2 if "shape_pairs" in over else 1))


def sized(sim, peaks, n, dt, t0, attempt):
    """
    A world's capacities from its peaks, each 1.5 times, and the record of the measurement. The triangle pairs
    size the table reducing the mesh contacts as well, so they hold the contacts it is given too. Under a broad
    phase that finds its pairs (`--broad-phase`), its candidate shape pairs as well.
    """
    f = 1.0 + C.OVERHEAD
    sizes = dict(contacts_per_world=max(1, math.ceil(f * peaks["contacts"])),
                 triangle_pairs_per_world=max(1, math.ceil(f * max(peaks["triangle_pairs"], peaks["reduced"]))),
                 njmax=max(16, math.ceil(f * peaks["rows"])))
    if sim.options.get("broad_phase"):
        sizes["shape_pairs_per_world"] = max(1, math.ceil(f * peaks["shape_pairs"]))
    return dict(peaks=peaks, sizes=sizes, steps=n, dt=dt, seconds=time.perf_counter() - t0, attempts=attempt + 1,
                measured_caps=dict(contacts_per_world=int(sim.caps.contacts_per_world),
                                   triangle_pairs_per_world=int(sim.caps.triangle_pairs_per_world)))


def point(pkg, opts, sizes, B, q, v, rows, period):
    """B worlds from their own starts for the whole run, timed; its demand, overflows and lost worlds."""
    q, v = q[:B], v[:B]
    t_build = time.perf_counter()
    sim = NS.Sim(pkg, "mjwarp", B=B, caps=NS.Caps(**sizes), options=opts)
    sim.track()
    ticks = Ticks(sim, rows, period)
    start(sim, q, v)
    sim.step(3)  # loading and capturing everything
    wp.synchronize()
    build_s = time.perf_counter() - t_build
    dt, n = sim.dt, C.steps(pkg["name"], sim.dt)
    start(sim, q, v)
    sim.reset_peaks()
    ticks.k = None
    ticks.at(0)
    wp.synchronize()
    # A full table reducing the mesh contacts makes a step take seconds (its failed inserts probe the whole table):
    # the tracked losses, copied to pinned memory every fiftieth of the run and read at the next, end such a run
    # early, an overflow the driver redoes. The host waits for the last copy before it reads it, so it runs at most
    # a fiftieth ahead of the device, which the steps queued behind the copy keep busy.
    seen = wp.zeros(sim.peak.shape[0], dtype=wp.int32, device="cpu", pinned=True)
    every, aborted, copied = max(1, n // 50), None, None
    t0 = time.perf_counter()
    for i in range(n):
        ticks.step(i, dt)
        sim.step(1)
        if (i + 1) % every == 0:
            if copied is not None:
                wp.synchronize_event(copied)
                if seen.numpy()[4] > 0:
                    aborted = i + 1
                    break
            wp.copy(seen, sim.peak)
            copied = wp.record_event()
        if (i + 1) % max(1, n // 10) == 0:  # when each tenth was queued, a stall's trace in the log
            print(f"step {i + 1}: {time.perf_counter() - t0:.1f} s", flush=True)
    wp.synchronize()
    wall = time.perf_counter() - t0
    peaks = sim.peaks()
    over = overflows(sim, peaks)
    qe, ve = sim.get_state()
    nan = ~(np.isfinite(qe).all(1) & np.isfinite(ve).all(1))
    m = sim.mjm
    free = [int(m.jnt_qposadr[j]) for j in range(m.njnt) if int(m.jnt_type[j]) == 0]  # mjJNT_FREE
    esc = C.escaped(np.asarray(qe, np.float64), q, free) & ~nan
    k = aborted or n
    return dict(B=B, steps=k, dt=dt, seconds=wall, world_steps_per_s=B * k / wall, sim_seconds_per_s=B * k * dt / wall,
                aborted_at=aborted, build_seconds=build_s, overflowed=bool(over), overflow=over, peaks=peaks,
                nan_worlds=int(nan.sum()),
                escaped_worlds=int(esc.sum()), sizes=sizes, final_qpos0=np.asarray(qe[0], np.float64).tolist(),
                capacity=dict(contacts=int(sim.contacts.rigid_contact_max), triangle_pairs=sim.caps.triangle_pairs(B),
                              rows=int(sim.solver.mjw_data.njmax), shape_pairs=int(sim.pipeline.shape_pairs_max)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=("measure", "point"))
    for k in ("pkg", "options", "starts", "controls", "out"):
        ap.add_argument(k)
    ap.add_argument("--sizes")
    ap.add_argument("--B", type=int)
    ap.add_argument("--batched", action="store_true", help="measure: the starts at once (`measure_batched`)")
    ap.add_argument("--broad-phase", choices=("sap",), help="Newton's broad phase, in place of its explicit pairs")
    a = ap.parse_args()
    t0 = time.perf_counter()
    res = dict(task=a.task, sim="mjwarp", broad_phase=a.broad_phase or "explicit")
    try:
        pkg, opts, q, v, (rows, period) = load(a)
        if a.broad_phase:
            opts["broad_phase"] = a.broad_phase
        res["scene"] = pkg["name"]
        if a.task == "measure":
            res.update((measure_batched if a.batched else measure)(pkg, opts, q, v, rows, period))
        else:
            res.update(point(pkg, opts, C.read_json(a.sizes)["sizes"], a.B, q, v, rows, period))
        res["ok"] = True
    except Exception as e:  # noqa: BLE001 -- recorded, the driver decides
        traceback.print_exc()
        res.update(ok=False, error=f"{type(e).__name__}: {e}"[:500])
    res["process_seconds"] = time.perf_counter() - t0
    C.write_json(a.out, res)


if __name__ == "__main__":
    main()
