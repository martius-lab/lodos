"""
One measurement of Lodos's mj mode on a scene, in a process of its own (`run.py` drives it):

    python -m bench.lodos_mjwarp.lodos_point measure SCENE GROUPS OUT
    python -m bench.lodos_mjwarp.lodos_point ab SCENE GROUPS OUT --sizes SIZES
    python -m bench.lodos_mjwarp.lodos_point point SCENE GROUPS OUT --sizes SIZES --B B --variant V --starts NPZ

`measure`: every capacity's largest demand of one world, over `common.N_TUNE` starts each simulated alone for the
scene's whole run (`lodos.utils.calibrate.Tracker`), and the capacities a world gets from it
(`bench.lodos_setting.size`, each 1.5 times its peak); the states half-way through, which `ab` times from. `ab`:
the execution options (`common.VARIANTS`: the shared Hessian, the fused Newton solve) timed against the default,
from those states. `point`: B worlds, each from its own start, for the scene's whole run, timed; the demand on the
capacities tracked in the step (an overflowed run says so), and the worlds that went non-finite or escaped counted.
"""

from __future__ import annotations

import argparse
import gc
import time
import traceback

import mujoco
import numpy as np
import warp as wp

from bench.lodos_mjwarp import common as C
from bench import lodos_setting as LSET
from lodos.utils import calibrate as lcalibrate
from viewer.scenes import catalog as LS


@wp.kernel
def _row(table: wp.array2d(dtype=wp.float32), k: int, ctrl: wp.array2d(dtype=wp.float32)):  # type: ignore
    w, j = wp.tid()
    ctrl[w, j] = table[k, j]


def setting(source, groups):
    """The viewer's mj defaults for the scene, the implicitfast integrator, its contacts reduced, `groups`."""
    s = LSET.default_setting(source, "mj")
    s["integrator"] = "implicitfast"
    if "contact_reduction" in s:
        s["contact_reduction"] = 1
    if "groups" in s:
        s["groups"] = groups
    return s


def set_solref(sim, solref):
    """Every contact's solref: Lodos's pairs' (its groups read the same rows) and MJWarp's geoms' and pairs'."""
    r = np.asarray(solref, np.float32)
    lc = sim.solver.lc
    if lc is not None:
        lc.cpr_solref.assign(np.tile(r, (lc.cpr_solref.shape[0], 1)))
    for name in ("geom_solref", "pair_solref"):
        a = getattr(sim.m, name, None)
        if a is not None and a.size:
            a.assign(np.ascontiguousarray(np.broadcast_to(r, a.numpy().shape)))


def build(source, groups, B, sizes=None, variant="default", qpos=None, qvel=None):
    caps = dict(sizes["caps"]) if sizes else None
    col = dict(sizes["collision"]) if sizes else None
    sim = LSET.build(source, setting(source, groups), "mj", B=B, caps=caps, collision=col,
                     solver=dict(C.VARIANTS[variant]), qpos=qpos, qvel=qvel)
    set_solref(sim, C.SOLREF[source.name])
    return sim


class Ticks:
    """A scene's recorded controls on the device, each tick's row written to every world by a kernel."""

    def __init__(self, source, sim, n):
        c = C.controls(source.controls, n)
        self.table = None if c is None else wp.array(c[0].astype(np.float32), dtype=wp.float32)
        self.period = None if c is None else c[1]
        self.sim, self.k = sim, None

    def at(self, k):
        if self.table is not None and k != self.k:
            k = min(k, self.table.shape[0] - 1)
            wp.launch(_row, dim=self.sim.drbd.ctrl.shape, inputs=[self.table, int(k)], outputs=[self.sim.drbd.ctrl])
            self.k = k

    def step(self, i, dt):
        if self.table is not None:
            self.at(C.tick(i, dt, self.period))


def free_joints(m):
    return [int(m.jnt_qposadr[j]) for j in range(m.njnt) if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)]


# -------------------------------------------------------------------------------------------------------------------


def measure(source, groups, out):
    """`common.N_TUNE` starts, each alone for the whole run: the peaks, the capacities, the half-way states."""
    dt = float(setting(source, groups)["timestep"])
    n = C.steps(source.name, dt)
    q, v = C.starts(source, C.N_TUNE, 0)
    caps = dict(source.caps)
    for attempt in range(6):
        sim = build(source, groups, 1, sizes=dict(caps=caps, collision={}))
        tr = lcalibrate.Tracker(sim)
        replay = tr.capture()
        tr.reset()
        ticks = Ticks(source, sim, n + 1)
        mid_q, mid_v = [], []
        t0 = time.perf_counter()
        for s in range(C.N_TUNE):
            LSET.start(sim, q[s:s + 1], v[s:s + 1], [0.0])
            ticks.k = None
            for i in range(n):
                ticks.step(i, dt)
                replay()
                if i == n // 2:
                    mid_q.append(sim.drbd.qpos.numpy()[0].astype(np.float64))
                    mid_v.append(sim.drbd.qvel.numpy()[0].astype(np.float64))
        rep = tr.report()
        if not rep["overflowed"]:
            break
        over = lcalibrate.which_overflowed(rep)
        for k in ("nconmax", "njmax", "nccdmax"):
            if (k == "nconmax" and "naconmax" in over) or k in over:
                caps[k] = 2 * int(caps.get(k) or 1024)
        if not any(k in over for k in ("naconmax", "njmax", "nccdmax")):
            raise RuntimeError(f"Lodos's arenas overflowed at their worst case: {over}")
        del sim, tr, replay
    else:
        raise RuntimeError(f"{source.name}: the capacities kept overflowing")
    peaks = LSET.peaks(rep)
    sizes = LSET.size(peaks, C.OVERHEAD)
    np.savez(str(out).replace(".json", ".mid.npz"), qpos=np.stack(mid_q), qvel=np.stack(mid_v),
             time=np.full(len(mid_q), (n // 2 + 1) * dt))
    return dict(peaks={k: p for k, p in peaks.items() if not k.startswith("_")}, sizes=sizes, steps=n, dt=dt,
                seconds=time.perf_counter() - t0, attempts=attempt + 1, measured_caps=caps)


def out_of_memory(e):
    return any(s in str(e) for s in ("Failed to allocate", "out of memory", "cudaErrorMemory"))


def ab(source, groups, sizes, mid, B, reps=3, nsteps=100, warmup=10):
    """
    Each execution option's world-steps/s at `B` worlds from the half-way states, its median of `reps`; at half as
    many worlds while the default runs out of memory.
    """
    while True:
        out = variants(source, groups, sizes, mid, B, reps, nsteps, warmup)
        if "error" not in out["default"] or not out_of_memory(out["default"]["error"]) or B <= C.B0:
            break
        B //= 2
    base = out["default"].get("world_steps_per_s")
    if not base:
        raise RuntimeError(f"the default failed: {out['default']['error']}")
    best = max((n for n in out if "world_steps_per_s" in out[n]), key=lambda n: out[n]["world_steps_per_s"])
    chosen = best if out[best]["world_steps_per_s"] > (1.0 + C.MIN_GAIN) * base else "default"
    return dict(B=B, variants=out, chosen=chosen)


def variants(source, groups, sizes, mid, B, reps, nsteps, warmup):
    k = np.arange(B) % len(mid["qpos"])
    q, v, t = mid["qpos"][k], mid["qvel"][k], mid["time"][k]
    out = {}
    for name in C.VARIANTS:
        sim = replay = ticks = None
        gc.collect()  # the variant before, gone before this one is built
        try:
            sim = build(source, groups, B, sizes=sizes, variant=name)
            replay = lcalibrate.Tracker(sim).capture()
            dt = float(sim.mjm.opt.timestep)
            ticks = Ticks(source, sim, C.steps(source.name, dt) + 1)
            took = []
            for r in range(reps + 1):
                LSET.start(sim, q, v, t)
                ticks.k = None
                ticks.step(int(round(float(t[0]) / dt)), dt)
                wp.synchronize()
                t0 = time.perf_counter()
                for _ in range(warmup if r == 0 else nsteps):
                    replay()
                wp.synchronize()
                if r:
                    took.append(time.perf_counter() - t0)
            out[name] = dict(world_steps_per_s=B * nsteps / float(np.median(took)))
        except Exception as e:  # noqa: BLE001 -- the fused solve refuses a model too large for it
            out[name] = dict(error=f"{type(e).__name__}: {e}"[:300])
        sim = replay = ticks = None
        gc.collect()
        wp.synchronize()
    return out


def point(source, groups, sizes, variant, B, starts):
    """B worlds from their own starts for the whole run, timed; its demand, overflows and lost worlds."""
    z = np.load(starts)
    q, v = z["qpos"][:B], z["qvel"][:B]
    t_build = time.perf_counter()
    sim = build(source, groups, B, sizes=sizes, variant=variant, qpos=q, qvel=v)
    tr = lcalibrate.Tracker(sim)
    replay = tr.capture()  # one step, loading and capturing everything
    for _ in range(2):
        replay()
    wp.synchronize()
    build_s = time.perf_counter() - t_build
    dt = float(sim.mjm.opt.timestep)
    n = C.steps(source.name, dt)
    ticks = Ticks(source, sim, n + 1)
    LSET.start(sim, q, v, np.zeros(B))
    tr.reset()
    ticks.at(0)
    wp.synchronize()
    t0 = time.perf_counter()
    for i in range(n):
        ticks.step(i, dt)
        replay()
        if (i + 1) % max(1, n // 10) == 0:  # when each tenth was queued, a stall's trace in the log
            print(f"step {i + 1}: {time.perf_counter() - t0:.1f} s", flush=True)
    wp.synchronize()
    wall = time.perf_counter() - t0
    rep = tr.report()
    qe = sim.drbd.qpos.numpy().astype(np.float64)
    ve = sim.drbd.qvel.numpy().astype(np.float64)
    nan = ~(np.isfinite(qe).all(1) & np.isfinite(ve).all(1))
    esc = C.escaped(qe, q, free_joints(source.mjm)) & ~nan
    used = {k: getattr(wp, f"get_mempool_used_mem_{k}", lambda *a: None)("cuda:0") for k in ("high",)}
    return dict(B=B, steps=n, dt=dt, seconds=wall, world_steps_per_s=B * n / wall, sim_seconds_per_s=B * n * dt / wall,
                build_seconds=build_s, overflowed=bool(rep["overflowed"]), overflow=lcalibrate.which_overflowed(rep),
                peaks=dict(runtime=rep["runtime"], lodos_col=rep.get("lodos_col")), nan_worlds=int(nan.sum()),
                escaped_worlds=int(esc.sum()), variant=variant, sizes=sizes, final_qpos0=qe[0].tolist(),
                mempool_high_bytes=used.get("high"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=("measure", "ab", "point"))
    ap.add_argument("scene")
    ap.add_argument("groups", choices=C.GROUPS)
    ap.add_argument("out")
    ap.add_argument("--sizes")
    ap.add_argument("--B", type=int)
    ap.add_argument("--variant", default="default")
    ap.add_argument("--starts")
    a = ap.parse_args()
    t0 = time.perf_counter()
    res = dict(task=a.task, scene=a.scene, sim="lodos", groups=a.groups)
    try:
        source = LS.builtin(a.scene)
        sizes = C.read_json(a.sizes)["sizes"] if a.sizes else None
        if a.task == "measure":
            res.update(measure(source, a.groups, a.out))
        elif a.task == "ab":
            mid = dict(np.load(a.sizes.replace(".json", ".mid.npz")))
            res.update(ab(source, a.groups, sizes, mid, a.B or C.ab_batch(a.scene)))
        else:
            res.update(point(source, a.groups, sizes, a.variant, a.B, a.starts))
        res["ok"] = True
    except Exception as e:  # noqa: BLE001 -- recorded, the driver decides
        traceback.print_exc()
        res.update(ok=False, error=f"{type(e).__name__}: {e}"[:500])
    res["process_seconds"] = time.perf_counter() - t0
    C.write_json(a.out, res)


if __name__ == "__main__":
    main()
