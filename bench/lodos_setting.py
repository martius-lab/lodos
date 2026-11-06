"""
How the benchmarks set Lodos up (`bench.lodos_mjwarp`, `bench.lodos_kamino`): a scene's physics setting at the
viewer's defaults, a build of it, the capacities a world needs from its peaks, and worlds started from states.

A *setting* maps the physics options a scene has to values. `default_setting` reads a scene's own, as the viewer
builds it (`viewer.session.default_config`); `build` makes a `lodos.sim.Sim` of a setting, each option where a build
takes it. `peaks` and `size` turn one world's demands, tracked alone (`lodos.utils.calibrate.Tracker`), into
capacities that hold at any B: MJWarp's per world, Lodos's arenas as fractions of a world's worst case.
"""

from __future__ import annotations

import dataclasses
import math

import mujoco
import numpy as np

from lodos import sim as lsim
from lodos.rbd import io as rbd_io
from viewer import pairs as VP

# Each physics option a scene may have: where a build takes it -- "solver" (`solver_kwargs`, MuJoCo's own options
# among them, written on the build's copy of the model), "sim" (`sim_options`), "override" (an MJWarp model
# override) or "sided" (the sidedness of every Lodos pair without a plane) -- and which scenes have it: "all",
# "exact" (the exact mode's proximal loop), "lodos" (Lodos pairs) or "convex" (MJWarp's convex pairs).
PHYSICS = {
    "timestep": ("sim", "all"), "integrator": ("sim", "all"), "iterations": ("solver", "all"),
    "tolerance": ("solver", "all"), "ls_iterations": ("solver", "all"), "ls_tolerance": ("solver", "all"),
    "prox_iters_per_qp": ("solver", "exact"), "prox_tol_z": ("solver", "exact"),
    "prox_tol_sigma": ("solver", "exact"), "prox_rho": ("solver", "exact"),
    "refresh_de_saxce_per_prox": ("solver", "exact"), "warm_start": ("solver", "all"),
    "contact_reduction": ("solver", "lodos"), "groups": ("solver", "lodos"), "normal_bins": ("solver", "lodos"),
    "moment_bins": ("solver", "lodos"), "spline_type": ("sim", "lodos"), "sided": ("sided", "lodos"),
    "ccd_iterations": ("override", "convex"), "ccd_tolerance": ("override", "convex"), "hybrid": ("sim", "convex"),
}
NJMAX_FLOOR = 16


def _convex(source):
    """Whether MJWarp collides convex pieces (MJCF meshes) in `source`: the options of its convex narrowphase."""
    m = source.mjm
    keyed = set() if source.surfaces is None else {int(g) for g in source.surfaces.geoms}
    mesh = np.asarray(m.geom_type) == int(mujoco.mjtGeom.mjGEOM_MESH)
    collides = (np.asarray(m.geom_contype) | np.asarray(m.geom_conaffinity)) != 0
    return any(mesh[g] and collides[g] and g not in keyed for g in range(int(m.ngeom)))


def has(source, name, mode):
    """Whether `source` has the physics option `name` under `mode`: one a scene does not have is not in its setting."""
    applies = PHYSICS[name][1]
    if applies == "exact":
        return mode == "exact"
    if applies == "lodos":
        s = source.surfaces
        if s is None or not len(s.pairs):
            return False
        return name != "sided" or bool((~VP.plane_pairs(s)).any())
    if applies == "convex":
        return _convex(source)
    return True


def default_setting(source, mode):
    """The physics options of `source` at the values it builds with under `mode`, the viewer's own."""
    from viewer.session import default_config, edit_config

    cfg = default_config(source, "lodos")
    if cfg["options"]["mode"] != mode:
        cfg = edit_config(source, cfg, "mode", mode)
    o, out = cfg["options"], {}
    for name in PHYSICS:
        if not has(source, name, mode):
            continue
        if name in o:
            out[name] = o[name]
        elif name == "sided":
            free = ~VP.plane_pairs(source.surfaces)
            out["sided"] = 2 if (VP.scene_sided(source.surfaces)[free] == 2).all() else 1
        elif name in ("ccd_iterations", "ccd_tolerance"):
            out[name] = type(getattr(source.mjm.opt, name))(getattr(source.mjm.opt, name))
        elif name == "hybrid":
            out["hybrid"] = "on"
    return out


def build(source, setting, mode, B=1, caps=None, qpos=None, qvel=None, collision=None, solver=None):
    """
    `B` worlds of `source` at the physics `setting` under `mode` (`SceneSource.make_sim`), the rest the scene's:
    `caps` MJWarp's per-world capacities, `collision` Lodos's (`make_collision_context`'s fractions and tiles) over
    the scene's, `solver` further solver options (an execution variant's, the Baumgarte bias). MJWarp's warnings
    that a solve ran to its iteration caps are off, as in every run the benchmarks recorded.
    """
    if collision and source.surfaces is not None:
        source = dataclasses.replace(source, surfaces=dataclasses.replace(
            source.surfaces, col_kwargs={**source.surfaces.col_kwargs, **collision}))
    kw, sim_options, sided = {"mode": mode, **(solver or {})}, {}, None
    overrides = ["opt.warn_overflow=~ITERATIONS|~LS_ITERATIONS"]
    for name, v in setting.items():
        if name not in PHYSICS:
            raise ValueError(f"{name} is not a physics option")
        where = PHYSICS[name][0]
        if where == "solver":
            kw[name] = v
        elif where == "sim":
            sim_options[name] = (lsim.ALL_TYPES if v == "on" else ()) if name == "hybrid" else v
        elif where == "override":
            overrides.append(f"opt.{name}={v}")
        elif where == "sided":
            sided = VP.sided_changes(source.surfaces, {"all": {"sided": v}, "sided": {}})
    if kw.get("groups", "contact") == "contact":  # the bins bin contact groups: without groups there is nothing
        for k in ("groups", "normal_bins", "moment_bins"):
            kw.pop(k, None)
    return source.make_sim(solver_kwargs=kw, overrides=tuple(overrides), caps=caps, B=B, qpos=qpos, qvel=qvel,
                           sim_options=sim_options, sided=sided)


def start(sim, qpos, qvel, times):
    """Loads each world's state and time, MJWarp's warm start and the proximal loop's carried values forgotten."""
    rbd_io.set_state(sim.drbd, qpos=qpos, qvel=qvel)
    sim.drbd.time.assign(np.asarray(times, np.float32))
    d = sim.d
    d.qacc_warmstart.zero_()
    if d.prox_zcache_n.shape[0]:
        d.prox_zcache_n.zero_()
    if sim.solver.groups is not None:
        sim.solver.groups.forget()


def peaks(rep):
    """A tracker report's demands, one world's: `{capacity: peak}`, and under "_tiles" the shapes `size` needs."""
    m = rep["runtime"]
    out = dict(nconmax=max(m["nacon"], m.get("ncollision", 0)), njmax=m["nefc_per_world"])
    ccd = m.get("ccd")
    if ccd is not None:
        out["nccdmax"] = max(ccd["epa"], ccd["manifold"], math.ceil(ccd["contacts"] / 4))
    col = rep.get("lodos_col")
    if col is not None:
        out.update({k: col["peak"][k] for k in ("nTa", "nTb", "nCa2")})
        out["_tiles"] = dict(slots=col["tile_slots"], per_world=col["slots_per_world"], uniform=col["uniform_K"],
                             K=col["tile_K"], compacts=col["compacts"])
    return out


def size(peaks, overhead):
    """
    Capacities from one world's `peaks`, each its peak times 1 + `overhead`: `{"caps": MJWarp's per world,
    "collision": Lodos's fractions}`.
    """
    f = 1.0 + overhead
    caps = dict(nconmax=max(1, math.ceil(f * peaks["nconmax"])),
                njmax=max(NJMAX_FLOOR, math.ceil(f * peaks["njmax"])))
    if "nccdmax" in peaks:
        caps["nccdmax"] = min(caps["nconmax"], max(1, math.ceil(f * peaks["nccdmax"])))
    out = {"caps": caps, "collision": {}}
    t = peaks.get("_tiles")
    if t is not None:
        worst = max(1, t["slots"])
        cap_t = min(1.0, f * max(1, peaks["nTa"]) / worst)
        nca1 = max(1, math.ceil(cap_t * worst)) * t["K"] if t["uniform"] else max(1, math.ceil(cap_t * t["per_world"]))
        col = dict(capacity_tile=cap_t, capacity2=min(1.0, f * max(1, peaks["nCa2"]) / nca1))
        if t["compacts"]:
            col["capacity1"] = min(1.0, f * max(1, peaks["nTb"]) / worst)
        out["collision"] = col
    return out
