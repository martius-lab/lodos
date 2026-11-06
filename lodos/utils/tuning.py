"""
A scene's throughput settings, measured once and kept in a file.

Every capacity of a `lodos.sim.Sim` is fixed when it is built -- CUDA graphs need stable
addresses -- and every one of them is a speed or a memory trade: too small drops contacts,
too large launches idle threads and holds memory a larger batch could use. The other
throughput knobs (tile sizes, the solver's budgets, contact reduction) are per scene too.
A tuning file holds all of them, measured by `tools/performance_tuning/` for one scene, with how
they were measured:

    sim = make_sim(mjm, B, surfaces=surf, tuning="scene.tuning.json")

Arguments given to `make_sim` explicitly win over the file; the file wins over a scene's
built-in defaults. `apply` says exactly what is taken from it.

## The file

    {
      "format": "lodos-tuning/2",
      "scene": "...",                        what it was measured on
      "B": 4096, "layout": "independent",    the batch it was measured at, and how its
                                             worlds were assigned to states
      "fingerprint": {...},                  the model and every asset (`scene_fingerprint`)
      "states": {...},                       the state set: path, digest, size, horizon,
                                             controls (`lodos.utils.states`)
      "safety": 1.5, "created": "...", "commit": "...", "device": "...",
      "collision":     {...},                make_collision_context: capacity_tile, capacity2,
                                             capacity1, tile_size, super_size
      "runtime":       {...},                naconmax, nconmax, njmax, nccdmax, njmax_nnz
      "sim":           {...},                contact_reduction
      "solver_kwargs": {...},                any budget the tuning changed (the solver's
                                             iterations, ...)
      "peaks":         {...},                the calibration report the capacities came from
      "throughput":    {...},                world-steps/s measured with these settings
      "baseline":      {...}                 the settings before tuning, for the record
    }

## Batch size

The collision capacities and `naconmax` are *batch totals*, fractions of
what `B` worlds could need, and a batch's total is closer to `B` times the mean load the
larger it is (`lodos.utils.states`, layouts). So a file is applied whole only at the `B` it was
measured at or above; below it, the batch totals are left out (and `apply` says so) and
only the per-world settings and the knobs that are not capacities are taken. The live
viewer runs one world, and takes the file that way.

## Staleness

Capacities follow the geometry: a re-baked surface with twice the vertices needs twice the
stage-1 arena. The file records a fingerprint of the model and of every asset --
per Lodos surface its vertex and face counts, SDF grid, spline type, inner spheres and a
hash of the arrays, plus the mesh pipeline's records when they sit beside the bake; per
convex-hull set its hull and vertex counts and a hash. `stale(file, mjm, surfaces)` lists
what changed; `make_sim` warns on a stale file and `tools/performance_tuning/calibrate.py`
re-measures it.
"""

from __future__ import annotations

import copy
import datetime
import hashlib
import json
import math
import subprocess
import warnings
import weakref
from pathlib import Path

import mujoco
import numpy as np

from lodos.utils.states import _sha, model_signature

FORMAT = "lodos-tuning/2"  # the format every tuning file names; `load` refuses any other
SECTIONS = ("collision", "runtime", "sim", "solver_kwargs")  # a file's sections of settings, which `make_sim` takes
# The settings that are batch totals, and so only valid at the `B` they were measured at or above.
# `nconmax` and `nccdmax` are per world in name only: MJWarp allocates them times `B` and lets
# any world use any of it, so what a calibration gives them is a share of a batch total.
BATCH_TOTALS = {"collision": ("capacity_tile", "capacity2", "capacity1"), "runtime": ("naconmax", "nconmax", "nccdmax")}
# The settings that are capacities -- sizes a calibration measures -- as against the knobs
# that choose how the work is done (tile sizes, budgets, contact reduction).
CAPACITIES = {"collision": ("capacity_tile", "capacity2", "capacity1"),
              "runtime": ("naconmax", "nconmax", "njmax", "nccdmax", "njmax_nnz")}
# The batch totals that are absolute counts rather than fractions of the batch.
ABSOLUTE_TOTALS = ("naconmax",)


#######################################################################################
# Fingerprints
#######################################################################################


# Each surface's fingerprint, by the surface: a scene holds one surface many times over, and
# every build checks the fingerprint. Weak, so that a freed surface's id is never mistaken
# for a new one's.
_SURF_FP = weakref.WeakKeyDictionary()


def surface_fingerprint(surf) -> dict:
    """
    How a Lodos surface was made, as far as the bake says: its vertex and face counts, its
    SDF grid (`N` cells of `h`), the spline it is evaluated with, its inner spheres, a hash
    of the arrays the collision pipeline reads, and -- when it was loaded from a directory
    that holds them -- the mesh pipeline's per-step records (`*.json` beside the bake).

    Called by `scene_fingerprint` and `tools.performance_tuning.tune.tile_groups`.
    In: surf, a baked surface (`lodos.col.sdf.load_surface_file`) or a `PlaneSurface`: v (nv, 3) float64, f (nf, 3)
    int64, N (3,) int32, h (3,) float32, sdf_coeffs (N[0] + 3, N[1] + 3, N[2] + 3) float32, sph_r (nsph,) float64.
    Out: dict: "kind", and for a surface "vertices", "faces", "grid_N", "voxel_h", "spline_type", "spheres", "extent"
    and "sha", with "file" and "records" when it has a source; a copy of the one cached in `_SURF_FP`.
    """
    from lodos.col.sdf import _is_plane

    if _is_plane(surf):
        return dict(kind="plane")
    # hashed once per surface and arrays: a rescaled surface has new arrays, and so a new key
    key = (id(surf.v), id(surf.sdf_coeffs), id(getattr(surf, "sph_r", None)))  # (int, int, int): the arrays' ids
    hit = _SURF_FP.get(surf)  # (key, dict) or None: the cached fingerprint
    if hit is not None and hit[0] == key:
        return copy.deepcopy(hit[1])
    v, f = np.asarray(surf.v), np.asarray(surf.f)  # (nv, 3) float64, (nf, 3) int64: the vertices and faces
    out = dict(  # dict: the fingerprint
        kind="surface",
        vertices=int(len(v)), faces=int(len(f)),
        grid_N=[int(x) for x in np.asarray(surf.N).reshape(-1)],
        voxel_h=[float(x) for x in np.asarray(surf.h, dtype=np.float64).reshape(-1)],
        spline_type=str(np.asarray(getattr(surf, "spline_type", "cubic")).item()),
        spheres=int(len(getattr(surf, "sph_r", ()))),
        extent=[float(x) for x in (v.max(0) - v.min(0))] if len(v) else [0.0, 0.0, 0.0],
        sha=_sha(v.astype(np.float32), f, np.asarray(surf.sdf_coeffs, dtype=np.float32),
                 np.asarray(getattr(surf, "sph_r", np.zeros(0)), dtype=np.float32)),
    )
    src = getattr(surf, "source", None)  # str or None: the bake's path
    if src:
        out["file"] = _portable(src)
        records = _records(Path(src))  # dict: the mesh pipeline's records of this body
        if records:
            out["records"] = records
    _SURF_FP[surf] = (key, copy.deepcopy(out))
    return out


def _portable(path) -> str:
    """
    `path` relative to the repository when it is inside it, so that a file reads the same from any clone.

    Called by `surface_fingerprint`. In: path str or Path. Out: str.
    """
    ROOT = Path(__file__).resolve().parents[2]  # Path: the repository's root

    p = Path(path).resolve()  # Path: `path`, absolute
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def _records(path):
    """
    This surface's rows of the mesh pipeline's records beside its bake: every step writes
    a list of per-body rows (`tools/mesh_processing/`; the bake step gathers the earlier steps'
    into its own directory). The body is the bake's name -- or, for a surface installed as
    `<name>/surface_data.npz`, its directory's.

    Called by `surface_fingerprint`. In: path Path, the bake's `.npz`.
    Out: dict: a record file's stem -> the body's row (dict) or rows (list), or the whole record when it is a dict of
    under 4096 characters of JSON.
    """
    body = path.parent.name if path.stem == "surface_data" else path.stem  # str: the body's name
    out = {}  # dict: record stem -> rows
    for p in sorted(path.parent.glob("*.json")):  # p: Path, a record beside the bake
        try:
            data = json.loads(p.read_text())  # list or dict: the record
        except (OSError, ValueError):
            continue
        if isinstance(data, list):
            rows = [r for r in data if isinstance(r, dict) and r.get("body") == body]  # list of dict: the body's rows
            if rows:
                out[p.stem] = rows[0] if len(rows) == 1 else rows
        elif isinstance(data, dict) and len(json.dumps(data)) < 4096:
            out[p.stem] = data
    return out


def hull_fingerprint(mjm) -> dict:
    """
    The MJCF's collision meshes: hulls per body, their vertex counts, and a hash of the vertices.

    Called by `scene_fingerprint`.
    In: mjm mujoco.MjModel. Out: dict: "hulls" int, and when there are any, "vertices", "max_vertices", "bodies" and
    "hulls_per_body_max" int and "sha" str.
    """
    mesh_geoms = [g for g in range(mjm.ngeom)  # list of int: the mesh geoms that collide
                  if mjm.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and (mjm.geom_contype[g] or mjm.geom_conaffinity[g])]
    if not mesh_geoms:
        return dict(hulls=0)
    per_body = {}  # dict: body name -> its hulls
    nvert = []  # list of int: each hull's vertex count
    verts = []  # list of (n, 3) float32: each hull's vertices
    for g in mesh_geoms:  # g: int, a mesh geom
        k = int(mjm.geom_dataid[g])  # int: its mesh
        n = int(mjm.mesh_vertnum[k])  # int: the mesh's vertex count
        a = int(mjm.mesh_vertadr[k])  # int: the mesh's first vertex
        nvert.append(n)
        verts.append(np.asarray(mjm.mesh_vert[a:a + n], dtype=np.float32))
        # str: the geom's body's name, or its id
        b = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_BODY, int(mjm.geom_bodyid[g])) or str(int(mjm.geom_bodyid[g]))
        per_body[b] = per_body.get(b, 0) + 1
    return dict(hulls=len(mesh_geoms), vertices=int(sum(nvert)), max_vertices=int(max(nvert)),
                bodies=len(per_body), hulls_per_body_max=int(max(per_body.values())),
                sha=_sha(*verts))


def scene_fingerprint(mjm, surfaces=None) -> dict:
    """
    What a tuning file's numbers depend on: the model (`lodos.utils.states.model_signature`), its
    collision geometry and timestep, its convex hulls and every Lodos surface.

    Called by `stale`, `tools.performance_tuning.common._fingerprint`, `tools.performance_tuning.calibrate.main` and
    `tools.performance_tuning.tune.main`.
    In: mjm mujoco.MjModel; surfaces, a `lodos.sim.Surfaces` or None.
    Out: dict: "model" (`model_signature`), "geometry" str, "timestep" float (s), "hulls" (`hull_fingerprint`), and
    with surfaces "surfaces" (a `surface_fingerprint` each) and "pairs" int.
    """
    # str: a hash of the geoms' types, sizes, bodies, contact bits and positions
    geometry = _sha(np.asarray(mjm.geom_type), np.asarray(mjm.geom_size, dtype=np.float32),
                    np.asarray(mjm.geom_bodyid), np.asarray(mjm.geom_contype), np.asarray(mjm.geom_conaffinity),
                    np.asarray(mjm.geom_pos, dtype=np.float32))
    out = dict(model=model_signature(mjm), geometry=geometry, timestep=float(mjm.opt.timestep),  # dict
               hulls=hull_fingerprint(mjm))
    if surfaces is not None:
        out["surfaces"] = [surface_fingerprint(s) for s in surfaces.surfaces]
        out["pairs"] = int(len(surfaces.pairs))
    return out


def _strip(fp):
    """
    A fingerprint without what does not decide a capacity (file paths, the records' wording).

    Called by `stale`. In: fp dict, a `scene_fingerprint`. Out: dict, a copy.
    """
    fp = copy.deepcopy(fp)
    for s in fp.get("surfaces", []):  # s: dict, a surface's fingerprint
        s.pop("file", None)
        s.pop("records", None)
    return fp


def stale(tuning, mjm, surfaces=None) -> list:
    """
    What differs between the scene the file was measured on and this one; empty if nothing does.

    Called by `merge_into_make_sim`, `viewer.performance.Performance.load_tuning` and
    `tools.performance_tuning.tune.main`.
    In: tuning, a path or a loaded dict; mjm mujoco.MjModel; surfaces, a `lodos.sim.Surfaces` or None.
    Out: list of str, a line per difference.
    """
    t = load(tuning)  # dict: the file
    # dict, dict: the file's fingerprint and this scene's
    then, now = _strip(t.get("fingerprint", {})), _strip(scene_fingerprint(mjm, surfaces))
    out = []  # list of str
    for k in sorted(set(then) | set(now)):  # k: str, a fingerprint entry
        if k == "surfaces":
            a, b = then.get(k, []), now.get(k, [])  # list, list: the surfaces' fingerprints then and now
            if len(a) != len(b):
                out.append(f"surfaces: {len(a)} then, {len(b)} now")
                continue
            for i, (x, y) in enumerate(zip(a, b)):  # i: int, a surface; x, y: dict, its fingerprints
                if x != y:
                    diff = sorted(j for j in set(x) | set(y) if x.get(j) != y.get(j))  # list of str: what differs
                    out.append(f"surface {i}: {', '.join(diff)}")
        elif then.get(k) != now.get(k):
            out.append(f"{k}: {then.get(k)} then, {now.get(k)} now")
    return out


#######################################################################################
# The file
#######################################################################################


def new(scene, B, layout, fingerprint, states=None, safety=None, **extra) -> dict:
    """
    An empty tuning file for `scene`, measured at `B` under `layout`.

    Called by `tools.performance_tuning.calibrate.main`, `tools.performance_tuning.common.as_tuning` and
    `tools.performance_tuning.tune.main`.
    In: scene str; B int; layout str (`lodos.utils.states.LAYOUTS`); fingerprint dict (`scene_fingerprint`); states
    dict or None, the state set's record; safety float or None; extra, more top-level entries.
    Out: dict, its four sections empty and "created", "commit" and "device" filled in.
    """
    return dict(format=FORMAT, scene=scene, B=int(B), layout=layout, fingerprint=fingerprint,
                states=states or {}, safety=safety, created=datetime.datetime.now().isoformat(timespec="seconds"),
                commit=_commit(), device=_device(), **{k: {} for k in SECTIONS}, **extra)


def _old_sections(t, path="") -> set:
    """
    The sections named "mjwarp" (capacities) or "mjw" (peaks), which this format calls "runtime".

    Called by `load`, and by itself on every nested dict.
    In: t, a loaded file or a part of it; path str, the keys above `t`, dotted. Out: set of str, the sections' paths.
    """
    out = set()  # set of str
    if isinstance(t, dict):
        for k, v in t.items():  # k: str, an entry; v: its value
            if k in ("mjwarp", "mjw") and isinstance(v, dict):
                out.add(f'{path}"{k}"')
            out |= _old_sections(v, f"{path}{k}.")
    return out


def load(tuning) -> dict:
    """
    A tuning file: a path, or an already loaded dict (returned as is).

    Raises `ValueError` on another format, or on a file that still uses the sections' old names. Called by `stale`,
    `apply`, `viewer.performance.Performance.load_tuning` and 7 others.
    In: tuning str, Path or dict. Out: dict, its per-pair tile sizes keyed by `(vertex geom, SDF geom)` tuples, None
    the default.
    """
    if isinstance(tuning, dict):
        return tuning
    t = json.loads(Path(tuning).read_text())  # dict: the file
    if t.get("format") != FORMAT:
        raise ValueError(f"{tuning}: not a tuning file of {FORMAT!r} ({t.get('format')!r})")
    old = sorted(_old_sections(t))  # list of str: the old sections' paths
    if old:
        raise ValueError(f"{tuning}: in an older format, with {', '.join(old)} (now \"runtime\"); migrate it: "
                         f"python tools/performance_tuning/migrate_formats.py {tuning}")
    col = t.get("collision", {})  # dict: the collision section
    for k in ("tile_size", "super_size"):  # k: str, a per-pair setting
        if isinstance(col.get(k), dict):
            col[k] = _keys_in(col[k])
    return t


def save(tuning, path):
    """
    Writes a tuning file; per-pair tile sizes keyed by `(vertex geom, SDF geom)` are written as "v,s".

    Called by `tools.performance_tuning.calibrate.main`, `tools.performance_tuning.tune.main` and
    `tools.performance_tuning.tile_tuner.main`.
    In: tuning dict; path str or Path. Out: Path, the file written.
    """
    t = copy.deepcopy(tuning)  # dict: the file, its keys made JSON's
    col = t.get("collision", {})  # dict: the collision section
    for k in ("tile_size", "super_size"):  # k: str, a per-pair setting
        if isinstance(col.get(k), dict):
            col[k] = _keys_out(col[k])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(t, indent=1, default=_json_default) + "\n")
    return path


def _keys_out(d):
    """
    A per-pair setting's keys as JSON's strings: `None`, the default, as "default", and `(v, s)` as "v,s".

    Called by `save`. In: d dict, `(vertex geom, SDF geom)` or None -> value. Out: dict, str -> value.
    """
    return {("default" if k is None else f"{k[0]},{k[1]}"): v for k, v in d.items()}


def _keys_in(d):
    """
    `_keys_out` undone: "default" as `None`, "v,s" as `(v, s)`.

    Called by `load`. In: d dict, str -> value. Out: dict, `(int, int)` or None -> value.
    """
    out = {}  # dict: the keys read back
    for k, v in d.items():  # k: str, a key; v: its value
        out[None if k == "default" else tuple(int(x) for x in k.split(","))] = v
    return out


def _json_default(o):
    """
    numpy scalars and arrays as JSON's numbers and lists; `TypeError` for anything else.

    Passed as `json.dumps`'s `default` by `save` and `tools.performance_tuning.common.write_json`.
    In: o, an object `json` cannot write. Out: int, float or list.
    """
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"{type(o)} is not JSON")


def _commit():
    """
    The repository's HEAD as a short hash, "+dirty" when tracked files differ from it; None when git cannot say.

    Called by `new`. Out: str or None.
    """
    try:
        root = Path(__file__).resolve().parents[2]  # Path: the repository's root
        # subprocess.CompletedProcess: `git rev-parse`'s
        r = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                           timeout=5)
        # str: the tracked files that differ from HEAD, empty when none do
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return r.stdout.strip() + ("+dirty" if dirty else "") if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _device():
    """
    The name of Warp's current device; None when Warp cannot say.

    Called by `new`. Out: str or None.
    """
    try:
        import warp as wp

        return str(wp.get_device().name)
    except Exception:  # noqa: BLE001
        return None


#######################################################################################
# Applying a file
#######################################################################################


def apply(tuning, B, quiet=False) -> dict:
    """
    What `make_sim` takes from a file at batch size `B`: a dict with the four sections,
    measured at the file's `B`:

    - at that `B`, they are taken whole;
    - above it, the absolute batch total (`naconmax`) is scaled by the
      ratio of the batch sizes -- the fractions (`capacity_tile`, `capacity2`,
      `capacity1`) already scale, and both are conservative there, since a larger batch
      of independently assigned worlds is closer to its mean;
    - below it, the batch totals are left out, with a warning: a smaller batch strays
      further from the mean than the one measured.

    Called by `merge_into_make_sim` and `viewer.scenes.catalog.SceneSource.make_sim`.
    In: tuning, a path or a loaded dict; B int; quiet bool, no warning. Out: dict: section -> {setting: value}.
    """
    t = load(tuning)  # dict: the file
    out = {k: {} for k in SECTIONS}  # dict: section -> {setting: value}
    B, B_sec = int(B), int(t.get("B", 0) or 0)  # int, int: the batch asked for, and the file's
    small = B < B_sec  # bool: below the file's batch
    dropped = []  # list of str: the batch totals left out
    for k in SECTIONS:  # k: str, a section
        for name, value in t.get(k, {}).items():  # name: str, a setting; value: its value
            if name in BATCH_TOTALS.get(k, ()):
                if small:
                    dropped.append(name)
                    continue
                if name in ABSOLUTE_TOTALS and B > B_sec > 0:
                    value = int(math.ceil(value * B / B_sec))
            out[k][name] = value
    if "naconmax" in out["runtime"]:
        out["runtime"]["nconmax"] = max(1, int(math.ceil(out["runtime"]["naconmax"] / B)))
        if "nccdmax" in out["runtime"]:
            out["runtime"]["nccdmax"] = min(out["runtime"]["nccdmax"], out["runtime"]["nconmax"])
    if dropped and not quiet:
        warnings.warn(f"the tuning file was measured at B={B_sec}; at B={B} its batch totals "
                      f"({', '.join(dropped)}) are left at the scene's own", stacklevel=3)
    return out


def merge_into_make_sim(tuning, B, surfaces, solver_kwargs, caps, sim_kwargs, mjm=None):
    """
    `make_sim`'s side of `apply`: fills what the caller left out. `caps` and `sim_kwargs`
    hold the caller's values, None where not given; `surfaces.col_kwargs` and
    `solver_kwargs` are filled where they do not name a key. Returns the new
    `(surfaces, solver_kwargs, caps, sim_kwargs)`; nothing passed in is changed.

    Called by `lodos.sim.make_sim`.
    In: tuning, a path or a loaded dict; B int; surfaces, a `lodos.sim.Surfaces` or None; solver_kwargs dict or None;
    caps dict: "nconmax", "njmax", "nccdmax", "naconmax" and "njmax_nnz", int or None; sim_kwargs dict:
    "contact_reduction", bool or None; mjm mujoco.MjModel or None, checked for staleness when given.
    Out: (Surfaces or None, dict, dict, dict).
    """
    import dataclasses

    if mjm is not None:
        diff = stale(tuning, mjm, surfaces)  # list of str: what changed since the file was measured
        if diff:
            warnings.warn("the tuning file is stale -- the scene changed since it was measured: "
                          + "; ".join(diff[:6]) + ("; ..." if len(diff) > 6 else ""), stacklevel=3)
    a = apply(tuning, B)  # dict: the file's settings at B
    caps = dict(caps)
    given = {k for k, v in caps.items() if v is not None}  # set of str: the capacities the caller gave
    for k, v in a["runtime"].items():  # k: str, a capacity; v: int, the file's
        if k in given or (k == "naconmax" and "nconmax" in given):
            continue  # a given `nconmax` sizes the batch's contacts; the file's `naconmax` would override it
        caps[k] = v
    if caps.get("nccdmax") is not None and caps.get("nconmax") is not None:
        caps["nccdmax"] = min(int(caps["nccdmax"]), int(caps["nconmax"]))
    sim_kwargs = dict(sim_kwargs)
    for k, v in a["sim"].items():
        if sim_kwargs.get(k) is None:
            sim_kwargs[k] = v
    solver_kwargs = dict(solver_kwargs or {})
    for k, v in a["solver_kwargs"].items():
        solver_kwargs.setdefault(k, v)
    if surfaces is not None and a["collision"]:
        col = dict(a["collision"])  # dict: the collision options, the caller's over the file's
        col.update(surfaces.col_kwargs)
        surfaces = dataclasses.replace(surfaces, col_kwargs=col)
    return surfaces, solver_kwargs, caps, sim_kwargs


def knobs_only(tuning) -> dict:
    """A copy of a file without its capacities: what a run at another scale (the live viewer's
    one world, which grows its own) takes from it.

    Called by `viewer.performance.Performance.apply_tuning`. In: tuning, a path or a loaded dict. Out: dict.
    """
    t = copy.deepcopy(load(tuning))  # dict: the file, copied
    for k, names in CAPACITIES.items():  # k: str, a section; names: tuple of str, its capacities
        for n in names:  # n: str, a capacity
            t.get(k, {}).pop(n, None)
    return t


def describe(tuning) -> str:
    """
    A short text summary of a file, for a log or a panel.

    Called by `tools.performance_tuning.calibrate.main` and `viewer.performance.Performance.ui`.
    In: tuning, a path or a loaded dict. Out: str, two lines: the provenance, then every setting.
    """
    t = load(tuning)  # dict: the file
    # list of str: the summary's lines
    lines = [f"{t.get('scene')}: B={t.get('B')} {t.get('layout')}, {t.get('states', {}).get('size', '?')} states, "
             f"safety {t.get('safety')}, {t.get('created')} @ {t.get('commit')}"]
    parts = []  # list of str: "setting=value" each, then the throughput
    for k in SECTIONS:  # k: str, a section
        for key, v in t.get(k, {}).items():  # key: str, a setting; v: its value
            parts.append(f"{key}={_short(v)}")
    thr = t.get("throughput", {})  # dict: the throughput measured
    if thr:
        parts.append(f"-> {thr.get('world_steps_per_s', 0):.3g} world-steps/s")
    lines.append("  " + " ".join(parts))
    return "\n".join(lines)


def _short(v):
    """
    A setting's value in a few characters: a float to 3 significant digits, a per-pair dict as "{key:value,...}"
    with `*` for the default's key.

    Called by `describe`. In: v, a setting's value. Out: str.
    """
    if isinstance(v, float):
        return f"{v:.3g}"
    if isinstance(v, dict):
        return "{" + ",".join(f"{'*' if k is None else k}:{x}" for k, x in v.items()) + "}"
    return str(v)


def digest(path) -> str:
    """
    A short hash of a file's bytes (a state set's, for the tuning file's provenance).

    Not called in the repository. In: path str or Path. Out: str, 16 hex digits.
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
