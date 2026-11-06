"""
The files a scene's runs read, in a process of its own (the driver, `run.py`, holds no GPU):

    python -m bench.lodos_kamino.prepare SCENE [--starts N]

Its starts -- `common.N_TUNE` for the capacities, and the ladder's first N (extended when a larger batch needs
more) -- and on a scene Kamino runs (`common.configs`) its Newton package (`viewer.newton_link.scene_package`),
Kamino's options (the viewer's, its iterations `common.kamino_iterations`) and the recorded controls Newton plays
(`SceneSource.newton_controls` where the scene has them, else its controls); and what the report says of the scene.
"""

from __future__ import annotations

import argparse
import math
import pickle

import numpy as np

from bench.lodos_kamino import common as C
from bench.lodos_kamino import lodos_point as LP
from viewer import newton_link as NL
from viewer.scenes import catalog as LS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--starts", type=int, default=0)
    a = ap.parse_args()
    src = LS.builtin(a.scene)
    d = C.OUT / "scenes"
    d.mkdir(parents=True, exist_ok=True)
    tune = d / f"{a.scene}.tune.npz"
    if not tune.exists():
        q, v = C.starts(src, C.N_TUNE, 0)
        np.savez(tune, qpos=q, qvel=v)
    ladder = d / f"{a.scene}.starts.npz"
    have = len(np.load(ladder)["qpos"]) if ladder.exists() else 0
    if a.starts > have:
        q, v = C.starts(src, a.starts, C.FIRST)
        np.savez(ladder, qpos=q, qvel=v)
    configs = C.configs(src)
    pkg = d / f"{a.scene}.pkg"
    if "kamino" in configs and not pkg.exists():
        p = NL.scene_package(src, "kamino")
        p["name"] = a.scene
        pkg.write_bytes(pickle.dumps(p, protocol=5))
        C.write_json(d / f"{a.scene}.kamino.json", dict(NL.default_options(src, "kamino"),
                                                        max_iterations=C.kamino_iterations(a.scene)))
        c = src.newton_controls if src.newton_controls is not None else src.controls
        if c is not None:
            rows, period = C.controls(c, math.ceil(C.SCENES[a.scene] / c.period) + 2)
            np.savez(d / f"{a.scene}.controls.npz", rows=rows, period=period)
    info = d / f"{a.scene}.json"
    old = C.read_json(info) if info.exists() else {}
    if old.get("configs") != configs:  # a scene prepared before Kamino ran keeps the rest of its record
        s = LP.setting(src, "contact")
        C.write_json(info, dict(
            old, scene=a.scene, seconds=C.SCENES[a.scene], dt=s["timestep"], steps=C.steps(a.scene, s["timestep"]),
            configs=configs, lodos_setting=s, lodos_solver=C.BAUMGARTE,
            kamino_options=C.read_json(d / f"{a.scene}.kamino.json") if "kamino" in configs else None,
            nq=int(src.mjm.nq), nv=int(src.mjm.nv), nbody=int(src.mjm.nbody), ngeom=int(src.mjm.ngeom),
            surfaces=0 if src.surfaces is None else len(src.surfaces.surfaces),
            lodos_pairs=0 if src.surfaces is None else len(src.surfaces.pairs)))


if __name__ == "__main__":
    main()
