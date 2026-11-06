"""
    python -m viewer [SCENE] [--sim lodos] [--seed 0] [--stride 1] [--root DIR ...] [--tuned] [--controls FILE.npz]

SCENE is a built-in scene's name (`viewer.scenes.catalog.BUILTIN`) or an MJCF file;
without one the window opens empty, and a scene is loaded from its Scene panel. `--sim`
is the sim it starts under: Lodos, or Newton's MuJoCo Warp or Kamino (`mjwarp`, `kamino`)
where `viewer/newton/`'s environment is installed.
`--tuned` applies each scene's tuning file (`tools/performance_tuning/tuned/`) on every load
and sim switch, as the Performance panel's checkbox does. `--controls` plays
pre-recorded controls in SCENE from its initial state on: an `.npz` of `ctrl`,
`(T, nu)` for the scene's `nu` actuators, and `period`, the seconds each row holds
(`viewer.scenes.catalog.RecordedControls`).
"""

import argparse
import os

from viewer.scenes import catalog as LS
from viewer.session import SIMS, Session


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m viewer", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", nargs="?", help="a built-in scene or an MJCF file")
    ap.add_argument("--sim", default="lodos", choices=SIMS, help="the sim the scene starts under")
    ap.add_argument("--seed", type=int, default=0, help="a built-in scene's random stacks")
    ap.add_argument("--stride", type=int, default=1, help="record every n-th step")
    ap.add_argument("--root", action="append", help="where the MJCF search looks (repeatable)")
    ap.add_argument("--width", type=int, default=None, help="window width (default: 85%% of the screen)")
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--tuned", action="store_true", help="apply each scene's tuning file on every load and sim switch")
    ap.add_argument("--controls", help="pre-recorded controls for SCENE to play (an .npz of ctrl and period)")
    a = ap.parse_args(argv)
    if a.scene and a.scene not in LS.BUILTIN and not os.path.exists(a.scene):
        ap.error(f"{a.scene!r} is neither a built-in scene ({', '.join(LS.BUILTIN)}) nor a file")
    if a.controls and not a.scene:
        ap.error("--controls plays in a scene: name one")
    source = None
    if a.controls:
        source = LS.builtin(a.scene, a.seed) if a.scene in LS.BUILTIN else LS.from_mjcf(a.scene)
        try:
            source = LS.with_controls(source, LS.RecordedControls.load(a.controls))
        except (OSError, ValueError) as exc:
            ap.error(f"--controls {a.controls}: {exc}")

    from viewer.app import LiveViewer

    app = LiveViewer(Session(stride=a.stride), roots=a.root, width=a.width, height=a.height)
    app.sim0 = a.sim
    app.perf.auto = a.tuned
    if a.scene in LS.BUILTIN:
        app.builtin, app.seed = a.scene, a.seed
    if source is not None:
        app.load(source, a.sim)
    elif a.scene in LS.BUILTIN:
        app.load_builtin(a.scene, a.seed, a.sim)
    elif a.scene:
        app.load_file(a.scene, a.sim)
    app.run()


if __name__ == "__main__":
    main()
