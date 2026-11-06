"""
Picks the tile size of each kind of Lodos segment by timing the collision pipeline alone.

    python tools/performance_tuning/tile_tuner.py arm_20_sdf --B 2048 [--states S.npz] \\
        [--tuning tools/performance_tuning/tuned/arm_20_sdf.tuning.json] [--write]

Segments are grouped by their (vertex surface, SDF surface) assets -- the twenty armadillos'
380 armadillo-armadillo segments are one group, their twenty against the floor another -- and
each group's size is searched over 4, 8, 16 and 32 with the others held, the largest group
first (one pass of coordinate descent). Each candidate is timed on the same frozen batch of
states at 1.5x its own demand, so that a size is not charged for an arena sized for another.
`--write` puts the sizes found into the tuning file; its capacities then need
`calibrate.py --update`, since the tiles change the demand.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibrate as CAL  # noqa: E402
import common as C  # noqa: E402
import tune as T  # noqa: E402

from lodos.utils import states as lstates  # noqa: E402
from lodos.utils import tuning as ltuning  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene")
    ap.add_argument("--B", type=int, default=1024)
    ap.add_argument("--states")
    ap.add_argument("--tuning")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    src = C.scene(a.scene)
    if src.surfaces is None:
        raise SystemExit(f"{a.scene} has no Lodos surfaces: nothing is tiled")
    sset = (lstates.load(a.states, src.mjm) if a.states
            else lstates.from_qpos(src.mjm, src.qpos0[None], qvel=src.qvel0[None], ctrl=src.ctrl))
    start, _, _ = CAL.calibrate_section(src, sset, a.B, safety=3.0, log=lambda *x: None)
    tiles, rec = T.tune_tiles(src, sset, a.B, T.section_without_peaks(start), C.log)
    C.log(f"{rec['base_ms']:.3f} ms at 8 everywhere -> {rec['best_ms']:.3f} ms: {rec['sizes']}")
    if a.write and a.tuning:
        t = ltuning.load(a.tuning)
        t.setdefault("collision", {})["tile_size"] = tiles if tiles else 8
        t["tiles"] = rec
        ltuning.save(t, a.tuning)
        C.log(f"wrote the sizes to {a.tuning}; recalibrate its capacities (calibrate.py --update)")


if __name__ == "__main__":
    main()
