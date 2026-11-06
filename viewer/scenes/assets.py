"""
The collision surfaces and meshes the scenes ship with (`viewer/assets/`), and the derived
files they need.

`load_surface` loads a shipped surface by its name, as `lodos.col.sdf.load_surface_file`
loads a bake from anywhere.

`ASSET_DIR` is `viewer/assets/`, in a clone and in an installed package alike (a wheel
carries it in `viewer`). `$LODOS_ASSETS` overrides it.
"""

import os
from pathlib import Path

from lodos.col.sdf import load_surface_file, write_surface_obj

ROOT = Path(__file__).resolve().parents[2]


def _asset_dir():
    if os.environ.get("LODOS_ASSETS"):
        return Path(os.environ["LODOS_ASSETS"])
    return Path(__file__).resolve().parents[1] / "assets"


ASSET_DIR = _asset_dir()
SURF_DIR = ASSET_DIR / "surfaces"
# The armadillo's 34-piece convex decomposition (y-up, ~1 m tall), for scenes that
# collide armadillos through MJWarp's pipeline. `scenes.drop` rotates and scales it
# onto the baked surface.
ARMADILLO_CP_DIR = ASSET_DIR / "armadillo_cp"

def _cache_dir():
    if os.environ.get("LODOS_CACHE"):
        return Path(os.environ["LODOS_CACHE"])
    if (ROOT / ".git").exists():  # a clone or a worktree: beside the tree
        return ROOT / ".scratch"
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "lodos"


# Derived artifacts -- the OBJ MuJoCo takes a surface's inertia from -- are regenerated on
# demand and live here, out of the tracked tree: `.scratch` in a clone, the user's cache
# directory for an installed package, `$LODOS_CACHE` if set.
CACHE_DIR = _cache_dir()


def load_surface(name, spline_type=None):
    """A shipped surface by its name (`SURF_DIR`), as `lodos.col.sdf.load_surface_file` loads it."""
    return load_surface_file(SURF_DIR / name / "surface_data.npz", spline_type)


def surface_obj(name, spline_type=None):
    """The cached OBJ of a shipped surface, written on first use."""
    path = CACHE_DIR / "surface_obj" / f"{name}_surface.obj"
    if not path.exists():
        write_surface_obj(load_surface(name, spline_type), path)
    return path
