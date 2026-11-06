"""
Reading geometry out of an MJCF, in the frame the collision pipeline authors in.

Shared by every step under `tools/mesh_processing/`. Nothing here imports from outside
this directory.
"""

from __future__ import annotations

import mujoco
import numpy as np
import trimesh

from pathlib import Path


# One suffix per step. A step *appends* its own to whatever its input carried, so a
# directory's name is the record of how it was made and the chain ends at
# `robot_wt_cp_fused_im_man_sdata`. `model_stem` strips the lot to recover the
# model's name, which is how each script finds the MJCF its world poses need.
PIPELINE_SUFFIXES = ("_world", "_sdata", "_man", "_im", "_fused", "_cp", "_wt")


def step_dir(path, suffix):
    """Where a step writes: `robot_wt` + `_cp_fused` -> `robot_wt_cp_fused`."""
    p = Path(path)
    base = p.stem if p.suffix else p.name       # `robot.xml` -> `robot`
    return p.parent / f"{base}{suffix}"


def model_stem(name):
    """`right_hand_cp_fused` -> `right_hand`, so the MJCF beside it can be found."""
    stem = str(name)
    changed = True
    while changed:
        changed = False
        for suffix in PIPELINE_SUFFIXES:
            if stem.endswith(suffix):
                stem, changed = stem[: -len(suffix)], True
    return stem


def collides(model, g):
    return (model.geom_contype[g] | model.geom_conaffinity[g]) != 0


def mesh_geoms(model, body, which="visual"):
    """
    The mesh geoms on `body`, filtered by whether MuJoCo collides with them.

    `visual` is the default because that is how a MuJoCo model is built: meshes are
    shipped to look at, marked `contype="0" conaffinity="0"`, and the physics runs
    against hand-fitted capsules. lodos collides the mesh itself, so the visual set
    is the input and the fitted primitives are of no use.
    """
    return [
        g
        for g in range(model.ngeom)
        if model.geom_bodyid[g] == body
        and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
        and (
            which == "all"
            or (collides(model, g) if which == "collision" else not collides(model, g))
        )
    ]


def body_frame_mesh(model, g):
    """
    Geom `g`'s mesh, moved from its own frame into its body's.

    MuJoCo's compiler recenters mesh vertices on their inertial frame and folds the
    correction into `geom_pos`/`geom_quat`, so `R(geom_quat) @ v + geom_pos` is exact
    -- verified against `geom_xpos`/`geom_xmat` at a random `qpos` to 3e-17.
    """
    mid = model.geom_dataid[g]
    va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
    v = model.mesh_vert[va : va + vn].astype(np.float64)
    f = model.mesh_face[fa : fa + fn].astype(np.int64)

    r = np.zeros(9)
    mujoco.mju_quat2Mat(r, model.geom_quat[g])
    return trimesh.Trimesh(v @ r.reshape(3, 3).T + model.geom_pos[g], f, process=False)


def merge_by_body(model, which="visual"):
    """
    `{body name: Trimesh}` in body frame, one entry per body that has any mesh.

    Concatenation, not a boolean union: the parts of one body are separate closed
    solids that touch, and a union of raw visual meshes is not even well posed when
    some of them are open. Making the result a solid is a separate problem.
    """
    out = {}
    for b in range(1, model.nbody):
        geoms = mesh_geoms(model, b, which)
        if not geoms:
            continue
        parts = [body_frame_mesh(model, g) for g in geoms]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
        out[name] = trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]
    return out


def body_poses(model):
    """`{body name: (R, t)}` at `qpos0` -- the pose to draw a body-frame mesh at."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b): (
            data.xmat[b].reshape(3, 3),
            data.xpos[b].copy(),
        )
        for b in range(1, model.nbody)
    }
