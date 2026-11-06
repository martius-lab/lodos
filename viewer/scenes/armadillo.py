"""
Five armadillos dropped onto a plane, batched over randomized initial stacks, on
baked surfaces only: Lodos-v1's armadillo drop (`papers/lodos_v1.pdf`, which
drops twenty), with five.

Every ordered body pair is declared, both ways round -- a collision pair is
one-directional, so `[i, j]` and `[j, i]` are two pairs -- plus each body against
the ground: 20 body-body pairs and 5 ground pairs for five bodies. The MJCF carries
one mesh per armadillo, the surface itself, for MuJoCo to take the inertia from;
it is keyed to the surface and so takes no part in MJWarp's collision, which has
nothing left to pair.

The initial stack is provably intersection-free: each body is placed by its own
rotated vertical extent, leaving `GAP` between the top of one and the bottom of the
next. `build` checks there are no contacts at the first step.

`build` returns a `lodos.sim.Sim`.
"""

from __future__ import annotations

import mujoco
import numpy as np

from lodos import sim as lsim
from viewer.scenes.assets import load_surface, surface_obj
from lodos.col.sdf import types as sdf_types

N_BODIES = 5
TIMESTEP = 0.002
DURATION = 10.0
GAP = 0.15
MU = 0.8
STIFFNESS = 5.0e3
DISSIPATION = 0.15


def scene_xml(n_bodies, obj_path, timestep):
    """The MJCF, on the elliptic friction cone. No cameras, textures or lights: nothing here renders."""
    bodies = "\n".join(
        f"""    <body name="b{i}" pos="0 0 {1.0 + i}">
      <freejoint/>
      <geom name="g{i}" type="mesh" mesh="arm" density="100"/>
    </body>"""
        for i in range(n_bodies)
    )
    return f"""<mujoco>
  <option timestep="{timestep}" integrator="Euler" cone="elliptic">
    <flag eulerdamp="disable"/>
  </option>
  <asset>
    <mesh name="arm" file="{obj_path}"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" pos="0 0 0" size="12 12 0.1"/>
{bodies}
  </worldbody>
</mujoco>"""


def quat_to_mat(q):
    """Rotation matrix from a MuJoCo (w, x, y, z) quaternion."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def initial_qpos(n_bodies, verts, gap, rng):
    """One stack that provably cannot self-intersect, packed as tightly as that allows."""
    q = np.zeros(7 * n_bodies)
    top = 0.0  # highest occupied height so far; the floor is at z = 0

    for i in range(n_bodies):
        quat = rng.normal(size=4)
        quat /= np.linalg.norm(quat)
        vz = (verts @ quat_to_mat(quat).T)[:, 2]

        z = top + gap - vz.min()
        q[7 * i : 7 * i + 2] = rng.uniform(-0.12, 0.12, 2)
        q[7 * i + 2] = z
        q[7 * i + 3 : 7 * i + 7] = quat
        top = z + vz.max()

    return q


def initial_qpos_batch(n_bodies, verts, gap, seed, batch):
    """`(batch, 7*n_bodies)`: one independently randomized stack per world, one stream."""
    rng = np.random.default_rng(seed)
    return np.stack([initial_qpos(n_bodies, verts, gap, rng) for _ in range(batch)])


def pairs(n_bodies, one_sided=False):
    """Every ordered body pair (or each interface once), then each body against the ground."""
    if one_sided:
        cprs = [[i, j] for i in range(n_bodies) for j in range(i + 1, n_bodies)]
    else:
        cprs = [[i, j] for i in range(n_bodies) for j in range(n_bodies) if i != j]
    return cprs + [[i, n_bodies] for i in range(n_bodies)]


def model(n_bodies=N_BODIES, timestep=TIMESTEP, spline="cubic"):
    """`(mjm, surf)`: the MJCF model and the baked armadillo surface."""
    surf = load_surface("armadillo", spline_type=spline)
    mjm = mujoco.MjModel.from_xml_string(scene_xml(n_bodies, surface_obj("armadillo"), timestep))
    return mjm, surf


def build(
    batch=64,
    seed=0,
    n_bodies=N_BODIES,
    timestep=TIMESTEP,
    gap=GAP,
    mu=MU,
    stiffness=STIFFNESS,
    dissipation=DISSIPATION,
    one_sided=False,
    warm_start=None,
    spline="cubic",
    col_kwargs=None,
    params=None,
    solver_options=None,
    **sim_kwargs,
):
    """
    The batched scene as a `lodos.sim.Sim`, ready to step.

    `solver_options` are the solver's (`lodos.cd.driver.resolve_options`), with
    `warm_start`, which None leaves at its default. `params`, a `(nCPR, 13)` table,
    replaces the default one (`lodos.cd.params`, with this experiment's stiffness and
    dissipation). Everything in `sim_kwargs` goes to `lodos.sim.make_sim` -- MJWarp's
    capacities, for instance, which must hold every Lodos contact.
    """
    mjm, surf = model(n_bodies, timestep, spline)
    q0 = initial_qpos_batch(n_bodies, np.asarray(surf.v), gap, seed, batch)
    cprs = pairs(n_bodies, one_sided)
    if params is None:
        params = lsim.default_pair_params(mu, len(cprs))
        params[:, 1] = stiffness
        params[:, 3] = dissipation
    surfaces = lsim.Surfaces(
        surfaces=[surf] * n_bodies + [sdf_types.PlaneSurface()],
        geoms=[mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, f"g{i}") for i in range(n_bodies)]
        + [mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "floor")],
        pairs=cprs,
        params=params,
        mu=mu,
        col_kwargs=dict(col_kwargs or {}),
    )
    solver_kwargs = dict(solver_options or {})
    if warm_start is not None:
        solver_kwargs["warm_start"] = warm_start
    return lsim.make_sim(mjm, batch, surfaces=surfaces, qpos=q0, solver_kwargs=solver_kwargs, **sim_kwargs)
