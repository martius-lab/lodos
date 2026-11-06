"""
The Lodos pairs' contact parameters, one row of thirteen columns a pair:

    [mu, stiffness, sharpness, dissipation, stiction, solref (2), solimp (5), w]

`mu` is every contact's friction. The spring-damper four define the force law below,
which weighs a contact group's members by the normal force each would carry
(`lodos.cd.groups`); the mj mode reads `solref` and `solimp`; `w`, an impedance scale,
is read by nothing.
"""

import numpy as np
import warp as wp


def make_default_contact_params(mu, nCPR):
    """
    `(nCPR, 13)` rows of the defaults with friction `mu`. The solref and solimp defaults
    are MuJoCo's, `dmin` 0.9 among them, so that under the mj mode Lodos's contacts have
    the impedance MuJoCo Warp gives its geoms by default.

    Called by `lodos.sim.default_pair_params` and `viewer.newton_link._newton_contact`.
    In: mu float, every pair's friction; nCPR int, the pairs. Out: (nCPR, 13) float32, the module docstring's columns.
    """
    row = [  # list of 13 floats: one pair's columns
        mu,
        5.0e03,    # stiffness
        5.0e02,    # sharpness
        1.0,       # dissipation
        5.0e-02,   # stiction
        0.02,      # solref - timeconst
        1.0,       # solref - dampratio
        0.9,       # solimp - dmin
        9.5e-01,   # solimp - dmax
        1.0e-03,   # solimp - width
        5.0e-01,   # solimp - mid
        2.0e00,    # solimp - power
        1.0,       # w, an impedance scale; nothing reads it
    ]
    return np.tile(np.array(row, dtype=np.float32), (nCPR, 1))


@wp.func
def dissipation_fn(x: float):
    """
    Velocity-dependent damping factor, C1 and compactly supported above: grows linearly
    as the contact approaches, decays quadratically to zero as it separates, and
    vanishes once separation is fast enough.

    Called by `lodos.cd.groups.group_contacts`. In: x float, the contact's normal velocity over its pair's
    dissipation, negative approaching. Out: float.
    """
    if x < 0.0:
        return 1.0 - x
    if x < 2.0:
        s = x - 2.0  # float
        return 0.25 * s * s
    return 0.0


@wp.func
def complience_fn(x: float):
    """
    The force law's elastic part, `max(x, 0)`: linear in `x`, the penetration times the pair's sharpness, and zero
    once the contact separates.

    Called by `lodos.cd.groups.group_contacts`. In: x float. Out: float.
    """
    return wp.max(x, 0.0)


def ancestor_dofs_np(m, body_id):
    """
    The dofs whose motion moves `body_id`: its own and every ancestor's, sorted.

    Called by `lodos.cd.groups.Groups.__init__`. In: m, MuJoCo's `MjModel` (`dof_bodyid` (nv,), `body_parentid`
    (nbody,)); body_id int. Out: (n,) int64, n the dofs found.
    """
    dof_bodyid = np.asarray(m.dof_bodyid)  # (nv,) int32
    body_parentid = np.asarray(m.body_parentid)  # (nbody,) int32
    dofs = []  # list of (k,) int64: each body's own dofs
    b = int(body_id)  # int: the body, then each ancestor in turn
    while b > 0:
        dofs.append(np.nonzero(dof_bodyid == b)[0])
        b = int(body_parentid[b])
    if not dofs:
        return np.zeros(0, dtype=int)
    return np.sort(np.concatenate(dofs)).astype(int)
