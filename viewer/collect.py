"""
One frame of a live session, read off a one-world `lodos.sim.Sim`.

A live session is one world and reads every frame back. `Collector` reads the contacts
from MJWarp's `d.contact`, Lodos's exported ones among them (typed `NOCARRY`), with each
contact's constraint force decoded by `lodos.cd.solver.constraint.contact_force` into the world frame:
the force on `geom[1]`, along a normal from `geom[0]` to `geom[1]`. A contact group's
force is drawn on its members, each its share. The relative velocity follows the same
convention: second body minus first, at the point.

A frame is collected after `lodos.step.forward` and before the integrator, the one
moment everything in it belongs to one state: `qpos`/`qvel` are the state, the poses
and contacts are computed from it, and the forces are the ones the step applies.
"""

from __future__ import annotations

import dataclasses

import mujoco
import numpy as np
import warp as wp

from viewer.geometry import geom_xforms
from viewer.trajectory import Frame

from lodos.cd.solver import constraint as solver_constraint
from lodos.mjcf.types import ContactType

SRC_LODOS, SRC_PRIM = 0, 1
NOCARRY = int(ContactType.NOCARRY)
GROUP = int(ContactType.GROUP)


@dataclasses.dataclass
class LiveFrame(Frame):
    """
    A `viewer.trajectory.Frame` with the state it was collected at, each contact's
    pipeline and bodies, the bodies' velocities and the frame's scalar traces.
    """

    qpos: np.ndarray = None  # (nq,)
    qvel: np.ndarray = None  # (nv,)
    con_src: np.ndarray = None  # (n,) SRC_LODOS or SRC_PRIM
    con_body: np.ndarray = None  # (n, 2) first and second body
    body_v: np.ndarray = None  # (nbody, 3) world velocity of each body's origin
    body_w: np.ndarray = None  # (nbody, 3) world angular velocity
    warm: np.ndarray = None  # (nv,) MJWarp's warm start, under Lodos
    stats: dict = None  # ncon_lodos, ncon_prim, depth, ke, niter; rounds, converged (the solve's)


def ancestor_matrix(mjm):
    """`(nbody, nv)`: 1 where dof `i` moves body `b` -- its own dofs and its ancestors'."""
    nbody, nv = int(mjm.nbody), int(mjm.nv)
    A = np.zeros((nbody, nv))
    parent = np.asarray(mjm.dof_parentid)
    for b in range(1, nbody):
        # The body's last dof, or the nearest ancestor's: a body without joints moves
        # with its parent.
        bb = b
        while bb > 0 and int(mjm.body_dofnum[bb]) == 0:
            bb = int(mjm.body_parentid[bb])
        if bb == 0:
            continue
        i = int(mjm.body_dofadr[bb]) + int(mjm.body_dofnum[bb]) - 1
        while i >= 0:
            A[b, i] = 1.0
            i = int(parent[i])
    return A


class Collector:
    """Reads `LiveFrame`s off one `lodos.sim.Sim` (`B = 1`) of `scene`."""

    def __init__(self, sim, scene):
        if sim.B != 1:
            raise ValueError("a live session is one world")
        self.sim, self.scene = sim, scene
        mjm = sim.mjm
        self.mjm = mjm
        self.A = ancestor_matrix(mjm)
        self.rootid = np.asarray(mjm.body_rootid, int)
        self.geom_bodyid = np.asarray(mjm.geom_bodyid, int)
        tb = sim.mrbd.tree_beg.numpy().astype(int)
        bb = sim.mrbd.blk_beg.numpy().astype(int)
        self.trees = [(tb[t], tb[t + 1], bb[t]) for t in range(len(tb) - 1)]

        n = int(sim.d.naconmax)
        self.ids = wp.array(np.arange(n, dtype=np.int32), dtype=int)
        self.force = wp.zeros(n, dtype=wp.spatial_vector)

    # ---------------------------------------------------------------------------------

    def _bodies(self, qvel, cdof, com, xpos):
        """Per body: angular velocity, linear velocity at its root's COM, that COM."""
        w = self.A @ (cdof[:, :3] * qvel[:, None])
        vc = self.A @ (cdof[:, 3:] * qvel[:, None])
        c = com[self.rootid]
        v0 = vc + np.cross(w, xpos - c)
        return w, vc, c, v0

    def _point_velocity(self, w, vc, c, body, p):
        return vc[body] + np.cross(w[body], p - c[body])

    def _ke(self, M_blk, qvel):
        ke = 0.0
        for lo, hi, beg in self.trees:
            n = hi - lo
            M = M_blk[beg:beg + n * n].reshape(n, n)
            M = np.tril(M) + np.tril(M, -1).T
            v = qvel[lo:hi]
            ke += 0.5 * float(v @ M @ v)
        return ke

    def _prim_contacts(self):
        sim, d = self.sim, self.sim.d
        nacon = min(int(d.nacon.numpy()[0]), int(d.naconmax))
        if nacon == 0:
            return None
        c = d.contact
        efc = c.efc_address[:nacon].numpy()[:, 0]
        keep = efc >= 0
        solver_constraint.contact_force(sim.m, d, self.ids[:nacon], True, self.force[:nacon])
        f6 = self.force[:nacon].numpy()
        p = c.pos[:nacon].numpy().astype(np.float64)
        frame = c.frame[:nacon].numpy().astype(np.float64)
        dist = c.dist[:nacon].numpy().astype(np.float64)
        geom = c.geom[:nacon].numpy().astype(int)
        typ = c.type[:nacon].numpy().astype(int)
        bodies = self.geom_bodyid[geom]
        src = np.where(typ & NOCARRY, SRC_LODOS, SRC_PRIM)
        group = (typ & GROUP) != 0
        k = keep & ~group
        out = [p[k], frame[k, 0], dist[k], f6[k, :3].astype(np.float64), src[k], bodies[k]]
        if (keep & group).any():
            # A contact group's force, `lambda` in its own coordinates, reaches each of its members as
            # a_i (lambda_n n_i + lambda_t1 t1_i + lambda_t2 t2_i), a ray's as a_i lambda (n_i - r_i t1_i).
            solver_constraint.contact_force(sim.m, d, self.ids[:nacon], False, self.force[:nacon])
            lam = self.force[:nacon].numpy()[:, :3].astype(np.float64)
            dim = c.dim[:nacon].numpy()
            gk = np.flatnonzero(keep & group)
            adr = d.contact_group_adr[:nacon].numpy()[gk]
            num = d.contact_group_num[:nacon].numpy()[gk]
            top = int((adr + num).max())
            mp = d.group_member_pos[:top].numpy().astype(np.float64)
            mf = d.group_member_frame[:top].numpy().astype(np.float64)
            md = d.group_member_dist[:top].numpy().astype(np.float64)
            ma = d.group_member_weight[:top].numpy().astype(np.float64)
            ms = d.group_member_slip[:top].numpy().astype(np.float64)
            idx = np.concatenate([np.arange(a, a + n) for a, n in zip(adr, num)])
            owner = np.repeat(gk, num)
            fr = mf[idx]
            l = lam[owner]
            ray = (dim[owner] == 1)[:, None]
            f_cone = l[:, :1] * fr[:, 0] + l[:, 1:2] * fr[:, 1] + l[:, 2:3] * fr[:, 2]
            f_ray = l[:, :1] * (fr[:, 0] - ms[idx, None] * fr[:, 1])
            f = ma[idx, None] * np.where(ray, f_ray, f_cone)
            parts = [mp[idx], fr[:, 0], md[idx], f, np.full(len(idx), SRC_LODOS), bodies[owner]]
            out = [np.concatenate([a, b]) for a, b in zip(out, parts)]
        p, n, depth, f, src, bodies = out
        return p, n, depth, f, np.full(len(p), -1), src, bodies

    # ---------------------------------------------------------------------------------

    def collect(self):
        """The current state of the simulation as a `LiveFrame`. Synchronizes."""
        sim = self.sim
        drbd = sim.drbd
        qpos = drbd.qpos.numpy()[0].astype(np.float64)
        qvel = drbd.qvel.numpy()[0].astype(np.float64)
        xpos = drbd.xpos.numpy()[0].astype(np.float64)
        xquat = drbd.xquat.numpy()[0].astype(np.float64)
        cdof = drbd.cdof.numpy()[0].astype(np.float64)
        com = drbd.subtree_com.numpy()[0].astype(np.float64)
        time = float(drbd.time.numpy()[0])

        w, vc, c, v0 = self._bodies(qvel, cdof, com, xpos)

        con = self._prim_contacts()
        if con is None:
            z3 = np.zeros((0, 3))
            p, n, depth, f, seg, src, bodies = z3, z3, np.zeros(0), z3, np.zeros(0, int), np.zeros(0, int), \
                np.zeros((0, 2), int)
        else:
            p, n, depth, f, seg, src, bodies = con
        v = (self._point_velocity(w, vc, c, bodies[:, 1], p) - self._point_velocity(w, vc, c, bodies[:, 0], p)) \
            if len(p) else np.zeros((0, 3))

        niter = float(sim.d.solver_niter.numpy()[0])
        warm = sim.d.qacc_warmstart.numpy()[0].astype(np.float64)
        # Whether the solve converged: the proximal loop's flag and its rounds where it runs
        # (none recorded by the fused kernel); else the Newton solve's, stopped short of its cap.
        rounds, converged = float("nan"), float("nan")
        m = sim.m
        if sim.solver is not None and sim.solver.fused is None and bool(m.opt.prox_enable):
            rounds = float(sim.d.prox_rounds.numpy()[0])
            converged = bool(sim.d.prox_converged.numpy()[0])
        elif sim.solver is not None and sim.solver.fused is None:
            converged = niter < int(m.opt.iterations)

        M_blk = drbd.M_blk.numpy()[:, 0].astype(np.float64)
        stats = dict(
            ncon_lodos=int((src == SRC_LODOS).sum()),
            ncon_prim=int((src == SRC_PRIM).sum()),
            depth=float(depth.min()) if len(depth) else 0.0,
            ke=self._ke(M_blk, qvel),
            niter=niter,
            rounds=rounds,
            converged=converged,
        )

        body_xform = np.concatenate([xpos, xquat[:, 1:], xquat[:, :1]], axis=1)
        jnt_anchor = drbd.xanchor.numpy()[0].astype(np.float64) if int(self.mjm.njnt) else np.zeros((0, 3))
        jnt_axis = drbd.xaxis.numpy()[0].astype(np.float64) if int(self.mjm.njnt) else np.zeros((0, 3))
        return LiveFrame(
            time=time,
            con_p=p.astype(np.float32), con_n=n.astype(np.float32), con_v=v.astype(np.float32),
            con_f=f.astype(np.float32), con_depth=depth.astype(np.float32), con_seg=seg.astype(np.int32),
            geom_xform=geom_xforms(self.scene, xpos, xquat).astype(np.float32),
            body_xform=body_xform.astype(np.float32),
            jnt_anchor=jnt_anchor.astype(np.float32), jnt_axis=jnt_axis.astype(np.float32),
            qpos=qpos, qvel=qvel, con_src=src.astype(np.int8), con_body=bodies.astype(np.int32),
            body_v=v0.astype(np.float32), body_w=w.astype(np.float32), warm=warm, stats=stats,
        )


def total_force(frame, body):
    """The sum of a frame's contact forces on `body`: +f on the second, -f on the first."""
    f = frame.con_f.astype(np.float64)
    out = f[frame.con_body[:, 1] == body].sum(0) - f[frame.con_body[:, 0] == body].sum(0)
    return out


def energy_reference(mjm, qpos, qvel):
    """MuJoCo's own kinetic energy of a state, for checking `Collector`'s."""
    d = mujoco.MjData(mjm)
    d.qpos[:], d.qvel[:] = qpos, qvel
    mujoco.mj_kinematics(mjm, d)
    mujoco.mj_comPos(mjm, d)
    mujoco.mj_crb(mjm, d)
    Mv = np.zeros(mjm.nv)
    mujoco.mj_mulM(mjm, d, Mv, d.qvel)
    return 0.5 * float(d.qvel @ Mv)
