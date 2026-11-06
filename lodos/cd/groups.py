"""
Contact groups for the NLSP solver: a Lodos pair segment's contacts in one normal bin and moment sector, solved as one
super-contact in MJWarp's solve, beside its per-contact rows, joint limits and friction loss.

The design is the Contact-Group Reparameterization section of `papers/lodos_v2.pdf`. A group of member contacts `i`,
with normalized weights `a_i` (summing to 1), takes one force `lambda`, and each member receives

    cone (condim 3):  f_i = a_i (lambda_n n_i + lambda_t1 t1_i + lambda_t2 t2_i),  lambda in the friction cone
    ray (condim 1):   f_i = a_i lambda (n_i - r_i t1_i),                            lambda >= 0

-- that section's cone and ray parameterizations, in coordinates where `lambda_n` is the group's total normal force.

- The weights are the spring-damper normal force with its damping at the step's start,
  `fbar_n = D(v_n / dissipation) (stiffness / sharpness) max(-sdf sharpness, 0)`, normalized over the group. A group
  whose total is zero -- every member separating at twice its dissipation velocity or faster -- carries no force:
  it is not exported, and neither are its contacts.
- A cone member's `t1` follows its own slip at the step's start, `t1 = normalize(v_t + eps t1~)`, `t1~` its last
  step's `t1` carried with the vertex body's rotation and onto the new normal by the minimal rotation; a contact new
  this step takes its bin's tangent, turned onto its normal. `t2 = n x t1`. A contact keeps its identity from step
  to step as (world, segment, vertex): a segment lists its contacts in vertex order, so last step's is found by
  bisection.
- A ray member's row is the spring-damper force per unit normal force (`lodos.cd.params`): its smooth-stiction friction at
  the ratio `r_i = mu |v_t| / sqrt(stiction^2 + |v_t|^2)` against its slip, `t1 = v_t / |v_t|`.

MJWarp builds a group's rows `sum_i a_i d_i^T J_i`, MuJoCo's weights and references from the members' own, and the
prox solver's Baumgarte bias `sum_i a_i gamma_n(d_i)` and de Saxce term (cone `mu sum_i a_i |sigma_i,t|`, ray
`sum_i a_i r_i t1_i . sigma_i`, which leaves the ray holding its members' normal velocities) member by member (`ContactType.GROUP`, `lodos/cd/solver/constraint.py` and
`nlsp.py`). This module decides the groups and writes their members and contacts. Capturable: every array is
allocated here.
"""

from __future__ import annotations

import numpy as np
import warp as wp


from lodos.cd import bins as cd_bins
from lodos.cd.bins import MOMENT_BINS
from lodos.cd.params import ancestor_dofs_np
from lodos.cd.params import complience_fn
from lodos.cd.params import dissipation_fn
from lodos.cd.solver.io import alloc_contact_groups
from lodos.col.prim.driver import ContactOut
from lodos.mjcf.types import ContactType
from lodos.mjcf.types import vec5

wp.set_module_options({"enable_backward": False})

KINDS = ("contact", "ray", "cone")  # the `groups` option's kinds, in the order of their codes below
CONTACT = wp.constant(0)  # a pair's contacts, each its own MJWarp contact
RAY = wp.constant(1)  # a pair's contacts in condim-1 groups
CONE = wp.constant(2)  # a pair's contacts in condim-3 groups
# Lodos's own contacts' type (`lodos.cd.driver.LODOS_CONTACT`), as a group: never in the prox solver's anchor cache.
GROUP_CONTACT = wp.constant(int(ContactType.CONSTRAINT | ContactType.NOCARRY | ContactType.GROUP))


@wp.struct
class GroupModel:
    """
    What does not change from step to step: the pairs' treatment, the bins, the dof walks, the force law.

    Built by `Groups.__init__`.
    """

    nnorm: int  # normal bins
    nmom: int  # moment sectors per normal bin
    nbin: int  # normal bins x moment sectors
    nGSEG: int  # grouped segments
    reduced: int  # 1: a grouped contact the reduction (`lodos.cd.reduce`) did not keep is in no group
    eps: float  # the slip (m/s) below which a cone member's frame keeps its carried direction
    gseg: wp.array(dtype=int)  # (nGSEG,) the grouped segments
    seg_kind: wp.array(dtype=int)  # (nSEG,) CONTACT, RAY or CONE
    seg_cpr: wp.array(dtype=int)  # (nSEG,) the segment's pair (mcol's)
    seg_g_v: wp.array(dtype=int)  # (nSEG,) its vertex geometry (mcol's)
    seg_g_s: wp.array(dtype=int)  # (nSEG,) its SDF geometry (mcol's)
    g_geom: wp.array(dtype=int)  # (nB,) the MJCF geom of each Lodos geometry
    g_bid: wp.array(dtype=int)  # (nB,) its body
    g_rootid: wp.array(dtype=int)  # (nB,) its body's root
    g_anc_beg: wp.array(dtype=int)  # (nB + 1,)
    g_anc_dof: wp.array(dtype=int)  # the dofs that move it
    cpr_stiffness: wp.array(dtype=float)  # (nCPR,) the pair's spring-damper stiffness
    cpr_sharpness: wp.array(dtype=float)  # (nCPR,) its sharpness
    cpr_dissipation: wp.array(dtype=float)  # (nCPR,) its dissipation velocity (m/s)
    cpr_stiction: wp.array(dtype=float)  # (nCPR,) its stiction velocity (m/s)
    cpr_friction: wp.array(dtype=vec5)  # Lodos's per-contact export's, so a live friction applies to both
    cpr_solref: wp.array(dtype=wp.vec2)  # (nCPR,) the pair's MuJoCo solref
    cpr_solimp: wp.array(dtype=vec5)  # (nCPR,) its MuJoCo solimp
    bin_n: wp.array(dtype=wp.vec3)  # (nnorm,) the normal bins' directions
    bin_t1: wp.array(dtype=wp.vec3)  # (nnorm,) each bin's first tangent
    bin_t2: wp.array(dtype=wp.vec3)  # (nnorm,) its second


@wp.struct
class GroupData:
    """
    A step's groups -- which contacts, their weights, frames and members -- and the frames carried from the last.

    Built by `Groups.__init__`.
    """

    key_nCa3: wp.array(dtype=int)  # (nCa3_max,) a grouped contact's bin x sector; -1 on a per-contact pair
    fbar_nCa3: wp.array(dtype=float)  # (nCa3_max,) its spring-damper normal force
    t1_nCa3: wp.array(dtype=wp.vec3)  # (nCa3_max,) its first tangent: a cone's frame, a ray's slip
    slip_nCa3: wp.array(dtype=float)  # (nCa3_max,) a ray member's friction ratio r_i
    cnt: wp.array2d(dtype=int)  # (B, nGSEG * nbin) contacts in each (grouped segment, bin x sector)
    fsum: wp.array2d(dtype=float)  # (B, nGSEG * nbin) their spring-damper normal forces summed
    cur: wp.array2d(dtype=int)  # (B, nGSEG * nbin) the next member slot of its group; -1 where none
    ngrp: wp.array(dtype=int)  # (B,)
    grp_seg: wp.array2d(dtype=int)  # (B, G_max)
    grp_slot: wp.array2d(dtype=int)  # (B, G_max) its (grouped segment, bin x sector)
    grp_adr: wp.array2d(dtype=int)  # (B, G_max) its first member
    grp_num: wp.array2d(dtype=int)  # (B, G_max) its members
    grp_cid: wp.array2d(dtype=int)  # (B, G_max) the MJWarp contact standing for it; -1 if it overflowed
    # A cone member's carried frame, two steps' worth: row `parity` this step's, the other row last step's. The rows
    # swap roles every step (`flip`), so nothing is copied: the arrays are the whole contact capacity.
    t1_body: wp.array2d(dtype=wp.vec3)  # (2, nCa3_max) a cone member's t1, in its vertex body's frame
    n_body: wp.array2d(dtype=wp.vec3)  # (2, nCa3_max) its normal, the same way
    vtx: wp.array2d(dtype=int)  # (2, nCa3_max) a cone member's vertex, which finds it again next step
    parity: wp.array(dtype=int)  # (1,) this step's row
    idul_prev: wp.array2d(dtype=int)  # (B, nSEG + 1) last step's segment runs
    overflow: wp.array(dtype=int)  # (1,) a group found no MJWarp contact
    peak: wp.array(dtype=int)  # (1,) the most groups a world had


@wp.func
def _key(gm: GroupModel, n: wp.vec3, p: wp.vec3, rc: wp.vec3):  # type: ignore
    """
    A contact's normal bin (`lodos.cd.bins`), and its bin x sector: the sector of its point about the vertex body's
    root COM.

    Called by `group_contacts`.
    In: n wp.vec3, the contact's unit normal; p wp.vec3, its point; rc wp.vec3, the vertex body's root COM.
    Out: int, int: the normal bin, 0 to nnorm - 1, and the bin x sector, 0 to nbin - 1.
    """
    kn = int(0)  # int: the closest bin so far
    best = wp.dot(gm.bin_n[0], n)  # float: its direction's dot with n
    for k in range(1, gm.nnorm):  # k: int, a bin
        d = wp.dot(gm.bin_n[k], n)  # float
        if d > best:
            best = d
            kn = k
    key = kn  # int: the bin x sector
    if gm.nmom > 1:
        dirn = gm.bin_n[kn]  # wp.vec3: the bin's direction
        r = p - rc  # wp.vec3: the moment arm, then its part in the bin's plane
        r = r - wp.dot(r, dirn) * dirn
        a = wp.atan2(wp.dot(r, gm.bin_t2[kn]), wp.dot(r, gm.bin_t1[kn])) + wp.pi  # float: its angle, 0 to 2 pi (rad)
        sector = wp.min(int(a * float(gm.nmom) / (2.0 * wp.pi)), gm.nmom - 1)  # int: its sector
        key = kn * gm.nmom + sector
    return kn, key


@wp.func
def _turn(a: wp.vec3, b: wp.vec3, v: wp.vec3):
    """
    `v` turned by the minimal rotation taking the unit `a` onto the unit `b`, or `-a` onto `b` where they oppose.

    Called by `group_contacts`.
    In: a, b, v wp.vec3. Out: wp.vec3.
    """
    u = a  # wp.vec3: a, or -a where it opposes b
    if wp.dot(a, b) < 0.0:
        u = -a
    k = wp.cross(u, b)  # wp.vec3: the rotation's axis times its sine
    c = wp.dot(u, b)  # float: its cosine
    return v * c + wp.cross(k, v) + k * (wp.dot(k, v) / (1.0 + c))


@wp.func
def _tangent(n: wp.vec3):
    """
    A unit tangent of the unit `n`, for a direction nothing else decides.

    Called by `group_contacts` and `export_groups`.
    In: n wp.vec3. Out: wp.vec3.
    """
    t = wp.cross(n, wp.vec3(1.0, 0.0, 0.0))  # wp.vec3
    if wp.abs(n[0]) > 0.9:
        t = wp.cross(n, wp.vec3(0.0, 1.0, 0.0))
    return wp.normalize(t)


@wp.func
def _side_motion(gm: GroupModel, b: int, g: int, qvel: wp.array2d(dtype=float),  # type: ignore
                 cdof: wp.array2d(dtype=wp.spatial_vector), subtree_com: wp.array2d(dtype=wp.vec3)):  # type: ignore
    """
    A geometry's angular and linear velocity about its tree's root COM, and that point.

    Called by `group_contacts`, for each side of a contact.
    In: b int, the world; g int, the Lodos geometry; qvel (B, nv) float32, cdof (B, nv) wp.spatial_vector,
    subtree_com (B, nbody) wp.vec3. Out: wp.vec3, wp.vec3, wp.vec3: the angular velocity (rad/s), the velocity of
    the point at the root COM (m/s), and that point.
    """
    w = wp.vec3(0.0)  # wp.vec3: the angular velocity
    v = wp.vec3(0.0)  # wp.vec3: the linear velocity at the root COM
    for idx in range(gm.g_anc_beg[g], gm.g_anc_beg[g + 1]):  # idx: int, a slot of g's dof walk
        i = gm.g_anc_dof[idx]  # int: a dof that moves g
        cd = cdof[b, i]  # wp.spatial_vector: its motion axis, [angular, linear]
        qv = qvel[b, i]  # float: its velocity
        w += qv * wp.vec3(cd[0], cd[1], cd[2])
        v += qv * wp.vec3(cd[3], cd[4], cd[5])
    return w, v, subtree_com[b, gm.g_rootid[g]]


@wp.kernel
def group_contacts(
    gm: GroupModel,  # type: ignore
    gd: GroupData,  # type: ignore
    ct: wp.array(dtype=int),  # type: ignore
    b_nCa3: wp.array(dtype=int),  # type: ignore
    seg_nCa3: wp.array(dtype=int),  # type: ignore
    v_nCa3: wp.array(dtype=int),  # type: ignore
    p_nCa3: wp.array(dtype=wp.vec3),  # type: ignore
    n_nCa3: wp.array(dtype=wp.vec3),  # type: ignore
    sdf_nCa3: wp.array(dtype=float),  # type: ignore
    qvel: wp.array2d(dtype=float),  # type: ignore
    cdof: wp.array2d(dtype=wp.spatial_vector),  # type: ignore
    subtree_com: wp.array2d(dtype=wp.vec3),  # type: ignore
    xmat: wp.array2d(dtype=wp.mat33),  # type: ignore
    keep: wp.array(dtype=wp.uint8),  # (nCa3_max,) the reduction's, under `gm.reduced` # type: ignore
):
    """
    A grouped contact's bin x sector, its spring-damper normal force, and its first tangent: a cone member's frame
    carried from last step and turned toward its slip, a ray member's slip and friction ratio. -1 on the others, and
    on a contact the reduction dropped, whose frame is still carried, for a step that keeps it.

    Launch: dim=(nCa3_max,), a thread per contact slot, by `Groups.before_export`.
    Reads: ct (5,) int32 (ct[2] the contacts); b_nCa3, seg_nCa3 and v_nCa3 (nCa3_max,) int32; p_nCa3 and n_nCa3
    (nCa3_max,) wp.vec3; sdf_nCa3 (nCa3_max,) float32; qvel (B, nv) float32; cdof (B, nv) wp.spatial_vector;
    subtree_com (B, nbody) wp.vec3; xmat (B, nbody) wp.mat33; keep (nCa3_max,) uint8, or (1,) unread when not
    reduced; gd.parity (1,), gd.idul_prev (B, nSEG + 1) and last step's rows of gd.vtx, gd.n_body and gd.t1_body;
    gm's segments, pair parameters, bins and dof walks.
    Writes: gd.key_nCa3, gd.fbar_nCa3, gd.t1_nCa3 and gd.slip_nCa3 (nCa3_max,); this step's rows of
    gd.vtx (2, nCa3_max) int32 and of gd.t1_body and gd.n_body (2, nCa3_max) wp.vec3.
    """
    i = wp.tid()  # int: the contact
    if i >= ct[2]:
        return
    sg = seg_nCa3[i]  # int: its segment
    kind = gm.seg_kind[sg]  # int: CONTACT, RAY or CONE
    if kind == CONTACT:
        gd.key_nCa3[i] = -1
        return
    b = b_nCa3[i]  # int: its world
    p = p_nCa3[i]  # wp.vec3: its point
    n = n_nCa3[i]  # wp.vec3: its normal
    gv = gm.seg_g_v[sg]  # int: the vertex geometry
    wv, vv, cv = _side_motion(gm, b, gv, qvel, cdof, subtree_com)  # wp.vec3 each: the vertex side's (w, v, root COM)
    ws, vs, cs = _side_motion(gm, b, gm.seg_g_s[sg], qvel, cdof, subtree_com)  # wp.vec3 each: the SDF side's
    # wp.vec3: the vertex side's velocity at p relative to the SDF side's
    vel = (vv + wp.cross(wv, p - cv)) - (vs + wp.cross(ws, p - cs))
    vn = wp.dot(vel, n)  # float: its normal part (m/s)
    vt = vel - vn * n  # wp.vec3: its tangential part, the slip
    j = gm.seg_cpr[sg]  # int: the pair
    sharp = gm.cpr_sharpness[j]  # float
    gd.fbar_nCa3[i] = (dissipation_fn(vn / gm.cpr_dissipation[j]) * (gm.cpr_stiffness[j] / sharp)
                       * complience_fn(-sdf_nCa3[i] * sharp))
    kn, key = _key(gm, n, p, cv)  # int, int: the normal bin, the bin x sector
    if gm.reduced != 0 and keep[i] == wp.uint8(0):
        key = -1
    gd.key_nCa3[i] = key
    slip = wp.length(vt)  # float: the slip speed (m/s)

    if kind == RAY:
        t1 = _tangent(n)  # wp.vec3: the member's first tangent
        if slip > 0.0:
            t1 = vt / slip
        st = gm.cpr_stiction[j]  # float: the stiction velocity (m/s)
        gd.t1_nCa3[i] = t1
        gd.slip_nCa3[i] = gm.cpr_friction[j][0] * slip / wp.sqrt(st * st + slip * slip)
        return

    # A cone member's frame: its last step's, carried with the vertex body and onto the new normal, else its
    # bin's tangent turned onto it; then toward its slip.
    cur = gd.parity[0]  # int: this step's row of the carried frames
    old = 1 - cur  # int: last step's
    R = xmat[b, gm.g_bid[gv]]  # wp.mat33: the vertex body's orientation
    v = v_nCa3[i]  # int: the contact's vertex
    gd.vtx[cur, i] = v
    end = gd.idul_prev[b, sg + 1]  # int: one past the segment's last contact last step
    lo = gd.idul_prev[b, sg]  # int: the bisection's lower bound, from the segment's first contact last step
    hi = end  # int: its upper bound
    while lo < hi:
        mid = (lo + hi) >> 1  # int
        if gd.vtx[old, mid] < v:
            lo = mid + 1
        else:
            hi = mid
    carried = _turn(gm.bin_n[kn], n, gm.bin_t1[kn])  # wp.vec3: the carried t1, the bin's for a new contact
    if lo < end:
        if gd.vtx[old, lo] == v:
            carried = _turn(R * gd.n_body[old, lo], n, R * gd.t1_body[old, lo])
    t1 = vt + gm.eps * carried
    t1 = t1 - wp.dot(t1, n) * n
    norm = wp.length(t1)  # float
    if norm > 1.0e-6 * gm.eps:
        t1 = t1 / norm
    else:
        t1 = _tangent(n)
    gd.t1_nCa3[i] = t1
    gd.slip_nCa3[i] = 0.0
    gd.t1_body[cur, i] = wp.transpose(R) * t1
    gd.n_body[cur, i] = wp.transpose(R) * n


@wp.kernel
def flip(gd: GroupData):  # type: ignore
    """
    This step's frames become last step's.

    Launch: dim=1, a single thread, by `Groups.before_export`.
    Reads and writes: gd.parity (1,) int32.
    """
    gd.parity[0] = 1 - gd.parity[0]


@wp.kernel
def group_count(
    gm: GroupModel,  # type: ignore
    gd: GroupData,  # type: ignore
    idul: wp.array2d(dtype=int),  # (B, nSEG + 1) each (world, segment)'s contacts # type: ignore
):
    """
    The contacts kept in each (grouped segment, bin x sector) and their spring-damper normal forces, summed in order.

    Launch: dim=(B, nGSEG), a thread per world and grouped segment, by `Groups.before_export`.
    Reads: idul (B, nSEG + 1) int32, gd.key_nCa3 (nCa3_max,) int32, gd.fbar_nCa3 (nCa3_max,) float32.
    Writes: gd.cnt (B, nGSEG * nbin) int32, gd.fsum (B, nGSEG * nbin) float32.
    """
    b, s = wp.tid()  # int, int: the world, the grouped segment
    sg = gm.gseg[s]  # int: its segment
    base = s * gm.nbin  # int: its first (grouped segment, bin x sector) slot
    for k in range(gm.nbin):  # k: int, a bin x sector
        gd.cnt[b, base + k] = 0
        gd.fsum[b, base + k] = 0.0
    for i in range(idul[b, sg], idul[b, sg + 1]):  # i: int, a contact of the segment
        key = gd.key_nCa3[i]  # int: its bin x sector, -1 in no group
        if key < 0:
            continue
        slot = base + key  # int: its slot
        gd.cnt[b, slot] = gd.cnt[b, slot] + 1
        gd.fsum[b, slot] = gd.fsum[b, slot] + gd.fbar_nCa3[i]


@wp.kernel
def group_assign(
    gm: GroupModel,  # type: ignore
    gd: GroupData,  # type: ignore
    idul: wp.array2d(dtype=int),  # type: ignore
):
    """
    A world's groups, in order: every (grouped segment, bin x sector) with contacts whose spring-damper normal force
    does not vanish. Its members take consecutive slots from the world's first contact's on.

    Launch: dim=B, a thread per world, by `Groups.before_export`.
    Reads: idul (B, nSEG + 1) int32 (its first column, each world's first contact), gd.cnt (B, nGSEG * nbin) int32,
    gd.fsum (B, nGSEG * nbin) float32. Writes: gd.cur (B, nGSEG * nbin) int32; gd.grp_seg, gd.grp_slot, gd.grp_adr
    and gd.grp_num (B, G_max) int32; gd.ngrp (B,) int32; gd.peak (1,) int32, by atomic max.
    """
    b = wp.tid()  # int: the world
    g = int(0)  # int: its groups so far
    adr = idul[b, 0]  # int: the next group's first member slot
    for s in range(gm.nGSEG):  # s: int, a grouped segment
        for k in range(gm.nbin):  # k: int, a bin x sector
            slot = s * gm.nbin + k  # int: the (grouped segment, bin x sector) slot
            gd.cur[b, slot] = -1
            num = gd.cnt[b, slot]  # int: its contacts
            if num > 0 and gd.fsum[b, slot] > 0.0:
                gd.grp_seg[b, g] = gm.gseg[s]
                gd.grp_slot[b, g] = slot
                gd.grp_adr[b, g] = adr
                gd.grp_num[b, g] = num
                gd.cur[b, slot] = adr
                adr += num
                g += 1
    gd.ngrp[b] = g
    wp.atomic_max(gd.peak, 0, g)


@wp.kernel
def group_place(
    gm: GroupModel,  # type: ignore
    gd: GroupData,  # type: ignore
    idul: wp.array2d(dtype=int),  # type: ignore
    p_nCa3: wp.array(dtype=wp.vec3),  # type: ignore
    n_nCa3: wp.array(dtype=wp.vec3),  # type: ignore
    sdf_nCa3: wp.array(dtype=float),  # type: ignore
    member_pos: wp.array(dtype=wp.vec3),  # type: ignore
    member_frame: wp.array(dtype=wp.mat33),  # type: ignore
    member_dist: wp.array(dtype=float),  # type: ignore
    member_weight: wp.array(dtype=float),  # type: ignore
    member_slip: wp.array(dtype=float),  # type: ignore
):
    """
    Each grouped contact kept as its group's next member, in contact order: point, frame, depth, weight, friction ratio.

    Launch: dim=(B, nGSEG), a thread per world and grouped segment, by `Groups.before_export`.
    Reads: idul (B, nSEG + 1) int32; p_nCa3 and n_nCa3 (nCa3_max,) wp.vec3; sdf_nCa3 (nCa3_max,) float32;
    gd.key_nCa3 (nCa3_max,) int32; gd.t1_nCa3 (nCa3_max,) wp.vec3; gd.fbar_nCa3 and gd.slip_nCa3 (nCa3_max,)
    float32; gd.fsum (B, nGSEG * nbin) float32; gd.cur (B, nGSEG * nbin) int32, advanced past each member placed.
    Writes: MJWarp's members (ngroupmember = nCa3_max): member_pos (nCa3_max,) wp.vec3, member_frame (nCa3_max,)
    wp.mat33 with rows (n, t1, t2), member_dist, member_weight and member_slip (nCa3_max,) float32.
    """
    b, s = wp.tid()  # int, int: the world, the grouped segment
    base = s * gm.nbin  # int: its first slot
    sg = gm.gseg[s]  # int: its segment
    for i in range(idul[b, sg], idul[b, sg + 1]):  # i: int, a contact of the segment
        key = gd.key_nCa3[i]  # int: its bin x sector, -1 in no group
        if key < 0:
            continue
        slot = base + key  # int: its slot
        m = gd.cur[b, slot]  # int: its member slot, -1 where its group carries no force
        if m < 0:
            continue
        gd.cur[b, slot] = m + 1
        n = n_nCa3[i]  # wp.vec3: its normal
        t1 = gd.t1_nCa3[i]  # wp.vec3: its first tangent
        t2 = wp.cross(n, t1)  # wp.vec3: its second
        member_pos[m] = p_nCa3[i]
        member_frame[m] = wp.mat33(n[0], n[1], n[2], t1[0], t1[1], t1[2], t2[0], t2[1], t2[2])
        member_dist[m] = sdf_nCa3[i]
        member_weight[m] = gd.fbar_nCa3[i] / gd.fsum[b, slot]
        member_slip[m] = gd.slip_nCa3[i]


@wp.kernel
def export_groups(
    gm: GroupModel,  # type: ignore
    gd: GroupData,  # type: ignore
    co: ContactOut,  # type: ignore
    member_pos: wp.array(dtype=wp.vec3),  # type: ignore
    member_frame: wp.array(dtype=wp.mat33),  # type: ignore
    member_dist: wp.array(dtype=float),  # type: ignore
    member_weight: wp.array(dtype=float),  # type: ignore
    contact_group_adr: wp.array(dtype=int),  # type: ignore
    contact_group_num: wp.array(dtype=int),  # type: ignore
):
    """
    Each group as one MJWarp contact of its pair's geoms, the SDF side's first, after Lodos's per-contact ones: of
    condim 3 (cone) or 1 (ray), typed GROUP, pointing at its members. Its point, normal and depth are nominal, the
    members' weighted means; MJWarp reads the members.

    Launch: dim=(B, G_max), a thread per world and group slot, by `Groups.export`; G_max = nGSEG * nbin
    (`Groups.G`), the groups a world can have.
    Reads: gd.ngrp (B,) int32; gd.grp_seg, gd.grp_adr and gd.grp_num (B, G_max) int32; member_pos (nCa3_max,)
    wp.vec3, member_frame (nCa3_max,) wp.mat33, member_dist and member_weight (nCa3_max,) float32; gm's segments,
    geoms and pair parameters.
    Writes: co.nacon (1,) int32, by atomic add; at the contact slot taken, co.dist, pos, frame, geom, worldid,
    includemargin, dim, friction, solref, solreffriction, solimp, adhesion, type and geomcollisionid (naconmax,),
    and co.efc_address (naconmax, nmaxpyramid) int32, set to -1; contact_group_adr and contact_group_num
    (naconmax,) int32; gd.grp_cid (B, G_max) int32; gd.overflow (1,) int32, set when the contacts overflow.
    """
    b, g = wp.tid()  # int, int: the world, the group
    if g >= gd.ngrp[b]:
        return
    sg = gd.grp_seg[b, g]  # int: its segment
    adr = gd.grp_adr[b, g]  # int: its first member
    num = gd.grp_num[b, g]  # int: its members
    pbar = wp.vec3(0.0)  # wp.vec3: the members' weighted mean point
    nbar = wp.vec3(0.0)  # wp.vec3: their weighted mean normal
    dbar = float(0.0)  # float: their weighted mean signed distance (m)
    for m in range(adr, adr + num):  # m: int, a member
        a = member_weight[m]  # float: its weight
        fr = member_frame[m]  # wp.mat33: its frame, rows (n, t1, t2)
        pbar += a * member_pos[m]
        nbar += a * wp.vec3(fr[0, 0], fr[0, 1], fr[0, 2])
        dbar += a * member_dist[m]
    cid = wp.atomic_add(co.nacon, 0, 1)  # int: the group's MJWarp contact
    if cid >= co.naconmax:
        gd.grp_cid[b, g] = -1
        gd.overflow[0] = 1
        return
    gd.grp_cid[b, g] = cid
    nrm = wp.normalize(nbar)  # wp.vec3: the nominal normal
    t1 = _tangent(nrm)  # wp.vec3
    t2 = wp.cross(nrm, t1)  # wp.vec3
    j = gm.seg_cpr[sg]  # int: the pair
    co.dist[cid] = dbar
    co.pos[cid] = pbar
    co.frame[cid] = wp.mat33(nrm[0], nrm[1], nrm[2], t1[0], t1[1], t1[2], t2[0], t2[1], t2[2])
    co.geom[cid] = wp.vec2i(gm.g_geom[gm.seg_g_s[sg]], gm.g_geom[gm.seg_g_v[sg]])
    co.worldid[cid] = b
    co.includemargin[cid] = 0.0
    if gm.seg_kind[sg] == CONE:
        co.dim[cid] = 3
    else:
        co.dim[cid] = 1
    co.friction[cid] = gm.cpr_friction[j]
    co.solref[cid] = gm.cpr_solref[j]
    co.solreffriction[cid] = wp.vec2(0.0, 0.0)
    co.solimp[cid] = gm.cpr_solimp[j]
    co.adhesion[cid] = 0.0
    co.type[cid] = GROUP_CONTACT
    co.geomcollisionid[cid] = 0
    for k in range(co.efc_address.shape[1]):  # k: int, one of the contact's efc address slots
        co.efc_address[cid, k] = -1
    contact_group_adr[cid] = adr
    contact_group_num[cid] = num


def resolve_kinds(groups, nCPR):
    """
    The solver's `groups` option as one kind per Lodos pair: None (every pair per contact), a kind for every pair
    ("contact", "ray", "cone"), or a sequence of `nCPR` kinds.

    Called by `lodos.sim.make_sim`.
    In: groups, None, a str or a sequence of nCPR str; nCPR int. Out: (nCPR,) list of int, CONTACT, RAY or CONE.
    Raises ValueError on a wrong count or an unknown kind.
    """
    if groups is None:
        return [0] * nCPR
    if isinstance(groups, str):
        groups = [groups] * nCPR
    groups = list(groups)
    if len(groups) != nCPR:
        raise ValueError(f"groups: {len(groups)} kinds for {nCPR} Lodos pairs")
    for k in groups:  # k: str, a pair's kind
        if k not in KINDS:
            raise ValueError(f"groups: unknown kind {k!r}, expected one of {KINDS}")
    return [KINDS.index(k) for k in groups]


def check_bins(nnorm, nmom):
    """
    The bins a group takes: `lodos.cd.bins`' normal bins and moment sectors.

    Called by `Groups.__init__` and `lodos.cd.driver.resolve_options`.
    In: nnorm, nmom int. Raises ValueError where `lodos.cd.bins` has no such bins.
    """
    if int(nnorm) not in cd_bins.NORMAL_BINS:
        raise ValueError(f"normal_bins: {nnorm}, expected one of {cd_bins.NORMAL_BINS}")
    if int(nmom) not in MOMENT_BINS:
        raise ValueError(f"moment_bins: {nmom}, expected one of {MOMENT_BINS}")


class Groups:
    """
    The solver's contact groups: the model, the step's data, and the step's two passes -- `before_export` (keys, weights and
    frames; the groups and their members) and `export` (their contacts, after Lodos's per-contact ones). Builds
    MJWarp's group arrays (`lodos.cd.solver.io.alloc_contact_groups`) when a pair is grouped (`active`).

    mjm, m, d: MuJoCo's model and MJWarp's; drbd: Lodos's rigid bodies (qvel, cdof, subtree_com, xmat);
    mcol, dcol: Lodos's collision context; g_geom: the MJCF geom of each Lodos geometry; params13: the pairs'
    thirteen-column parameters; kinds: one per pair (`resolve_kinds`); nnorm, nmom: the normal bins and moment
    sectors; eps: the slip (m/s) below which a cone member's frame keeps its carried direction; friction, solref,
    solimp: the pairs' MJWarp parameters as Lodos's per-contact export holds them (`lodos.cd.driver.LodosContacts`),
    shared, or None to take them from `params13`; keep: the contact reduction's flags (`lodos.cd.reduce`), which
    then decide the grouped contacts too, or None for every contact.

    Built by `lodos.cd.driver.SolverDriver.__init__`; stepped by `lodos.cd.driver.SolverDriver.contact_dynamics`.
    """

    def __init__(self, mjm, m, d, drbd, mcol, dcol, g_geom, params13, kinds, nnorm=1, nmom=1, eps=1.0e-4,
                 friction=None, solref=None, solimp=None, keep=None):
        """
        The groups' model and data, and MJWarp's group arrays for `nCa3_max` members; only `seg_grouped` and
        `active` when no pair is grouped.

        Called by `lodos.cd.driver.SolverDriver.__init__`.
        In: the class docstring's, with g_geom (nB,) int; params13 (nCPR, 13) float; kinds (nCPR,) int; friction
        (nCPR,) vec5, solref (nCPR,) wp.vec2 and solimp (nCPR,) vec5 Warp arrays or None; keep (nCa3_max,) uint8 or
        None. Out: d's contact_group_* (naconmax,) and group_member_* (nCa3_max,) arrays, replaced.
        Raises ValueError on bins `check_bins` refuses.
        """
        check_bins(nnorm, nmom)
        nSEG, nCPR, B = int(mcol.nSEG), int(mcol.nCPR), int(d.nworld)  # int: segments, pairs, worlds
        seg_cpr = mcol.seg_cpr.numpy().astype(int)  # (nSEG,) int64: each segment's pair
        # (nSEG,) int64: each segment's kind, CONTACT, RAY or CONE
        seg_kind = np.asarray(kinds, dtype=int)[seg_cpr] if nSEG else np.zeros(0, dtype=int)
        self.seg_grouped = (seg_kind != 0).astype(np.int32)  # (nSEG,) int32: 1 for a grouped segment
        self.active = bool(self.seg_grouped.any())  # bool: some pair is grouped
        if not self.active:
            return
        gseg = np.flatnonzero(seg_kind != 0).astype(np.int32)  # (nGSEG,) int32: the grouped segments
        gm = GroupModel()  # GroupModel
        gm.nnorm, gm.nmom, gm.nbin = int(nnorm), int(nmom), int(nnorm) * int(nmom)
        gm.nGSEG = int(len(gseg))
        gm.eps = float(eps)
        gm.reduced = int(keep is not None)
        # (nCa3_max,) uint8: the reduction's flags; (1,), unread, without the reduction
        self.keep = keep if keep is not None else wp.zeros(1, dtype=wp.uint8)
        gm.gseg = wp.array(gseg, dtype=int)
        gm.seg_kind = wp.array(seg_kind.astype(np.int32), dtype=int)
        gm.seg_cpr, gm.seg_g_v, gm.seg_g_s = mcol.seg_cpr, mcol.seg_g_v, mcol.seg_g_s
        g_bid = mcol.g_bid.numpy().astype(int)  # (nB,) int64: each Lodos geometry's body
        anc = [ancestor_dofs_np(mjm, bid) for bid in g_bid]  # nB (k,) int64 arrays: the dofs that move each geometry
        gm.g_geom = wp.array(np.asarray(g_geom, dtype=np.int32), dtype=int)
        gm.g_bid = wp.array(g_bid.astype(np.int32), dtype=int)
        gm.g_rootid = wp.array(np.asarray(mjm.body_rootid)[g_bid].astype(np.int32), dtype=int)
        gm.g_anc_beg = wp.array(np.concatenate([[0], np.cumsum([len(a) for a in anc])]).astype(np.int32), dtype=int)
        gm.g_anc_dof = wp.array(np.concatenate(anc).astype(np.int32) if sum(len(a) for a in anc) else
                                np.zeros(1, np.int32), dtype=int)
        P = np.asarray(params13, dtype=np.float32).reshape(nCPR, 13)  # (nCPR, 13) float32: the pairs' parameters
        gm.cpr_stiffness = wp.array(P[:, 1], dtype=float)
        gm.cpr_sharpness = wp.array(P[:, 2], dtype=float)
        gm.cpr_dissipation = wp.array(P[:, 3], dtype=float)
        gm.cpr_stiction = wp.array(P[:, 4], dtype=float)
        mu = P[:, 0]  # (nCPR,) float32: the pairs' friction
        # as `lodos.cd.driver.SolverDriver` gives Lodos's own contacts: MuJoCo's torsional and rolling defaults
        # (nCPR, 5) float32
        fr = np.stack([mu, mu, np.full_like(mu, 0.005), np.full_like(mu, 1e-4), np.full_like(mu, 1e-4)], 1)
        gm.cpr_friction = wp.array(fr, dtype=vec5) if friction is None else friction
        gm.cpr_solref = wp.array(np.ascontiguousarray(P[:, 5:7]), dtype=wp.vec2) if solref is None else solref
        gm.cpr_solimp = wp.array(np.ascontiguousarray(P[:, 7:12]), dtype=vec5) if solimp is None else solimp
        dirs = cd_bins.bin_normals(int(nnorm))  # (nnorm, 3) float64: the bins' directions
        t1, t2 = cd_bins.bin_tangents(dirs)  # (nnorm, 3), (nnorm, 3) float64: each bin's tangents
        gm.bin_n = wp.array(dirs.astype(np.float32), dtype=wp.vec3)
        gm.bin_t1 = wp.array(t1.astype(np.float32), dtype=wp.vec3)
        gm.bin_t2 = wp.array(t2.astype(np.float32), dtype=wp.vec3)
        self.gm = gm  # GroupModel

        n = int(dcol.nCa3_max)  # int: the contact capacity, nCa3_max
        nslot = gm.nGSEG * gm.nbin  # int: (grouped segment, bin x sector) slots a world
        self.G = nslot  # groups a world: one per (grouped segment, bin x sector) at most
        gd = GroupData()  # GroupData
        gd.key_nCa3 = wp.full(n, -1, dtype=int)
        gd.fbar_nCa3 = wp.zeros(n, dtype=float)
        gd.t1_nCa3 = wp.zeros(n, dtype=wp.vec3)
        gd.slip_nCa3 = wp.zeros(n, dtype=float)
        gd.cnt = wp.zeros((B, nslot), dtype=int)
        gd.fsum = wp.zeros((B, nslot), dtype=float)
        gd.cur = wp.zeros((B, nslot), dtype=int)
        gd.ngrp = wp.zeros(B, dtype=int)
        gd.grp_seg = wp.zeros((B, nslot), dtype=int)
        gd.grp_slot = wp.zeros((B, nslot), dtype=int)
        gd.grp_adr = wp.zeros((B, nslot), dtype=int)
        gd.grp_num = wp.zeros((B, nslot), dtype=int)
        gd.grp_cid = wp.full((B, nslot), -1, dtype=int)
        gd.t1_body = wp.zeros((2, n), dtype=wp.vec3)
        gd.n_body = wp.zeros((2, n), dtype=wp.vec3)
        gd.vtx = wp.zeros((2, n), dtype=int)
        gd.parity = wp.zeros(1, dtype=int)
        gd.idul_prev = wp.zeros((B, nSEG + 1), dtype=int)
        gd.overflow = wp.zeros(1, dtype=int)
        gd.peak = wp.zeros(1, dtype=int)
        self.gd = gd  # GroupData
        self.B, self.n = B, n  # int, int: worlds, the contact capacity nCa3_max
        self.dcol, self.drbd, self.d = dcol, drbd, d  # CollisionData, RbdData, MJWarp's Data: the step's arrays
        alloc_contact_groups(m, d, n)

    def forget(self):
        """
        Drops the frames carried from the last step, for a state that does not follow it (a reset).

        Called by `viewer.session.Runner.set_state` and `bench.lodos_setting.start`.
        Out: gd.idul_prev (B, nSEG + 1) int32, zeroed: no contact finds last step's.
        """
        self.gd.idul_prev.zero_()

    def before_export(self):
        """
        This step's keys, weights and frames, its groups and their members; then its frames kept for the next.

        Called by `lodos.cd.driver.SolverDriver.contact_dynamics`, every step, before Lodos's contacts are exported.
        In: dcol's contacts (nCa3_max,) and idul_B_nSEG_to_nCa3 (B, nSEG + 1) int32; drbd.qvel (B, nv) float32,
        drbd.cdof (B, nv) wp.spatial_vector, drbd.subtree_com (B, nbody) wp.vec3, drbd.xmat (B, nbody) wp.mat33;
        `keep`. Out: gd; d.group_member_pos, frame, dist, weight and slip (nCa3_max,).
        """
        # GroupModel, GroupData, CollisionData, RbdData, Data
        gm, gd, dcol, drbd, d = self.gm, self.gd, self.dcol, self.drbd, self.d
        idul = dcol.idul_B_nSEG_to_nCa3  # (B, nSEG + 1) int32: each (world, segment)'s contacts
        wp.launch(group_contacts, dim=self.n,
                  inputs=(gm, gd, dcol.ct, dcol.b_nCa3, dcol.seg_nCa3, dcol.v_nCa3, dcol.p_nCa3, dcol.n_nCa3,
                          dcol.sdf_nCa3, drbd.qvel, drbd.cdof, drbd.subtree_com, drbd.xmat, self.keep))
        wp.launch(group_count, dim=(self.B, gm.nGSEG), inputs=(gm, gd, idul))
        wp.launch(group_assign, dim=self.B, inputs=(gm, gd, idul))
        wp.launch(group_place, dim=(self.B, gm.nGSEG),
                  inputs=(gm, gd, idul, dcol.p_nCa3, dcol.n_nCa3, dcol.sdf_nCa3, d.group_member_pos,
                          d.group_member_frame, d.group_member_dist, d.group_member_weight, d.group_member_slip))
        wp.launch(flip, dim=1, inputs=(gd,))
        wp.copy(gd.idul_prev, idul)

    def export(self, co):
        """
        The groups as MJWarp contacts, after Lodos's per-contact export.

        Called by `lodos.cd.driver.SolverDriver.contact_dynamics`, every step, after `export_lodos`.
        In: co (`lodos.col.prim.driver.ContactOut`), MJWarp's contact arrays (naconmax,) and their count.
        Out: a contact per group from co.nacon on; d.contact_group_adr and d.contact_group_num (naconmax,) int32;
        gd.grp_cid (B, G_max) int32, gd.overflow (1,) int32.
        """
        d = self.d  # Data: MJWarp's
        wp.launch(export_groups, dim=(self.B, self.G),
                  inputs=(self.gm, self.gd, co, d.group_member_pos, d.group_member_frame, d.group_member_dist,
                          d.group_member_weight, d.contact_group_adr, d.contact_group_num))
