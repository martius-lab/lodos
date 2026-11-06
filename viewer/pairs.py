"""
The collision pairs whose contact parameters the viewer's Collision pairs folder sets: friction, solref and
solimp per pair, and whether a Lodos pair is one- or two-sided.

Three kinds of pair, the first two under Lodos, the third under Newton's MuJoCo Warp:

- **Lodos's** (`SceneSource.surfaces.pairs`): the solver reads a row of parameters per pair (`lodos.cd.driver`'s
  `cpr_friction`, `cpr_solref`, `cpr_solimp`), so a pair takes what it is set. A pair is one-sided -- its
  first surface's vertices against its second's SDF -- or two-sided, both ways (`cpr_is_symmetric`), which
  rebuilds the sim. A pair with a plane is one-sided: a plane has no vertices.
- **MJWarp's** (its model's filtered geom pairs, `nxn_geom_pair_filtered`): MuJoCo resolves a contact's
  parameters from an explicit `<pair>`'s own, else from its two geoms -- friction the larger of theirs, solref
  and solimp mixed by `solmix`, a geom of higher priority taking all (MJWarp's `collision_core`). An explicit
  pair takes what it is set. For the rest each geom takes the min, max or mean (`GEOM_MODES`, the
  configuration's `mode`) of what is set on the pairs it is in, elementwise, and a geom on no set pair keeps
  its own; what MJWarp then applies to a pair (`PairTable.values`) can differ from what was set on it.
- **Newton's** (the shape pairs its collision pipeline collides, `newton_scene.Sim.pair_info`), under both of
  Newton's sims: SolverMuJoCo hands Newton's contacts to MuJoCo Warp, which resolves their parameters from the
  two shapes' geoms as above; Kamino takes friction alone, a contact's its two shapes' mixed by its rule (`mix`:
  their average, by default). Each shape takes the mode of what is set on its pairs, on the running model
  (`set_shape_params`).

A configuration's overrides (`no_overrides`, its `pairs`) hold, per kind, a value for every pair of the kind
(`all`, by field) and each pair's own (by field, by pair index), which applies over it: friction a float,
solref two, solimp five, sidedness 1 or 2 (for all, the pairs without a plane). A scene's twenty piled hands
as convex pieces have a million MJCF pairs, which is why an edit of all of them is one value, not a million.
The exact mode applies friction and sidedness alone: it takes no MuJoCo impedance.
"""

from __future__ import annotations

import mujoco
import numpy as np

from lodos.col.sdf import types as sdf_types

KINDS = ("lodos", "prim", "newton")
FIELDS = ("mu", "solref", "solimp")
WIDTH = {"mu": 1, "solref": 2, "solimp": 5}
GEOM_MODES = ("min", "max", "mean")
MINVAL = 1e-15  # MuJoCo's mjMINVAL: solmix below it counts as none (`collision_core`)


def kind_fields(kind):
    """What a kind of pair takes: friction, solref, solimp, and Lodos's sidedness."""
    return FIELDS + (("sided",) if kind == "lodos" else ())


def no_overrides():
    """A configuration's pair overrides with nothing set."""
    return {**{k: dict(all={}, **{f: {} for f in kind_fields(k)}) for k in KINDS}, "mode": "min"}


def has_overrides(ov, exact):
    """Whether a kind's overrides `ov` change anything the solver takes (the exact mode takes friction alone)."""
    fields = ("mu",) if exact else FIELDS
    return any(ov[f] or f in ov["all"] for f in fields)


def check(field, value):
    """`value` as a configuration holds it: a float, a tuple of two or five, or 1 or 2; ValueError else."""
    if field == "sided":
        if value not in (1, 2):
            raise ValueError(f"a pair is one-sided (1) or two-sided (2), not {value!r}")
        return int(value)
    v = np.asarray(value, dtype=np.float64).reshape(-1)
    if v.size != WIDTH[field] or not np.isfinite(v).all():
        raise ValueError(f"{field} takes {WIDTH[field]} finite numbers, not {value!r}")
    if field == "mu":
        if v[0] < 0:
            raise ValueError(f"friction cannot be negative ({v[0]:g})")
        return float(v[0])
    return tuple(float(x) for x in v)


def plane_pairs(surfaces):
    """`(npairs,)`: the Lodos pairs with a plane in them."""
    planes = [isinstance(s, sdf_types.PlaneSurface) for s in surfaces.surfaces]
    return np.array([planes[a] or planes[b] for a, b in surfaces.pairs], dtype=bool)


def scene_sided(surfaces):
    """`(npairs,)`: each Lodos pair's sidedness as the scene builds it, 1 or 2."""
    sym = surfaces.col_kwargs.get("cpr_is_symmetric")
    return np.where(np.asarray(sym, dtype=bool), 2, 1) if sym is not None else np.ones(len(surfaces.pairs), int)


def sided(surfaces, ov):
    """`(npairs,)`: each Lodos pair's sidedness under the overrides `ov` (a configuration's `pairs["lodos"]`)."""
    out = scene_sided(surfaces).copy()
    if "sided" in ov["all"]:
        out[~plane_pairs(surfaces)] = ov["all"]["sided"]
    for j, s in ov["sided"].items():
        out[j] = s
    return out


def sided_changes(surfaces, ov):
    """`{pair: 1 or 2}`: the Lodos pairs the overrides `ov` make other than the scene builds them."""
    own, now = scene_sided(surfaces), sided(surfaces, ov)
    return {int(j): int(now[j]) for j in np.nonzero(now != own)[0]}


def geom_label(mjm, g):
    """A geom's name, else its body's and its index."""
    name = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_GEOM, int(g))
    if name:
        return name
    b = int(mjm.geom_bodyid[g])
    return f"{mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_BODY, b) or f'body {b}'}/geom {int(g)}"


def _fmt(v):
    return " ".join(f"{x:.3g}" for x in np.atleast_1d(v))


def label(pairs, exact, kinds=("lodos", "prim")):
    """What a configuration's overrides of the pairs of `kinds` -- those its sim applies -- change, for a take's label."""
    parts = []
    names = dict(lodos="Lodos", prim="MJCF", newton="Newton")
    for kind in kinds:
        ov, name = pairs[kind], names[kind]
        for f in FIELDS:
            if exact and f != "mu":
                continue
            own = ov[f]
            if f in ov["all"]:
                parts.append(f"{name} {f} {_fmt(ov['all'][f])} on all pairs" + (f", {len(own)} their own" if own
                                                                                 else ""))
            elif own:
                vals = set(own.values())
                parts.append(f"{name} {f} {_fmt(next(iter(vals))) if len(vals) == 1 else 'set'} on {len(own)} pairs")
    if "lodos" in kinds:
        ov = pairs["lodos"]
        if "sided" in ov["all"]:
            parts.append(f"Lodos pairs all {'two' if ov['all']['sided'] == 2 else 'one'}-sided")
        for s, name in ((2, "two-sided"), (1, "one-sided")):
            n = sum(1 for v in ov["sided"].values() if v == s)
            if n:
                parts.append(f"{n} Lodos pairs {name}")
    for kind, name in (("prim", "MJCF geoms"), ("newton", "Newton shapes")):
        if kind in kinds and has_overrides(pairs[kind], exact):
            parts.append(f"{name} by {pairs['mode']}")
    return parts


def _reduce(base, idx, vals, mode):
    """`base` (n, w) with each row named in `idx` the min, max or mean of the `vals` rows naming it."""
    out = base.copy()
    if not len(idx):
        return out
    if mode == "min":
        acc = np.full(base.shape, np.inf)
        np.minimum.at(acc, idx, vals)
    elif mode == "max":
        acc = np.full(base.shape, -np.inf)
        np.maximum.at(acc, idx, vals)
    elif mode == "mean":
        acc = np.zeros(base.shape)
        np.add.at(acc, idx, vals)
        acc /= np.maximum(np.bincount(idx, minlength=len(base)), 1)[:, None]
    else:
        raise ValueError(f"a geom takes the min, max or mean of its pairs' values, not {mode!r}")
    hit = np.unique(idx)
    out[hit] = acc[hit]
    return out


def _own(d, width):
    """A field's own values by pair, `d`, as `(indices, (k, width) values)`."""
    k = np.fromiter(d.keys(), int, len(d))
    return k, np.asarray(list(d.values()), np.float64).reshape(len(k), width)


class PairTable:
    """
    One kind of pair of a built Lodos sim: `names` and each pair's two `ends` in them, the scene's values (`mu`
    (n,), `solref` (n, 2), `solimp` (n, 5); Lodos's `sided` and `plane` (n,)), `match` to filter the pairs by
    name, `values` and `marks` for what a configuration's overrides make of them. Make one with `lodos` or
    `prim`.
    """

    def __init__(self, kind, names, ends, mu, solref, solimp):
        self.kind = kind
        self.names, self.ends = names, ends
        self._lower = [s.lower() for s in names]
        self.mu, self.solref, self.solimp = mu, solref, solimp
        self.sided = self.plane = None  # Lodos's: (n,) 1 or 2 as the scene builds it, and the pairs with a plane
        # MJWarp's: each pair's explicit pair id (-1 for none); the scene's geom parameters (`friction`, `solref`,
        # `solimp`), solmix and priority; the explicit pairs' own
        self.pairid = self.geom = self.solmix = self.priority = self.pair = None
        # Newton's under Kamino: how a contact's friction mixes its two shapes' (None: MuJoCo's rules); the fields
        # its sim applies, which its list shows
        self.mix, self.fields = None, FIELDS
        self._matches = {}

    def __len__(self):
        return len(self.ends)

    @classmethod
    def lodos(cls, sim, defaults, scene):
        """
        Lodos's pairs of `sim`, at the parameters it was built with (`Runner._read_defaults`), and their
        sidedness as the scene's surfaces `scene` have it (a sim built with pairs made two-sided has others).
        """
        s, mjm = sim.surfaces, sim.mjm
        fr = np.asarray(defaults["cpr_friction"], np.float64)
        t = cls("lodos", [geom_label(mjm, g) for g in s.geoms], np.asarray(s.pairs, int).reshape(-1, 2),
                fr[:, 0].copy(), np.asarray(defaults["cpr_solref"], np.float64),
                np.asarray(defaults["cpr_solimp"], np.float64))
        t.sided, t.plane = scene_sided(scene), plane_pairs(s)
        return t

    @classmethod
    def prim(cls, sim, defaults):
        """MJWarp's pairs of `sim` (its filtered geom pairs), as it resolves their parameters from the scene's."""
        m, mjm = sim.m, sim.mjm
        ends = m.nxn_geom_pair_filtered.numpy().astype(int).reshape(-1, 2)
        n = len(ends)
        t = cls("prim", [geom_label(mjm, g) for g in range(int(mjm.ngeom))], ends, np.zeros(n), np.zeros((n, 2)),
                np.zeros((n, 5)))
        t.pairid = m.nxn_pairid_filtered.numpy().reshape(-1, 2)[:, 0].astype(int) if n else np.zeros(0, int)
        t.geom = {k: np.asarray(defaults[f"geom_{k}"], np.float64)[0] for k in ("friction", "solref", "solimp")}
        t.solmix = np.asarray(mjm.geom_solmix, np.float64)
        t.priority = np.asarray(mjm.geom_priority, int)
        t.pair = ({k: np.asarray(defaults[f"pair_{k}"], np.float64)[0] for k in ("friction", "solref", "solimp")}
                  if "pair_friction" in defaults else None)
        t.mu, t.solref, t.solimp = t._resolve(t.geom, t.pair)
        return t

    # -------------------------------------------------------------------------------

    @classmethod
    def newton(cls, info):
        """
        Newton's shape pairs, from the Newton process (`newton_scene.Sim.pair_info`): its shape names, the pairs, and
        each shape's parameters as built, which MuJoCo Warp resolves a pair's from -- or under Kamino its friction,
        mixed by Kamino's rule (`mix`).
        """
        ends = np.asarray(info["pairs"], int).reshape(-1, 2)
        n = len(ends)
        t = cls("newton", list(info["names"]), ends, np.zeros(n), np.zeros((n, 2)), np.zeros((n, 5)))
        t.pairid = np.full(n, -1, int)  # no explicit pairs: Newton's contacts are its pipeline's
        t.geom = {k: np.asarray(info[k], np.float64) for k in ("friction", "solref", "solimp")}
        t.solmix = np.asarray(info["solmix"], np.float64)
        t.priority = np.asarray(info["priority"], int)
        t.mix = info.get("mix")
        if t.mix is not None:
            t.fields = ("mu",)
        t.mu, t.solref, t.solimp = t._resolve(t.geom, t.pair)
        return t

    def label(self, i):
        a, b = self.ends[i]
        return f"{self.names[a]} / {self.names[b]}"

    def _holding(self, words, side=None):
        """`(n,)`: the pairs whose names hold every word -- the first name (`side` 0), the second (1), or either."""
        ok = np.ones(len(self), bool)
        for w in words:
            has = np.array([w in s for s in self._lower], bool)
            ok &= has[self.ends[:, side]] if side is not None else (has[self.ends[:, 0]] | has[self.ends[:, 1]])
        return ok

    def match(self, text):
        """
        The indices of the pairs `text` names (any case): every word in the pair's names, or with a `/`, the words
        before it in the first name and those after it in the second. All of them for no words.
        """
        key = " ".join(text.lower().split())
        hit = self._matches.get(key)
        if hit is None:
            left, sep, right = key.partition("/")
            ok = (self._holding(left.split(), 0) & self._holding(right.split(), 1)) if sep else \
                self._holding(key.split())
            hit = np.nonzero(ok)[0]
            if len(self._matches) > 64:
                self._matches.clear()
            self._matches[key] = hit
        return hit

    def values(self, pairs, exact):
        """
        `{field: (n, ...)}` as the solver applies them under the configuration's overrides `pairs` -- and Lodos's
        `sided` -- with `exact` (the exact mode) taking friction alone.
        """
        if self.kind == "lodos":
            ov = pairs["lodos"]
            out = {"mu": self.mu.copy(), "solref": self.solref.copy(), "solimp": self.solimp.copy()}
            for f in FIELDS:
                if exact and f != "mu":
                    continue
                if f in ov["all"]:
                    out[f][:] = ov["all"][f]
                if ov[f]:
                    k, v = _own(ov[f], WIDTH[f])
                    out[f][k] = v.reshape(out[f][k].shape)
            s = self.sided.copy()
            if "sided" in ov["all"]:
                s[~self.plane] = ov["all"]["sided"]
            for j, x in ov["sided"].items():
                s[j] = x
            out["sided"] = s
            return out
        g = self.geom_arrays(pairs, exact)
        mu, solref, solimp = self._resolve(g, g["pair"])
        return {"mu": mu, "solref": solref, "solimp": solimp}

    def marks(self, pairs):
        """`{field: (n,) bool}`: the pairs a field is set on, by the value for all or by their own."""
        ov = pairs[self.kind]
        out = {}
        for f in kind_fields(self.kind):
            m = np.zeros(len(self), bool)
            if f in ov["all"]:
                m[:] = True if f != "sided" else ~self.plane
            if ov[f]:
                m[np.fromiter(ov[f].keys(), int, len(ov[f]))] = True
            out[f] = m
        return out

    def geom_arrays(self, pairs, exact):
        """
        MJWarp's and Newton's: `{"friction": (ngeom, 3), "solref": (ngeom, 2), "solimp": (ngeom, 5), "pair": ...}`,
        the geoms' (Newton's shapes') parameters under the overrides `pairs` (each geom the `mode` of what is set on
        its pairs), and the explicit pairs' own (`pair`: `friction` (npair, 5), `solref`, `solimp`; None without any).
        """
        ov, mode = pairs[self.kind], pairs["mode"]
        out = {k: v.copy() for k, v in self.geom.items()}
        pair = None if self.pair is None else {k: v.copy() for k, v in self.pair.items()}
        n = len(self)
        for f in FIELDS:
            every = ov["all"].get(f)
            if (every is None and not ov[f]) or (exact and f != "mu"):
                continue
            if every is not None:  # every pair: the value for all, a pair's own over it
                k = np.arange(n)
                v = np.tile(np.atleast_1d(np.asarray(every, np.float64)), (n, 1))
                if ov[f]:
                    ko, vo = _own(ov[f], WIDTH[f])
                    v[ko] = vo
            else:
                k, v = _own(ov[f], WIDTH[f])
            own = self.pairid[k] >= 0
            if own.any():  # an explicit pair takes what it is set
                pid = self.pairid[k[own]]
                if f == "mu":
                    pair["friction"][pid, :2] = v[own]
                else:
                    pair[f][pid] = v[own]
            k, v = k[~own], v[~own]
            idx = np.concatenate([self.ends[k, 0], self.ends[k, 1]])
            vals = np.concatenate([v, v])
            if f == "mu":
                out["friction"][:, :1] = _reduce(out["friction"][:, :1], idx, vals, mode)
            else:
                out[f] = _reduce(out[f], idx, vals, mode)
        out["pair"] = pair
        return out

    def _resolve(self, geom, pair):
        """
        `(mu, solref, solimp)` per pair, as MJWarp's `contact_material_params` resolves them; under Kamino the
        friction as its rule mixes it (`mix`), with no solref or solimp.
        """
        g1, g2 = self.ends[:, 0], self.ends[:, 1]
        if self.mix is not None:
            a, b = geom["friction"][g1, 0], geom["friction"][g2, 0]
            mu = {"average": 0.5 * (a + b), "multiply": a * b, "max": np.maximum(a, b), "min": np.minimum(a, b)}
            return mu[self.mix], np.zeros((len(self), 2)), np.zeros((len(self), 5))
        p1, p2 = self.priority[g1], self.priority[g2]
        s1, s2 = self.solmix[g1], self.solmix[g2]
        with np.errstate(divide="ignore", invalid="ignore"):
            mix = np.where(s1 + s2 > 0, s1 / (s1 + s2), 0.5)
        mix = np.where((s1 < MINVAL) & (s2 < MINVAL), 0.5, mix)
        mix = np.where((s1 < MINVAL) & (s2 >= MINVAL), 0.0, mix)
        mix = np.where((s1 >= MINVAL) & (s2 < MINVAL), 1.0, mix)
        mix = np.where(p1 > p2, 1.0, np.where(p2 > p1, 0.0, mix))[:, None]
        f1, f2 = geom["friction"][g1], geom["friction"][g2]
        fr = np.where((p1 > p2)[:, None], f1, np.where((p2 > p1)[:, None], f2, np.maximum(f1, f2)))
        r1, r2 = geom["solref"][g1], geom["solref"][g2]
        both = ((r1[:, 0] > 0) & (r2[:, 0] > 0))[:, None]
        solref = np.where(both, mix * r1 + (1 - mix) * r2, np.minimum(r1, r2))
        solimp = mix * geom["solimp"][g1] + (1 - mix) * geom["solimp"][g2]
        mu = fr[:, 0].copy()
        own = self.pairid >= 0
        if own.any() and pair is not None:
            pid = self.pairid[own]
            mu[own] = pair["friction"][pid, 0]
            solref[own] = pair["solref"][pid]
            solimp[own] = pair["solimp"][pid]
        return mu, solref, solimp

    def row(self, i, vals, marks):
        """Pair `i` as the folder's list shows it: its values as applied, `*` on those set on it."""
        mark = lambda f: "*" if marks[f][i] else ""  # noqa: E731
        txt = f"{self.label(i)}   mu{mark('mu')} {vals['mu'][i]:.3g}"
        if "solref" in self.fields:
            txt += f"   solref{mark('solref')} {_fmt(vals['solref'][i])}   solimp{mark('solimp')} {_fmt(vals['solimp'][i])}"
        if self.kind == "lodos":
            txt += ("   two-sided" if vals["sided"][i] == 2 else "   one-sided") + mark("sided")
        return txt
