"""
Whether a mesh is a single closed 2-manifold, and how it fails if not.

Shared by the step that produces the failures (`interactive_remesh.py`) and the one
that repairs them (`fix_nonmanifold_meshes.py`). Nothing here imports from outside
this directory.

`trimesh.is_watertight` answers only the combined question -- every edge has exactly
two faces -- so it reads the same whichever way the mesh broke. The three counts
here separate cases that want different answers:

    bnd   edges with a single face. A hole. Nothing can sign a field through one,
          and no local edit closes it -- the vertex count has to change.
    nm    edges with more than two. The surface touches itself, where a pinch in the
          input came out thinner than one output triangle. The multiplicity stays
          even, so the enclosed volume is still well defined.
    dup   a face repeated. In practice the repeats are oppositely wound, a
          zero-thickness flap enclosing nothing, and deleting every copy restores
          the manifold without moving a vertex or changing the volume.

`shells` counts face-connected components, which a pinch inflates: cutting the
adjacency at a non-manifold edge separates sheets that share a vertex.
"""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import numpy as np
import trimesh


def topology(mesh):
    """`bnd`, `nm`, `dup`, `shells` and the `ok` that is all four being trivial."""
    f = np.asarray(mesh.faces)
    incident = Counter(map(tuple, np.sort(np.asarray(mesh.edges), axis=1)))
    t = SimpleNamespace(
        bnd=sum(1 for n in incident.values() if n == 1),
        nm=sum(1 for n in incident.values() if n > 2),
        dup=len(f) - len(np.unique(np.sort(f, axis=1), axis=0)),
        shells=len(mesh.split(only_watertight=False)),
    )
    t.ok = t.bnd == 0 and t.nm == 0 and t.dup == 0 and t.shells == 1
    return t


def topo_tag(t, clean=""):
    """A topology as a short label, `clean` when there is nothing to say."""
    if t.ok:
        return clean
    parts = []
    if t.bnd:
        parts.append(f"open {t.bnd}e")
    if t.nm:
        parts.append(f"nonman {t.nm}e")
    if t.dup:
        parts.append(f"{t.dup} dup f")
    if t.shells != 1:
        parts.append(f"{t.shells}sh")
    return " ".join(parts)


def drop_flaps(mesh):
    """
    Delete every copy of each repeated face; returns `(mesh, faces_dropped, verts_dropped)`.

    Every copy, not all but one. A repeated face here is a zero-thickness flap: the
    copies are oppositely wound and enclose nothing, so both belong to the same
    defect and removing one would leave the other bounding a hole. Removing both
    takes each of the flap's edges from four incident faces down to two, and the
    volume is unchanged to the last bit because the flap bounded none of it.

    A vertex used only by the flap is then left with no face at all, and is dropped
    here: `Trimesh.export` writes only referenced vertices, so keeping them would
    mean the mesh in memory and the OBJ on disk disagreed about the contact set.
    Doing it here makes the count visible and reportable instead. On the SHARPA hand
    it is 0 to 6 vertices out of hundreds, and they sit on the degenerate flap rather
    than on real surface.

    Every *surviving* vertex is passed through by value, and `process=False` keeps
    trimesh from welding or reordering on its own.
    """
    f = np.asarray(mesh.faces)
    v = np.asarray(mesh.vertices)
    if len(f) == 0:
        return mesh, 0, 0
    repeats = Counter(map(tuple, np.sort(f, axis=1)))
    keep = np.array([repeats[tuple(sorted(t))] == 1 for t in f])
    if keep.all():
        return mesh, 0, 0

    f = f[keep]
    used = np.unique(f)
    remap = np.full(len(v), -1)
    remap[used] = np.arange(len(used))
    out = trimesh.Trimesh(v[used].copy(), remap[f], process=False)
    return out, int((~keep).sum()), int(len(v) - len(used))


def fan(verts, polys):
    """A triangle `Trimesh` from a polygon mesh, for the checks above.

    For *display* only. Do not run `topology` on the result of this: fanning is
    not topology preserving, and the failure is not subtle.

    A diagonal introduced by fanning is two-incident on its own, but nothing stops
    it coinciding with an edge that already exists elsewhere in the mesh -- on a
    quad mesh that is common, since one quad's diagonal is often a neighbour's
    edge. That edge then counts four incidences and reads as non-manifold, and two
    distinct quads can fan to the same triangle and read as duplicates. Measured on
    the armadillo quad remesh: fanned it reports 9 shells, 8 non-manifold edges and
    8 duplicate faces; the mesh itself is a single closed surface with none of
    those, Euler characteristic exactly 2. Use `topology_polys`.
    """
    tris = [[p[0], p[i], p[i + 1]] for p in polys for i in range(1, len(p) - 1)]
    return trimesh.Trimesh(np.asarray(verts, dtype=np.float64),
                           np.asarray(tris, dtype=np.int64), process=False)


def drop_flaps_polys(verts, polys):
    """
    `drop_flaps` for a polygon mesh: `(verts, polys, faces_dropped, verts_dropped)`.

    Same rule and the same reasoning -- every copy of a repeated face goes, not all
    but one, because the copies are oppositely wound and enclose nothing, so
    removing one would leave the other bounding a hole.

    Done on the polygons rather than on a fanned triangulation because a quad and
    its flap share a vertex *set*, which is what identifies them, while their
    fanned triangles need not even be the same triangles -- a quad fanned from a
    different starting corner splits along the other diagonal. Matching sets here
    is both simpler and stricter.
    """
    verts = np.asarray(verts, dtype=np.float64)
    if not polys:
        return verts, polys, 0, 0

    repeats = Counter(tuple(sorted(p)) for p in polys)
    keep = [p for p in polys if repeats[tuple(sorted(p))] == 1]
    dropped_f = len(polys) - len(keep)
    if dropped_f == 0:
        return verts, polys, 0, 0

    used = (np.unique(np.concatenate([np.asarray(p) for p in keep]))
            if keep else np.zeros(0, dtype=np.int64))
    remap = -np.ones(len(verts), dtype=np.int64)
    remap[used] = np.arange(len(used))
    return (verts[used],
            [[int(remap[i]) for i in p] for p in keep],
            dropped_f,
            int(len(verts) - len(used)))


def topology_polys(verts, polys):
    """`topology`, computed on the polygons themselves rather than a triangulation.

    Same fields, so it is a drop-in. On a triangle mesh it agrees with `topology`
    exactly; on a quad mesh `topology` cannot be trusted at all -- see `fan`.

    Shells are connected components over faces sharing an *edge*, which is the
    same relation `Trimesh.split` uses, so the counts mean the same thing.
    """
    inc = Counter()
    for p in polys:
        for i in range(len(p)):
            inc[tuple(sorted((p[i], p[(i + 1) % len(p)])))] += 1

    parent = list(range(len(polys)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    owner = {}
    for fi, p in enumerate(polys):
        for i in range(len(p)):
            e = tuple(sorted((p[i], p[(i + 1) % len(p)])))
            if e in owner:
                ra, rb = find(owner[e]), find(fi)
                if ra != rb:
                    parent[ra] = rb
            else:
                owner[e] = fi

    t = SimpleNamespace(
        bnd=sum(1 for n in inc.values() if n == 1),
        nm=sum(1 for n in inc.values() if n > 2),
        dup=len(polys) - len({tuple(sorted(p)) for p in polys}),
        shells=len({find(i) for i in range(len(polys))}) if polys else 0,
        euler=len(verts) - len(inc) + len(polys),
    )
    t.ok = t.bnd == 0 and t.nm == 0 and t.dup == 0 and t.shells == 1
    return t
