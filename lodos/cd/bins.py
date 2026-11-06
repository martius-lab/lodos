"""
Direction bins for contacts: the polyhedra whose face normals split a pair's contacts
by the direction of their normals.

A contact group (`lodos.cd.groups`) is a pair segment's contacts in one normal bin and
moment sector, and `lodos.cd.reduce` keeps a few contacts of each bin, as Newton's
collision pipeline does with the same icosahedron.
"""

import numpy as np

# How many normal-direction bins contacts can be split into: 1, or 6/12/20, the face
# normals of a cube, a dodecahedron and an icosahedron. See `bin_normals`.
NORMAL_BINS = (1, 6, 12, 20)
# How many sectors each normal bin is split into by the direction of a contact's moment
# arm, so that contacts pulling the same way on opposite sides of a body are grouped apart.
MOMENT_BINS = (1, 2, 3, 4, 6)


def bin_normals(nbin):
    """
    The bin directions for `normal_bins=nbin`, `(nbin, 3)` unit vectors; a contact
    belongs to the bin whose direction is closest to its normal.

    Each set is turned so that +z is the centre of a bin -- a cube's face, a
    dodecahedron's face (an icosahedron's vertex), an icosahedron's face -- so the
    contacts of a floor, whose normals are +z to rounding, share one bin rather than
    landing on a bin boundary and splitting by noise. Newton's binning, which this
    follows, puts +y through an icosahedron vertex instead; with z up that leaves a
    floor's normal on the edge between two bins.

    Called by `lodos.cd.groups.Groups.__init__` and `lodos.cd.reduce.make_reduce_context`.
    In: nbin int, one of `NORMAL_BINS`. Out: (nbin, 3) float64.
    """
    nbin = int(nbin)
    assert nbin in NORMAL_BINS, f"normal_bins={nbin}, expected one of {NORMAL_BINS}"
    if nbin == 1:
        return np.array([[0.0, 0.0, 1.0]])
    if nbin == 6:
        return np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float)
    phi = (1.0 + np.sqrt(5.0)) / 2.0  # float: the golden ratio
    # (12, 3) float64: the icosahedron's vertices
    ico = np.array([[0, s1, s2 * phi] for s1 in (-1, 1) for s2 in (-1, 1)]
                   + [[s1, s2 * phi, 0] for s1 in (-1, 1) for s2 in (-1, 1)]
                   + [[s2 * phi, 0, s1] for s1 in (-1, 1) for s2 in (-1, 1)], dtype=float)
    if nbin == 12:
        d = ico  # (nbin, 3) float64: the bin directions, not yet of unit length
    else:
        # The icosahedron's faces: the vertex triples that are pairwise an edge apart.
        edge = np.min(np.linalg.norm(ico[1:] - ico[0], axis=1))  # float: the edge length
        # (12, 12) bool: the vertex pairs an edge apart
        adj = np.abs(np.linalg.norm(ico[:, None] - ico[None], axis=2) - edge) < 1e-9
        # list of 20 (int, int, int): the faces' vertex triples
        faces = [(i, j, k) for i in range(12) for j in range(i + 1, 12) for k in range(j + 1, 12)
                 if adj[i, j] and adj[j, k] and adj[i, k]]
        assert len(faces) == 20
        d = np.array([ico[list(f)].sum(0) for f in faces])
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    # Rotate the direction nearest +z onto +z (Rodrigues), then sort for a fixed order.
    a = d[np.argmax(d[:, 2])]  # (3,) float64: the direction nearest +z
    v, c = np.cross(a, [0.0, 0.0, 1.0]), float(a[2])  # (3,) float64, float: a x +z and a . +z
    if np.linalg.norm(v) > 1e-12:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])  # (3, 3) float64: v's cross matrix
        R = np.eye(3) + vx + vx @ vx / (1.0 + c)  # (3, 3) float64: the rotation taking a onto +z
        d = d @ R.T
    d[np.abs(d) < 1e-12] = 0.0
    return d[np.lexsort((d[:, 0], d[:, 1], -d[:, 2]))]


def bin_tangents(dirs):
    """
    A fixed orthonormal pair (t1, t2) in the plane normal to each bin direction.

    Called by `lodos.cd.groups.Groups.__init__` and `lodos.cd.reduce.make_reduce_context`.
    In: dirs (nbin, 3) float64, unit directions (`bin_normals`). Out: t1, t2 (nbin, 3) float64.
    """
    t1 = np.zeros_like(dirs)  # (nbin, 3) float64
    for k, d in enumerate(dirs):  # k: int, d: (3,) float64, a bin and its direction
        # (3,) float64: an axis away from d
        a = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        t = a - (a @ d) * d  # (3,) float64: a's part normal to d
        t1[k] = t / np.linalg.norm(t)
    return t1, np.cross(dirs, t1)
