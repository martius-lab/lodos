"""Close a CAD tessellation's T-junction seams exactly: split each face at the vertices lying on its edges."""
import sys, numpy as np, trimesh
from collections import defaultdict
from scipy.spatial import cKDTree


def boundary_edges(F):
    e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    u, cnt = np.unique(e, axis=0, return_counts=True)
    return u[cnt == 1], u[cnt > 2]


def split_points(V, F, tol):
    """For every boundary edge, the boundary vertices strictly inside it, ordered from its first endpoint."""
    bnd, _ = boundary_edges(F)
    bv = np.unique(bnd.ravel())
    tree = cKDTree(V[bv])
    out = {}
    for a, b in bnd:
        pa, pb = V[a], V[b]
        d = pb - pa
        L = np.linalg.norm(d)
        cand = tree.query_ball_point((pa + pb) / 2, L / 2 + tol)
        pts = []
        for ci in cand:
            k = bv[ci]
            if k == a or k == b:
                continue
            t = np.dot(V[k] - pa, d) / (L * L)
            if t <= 1e-9 or t >= 1 - 1e-9:
                continue
            if np.linalg.norm(pa + t * d - V[k]) <= tol:
                pts.append((t, k))
        if pts:
            out[(a, b)] = [k for _, k in sorted(pts)]
    return out


def triangulate(poly, V, n):
    """
    A convex polygon with collinear runs (a triangle with points inserted on its edges),
    by ear clipping. An ear is valid only if it has positive area and no other polygon
    vertex lies inside it or on its boundary -- a corner whose diagonal runs along an
    edge carrying inserted points would otherwise hand back the original face.
    """
    poly = list(poly)
    tris = []
    nn = n / np.linalg.norm(n)
    def area(i, j, k):
        return np.dot(np.cross(V[j] - V[i], V[k] - V[i]), nn)
    def inside(s, i, j, k, A):
        # barycentric coordinates of s in (i, j, k), relative to the ear's area
        w0 = area(s, j, k) / A
        w1 = area(i, s, k) / A
        w2 = area(i, j, s) / A
        return min(w0, w1, w2) >= -1e-9
    while len(poly) > 3:
        m = len(poly)
        best, bi = -1.0, -1
        for i in range(m):
            p, q, r = poly[i - 1], poly[i], poly[(i + 1) % m]
            A = area(p, q, r)
            if A <= 1e-18:
                continue
            if any(inside(s, p, q, r, A) for s in poly if s not in (p, q, r)):
                continue
            # prefer the ear with the best shape: largest area over squared longest edge
            L = max(np.linalg.norm(V[q] - V[p]), np.linalg.norm(V[r] - V[q]), np.linalg.norm(V[p] - V[r]))
            score = A / (L * L)
            if score > best:
                best, bi = score, i
        if bi < 0:
            raise RuntimeError("no valid ear")
        m = len(poly)
        tris.append((poly[bi - 1], poly[bi], poly[(bi + 1) % m]))
        poly.pop(bi)
    tris.append(tuple(poly))
    return tris


def repair(V, F, tol):
    for it in range(6):
        sp = split_points(V, F, tol)
        if not sp:
            break
        newF = []
        nsplit = 0
        for f in F:
            a, b, c = (int(x) for x in f)
            poly = []
            hit = False
            for p, q in ((a, b), (b, c), (c, a)):
                poly.append(p)
                key, rev = ((p, q), False) if p < q else ((q, p), True)
                if key in sp:
                    ins = sp[key][::-1] if rev else sp[key]
                    poly.extend(ins)
                    hit = True
            if not hit:
                newF.append((a, b, c))
                continue
            nsplit += 1
            n = np.cross(V[b] - V[a], V[c] - V[a])
            newF.extend(triangulate(poly, V, n))
        F = np.array(newF, dtype=np.int64)
        print(f"   pass {it}: {len(sp)} edges carried vertices, {nsplit} faces split -> {len(F)} faces", flush=True)
    return F


# Usage: tjunction.py TOL OUT_DIR SRC.obj ...   (writes OUT_DIR/<src>_wt.obj)
TOL, OUT_DIR = float(sys.argv[1]), sys.argv[2]
for src in sys.argv[3:]:
    m = trimesh.load(src, force="mesh")
    m = trimesh.Trimesh(m.vertices, m.faces, process=True)  # merge near-coincident vertices first
    V, F = np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int64)
    bnd0, nm0 = boundary_edges(F)
    print(f"   after merging: {len(V)} verts, {len(F)} faces, boundary edges {len(bnd0)}, non-manifold {len(nm0)}")
    area0 = m.area
    # drop zero-area faces first: they carry no surface and confuse the edge counts
    A = np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1) / 2
    F = F[A > 1e-16]
    print(f"== {src.split('/')[-1]}: {len(V)} verts, {len(F)} faces (dropped {int((A <= 1e-16).sum())} zero-area)")
    F = repair(V, F, tol=TOL)
    out = trimesh.Trimesh(V, F, process=True)  # merge, drop unreferenced
    bnd, nonman = boundary_edges(np.asarray(out.faces))
    Af = out.area_faces
    print(f"   result: {len(out.vertices)} verts, {len(out.faces)} faces, watertight {out.is_watertight}, winding consistent {out.is_winding_consistent}, "
          f"boundary edges {len(bnd)}, non-manifold edges {len(nonman)}, euler {out.euler_number}, volume {out.volume * 1e6:.4f} cm^3, "
          f"area {out.area * 1e4:.4f} cm^2 (input {area0 * 1e4:.4f}), min face area {Af.min():.2e}")
    out.export(f"{OUT_DIR}/{src.split('/')[-1].replace('.obj', '_wt.obj')}")
