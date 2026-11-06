"""Instant Meshes at a fixed edge length, reprojected onto the repaired original, flaps removed, small holes closed."""
import sys, subprocess, numpy as np, trimesh, warp as wp
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # mesh_processing
from instant_meshes import read_obj, BINARY
from topology import topology, drop_flaps, topo_tag
wp.config.quiet = True


def reproject(orig, V):
    wm = wp.Mesh(points=wp.array(orig.vertices.astype(np.float32), dtype=wp.vec3),
                 indices=wp.array(orig.faces.astype(np.int32).ravel(), dtype=int))
    @wp.kernel
    def closest(mid: wp.uint64, p: wp.array(dtype=wp.vec3), cp: wp.array(dtype=wp.vec3)):
        i = wp.tid()
        q = wp.mesh_query_point_no_sign(mid, p[i], 1.0)
        if q.result:
            cp[i] = wp.mesh_eval_position(mid, q.face, q.u, q.v)
    P = wp.array(V.astype(np.float32), dtype=wp.vec3)
    CP = wp.zeros_like(P)
    wp.launch(closest, dim=len(V), inputs=[wm.id, P, CP])
    return CP.numpy().astype(np.float64)


def fill_small_holes(m, max_edges=8):
    """
    Close every boundary loop of at most `max_edges` edges with a fan wound against the loop.

    Only a boundary that closes into a simple loop is filled. One that branches at a vertex (two
    boundary edges leaving it, as Instant Meshes left on the M16 NIST nut at 0.2 mm) stays open:
    tracing it runs into a vertex with no next edge, or ends on another loop's vertex.
    """
    F = np.asarray(m.faces)
    half = set()
    for a, b, c in F:
        half.update(((a, b), (b, c), (c, a)))
    nxt = {}
    for a, b in half:
        if (b, a) not in half:
            nxt[b] = a  # the hole's own half-edge runs opposite the boundary one
    loops, seen = [], set()
    for s in list(nxt):
        if s in seen:
            continue
        loop, v = [], s
        while v not in seen and v in nxt:
            seen.add(v); loop.append(v); v = nxt[v]
        if v == s:
            loops.append(loop)
    new = [(l[0], l[i], l[i + 1]) for l in loops if 3 <= len(l) <= max_edges for i in range(1, len(l) - 1)]
    big = [len(l) for l in loops if len(l) > max_edges]
    return trimesh.Trimesh(m.vertices, np.concatenate([F, np.array(new, dtype=F.dtype).reshape(-1, 3)]), process=False), len(loops) - len(big), big


# Usage: remesh.py REPAIRED.obj EDGE_M OUT.obj [CREASE_DEG]
orig_path, scale, out_path = sys.argv[1], float(sys.argv[2]), sys.argv[3]
crease = sys.argv[4] if len(sys.argv) > 4 else "30"
orig = trimesh.load(orig_path, force="mesh", process=True)
raw = out_path.replace(".obj", "_raw.obj")
proc = subprocess.run([str(BINARY), "-r", "6", "-p", "6", "-s", str(scale), "-c", crease, "-d", "-t", "16", "-o", raw, orig_path],
                      capture_output=True, text=True)
assert proc.returncode == 0, proc.stderr[-500:]
im = read_obj(raw)
im = trimesh.Trimesh(im.vertices, im.faces, process=True)
t0 = topology(im)
V = reproject(orig, np.asarray(im.vertices))
m = trimesh.Trimesh(V, im.faces, process=False)
m, nf, nv = drop_flaps(m)
m, nholes, big = fill_small_holes(m)
m = trimesh.Trimesh(m.vertices, m.faces, process=True)
t = topology(m)
A = m.area_faces
el = m.edges_unique_length
print(f"{out_path.split('/')[-1]}: IM {len(im.vertices)} verts [{topo_tag(t0, 'clean')}] -> dropped {nf} flap faces / {nv} verts, filled {nholes} holes"
      f"{' (left open: ' + str(big) + ')' if big else ''} -> {len(m.vertices)} verts, {len(m.faces)} faces [{topo_tag(t, 'clean')}], "
      f"watertight {m.is_watertight}, euler {m.euler_number} (original {orig.euler_number}), volume {m.volume * 1e6:.4f} vs {orig.volume * 1e6:.4f} cm^3, "
      f"edge median {np.median(el) * 1e3:.3f} mm CV {el.std() / el.mean():.2f}, min face area {A.min():.1e}", flush=True)
m.export(out_path)
