import numpy as np

import torch
from scipy.spatial import cKDTree

import trimesh

from lodos.col.sdf import serialize

#######################################################################################
# Utilities for Vertex-Edge Reordering for Sequential Memory Access
#######################################################################################


def quantize_v(v, eps_divisor=1):
    """
    The vertices on an integer grid of spacing `eps / eps_divisor`, `eps` the smallest distance between two of
    them, and the curve depth that grid needs. Prints the spacing.

    Called by `process_mesh`, `tools.mesh_processing.create_surface_data.bake` and
    `tools.mesh_processing.nutbolt.bake.build`.
    In: v (n, 3) float, the vertices; eps_divisor, number, grid cells per `eps`.
    Out: vq (n, 3) int, each vertex's grid cell from the bounding box's minimum corner; recommended_depth, int,
    the bits per axis that cover the longest side.
    """
    max_extent = (v.max(axis=0) - v.min(axis=0)).max()  # float: the bounding box's longest side
    if v.shape[0] < 2:
        eps = max_extent  # float: the grid spacing before `eps_divisor`
    else:
        # The smallest distance between two vertices, by nearest-neighbour query. The
        # closest pair's distance is recomputed the way an all-pairs distance matrix
        # computes it, so the quantization is the same to the bit as an all-pairs search's.
        dist, nbr = cKDTree(v).query(v, k=2)  # (n, 2) float, (n, 2) int: each vertex itself, then its nearest
        i = int(np.argmin(dist[:, 1]))  # int: a vertex of the closest pair
        j = int(nbr[i, 1])  # int: the other
        eps = np.linalg.norm(v[None, j, :] - v[None, i, :], axis=-1)[0]
    print("Quantization eps is: {}".format(eps / eps_divisor))
    vq = ((v - v.min(axis=0)) / eps * eps_divisor).astype(int)  # (n, 3) int
    recommended_depth = int(np.ceil(np.log2(max_extent / eps * eps_divisor)).item())  # int

    return vq, recommended_depth


def reorder_v(v, e, f, vq, depth, encoding="hilbert"):
    """
    The vertices sorted along a Hilbert curve (or a Morton one, `encoding="morton"`) through their grid cells,
    and the edges and faces renumbered to match.

    Called by `process_mesh`.
    In: v (n, 3) float, e (ne, 2) int, f (nf, 3) int; vq (n, 3) int, the grid cells from `quantize_v`; depth, int,
    the curve's bits per axis; encoding, str. Out: v (n, 3), e (ne, 2) and f (nf, 3), reordered.
    """
    if encoding == "morton":
        encode = serialize.morton_encode  # (n, 3) int tensor, depth -> (n,) int64 tensor of curve keys
    else:
        encode = serialize.hilbert_encode

    v_idx = np.array(encode(torch.from_dlpack(vq), depth))  # (n,) int64: each vertex's key along the curve
    v_permutation = np.argsort(v_idx)  # (n,) int: the old index of each new vertex
    v_permutation_inv = np.argsort(v_permutation)  # (n,) int: the new index of each old vertex
    v_sorted = v[v_permutation]  # (n, 3)
    e_sorted = v_permutation_inv[e]  # (ne, 2) int
    f_sorted = v_permutation_inv[f]  # (nf, 3) int

    return v_sorted, e_sorted, f_sorted


def get_e_l(v, e):
    """
    The length of every edge.

    Called by `process_mesh`.
    In: v (n, 3) float, e (ne, 2) int. Out: (ne,) float.
    """
    return np.linalg.norm(v[e][:, 0] - v[e][:, 1], axis=-1)


def process_mesh(v, e, f, eps_divisor=1, depth=None, encoding="hilbert"):
    """
    A mesh with its vertices in curve order (`quantize_v`, then `reorder_v`), and its edge lengths. Prints the
    recommended depth.

    Called by `load_and_process_mesh`, `tools.mesh_processing.create_surface_data.bake` and
    `tools.mesh_processing.nutbolt.bake.build`.
    In: v (n, 3) float, e (ne, 2) int, f (nf, 3) int; eps_divisor, number, as for `quantize_v`; depth, int, the
    curve's bits per axis, None for the recommended one; encoding, "hilbert" or "morton".
    Out: v (n, 3), e (ne, 2) and f (nf, 3), reordered, and e_l (ne,) float, the edge lengths.
    """
    vq, recommended_depth = quantize_v(v, eps_divisor)  # (n, 3) int, int
    print("Recommended minimum encoding depth is: {}".format(recommended_depth))
    if depth == None:
        depth = recommended_depth  # int

    v, e, f = reorder_v(v, e, f, vq, depth, encoding)  # (n, 3), (ne, 2), (nf, 3), reordered

    # `e` and `e_l` are mesh topology, kept for the asset tools: they place the
    # normal glyphs in `tools/mesh_processing/create_surface_data.py` and set its
    # scale. Nothing in the collision pipeline reads them.
    return v, e, f, get_e_l(v, e)


def load_mesh(path):
    """
    A mesh file's vertices, unique edges and faces, as `trimesh` reads them.

    Called by `tools.mesh_processing.create_surface_data.bake`.
    In: path, str or Path. Out: v (n, 3) float64, e (ne, 2) int64, f (nf, 3) int64.
    """
    mesh = trimesh.load_mesh(path)  # trimesh.Trimesh
    v = np.array(mesh.vertices)  # (n, 3) float64
    e = np.array(mesh.edges_unique)  # (ne, 2) int64
    f = np.array(mesh.faces)  # (nf, 3) int64

    return v, e, f


def load_and_process_mesh(path, eps_divisor=50, depth=16, encoding="hilbert"):
    """
    A mesh file, read as `load_mesh` reads it, through `process_mesh`.

    Not called in the repository.
    In: path, str or Path; eps_divisor, depth and encoding, as for `process_mesh`.
    Out: v (n, 3), e (ne, 2), f (nf, 3) and e_l (ne,), as `process_mesh` returns them.
    """
    mesh = trimesh.load_mesh(path)  # trimesh.Trimesh
    v = np.array(mesh.vertices)  # (n, 3) float64
    e = np.array(mesh.edges_unique)  # (ne, 2) int64
    f = np.array(mesh.faces)  # (nf, 3) int64
    return process_mesh(v, e, f, eps_divisor=eps_divisor, depth=depth,
                        encoding=encoding)
