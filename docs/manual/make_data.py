"""The data the manuals' figures plot, into docs/manual/data/. Needs the GPU.

    python docs/manual/make_data.py [slope] [funnel] [reduce] [sdf] [tiles]

    slope    a unit cube on a 20 degree slope, one world per configuration of the solver: held
             at mu 0.5 (its displacement down the slope) and sliding at mu 0.2 (its height
             above rest)
    funnel   the Lodos pipeline's survivors per stage on the five-armadillo drop, 1024
             worlds, at the impact and once settled
    reduce   one world of the nut-bolt seated under the mj mode: every Lodos contact of the nut
             on the bolt, and which ones the contact reduction keeps
    sdf      the armadillo's SDF on a slice through it (tricubic, trilinear and voxel),
             and along a line through its surface against the exact mesh distance
    tiles    the armadillo's Hilbert-ordered vertices and their tiles' bounding spheres

Each section writes its own files, so one can be redone alone. The figures are drawn by
make_figures.py from these files, without the GPU.
"""
import json
import os
import sys
from pathlib import Path

import mujoco
import numpy as np
import warp as wp

from lodos import sim as lsim, step
from lodos.col.sdf import PlaneSurface, collide as sdf_collide
from viewer.scenes.assets import load_surface

wp.config.quiet = True
OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(exist_ok=True)


def slope_sim(mu, opts):
    g, theta = 9.81, np.radians(20.0)
    mjm = mujoco.MjModel.from_xml_string(f"""<mujoco>
      <option timestep="0.002" gravity="{g * np.sin(theta)} 0 {-g * np.cos(theta)}" cone="elliptic"/>
      <worldbody><geom name="floor" type="plane" size="50 50 0.1" friction="{mu}"/>
        <body name="cube" pos="0 0 0.501"><freejoint/>
          <geom name="cube" type="box" size="0.5 0.5 0.5" density="100" contype="0" conaffinity="0" friction="{mu}"/></body>
      </worldbody></mujoco>""")
    gid = lambda n: mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, n)
    surfaces = lsim.Surfaces(surfaces=[load_surface("cube"), PlaneSurface()], geoms=[gid("cube"), gid("floor")],
                             pairs=[(0, 1)], mu=mu)
    return lsim.make_sim(mjm, 1, surfaces=surfaces, solver_kwargs=opts, nconmax=64, njmax=256)


# The solver's configurations: the exact mode (its default) with and without the de Saxce
# term, the mj mode, and the exact mode on the cube's contacts gathered into ray and cone
# groups (`lodos.cd.groups`).
SLOPE = [
    ("exact", {}),
    ("exact_nods", {"de_saxce": False}),
    ("mj", {"mode": "mj"}),
    ("ray", {"groups": "ray"}),
    ("cone", {"groups": "cone"}),
]


def slope():
    """Held (mu 0.5): displacement down the slope over 3 s. Sliding (mu 0.2): height above rest over 2 s."""
    t_hold = np.arange(1, 1501) * 0.002
    t_slide = np.arange(1, 1001) * 0.002
    hold = {"t": t_hold}
    slide = {"t": t_slide}
    for name, opts in SLOPE:
        s = slope_sim(0.5, opts)
        replay = step.make_step_graph(s)
        x0 = float(s.drbd.qpos.numpy()[0, 0])
        xs = []
        for _ in range(len(t_hold)):
            replay()
            xs.append(float(s.drbd.qpos.numpy()[0, 0]) - x0)
        hold[name] = np.array(xs) * 1e3  # mm
        s = slope_sim(0.2, opts)
        replay = step.make_step_graph(s)
        zs = []
        for _ in range(len(t_slide)):
            replay()
            zs.append(float(s.drbd.qpos.numpy()[0, 2]))
        slide[name] = (np.array(zs) - 0.5) * 1e3  # mm above the resting height
        print(f"slope {name}: held, creep {hold[name][-1] - hold[name][-501]:.3f} mm over the last s; "
              f"sliding, height max {slide[name][250:].max():.3f} mm", flush=True)
    for fname, d in (("slope_hold.csv", hold), ("slope_slide.csv", slide)):
        keys = list(d)
        np.savetxt(OUT / fname, np.column_stack([d[k] for k in keys]), delimiter=",", header=",".join(keys),
                   comments="", fmt="%.6g")


def funnel():
    """Per world, mean over the batch: what each stage of the Lodos pipeline leaves."""
    from viewer.scenes import armadillo
    B = 1024
    s = armadillo.build(batch=B, seed=0, nconmax=256, njmax=1024)
    mcol, dcol = s.mcol, s.dcol
    replay = step.make_step_graph(s)
    rows = []
    marks = {650: "impact", 2550: "settled"}
    for i in range(1, 2551):
        replay()
        if i in marks:
            ct = dcol.ct.numpy()
            active = dcol.cpr_active.numpy()
            row = dict(window=marks[i], step=i,
                       pairs=int(mcol.nCPR), pairs_active=float(active.sum()) / B,
                       candidates=int(mcol.nC), tiles=int(mcol.nT_world),
                       tiles_active=float(ct[3]) / B, slots=float(ct[0]) / B,
                       prefilter=float(ct[1]) / B, contacts=float(ct[2]) / B)
            rows.append(row)
            print("funnel", json.dumps(row), flush=True)
    keys = list(rows[0])
    with open(OUT / "funnel.csv", "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")


def reduce_():
    """The nut seated (5 s in) under the mj mode, with the reduction on."""
    from viewer.scenes import nutbolt
    s = nutbolt.build(1, nut="nut_s1.0", solver_kwargs={"mode": "mj"}, contact_reduction=True, nconmax=768,
                      njmax=3072)
    replay = step.make_step_graph(s)
    for _ in range(int(5.0 * 480)):
        replay()
    c = sdf_collide.get_contacts(s.mcol, s.dcol)
    keep = s.solver.rd.keep.numpy()[:len(c["p"])]
    q = s.drbd.qpos.numpy()[0]
    bolt = q[0:3]
    seg = c["seg"]
    sel = seg == 0  # the nut's vertices on the bolt's SDF
    p = c["p"][sel] - bolt
    k = keep[sel]
    theta = np.arctan2(p[:, 1], p[:, 0])
    rr = np.hypot(p[:, 0], p[:, 1])
    data = np.column_stack([p, theta, rr, k, c["n"][sel], c["sdf"][sel]])
    np.savetxt(OUT / "reduce.csv", data, delimiter=",", header="x,y,z,theta,r,keep,nx,ny,nz,sdf", comments="",
               fmt="%.6g")
    print(f"reduce: {sel.sum()} nut-on-bolt contacts, {int(k.sum())} kept; all segments {len(seg)}", flush=True)


def _armadillo():
    surf = load_surface("armadillo")
    return surf


def sdf():
    """A slice of the armadillo's SDF (c2, c0, vox) and a line through its surface against the mesh."""
    from lodos.col.sdf import implicit as imp
    cubic = load_surface("armadillo", spline_type="cubic")
    linear = load_surface("armadillo", spline_type="linear")
    v = np.asarray(cubic.v, dtype=np.float64)
    f = np.asarray(cubic.f, dtype=np.int32)
    gp = cubic.imps.grid_params
    p_min, p_max = np.asarray(gp.p_min, dtype=float), np.asarray(gp.p_max, dtype=float)
    centre = 0.5 * (v.min(axis=0) + v.max(axis=0))
    # The slice: the plane y = centre[1] (front view, x to the right, z up).
    nx, nz = 420, 480
    xs = np.linspace(p_min[0], p_max[0], nx)
    zs = np.linspace(p_min[2], p_max[2], nz)
    X, Z = np.meshgrid(xs, zs)
    P = np.column_stack([X.ravel(), np.full(X.size, centre[1]), Z.ravel()]).astype(np.float32)

    def evaluate(surf, func):
        pts = wp.array(P, dtype=wp.vec3)
        out = wp.zeros(len(P), dtype=float)
        wp.launch(func, dim=len(P), inputs=[pts, out, surf.imps])
        return out.numpy().reshape(X.shape)

    @wp.kernel
    def k_sdf(p: wp.array(dtype=wp.vec3), out: wp.array(dtype=float), imps: imp.ImplicitSurface):  # type: ignore
        i = wp.tid()
        out[i] = imp.eval_sdf(p[i], imps)

    @wp.kernel
    def k_vsdf(p: wp.array(dtype=wp.vec3), out: wp.array(dtype=float), imps: imp.ImplicitSurface):  # type: ignore
        i = wp.tid()
        out[i] = imp.eval_vsdf(p[i], imps)

    c2 = evaluate(cubic, k_sdf)
    c0 = evaluate(linear, k_sdf)
    vox = evaluate(cubic, k_vsdf)

    # The exact signed distance to the mesh, for the line.
    mesh = wp.Mesh(points=wp.array(v.astype(np.float32), dtype=wp.vec3), indices=wp.array(f.ravel(), dtype=int))

    @wp.kernel
    def k_exact(mesh_id: wp.uint64, p: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):  # type: ignore
        i = wp.tid()
        q = wp.mesh_query_point_sign_normal(mesh_id, p[i], 1.0e6)
        if q.result:
            c = wp.mesh_eval_position(mesh_id, q.face, q.u, q.v)
            out[i] = q.sign * wp.length(p[i] - c)

    # The line: horizontal, through the slice at the height of the armadillo's belly, along x.
    zl = centre[2] + 0.05 * (v[:, 2].max() - v[:, 2].min())
    sl = np.linspace(p_min[0], p_max[0], 1500)
    L = np.column_stack([sl, np.full(sl.size, centre[1]), np.full(sl.size, zl)]).astype(np.float32)
    pts = wp.array(L, dtype=wp.vec3)
    outs = {}
    for name, surf, func in (("c2", cubic, k_sdf), ("c0", linear, k_sdf), ("vox", cubic, k_vsdf)):
        o = wp.zeros(len(L), dtype=float)
        wp.launch(func, dim=len(L), inputs=[pts, o, surf.imps])
        outs[name] = o.numpy()
    o = wp.zeros(len(L), dtype=float)
    wp.launch(k_exact, dim=len(L), inputs=[mesh.id, pts, o])
    outs["exact"] = o.numpy()
    # Vertices near the slice plane, for the dots.
    near = np.abs(v[:, 1] - centre[1]) < 0.5 * float(np.asarray(gp.h).max())
    np.savez_compressed(OUT / "sdf_slice.npz", xs=xs, zs=zs, c2=c2, c0=c0, vox=vox, y=centre[1], zl=zl,
                        h=np.asarray(gp.h, dtype=float), r=float(gp.r), p_min=p_min, p_max=p_max,
                        verts=v[near][:, [0, 2]])
    np.savetxt(OUT / "sdf_line.csv", np.column_stack([sl, outs["exact"], outs["c2"], outs["c0"], outs["vox"]]),
               delimiter=",", header="x,exact,c2,c0,vox", comments="", fmt="%.6g")
    print(f"sdf: grid {np.asarray(gp.N)}, h {np.asarray(gp.h)}, r {float(gp.r):.5f}; slice {c2.shape}", flush=True)


def tiles():
    """The armadillo's vertices in their stored (Hilbert) order, and tile_size-8 bounding spheres."""
    surf = _armadillo()
    v = np.asarray(surf.v, dtype=np.float64)
    K = 8
    rows = []
    for t in range(int(np.ceil(len(v) / K))):
        q = v[t * K:(t + 1) * K]
        c = 0.5 * (q.min(axis=0) + q.max(axis=0))
        r = float(np.linalg.norm(q - c, axis=1).max())
        rows.append([t, *c, r])
    np.savetxt(OUT / "tiles_vertices.csv", np.column_stack([np.arange(len(v)), v, np.arange(len(v)) // K]),
               delimiter=",", header="i,x,y,z,tile", comments="", fmt="%.6g")
    np.savetxt(OUT / "tiles_spheres.csv", np.array(rows), delimiter=",", header="tile,cx,cy,cz,r", comments="",
               fmt="%.6g")
    # The same vertices shuffled, for the comparison: what a tile would bound without the curve.
    rng = np.random.default_rng(0)
    w = v[rng.permutation(len(v))]
    rr = []
    for t in range(int(np.ceil(len(w) / K))):
        q = w[t * K:(t + 1) * K]
        c = 0.5 * (q.min(axis=0) + q.max(axis=0))
        rr.append(float(np.linalg.norm(q - c, axis=1).max()))
    print(f"tiles: {len(v)} vertices, {len(rows)} tiles; mean radius {np.mean([r[-1] for r in rows]):.4f} "
          f"Hilbert, {np.mean(rr):.4f} shuffled", flush=True)
    with open(OUT / "tiles_summary.json", "w") as fh:
        json.dump({"n": len(v), "tiles": len(rows), "mean_r_hilbert": float(np.mean([r[-1] for r in rows])),
                   "mean_r_shuffled": float(np.mean(rr)), "extent": float((v.max(0) - v.min(0)).max())}, fh)


SECTIONS = {"slope": slope, "funnel": funnel, "reduce": reduce_, "sdf": sdf, "tiles": tiles}

if __name__ == "__main__":
    for name in sys.argv[1:] or list(SECTIONS):
        SECTIONS[name]()
