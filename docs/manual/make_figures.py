"""The manuals' data figures, from data/ into fig/ as PDF (for LaTeX) and PNG (for Markdown).

    python docs/manual/make_figures.py

No GPU: the data comes from make_data.py. The diagrams are TikZ, under fig/*.tex.

Colours follow the dataviz skill's validated categorical order (blue, orange, aqua, yellow,
magenta), assigned to entities in that fixed order and never cycled; a configuration keeps its
colour across panels, and a variant of it (the de Saxce term off) its line style. Text is
ink, never a series colour. The signed distance is diverging, blue inside and orange
outside, with a neutral grey at zero.
"""
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib import font_manager, pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Circle, Rectangle

HERE = Path(__file__).resolve().parent
DATA, FIG = HERE / "data", HERE / "fig"
FIG.mkdir(exist_ok=True)

for f in glob.glob(str(Path.home() / "texmf/fonts/opentype/public/libertine/LinLibertine_R*.otf")):
    font_manager.fontManager.addfont(f)

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SDF_MAP = LinearSegmentedColormap.from_list(
    "sdf", ["#104281", "#2a78d6", "#9ec5f4", "#f0efec", "#f8cdb6", "#f19a73", "#d45a28"])

plt.rcParams.update({
    "font.family": "Linux Libertine O", "font.size": 9, "mathtext.fontset": "custom",
    "mathtext.rm": "Linux Libertine O", "mathtext.it": "Linux Libertine O:italic",
    "mathtext.bf": "Linux Libertine O:bold", "pdf.fonttype": 42,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK, "axes.linewidth": 0.8, "axes.titlesize": 9,
    "axes.titleweight": "regular", "axes.titlecolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "grid.linestyle": "-", "legend.frameon": False, "legend.fontsize": 8,
    "lines.linewidth": 1.6, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200,
})
CM = 1 / 2.54


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)
    print("figure", name)


def csv(name):
    return np.genfromtxt(DATA / name, delimiter=",", names=True)


#######################################################################################
# Collision detection
#######################################################################################


def fig_sdf():
    d = np.load(DATA / "sdf_slice.npz")
    xs, zs, c2, c0, vox = d["xs"], d["zs"], d["c2"], d["c0"], d["vox"]
    h, r, p_min = d["h"], float(d["r"]), d["p_min"]
    verts, zl = d["verts"], float(d["zl"])
    line = csv("sdf_line.csv")
    fig = plt.figure(figsize=(16.5 * CM, 6.4 * CM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.55], wspace=0.32)
    norm = TwoSlopeNorm(vmin=-0.12, vcenter=0.0, vmax=0.6)
    ext = [xs[0], xs[-1], zs[0], zs[-1]]

    ax = fig.add_subplot(gs[0])
    im = ax.imshow(np.clip(c2, -0.12, 0.6), origin="lower", extent=ext, cmap=SDF_MAP, norm=norm,
                   interpolation="bilinear")
    cax = ax.inset_axes([0.06, 0.05, 0.5, 0.035])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-0.1, 0, 0.6])
    cb.ax.tick_params(labelsize=6.5, length=1.5, pad=1)
    cb.outline.set_visible(False)
    cax.set_title("signed distance [m]", fontsize=6.5, pad=1.5, loc="left")
    ax.contour(xs, zs, c2, levels=[0.0], colors=INK, linewidths=0.9)
    ax.plot(verts[:, 0], verts[:, 1], ".", ms=2.2, color=INK, mec="none")
    ax.axhline(zl, color=INK2, lw=0.7, ls=(0, (3, 2)))
    zx = (xs[0] + 0.30 * (xs[-1] - xs[0]), xs[0] + 0.62 * (xs[-1] - xs[0]))
    zz = (zl - 0.26, zl + 0.26)
    ax.add_patch(Rectangle((zx[0], zz[0]), zx[1] - zx[0], zz[1] - zz[0], fill=False, ec=INK, lw=0.8))
    ax.set_title(r"(a) the tricubic SDF $S_{\mathrm{c2}}$ on a slice", loc="left")
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xlabel("$x$ [m]")
    ax.set_ylabel("$z$ [m]")

    ax = fig.add_subplot(gs[1])
    # vox is piecewise constant on the voxels: draw it at its own resolution.
    ix = (xs >= zx[0]) & (xs <= zx[1])
    iz = (zs >= zz[0]) & (zs <= zz[1])
    ax.imshow(np.clip(vox[np.ix_(iz, ix)], -0.12, 0.6), origin="lower", extent=[zx[0], zx[1], zz[0], zz[1]],
              cmap=SDF_MAP, norm=norm, interpolation="nearest")
    for k in range(-2, 200):
        gx = p_min[0] + k * h[0]
        if zx[0] <= gx <= zx[1]:
            ax.axvline(gx, color="white", lw=0.35)
        gz = p_min[2] + k * h[2]
        if zz[0] <= gz <= zz[1]:
            ax.axhline(gz, color="white", lw=0.35)
    ax.contour(xs[ix], zs[iz], c2[np.ix_(iz, ix)], levels=[0.0], colors=INK, linewidths=1.0)
    ax.contour(xs[ix], zs[iz], c0[np.ix_(iz, ix)], levels=[0.0], colors=INK, linewidths=0.9, linestyles=[(0, (3, 2))])
    ax.contour(xs[ix], zs[iz], vox[np.ix_(iz, ix)], levels=[0.0], colors=INK2, linewidths=0.7, linestyles=[(0, (1, 1.5))])
    sel = (verts[:, 0] >= zx[0]) & (verts[:, 0] <= zx[1]) & (verts[:, 1] >= zz[0]) & (verts[:, 1] <= zz[1])
    ax.plot(verts[sel, 0], verts[sel, 1], "o", ms=3.2, color=INK, mec="white", mew=0.6)
    ax.set_xlim(*zx)
    ax.set_ylim(*zz)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_title(r"(b) voxels of $S_{\mathrm{vox}}$", loc="left")
    ax.set_xlabel("$x$ [m]")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=INK, lw=1.0, label=r"$S_{\mathrm{c2}} = 0$"),
                       Line2D([], [], color=INK, lw=0.9, ls=(0, (3, 2)), label=r"$S_{\mathrm{c0}} = 0$"),
                       Line2D([], [], color=INK2, lw=0.7, ls=(0, (1, 1.5)), label=r"$S_{\mathrm{vox}} = 0$")],
              loc="lower left", fontsize=7, handlelength=2.0, borderpad=0.2, labelspacing=0.2,
              facecolor="white", framealpha=0.85, frameon=True, edgecolor="none")

    ax = fig.add_subplot(gs[2])
    x = line["x"]
    sel = (x >= zx[0] - 0.15) & (x <= zx[1] + 0.15)
    ax.fill_between(x[sel], line["exact"][sel] - r, line["exact"][sel] + r, color=BLUE, alpha=0.10, lw=0,
                    label=r"exact $\pm\, r$")
    ax.plot(x[sel], line["exact"][sel], color=INK, lw=1.0, label="exact distance to the mesh")
    ax.plot(x[sel], line["vox"][sel], color=ORANGE, lw=1.2, drawstyle="steps-mid", label=r"$S_{\mathrm{vox}}$, voxel centres")
    ax.plot(x[sel], line["c0"][sel], color=AQUA, lw=1.4, ls=(0, (4, 2)), label=r"$S_{\mathrm{c0}}$, trilinear")
    ax.plot(x[sel], line["c2"][sel], color=BLUE, lw=1.6, label=r"$S_{\mathrm{c2}}$, tricubic")
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_title("(c) along the dashed line in (a)", loc="left")
    ax.set_xlabel("$x$ [m]")
    ax.set_ylabel("signed distance [m]")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=7, ncol=3, columnspacing=1.0,
              handlelength=2.2)
    ax.set_ylim(min(line["exact"][sel].min(), line["vox"][sel].min()) - 0.03,
                max(line["exact"][sel].max(), line["vox"][sel].max()) + 0.03)
    save(fig, "fig_sdf")


def fig_tiles():
    v = csv("tiles_vertices.csv")
    sph = csv("tiles_spheres.csv")
    summ = json.load(open(DATA / "tiles_summary.json"))
    P = np.column_stack([v["x"], v["y"], v["z"]])
    fig, axs = plt.subplots(1, 3, figsize=(16.5 * CM, 6.2 * CM), gridspec_kw=dict(wspace=0.05))
    cmap = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#5598e7", "#104281"])
    ax = axs[0]
    ax.plot(P[:, 0], P[:, 2], color=AXIS, lw=0.5, zorder=1)
    ax.scatter(P[:, 0], P[:, 2], c=np.arange(len(P)), cmap=cmap, s=7, zorder=2, linewidths=0)
    ax.set_title("(a) stored order: a Hilbert curve", loc="left")
    for k, (ax, title) in enumerate(((axs[1], f"(b) tiles of 8: mean $R$ {summ['mean_r_hilbert']:.2f} m"),
                                     (axs[2], f"(c) shuffled: mean $R$ {summ['mean_r_shuffled']:.2f} m"))):
        if k == 0:
            C = np.column_stack([sph["cx"], sph["cz"]])
            R = sph["r"]
            Q = P
        else:
            rng = np.random.default_rng(0)
            Q = P[rng.permutation(len(P))]
            C, R = [], []
            for t in range(int(np.ceil(len(Q) / 8))):
                q = Q[t * 8:(t + 1) * 8]
                c = 0.5 * (q.min(0) + q.max(0))
                C.append(c[[0, 2]])
                R.append(np.linalg.norm(q - c, axis=1).max())
            C, R = np.array(C), np.array(R)
        ax.scatter(P[:, 0], P[:, 2], color=INK2, s=3, linewidths=0, zorder=2)
        col = BLUE if k == 0 else ORANGE
        for c, rr in zip(C, R):
            ax.add_patch(Circle(c, rr, fc=col, ec=col, alpha=0.05 if k == 0 else 0.025, lw=0.6, zorder=1))
            ax.add_patch(Circle(c, rr, fc="none", ec=col, alpha=0.55 if k == 0 else 0.35, lw=0.5, zorder=1))
        ax.set_title(title, loc="left")
    for ax in axs:
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xlim(P[:, 0].min() - 0.7, P[:, 0].max() + 0.7)
        ax.set_ylim(P[:, 2].min() - 0.7, P[:, 2].max() + 0.7)
    save(fig, "fig_tiles")


def fig_funnel():
    stages = ["all candidates", "after the box test", "vertex slots of surviving tiles",
              "after the voxel prefilter", "contacts (spline)"]
    raw = open(DATA / "funnel.csv").read().splitlines()
    keys = raw[0].split(",")
    vals = {}
    for line in raw[1:]:
        rr = dict(zip(keys, line.split(",")))
        f = {k: float(x) for k, x in rr.items() if k != "window"}
        vals[rr["window"]] = [f["candidates"], f["pairs_active"] * f["candidates"] / f["pairs"], f["slots"],
                              f["prefilter"], f["contacts"]]
    fig, ax = plt.subplots(figsize=(16.5 * CM, 5.6 * CM))
    y = np.arange(len(stages))[::-1]
    hgt = 0.36
    for k, (w, col) in enumerate((("impact", BLUE), ("settled", ORANGE))):
        yy = y + (hgt / 2 + 0.03) * (1 if k == 0 else -1)
        ax.barh(yy, vals[w], height=hgt, color=col, label=f"{w} (step {650 if w == 'impact' else 2550})", zorder=2)
        for yv, x in zip(yy, vals[w]):
            ax.text(x * 1.12, yv, f"{x:,.0f}" if x >= 100 else f"{x:.1f}", va="center", ha="left", fontsize=7.5,
                    color=INK2)
    ax.set_xscale("log")
    ax.set_xlim(1, 6e4)
    ax.set_yticks(y)
    ax.set_yticklabels(stages)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("per world, mean over 1024 worlds (log scale)")
    ax.legend(loc="lower right")
    save(fig, "fig_funnel")


def fig_reduce():
    raw = csv("reduce.csv")
    keep = raw["keep"] > 0
    fig, axs = plt.subplots(1, 2, figsize=(16.5 * CM, 6.0 * CM), gridspec_kw=dict(width_ratios=[1, 2.1], wspace=0.25))
    ax = axs[0]
    ax.scatter(raw["x"] * 1e3, raw["y"] * 1e3, s=4, color=MUTED, linewidths=0, label=f"all {len(keep)} contacts")
    ax.scatter(raw["x"][keep] * 1e3, raw["y"][keep] * 1e3, s=16, color=BLUE, edgecolors="white", linewidths=0.6,
               label=f"the {keep.sum()} kept")
    ax.set_aspect("equal")
    ax.set_xlabel("$x$ [mm]")
    ax.set_ylabel("$y$ [mm]")
    ax.set_title("(a) from above", loc="left")
    ax = axs[1]
    th = np.degrees(raw["theta"])
    ax.scatter(th, raw["z"] * 1e3, s=4, color=MUTED, linewidths=0, label=f"all {len(keep)} nut-on-bolt contacts")
    ax.scatter(th[keep], raw["z"][keep] * 1e3, s=16, color=BLUE, edgecolors="white", linewidths=0.6,
               label=f"the {keep.sum()} the reduction keeps")
    ax.set_xlabel("angle about the bolt's axis [deg]")
    ax.set_ylabel("height [mm]")
    ax.set_xlim(-180, 180)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_title("(b) unrolled about the axis", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=2)
    lo, hi = np.percentile(raw["z"] * 1e3, [0, 100])
    ax.set_ylim(lo - 1.0, hi + 3.5)
    save(fig, "fig_reduce")


#######################################################################################
# Contact dynamics
#######################################################################################


def fig_spring_damper():
    """The spring-damper law that weighs a contact group's members, a ray member's friction, and a member's de
    Saxce term: the ray (aqua) and the cone (yellow) keep fig_slope's colours."""
    fig, axs = plt.subplots(1, 3, figsize=(16.5 * CM, 5.2 * CM), gridspec_kw=dict(wspace=0.38))
    x = np.linspace(-1.5, 3.0, 400)
    D = np.where(x < 0, 1 - x, np.where(x < 2, (x - 2) ** 2 / 4, 0.0))
    ax = axs[0]
    ax.plot(x, D, color=BLUE)
    ax.axvline(0, color=AXIS, lw=0.8)
    ax.set_xlabel(r"$v_n/\delta$  (approaching $<0<$ separating)")
    ax.set_ylabel(r"dissipation factor $D$")
    ax.set_title("(a) the damping factor", loc="left")
    ax.annotate("$1 - x$", (-1.0, 2.0), xytext=(-0.55, 2.2), fontsize=8, color=INK2)
    ax.annotate(r"$(x-2)^2/4$", (1.0, 0.25), xytext=(0.95, 0.55), fontsize=8, color=INK2)
    eps = 0.05
    v = np.linspace(0, 0.4, 400)
    ax = axs[1]
    ax.plot(v, v / np.sqrt(eps ** 2 + v ** 2), color=AQUA, label="smooth, $\\varepsilon$ = 0.05 m/s")
    ax.plot([0, 0, 0.4], [0, 1, 1], color=INK2, lw=1.0, ls=(0, (3, 2)), label="Coulomb")
    ax.set_xlabel(r"slip speed $\Vert v_t\Vert$ [m/s]")
    ax.set_ylabel(r"$r / \mu = \Vert f_t\Vert / (\mu f_n)$")
    ax.set_ylim(0, 1.12)
    ax.set_title("(b) a ray member's friction", loc="left")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.04), fontsize=7)
    mu = 0.5
    ax = axs[2]
    v = np.linspace(0, 0.15, 400)  # where the two differ: below a few stiction velocities
    ax.plot(v, mu * v, color=YELLOW, label=r"cone: $\mu \Vert\sigma_t\Vert$")
    ax.plot(v, mu * v ** 2 / np.sqrt(eps ** 2 + v ** 2), color=AQUA, label=r"ray: $r\, t_1^\top\sigma$")
    ax.set_xlabel(r"slip speed $\Vert\sigma_t\Vert$ [m/s]")
    ax.set_ylabel(r"de Saxcé term [m/s]")
    ax.set_title(r"(c) a member's de Saxcé term, $\mu$ = 0.5", loc="left")
    ax.legend(loc="upper left", fontsize=7)
    save(fig, "fig_spring_damper")


# The solver's configurations in fig_slope, as make_data.py's slope section names its columns: each an entity
# with its colour in the categorical order, the exact mode without the de Saxce term a variant of the exact mode
# (its colour, dashed).
SLOPE = [  # column, colour, line style, label
    ("exact", BLUE, "-", "exact"),
    ("exact_nods", BLUE, (0, (3, 2)), "exact, no de Saxcé term"),
    ("mj", ORANGE, "-", "mj"),
    ("ray", AQUA, "-", "exact, ray groups"),
    ("cone", YELLOW, "-", "exact, cone groups"),
]
HELD = ("exact", "exact_nods", "cone")  # the configurations that hold the cube at mu 0.5


def fig_slope():
    hold = csv("slope_hold.csv")
    slide = csv("slope_slide.csv")
    t = hold["t"]
    last = np.searchsorted(t, t[-1] - 1.0 - 1e-9)  # the last second: make_data.py's creep window
    creep = {k: hold[k][-1] - hold[k][last] for k, *_ in SLOPE}  # mm in the last second
    ts = slide["t"]
    landed = ts >= 0.5 - 1e-9  # make_data.py's: after the first half second
    top = {k: slide[k][landed].max() for k, *_ in SLOPE}  # the highest the cube rides once landed, mm
    low = {k: slide[k][landed].min() for k, *_ in SLOPE}
    fig, axs = plt.subplots(1, 2, figsize=(16.5 * CM, 6.4 * CM), gridspec_kw=dict(wspace=0.28))

    def lw(ls):
        return 1.6 if ls == "-" else 1.1

    ax = axs[0]
    for key, col, ls, label in SLOPE:
        ax.plot(t, hold[key], color=col, ls=ls, label=label, lw=lw(ls), zorder=2 if key in ("ray", "mj") else 3)
    ax.text(t[-1] * 0.97, hold["ray"][-1] * 0.93, f"ray groups: {creep['ray']:.0f} mm in the last s",
            ha="right", va="top", fontsize=7.5, color=INK2)
    ax.text(t[-1] * 0.97, hold["mj"][-1] + 6, f"mj: {creep['mj']:.1f} mm in the last s",
            ha="right", va="bottom", fontsize=7.5, color=INK2)
    held = max(creep[k] for k in HELD)
    ax.text(t[-1] * 0.97, -10, f"exact, cone groups: held ({held:.3f} mm or less in the last s)",
            ha="right", va="top", fontsize=7.5, color=INK2)
    ax.set_ylim(-22, max(hold["ray"]) * 1.08)
    ax.set_xlim(0, t[-1])
    ax.set_xlabel("$t$ [s]")
    ax.set_ylabel("displacement down the slope [mm]")
    ax.set_title(r"(a) held: a cube on a 20° slope at $\mu$ = 0.5", loc="left")

    ax = axs[1]
    for key, col, ls, label in SLOPE:
        ax.plot(ts, slide[key], color=col, ls=ls, label=label, lw=lw(ls))
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_xlim(0, ts[-1])
    ax.set_ylim(-2, 16)
    ax.set_xlabel("$t$ [s]")
    ax.set_ylabel("height above rest [mm]")
    ax.set_title(r"(b) sliding down it at $\mu$ = 0.2", loc="left")
    ax.text(ts[-1] * 0.985, top["exact_nods"] + 0.4, f"no de Saxcé term: up to {top['exact_nods']:.1f} mm",
            ha="right", va="bottom", fontsize=7.5, color=INK2)
    ax.text(ts[-1] * 0.985, top["mj"] + 0.4, f"mj, no term by default: up to {top['mj']:.1f} mm",
            ha="right", va="bottom", fontsize=7.5, color=INK2)
    # The configurations with the term -- the exact mode and its groups -- stay near rest: an inset at their own
    # scale, from the landing on, where the hop leaves room (the upper left).
    on = [k for k, *_ in SLOPE if k not in ("exact_nods", "mj")]
    ins = ax.inset_axes([0.11, 0.45, 0.42, 0.30])
    for key, col, ls, label in SLOPE:
        if key in on:
            ins.plot(ts[landed], slide[key][landed], color=col, ls=ls, lw=1.3)
    ins.axhline(0, color=AXIS, lw=0.8)
    ins.set_xlim(ts[landed][0], ts[-1])
    lo = min(low[k] for k in on)
    ins.set_ylim(lo * 1.3, 0.06)
    ins.set_yticks(np.arange(np.ceil(lo * 10) / 10, 0.01, 0.1))
    ins.tick_params(labelsize=6.5, length=2, pad=1)
    ins.set_title("with the term, from 0.5 s [mm]", fontsize=6.5, loc="left", pad=2, color=INK2)
    ins.grid(True)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=len(SLOPE), fontsize=7,
               handlelength=2.4, columnspacing=1.4)
    save(fig, "fig_slope")


def fig_throughput():
    """
    The nut-bolt's throughput, from data/nutbolt_throughput.csv (the 1 mm nut at
    4096 worlds; Newton's own SDF example beside it): the solver's configurations in blue, Newton's in grey,
    fastest first. Beside each of the solver's bars, the contacts a world and the busiest world's rows at the end
    of the window, and where the nut was not seated then (not within -1..3 mm of its seat), how
    far above it: such a configuration timed a different motion.
    """
    import csv as _csv
    with open(DATA / "nutbolt_throughput.csv", newline="") as f:
        rows = list(_csv.DictReader(f))
    rows.sort(key=lambda r: -float(r["world_s_per_s"]))

    def name(r):
        lab = r["label"].replace("6x4", "6 × 4")
        return lab + ", per contact" if lab in ("exact", "mj") else lab

    fig, ax = plt.subplots(figsize=(16.5 * CM, 5.4 * CM))
    y = np.arange(len(rows))[::-1]
    vals = [float(r["world_s_per_s"]) for r in rows]
    cols = [MUTED if r["family"] != "lodos" else BLUE for r in rows]
    ax.barh(y, vals, height=0.55, color=cols, zorder=2)
    top = max(vals)
    right = top * 1.5
    for yy, r, val in zip(y, rows, vals):
        ax.text(val + 0.012 * top, yy, f"{val:,.0f}", va="center", ha="left", fontsize=7.5, color=INK)
        if r["family"] != "lodos":
            continue
        note = f"{float(r['contacts_per_world']):.0f} contacts, {int(r['rows_max'])} rows"
        z = float(r["nut_z_end_mm"])
        if not -1.0 <= z <= 3.0:
            note = f"the nut still {z:.1f} mm above its seat; " + note
        ax.text(right, yy, note, va="center", ha="right", fontsize=7, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([name(r) for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, right)
    ax.set_xticks([x for x in ax.get_xticks() if x <= top * 1.1])
    ax.text(right, y[0] + 0.62, "at the window's end: contacts a world, the busiest world's rows", va="bottom",
            ha="right", fontsize=7, color=MUTED)
    ax.set_xlabel("world-seconds per second, 1 mm nut, 4096 worlds (RTX 4080)")
    save(fig, "fig_throughput")


FIGS = [fig_sdf, fig_tiles, fig_funnel, fig_reduce, fig_spring_damper, fig_slope, fig_throughput]

if __name__ == "__main__":
    import sys
    want = set(sys.argv[1:])
    for f in FIGS:
        if not want or f.__name__ in want:
            f()
