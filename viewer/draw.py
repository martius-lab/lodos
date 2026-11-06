"""
A recorded trajectory in Newton's viewer.

`newton.viewer.ViewerGL` supplies the parts that are not specific to this
simulator -- an OpenGL window, a free-flying camera, an ImGui sidebar and an
offscreen frame grab -- and this module supplies the parts that are: what a lodos
scene looks like, what a contact overlay means, and a panel to scrub and toggle
them with.

Nothing here talks to the device. It draws `viewer.trajectory.Trajectory`,
which is host-side by the time it arrives, so the viewer can be pointed at a
recording loaded from disk with no simulation context present at all.

## What gets drawn

Geometry comes from the collision surfaces' own vertices and faces, so what you
see is what was simulated -- not a proxy, and with no alignment to get wrong.
A half-space has no vertices, so it gets a quad; `plane_size` sets how big.

Each geometry has three independent controls: `geom_show` draws the solid mesh,
`geom_wire` draws its edges, and `geom_transparency` fades the solid one out. The
first two combine, so a geometry can be a wireframe laid over its own surface,
which is what reads best on a curved body.

`transparency` is the master above them, and it *multiplies* rather than
overwrites: the effective alpha of a geometry is `(1 - transparency)` times its
own. So the global knob is the one to reach for when the whole scene is in the
way of the overlays -- a masonry arch hiding the thrust line in its joints -- and
moving it does not cost the per-geometry settings underneath.

Transparency needed Newton to grow an alpha channel; `enable_transparency` is
where that happens and why.

The five contact overlays -- normals, relative velocities, and the contact force
with its normal and tangential parts -- are drawn as instanced 3D arrows, one mesh
scaled per contact. Newton's `log_lines`/`log_arrows` are screen-space and take
their width from a single renderer-wide setting, so per-overlay thickness is not
expressible through them; an instanced mesh gives every overlay its own shaft
radius in metres, and reads correctly at any zoom. Body frames and joint axes stay
as lines, sharing the renderer's `line_width`, because they are reference geometry
rather than measured vectors.

The overlays are independent checkboxes, so several show at once: the
interesting readings are comparisons, normal force against normal velocity, or
tangential force against sliding direction.

## Scales

An arrow draws a force or a velocity as a length, so every arrow overlay has a scale factor, and
`auto_scale` sets each so that a `scale_percentile` arrow over *the whole
trajectory* is one `arrow_length`. Both halves of that matter. Normalizing over
the trajectory rather than the frame is what keeps one frame's arrows comparable
with the next's. Normalizing to a percentile rather than the maximum is what keeps
them visible at all -- see `Trajectory.peak`, and the slider that exposes it,
because contact magnitudes span orders of magnitude and no one value reads well
across them.

## Backends

The drawing goes through `newton.viewer.ViewerBase`, so `ViewerViser` (browser),
`ViewerRerun` and `ViewerUSD` all render the scene. The panel exists twice, in
ImGui for `ViewerGL` and in viser's own GUI for `ViewerViser`, because the two
toolkits share no API and the browser one is what makes a headless machine
usable. Both drive the same attributes, and `OVERLAYS` and `ARROWS` are the single
lists they both build from.
Video rendering is `ViewerGL`-only -- it needs the offscreen frame grab -- and so
is per-geometry wireframe, `log_wireframe_shape` being a no-op elsewhere.

On any other backend `run` draws frames with no panel at all.
`TrajectoryViewer.draw` is public for that reason: it is the whole of the
rendering, and a caller driving its own loop can use it directly.
"""

import itertools
import time
from dataclasses import dataclass

import numpy as np
import warp as wp

# Overlay colors, RGB in [0, 1]: the normal force blue, the tangential force green.
COLOR_POINT = (0.05, 0.05, 0.05)
COLOR_NORMAL = (0.20, 0.80, 1.00)
COLOR_VELOCITY = (1.00, 0.85, 0.10)
COLOR_FORCE = (1.00, 0.10, 0.90)
COLOR_FORCE_N = (0.20, 0.30, 1.00)
COLOR_FORCE_T = (0.20, 0.90, 0.30)
COLOR_JOINT = (1.00, 0.50, 0.90)
COLOR_AXES = ((0.9, 0.1, 0.1), (0.1, 0.9, 0.1), (0.1, 0.4, 1.0))
COLOR_WIRE = (0.15, 0.15, 0.18)

# Geometry colors, cycled. A distinguishable set at mesh opacity, which is what
# separating five armadillos in a pile needs. Half-spaces are not cycled: a ground
# is a ground, and giving it a body's colour makes it read as one more object.
PALETTE = (
    (0.85, 0.35, 0.30),
    (0.35, 0.60, 0.85),
    (0.45, 0.75, 0.40),
    (0.90, 0.70, 0.25),
    (0.65, 0.45, 0.80),
    (0.40, 0.75, 0.75),
    (0.85, 0.55, 0.70),
    (0.60, 0.60, 0.60),
)
GROUND_COLOR = (0.80, 0.72, 0.58)

# What `geom_transparency` blends towards where the shader patch could not be
# applied and the slider has to fall back to de-emphasis. See `enable_transparency`.
FADE_TARGET = (0.72, 0.70, 0.66)

# Wireframe edges are pushed this far out along the surface normal, as a fraction
# of the mesh's own diagonal, so that a wireframe laid over its solid mesh reads as
# an outline instead of z-fighting with the surface it traces.
WIRE_OFFSET = 0.0025

# The overlay toggles, in panel order: attribute, label, default.
OVERLAYS = (
    ("show_points", "Contact points", True),
    ("show_normals", "Contact normals", True),
    ("show_velocity", "Contact velocities", False),
    ("show_force", "Contact force", True),
    ("show_force_n", "  normal component", False),
    ("show_force_t", "  tangential component", False),
    ("show_frames", "Body frames", False),
    ("show_joints", "Joint axes", False),
)

# The arrow overlays, in draw order: key and colour. `show_<key>` is the toggle
# and `thickness[<key>]` the shaft radius in metres; the vector each one draws is
# assembled in `_log_contacts`.
ARROWS = (
    ("normals", COLOR_NORMAL),
    ("velocity", COLOR_VELOCITY),
    ("force", COLOR_FORCE),
    ("force_n", COLOR_FORCE_N),
    ("force_t", COLOR_FORCE_T),
)

# Frame rate a video defaults to at most. Above this, playback is slow motion by
# `playback_rate` rather than the file being unopenable.
MAX_FPS = 120

# How much of the framed scene the WASDQE keys cross in a second. Newton's own
# default is a flat 4 m/s, which is a sensible number for a room and a useless
# one for a hand: a hand is 0.2 m across, so a tap of W is the whole thing.
# Tying the speed to what the camera was framed on makes it right in both, and
# the panel's slider is for taste on top of that.
CAMERA_SPEED_FACTOR = 0.6

# Slider bounds, as multiples of the framed extent per second. Wide, and
# logarithmic, because "too fast" and "too slow" are both a factor of ten away.
CAMERA_SPEED_RANGE = (0.02, 4.0)

# The mouse wheel's zoom: a notch dollies the camera toward its pivot, the distance
# shrinking by exp(-speed) whatever the scene's size (Newton's default 0.15 is 14 % a
# notch). Slider bounds, logarithmic: 1 % to 63 % a notch.
ZOOM_SPEED_RANGE = (0.01, 1.0)

# MuJoCo `mjtJoint`: free, ball, slide, hinge. Only the last two have an axis
# worth drawing -- a free joint has an `xaxis` too, and it means nothing.
JNT_FREE, JNT_BALL, JNT_SLIDE, JNT_HINGE = 0, 1, 2, 3



#######################################################################################
# Transparency
#######################################################################################

# An ordered 4x4 dither, in GLSL. Screen-door rather than blended transparency,
# which is what makes this a shader patch and not a renderer rewrite: blending
# would need the transparent objects split out of the draw list, sorted
# back-to-front and drawn with depth writes off, which is a hundred lines of
# `RendererGL.render` rather than a string. Dithering discards fragments instead,
# so the depth buffer stays exact, the draw order stops mattering, and two
# transparent bodies overlap correctly.
_DITHER = """
    if (alpha < 0.999) {
        const float BAYER[16] = float[16](
             0.0,  8.0,  2.0, 10.0,
            12.0,  4.0, 14.0,  6.0,
             3.0, 11.0,  1.0,  9.0,
            15.0,  7.0, 13.0,  5.0);
        ivec2 pix = ivec2(gl_FragCoord.xy) % 4;
        if (alpha * 16.0 <= BAYER[pix.y * 4 + pix.x]) discard;
    }
"""

# Alpha rides in `Material.z`, encoded as `alpha - 1`. That component is Newton's
# `checker_enable`, a flag it only ever sets non-negative, so the negative half of
# its range is free and the patched shader keeps the checker working for whatever
# still uses it. The encoding puts opaque at exactly 0.0, which is also what an
# unset material buffer contains -- so anything that does not know about this,
# Newton's own objects included, is opaque by construction.
_ALPHA_DECODE = """    float checker_enable = max(Material.z, 0.0);
    float alpha = 1.0 + min(Material.z, 0.0);"""

_SHADOW_IN = "layout (location = 8) in vec4 aMaterial;\nout float Alpha;\n\nuniform mat4 light_space_matrix;"
_SHADOW_SET = "    Alpha = 1.0 + min(aMaterial.z, 0.0);\n    mat4 transform = mat4("

# The checker's positive half: Newton draws squares 1 unit wide in the mesh's own frame,
# the darker at 0.7 of the lighter's linear colour (0.85 on screen). A positive `z` is
# now the square's width, so `z = 1` is still Newton's, and the darker square is at 0.4
# (about two thirds on screen), MuJoCo's contrast between its `rgb1` and `rgb2`. Both are
# optional: where Newton's lines have moved, the checker stays Newton's.
_CHECKER_PATCHES = (
    ("    float checker_scale = 1.0;", "    float checker_scale = checker_enable > 0.0 ? 1.0 / checker_enable : 1.0;"),
    ("        vec3 albedo2 = albedo*0.7;", "        vec3 albedo2 = albedo*0.4;"),
)

TRANSPARENCY = False


def enable_transparency():
    """
    Give Newton's GL viewer an alpha channel, by patching two shader sources.

    Newton 1.5's mesh shader ends in `FragColor = vec4(color, 1.0)` and its mesh
    pass never enables blending, so there is no alpha to set and nowhere to set it.
    Everything else needed is already there: `log_instances` takes a per-instance
    material `vec4` that reaches the fragment shader untouched, and one of its
    components has a free half. So this replaces two exact substrings -- the line
    that decodes `Material.z` and the line that writes the fragment -- and does the
    same for the shadow pass, so a transparent body casts a transparent shadow
    instead of a solid one. It also gives the checker a square of any width
    (`_CHECKER_PATCHES`), for the live viewer's ground.

    Called at import, because the shader program is built when a `ViewerGL` is
    constructed and a patch after that is too late. Idempotent, and it checks every
    substring before touching anything: on a Newton whose shaders have moved on it
    returns False, changes nothing, and `geom_transparency` falls back to fading the
    colour towards the background.
    """
    global TRANSPARENCY

    try:
        from newton._src.viewer.gl import shaders
    except Exception:
        return False

    if getattr(shaders, "_lodos_alpha", False):
        TRANSPARENCY = True
        return True

    frag = shaders.shape_fragment_shader
    sh_v = shaders.shadow_vertex_shader
    sh_f = shaders.shadow_fragment_shader

    checker = "    float checker_enable = Material.z;"
    out = "    FragColor = vec4(color, 1.0);"
    transform = "    mat4 transform = mat4("
    uniform = "uniform mat4 light_space_matrix;"
    empty = "void main() { }"

    if not (checker in frag and out in frag):
        return False
    if not (uniform in sh_v and transform in sh_v and empty in sh_f):
        return False

    frag = frag.replace(checker, _ALPHA_DECODE).replace(out, _DITHER + out)
    for old, new in _CHECKER_PATCHES:
        frag = frag.replace(old, new)
    shaders.shape_fragment_shader = frag
    # The shadow pass reads position and the instance transform only, so the
    # material attribute has to be declared before it can dither. Location 8 is
    # already bound on the same VAO the lit pass uses, and where it is not it
    # defaults to (0, 0, 0, 1), which decodes to alpha 1.
    shaders.shadow_vertex_shader = sh_v.replace(uniform, _SHADOW_IN).replace(
        transform, _SHADOW_SET
    )
    shaders.shadow_fragment_shader = sh_f.replace(
        empty, "in float Alpha;\n\nvoid main()\n{\n    float alpha = Alpha;\n" + _DITHER + "}"
    )

    shaders._lodos_alpha = True
    TRANSPARENCY = True
    return True


enable_transparency()


#######################################################################################
# Scene: the host-side rendering metadata
#######################################################################################


@dataclass
class VisScene:
    """
    Everything static a renderer needs and a `Trajectory` does not carry:
    meshes, colors, labels, and which joints have a drawable axis.
    """

    meshes: list  # [(vertices (V, 3), faces (F, 3))], one per distinct surface
    geom_mesh: np.ndarray  # (nB,) index into `meshes`
    geom_color: np.ndarray  # (nB, 3)
    geom_label: np.ndarray  # (nB,) strings, for the panel
    jnt_axis_ok: np.ndarray  # (njnt,) bool, hinge and slide joints
    scale: float = 1.0  # characteristic scene length, sets the overlay defaults

    def as_dict(self):
        """Flat numpy arrays, ready for `np.savez`. Meshes are ragged, so they
        concatenate behind offset arrays, as the trajectory's contacts do."""
        v = [np.asarray(a, np.float32) for a, _ in self.meshes]
        f = [np.asarray(b, np.int32) for _, b in self.meshes]
        return {
            "mesh_v": np.concatenate(v) if v else np.zeros((0, 3), np.float32),
            "mesh_f": np.concatenate(f) if f else np.zeros((0, 3), np.int32),
            "mesh_v_off": np.concatenate([[0], np.cumsum([len(a) for a in v])]).astype(np.int64),
            "mesh_f_off": np.concatenate([[0], np.cumsum([len(b) for b in f])]).astype(np.int64),
            "geom_mesh": np.asarray(self.geom_mesh, np.int64),
            "geom_color": np.asarray(self.geom_color, np.float32),
            "geom_label": np.asarray(self.geom_label, dtype="U32"),
            "jnt_axis_ok": np.asarray(self.jnt_axis_ok, bool),
            "scale": self.scale,
        }

    @classmethod
    def from_dict(cls, z):
        """Rebuild what `as_dict` produced."""
        vo, fo = z["mesh_v_off"], z["mesh_f_off"]
        meshes = [
            (z["mesh_v"][vo[k] : vo[k + 1]], z["mesh_f"][fo[k] : fo[k + 1]])
            for k in range(len(vo) - 1)
        ]
        return cls(
            meshes=meshes,
            geom_mesh=z["geom_mesh"],
            geom_color=z["geom_color"],
            geom_label=z["geom_label"],
            jnt_axis_ok=z["jnt_axis_ok"],
            scale=float(z["scale"]),
        )


def plane_mesh(size):
    """A quad for a half-space, in its geometry's body frame, where the SDF is z."""
    v = np.array(
        [[-size, -size, 0.0], [size, -size, 0.0], [size, size, 0.0], [-size, size, 0.0]],
        dtype=np.float32,
    )
    return v, np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)


def sphere_mesh(centres, radii, nu=14, nv=9):
    """A geometry's inner spheres, as one triangulated mesh.

    What a sphere-only geometry is drawn as. It carries no surface of its own, so
    it is drawn as the primitives it collides with.
    """
    u = np.linspace(0.0, 2.0 * np.pi, nu, endpoint=False)
    w = np.linspace(0.0, np.pi, nv)
    uu, ww = np.meshgrid(u, w, indexing="ij")
    unit = np.stack([np.cos(uu) * np.sin(ww),
                     np.sin(uu) * np.sin(ww),
                     np.cos(ww)], axis=-1).reshape(-1, 3)
    tri = []
    for i in range(nu):
        for j in range(nv - 1):
            a, b = i * nv + j, ((i + 1) % nu) * nv + j
            tri += [[a, b, a + 1], [b, b + 1, a + 1]]
    tri = np.asarray(tri, dtype=np.int32)

    v, f = [], []
    for k, (c, r) in enumerate(zip(centres, radii)):
        v.append(c + r * unit)
        f.append(tri + k * len(unit))
    if not v:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32)
    return (np.concatenate(v).astype(np.float32),
            np.concatenate(f).astype(np.int32))


def make_scene(m, mcol, surfaces, colors=None, plane_size=None):
    """
    Rendering metadata for the geometries `make_collision_context` was given.

    surfaces: the same list, in the same order. Surfaces are grouped by identity,
        so the usual `[surf] * n + [PlaneSurface()]` registers two meshes rather
        than `n + 1` and the instancing is what the renderer sees.

    colors: `(nB, 3)` in [0, 1], or None to cycle `PALETTE` over the real
        geometries and give every half-space `GROUND_COLOR`.

    plane_size: half-extent of the quad standing in for each half-space. Defaults
        to twenty times the scene's characteristic length, which puts the quad's
        edge outside the default camera framing so the ground reads as a ground.
    """
    spans = [
        float(np.ptp(np.asarray(s.v), axis=0).max())
        for s in surfaces
        if len(getattr(s, "v", ())) > 0
    ]
    if not spans:
        # Every geometry is spheres or a plane; fall back to the spheres' own
        # extent so the camera and the overlay defaults still have a length.
        r = mcol.prim_r.numpy()
        spans = [float(4.0 * r.max())] if len(r) and r.max() > 0 else []
    scale = float(np.median(spans)) if spans else 1.0
    plane_size = 20.0 * scale if plane_size is None else float(plane_size)

    is_plane = mcol.g_plane.numpy().astype(bool)
    g_bid = mcol.g_bid.numpy().astype(int)
    nB = len(surfaces)

    # `mcol.g_plane` decides what a half-space is, not "this surface has no
    # vertices": a geometry of inner spheres alone has none either, and taken for a
    # plane it would be drawn as a quad the size of the scene -- silently, since a
    # plane is a perfectly ordinary thing to see.
    g_nS = mcol.g_nS.numpy().astype(int)
    prim_r = mcol.prim_r.numpy()
    v_b = mcol.v_b.numpy()
    beg = np.concatenate([[0], np.cumsum(mcol.g_nP.numpy())]).astype(int)

    meshes, index, geom_mesh = [], {}, []
    for g, surf in enumerate(surfaces):
        key = id(surf)
        if key not in index:
            index[key] = len(meshes)
            v = np.asarray(getattr(surf, "v", ()), dtype=np.float32)
            if is_plane[g]:
                meshes.append(plane_mesh(plane_size))
            elif len(v):
                meshes.append((v, np.asarray(surf.f, dtype=np.int32)))
            else:
                # No surface and not a plane: it collides through its spheres, so
                # that is what it is.
                lo = beg[g] + int(mcol.g_nV.numpy()[g])
                hi = beg[g] + int(mcol.g_nP.numpy()[g])
                meshes.append(sphere_mesh(v_b[lo:hi], prim_r[lo:hi]))
                assert g_nS[g], (
                    f"geometry {g} has no vertices, no spheres and is not a "
                    "plane, so there is nothing to draw it as"
                )
        geom_mesh.append(index[key])

    if colors is None:
        body = 0
        colors = []
        for g in range(nB):
            if is_plane[g]:
                colors.append(GROUND_COLOR)
            else:
                colors.append(PALETTE[body % len(PALETTE)])
                body += 1
        colors = np.array(colors, dtype=np.float32)

    labels = [
        f"{g}: plane" if is_plane[g] else f"{g}: body {g_bid[g]}" for g in range(nB)
    ]

    jnt_type = np.asarray(m.jnt_type).astype(int)
    return VisScene(
        meshes=meshes,
        geom_mesh=np.array(geom_mesh, dtype=int),
        geom_color=np.asarray(colors, dtype=np.float32),
        geom_label=np.array(labels, dtype="U32"),
        jnt_axis_ok=np.isin(jnt_type, (JNT_SLIDE, JNT_HINGE)),
        scale=scale,
    )


def save_recording(path, traj, scene):
    """
    Write a trajectory and the scene it was recorded in to one `.npz`.

    Self-contained: a recording saved this way needs neither the assets nor a GPU
    nor lodos itself to be viewed again, which is what makes recording on one
    machine and looking at it on another work.
    """
    out = {f"traj/{k}": v for k, v in traj.as_dict().items()}
    out.update({f"scene/{k}": v for k, v in scene.as_dict().items()})
    np.savez_compressed(path, **out)
    return path


def load_recording(path):
    """The `(trajectory, scene)` that `save_recording` wrote."""
    from viewer.trajectory import Trajectory

    z = np.load(path)
    sub = lambda p: {k[len(p) :]: z[k] for k in z.files if k.startswith(p)}
    return Trajectory.from_dict(sub("traj/")), VisScene.from_dict(sub("scene/"))


#######################################################################################
# Overlay geometry, in numpy
#######################################################################################


def quat_mat(q):
    """`(..., 4)` quaternions in Warp's `(x, y, z, w)` order to `(..., 3, 3)`."""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def mat44(xform):
    """
    A 7-element Warp transform as the flat column-major 4x4 the GL wireframe
    shader wants. Column-major because `glUniformMatrix4fv` is called with
    `transpose=GL_FALSE`; `newton._src.viewer.viewer.ViewerBase._transform_to_mat44`
    is the same conversion, restated so this package needs nothing private.
    """
    m = np.zeros((4, 4), dtype=np.float32)
    m[:3, :3] = quat_mat(np.asarray(xform[3:], np.float32))
    m[:3, 3] = xform[:3]
    m[3, 3] = 1.0
    return np.ascontiguousarray(m.T).reshape(-1)


def vertex_normals(vertices, faces):
    """Area-weighted vertex normals, which is the cheap accumulation of face normals."""
    v = np.asarray(vertices, np.float64).reshape(-1, 3)
    f = np.asarray(faces, np.int64).reshape(-1, 3)

    tri = v[f]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])  # length is 2 * area

    n = np.zeros_like(v)
    for k in range(3):
        np.add.at(n, f[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)


def wireframe_edges(vertices, faces, color, offset=WIRE_OFFSET):
    """
    A mesh's deduplicated edges as the `(E * 2, 6)` interleaved
    `[px, py, pz, cr, cg, cb]` array `log_wireframe_shape` takes.

    Deduplicated because every interior edge is shared by two triangles, so the
    naive list is twice the size for the same picture.

    `offset` lifts each vertex along its normal, as a fraction of the mesh's own
    diagonal. Without it a wireframe drawn over its solid mesh z-fights with the
    surface it traces -- the wireframe pass has no polygon offset and the lines sit
    at exactly the surface depth. Lifting along the normal rather than scaling
    about the centroid is what makes it hold on a concave body.
    """
    v = np.asarray(vertices, np.float64).reshape(-1, 3)
    f = np.asarray(faces, np.int64).reshape(-1, 3)

    if offset:
        v = v + (offset * float(np.linalg.norm(np.ptp(v, axis=0)))) * vertex_normals(v, f)

    pairs = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    pairs = np.unique(np.sort(pairs, axis=1), axis=0)

    data = np.empty((len(pairs) * 2, 6), dtype=np.float32)
    data[0::2, :3], data[1::2, :3] = v[pairs[:, 0]], v[pairs[:, 1]]
    data[:, 3:] = np.asarray(color, np.float32)
    return data


def arrow_mesh(segments=12, head_frac=0.28, head_radius=2.6):
    """
    A unit arrow along +z: shaft radius 1, total length 1, pointing from the
    origin to `z = 1`.

    Instanced with a scale of `(r, r, L)` this gives an arrow of shaft radius `r`
    and length `L`, with the head a fixed fraction of the length -- which is the
    convention arrows are normally drawn with, and it comes out of the
    non-uniform scale for free.
    """
    a = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    ring = np.stack([np.cos(a), np.sin(a)], axis=-1)
    z0, z1 = 0.0, 1.0 - head_frac

    verts = [
        np.column_stack([ring, np.full(segments, z0)]),            # 0: shaft base
        np.column_stack([ring, np.full(segments, z1)]),            # 1: shaft top
        np.column_stack([head_radius * ring, np.full(segments, z1)]),  # 2: head base
        np.array([[0.0, 0.0, z0], [0.0, 0.0, z1], [0.0, 0.0, 1.0]]),   # centres, tip
    ]
    v = np.concatenate(verts).astype(np.float32)
    base, top, head = 0, segments, 2 * segments
    c_base, c_head, tip = 3 * segments, 3 * segments + 1, 3 * segments + 2

    f = []
    for i in range(segments):
        j = (i + 1) % segments
        f += [[base + i, base + j, top + j], [base + i, top + j, top + i]]  # shaft
        f += [[c_base, base + j, base + i]]                                 # bottom cap
        f += [[c_head, head + i, head + j]]                                 # head underside
        f += [[head + i, tip, head + j]]                                    # cone
    return v, np.array(f, dtype=np.int32)


def align_z(d):
    """
    `(N, 3)` directions to `(N, 4)` Warp-order quaternions taking `+z` onto them.

    The half-angle form, `q = [z x d, 1 + z . d]` normalized, which degenerates
    only for `d` antiparallel to `z` -- where any axis perpendicular to `z` is
    correct and `x` is chosen.
    """
    n = np.linalg.norm(d, axis=-1, keepdims=True)
    u = d / np.maximum(n, 1e-12)

    q = np.empty((len(u), 4), dtype=np.float64)
    q[:, 0], q[:, 1], q[:, 2] = -u[:, 1], u[:, 0], 0.0  # cross([0,0,1], u)
    q[:, 3] = 1.0 + u[:, 2]

    q[q[:, 3] < 1e-8] = (1.0, 0.0, 0.0, 0.0)  # 180 degrees about x
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def arrow_instances(starts, ends, radius):
    """`(xforms (N, 7), scales (N, 3))` drawing `arrow_mesh` from each start to each end."""
    d = np.asarray(ends, np.float64) - np.asarray(starts, np.float64)
    length = np.linalg.norm(d, axis=-1)

    xforms = np.concatenate([np.asarray(starts, np.float64), align_z(d)], axis=-1)
    scales = np.column_stack([np.full(len(d), radius), np.full(len(d), radius), length])
    return xforms, scales


def frame_segments(body_xform, length):
    """Three coloured axis segments per body, as `(starts, ends, colors)`."""
    pos, rot = body_xform[:, :3], quat_mat(body_xform[:, 3:])
    starts = np.repeat(pos, 3, axis=0)
    # Columns of the rotation are the body axes in world coordinates.
    ends = starts + length * np.transpose(rot, (0, 2, 1)).reshape(-1, 3)
    colors = np.tile(np.array(COLOR_AXES, dtype=np.float32), (len(body_xform), 1))
    return starts, ends, colors


def joint_segments(anchor, axis, ok, length):
    """One segment through each hinge or slide joint's anchor, along its axis."""
    a, u = anchor[ok], axis[ok]
    return a - 0.5 * length * u, a + 0.5 * length * u


#######################################################################################
# The viewer
#######################################################################################


class TrajectoryViewer:
    """
    Draws a `Trajectory` and gives it a panel to be driven from.

    traj: the recording, from `viewer.trajectory`.
    scene: from `make_scene`.
    viewer: a `newton.viewer.ViewerBase`. Defaults to a fresh `ViewerGL`.

    on_save_state: called with the frame index behind a "Save state" button,
        returning something printable -- normally where it wrote. None hides the
        button. A `Trajectory` holds drawable state, not `qpos`, so this viewer
        cannot write a simulation state on its own; whoever supplied the
        trajectory knows what it came from and does the writing, in this callback.
    """

    # Every path this logs is namespaced per instance, so that two `TrajectoryViewer`s
    # can share one backend -- two engines' runs of the same scene, side by side.
    # Without the namespace the second one rebinds the first's mesh prototypes, which
    # Newton refuses when the vertex counts differ ("Number of points does not match")
    # rather than resizing.
    _COUNT = itertools.count()

    def __init__(
        self,
        traj,
        scene,
        viewer=None,
        auto_scale=True,
        video_path="trajectory.mp4",
        frame_camera=True,
        on_save_state=None,
    ):
        if viewer is None:
            from newton.viewer import ViewerGL

            viewer = ViewerGL()

        self.traj = traj
        self.scene = scene
        self.viewer = viewer

        self.frame_index = 0
        self.playing = False
        # A multiple of *real time*, not of frames per render tick -- see
        # `_advance`. 1.0 plays the trajectory at the speed it was simulated at.
        self.play_rate = 1.0
        self._play_accum = 0.0
        self._play_last = None

        for attr, _, default in OVERLAYS:
            setattr(self, attr, default)

        # Per-geometry display, three independent controls:
        # `geom_show` draws the solid mesh, `geom_wire` draws its edges, and the two
        # together give a wireframe laid over its own surface, which is what reads
        # best on a curved body. `geom_transparency` is 0 opaque, 1 invisible.
        n = len(scene.geom_mesh)
        self.geom_show = [True] * n
        self.geom_wire = [False] * n
        self.geom_transparency = [0.0] * n
        # The master, composed with the per-geometry values rather than replacing
        # them. Alphas multiply, so 0 here is exactly a no-op.
        self.transparency = 0.0

        # Overlay sizes, in metres, from the scene's own scale.
        s = scene.scale
        self.point_radius = 0.02 * s
        self.normal_length = 0.15 * s
        self.frame_length = 0.30 * s
        self.arrow_length = 0.30 * s
        self.line_width = 1.5  # pixels, shared by the frame and joint lines
        self.thickness = {key: 0.010 * s for key, _ in ARROWS}
        self.auto_scale = auto_scale
        self.scale_percentile = 95.0
        self.velocity_scale = 1.0
        self.force_scale = 1.0
        if auto_scale:
            self.rescale()

        # What `frame_camera` last fitted, which is the length scale the camera
        # speed and its slider bounds are derived from. None until it runs.
        self.framed_extent = None
        self.camera_speed_factor = CAMERA_SPEED_FACTOR

        self.on_save_state = on_save_state
        self.video_path = video_path
        # Real time where that is playable, and capped where it is not: a 250 Hz
        # recording encoded at 250 fps is a file most players will not open, so the
        # cap trades exact timing for a video that runs. `playback_rate` says by
        # how much, and both the panel and `--fps` override it.
        self.video_fps = min(max(int(round(1.0 / traj.dt)), 1), MAX_FPS) if traj.dt > 0 else 30
        self.status = ""

        self._ns = f"/traj{next(TrajectoryViewer._COUNT)}"
        self.arrow_mesh = f"{self._ns}/mesh/arrow"

        self._meshes_logged = False
        self._panel_registered = False
        # Kept alive because `log_wireframe_shape` caches its VBO by `id()` of the
        # array it is handed; a temporary would be rebuilt every frame, and its
        # address could be reused by a different mesh.
        self._wire_data = {}

        if frame_camera:
            self.frame_camera()

    #######################################################################################
    # Camera
    #######################################################################################

    def set_camera_speed(self, metres_per_second):
        """Set how fast WASDQE fly the camera.

        Args:
            metres_per_second: The speed. Clamped to be positive, since the
                viewer rejects a negative one.

        Returns:
            The speed actually set, or None if this viewer has no such control
            -- `camera_speed` arrived in Newton after the file and null viewers,
            which are not flown by hand.
        """
        speed = max(float(metres_per_second), 1e-6)
        if not hasattr(type(self.viewer), "camera_speed"):
            return None
        self.viewer.camera_speed = speed
        return speed

    def camera_speed(self):
        """The current fly-through speed, in metres per second.

        Returns:
            The speed, or None if this viewer does not expose one.
        """
        return getattr(self.viewer, "camera_speed", None)

    def set_zoom_speed(self, speed):
        """Set how far a notch of the mouse wheel dollies the camera toward its pivot:
        the distance shrinks by `exp(-speed)`.

        Newton's viewer keeps this in its GUI's `_camera_dolly_scroll_sensitivity`,
        which nothing public sets.

        Args:
            speed: The zoom speed. Clamped to be positive.

        Returns:
            The speed actually set, or None if this viewer has no such control (the
            file and null viewers, or a Newton without it).
        """
        gui = getattr(self.viewer, "gui", None)
        if gui is None or not hasattr(gui, "_camera_dolly_scroll_sensitivity"):
            return None
        gui._camera_dolly_scroll_sensitivity = max(float(speed), 1e-6)
        return gui._camera_dolly_scroll_sensitivity

    def zoom_speed(self):
        """The current zoom speed (`set_zoom_speed`), or None if this viewer has none."""
        return getattr(getattr(self.viewer, "gui", None), "_camera_dolly_scroll_sensitivity", None)

    def frame_camera(self, t=None, azimuth=45.0, elevation=-20.0, distance=None):
        """
        Point the camera at the trajectory, so the first thing drawn is in view.
        After this the camera belongs to whoever is flying it.

        t: a timestep to frame, or None for the union of all of them. The union is
            the safe default and often a distant one -- a stack of armadillos falls
            eight metres and settles into one, so a framing that holds both holds
            the pile small. The panel's "Frame camera" button passes the timestep
            being looked at.

        azimuth, elevation: degrees. Negative elevation puts the camera above.
        distance: metres from the centre, or None for 1.2x the framed diagonal.
        """
        if not len(self.traj):
            return

        frames = self.traj if t is None else [self.traj[int(t)]]
        pts = np.concatenate([f.geom_xform[:, :3] for f in frames])
        if not len(pts):
            return

        lo, hi = pts.min(axis=0), pts.max(axis=0)
        centre = 0.5 * (lo + hi)
        extent = float(np.linalg.norm(hi - lo)) + self.scene.scale
        distance = 1.2 * extent if distance is None else float(distance)

        # Reframing is also the moment the scene's size is known, so it is where
        # the fly-through speed gets set. The panel can override it afterwards;
        # pressing "Frame camera" again puts it back.
        self.framed_extent = extent
        self.set_camera_speed(self.camera_speed_factor * extent)

        a, e = np.radians(azimuth), np.radians(elevation)
        offset = distance * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), -np.sin(e)])
        pos = centre + offset

        # `set_camera` takes the angles rather than the target, and the viewer's
        # camera is Z-up, so the direction has to be turned back into pitch and yaw.
        d = -offset / max(np.linalg.norm(offset), 1e-9)
        pitch = float(np.degrees(np.arcsin(np.clip(d[2], -1.0, 1.0))))
        yaw = float(np.degrees(np.arctan2(d[1], d[0])))
        self.viewer.set_camera(wp.vec3(*pos.astype(float)), pitch, yaw)

    #######################################################################################
    # Scaling
    #######################################################################################

    def rescale(self):
        """
        Set the velocity and force scales so that a `scale_percentile` arrow is
        `arrow_length` long, over the whole trajectory rather than the frame.

        The percentile is the parameter of this, and the panel exposes it,
        because contact magnitudes over a trajectory span orders of magnitude and
        no one setting reads well across all of them. Armadillos dropped onto a
        plane have a median contact force of 48 N, a 95th percentile of 586 and a
        maximum of 36 kN: normalize to the maximum and the settled pile draws
        nothing, normalize to the median and the landing fills the screen.
        """
        for name, field_name in (("velocity_scale", "con_v"), ("force_scale", "con_f")):
            peak = self.traj.peak(field_name, self.scale_percentile)
            setattr(self, name, self.arrow_length / peak if peak > 0 else 1.0)

    #######################################################################################
    # Drawing
    #######################################################################################

    def _vec3(self, a):
        """A contiguous float32 `wp.array` of vec3 from anything numpy-shaped."""
        return wp.array(np.ascontiguousarray(a, dtype=np.float32), dtype=wp.vec3)

    def _log_meshes(self):
        """
        Register the mesh prototypes, once. Hidden, every one of them.

        `log_mesh` does not merely register: it creates a scene object drawn at the
        identity transform in a default colour. Leaving that visible puts an
        untextured ghost of every mesh at the world origin, taking part in nothing.
        """
        for k, (v, f) in enumerate(self.scene.meshes):
            self.viewer.log_mesh(
                f"{self._ns}/mesh/{k}",
                self._vec3(v),
                wp.array(np.ascontiguousarray(f, dtype=np.int32).reshape(-1), dtype=wp.int32),
                hidden=True,
            )
            self._wire_data[k] = wireframe_edges(v, f, COLOR_WIRE)

        av, af = arrow_mesh()
        self.viewer.log_mesh(
            self.arrow_mesh,
            self._vec3(av),
            wp.array(np.ascontiguousarray(af, dtype=np.int32).reshape(-1), dtype=wp.int32),
            hidden=True,
        )
        self._meshes_logged = True

    def _appearance(self, which):
        """
        `(colors, materials)` for a set of geometries, carrying their transparency.

        Alpha rides in the material's `z` as `alpha - 1`; see `enable_transparency`.
        Where that patch could not be applied there is no alpha to ride in, so the
        slider falls back to blending the colour towards the background -- which
        de-emphasizes a geometry without letting you see through it.
        """
        alpha = (1.0 - self.transparency) * (
            1.0 - np.asarray([self.geom_transparency[g] for g in which], np.float32)
        )
        colors = self.scene.geom_color[which]

        if not TRANSPARENCY:
            t = (1.0 - alpha)[:, None]
            return colors * (1.0 - t) + np.asarray(FADE_TARGET, np.float32) * t, None

        materials = np.zeros((len(which), 4), dtype=np.float32)
        materials[:, 2] = alpha - 1.0
        return colors, materials

    def _log_geoms(self, frame):
        """
        One instance batch per mesh over the geometries drawn solid, and one
        wireframe shape per geometry drawn as edges. The two sets are independent,
        so a geometry can be in both.

        The solid batch is rebuilt each frame from the current settings, because an
        instance batch is all-or-nothing: hiding one geometry of a mesh means
        logging the others.
        """
        n = len(self.scene.geom_mesh)

        for k in range(len(self.scene.meshes)):
            which = [g for g in range(n) if self.geom_show[g] and self.scene.geom_mesh[g] == k]
            if not which:
                self.viewer.log_instances(
                    f"{self._ns}/geoms/{k}", f"{self._ns}/mesh/{k}",
                    None, None, None, None, hidden=True,
                )
                continue

            colors, materials = self._appearance(which)
            self.viewer.log_instances(
                f"{self._ns}/geoms/{k}",
                f"{self._ns}/mesh/{k}",
                wp.array(
                    np.ascontiguousarray(frame.geom_xform[which], dtype=np.float32),
                    dtype=wp.transformf,
                ),
                self._vec3(np.ones((len(which), 3))),
                self._vec3(colors),
                None if materials is None
                else wp.array(np.ascontiguousarray(materials), dtype=wp.vec4),
            )

        for g in range(n):
            data = self._wire_data.get(int(self.scene.geom_mesh[g])) if self.geom_wire[g] else None
            self.viewer.log_wireframe_shape(
                f"{self._ns}/wire/{g}", data, mat44(frame.geom_xform[g]),
                hidden=not self.geom_wire[g],
            )

    def _log_arrows(self, name, starts, ends, color, radius, hidden):
        """One arrow per segment, as instances of the shared arrow mesh."""
        if hidden or not len(starts):
            self.viewer.log_instances(f"{self._ns}/contacts/{name}", self.arrow_mesh,
                                      None, None, None, None, hidden=True)
            return

        xforms, scales = arrow_instances(starts, ends, radius)
        self.viewer.log_instances(
            f"{self._ns}/contacts/{name}",
            self.arrow_mesh,
            wp.array(np.ascontiguousarray(xforms, dtype=np.float32), dtype=wp.transformf),
            self._vec3(scales),
            self._vec3(np.tile(color, (len(xforms), 1))),
            None,
        )

    def _log_contacts(self, frame):
        """
        The six contact overlays.

        Everything is logged every frame, with `hidden` doing the toggling, so that
        switching one off leaves nothing behind from the frame before -- a viewer
        keeps whatever was last logged under a name until something replaces it.
        """
        p = frame.con_p
        fn, ft = frame.decompose("con_f")

        # `log_points` wants a per-point color array. Its signature says a triplet
        # will do, and on the GL backend it will not, so the expansion happens here.
        self.viewer.log_points(
            f"{self._ns}/contacts/points",
            self._vec3(p),
            radii=self.point_radius,
            colors=self._vec3(self._point_colors(frame)),
            hidden=not self.show_points,
        )

        vectors = {
            "normals": (frame.con_n, self.normal_length),
            "velocity": (frame.con_v, self.velocity_scale),
            "force": (frame.con_f, self.force_scale),
            "force_n": (fn, self.force_scale),
            "force_t": (ft, self.force_scale),
        }
        for key, color in ARROWS:
            vec, scale = vectors[key]
            self._log_arrows(key, p, p + scale * vec, color, self.thickness[key],
                             not getattr(self, f"show_{key}"))

    def _point_colors(self, frame):
        """One colour per contact point."""
        return np.tile(COLOR_POINT, (len(frame.con_p), 1))

    def _log_frames(self, frame):
        """Body frames and joint axes, as lines -- reference geometry, not measurements."""
        if hasattr(self.viewer, "renderer"):
            self.viewer.renderer.line_width = self.line_width

        starts, ends, colors = frame_segments(frame.body_xform, self.frame_length)
        self.viewer.log_lines(
            f"{self._ns}/frames/body",
            self._vec3(starts),
            self._vec3(ends),
            self._vec3(colors),
            hidden=not self.show_frames,
        )

        starts, ends = joint_segments(
            frame.jnt_anchor, frame.jnt_axis, self.scene.jnt_axis_ok, self.frame_length
        )
        self.viewer.log_lines(
            f"{self._ns}/frames/joint",
            self._vec3(starts),
            self._vec3(ends),
            self._vec3(np.tile(COLOR_JOINT, (len(starts), 1))),
            hidden=not self.show_joints,
        )

    def draw(self, t=None):
        """
        Draw one timestep, beginning and ending the viewer frame.

        The whole of the rendering: `run` calls it in a loop, and so does the video
        render, which is what makes the video show exactly what the window shows.
        """
        t = self.frame_index if t is None else t
        frame = self.traj[int(np.clip(t, 0, len(self.traj) - 1))]

        self.viewer.begin_frame(frame.time)
        self.log_frame(frame)
        self.viewer.end_frame()

    def log_frame(self, frame):
        """
        Everything one frame draws, between the viewer's `begin_frame` and
        `end_frame`: the geometries, the contact overlays and the reference frames.
        Separate from `draw` so that a caller drawing more into the same viewer
        frame -- `viewer` adds the bodies' velocities -- logs this
        and its own objects under one `begin_frame`.
        """
        if not self._meshes_logged:
            self._log_meshes()

        self._log_geoms(frame)
        self._log_contacts(frame)
        self._log_frames(frame)

    #######################################################################################
    # Video
    #######################################################################################

    @property
    def playback_rate(self):
        """Video speed against the simulation: 1.0 is real time, less is slower."""
        return self.video_fps * self.traj.dt if self.traj.dt > 0 else 1.0

    def render_video(self, path=None, fps=None):
        """
        Render the whole trajectory to a video from the camera's current pose, with
        the overlays that are currently switched on.

        Both of those come for free: the frames are drawn by `draw`, the same call
        the window uses, and the camera is whatever the user has flown it to. Frames
        are streamed to the encoder rather than accumulated -- 5000 of them at
        1080p is 31 GB in memory and a few hundred megabytes on disk.
        """
        import mediapy as media

        path = path or self.video_path
        fps = fps or self.video_fps

        self.draw(0)
        h, w = self.viewer.get_frame().shape[:2]

        with media.VideoWriter(path, shape=(h, w), fps=fps) as writer:
            for t in range(len(self.traj)):
                self.draw(t)
                writer.add_image(self.viewer.get_frame().numpy())

        return path

    #######################################################################################
    # Panel, for the GL backend
    #######################################################################################

    def _camera_ui(self, imgui):
        """The camera section: how fast the keys fly it, and how far the wheel zooms it.

        Args:
            imgui: The imgui module the panel is being drawn with.
        """
        speed = self.camera_speed()
        if speed is None:
            return

        imgui.set_next_item_open(True, imgui.Cond_.appearing)
        if not imgui.collapsing_header("Camera"):
            return

        extent = self.framed_extent or (20.0 * self.scene.scale)
        lo = CAMERA_SPEED_RANGE[0] * extent
        hi = CAMERA_SPEED_RANGE[1] * extent
        changed, value = imgui.slider_float(
            "Fly speed", speed, lo, hi, "%.3f m/s",
            imgui.SliderFlags_.logarithmic,
        )
        if changed:
            self.set_camera_speed(value)
            if extent > 0:
                self.camera_speed_factor = value / extent
        imgui.text(
            f"WASDQE; {self.camera_speed_factor:.2f} of the framed "
            f"{extent:.2f} m per second"
        )

        zoom = self.zoom_speed()
        if zoom is None:
            return
        changed, value = imgui.slider_float(
            "Zoom speed", zoom, ZOOM_SPEED_RANGE[0], ZOOM_SPEED_RANGE[1], "%.3f",
            imgui.SliderFlags_.logarithmic,
        )
        if changed:
            self.set_zoom_speed(value)
            zoom = value
        imgui.text(f"the wheel; {100.0 * (1.0 - np.exp(-zoom)):.0f} % closer a notch")

    def _ui(self, imgui):
        """The sidebar panel. Registered on `ViewerGL` only."""
        self._ui_trajectory(imgui)
        self._camera_ui(imgui)
        self._ui_overlays(imgui)
        self._ui_geometries(imgui)
        self._ui_scales(imgui)
        self._ui_thickness(imgui)
        self._ui_render(imgui)

    # The panel's sections, one method each so that `viewer` can compose its
    # own panel from the ones it keeps.

    def _ui_trajectory(self, imgui):
        """Scrubbing and playback."""
        frame = self.traj[self.frame_index]

        imgui.set_next_item_open(True, imgui.Cond_.appearing)
        if imgui.collapsing_header("Trajectory"):
            changed, value = imgui.slider_int(
                "Timestep", self.frame_index, 0, max(len(self.traj) - 1, 0)
            )
            if changed:
                self.frame_index = value
                self.playing = False

            imgui.text(f"t = {frame.time:.4f} s     contacts: {frame.ncon}")

            if imgui.button("|<"):
                self.frame_index, self.playing = 0, False
            imgui.same_line()
            if imgui.button("<"):
                self.frame_index, self.playing = max(self.frame_index - 1, 0), False
            imgui.same_line()
            if imgui.button(">"):
                self.frame_index = min(self.frame_index + 1, len(self.traj) - 1)
                self.playing = False
            imgui.same_line()
            _, self.playing = imgui.checkbox("Play", self.playing)

            _, self.play_rate = imgui.slider_float("Speed", self.play_rate, 0.1, 10.0, "%.1fx")

            if imgui.button("Frame camera"):
                self.frame_camera(self.frame_index)

            if self.on_save_state is not None:
                imgui.same_line()
                if imgui.button("Save state"):
                    try:
                        self.status = f"wrote {self.on_save_state(self.frame_index)}"
                    except Exception as exc:  # noqa: BLE001
                        # A failed save must not take the window with it: the
                        # button is pressed while looking at something worth
                        # keeping.
                        self.status = f"save failed: {exc}"

    def _ui_overlays(self, imgui):
        """The overlay toggles."""
        imgui.set_next_item_open(True, imgui.Cond_.appearing)
        if imgui.collapsing_header("Overlays"):
            for attr, label, _ in OVERLAYS:
                _, value = imgui.checkbox(label, getattr(self, attr))
                setattr(self, attr, value)

    def _ui_geometries(self, imgui):
        """Per-geometry display: show, wireframe, transparency."""
        if imgui.collapsing_header("Geometries"):
            self._ui_geometry_bulk(imgui)
            imgui.text("show  wire  transparency")
            for g, label in enumerate(self.scene.geom_label):
                self._ui_geometry_row(imgui, g, label)
            if not TRANSPARENCY:
                imgui.text("no alpha in this Newton build:")
                imgui.text("the slider fades colour instead.")

    def _ui_geometry_bulk(self, imgui):
        """The bulk buttons and the master transparency."""
        if imgui.button("Reset"):
            self._set_all(show=True, wire=False, transparency=0.0)
        imgui.same_line()
        if imgui.button("All wire"):
            self._set_all(wire=True)
        imgui.same_line()
        if imgui.button("No wire"):
            self._set_all(wire=False)

        _, self.transparency = imgui.slider_float(
            "All##globalalpha", self.transparency, 0.0, 1.0, "%.2f"
        )

    def _ui_geometry_row(self, imgui, g, label):
        """One geometry's show, wire and transparency controls."""
        _, self.geom_show[g] = imgui.checkbox(f"##show{g}", self.geom_show[g])
        imgui.same_line()
        _, self.geom_wire[g] = imgui.checkbox(f"##wire{g}", self.geom_wire[g])
        imgui.same_line()
        _, self.geom_transparency[g] = imgui.slider_float(
            f"{label}##alpha{g}", self.geom_transparency[g], 0.0, 1.0, "%.2f"
        )

    def _ui_scales(self, imgui):
        """Overlay sizes and the arrow scales."""
        if imgui.collapsing_header("Scales"):
            changed, self.auto_scale = imgui.checkbox("Auto scale", self.auto_scale)
            if changed and self.auto_scale:
                self.rescale()

            _, self.point_radius = imgui.slider_float(
                "Point radius", self.point_radius, 0.0, 0.2 * self.scene.scale, "%.4f"
            )
            _, self.normal_length = imgui.slider_float(
                "Normal length", self.normal_length, 0.0, self.scene.scale, "%.3f"
            )
            _, self.frame_length = imgui.slider_float(
                "Frame length", self.frame_length, 0.0, 2.0 * self.scene.scale, "%.3f"
            )
            _, self.line_width = imgui.slider_float(
                "Line width", self.line_width, 0.5, 10.0, "%.1f px"
            )

            if self.auto_scale:
                changed, self.arrow_length = imgui.slider_float(
                    "Arrow length", self.arrow_length, 0.0, 2.0 * self.scene.scale, "%.3f"
                )
                moved, self.scale_percentile = imgui.slider_float(
                    "Percentile", self.scale_percentile, 1.0, 100.0, "%.0f"
                )
                if changed or moved:
                    self.rescale()
                imgui.text(f"velocity x{self.velocity_scale:.3g}  force x{self.force_scale:.3g}")
            else:
                _, self.velocity_scale = imgui.input_float("Velocity scale", self.velocity_scale)
                _, self.force_scale = imgui.input_float("Force scale", self.force_scale)

    def _ui_thickness(self, imgui):
        """Arrow shaft radii."""
        if imgui.collapsing_header("Thickness"):
            hi = 0.1 * self.scene.scale
            for key in self.thickness:
                _, self.thickness[key] = imgui.slider_float(
                    key, self.thickness[key], 0.0, hi, "%.4f"
                )

    def _ui_render(self, imgui):
        """The video render."""
        if imgui.collapsing_header("Render"):
            _, self.video_path = imgui.input_text("Output", self.video_path)
            _, self.video_fps = imgui.slider_int("FPS", self.video_fps, 1, 240)
            imgui.text(f"{len(self.traj)} frames from the current camera")
            imgui.text(f"{self.playback_rate:.2f}x real time")
            if imgui.button("Render video"):
                self.status = f"rendering {len(self.traj)} frames..."
                try:
                    self.render_video()
                    self.status = f"wrote {self.video_path}"
                except Exception as exc:  # surfaced in the panel, not the console
                    self.status = f"render failed: {exc}"
            if self.status:
                imgui.text(self.status)

    def _set_all(self, show=None, wire=None, transparency=None):
        """The panel's bulk buttons, which is most of what a long geometry list needs."""
        for g in range(len(self.scene.geom_mesh)):
            if show is not None:
                self.geom_show[g] = show
            if wire is not None:
                self.geom_wire[g] = wire
            if transparency is not None:
                self.geom_transparency[g] = transparency

    #######################################################################################
    # Panel, for the browser backend
    #######################################################################################

    def _viser_panel(self, server):
        """
        The same controls in viser's GUI, for `ViewerViser`.

        Separate code from `_ui`: ImGui is immediate-mode and viser is retained-mode
        with callbacks, and the two do not meet in the middle. What they do share is
        the state they drive and the `OVERLAYS`/`ARROWS` tables, so a toggle added
        there appears in both.

        No render button -- the video path needs `ViewerGL.get_frame`. Per-geometry
        wireframe and transparency are present but inert: `log_wireframe_shape` is a
        GL-only path, and the alpha channel `enable_transparency` adds is a patch to
        the GL shaders. On this backend the transparency slider falls back to the
        colour blend, as it does on a GL build the patch did not fit.
        """
        with server.gui.add_folder("Trajectory"):
            slider = server.gui.add_slider(
                "Timestep", min=0, max=max(len(self.traj) - 1, 1), step=1,
                initial_value=self.frame_index,
            )
            play = server.gui.add_checkbox("Play", self.playing)
            rate = server.gui.add_number("Speed", initial_value=self.play_rate, step=0.1)
            frame_btn = server.gui.add_button("Frame camera")

            # Playback writes the slider, so its handler must not pause: it would
            # stop the playback that moved it.
            @slider.on_update
            def _(_):
                self.frame_index = int(slider.value)

            @play.on_update
            def _(_):
                self.playing = bool(play.value)

            @rate.on_update
            def _(_):
                self.play_rate = float(rate.value)

            @frame_btn.on_click
            def _(_):
                self.frame_camera(self.frame_index)

        with server.gui.add_folder("Overlays"):
            for attr, label, _ in OVERLAYS:
                box = server.gui.add_checkbox(label.strip(), getattr(self, attr))
                box.on_update(lambda _, a=attr, b=box: setattr(self, a, bool(b.value)))

        with server.gui.add_folder("Geometries"):
            master = server.gui.add_slider("Transparency (all)", min=0.0, max=1.0,
                                           step=0.01, initial_value=self.transparency)
            master.on_update(lambda _, h=master: setattr(self, "transparency", float(h.value)))

            for g, label in enumerate(self.scene.geom_label):
                with server.gui.add_folder(str(label)):
                    show = server.gui.add_checkbox("Show", self.geom_show[g])
                    wire = server.gui.add_checkbox("Wireframe", self.geom_wire[g])
                    alpha = server.gui.add_slider("Transparency", min=0.0, max=1.0, step=0.01,
                                                  initial_value=self.geom_transparency[g])
                    show.on_update(lambda _, i=g, h=show: self.geom_show.__setitem__(i, bool(h.value)))
                    wire.on_update(lambda _, i=g, h=wire: self.geom_wire.__setitem__(i, bool(h.value)))
                    alpha.on_update(
                        lambda _, i=g, h=alpha: self.geom_transparency.__setitem__(i, float(h.value))
                    )

        with server.gui.add_folder("Scales"):
            for attr, label, step in (
                ("point_radius", "Point radius", 1e-3),
                ("normal_length", "Normal length", 1e-2),
                ("frame_length", "Frame length", 1e-2),
                ("arrow_length", "Arrow length", 1e-2),
                ("scale_percentile", "Percentile", 1.0),
            ):
                num = server.gui.add_number(label, initial_value=getattr(self, attr), step=step)

                def _(_, a=attr, n=num):
                    setattr(self, a, float(n.value))
                    if a in ("arrow_length", "scale_percentile"):
                        self.rescale()

                num.on_update(_)

        with server.gui.add_folder("Thickness"):
            for key, _c in ARROWS:
                num = server.gui.add_number(key, initial_value=self.thickness[key], step=1e-3)
                num.on_update(lambda _, k=key, n=num: self.thickness.__setitem__(k, float(n.value)))

        return slider

    #######################################################################################
    # The loop
    #######################################################################################

    def _advance(self):
        """One tick of playback, paced against the wall clock.

        `play_rate` is a multiple of *real time*, not of frames per render
        tick, which is only the same thing when the display rate happens to
        equal `1 / traj.dt`: at a frame a tick, a recording taken every 2
        control steps of 10 ms would play at 1.2x on a 60 Hz window, and one
        taken every step at 0.6x. Pacing on elapsed time makes 1.0 mean real
        time whatever the display does and whatever `record_every` the
        recording used.

        A long stall -- a window resize, a slow first draw, a debugger -- would
        otherwise jump the playhead by however many frames the gap covers, so
        the elapsed time is capped before it is converted.
        """
        if not self.playing or len(self.traj) < 2:
            self._play_last = None
            return
        now = time.perf_counter()
        if self._play_last is None:
            self._play_last = now
            return
        elapsed, self._play_last = now - self._play_last, now
        dt = self.traj.dt if self.traj.dt > 0 else 1.0 / 60.0
        self._play_accum += min(elapsed, 0.25) * self.play_rate / dt
        step, self._play_accum = divmod(self._play_accum, 1.0)
        self.frame_index = int(self.frame_index + step) % len(self.traj)

    def run(self, fps=60.0):
        """
        Open the viewer and draw until it is closed.

        fps: the loop's pace on backends that do not block on a display. `ViewerGL`
            paces itself against the window, so this only bites on `ViewerViser`,
            where an unpaced loop would push frames at whatever the CPU manages.
        """
        slider = None
        if not self._panel_registered:
            if hasattr(self.viewer, "register_ui_callback"):
                self.viewer.register_ui_callback(self._ui, "panel")
            elif hasattr(self.viewer, "_server"):
                slider = self._viser_panel(self.viewer._server)
            self._panel_registered = True

        paced = not hasattr(self.viewer, "register_ui_callback")
        while self.viewer.is_running():
            self._advance()
            self.draw()
            if slider is not None and slider.value != self.frame_index:
                slider.value = self.frame_index
            if paced:
                time.sleep(1.0 / fps)

        self.viewer.close()
