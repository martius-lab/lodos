"""
The viewer every step under `tools/mesh_processing/` opens when it finishes.

One panel, one row per body: a visibility checkbox, a focus checkbox that shows
that body alone, and whatever the step wants to say about it. Above the rows, one
checkbox per *variant* -- the named versions of each body a step produced, such as
`input` and `watertight` -- so a whole set can be switched off at once. Colour is
per body and shared across its variants, so a part can be followed between them.

A variant may be a single mesh or a *list* of them, which is how a convex
decomposition is shown: the parts get distinct colours of their own rather than
the body's, because the thing being judged there is where the cuts fell.

It may also be `Lines` or `Points` rather than a surface, for the things a step
wants to show that a mesh cannot say -- the order a space-filling curve put the
vertices in, a voxel grid, a field of normals. They carry `.vertices` so the
camera frames them like anything else.

`exclusive=True` drops the per-body visibility checkbox and leaves the focus one,
which turns the panel into a radio button over the bodies. That is for a step whose
subject is one body at a time: several bodies drawn in their own frames land on top
of each other, so showing two is not a view of anything.

A variant named in `wire` is drawn as edges instead of a surface: a remesh sits
*inside* the mesh it came from, so drawing both solid shows only the original and
hides the thing being judged.

A step that recomputes geometry while the window is open -- anything with a
slider -- subclasses this: `on_frame` to pump its work, `extra_header` and
`extra_row` to add widgets, and `Body.dirty` to say a body's meshes changed under
a variant name the viewer already knows about.

Three things about Newton this works around.

**Layer visibility does not hide a `log_mesh`.** `set_layer_visible` documents that
it hides "every object the layer owns ... on the next `log_state` / `log_contacts`
cycle", which is how a *model* is drawn; meshes logged directly are never
revisited, so the flag flips and the geometry stays on screen. Hiding here is
therefore done by logging nothing, and `_sync` reads the flag back each frame so
that Newton's own Layers panel drives the same path and the two agree.

**A panel registered at "side" or "free" disappears.** `remove_layer` calls
`clear_model`, which calls `clear_example_callbacks`, which drops every "side" and
"free" callback -- so a panel in either slot survives exactly until the first
redraw. "panel" survives, and the window is opened from there.

**`log_mesh` binds a name to a fixed vertex count.** Relogging different geometry
under the same name is refused rather than resized, so a body's layer is torn down
with `remove_layer` and rebuilt. That also means one layer per body, not per mesh:
Newton lists a checkbox per layer, and 46 of them is not a panel anyone can use.
"""

from __future__ import annotations

import colorsys
import traceback

import numpy as np
import warp as wp

WIRE_RGB = (0.05, 0.05, 0.07)
# What `transparency` blends towards when the caller says alpha is not available.
# Newton's own shape shader writes `vec4(color, 1.0)`, so a solid is opaque unless
# something has patched it -- `viewer.draw.enable_transparency` does, and a
# step that has called it passes `transparency=True` here. Without it, fading the
# colour de-emphasizes a solid without letting you see through it.
FADE_RGB = (0.55, 0.58, 0.62)
GOLDEN = 0.6180339887498949
SAT = (0.90, 0.62, 0.78)
VAL = (0.95, 0.72, 0.86)


def wireframe(mesh):
    """`(starts, ends)` for every unique edge."""
    e = mesh.edges_unique
    return mesh.vertices[e[:, 0]], mesh.vertices[e[:, 1]]


class Lines:
    """Explicit segments as a variant. `color=None` takes the body's own."""

    def __init__(self, starts, ends, color=None, width=None):
        self.starts = np.asarray(starts, dtype=np.float64).reshape(-1, 3)
        self.ends = np.asarray(ends, dtype=np.float64).reshape(-1, 3)
        self.color = color
        self.width = width

    @property
    def vertices(self):
        """What the camera frames on, and what `Body.posed` transforms."""
        return np.vstack([self.starts, self.ends])


class Points(Lines):
    """A point cloud as a variant, drawn at `radius`."""

    def __init__(self, points, radius, color=None):
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        super().__init__(pts, pts, color)
        self.radius = radius


def body_color(i):
    """Successive bodies ~137 degrees apart on the hue wheel, so no two are close."""
    return colorsys.hsv_to_rgb((i * GOLDEN) % 1.0, SAT[i % 3], VAL[i % 3])


def part_color(i):
    """The same stepping, for the parts *within* one body's list variant."""
    return colorsys.hsv_to_rgb((i * GOLDEN) % 1.0, 0.72 + 0.18 * (i % 2), 0.95 - 0.15 * (i % 3))


class Body:
    """One body, its named variants, and what the viewer last drew for it."""

    def __init__(self, name, variants, color, pose=None, info="", fade=0.0):
        self.name = name
        self.variants = variants          # {variant name: Trimesh | [Trimesh]}
        self.color = color
        self.pose = pose
        self.info = info
        # Drawn this much more transparent than the viewer's own slider, so one
        # step can show a thing *inside* another: the outer body follows the
        # slider and the inner one stays legible under it. Added, then clamped.
        self.fade = float(fade)
        self.visible = True
        self.dirty = False                # geometry changed under an unchanged name
        self.rendered = None              # (visible, tuple of shown variants)

    def posed(self, v):
        if self.pose is None:
            return v
        r, t = self.pose
        return v @ r.T + t


class MeshViewer:
    """
    `MeshViewer(bodies, variants, title=...).run()`.

    bodies: list of `Body`. variants: the variant names, in panel order; the first
    is shown by default and the rest are off, which keeps a two-variant step from
    opening with one set buried inside the other.
    """

    def __init__(self, bodies, variants, title="Meshes", subtitle="", default_on=None,
                 wire=(), line_width=2.0e-4, line_px=1.5, exclusive=False,
                 transparency=False):
        self.bodies = bodies
        self.variants = list(variants)
        self.wire = set(wire)
        self.exclusive = exclusive
        self.alpha_ok = bool(transparency)
        self.line_width = line_width
        # Newton draws lines as screen-space quads and documents `log_lines(width=)` as
        # "currently ignored"; the pixel width comes from the renderer instead, so a
        # slider has to drive this and not the per-batch argument. It is a uniform read
        # at draw time, so changing it needs no redraw.
        self.line_px = line_px
        self.title = title
        self.subtitle = subtitle
        on = self.variants[:1] if default_on is None else list(default_on)
        self.show = {v: (v in on) for v in self.variants}
        self.focused = None
        self.fade = 0.0                   # 0 opaque, 1 invisible (or fully faded)
        self.message = ""
        self._ui_failed = False
        self.viewer = None

    #######################################################################################
    # Drawing
    #######################################################################################

    def _shown_variants(self):
        return tuple(v for v in self.variants if self.show[v])

    def _entries(self, b, name):
        """`[(mesh, rgb)]` for one variant -- a list variant colours its parts apart."""
        value = b.variants.get(name)
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return [(m, part_color(i)) for i, m in enumerate(value)]
        return [(value, b.color)]

    def _redraw(self, b):
        v = self.viewer
        if b.name in v.layers:
            b.visible = v.layers[b.name].visible
            v.remove_layer(b.name)
        v.activate(b.name)

        if b.visible:
            shown = self._shown_variants()
            # A wireframe reads as dark over a surface and as its own colour against the
            # background, so which one it gets depends on whether a solid is under it.
            over_solid = any(
                n not in self.wire
                and any(not isinstance(m, Lines) for m, _ in self._entries(b, n))
                for n in shown
            )
            for name in shown:
                for i, (mesh, rgb) in enumerate(self._entries(b, name)):
                    key = f"{b.name}/{name}{i:03d}"
                    if isinstance(mesh, Points):
                        pts = np.ascontiguousarray(b.posed(mesh.starts), np.float32)
                        # One colour per point, not one colour. `log_points` documents a
                        # tuple as acceptable and the GL backend then calls `.numpy()` on
                        # it; `log_mesh` and `log_lines` really do take the tuple.
                        v.log_points(
                            key,
                            wp.array(pts, dtype=wp.vec3),
                            radii=float(mesh.radius),
                            colors=wp.array(
                                np.tile(np.float32(mesh.color or rgb), (len(pts), 1)),
                                dtype=wp.vec3,
                            ),
                        )
                        continue
                    if isinstance(mesh, Lines):
                        v.log_lines(
                            key,
                            wp.array(np.ascontiguousarray(b.posed(mesh.starts), np.float32), dtype=wp.vec3),
                            wp.array(np.ascontiguousarray(b.posed(mesh.ends), np.float32), dtype=wp.vec3),
                            tuple(float(c) for c in (mesh.color or rgb)),
                            width=mesh.width or self.line_width,
                        )
                        continue
                    if name in self.wire:
                        starts, ends = wireframe(mesh)
                        v.log_lines(
                            key,
                            wp.array(np.ascontiguousarray(b.posed(starts), np.float32), dtype=wp.vec3),
                            wp.array(np.ascontiguousarray(b.posed(ends), np.float32), dtype=wp.vec3),
                            WIRE_RGB if over_solid else tuple(float(c) for c in rgb),
                            width=self.line_width,
                        )
                        continue
                    # Registered hidden, then drawn as a single instance at identity.
                    # `log_mesh` takes no material, and the material's `z` is the only
                    # place an alpha can ride; see `enable_transparency`. The vertices
                    # are already posed, so the instance transform is the identity.
                    v.log_mesh(
                        key,
                        wp.array(np.ascontiguousarray(b.posed(mesh.vertices), np.float32), dtype=wp.vec3),
                        wp.array(np.ascontiguousarray(mesh.faces, np.int32).reshape(-1), dtype=wp.int32),
                        hidden=True,
                    )
                    v.log_instances(
                        f"{key}/i", key,
                        wp.array([wp.transformf(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())],
                                 dtype=wp.transformf),
                        wp.array(np.ones((1, 3), np.float32), dtype=wp.vec3),
                        wp.array(np.asarray([self._solid_rgb(rgb, b)], np.float32), dtype=wp.vec3),
                        self._material(b),
                    )

        v.set_layer_visible(b.name, b.visible)
        b.dirty = False
        b.rendered = (b.visible, self._shown_variants())

    def _fade(self, b=None):
        """The fade a body is drawn at: the slider plus its own offset."""
        return min(1.0, self.fade + (b.fade if b is not None else 0.0))

    def _solid_rgb(self, rgb, b=None):
        """A solid's colour: its own where alpha is real, faded where it is not."""
        if self.alpha_ok:
            return tuple(float(c) for c in rgb)
        f = self._fade(b)
        return tuple(float((1.0 - f) * c + f * bg) for c, bg in zip(rgb, FADE_RGB))

    def _material(self, b=None):
        """`alpha - 1` in the material's `z`, or None where the shader has no alpha."""
        if not self.alpha_ok:
            return None
        m = np.zeros((1, 4), dtype=np.float32)
        m[0, 2] = -self._fade(b)      # alpha - 1, with alpha = 1 - fade
        return wp.array(m, dtype=wp.vec4)

    def redraw_all(self):
        """Mark every body for redraw, after something that changes how all are drawn."""
        for b in self.bodies:
            b.dirty = True

    def _sync(self):
        """Pick up visibility set anywhere else -- Newton's own Layers panel, mostly."""
        self.viewer.renderer.line_width = float(self.line_px)
        want = self._shown_variants()
        for b in self.bodies:
            layer = self.viewer.layers.get(b.name)
            if layer is not None:
                b.visible = layer.visible
            if b.dirty or b.rendered != (b.visible, want):
                self._redraw(b)

    #######################################################################################
    # Visibility
    #######################################################################################

    def _is_visible(self, b):
        layer = self.viewer.layers.get(b.name)
        return layer.visible if layer is not None else b.visible

    def set_all(self, visible):
        self.focused = None
        for b in self.bodies:
            self._set_visible(b, visible)

    def focus(self, name):
        """
        Show `name` alone, or everything when `name` is None.

        Exclusive by construction: the state is one body name, so focusing a second
        releases the first with no way for two to be set.
        """
        self.focused = name
        for b in self.bodies:
            self._set_visible(b, name is None or b.name == name)

    def _set_visible(self, b, visible):
        """
        The flag first, the layer only if it exists.

        A layer is created by the first `_redraw`, so anything that sets visibility
        before the first frame -- a step that opens already focused on one body --
        has no layer to set and `set_layer_visible` raises `KeyError`. Writing
        `b.visible` is enough on its own: `_sync` redraws whenever it disagrees with
        what was last drawn, and reads the layer back when there is one.
        """
        b.visible = visible
        if self.viewer is not None and b.name in self.viewer.layers:
            self.viewer.set_layer_visible(b.name, visible)

    #######################################################################################
    # Panel
    #######################################################################################

    def ui(self, imgui):
        """Draw the panel, reporting rather than swallowing anything it raises."""
        try:
            self._ui(imgui)
        except Exception:
            if not self._ui_failed:
                self._ui_failed = True
                traceback.print_exc()

    def _ui(self, imgui):
        imgui.set_next_window_size(imgui.ImVec2(560, 700), imgui.Cond_.first_use_ever)
        imgui.set_next_window_pos(imgui.ImVec2(1000, 40), imgui.Cond_.first_use_ever)
        expanded, _ = imgui.begin(self.title)
        if not expanded:
            imgui.end()
            return

        if self.subtitle:
            imgui.text(self.subtitle)
        self.extra_header(imgui)

        for name in self.variants:
            changed, value = imgui.checkbox(name, self.show[name])
            if changed:
                self.show[name] = value
            imgui.same_line()
        imgui.new_line()

        if not self.exclusive:
            if imgui.button("All on"):
                self.set_all(True)
            imgui.same_line()
            if imgui.button("All off"):
                self.set_all(False)
            imgui.same_line()
        if imgui.button("Clear focus"):
            self.focus(None)

        imgui.separator()
        imgui.text("     body" if self.exclusive else "vis  foc   body")

        for i, b in enumerate(self.bodies):
            if not self.exclusive:
                changed, value = imgui.checkbox(f"##vis{i}", self._is_visible(b))
                if changed:
                    self.focused = None
                    self.viewer.set_layer_visible(b.name, value)
                imgui.same_line()

            changed, value = imgui.checkbox(f"##foc{i}", self.focused == b.name)
            if changed:
                self.focus(b.name if value else None)
            imgui.same_line()

            if self.extra_row(imgui, i, b):
                imgui.same_line()

            imgui.text(f"{b.name:<26s} {b.info}")

        imgui.end()

    #######################################################################################
    # Hooks -- no-ops here, for steps that recompute while the window is open
    #######################################################################################

    def extra_header(self, imgui):
        """Widgets under the subtitle. Return value ignored."""

    def extra_row(self, imgui, i, body):
        """Widgets between `foc` and the name. Return True if anything was drawn."""
        return False

    def on_frame(self):
        """Called once per frame before anything is redrawn."""

    def on_close(self):
        """Called after the window closes."""

    #######################################################################################

    def frame_camera(self, bodies=None):
        pts = np.vstack([
            b.posed(mesh.vertices)
            for b in (self.bodies if bodies is None else bodies)
            for name in b.variants
            for mesh, _ in self._entries(b, name)
        ] or [np.zeros((1, 3))])
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        centre = 0.5 * (lo + hi)
        dist = 1.6 * float(np.linalg.norm(hi - lo))
        a, e = np.radians(35.0), np.radians(-12.0)
        off = dist * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), -np.sin(e)])
        d = -off / np.linalg.norm(off)
        self.viewer.set_camera(
            wp.vec3(*(centre + off).astype(float)),
            float(np.degrees(np.arcsin(np.clip(d[2], -1, 1)))),
            float(np.degrees(np.arctan2(d[1], d[0]))),
        )
        # Newton's camera assumes a scene metres across and starts its orbit pivot 5 m
        # in front of itself, which is what makes panning unusable on a 10 cm object.
        self.viewer.camera_speed = 0.4 * dist
        self.viewer.camera.near = max(1e-4, 0.002 * dist)
        # `far` has to follow `near`, and Newton leaves it at 1000 m. On a 3 cm body
        # that is a ratio of ten million, which a 24-bit depth buffer cannot resolve:
        # the sky quad and the cleared background then win the depth test in
        # alternating patches, and the window fills with blue polygons behind the
        # geometry. Ten thousand is the usual ceiling for the ratio.
        self.viewer.camera.far = 1.0e4 * self.viewer.camera.near
        self.viewer.camera.set_pivot(wp.vec3(*centre.astype(float)))

    def run(self, viewer=None):
        if viewer is None:
            from newton.viewer import ViewerGL

            viewer = ViewerGL()
        self.viewer = viewer
        self.frame_camera()
        viewer.register_ui_callback(self.ui, "panel")
        while viewer.is_running():
            self.on_frame()
            viewer.begin_frame(0.0)
            self._sync()
            viewer.end_frame()
        viewer.close()
        self.on_close()
