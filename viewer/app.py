"""
The live viewer: simulate a scene, edit its state, switch sims, scrub its takes.

    python -m viewer                    # an empty window: load a scene from its panel
    python -m viewer arm_box_6          # a built-in scene (see `scenes.BUILTIN`)
    python -m viewer path/to/model.xml  # an MJCF file

`LiveViewer` is a `viewer.draw.TrajectoryViewer` whose trajectory is the take a
`viewer.session.Session` is looking at, so everything that viewer draws --
geometry with per-geometry wireframe and transparency, contact points, normals,
velocities and forces, body frames, joint axes, the video render -- is drawn the same
way here. What this adds:

- **Scene** (sidebar): a built-in scene, or an MJCF file found by typing part of its
  path into the search box (every MJCF under the search roots is indexed in the
  background), or picked in the system's file dialog.
- **Sim** (sidebar): the sim the scene runs under -- Lodos, or for comparison Newton's
  MuJoCo Warp or Kamino -- and its options, each with an arrow for the way that
  makes the sim faster (`viewer.session.speed_arrow`). A sim's first use builds it. Export
  sim config writes everything the panel sets to a JSON file (`Session.export_config`).
- **Capacities** (sidebar): every capacity of the sim -- Lodos's, or Newton's under its
  sims -- and the largest demand on it so far; Tune capacities sets each to its peak times
  one plus the overhead (`Session.tune_capacities`).
- **Collision pairs** (sidebar): under Lodos, every Lodos pair and every pair MJWarp's
  pipeline collides, and under Newton's MuJoCo Warp every pair of shapes Newton's pipeline
  collides: friction, solref and solimp set on all of them at once, or on the pairs a filter
  on their names matches, and Lodos's pairs made one- or two-sided (`viewer.pairs`).
- **State** (sidebar): the selected body's joints, position and velocity fields per
  joint. Click a body in the view to select it; its joints get gizmos, dragged as in
  viser: a free joint's translate and rotate handles, a hinge's ring about its axis, a
  slide's arrow along it. Every body's linear and angular velocity is drawn as an
  arrow from its origin.
- **Timeline** (the strip at the bottom): the takes, play/pause, speed, and the
  current take's frames with four traces under them -- contacts per pipeline, the
  deepest penetration, kinetic energy and solver iterations. Click or drag in a trace
  to seek.
- **Recording** (sidebar): how often a take records, and a take saved whole to an
  `.npz` and loaded back, to scrub and to continue (`viewer.session.save_take`).

Editing (a gizmo, a field, the sim, an option or a parameter) makes a draft from the frame
being looked at; Play runs it as a new take. See `viewer.session`.

Keys: Space play/pause, Right/Left one frame, F frame the camera, H hide the panels,
Escape quit.
"""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path

import numpy as np
import warp as wp
from newton.viewer import ViewerGL

from viewer import draw as V
from viewer import joints as J
from viewer.scenes import catalog as LS
from viewer.collect import SRC_LODOS, SRC_PRIM
from viewer.geometry import KIND_PRIM, KIND_NAMES, KIND_PLANE
from viewer.performance import Performance
from viewer import pairs as VP
from viewer.session import (CAPACITY_OVERHEAD, OPTIONS, SIM_INFO, SIM_NAMES, SIMS, Session, TrajectoryView, arrow_tip,
                            default_config, edit_config, edit_all_pairs, edit_pairs, normalize_option, runner_key,
                            save_take, speed_arrow, take_meta)

# A second palette for MJWarp's contacts, so the two pipelines read apart in one pile:
# warm where Lodos's are cool.
COLORS = {
    SRC_LODOS: dict(points=V.COLOR_POINT, normals=V.COLOR_NORMAL, velocity=V.COLOR_VELOCITY, force=V.COLOR_FORCE,
                    force_n=V.COLOR_FORCE_N, force_t=V.COLOR_FORCE_T),
    SRC_PRIM: dict(points=(0.95, 0.45, 0.05), normals=(0.10, 0.65, 0.55), velocity=(0.95, 0.60, 0.20),
                  force=(1.00, 0.30, 0.10), force_n=(0.55, 0.20, 0.85), force_t=(0.55, 0.85, 0.15)),
}
PIPELINES = ((SRC_LODOS, "Lodos"), (SRC_PRIM, "MJCF"))
CONTACT_OVERLAYS = (("points", "points"), ("normals", "normals"), ("velocity", "velocities"),
                    ("force", "forces"), ("force_n", "  normal part"), ("force_t", "  tangential part"))
COLOR_BODY_V = (0.10, 0.75, 0.95)
COLOR_BODY_W = (0.95, 0.75, 0.10)
COLOR_SELECT = (1.0, 0.95, 0.3)
TRACE_COLORS = ((0.30, 0.55, 1.00), (1.00, 0.45, 0.10), (0.85, 0.25, 0.30), (0.35, 0.80, 0.40), (0.70, 0.50, 0.95))

TIMELINE_HEIGHT = 170  # pixels, at DPI scale 1
SIDEBAR = 320


def default_roots():
    """Where the MJCF search looks: the working directory, this repository, the home directory."""
    here = Path(__file__).resolve().parents[1]
    roots = [Path.cwd(), here, Path.home()]
    out = []
    for r in roots:
        if r not in out:
            out.append(r)
    return out


class FileIndex:
    """Every MJCF under some roots, found in a background thread."""

    def __init__(self, roots):
        self.roots = [Path(r) for r in roots]
        self.paths = []
        self.done = False
        self.scan()

    def scan(self):
        self.done = False

        def work():
            self.paths = LS.find_mjcf(self.roots)
            self.done = True

        threading.Thread(target=work, daemon=True).start()


class LiveViewerGL(ViewerGL):
    """Newton's GL viewer with the input the live viewer needs: gizmo drags, body picking, keys."""

    def __init__(self, app, **kw):
        self.app = app
        self._press = None
        super().__init__(**kw)
        self.renderer.set_title("Lodos live")

    def on_mouse_press(self, x, y, button, modifiers):
        self._press = (x, y, button)
        if self.app.gizmo_hot:
            return
        super().on_mouse_press(x, y, button, modifiers)

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        # A gizmo being dragged owns the mouse: without this the camera turns too.
        if self.app.gizmo_hot or self.app.gizmo_active:
            return
        super().on_mouse_drag(x, y, dx, dy, buttons, modifiers)

    def on_mouse_release(self, x, y, button, modifiers):
        import pyglet

        press, self._press = self._press, None
        super().on_mouse_release(x, y, button, modifiers)
        if press is None or button != pyglet.window.mouse.LEFT:
            return
        # A click, not the end of a camera drag; hands move a few pixels while clicking,
        # and more of them at a 2x UI scale.
        slack = 6.0 * (self.gui.ui.dpi_scale if self.gui is not None else 1.0)
        if abs(press[0] - x) + abs(press[1] - y) > slack or self.app.gizmo_hot or self.app.gizmo_active:
            return
        if self.gui is not None and self.gui.is_mouse_capturing():
            return
        fx, fy = self._to_framebuffer_coords(x, y)
        start, d = self.camera.get_world_ray(fx, fy)
        self.app.pick(np.array([start[0], start[1], start[2]]), np.array([d[0], d[1], d[2]]))

    def on_key_press(self, symbol, modifiers):
        import pyglet

        if self.gui is not None and self.gui.is_keyboard_capturing():
            return
        k = pyglet.window.key
        if symbol == k.SPACE:
            self.app.session.toggle() if self.app.session.loaded else None
        elif symbol == k.RIGHT:
            self.app.step_frame(+1)
        elif symbol == k.LEFT:
            self.app.step_frame(-1)
        elif symbol == k.F:
            self.app.frame_camera_here()
        elif symbol in (k.H, k.ESCAPE):
            super().on_key_press(symbol, modifiers)


class LiveViewer(V.TrajectoryViewer):
    """
    The live viewer. `session` defaults to an empty one; `roots` are where the MJCF
    search looks. `run` opens the window and returns when it is closed.
    """

    def __init__(self, session=None, roots=None, width=None, height=None, viewer=None):
        self.session = session or Session()
        if viewer is None:
            if width is None or height is None:
                width, height = _screen_size(width, height)
            viewer = LiveViewerGL(self, width=width, height=height, vsync=False)
        self.viewer = viewer
        self.index = FileIndex(roots or default_roots())
        self.scene = None
        self.query = ""
        self.matches = []
        self._match_key = None
        self.pick_file = None
        self.builtin = LS.PANEL[0]
        self.seed = 0
        self.sim0 = "lodos"
        self.live_status = ""
        self.pending = None  # (callable, message): blocking work run under the splash
        self._pending_shown = 0
        # the Collision pairs folder: each kind's filter, and its values for the configuration on screen
        self.pair_filter = {}
        self._pair_cache = {}
        self._take_path = ""
        self.cap_overhead = CAPACITY_OVERHEAD  # the Capacities section's: each capacity's room over its peak
        self.cap_note = None  # (configuration key, what its last Tune capacities did), shown under the button
        # gizmos and selection
        self.gizmo_hot = False
        self.gizmo_active = False
        self.selected = None  # body
        self.gizmo_local = True
        self.gizmo_all = False
        self.ang_world = False
        self.joints = []
        # body velocities and the contact overlays per pipeline, all off as every scene starts
        self.show_body_v = False
        self.show_body_w = False
        self.show_con = {(src, key): False for src, _ in PIPELINES for key, _ in CONTACT_OVERLAYS}
        self.checker = 0.0  # the ground's square (m); 0 draws it plain
        self._traces = None
        self._rescaled_at = None
        self._implot = False
        self.show_trace = [True, True, True, True]
        self.perf = Performance(self)
        self.viewer.register_ui_callback(self._ui, "panel")
        self.viewer.register_ui_callback(self._ui_free, "free")
        self.session.on_build = self._on_build

    # ---------------------------------------------------------------------------------
    # Scene

    def load(self, source, sim=None):
        """Start a session on `source` (a `SceneSource`), under the splash."""
        sim = sim or self.sim0
        self._defer(lambda: self._load_now(source, sim), f"loading {source.name} under {SIM_NAMES[sim]}")

    def load_builtin(self, name, seed=0, sim=None):
        sim = sim or self.sim0
        self._defer(lambda: self._load_now(LS.builtin(name, seed), sim), f"building {name} under {SIM_NAMES[sim]}")

    def load_file(self, path, sim=None):
        sim = sim or self.sim0
        self._defer(lambda: self._load_now(LS.from_mjcf(path), sim), f"loading {path}")

    def _load_now(self, source, sim):
        self.session.load(source, sim)
        self._attach_scene()
        self.live_status = f"loaded {source.name}"
        self.perf.on_load()

    def _attach_scene(self):
        """(Re)initialize the drawing for the session's scene, hiding the previous scene's objects."""
        old = getattr(self, "_ns", None)
        if old is not None:
            self._hide_namespace(old)
        ses = self.session
        V.TrajectoryViewer.__init__(self, TrajectoryView(ses.take), ses.scene, viewer=self.viewer,
                                    auto_scale=True, video_path="take.mp4", frame_camera=True)
        sc = ses.scene
        self.geom_show = [bool(x) for x in sc.geom_show0]
        self.geom_transparency = [float(1.0 - a) for a in sc.geom_alpha]
        # A scene starts without overlays, as the base class starts its frames and joints;
        # a sim switch, which keeps the scene, keeps what was switched on.
        self.show_body_v = self.show_body_w = False
        self.show_con = {k: False for k in self.show_con}
        self.checker = float(sc.checker)
        s = sc.scale
        self.thickness.update(body_v=0.02 * s, body_w=0.02 * s)
        self.joints = J.joints(ses.source.mjm)
        self.selected = None
        self._traces = None
        self._rescaled_at = (ses.take.id, len(ses.take.frames))

    def _hide_namespace(self, ns):
        """Hide every object a previous scene logged: Newton keeps whatever was last logged under a name."""
        for attr in ("objects", "wireframe_shapes", "lines", "arrows", "points"):
            d = getattr(self.viewer, attr, None)
            if not isinstance(d, dict):
                continue
            for name, obj in d.items():
                if isinstance(name, str) and name.startswith(ns) and hasattr(obj, "hidden"):
                    obj.hidden = True

    def _defer(self, fn, message):
        self.pending = (fn, message)
        self._pending_shown = 0

    def _on_build(self, message):
        self.live_status = message

    def _run_pending(self):
        fn, message = self.pending
        if self._pending_shown < 2:
            # Draw the splash before blocking, so the window says what it is doing.
            self.viewer.show_loading_splash(message)
            self._pending_shown += 1
            return
        self.pending = None
        try:
            fn()
            if self.live_status.startswith("building"):
                self.live_status = self.session.status  # what was built, and how long it took
        except Exception as exc:  # surfaced in the panel, not the console
            import traceback

            traceback.print_exc()
            self.live_status = f"failed: {exc}"
        finally:
            self.viewer.hide_loading_splash()

    def _maybe_build(self, cfg, message, fn):
        """Run `fn` now if the runner for `cfg` exists, else under the splash."""
        if runner_key(cfg) in self.session.runners:
            fn()
        else:
            self._defer(fn, message)

    def request_sim(self, sim):
        """
        What the sim menu does: switch to `sim` at the frame being looked at (a draft),
        building it under the splash the first time.

        A method taking the name: the switch may run a frame later, behind the splash, and a
        closure over the panel's local would pick up whatever the panel's next loop left in it.
        """
        ses = self.session
        self._maybe_build(default_config(ses.source, sim),
                          f"building {SIM_NAMES[sim]} (its first use compiles its kernels)",
                          lambda: (ses.set_sim(sim), self.perf.on_load()))

    def request_option(self, option, value):
        """What an option's field does; an option that allocates may build a runner."""
        value = normalize_option(option, value)
        ses = self.session
        cfg = ses.config
        self._maybe_build(edit_config(ses.source, cfg, option, value),
                          f"building {SIM_NAMES[cfg['sim']]} at {option} = {value}",
                          lambda: ses.set_option(option, value))

    def request_pairs(self, kind, indices, field, value):
        """What a Collision pairs field does: a pair made one- or two-sided builds a runner."""
        ses = self.session
        try:
            new = edit_pairs(ses.source, ses.config, kind, indices, field, value)
        except ValueError as exc:
            self.live_status = f"refused: {exc}"
            return
        what = "restored" if value is None else ("two-sided" if value == 2 else "one-sided")
        self._maybe_build(new, f"building Lodos with {'every' if indices is None else len(indices)} pairs {what}",
                          lambda: ses.set_pairs(kind, indices, field, value))

    def request_all_pairs(self, field, value):
        """What an All pairs field does: every pair of every kind; made one- or two-sided, a rebuild."""
        ses = self.session
        try:
            new = edit_all_pairs(ses.source, ses.config, list(ses.pair_tables()), field, value)
        except ValueError as exc:
            self.live_status = f"refused: {exc}"
            return
        what = "restored" if value is None else ("two-sided" if value == 2 else "one-sided")
        self._maybe_build(new, f"building Lodos with every pair {what}", lambda: ses.set_all_pairs(field, value))

    def request_load_take(self, path):
        """Add the take saved at `path`, loading its scene first when another is loaded."""
        try:
            meta = take_meta(path)
        except Exception as exc:  # noqa: BLE001
            self.live_status = f"load failed: {exc}"
            return
        ses = self.session

        def load():
            ses.load_take(path)
            self.live_status = f"loaded {Path(path).name}: {len(ses.take.frames)} frames"

        if ses.loaded and ses.source.name == meta["scene"]:
            self._defer(load, f"loading {Path(path).name}")
            return
        if meta.get("path"):
            source = lambda: LS.from_mjcf(meta["path"])  # noqa: E731
        elif meta["scene"] in LS.BUILTIN:
            source = lambda: LS.builtin(meta["scene"])  # noqa: E731
        else:
            self.live_status = f"load failed: {meta['scene']} is neither a built-in scene nor a file"
            return
        self._defer(lambda: (self._load_now(source(), self.sim0), load()),
                    f"loading {meta['scene']} for {Path(path).name}")

    # ---------------------------------------------------------------------------------
    # Drawing

    def _log_meshes(self):
        # Two-sided: MJCF meshes come in either winding, and a plane is seen from below.
        for k, (v, f) in enumerate(self.scene.meshes):
            self.viewer.log_mesh(
                f"{self._ns}/mesh/{k}", self._vec3(v),
                wp.array(np.ascontiguousarray(f, dtype=np.int32).reshape(-1), dtype=wp.int32),
                hidden=True, backface_culling=False,
            )
            self._wire_data[k] = V.wireframe_edges(v, f, V.COLOR_WIRE)
        av, af = V.arrow_mesh()
        self.viewer.log_mesh(self.arrow_mesh, self._vec3(av),
                             wp.array(np.ascontiguousarray(af, dtype=np.int32).reshape(-1), dtype=wp.int32),
                             hidden=True)
        self._meshes_logged = True

    def _appearance(self, which):
        """`TrajectoryViewer._appearance`, with an opaque plane checkered: its material's `z` is the square (m)."""
        colors, materials = super()._appearance(which)
        if materials is not None and self.checker > 0.0:
            plane = (self.scene.geom_kind[which] == KIND_PLANE) & (materials[:, 2] >= 0.0)
            materials[plane, 2] = self.checker
        return colors, materials

    def _log_geoms(self, frame):
        """
        `TrajectoryViewer._log_geoms`, at the cost of the geometries that are drawn: one
        instance batch per mesh from precomputed members, and a wireframe call only for a
        geometry drawn as edges or just switched off -- and nothing at all when neither the
        frame nor the display settings changed, since the viewer keeps what was last
        logged. The base class walks every geometry for every mesh and logs every
        wireframe every frame: a displayed frame of twenty armadillos as 34-piece
        polytopes took 18 ms with it, 10 with this while playing and 5 while paused.
        """
        sc = self.scene
        if getattr(self, "_members_of", None) is not sc:
            self._members = [np.flatnonzero(sc.geom_mesh == k) for k in range(len(sc.meshes))]
            self._members_of = sc
            self._wire_prev = None
            self._geoms_key = None
        show = np.asarray(self.geom_show, bool)
        key = (id(frame), show.tobytes(), np.asarray(self.geom_wire, bool).tobytes(),
               np.asarray(self.geom_transparency, np.float32).tobytes(), float(self.transparency), self.selected,
               float(self.checker))
        if key == self._geoms_key:
            return
        self._geoms_key = key
        for k, members in enumerate(self._members):
            which = members[show[members]]
            name, mesh = f"{self._ns}/geoms/{k}", f"{self._ns}/mesh/{k}"
            if not len(which):
                self.viewer.log_instances(name, mesh, None, None, None, None, hidden=True)
                continue
            colors, materials = self._appearance(which)
            self.viewer.log_instances(
                name, mesh,
                wp.array(np.ascontiguousarray(frame.geom_xform[which], dtype=np.float32), dtype=wp.transformf),
                self._vec3(np.ones((len(which), 3))),
                self._vec3(colors),
                None if materials is None else wp.array(np.ascontiguousarray(materials), dtype=wp.vec4),
            )
        # The selected body is outlined: its drawn geometries get their edges too.
        wire = np.asarray(self.geom_wire, bool) | ((sc.geom_body == self.selected) & show)
        prev = self._wire_prev if self._wire_prev is not None else np.ones_like(wire)
        for g in np.flatnonzero(wire | (prev & ~wire)):
            data = self._wire_data.get(int(sc.geom_mesh[g])) if wire[g] else None
            self.viewer.log_wireframe_shape(f"{self._ns}/wire/{g}", data, V.mat44(frame.geom_xform[g]),
                                            hidden=not wire[g])
        self._wire_prev = wire

    def _log_contacts(self, frame):
        """The contact overlays, per pipeline, each with its own toggles and colours."""
        fn, ft = frame.decompose("con_f")
        vectors = {
            "normals": (frame.con_n, self.normal_length),
            "velocity": (frame.con_v, self.velocity_scale),
            "force": (frame.con_f, self.force_scale),
            "force_n": (fn, self.force_scale),
            "force_t": (ft, self.force_scale),
        }
        for src, pname in PIPELINES:
            m = frame.con_src == src if frame.con_src is not None else np.zeros(frame.ncon, bool)
            p = frame.con_p[m]
            col = COLORS[src]
            self.viewer.log_points(
                f"{self._ns}/contacts/points_{pname}", self._vec3(p if len(p) else np.zeros((0, 3))),
                radii=self.point_radius, colors=self._vec3(np.tile(col["points"], (len(p), 1))),
                hidden=not self.show_con[(src, "points")] or not len(p),
            )
            for key, _ in CONTACT_OVERLAYS[1:]:
                vec, scale = vectors[key]
                self._log_arrows(f"{key}_{pname}", p, p + scale * vec[m], col[key], self.thickness[key],
                                 not self.show_con[(src, key)])

    def _log_bodies(self, frame):
        """Each body's velocity, linear and angular, as arrows from its origin."""
        moving = self._body_moving
        o = frame.body_xform[moving, :3]
        self._log_arrows("body_v", o, o + self.body_v_scale * frame.body_v[moving], COLOR_BODY_V,
                         self.thickness["body_v"], not self.show_body_v)
        self._log_arrows("body_w", o, o + self.body_w_scale * frame.body_w[moving], COLOR_BODY_W,
                         self.thickness["body_w"], not self.show_body_w)
        sel = [] if self.selected is None else [self.selected]
        self.viewer.log_points(f"{self._ns}/select", self._vec3(frame.body_xform[sel, :3] if sel else np.zeros((0, 3))),
                               radii=1.5 * self.point_radius, colors=self._vec3(np.tile(COLOR_SELECT, (len(sel), 1))),
                               hidden=not sel)

    def _has_dofs(self, b):
        mjm = self.session.source.mjm
        while b > 0:
            if int(mjm.body_dofnum[b]):
                return True
            b = int(mjm.body_parentid[b])
        return False

    @property
    def _body_moving(self):
        """`(nbody,)` whether each body has a dof, its own or an ancestor's."""
        mjm = self.session.source.mjm
        key = id(mjm)
        if getattr(self, "_moving_key", None) != key:
            self._moving = np.array([self._has_dofs(b) for b in range(int(mjm.nbody))])
            self._moving_key = key
        return self._moving

    def rescale(self):
        """
        The contact arrows as `TrajectoryViewer.rescale` sets them, and the bodies'
        velocity arrows the same way: a `scale_percentile` speed over the take is half
        a body long. Floors of 1 m/s and 2 rad/s keep a scene at rest -- a fresh draft --
        from blowing a nudge up to a body's length.
        """
        super().rescale()
        frames = list(self.traj)
        moving = self._body_moving
        pv = pw = 0.0
        if frames and moving.any():
            v = np.concatenate([np.linalg.norm(f.body_v[moving], axis=1) for f in frames])
            w = np.concatenate([np.linalg.norm(f.body_w[moving], axis=1) for f in frames])
            pv = float(np.percentile(v, self.scale_percentile))
            pw = float(np.percentile(w, self.scale_percentile))
        self.body_v_scale = 0.5 * self.scene.scale / max(pv, 1.0)
        self.body_w_scale = 0.5 * self.scene.scale / max(pw, 2.0)

    def log_frame(self, frame):
        super().log_frame(frame)
        self._log_bodies(frame)

    def draw(self, t=None):
        """One frame of the current take -- what the video render calls, too."""
        take = self.session.take
        t = self.session.frame_index if t is None else t
        frame = take.frames[int(np.clip(t, 0, len(take.frames) - 1))]
        self.viewer.begin_frame(frame.time)
        self.log_frame(frame)
        self.viewer.end_frame()

    def frame_camera(self, t=None, azimuth=45.0, elevation=-20.0, distance=None):
        """
        `TrajectoryViewer.frame_camera` over the bodies' geometries and the ground under
        them: a live scene starts as a stack in the air and ends on the floor, and a
        framing of the first frame alone loses the pile the moment it lands.
        """
        if self.scene is None or not len(self.traj):
            return
        frames = self.traj if t is None else [self.traj[int(t)]]
        # What moves: walls and other scenery would hold the framing on the box rather
        # than on the pile inside it.
        solid = (self.scene.geom_kind != KIND_PLANE) & self._body_moving[self.scene.geom_body]
        if not solid.any():
            solid = self.scene.geom_kind != KIND_PLANE
        pts = np.concatenate([f.geom_xform[solid, :3] for f in frames]).astype(np.float64)
        if not len(pts):
            return
        if (self.scene.geom_kind == KIND_PLANE).any():
            ground = pts.copy()
            ground[:, 2] = 0.0
            pts = np.concatenate([pts, ground])
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        centre = 0.5 * (lo + hi)
        extent = float(np.linalg.norm(hi - lo)) + self.scene.scale
        # The sidebar covers the left of the window and the timeline its bottom: aim at
        # the middle of what is left, from far enough back to fit it there.
        fs, ft = self._covered()
        shrink = max(min(1.0 - fs, 1.0 - ft), 0.3)
        distance = 1.1 * extent / shrink if distance is None else float(distance)
        self.framed_extent = extent
        self.set_camera_speed(self.camera_speed_factor * extent)
        a, e = np.radians(azimuth), np.radians(elevation)
        offset = distance * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), -np.sin(e)])
        d = -offset / max(np.linalg.norm(offset), 1e-9)
        right = np.cross(d, [0.0, 0.0, 1.0])
        right /= max(np.linalg.norm(right), 1e-9)
        up = np.cross(right, d)
        cam = self.viewer.camera
        hh = distance * np.tan(np.radians(cam.fov) / 2)
        hw = hh * cam.width / max(cam.height, 1)
        pos = centre + offset - right * (fs * hw) - up * (ft * hh)
        pitch = float(np.degrees(np.arcsin(np.clip(d[2], -1.0, 1.0))))
        yaw = float(np.degrees(np.arctan2(d[1], d[0])))
        self.viewer.set_camera(wp.vec3(*pos.astype(float)), pitch, yaw)

    def _covered(self):
        """The fractions of the window's width and height the sidebar and the timeline cover."""
        gui = self.viewer.gui
        if gui is None:
            return 0.0, 0.0
        cam = self.viewer.camera
        s = gui.ui.dpi_scale
        W, H = max(cam.width, 1), max(cam.height, 1)
        return min(0.5, (SIDEBAR + 10) * s / W), min(0.5, (TIMELINE_HEIGHT + 20) * s / H)

    def frame_camera_here(self):
        if self.session.loaded:
            self.frame_camera(self.session.frame_index)

    def step_frame(self, d):
        ses = self.session
        if not ses.loaded:
            return
        ses.pause()
        if d > 0:
            ses.step_once()
        else:
            ses.seek(ses.frame_index - 1)

    # ---------------------------------------------------------------------------------
    # Picking

    def pick(self, start, d):
        """
        Select the body under the ray, or none for empty space. Only bodies that can move
        are candidates: the walls of the drop scenes are geometry too, transparent and in
        front of the pile, and would otherwise be the first thing a click hits.
        """
        ses = self.session
        if not ses.loaded or ses.frame is None:
            return
        fr, sc = ses.frame, self.scene
        moving = self._body_moving
        best, body = np.inf, None
        for g in range(len(sc.geom_mesh)):
            if not self.geom_show[g] or sc.geom_kind[g] == KIND_PLANE or not moving[sc.geom_body[g]]:
                continue
            x = fr.geom_xform[g].astype(np.float64)
            R = J.Rotation.from_quat(x[3:]).as_matrix()
            o = R.T @ (start - x[:3])
            u = R.T @ d
            v, f = sc.meshes[sc.geom_mesh[g]]
            t = _ray_mesh(o, u, np.asarray(v, np.float64), np.asarray(f, np.int64))
            if t < best:
                best, body = t, int(sc.geom_body[g])
        self.selected = body
        self.live_status = (f"selected {sc.body_name[body]}: its joints are in the State panel" if body is not None
                            else "nothing to select under the cursor")

    # ---------------------------------------------------------------------------------
    # The panel

    def _ui(self, imgui):
        self._guarded(imgui, "Scene", self._ui_scene)
        if not self.session.loaded or self.scene is None:
            return
        for name, section in (("Sim", self._ui_sim), ("Capacities", self._ui_capacities),
                              ("Collision pairs", self._ui_pairs), ("State", self._ui_state),
                              ("Recording", self._ui_recording), ("Camera", self._camera_ui),
                              ("Overlays", self._ui_overlays), ("Geometries", self._ui_geometries),
                              ("Scales", self._ui_scales), ("Thickness", self._ui_thickness),
                              ("Render", self._ui_render), ("Performance", self.perf.ui)):
            self._guarded(imgui, name, section)

    def _guarded(self, imgui, name, section):
        """
        Draw one section of the panel. An error in it is reported in the panel and on the
        console, once, and ImGui's stacks are put back as they were before the section --
        rather than the exception unwinding through the window's frame and closing it, with
        every take in it.
        """
        state = imgui.internal.ErrorRecoveryState()
        imgui.internal.error_recovery_store_state(state)
        try:
            section(imgui)
        except Exception as exc:  # noqa: BLE001 -- the window stays open
            imgui.internal.error_recovery_try_to_recover_state(state)
            key = (name, type(exc).__name__, str(exc))
            if key != getattr(self, "_last_ui_error", None):
                import traceback

                traceback.print_exc()
                self._last_ui_error = key
            self.live_status = f"{name}: {type(exc).__name__}: {exc}"

    def _ui_scene(self, imgui):
        imgui.set_next_item_open(True, imgui.Cond_.appearing)
        if not imgui.collapsing_header("Scene"):
            return
        if self.session.loaded:
            src = self.session.source
            imgui.text_wrapped(f"{src.name}: {src.mjm.nbody - 1} bodies, {src.mjm.ngeom} geoms, "
                               f"{0 if src.surfaces is None else len(src.surfaces.surfaces)} Lodos surfaces")
        names = list(LS.PANEL)
        i = names.index(self.builtin) if self.builtin in names else 0
        imgui.set_next_item_width(140 * self._dpi())
        _, i = imgui.combo("##builtin", i, names)
        self.builtin = names[i]
        imgui.same_line()
        imgui.set_next_item_width(60 * self._dpi())
        _, self.seed = imgui.input_int("seed", self.seed, 0)
        imgui.same_line()
        if imgui.button("Load##builtin"):
            self.load_builtin(self.builtin, self.seed)
        imgui.text_disabled(LS.BUILTIN[self.builtin][0])

        imgui.separator()
        idx = self.index
        changed, self.query = imgui.input_text_with_hint("##mjcf", "search MJCF files...", self.query)
        key = (self.query, len(idx.paths), idx.done)
        if key != self._match_key:
            self.matches = LS.search(idx.paths, self.query, limit=300)
            self._match_key = key
        imgui.same_line()
        if imgui.button("Browse..."):
            self.viewer.gui.ui.open_load_file_dialog("Open MJCF")
        path = self.viewer.gui.ui.consume_file_dialog_result() if self.viewer.gui is not None else None
        if path:
            self.load_file(path)
        n = len(self.matches)
        state = "" if idx.done else " (indexing...)"
        imgui.text_disabled(f"{n} of {len(idx.paths)} MJCF files{state}")
        if imgui.begin_list_box("##matches", imgui.ImVec2(-1, 150)):
            for p in self.matches:
                shown = self._short(p)
                clicked, _ = imgui.selectable(shown, p == self.pick_file)
                if imgui.is_item_hovered() and imgui.is_mouse_double_clicked(0):
                    self.pick_file = p
                    self.load_file(p)
                elif clicked:
                    self.pick_file = p
                if imgui.is_item_hovered():
                    imgui.set_tooltip(p)
            imgui.end_list_box()
        imgui.begin_disabled(self.pick_file is None)
        if imgui.button("Load file"):
            self.load_file(self.pick_file)
        imgui.end_disabled()
        imgui.same_line()
        if imgui.button("Rescan"):
            idx.scan()
        imgui.same_line()
        j = SIMS.index(self.sim0)
        imgui.set_next_item_width(100 * self._dpi())
        _, j = imgui.combo("start sim", j, [SIM_NAMES[s] for s in SIMS])
        self.sim0 = SIMS[j]
        if self.session.loaded and self.session.source.xml is not None:
            if imgui.button("Export as MJCF"):
                fr = self.session.frame
                out = Path.cwd() / f"{self.session.source.name}_live.xml"
                try:
                    LS.export_mjcf(self.session.source, out, fr.qpos, fr.qvel, fr.time)
                    self.live_status = f"wrote {out}"
                except Exception as exc:  # noqa: BLE001
                    self.live_status = f"export failed: {exc}"
        if self.live_status:
            imgui.text_wrapped(self.live_status)

    def _dpi(self):
        gui = self.viewer.gui
        return gui.ui.dpi_scale if gui is not None else 1.0

    @staticmethod
    def _short(p):
        home = str(Path.home())
        return "~" + p[len(home):] if p.startswith(home) else p

    def _ui_sim(self, imgui):
        ses = self.session
        imgui.set_next_item_open(True, imgui.Cond_.appearing)
        if not imgui.collapsing_header("Sim"):
            return
        cfg = ses.config
        cur = cfg["sim"]
        i = SIMS.index(cur)
        changed, i = imgui.combo("sim", i, [SIM_NAMES[s] for s in SIMS])
        if changed and SIMS[i] != cur:
            self.request_sim(SIMS[i])
        imgui.text_disabled(SIM_INFO[cur])
        for opt, kind, lo, hi in OPTIONS[cur]:
            v = cfg["options"].get(opt)
            if v is None:
                continue
            if kind == "choice":
                imgui.set_next_item_width(120 * self._dpi())
                choices = list(lo)
                ch, ni = imgui.combo(opt, choices.index(v) if v in choices else 0, [str(c) for c in choices])
                if ch and choices[ni] != v:
                    self.request_option(opt, choices[ni])
            elif kind is int:
                imgui.set_next_item_width(120 * self._dpi())
                ch, nv = imgui.input_int(opt, int(v), 1)
                if imgui.is_item_deactivated_after_edit() or (ch and not imgui.is_item_active()):
                    self.request_option(opt, int(np.clip(nv, lo, hi)))
            else:
                imgui.set_next_item_width(120 * self._dpi())
                ch, nv = imgui.input_float(opt, float(v), 0.0, 0.0, "%.1e")
                if imgui.is_item_deactivated_after_edit():
                    self.request_option(opt, float(np.clip(nv, lo, hi)))
            self._speed_arrow(imgui, speed_arrow(cur, opt))
        self._ui_cost(imgui, cfg)
        if imgui.button("Export sim config"):
            stem = re.sub(r"[^\w.-]+", "_", ses.source.name).strip("_")  # a scene's name may have ", start 3"
            out = Path.cwd() / f"{stem}_{cur}_simconfig.json"
            try:
                ses.export_config(out)
                self.live_status = f"wrote {out}"
            except Exception as exc:  # noqa: BLE001
                self.live_status = f"export failed: {exc}"
        if imgui.is_item_hovered():
            imgui.set_tooltip("everything the panel sets on this configuration, to a JSON file in the working "
                              "folder: the sim's options, its capacities and every pair's contact parameters")

    def _speed_arrow(self, imgui, arrow):
        """The speed arrow after a field: the way that makes the sim faster (`speed_arrow`); none for ""."""
        if not arrow:
            return
        imgui.same_line()
        imgui.text_disabled(arrow)
        if imgui.is_item_hovered():
            imgui.set_tooltip(arrow_tip(arrow))

    def _ui_capacities(self, imgui):
        ses = self.session
        if not imgui.collapsing_header("Capacities"):
            return
        cfg = ses.config
        r = ses.runner(cfg)
        if imgui.begin_table("capacities", 3, imgui.TableFlags_.sizing_fixed_fit):
            for h in ("capacity", "allocated", "peak seen"):
                imgui.table_setup_column(h)
            imgui.table_headers_row()
            for name, alloc, peak in r.capacities():
                imgui.table_next_row()
                imgui.table_next_column()
                imgui.text(name)
                imgui.table_next_column()
                imgui.text(f"{alloc:,}")
                imgui.table_next_column()
                imgui.text(f"{peak:,}")
            imgui.end_table()
        fr = r.fractions()
        if fr:
            imgui.text_disabled("Lodos's arenas, fractions of a world's worst case: "
                                + ", ".join(f"{k} {v:.3g}" for k, v in fr.items()))
        imgui.text_disabled(f"the peaks over the {r.steps_seen:,} steps simulated under this configuration")
        imgui.set_next_item_width(120 * self._dpi())
        _, self.cap_overhead = imgui.input_float("capacity overhead", float(self.cap_overhead), 0.0, 0.0, "%.2f")
        if imgui.is_item_hovered():
            imgui.set_tooltip("each capacity's room over its peak: 0.5 sizes it 1.5x the peak")
        if imgui.button("Tune capacities"):
            self.request_tune_capacities(self.cap_overhead)
        if imgui.is_item_hovered():
            imgui.set_tooltip("every capacity at its peak seen times 1 + the overhead, then rebuilt from this state; "
                              "a step that outgrows one grows it again")
        if self.cap_note is not None and self.cap_note[0] == runner_key(cfg):
            imgui.text_wrapped(self.cap_note[1])

    def request_tune_capacities(self, overhead):
        """What Tune capacities does: every capacity at its peak times 1 + `overhead`, rebuilt under the splash."""
        ses = self.session

        def tune():
            key = runner_key(ses.config)
            try:
                ses.tune_capacities(overhead)
            except Exception as exc:  # noqa: BLE001 -- said under the button too, where it was pressed
                self.cap_note = (key, f"not tuned: {exc}")
                raise
            self.live_status = ses.status
            self.cap_note = (key, ses.status)

        self._defer(tune, "tuning the capacities")

    # The most pairs the Collision pairs folder lists at once: a filter narrows the rest.
    PAIR_ROWS = 300

    def _ui_pairs(self, imgui):
        ses = self.session
        if not imgui.collapsing_header("Collision pairs"):
            return
        cfg = ses.config
        tables = ses.pair_tables()
        if not tables:
            imgui.text_disabled("the scene collides nothing")
            return
        # friction alone: Lodos's exact mode, and Kamino
        kamino = cfg["sim"] == "kamino"
        exact = kamino or (cfg["sim"] == "lodos" and cfg["options"].get("mode") != "mj")
        if kamino:
            imgui.text_wrapped(f"Kamino takes friction alone, a contact's the {tables['newton'].mix} of its two "
                               "shapes': no solref or solimp reaches it.")
        elif exact:
            imgui.text_wrapped("The exact mode takes friction and sidedness alone: no solref, solimp or "
                               "impratio reaches it.")
        else:
            imgui.text_wrapped(f"A time constant below twice the step ({2e3 * ses.dt:.3g} ms) is raised to it "
                               "(MuJoCo's refsafe).")
        imgui.text_wrapped("A field sets every pair it covers: all of them, or those its filter matches. A "
                           "right-click on it restores the scene's values; * marks a value set on a pair.")
        info = {k: self._pair_info(k, t, cfg, exact) for k, t in tables.items()}
        imgui.separator()
        imgui.text(f"All pairs ({sum(len(t) for t in tables.values())})")
        imgui.push_id("all")
        for f in VP.FIELDS:
            parts = [info[k]["summary"][f] for k in tables]
            first = parts[0][0]
            mixed = any(m or not np.array_equal(v, first) for v, m, _ in parts)
            self._ui_pair_field(imgui, f, first, mixed, sum(n for *_, n in parts),
                                lambda x, f=f: self.request_all_pairs(f, x), disabled=exact and f != "mu")
        if "lodos" in tables and (~tables["lodos"].plane).any():
            free = ~tables["lodos"].plane
            self._ui_sided(imgui, info["lodos"]["vals"]["sided"][free], info["lodos"]["marks"]["sided"].any(),
                           lambda x: self.request_all_pairs("sided", x))
        imgui.pop_id()
        for kind, title in (("lodos", "Lodos pairs"), ("prim", "MJCF pairs"), ("newton", "Newton's pairs")):
            if kind in tables:
                imgui.separator()
                imgui.text(f"{title} ({len(tables[kind])})")
                imgui.push_id(kind)
                self._ui_pair_kind(imgui, kind, tables[kind], cfg, exact, info[kind])
                imgui.pop_id()

    def _pair_info(self, kind, table, cfg, exact):
        """
        The pairs' values as applied (`vals`), which are set (`marks`) and each field's `summary` over them all
        (`_summary`), for the configuration on screen; made again when it changes.
        """
        hit = self._pair_cache.get(kind)
        if hit is None or hit[0] is not cfg or hit[1] is not table:
            vals, marks = table.values(cfg["pairs"], exact), table.marks(cfg["pairs"])
            hit = (cfg, table, dict(vals=vals, marks=marks,
                                    summary={f: _summary(vals[f], marks[f]) for f in VP.FIELDS}))
            self._pair_cache[kind] = hit
        return hit[2]

    def _ui_pair_kind(self, imgui, kind, t, cfg, exact, info):
        filters = self.pair_filter
        imgui.set_next_item_width(-1)
        _, text = imgui.input_text_with_hint("##filter", "filter: words in the pair's names; a / b: in its first, "
                                             "its second", filters.get(kind, ""))
        filters[kind] = text
        idx = t.match(text)
        every = not text.split()  # every pair of the kind: one value for them all
        vals, marks = info["vals"], info["marks"]
        imgui.text_disabled(f"{len(idx)} of {len(t)} pairs match")
        if imgui.begin_list_box("##pairs", imgui.ImVec2(-1, 120 * self._dpi())):
            for i in idx[:self.PAIR_ROWS]:
                clicked, _ = imgui.selectable(t.row(i, vals, marks), False)
                if clicked:
                    filters[kind] = t.label(i)  # narrow to this pair
            if len(idx) > self.PAIR_ROWS:
                imgui.text_disabled(f"... {len(idx) - self.PAIR_ROWS} more: narrow the filter")
            imgui.end_list_box()
        if not len(idx):
            return
        pairs = cfg["pairs"]
        if kind in ("prim", "newton"):
            m = VP.GEOM_MODES.index(pairs["mode"])
            imgui.set_next_item_width(90 * self._dpi())
            ch, m = imgui.combo("a geom's value from its pairs'", m, list(VP.GEOM_MODES))
            if ch:
                self.session.set_geom_mode(VP.GEOM_MODES[m])
            if imgui.is_item_hovered():
                imgui.set_tooltip("MuJoCo mixes a contact's parameters from its two geoms: each geom takes the min, "
                                  "max or mean of what is set on the pairs it is in, so the list shows what MJWarp "
                                  "then applies")
        sel = None if every else idx
        for f in VP.FIELDS:
            first, mixed, n = info["summary"][f] if every else _summary(vals[f][idx], marks[f][idx])
            self._ui_pair_field(imgui, f, first, mixed, n, lambda x, f=f: self.request_pairs(kind, sel, f, x),
                                disabled=exact and f != "mu", under_all=not every and f in pairs[kind]["all"])
        if kind == "lodos":
            free = idx[~t.plane[idx]]  # a pair with a plane is one-sided
            if len(free):
                self._ui_sided(imgui, vals["sided"][free], marks["sided"][free].any(),
                               lambda x: self.request_pairs(kind, None if every else free, "sided", x))
            if len(free) < len(idx):
                imgui.text_disabled(f"{len(idx) - len(free)} of them have a plane: one-sided")

    def _ui_pair_field(self, imgui, field, first, mixed, n_set, request, disabled, under_all=False):
        """
        One parameter of a set of pairs: their common value, else the first's, marked mixed; `request(value)` sets
        it on them all, `request(None)` restores theirs -- the value for all the kind's pairs when `under_all`.
        """
        name = field + (" (mixed)" if mixed else "") + (" *" if n_set else "") + f"###{field}"  # the id: the field
        imgui.begin_disabled(disabled)
        imgui.set_next_item_width(220 * self._dpi())
        if field == "mu":
            ch, nv = imgui.input_float(name, float(first), 0.0, 0.0, "%.4g")
            if imgui.is_item_deactivated_after_edit():
                request(float(nv))
        elif field == "solref":
            ch, nv = imgui.input_float2(name, [float(x) for x in first], "%.4g")
            if imgui.is_item_deactivated_after_edit():
                request(tuple(nv))
        else:
            txt = " ".join(f"{x:.6g}" for x in first)
            ch, nv = imgui.input_text(name, txt)
            if imgui.is_item_deactivated_after_edit() and nv != txt:
                try:
                    request(tuple(float(x) for x in nv.replace(",", " ").split()))
                except ValueError as exc:
                    self.live_status = f"solimp: {exc}"
        if n_set and imgui.is_item_clicked(1):
            request(None)
        if imgui.is_item_hovered():
            tip = {"mu": "friction", "solref": "time constant (s), damping ratio",
                   "solimp": "dmin dmax width mid power"}[field]
            back = "back to the value for all of them" if under_all else "restores the scene's"
            imgui.set_tooltip(tip + (f"; set on {n_set} of these pairs, right-click {back}" if n_set else ""))
        imgui.end_disabled()

    def _ui_sided(self, imgui, sided, any_set, request):
        """Whether a set of Lodos pairs is one- or two-sided; `request(1 or 2)` makes them so, `request(None)` the scene's."""
        mixed = len(np.unique(sided)) > 1
        imgui.set_next_item_width(160 * self._dpi())
        ch, s = imgui.combo("sided" + (" (mixed)" if mixed else "") + (" *" if any_set else "") + "###sided",
                            int((sided == 2).all()), ["one-sided", "two-sided"])
        if ch:
            request(s + 1)
        if any_set and imgui.is_item_clicked(1):
            request(None)
        if imgui.is_item_hovered():
            imgui.set_tooltip("one-sided: the first surface's vertices against the second's SDF; two-sided: both ways "
                              "(a rebuild); a pair with a plane is one-sided")
        self._speed_arrow(imgui, "-> one-sided")

    def _ui_state(self, imgui):
        ses = self.session
        imgui.set_next_item_open(True, imgui.Cond_.appearing)
        if not imgui.collapsing_header("State"):
            return
        fr = ses.frame
        if fr is None:
            return
        bodies = sorted({j.body for j in self.joints})
        names = ["(none)"] + [self.scene.body_name[b] for b in bodies]
        i = 0 if self.selected not in bodies else bodies.index(self.selected) + 1
        imgui.set_next_item_width(160 * self._dpi())
        ch, i = imgui.combo("body", i, names)
        if ch:
            self.selected = None if i == 0 else bodies[i - 1]
        imgui.same_line()
        imgui.text_disabled("(or click it)")
        _, self.gizmo_local = imgui.checkbox("local axes", self.gizmo_local)
        imgui.same_line()
        _, self.gizmo_all = imgui.checkbox("gizmos on all bodies", self.gizmo_all)
        _, self.ang_world = imgui.checkbox("angular velocity in world frame", self.ang_world)
        if imgui.button("Zero velocities"):
            ses.edit_state(qvel=np.zeros_like(fr.qvel))
        imgui.same_line()
        if imgui.button("Initial state"):
            ses.reset_state()
        keys = ses.source.keyframes
        if keys:
            imgui.same_line()
            kn = [k[0] for k in keys]
            imgui.set_next_item_width(100 * self._dpi())
            ch, k = imgui.combo("##key", 0, ["keyframe..."] + kn)
            if ch and k > 0:
                ses.reset_state(k - 1)
        rec = ses.source.controls
        if rec is not None:
            ch, on = imgui.checkbox("play controls", ses.controls_playing)
            if ch:
                ses.play_controls(on)
            imgui.same_line()
            if not ses.controls_playing:
                imgui.text_disabled("held: the drives keep their last targets")
            elif isinstance(rec, LS.RecordedControls):
                imgui.text_disabled(f"tick {rec.tick(fr.time)} of {len(rec.ctrl)}")
        if imgui.button("Save state"):
            out = Path.cwd() / f"{ses.source.name}_state_{fr.time:.3f}.npz"
            np.savez(out, qpos=fr.qpos, qvel=fr.qvel, time=fr.time)
            self.live_status = f"wrote {out}"
        imgui.same_line()
        _, self._state_path = imgui.input_text_with_hint("##statepath", "state .npz to load",
                                                         getattr(self, "_state_path", ""))
        imgui.same_line()
        if imgui.button("Load##state"):
            try:
                z = np.load(os.path.expanduser(self._state_path))
                ses.edit_state(qpos=z["qpos"], qvel=z["qvel"])
            except Exception as exc:  # noqa: BLE001
                self.live_status = f"load failed: {exc}"

        if imgui.button("Copy state"):
            # One line of Python a script or the Paste button can take back.
            imgui.set_clipboard_text(f"dict(qpos={fr.qpos.tolist()}, qvel={fr.qvel.tolist()})")
            self.live_status = "state copied to the clipboard"
        imgui.same_line()
        if imgui.button("Paste state"):
            try:
                import ast

                text = imgui.get_clipboard_text() or ""
                tree = ast.parse(text.strip(), mode="eval").body
                kw = {k.arg: ast.literal_eval(k.value) for k in tree.keywords}
                ses.edit_state(qpos=kw.get("qpos"), qvel=kw.get("qvel"))
            except Exception as exc:  # noqa: BLE001
                self.live_status = f"paste failed: {exc}"

        if self.selected is None:
            imgui.text_disabled("select a body to edit its joints")
            return
        for jt in J.body_joints(self.joints, self.selected):
            self._ui_joint(imgui, ses, fr, jt)

    def _ui_joint(self, imgui, ses, fr, jt):
        imgui.push_id(f"jnt{jt.id}")
        imgui.text(f"{jt.name} ({jt.type_name})")
        a, dv = jt.qposadr, jt.dofadr
        q, v = fr.qpos.copy(), fr.qvel.copy()
        newq = newv = None
        if jt.type == J.FREE:
            ch, x = imgui.drag_float3("position", list(q[a:a + 3]), 0.005)
            if ch:
                q[a:a + 3] = x
                newq = q
            ch, w = imgui.input_float4("quaternion", list(q[a + 3:a + 7]), "%.4f")
            if imgui.is_item_deactivated_after_edit():
                q[a + 3:a + 7] = J.normalize(w)
                newq = q
            ch, e = imgui.drag_float3("roll pitch yaw", list(J.quat_to_euler(J.normalize(q[a + 3:a + 7]))), 0.5)
            if ch:
                q[a + 3:a + 7] = J.euler_to_quat(e)
                newq = q
            ch, lin = imgui.drag_float3("linear velocity", list(v[dv:dv + 3]), 0.01)
            if ch:
                v[dv:dv + 3] = lin
                newv = v
            w = self._ui_angular(imgui, fr, jt, v, dv + 3)
            if w is not None:
                newv = w
        elif jt.type == J.BALL:
            ch, w = imgui.input_float4("quaternion", list(q[a:a + 4]), "%.4f")
            if imgui.is_item_deactivated_after_edit():
                q[a:a + 4] = J.normalize(w)
                newq = q
            ch, e = imgui.drag_float3("roll pitch yaw", list(J.quat_to_euler(J.normalize(q[a:a + 4]))), 0.5)
            if ch:
                q[a:a + 4] = J.euler_to_quat(e)
                newq = q
            newv = self._ui_angular(imgui, fr, jt, v, dv)
        else:
            unit = "rad" if jt.type == J.HINGE else "m"
            ch, x = imgui.drag_float(f"position ({unit})", float(q[a]), 0.005)
            if ch:
                q[a] = x
                newq = q
            if jt.type == J.HINGE:
                imgui.same_line()
                imgui.text_disabled(f"{np.degrees(q[a]):.1f} deg")
            ch, x = imgui.drag_float(f"velocity ({unit}/s)", float(v[dv]), 0.01)
            if ch:
                v[dv] = x
                newv = v
        if newq is not None or newv is not None:
            ses.edit_state(qpos=newq, qvel=newv)
        imgui.pop_id()

    def _ui_angular(self, imgui, fr, jt, v, i):
        w = v[i:i + 3]
        shown = J.world_angular(fr, jt, w) if self.ang_world else w
        label = "angular velocity (world)" if self.ang_world else "angular velocity (body)"
        ch, x = imgui.drag_float3(label, list(shown), 0.01)
        if not ch:
            return None
        v[i:i + 3] = J.local_angular(fr, jt, x) if self.ang_world else x
        return v

    def _ui_recording(self, imgui):
        ses = self.session
        if not imgui.collapsing_header("Recording and traces"):
            return
        imgui.set_next_item_width(120 * self._dpi())
        ch, stride = imgui.input_int("record every n-th step", ses.stride, 1)
        if ch:
            ses.stride = max(1, int(stride))
        imgui.text_disabled("applies to takes started from now on")
        imgui.text(f"{len(ses.takes)} takes, {ses.memory() / 1e6:.1f} MB recorded")
        imgui.text("traces in the timeline:")
        for i, name in enumerate(("contacts", "deepest penetration", "kinetic energy", "solver iterations")):
            _, self.show_trace[i] = imgui.checkbox(name, self.show_trace[i])
        imgui.text_disabled(f"pauses at {ses.memory_budget / 1e9:.0f} GB recorded")
        take = ses.take
        if take is not None:
            if imgui.button("Save take"):
                out = Path.cwd() / f"{ses.source.name}_{'draft' if take.id < 0 else f'take{take.id}'}.npz"
                try:
                    save_take(take, ses.source, out)
                    self.live_status = f"wrote {out}: {len(take.frames)} frames"
                except Exception as exc:  # noqa: BLE001
                    self.live_status = f"save failed: {exc}"
            if imgui.is_item_hovered():
                imgui.set_tooltip("the take on screen, whole: every frame, its configuration and where it ends, "
                                  "to load back, scrub and continue")
            imgui.same_line()
        imgui.set_next_item_width(220 * self._dpi())
        _, self._take_path = imgui.input_text_with_hint("##takepath", "take .npz to load", self._take_path)
        imgui.same_line()
        if imgui.button("Load##take"):
            self.request_load_take(os.path.expanduser(self._take_path))
        if ses.status:
            imgui.text_wrapped(ses.status)

    def _ui_overlays(self, imgui):
        imgui.set_next_item_open(True, imgui.Cond_.appearing)
        if not imgui.collapsing_header("Overlays"):
            return
        if imgui.begin_table("overlays", 3, imgui.TableFlags_.sizing_fixed_fit):
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text("contacts")
            for _, pname in PIPELINES:
                imgui.table_next_column()
                imgui.text(pname)
            for key, label in CONTACT_OVERLAYS:
                imgui.table_next_row()
                imgui.table_next_column()
                imgui.text(label)
                for src, pname in PIPELINES:
                    imgui.table_next_column()
                    _, self.show_con[(src, key)] = imgui.checkbox(f"##{key}{src}", self.show_con[(src, key)])
            imgui.end_table()
        _, self.show_body_v = imgui.checkbox("Body linear velocity", self.show_body_v)
        _, self.show_body_w = imgui.checkbox("Body angular velocity", self.show_body_w)
        _, self.show_frames = imgui.checkbox("Body frames", self.show_frames)
        _, self.show_joints = imgui.checkbox("Joint axes", self.show_joints)

    def _ui_geometries(self, imgui):
        if not imgui.collapsing_header("Geometries"):
            return
        sc = self.scene
        self._ui_geometry_bulk(imgui)
        imgui.text("by kind: show / wire")
        for kind, name in enumerate(KIND_NAMES):
            idx = [g for g in range(len(sc.geom_kind)) if sc.geom_kind[g] == kind]
            if not idx:
                continue
            imgui.push_id(f"kind{kind}")
            shown = all(self.geom_show[g] for g in idx)
            ch, shown = imgui.checkbox("##s", shown)
            if ch:
                for g in idx:
                    self.geom_show[g] = shown
            imgui.same_line()
            wired = all(self.geom_wire[g] for g in idx)
            ch, wired = imgui.checkbox("##w", wired)
            if ch:
                for g in idx:
                    self.geom_wire[g] = wired
            imgui.same_line()
            imgui.text(f"{name} ({len(idx)})")
            imgui.pop_id()
        if len(sc.lodos_bodies) and any(sc.geom_kind == KIND_PRIM):
            imgui.text_disabled("MJCF geoms of Lodos bodies start hidden")
        if (sc.geom_kind == KIND_PLANE).any():
            imgui.set_next_item_width(90 * self._dpi())
            ch, v = imgui.input_float("ground squares (m, 0 plain)", self.checker, 0.0, 0.0, "%.3g")
            if ch:
                self.checker = max(0.0, float(v))
        imgui.separator()
        imgui.text("per body: show  wire  transparency")
        for b in sorted(set(sc.geom_body.tolist())):
            idx = [g for g in range(len(sc.geom_body)) if sc.geom_body[g] == b]
            if imgui.tree_node(f"{sc.body_name[b]} ({len(idx)})##body{b}"):
                for g in idx:
                    self._ui_geometry_row(imgui, g, sc.geom_label[g].split(": ", 1)[-1])
                imgui.tree_pop()

    def _ui_scales(self, imgui):
        super()._ui_scales(imgui)
        if imgui.collapsing_header("Body velocity scales"):
            s = self.scene.scale
            _, self.body_v_scale = imgui.slider_float("m per m/s", self.body_v_scale, 0.0, 2.0 * s, "%.3f",
                                                      imgui.SliderFlags_.logarithmic)
            _, self.body_w_scale = imgui.slider_float("m per rad/s", self.body_w_scale, 0.0, 2.0 * s, "%.3f",
                                                      imgui.SliderFlags_.logarithmic)

    # ---------------------------------------------------------------------------------
    # The timeline, and the gizmos: drawn outside the sidebar

    def _ui_free(self, imgui):
        if not self._implot:
            from imgui_bundle import implot

            implot.create_context()
            self._implot = True
        self._guarded(imgui, "Timeline", self._ui_timeline)
        self._guarded(imgui, "Gizmos", self._ui_gizmos)

    def _ui_timeline(self, imgui):
        from imgui_bundle import implot

        io = imgui.get_io()
        s = self.viewer.gui.ui.dpi_scale if self.viewer.gui is not None else 1.0
        h = TIMELINE_HEIGHT * s
        x0 = (SIDEBAR + 10) * s
        imgui.set_next_window_pos(imgui.ImVec2(x0, io.display_size[1] - h - 10 * s), imgui.Cond_.first_use_ever)
        imgui.set_next_window_size(imgui.ImVec2(io.display_size[0] - x0 - 10 * s, h), imgui.Cond_.first_use_ever)
        ses = self.session
        if ses.loaded and ses.take is not None:
            if ses.take.id < 0:
                state = "draft: Play runs it as a new take"
            elif ses.status.startswith("paused"):
                state = ses.status
            elif not ses.playing:
                state = "paused"
            elif ses.frame_index < len(ses.take.frames) - 1:
                state = f"replaying at {ses.rate:.2f}x"
            else:
                state = f"simulating at {ses.rtf:.2f}x real time"
            title = f"Timeline    {state}    {ses.memory() / 1e6:.0f} MB recorded###Timeline"
        else:
            title = "Timeline###Timeline"
        if not imgui.begin(title):
            imgui.end()
            return
        if not ses.loaded or ses.take is None:
            imgui.text_disabled("no scene: load one from the Scene panel")
            imgui.end()
            return

        # Row 1: the transport, then the takes.
        takes = ses.takes
        cur = ses.take
        if imgui.button("|<"):
            ses.pause()
            ses.seek(0)
        imgui.same_line()
        if imgui.button("<"):
            self.step_frame(-1)
        imgui.same_line()
        if imgui.button("Pause" if ses.playing else "Play", imgui.ImVec2(70 * s, 0)):
            ses.toggle()
        imgui.same_line()
        if imgui.button(">"):
            self.step_frame(+1)
        imgui.same_line()
        if imgui.button(">|"):
            ses.pause()
            ses.seek(len(ses.take.frames) - 1)
        imgui.same_line()
        imgui.set_next_item_width(110 * s)
        _, ses.rate = imgui.slider_float("##speed", ses.rate, 0.02, 10.0, "%.2fx", imgui.SliderFlags_.logarithmic)
        imgui.same_line()
        _, ses.max_speed = imgui.checkbox("max", ses.max_speed)
        imgui.same_line()
        imgui.set_next_item_width(max(imgui.get_content_region_avail().x - 70 * s, 80 * s))
        if imgui.begin_combo("##takes", cur.label()):
            for i, t in enumerate(takes):
                clicked, _ = imgui.selectable(f"{t.label()}##take{t.id}", t is cur)
                if clicked:
                    ses.select(i)
            imgui.end_combo()
        if imgui.is_item_hovered():
            imgui.set_tooltip(cur.label())
        imgui.same_line()
        imgui.begin_disabled(cur.id < 0)
        if imgui.button("Delete"):
            ses.delete(takes.index(cur))
        imgui.end_disabled()

        # Row 2: the frame slider, with the time on it.
        fr = ses.frame
        imgui.set_next_item_width(-1)
        ch, k = imgui.slider_int("##frame", ses.frame_index, 0, max(len(cur.frames) - 1, 0),
                                 f"frame %d of {len(cur.frames) - 1}    t = {fr.time:.3f} s")
        if ch:
            ses.pause()
            ses.seek(k)

        # Row 3: the traces, side by side, as many as are switched on.
        tr = self._trace_data(cur)
        plots = (
            ("contacts", (("Lodos contacts", tr["ncon_lodos"]), ("MJCF contacts", tr["ncon_prim"]))),
            ("penetration", (("deepest penetration [m]", tr["pen"]),)),
            ("energy", (("kinetic energy [J]", tr["ke"]),)),
            ("iterations", (("solver iterations", tr["niter"]),)),
        )
        shown = [i for i in range(4) if self.show_trace[i]]
        avail = imgui.get_content_region_avail()
        w = (avail.x - max(len(shown) - 1, 0) * imgui.get_style().item_spacing.x) / max(len(shown), 1)
        ph = max(avail.y, 40 * s)
        t_now = fr.time
        for n, i in enumerate(shown):
            title, series = plots[i]
            if n:
                imgui.same_line()
            flags = implot.Flags_.no_menus | implot.Flags_.no_box_select | implot.Flags_.no_title
            if implot.begin_plot(f"{title}##trace{i}", imgui.ImVec2(w, ph), flags):
                implot.setup_axes("", "", implot.AxisFlags_.auto_fit | implot.AxisFlags_.no_tick_labels,
                                  implot.AxisFlags_.auto_fit)
                implot.setup_legend(implot.Location_.north_west)
                for k2, (name, ys) in enumerate(series):
                    if len(ys):
                        c = TRACE_COLORS[k2 if i == 0 else i + 1]
                        implot.plot_line(name, tr["t"], ys,
                                         implot.Spec(line_color=imgui.ImVec4(*c, 1.0), line_weight=1.5))
                ch, x, _, _, _ = implot.drag_line_x(0, t_now, imgui.ImVec4(1, 1, 1, 0.8), 1.5)
                if ch:
                    self._seek_time(x)
                elif implot.is_plot_hovered() and imgui.is_mouse_clicked(0):
                    self._seek_time(implot.get_plot_mouse_pos().x)
                implot.end_plot()
        imgui.end()

    def _ui_cost(self, imgui, cfg):
        """
        What the solve costs, under any sim: the frame's iterations and whether it converged --
        Kamino's PADMM iterations against its cap, with its residuals (absolute, physical units);
        MuJoCo Warp's Newton iterations against its cap; Lodos's Newton iterations, summed over its
        proximal rounds and the rounds against their budget where the loop runs -- and, over the
        take's recorded frames, the mean and largest iteration count, the mean rounds, and how many
        frames did not converge.
        """
        ses = self.session
        o, sim = cfg["options"], cfg["sim"]
        st = ses.frame.stats if ses.frame is not None else {}
        n = float(st.get("niter", float("nan")))
        conv = st.get("converged", float("nan"))
        rounds = float(st.get("rounds", float("nan")))
        if np.isfinite(n):
            if sim == "kamino":
                how = "converged" if conv else f"stopped at the cap of {int(o['max_iterations'])}"
                imgui.text(f"PADMM, this frame: {int(n)} iterations, {how}")
                imgui.text_disabled(f"residuals: primal {st['r_p']:.2g}, dual {st['r_d']:.2g}, "
                                    f"complementarity {st['r_c']:.2g}")
            elif np.isfinite(rounds):
                budget = max(1, int(o.get("prox_iters_per_qp", 1)))
                how = "the loop converged" if conv else f"the loop spent its {budget} rounds"
                imgui.text(f"this frame: {int(n)} Newton iterations over {int(rounds)} proximal rounds; {how}")
            elif conv == conv:  # a flag, not NaN: a Newton solve against its cap
                how = "converged" if conv else f"stopped at the cap of {int(o['iterations'])}"
                imgui.text(f"this frame: {int(n)} Newton iterations, {how}")
            else:  # the fused kernel records no flag
                imgui.text(f"this frame: {int(n)} Newton iterations")
        if ses.take is not None and ses.take.frames:
            tr = self._trace_data(ses.take)
            ok = np.isfinite(tr["niter"])
            if ok.any():
                its, cv, rd = tr["niter"][ok], tr["converged"][ok], tr["rounds"][ok]
                line = f"this take: {its.mean():.1f} iterations a step, at most {int(its.max())}"
                if np.isfinite(rd).any():
                    line += f"; {np.nanmean(rd):.1f} proximal rounds a step"
                if np.isfinite(cv).any():
                    line += f"; {int((cv == 0).sum())} of {int(np.isfinite(cv).sum())} frames unconverged"
                imgui.text(line)

    def _seek_time(self, t):
        ses = self.session
        times = self._trace_data(ses.take)["t"]
        if len(times):
            ses.pause()
            ses.seek(int(np.clip(np.searchsorted(times, t), 0, len(times) - 1)))

    def _trace_data(self, take):
        key = (id(take), len(take.frames))
        if self._traces is not None and self._traces[0] == key:
            return self._traces[1]
        fs = take.frames
        st = [f.stats for f in fs]
        out = dict(
            t=np.array([f.time for f in fs], np.float64),
            ncon_lodos=np.array([x["ncon_lodos"] for x in st], np.float64),
            ncon_prim=np.array([x["ncon_prim"] for x in st], np.float64),
            pen=np.array([-min(x["depth"], 0.0) for x in st], np.float64),
            ke=np.array([x["ke"] for x in st], np.float64),
            niter=np.array([x["niter"] for x in st], np.float64),
            converged=np.array([x.get("converged", np.nan) for x in st], np.float64),  # 1, 0, or NaN: none recorded
            rounds=np.array([x.get("rounds", np.nan) for x in st], np.float64),  # Lodos's proximal rounds
        )
        if not out["ncon_prim"].any():
            out["ncon_prim"] = np.zeros(0)
        if not out["ncon_lodos"].any() and len(out["ncon_prim"]):
            out["ncon_lodos"] = np.zeros(0)
        self._traces = (key, out)
        return out

    def _ui_gizmos(self, imgui):
        ses = self.session
        self.gizmo_hot = False
        if not ses.loaded or ses.frame is None or self.viewer.gui is None:
            self.gizmo_active = False
            return
        fr = ses.frame
        if self.gizmo_all:
            targets = [j for j in self.joints]
        elif self.selected is not None:
            targets = J.body_joints(self.joints, self.selected)
        else:
            targets = []
        if not targets:
            self.gizmo_active = False
            return
        giz = self.viewer.gui.ui.giz
        io = imgui.get_io()
        giz.set_orthographic(False)
        giz.set_rect(0.0, 0.0, float(io.display_size[0]), float(io.display_size[1]))
        giz.set_gizmo_size_clip_space(0.09)
        cam = self.viewer.camera
        view = cam.get_view_matrix().reshape(4, 4).transpose()
        proj = cam.get_projection_matrix().reshape(4, 4).transpose()
        m16 = lambda M: giz.Matrix16(np.asarray(M, np.float32).flatten(order="F").tolist())
        view_, proj_ = m16(view), m16(proj)
        OP = giz.OPERATION
        tr_ops = {"x": OP.translate_x, "y": OP.translate_y, "z": OP.translate_z}
        ro_ops = {"x": OP.rotate_x, "y": OP.rotate_y, "z": OP.rotate_z}
        qpos = fr.qpos
        edited = None
        for jt in targets:
            if jt.type == J.FREE and not self._joint_top(jt):
                continue
            M = J.gizmo_frame(fr, jt)
            t_axes, r_axes = J.gizmo_ops(jt)
            ops = []
            if t_axes == "xyz":
                ops.append(OP.translate)
            else:
                ops += [tr_ops[a] for a in t_axes]
            if r_axes == "xyz":
                ops.append(OP.rotate)
            else:
                ops += [ro_ops[a] for a in r_axes]
            local = self.gizmo_local or jt.type in (J.HINGE, J.SLIDE)
            mode = giz.MODE.local if local else giz.MODE.world
            giz.push_id(f"j{jt.id}")
            M_ = m16(M)
            moved = False
            for op in ops:
                moved = bool(giz.manipulate(view_, proj_, op, mode, M_, None, None)) or moved
            self.gizmo_hot = self.gizmo_hot or bool(giz.is_over())
            giz.pop_id()
            if moved:
                M_new = np.asarray(M_.values, np.float64).reshape(4, 4, order="F")
                qpos = J.apply_gizmo(fr, jt, M, M_new, qpos if edited is None else edited)
                edited = qpos
        self.gizmo_active = bool(giz.is_using_any())
        self.gizmo_hot = self.gizmo_hot or self.gizmo_active
        if edited is not None:
            ses.edit_state(qpos=edited)

    def _joint_top(self, jt):
        """A free joint's gizmo sets its body's world pose, which is only its `qpos` at the top of a tree."""
        return int(self.session.source.mjm.body_parentid[jt.body]) == 0

    # ---------------------------------------------------------------------------------
    # The loop

    def run(self):
        """Open the window and run until it is closed."""
        while self.viewer.is_running():
            if self.pending is not None:
                self._run_pending()
            ses = self.session
            try:
                ses.tick()
            except Exception as exc:  # a failed step pauses the session instead of closing the window
                import traceback

                traceback.print_exc()
                ses.pause()
                self.live_status = f"step failed: {exc}"
            try:
                if ses.loaded and ses.frame is not None:
                    if self.scene is not ses.scene:
                        self._attach_scene()
                    self.traj = TrajectoryView(ses.take)
                    self.frame_index = ses.frame_index
                    self._autoscale()
                    self.draw()
                else:
                    self.viewer.begin_frame(0.0)
                    self.viewer.end_frame()
            except Exception as exc:  # drawing failed: say so and keep the session
                import traceback

                traceback.print_exc()
                ses.pause()
                self.live_status = f"drawing failed: {type(exc).__name__}: {exc}"
        self.viewer.close()

    def _autoscale(self):
        """Keep the arrow scales normalized over the take as it grows, without twitching every frame."""
        if not self.auto_scale:
            return
        t = self.session.take
        key = self._rescaled_at
        if key is None or key[0] != t.id or len(t.frames) > 1.25 * max(key[1], 1) + 8:
            self.rescale()
            self._rescaled_at = (t.id, len(t.frames))


def _summary(v, marks):
    """A parameter over a set of pairs (`v`, a row each): `(the first's, whether they differ, how many are set)`."""
    first = v[0]
    return first, bool(not np.all(v == first)), int(np.count_nonzero(marks))


def _screen_size(width, height, fill=0.85):
    """A window `fill` of the default screen, unless given; 1920x1080 if pyglet cannot say."""
    try:
        import pyglet

        sc = pyglet.display.get_display().get_default_screen()
        w, h = int(fill * sc.width), int(fill * sc.height)
    except Exception:  # noqa: BLE001 -- no display to ask
        w, h = 1920, 1080
    return width or w, height or h


def _ray_mesh(o, d, v, f):
    """Distance along the ray `o + t d` to its first hit on the triangles `v[f]`, or inf."""
    if not len(f):
        return np.inf
    c = v.mean(0)
    r = np.linalg.norm(v - c, axis=1).max()
    dn = d / max(np.linalg.norm(d), 1e-12)
    oc = c - o
    tc = oc @ dn
    if np.linalg.norm(oc - tc * dn) > r:
        return np.inf
    a, b, cc = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    e1, e2 = b - a, cc - a
    p = np.cross(dn, e2)
    det = (e1 * p).sum(1)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = o - a
    u = (s * p).sum(1) * inv
    q = np.cross(s, e1)
    w = (dn * q).sum(1) * inv
    t = (e2 * q).sum(1) * inv
    hit = ok & (u >= 0) & (w >= 0) & (u + w <= 1) & (t > 0)
    return float(t[hit].min()) if hit.any() else np.inf
