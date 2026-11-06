"""
The live viewer's Performance panel: a scene's tuning file, its state set, the tools of
`tools/performance_tuning/` run in the background, and what the state on screen costs.

Everything that measures throughput runs in a separate process, at a batch size of the
user's choosing, from the scene's state set: one world -- the viewer's -- says nothing about
a batched run's speed, and a timing taken while the viewer steps on the same GPU is
disturbed by it (pause the viewer for clean numbers). What the panel reads in-process is the
current state's demand on the viewer's own capacities: each Lodos stage's survivors, MJWarp's
contacts and rows.

Applying a tuning file here takes its knobs, not its capacities (`lodos.utils.tuning.knobs_only`):
the viewer runs one world and grows its own capacities when a step overflows them. The
collision knobs (tile sizes) rebuild the runners; the solver's budgets, and the contact
reduction, become the panel's options, where they can be seen and changed. A tuning file
tunes Lodos, not Newton's sims. With `auto` (`python -m viewer --tuned`) the file is
applied on every scene load and sim switch.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from lodos.utils import states as lstates
from lodos.utils import tuning as ltuning
from viewer.session import SIM_NAMES

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools" / "performance_tuning"
TUNED_DIR = TOOLS / "tuned"
WORK_DIR = ROOT / "results" / "tuning"


def scene_key(source):
    """The name a scene's files go by: a built-in's name, an MJCF file's stem."""
    if source.path:
        return Path(source.path).stem
    return source.name.split(",")[0]


def scene_arg(source):
    """How the tools are told the scene: an MJCF path, or a built-in's name."""
    return str(source.path) if source.path else scene_key(source)


class Performance:
    """The panel's state; `ui` draws it. The other methods do not draw."""

    def __init__(self, app):
        self.app = app
        self.key = None
        self.tuning_path = ""
        self.tuning = None
        self.stale = []
        self.applied = None
        self.sset = None
        self.sset_path = ""
        self.pick = -1
        self.take_stride = 10
        self.job = None
        self.job_kind = ""
        self.job_log = None
        self.job_out = None
        self.job_started = 0.0
        self.B = 1024
        self.counters = None
        self.profile = None
        self.tile_size = 8
        self.status = ""
        self.expand = False  # open the section when it first appears (a script driving the window)
        self.auto = False  # apply the scene's tuning file on every load and sim switch

    # -------------------------------------------------------------------------------
    # The scene

    @property
    def session(self):
        return self.app.session

    def on_scene(self):
        """A scene was loaded: its files' default paths, and its shipped tuning file if there is one."""
        src = self.session.source
        key = scene_key(src)
        if key == self.key:
            return
        self.key = key
        self.tuning, self.applied, self.stale, self.sset, self.pick = None, None, [], None, -1
        self.counters = self.profile = None
        self.tuning_path = str(TUNED_DIR / f"{key}.tuning.json")
        self.sset_path = str(WORK_DIR / "states" / f"{key}.states.npz")
        if Path(self.tuning_path).exists():
            self.load_tuning(self.tuning_path)
        if Path(self.sset_path).exists():
            try:
                self.sset = lstates.load(self.sset_path, src.mjm)
            except ValueError as e:
                self.status = f"state set: {e}"

    # -------------------------------------------------------------------------------
    # The tuning file

    def load_tuning(self, path):
        src = self.session.source
        self.tuning = ltuning.load(path)
        self.tuning_path = str(path)
        self.stale = ltuning.stale(self.tuning, src.mjm, src.surfaces)
        self.status = f"loaded {Path(path).name}" + (" -- STALE" if self.stale else "")

    def apply_tuning(self):
        """The file's knobs on the viewer: collision options rebuild the runners, budgets become options."""
        if self.tuning is None:
            return
        ses = self.session
        sim = ses.config["sim"]
        if sim != "lodos":
            self.status = f"a tuning file tunes Lodos, not {SIM_NAMES[sim]}"
            return
        sec = ltuning.knobs_only(self.tuning)
        # Only what differs: setting the collision options rebuilds every runner, and under
        # `auto` this runs on every sim switch.
        if ses.source.surfaces is not None:
            have = ses.source.surfaces.col_kwargs
            col = {k: v for k, v in sec.get("collision", {}).items() if have.get(k) != v}
            if col:
                self.set_collision(**col)
        # The budgets, and the make_sim knobs the panel has as options (the contact reduction).
        opts = dict(sec.get("solver_kwargs", {}))
        opts.update({k: int(bool(v)) for k, v in sec.get("sim", {}).items() if k == "contact_reduction"})
        for k, v in opts.items():
            if k in ses.config["options"] and ses.config["options"][k] != v:
                ses.set_option(k, v)
        self.applied = (self.tuning_path, sim)
        self.status = f"applied {Path(self.tuning_path).name}'s knobs"

    def on_load(self):
        """What a scene load or a sim switch does here: with `auto`, the scene's file applied."""
        if not self.auto:
            return
        self.on_scene()
        if self.tuning is None:
            self.status = "no tuning file for this scene: its own settings"
            return
        self.apply_tuning()

    def set_collision(self, **kw):
        """Lodos collision options for every runner from here on; the runners are rebuilt."""
        ses = self.session
        surf = ses.source.surfaces
        if surf is None:
            return
        col = dict(surf.col_kwargs)
        col.update(kw)
        ses.source.surfaces = dataclasses.replace(surf, col_kwargs=col)
        ses.runners.clear()
        cfg, st, parent = ses._base()
        ses._make_draft(cfg, st, parent)

    # -------------------------------------------------------------------------------
    # The state set

    def _set(self):
        src = self.session.source
        if self.sset is None:
            self.sset = lstates.from_qpos(src.mjm, np.zeros((0, src.mjm.nq)), source=dict(kind="viewer"))
        return self.sset

    def _from_frames(self, frames, label):
        src = self.session.source
        q = np.stack([f.qpos for f in frames])
        v = np.stack([f.qvel for f in frames])
        return lstates.from_qpos(src.mjm, q, qvel=v, ctrl=src.ctrl,
                                 labels=[f"{label} t={f.time:.3f}" for f in frames], source=dict(kind="viewer"))

    def add_frame(self):
        fr = self.session.frame
        if fr is None:
            return
        new = self._from_frames([fr], "frame")
        self.sset = new if self.sset is None or not len(self.sset) else lstates.concat(self.sset, new)
        self.status = f"{len(self.sset)} states"

    def add_take(self, stride=None):
        t = self.session.take
        if t is None or not t.frames:
            return
        frames = t.frames[:: max(1, int(stride or self.take_stride))]
        new = self._from_frames(frames, f"take {t.id}")
        self.sset = new if self.sset is None or not len(self.sset) else lstates.concat(self.sset, new)
        self.status = f"{len(self.sset)} states"

    def save_states(self, path=None):
        if self.sset is None or not len(self.sset):
            return
        self.sset_path = str(path or self.sset_path)
        self.sset.save(self.sset_path)
        self.status = f"saved {len(self.sset)} states to {self.sset_path}"

    def load_states(self, path=None):
        self.sset_path = str(path or self.sset_path)
        self.sset = lstates.load(self.sset_path, self.session.source.mjm)
        self.pick = -1
        self.status = f"loaded {len(self.sset)} states"

    def show_state(self, i):
        """State `i` of the set, as the viewer's state."""
        self.pick = int(i)
        self.session.edit_state(qpos=self.sset.qpos[i], qvel=self.sset.qvel[i])

    # -------------------------------------------------------------------------------
    # Background tools

    def command(self, kind):
        """The command line `kind` ("explore", "calibrate", "profile", "tune") runs."""
        src = self.session.source
        py = sys.executable
        scene = scene_arg(src)
        if kind == "explore":
            return [py, str(TOOLS / "explore.py"), scene, "--out", self.sset_path]
        if kind == "calibrate":
            return [py, str(TOOLS / "calibrate.py"), scene, "--states", self.sset_path, "--B", str(self.B),
                    "--out", self.tuning_path, "--update"]
        if kind == "profile":
            self.job_out = str(WORK_DIR / "logs" / f"{self.key}.profile.json")
            cmd = [py, str(TOOLS / "profile_scene.py"), scene, "--B", str(self.B), "--json", self.job_out]
            if Path(self.tuning_path).exists():
                cmd += ["--tuning", self.tuning_path]
            if Path(self.sset_path).exists():
                cmd += ["--states", self.sset_path]
            return cmd
        if kind == "tune":
            return [py, str(TOOLS / "tune.py"), scene, "--out-dir", str(Path(self.tuning_path).parent),
                    "--states", self.sset_path]
        raise ValueError(kind)

    def start(self, kind):
        if self.running:
            return
        if kind in ("calibrate", "tune") and not Path(self.sset_path).exists():
            self.status = "save or explore a state set first"
            return
        (WORK_DIR / "logs").mkdir(parents=True, exist_ok=True)
        log = WORK_DIR / "logs" / f"{self.key}.{kind}.log"
        self.job_log = log
        self.job_kind = kind
        self.job_started = time.time()
        self.job = subprocess.Popen(self.command(kind), stdout=open(log, "w"), stderr=subprocess.STDOUT, cwd=ROOT)

    @property
    def running(self):
        return self.job is not None and self.job.poll() is None

    def poll(self):
        """Called every frame: picks up what a finished job wrote."""
        if self.job is None or self.job.poll() is None:
            return
        code, kind = self.job.returncode, self.job_kind
        self.job = None
        if code:
            self.status = f"{kind} failed (exit {code}); see {self.job_log}"
            return
        self.status = f"{kind} done in {time.time() - self.job_started:.0f} s"
        if kind in ("calibrate", "tune") and Path(self.tuning_path).exists():
            self.load_tuning(self.tuning_path)
        if kind == "explore" and Path(self.sset_path).exists():
            self.load_states()
        if kind == "profile" and self.job_out and Path(self.job_out).exists():
            self.profile = json.loads(Path(self.job_out).read_text())

    def stop(self):
        if self.running:
            self.job.terminate()
            self.status = f"{self.job_kind} stopped"

    def log_tail(self, n=6):
        if self.job_log is None or not Path(self.job_log).exists():
            return []
        lines = [ln for ln in Path(self.job_log).read_text(errors="replace").splitlines()
                 if ln.strip() and not ln.startswith(("Warp ", "   ", "Module "))]
        return lines[-n:]

    # -------------------------------------------------------------------------------
    # The current state's demand

    def read_counters(self):
        """Each capacity of the viewer's runner against what the state on screen uses."""
        ses = self.session
        r = ses.runner(ses.config)
        sim = r.sim
        out = []
        if sim is None:  # one of Newton's sims, in its own process: its contacts against its buffer
            fr = ses.frame
            out.append(("Newton contacts", int(fr.stats.get("ncon_prim", 0)) if fr is not None else 0,
                        int(r.info.get("contacts_per_world", 0))))
            self.counters = out
            return out
        if sim.dcol is not None:
            d, ct = sim.dcol, sim.dcol.ct.numpy()
            out += [("Lodos candidates", int(ct[0]), int(d.nCa1_max)), ("Lodos tiles", int(ct[3]), int(d.nTa_max)),
                    ("Lodos voxel survivors", int(ct[1]), int(d.nCa2_max)),
                    ("Lodos contacts", int(ct[2]), int(d.nCa3_max))]
        d = sim.d
        out.append(("Runtime contacts", int(d.nacon.numpy()[0]), int(d.naconmax)))
        out.append(("Runtime rows", int(d.nefc.numpy().max()), int(d.njmax)))
        self.counters = out
        return out

    # -------------------------------------------------------------------------------
    # Drawing

    def ui(self, imgui):
        self.poll()
        imgui.set_next_item_open(self.expand, imgui.Cond_.appearing)
        if not imgui.collapsing_header("Performance"):
            return
        self.on_scene()
        dpi = self.app._dpi()
        ses = self.session

        imgui.text("tuning file")
        imgui.set_next_item_width(-1)
        _, self.tuning_path = imgui.input_text("##tuning", self.tuning_path)
        if imgui.button("Load##tuning") and Path(self.tuning_path).exists():
            self.load_tuning(self.tuning_path)
        imgui.same_line()
        imgui.begin_disabled(self.tuning is None)
        if imgui.button("Apply knobs"):
            self.apply_tuning()
        imgui.end_disabled()
        ch, self.auto = imgui.checkbox("apply on every load and sim switch", self.auto)
        if ch and self.auto:
            self.on_load()
        if self.tuning is not None:
            for line in ltuning.describe(self.tuning).splitlines():
                imgui.text_wrapped(line)
            base = self.tuning.get("baseline") or {}
            if base.get("throughput"):
                imgui.text_disabled(f"before tuning: {base['throughput']['world_steps_per_s']:.3g} world-steps/s "
                                    f"at B={base.get('B')}")
            if self.stale:
                imgui.text_colored(imgui.ImVec4(1.0, 0.5, 0.2, 1.0), "stale: " + "; ".join(self.stale[:3]))

        imgui.separator()
        n = 0 if self.sset is None else len(self.sset)
        imgui.text(f"state set: {n} states")
        imgui.set_next_item_width(-1)
        _, self.sset_path = imgui.input_text("##states", self.sset_path)
        if imgui.button("Add frame"):
            self.add_frame()
        imgui.same_line()
        if imgui.button("Add take"):
            self.add_take()
        imgui.same_line()
        imgui.set_next_item_width(50 * dpi)
        _, self.take_stride = imgui.input_int("every", self.take_stride, 0)
        if imgui.button("Save##states"):
            self.save_states()
        imgui.same_line()
        if imgui.button("Load##states") and Path(self.sset_path).exists():
            self.load_states()
        if n and imgui.begin_list_box("##statelist", imgui.ImVec2(-1, 90 * dpi)):
            for i in range(min(n, 2000)):
                clicked, _ = imgui.selectable(f"{i}: {self.sset.labels[i]}", i == self.pick)
                if clicked:
                    self.show_state(i)
            imgui.end_list_box()

        imgui.separator()
        imgui.text("tools (in the background, at a batch)")
        imgui.set_next_item_width(80 * dpi)
        _, self.B = imgui.input_int("B", self.B, 0)
        imgui.begin_disabled(self.running)
        for kind, tip in (("explore", "make a state set: the scene's own starts rolled out"),
                          ("calibrate", "size every capacity at B from the state set, into the tuning file"),
                          ("profile", "time each stage at B"),
                          ("tune", "the whole tuning: tiles, capacities, budgets, batch size")):
            if imgui.button(kind.capitalize()):
                self.start(kind)
            if imgui.is_item_hovered():
                imgui.set_tooltip(tip)
            imgui.same_line()
        imgui.end_disabled()
        imgui.new_line()
        if self.running:
            imgui.text(f"{self.job_kind}: running {time.time() - self.job_started:.0f} s")
            imgui.same_line()
            if imgui.button("Stop"):
                self.stop()
        for line in self.log_tail():
            imgui.text_disabled(line[:120])
        if self.profile is not None:
            p = self.profile
            imgui.text(f"profile at B={p['B']}: {p['world_steps_per_s']:.3g} world-steps/s")
            if imgui.begin_table("profile", 3, imgui.TableFlags_.sizing_stretch_same):
                for k, v in p["ms"].items():
                    imgui.table_next_row()
                    imgui.table_next_column()
                    imgui.text(k)
                    imgui.table_next_column()
                    imgui.text(f"{v:.3f} ms")
                    imgui.table_next_column()
                    if k != "whole_step":
                        imgui.progress_bar(v / max(p["stages_sum_ms"], 1e-9), imgui.ImVec2(-1, 0))
                imgui.end_table()

        imgui.separator()
        if imgui.button("Read counters"):
            self.read_counters()
        if imgui.is_item_hovered():
            imgui.set_tooltip("the state on screen against the viewer's own capacities (one world)")
        if self.counters:
            if imgui.begin_table("counters", 3, imgui.TableFlags_.sizing_stretch_same):
                for name, used, cap in self.counters:
                    imgui.table_next_row()
                    imgui.table_next_column()
                    imgui.text(name)
                    imgui.table_next_column()
                    imgui.text(f"{used} / {cap}")
                    imgui.table_next_column()
                    imgui.progress_bar(min(1.0, used / max(cap, 1)), imgui.ImVec2(-1, 0))
                imgui.end_table()
        if ses.source.surfaces is not None:
            imgui.set_next_item_width(80 * dpi)
            ch, self.tile_size = imgui.input_int("tile size (all pairs)", self.tile_size, 0)
            imgui.same_line()
            if imgui.button("Rebuild##tiles"):
                self.set_collision(tile_size=max(1, int(self.tile_size)))
        if self.status:
            imgui.text_wrapped(self.status)
