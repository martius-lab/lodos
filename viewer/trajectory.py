"""
A recorded run, on the host: what the slider scrubs through.

Scrubbing needs the frames copied off the device as they are produced
(`viewer.collect`). That copy is the reason a recorded run is slower than a
plain one -- it synchronizes, once per recorded frame -- and why it is not part of
the step: a step that reads back would not capture into a CUDA graph, and every step
would pay.

Frames are ragged, because contact counts are. They are stored that way rather
than padded to `ncon_max`: at a capacity sized for a whole batch the padding would
be most of the memory, and nothing downstream wants the padding. Only the
live prefix of each contact array is copied, which is also what keeps the per-frame
cost proportional to the contacts that exist rather than to the capacity.

Rough size: a frame with `n` contacts costs `56 n` bytes plus the poses, so 500
contacts over 5000 frames is about 140 MB. Record every k-th step (the driver's
`--record-every`) when that is too much; the trajectory is a rendering artifact,
not the simulation, and dropping frames from it changes nothing about the run.
"""

from dataclasses import dataclass

import numpy as np


#######################################################################################
# Types
#######################################################################################


@dataclass
class Frame:
    """One collected step. Every array is host-side numpy, live entries only."""

    time: float
    con_p: np.ndarray  # (n, 3) contact position
    con_n: np.ndarray  # (n, 3) unit normal, out of the SDF geometry
    con_v: np.ndarray  # (n, 3) relative velocity, vertex geometry minus SDF
    con_f: np.ndarray  # (n, 3) total contact force on the vertex geometry
    con_depth: np.ndarray  # (n,) penetration, < 0
    con_seg: np.ndarray  # (n,) segment of origin
    geom_xform: np.ndarray  # (nB, 7) [pos, quat] in Warp's (x, y, z, w) order
    body_xform: np.ndarray  # (nbody, 7)
    jnt_anchor: np.ndarray  # (njnt, 3)
    jnt_axis: np.ndarray  # (njnt, 3)

    @property
    def ncon(self):
        return self.con_p.shape[0]

    def decompose(self, field):
        """
        A contact vector split along the normal, as `(normal, tangential)`.

        Both the force and the velocity are drawn this way, and both are stored
        whole rather than pre-split: the normal is already there, the split is two
        lines, and storing three vectors where one will do triples the recording.
        """
        x = getattr(self, field)
        xn = np.sum(self.con_n * x, axis=-1, keepdims=True) * self.con_n
        return xn, x - xn


# The per-frame contact arrays, in the order `save`/`load` write them.
CONTACT_FIELDS = ("con_p", "con_n", "con_v", "con_f", "con_depth", "con_seg")
FRAME_FIELDS = ("geom_xform", "body_xform", "jnt_anchor", "jnt_axis")


class Trajectory:
    """
    A growing list of `Frame`, plus the sizes a viewer needs before it has seen
    one. Indexable and iterable; `traj[t]` is the frame the slider shows at `t`.
    """

    def __init__(self, nB, nbody, njnt, dt=0.0, frames=None):
        # `dt` is the interval between *recorded* frames, not the simulation
        # timestep -- a recording that keeps every k-th step has a k times coarser
        # one, and video timing is the only thing that reads it.
        self.nB = int(nB)
        self.nbody = int(nbody)
        self.njnt = int(njnt)
        self.dt = float(dt)
        self.frames = list(frames or [])

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, t):
        return self.frames[t]

    def __iter__(self):
        return iter(self.frames)

    def peak(self, field, percentile=99.0):
        """
        A representative large magnitude of a contact vector over the trajectory.

        What the viewer's automatic arrow scaling is normalized by: forces and
        velocities have no length in metres, so something has to set the scale, and
        a constant over the trajectory is what keeps a frame's arrows comparable
        with the next frame's.

        A percentile rather than the maximum, because the maximum is an impact. A
        stack of armadillos landing peaks at 36 kN and settles at two orders of
        magnitude less, so scaling to the peak leaves every settled contact drawn
        sub-pixel -- the frames stay comparable and all of them are empty. Pass
        `percentile=100` for the true maximum.
        """
        mags = [
            np.linalg.norm(getattr(frame, field), axis=-1)
            for frame in self.frames
            if frame.ncon
        ]
        if not mags:
            return 0.0
        return float(np.percentile(np.concatenate(mags), percentile))

    #######################################################################################
    # Persistence
    #######################################################################################

    def as_dict(self):
        """
        The recording as flat numpy arrays, ready for `np.savez`.

        Ragged contact arrays are concatenated with a `con_off` index -- the same
        shape the collision pipeline's own compaction produces -- so a frame's
        contacts are `arr[con_off[t]:con_off[t + 1]]`.
        """
        counts = [f.ncon for f in self.frames]
        out = {
            "nB": self.nB,
            "nbody": self.nbody,
            "njnt": self.njnt,
            "dt": self.dt,
            "time": np.array([f.time for f in self.frames], dtype=np.float64),
            "con_off": np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
        }

        for f in CONTACT_FIELDS:
            parts = [getattr(fr, f) for fr in self.frames]
            out[f] = np.concatenate(parts) if parts else np.zeros(0)
        for f in FRAME_FIELDS:
            parts = [getattr(fr, f) for fr in self.frames]
            out[f] = np.stack(parts) if parts else np.zeros(0)

        return out

    @classmethod
    def from_dict(cls, z):
        """Rebuild what `as_dict` produced."""
        off = z["con_off"]
        frames = [
            Frame(
                time=float(z["time"][t]),
                **{f: z[f][off[t] : off[t + 1]] for f in CONTACT_FIELDS},
                **{f: z[f][t] for f in FRAME_FIELDS},
            )
            for t in range(len(off) - 1)
        ]
        return cls(int(z["nB"]), int(z["nbody"]), int(z["njnt"]), float(z["dt"]), frames)

    def save(self, path):
        """Write the recording to a `.npz`. See `viewer.draw.save_recording`
        for the form that also carries the scene, and so can be viewed anywhere."""
        np.savez_compressed(path, **self.as_dict())
        return path

    @classmethod
    def load(cls, path):
        """Read back what `save` wrote."""
        return cls.from_dict(np.load(path))
