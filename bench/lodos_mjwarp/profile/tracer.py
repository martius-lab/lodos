"""
Stage timing inside a captured CUDA graph, shared by the Lodos and the Newton profilers.

`Tracer.wrap` puts a scope around a function (a module attribute or an instance method):
a timing event before it and one after, recorded where the function runs. Called during
`wp.ScopedCapture`, the events become nodes of the graph, so every launch of the graph
re-records them and `read` (after a synchronize) gives each scope's time in that launch.
Scopes nest by call: a scope's path is its enclosing scopes' names joined by "/".
Nothing is recorded inside a `capture_while`/`capture_if` body (MJWarp's event_scope rule).
"""

from __future__ import annotations

import contextlib
import functools
import sys
import types

import warp as wp


def _in_conditional():
    f = sys._getframe(2)
    while f is not None:
        if f.f_code.co_name in ("capture_while", "capture_if"):
            return True
        f = f.f_back
    return False


class Tracer:
    def __init__(self):
        self.records = []  # (path, beg, end), in recording order
        self.stack = []
        self.patches = []

    @contextlib.contextmanager
    def scope(self, name):
        self.stack.append(name)
        path = "/".join(self.stack)
        beg, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
        wp.record_event(beg)
        try:
            yield
        finally:
            wp.record_event(end)
            self.stack.pop()
            self.records.append((path, beg, end))

    def wrap(self, obj, attr, name):
        orig = getattr(obj, attr)
        tracer = self

        @functools.wraps(orig)
        def wrapper(*a, **k):
            if _in_conditional():
                return orig(*a, **k)
            with tracer.scope(name):
                return orig(*a, **k)

        own = isinstance(obj, types.ModuleType) or attr in getattr(obj, "__dict__", {})
        setattr(obj, attr, wrapper)
        self.patches.append((obj, attr, orig, own))

    def unwrap(self):
        for obj, attr, orig, own in reversed(self.patches):
            if own:
                setattr(obj, attr, orig)
            else:
                delattr(obj, attr)  # the instance's shadow of its class's method
        self.patches = []

    def read(self):
        """`{path: ms}` of the last launch; a scope entered twice in one step is summed."""
        out = {}
        for path, beg, end in self.records:
            out[path] = out.get(path, 0.0) + wp.get_event_elapsed_time(beg, end)
        return out


class Accumulator:
    """Per-path sums over sampled steps, and the counters read at them."""

    def __init__(self):
        self.ms, self.n, self.counts = {}, 0, {}

    def add(self, ms, counts=None):
        for k, v in ms.items():
            self.ms[k] = self.ms.get(k, 0.0) + v
        for k, v in (counts or {}).items():
            self.counts.setdefault(k, []).append(v)
        self.n += 1

    def result(self):
        mean = {k: v / max(1, self.n) for k, v in self.ms.items()}
        selfs = {}
        for k, v in mean.items():
            kids = [c for c in mean if c.startswith(k + "/") and "/" not in c[len(k) + 1:]]
            if kids:
                selfs[k + "/(self)"] = v - sum(mean[c] for c in kids)
        stats = {}
        for k, vs in self.counts.items():
            import numpy as np

            a = np.asarray(vs, np.float64)
            stats[k] = dict(mean=float(a.mean()), max=float(a.max()), min=float(a.min()))
        return dict(samples=self.n, ms=mean, self_ms=selfs, counts=stats)
