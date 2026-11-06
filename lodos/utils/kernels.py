# Copyright 2025 The Newton Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Adapted for Lodos from MuJoCo Warp 3.14.0's mujoco_warp/_src/warp_util.py: the parts Lodos uses; every function keeps
# its MuJoCo Warp name.

"""Warp utilities: a cache of kernels built by factories, timed event scopes, and the array-spec
and toolkit checks the MJCF io uses."""

import dataclasses
import functools
import inspect
import warnings

import warp as wp


_STACK = None


def is_array_spec(typ) -> bool:
  """Check if a type annotation is an array spec (wp.array instance or bracket annotation)."""
  return isinstance(typ, wp.array) or type(typ).__name__ == "_ArrayAnnotation"


def mark_batched(obj):
  """Recursively set _is_batched = True on all batched warp arrays within obj."""
  if not dataclasses.is_dataclass(obj):
    return
  for f in dataclasses.fields(obj):
    val = getattr(obj, f.name, None)
    if val is None:
      continue
    if dataclasses.is_dataclass(val):
      mark_batched(val)
      continue
    if not isinstance(val, wp.array):
      continue
    if not is_array_spec(f.type):
      continue
    spec_shape = getattr(f.type, "shape", ())
    if spec_shape and spec_shape[0] in ("*", "nworld"):
      val._is_batched = True


def _merge(a: dict, b: dict) -> dict:
  """Merges two event trace stacks."""
  ret = {}
  if not a or not b:
    return dict(**a, **b)
  if set(a) != set(b):
    raise ValueError("incompatible stacks")
  for key in a:
    a1_events, a1_substack = a[key]
    a2_events, a2_substack = b[key]
    ret[key] = (a1_events + a2_events, _merge(a1_substack, a2_substack))
  return ret


def event_scope(fn, name: str = ""):
  """Wraps a function and records an event before and after the function invocation."""
  name = name or getattr(fn, "__name__")

  @functools.wraps(fn)
  def wrapper(*args, **kwargs):
    global _STACK
    if _STACK is None:
      return fn(*args, **kwargs)

    for frame_info in inspect.stack():
      if frame_info.function in ("capture_while", "capture_if"):
        return fn(*args, **kwargs)

    # push into next level of stack
    saved_stack, _STACK = _STACK, {}
    beg = wp.Event(enable_timing=True)
    end = wp.Event(enable_timing=True)
    wp.record_event(beg)
    res = fn(*args, **kwargs)
    wp.record_event(end)
    # pop back up to current level
    sub_stack, _STACK = _STACK, saved_stack
    # append events and substack
    prev_events, prev_substack = _STACK.get(name, ((), {}))
    events = prev_events + ((beg, end),)
    sub_stack = _merge(prev_substack, sub_stack)
    _STACK[name] = (events, sub_stack)
    return res

  return wrapper


_KERNEL_CACHE = {}


def cache_kernel(func):
  # caching kernels to avoid crashes in graph_conditional code
  @functools.wraps(func)
  def wrapper(*args):
    def _hash_arg(a):
      if hasattr(a, "size"):
        return a.size
      if isinstance(a, list):
        return hash(tuple(a))
      return hash(a)

    key = tuple(_hash_arg(a) for a in args) + (hash(func.__name__),)
    if key not in _KERNEL_CACHE:
      _KERNEL_CACHE[key] = func(*args)
    return _KERNEL_CACHE[key]

  return wrapper


def check_toolkit_driver():
  wp.init()
  if wp.get_device().is_cuda:
    if not wp.is_conditional_graph_supported():
      warnings.warn(
        """
        CUDA version < 12.4 detected
        - graph capture may be unreliable for < 12.3
        - conditional graph nodes are not available for < 12.4
          Model.opt.graph_conditional should be set to False
        """,
        stacklevel=2,
      )
