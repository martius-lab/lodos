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
# Lodos's own allocation of its contact groups' arrays in lodos.mjcf.types.Data; the parts of MuJoCo Warp 3.14.0's
# mujoco_warp/_src/io.py that Lodos uses are adapted in lodos/mjcf/io.py.

"""The solver's allocations beside `lodos.mjcf.io.put_data`: the contact groups' arrays
(`alloc_contact_groups`)."""

import warp as wp

from lodos.mjcf import types


def alloc_contact_groups(m: types.Model, d: types.Data, nmember: int):
  """Allocates the contact-group arrays: per contact (naconmax), and nmember members.

  A contact typed ContactType.GROUP stands for the members
  group_member_*[contact_group_adr : contact_group_adr + contact_group_num]: whoever writes the
  contacts writes the members too, and gives the group condim 1 (a ray: one row along
  normal - slip * t1 per member) or 3 (a cone). Until this is called the arrays are empty, and
  make_constraint and the prox solver compile none of the group paths. Call it before any
  graph is captured: it replaces the arrays.

  Args:
    m: model; groups take neither flex nor geom surface velocities.
    d: data; its contact_group_* and group_member_* arrays are replaced.
    nmember: members the arrays hold, over all worlds (>= 1).
  """
  if nmember < 1:
    raise ValueError(f"alloc_contact_groups: nmember must be >= 1, got {nmember}")
  if m.nflex > 0:
    raise ValueError("alloc_contact_groups: contact groups do not support flex")
  if m.flg_surfacevel:
    raise ValueError("alloc_contact_groups: contact groups do not support geom surface velocities")
  n = d.naconmax
  d.contact_group_adr = wp.zeros((n,), dtype=int)
  d.contact_group_num = wp.zeros((n,), dtype=int)
  d.contact_group_force = wp.zeros((n,), dtype=wp.mat33)
  d.contact_group_moment = wp.zeros((n,), dtype=wp.mat33)
  d.contact_group_imp = wp.zeros((n,), dtype=wp.vec3)
  d.group_member_pos = wp.zeros((nmember,), dtype=wp.vec3)
  d.group_member_frame = wp.zeros((nmember,), dtype=wp.mat33)
  d.group_member_dist = wp.zeros((nmember,), dtype=float)
  d.group_member_weight = wp.zeros((nmember,), dtype=float)
  d.group_member_slip = wp.zeros((nmember,), dtype=float)
