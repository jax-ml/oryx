# Copyright 2024 The oryx Authors.
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

"""Wraps TFP bijectors for use with Jax."""
from jax import tree_util
from jax import util as jax_util
import jax.numpy as np

from oryx.core import primitive
from oryx.core.interpreters.inverse import core as inverse_core
from oryx.core.interpreters.inverse import slice as slc

__all__ = [
    'patch_bijector',
]

safe_map = jax_util.safe_map

InverseAndILDJ = inverse_core.InverseAndILDJ
NDSlice = slc.NDSlice

bijector_p = primitive.InitialStylePrimitive('bijector')


def bijector_ildj_rule(incells, outcells, *, in_tree, num_consts, direction,
                       num_bijector, **_):
  """Inverse/ILDJ rule for bijectors."""
  const_incells, flat_incells = jax_util.split_list(incells, [num_consts])
  flat_bijector_cells, arg_incells = jax_util.split_list(
      flat_incells, [num_bijector])
  if any(not cell.top() for cell in flat_bijector_cells):
    return (const_incells + flat_incells, outcells, None)
  bijector_vals = [cell.val for cell in flat_bijector_cells]
  bijector, _ = tree_util.tree_unflatten(
      in_tree, bijector_vals + [None] * len(arg_incells))
  if direction == 'forward':
    forward_func = bijector.forward
    inv_func = bijector.inverse
    ildj_func = bijector.inverse_log_det_jacobian
  elif direction == 'inverse':
    forward_func = bijector.inverse
    inv_func = bijector.forward
    ildj_func = bijector.forward_log_det_jacobian
  else:
    raise ValueError('Bijector direction must be ' '"forward" or "inverse".')

  outcell, = outcells
  incell = flat_incells[-1]
  if incell.bottom() and not outcell.bottom():
    val, ildj = outcell.val, outcell.ildj
    inildj = ildj + ildj_func(val, np.ndim(val))
    ndslice = NDSlice.new(inv_func(val), inildj)
    flat_incells = [InverseAndILDJ(incell.aval, [ndslice])]
    new_outcells = outcells
  elif outcell.is_unknown() and not incell.is_unknown():
    new_outcells = [InverseAndILDJ.new(forward_func(incell.val))]
  new_incells = flat_bijector_cells + flat_incells
  return (const_incells + new_incells, new_outcells, None)

inverse_core.ildj_registry[bijector_p] = bijector_ildj_rule


def patch_bijector(bij_type):
  """Patches a TFP bijector to use a primitive in its forward/inverse methods."""
  old_forward = bij_type.forward
  old_inverse = bij_type.inverse

  def bijector_bind(bijector, x, **kwargs):
    return primitive.initial_style_bind(
        bijector_p,
        direction=kwargs['direction'],
        num_bijector=len(tree_util.tree_leaves(bijector)),
        bijector_name=bijector.__class__.__name__)(_bijector)(bijector, x,
                                                              **kwargs)

  def _bijector(bij, x, **kwargs):
    direction = kwargs.pop('direction', 'forward')
    if direction == 'forward':
      return old_forward(bij, x, **kwargs)
    elif direction == 'inverse':
      return old_inverse(bij, x, **kwargs)
    else:
      raise ValueError('Bijector direction must be "forward" or "inverse".')

  def forward(self, x, **kwargs):
    return bijector_bind(self, x, direction='forward', **kwargs)

  def inverse(self, x, **kwargs):
    return bijector_bind(self, x, direction='inverse', **kwargs)

  bij_type.forward = forward
  bij_type.inverse = inverse
