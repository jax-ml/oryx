# Copyright 2023 The oryx Authors.
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

"""Registers inverse and ILDJ rules.

This module also monkey patches `jax.nn.sigmoid`, `jax.scipy.special.logit`, and
`jax.scipy.special.expit` to have custom inverses.
"""
import inspect
import jax
from jax import lax
from jax import util as jax_util
import jax.numpy as np

from oryx.core import primitive
from oryx.core.interpreters import harvest
from oryx.core.interpreters.inverse import bijector_extensions
from oryx.core.interpreters.inverse import core as inverse_core
from oryx.core.interpreters.inverse import custom_inverse as ci
from oryx.core.interpreters.inverse import slice as slc
from tensorflow_probability.substrates import jax as tfp  # pylint: disable=g-importing-member

tfb = tfp.bijectors

__all__ = [
]

# Register all bijectors from TFP.
for name in tfb.__all__:
  bij = getattr(tfb, name)
  if inspect.isclass(bij) and issubclass(bij, tfb.Bijector):
    if bij not in {tfb.Bijector, tfb.Composition}:
      bijector_extensions.patch_bijector(bij)

ildj_registry = inverse_core.ildj_registry
register_elementwise = inverse_core.register_elementwise
register_binary = inverse_core.register_binary
InverseAndILDJ = inverse_core.InverseAndILDJ
NDSlice = slc.NDSlice
Slice = slc.Slice
safe_map = jax_util.safe_map
custom_inverse = ci.custom_inverse

register_elementwise(lax.exp_p)(np.log)
register_elementwise(lax.log_p)(np.exp)
register_elementwise(lax.sin_p)(np.arcsin)
register_elementwise(lax.cos_p)(np.arccos)
register_elementwise(lax.expm1_p)(np.log1p)
register_elementwise(lax.log1p_p)(np.expm1)
register_elementwise(lax.neg_p)(lambda x: -x)
register_elementwise(lax.sqrt_p)(np.square)


@register_elementwise(lax.integer_pow_p)
def integer_pow_inverse(z, *, y):
  """Inverse for `integer_pow_p` primitive."""
  if y == 0:
    raise ValueError('Cannot invert raising to a value to the 0-th power.')
  elif y == 1:
    return z
  elif y == -1:
    return np.reciprocal(z)
  elif y == 2:
    return np.sqrt(z)
  return lax.pow(z, 1. / y)


def pow_left(x, z, ildj_):
  # x ** y = z
  # y = f^-1(z) = log(z) / log(x)
  # grad(f^-1)(z) = 1 / (z log(x))
  # log(grad(f^-1)(z)) = log(1 / (z log(x))) = -log(z) - log(log(x))
  return np.log(z) / np.log(x), ildj_ - np.log(z) - np.log(np.log(x))


def pow_right(y, z, ildj_):
  # x ** y = z
  # x = f^-1(z) = z ** (1 / y)
  # grad(f^-1)(z) = 1 / y * z ** (1 / y - 1)
  # log(grad(f^-1)(z)) = (1 / y - 1)log(z) - log(y)
  y_inv = np.reciprocal(y)
  return lax.pow(z, y_inv), ildj_ + (y_inv - 1.) * np.log(z) - np.log(y)
register_binary(lax.pow_p)(pow_left, pow_right)


def add_left(left_val, out_val, ildj_):
  return out_val - left_val, ildj_


def add_right(right_val, out_val, ildj_):
  return out_val - right_val, ildj_
register_binary(lax.add_p)(add_left, add_right)


def sub_left(left_val, out_val, ildj_):
  return left_val - out_val, ildj_


def sub_right(right_val, out_val, ildj_):
  return out_val + right_val, ildj_
register_binary(lax.sub_p)(sub_left, sub_right)


def mul_left(left_val, out_val, ildj_):
  return out_val / left_val, -np.log(np.abs(left_val)) + ildj_


def mul_right(right_val, out_val, ildj_):
  return out_val / right_val, -np.log(np.abs(right_val)) + ildj_
register_binary(lax.mul_p)(mul_left, mul_right)


def div_left(left_val, out_val, ildj_):
  return left_val / out_val, ((np.log(left_val) - 2 * np.log(out_val)) + ildj_)


def div_right(right_val, out_val, ildj_):
  return out_val * right_val, np.log(np.abs(right_val)) + ildj_
register_binary(lax.div_p)(div_left, div_right)


def slice_ildj(incells, outcells, **params):
  """InverseAndILDJ rule for slice primitive."""
  incell, = incells
  outcell, = outcells
  start_indices = params['start_indices']
  limit_indices = params['limit_indices']
  slices = tuple(Slice(s, l) for s, l in zip(start_indices, limit_indices,
                                             strict=True))
  if outcell.top() and not incell.top():
    incells = [
        InverseAndILDJ(incell.aval, [
            NDSlice(outcell.val, outcell.ildj, *slices)])
    ]
  elif incell.top() and not outcell.top():
    outval = lax.slice_p.bind(incell.val, **params)
    outcells = [InverseAndILDJ.new(outval)]
  return incells, outcells, None
ildj_registry[lax.slice_p] = slice_ildj


def concatenate_ildj(incells, outcells, *, dimension):
  """InverseAndILDJ rule for concatenate primitive."""
  outcell, = outcells
  if all(incell.top() for incell in incells):
    invals = [incell.val for incell in incells]
    out_val = lax.concatenate_p.bind(*invals, dimension=dimension)
    outcells = [InverseAndILDJ.new(out_val)]
  elif outcell.top():
    idx = 0
    sections = []
    outval = outcell.val
    outildj = outcell.ildj
    for incell in incells[:-1]:
      d = incell.aval.shape[dimension]
      idx += d
      sections.append(idx)
    invals = np.split(outval, sections, dimension)
    ildjs = np.split(outildj, sections, dimension)
    inslcs = [[NDSlice.new(inval, ildj_)]
              for inval, ildj_ in zip(invals, ildjs)]
    incells = [InverseAndILDJ(old_incell.aval, inslc)
               for old_incell, inslc in zip(incells, inslcs, strict=True)]
  return incells, outcells, None
ildj_registry[lax.concatenate_p] = concatenate_ildj


def tie_all_ildj(incells, outcells, **params):
  del params
  new_cells = [
      incell.join(outcell)
      for incell, outcell in zip(incells, outcells, strict=True)
  ]
  return new_cells, new_cells, None


ildj_registry[primitive.tie_all_p] = tie_all_ildj


def sow_ildj(incells, outcells, **params):
  if all(incell.top() for incell in incells) and all(
      not outcell.top() for outcell in outcells):
    # In forward evaluation mode, we want to still sow the values.
    invals = [incell.val for incell in incells]
    outvals = harvest.sow_p.bind(*invals, **params)
    new_outcells = [InverseAndILDJ.new(outval) for outval in outvals]
    return incells, new_outcells, None
  return tie_all_ildj(incells, outcells, **params)


ildj_registry[harvest.sow_p] = sow_ildj


def reshape_ildj(incells, outcells, **params):
  """InverseAndILDJ rule for reshape primitive."""
  incell, = incells
  outcell, = outcells
  if incell.top() and not outcell.top():
    return incells, [InverseAndILDJ.new(lax.reshape_p.bind(
        incell.val, **params
    ))], None
  elif outcell.top() and not incell.top():
    val = outcell.val
    new_incells = [
        InverseAndILDJ.new(np.reshape(val, incell.aval.shape))
    ]
    return new_incells, outcells, None
  return incells, outcells, None
ildj_registry[lax.reshape_p] = reshape_ildj


def expit_ildj(y):
  x = jax.scipy.special.logit(y)
  return jax.nn.softplus(-x) + jax.nn.softplus(x)


def logit_ildj(y):
  return -jax.nn.softplus(-y) - jax.nn.softplus(y)


def softplus_inv(y):
  return np.log(-np.expm1(-y)) + y


def softplus_ildj(y):
  return -np.log1p(-np.exp(-y))


def convert_element_type_ildj(incells, outcells, *, new_dtype, **params):
  """InverseAndILDJ rule for convert_element_type primitive."""
  incell, = incells
  outcell, = outcells
  if incell.top() and not outcell.top():
    outcells = [InverseAndILDJ.new(lax.convert_element_type_p.bind(
        incell.val, new_dtype=new_dtype, **params))]
  elif outcell.top() and not incell.top():
    val = outcell.val
    incells = [
        InverseAndILDJ.new(lax.convert_element_type_p.bind(
            val, new_dtype=incell.aval.dtype, **params))]
  return incells, outcells, None
ildj_registry[lax.convert_element_type_p] = convert_element_type_ildj


# Monkey patching JAX so we can define custom, more numerically stable inverses.
jax.scipy.special.expit = custom_inverse(jax.scipy.special.expit)
jax.scipy.special.logit = custom_inverse(jax.scipy.special.logit)
jax.nn.sigmoid = jax.scipy.special.expit
jax.nn.softplus = custom_inverse(jax.nn.softplus)
jax.scipy.special.expit.def_inverse_unary(f_inv=jax.scipy.special.logit,
                                          f_ildj=expit_ildj)
jax.scipy.special.logit.def_inverse_unary(f_inv=jax.scipy.special.expit,
                                          f_ildj=logit_ildj)
jax.nn.softplus.def_inverse_unary(f_inv=softplus_inv,
                                  f_ildj=softplus_ildj)

