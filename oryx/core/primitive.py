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

"""Module for higher order primitives."""
import itertools as it
from typing import Callable

from jax import api_util
from jax import tree_util
from jax import util as jax_util
from jax._src import core as jax_core
from jax.extend import linear_util as lu
from jax.interpreters import ad
from jax.interpreters import batching
from jax.interpreters import mlir
from jax.interpreters import partial_eval as pe

from oryx.core import trace_util

__all__ = [
    'HigherOrderPrimitive',
    'FlatPrimitive',
    'call_bind',
    'tie_all',
    'tie_in'
]

safe_map = jax_core.safe_map

custom_batch_rules = {}
hop_transformation_rules = {}
initial_transformation_rules = {}


def register_hop_transformation_rule(name: str, register_func: Callable[...,
                                                                        None]):
  hop_transformation_rules[name] = register_func


def register_initial_transformation_rule(name: str,
                                         register_func: Callable[..., None]):
  initial_transformation_rules[name] = register_func


class HigherOrderPrimitive(jax_core.CallPrimitive):
  """A primitive that appears in traces through transformations.

  In JAX, when functions composed of primitives are traced,
  only the primitives appear in the trace. A HigherOrderPrimitive (HOP)
  can be bound to a function using `call_bind`, which
  traces the function and surfaces its Jaxpr
  in the trace in the HOP's params.

  A HOP appears in the traces of transformed functions. Specifically,
  unlike `jax.custom_transforms` functions, which do not
  appear in a trace after a transformation like `jax.grad` or `jax.vmap`
  is applied, a HOP will create another HOP to appear in the trace
  after transformation, bound to the transformed function.
  """

  def __init__(self, name):
    super(HigherOrderPrimitive, self).__init__(name)
    self.multiple_results = True
    for register_func in hop_transformation_rules.values():
      register_func(self)

  def impl(self, f, *args, **params):
    del params
    with jax_core.eval_context():
      return f.call_wrapped(*args)

  def subcall(self, name):
    return self.__class__(f'{self.name}/{name}')


def hop_transpose_rule(prim):

  def rule(*args, **kwargs):
    return ad.call_transpose(prim.subcall('transpose'), *args, **kwargs)

  ad.primitive_transposes[prim] = rule
  return rule


register_hop_transformation_rule('transpose', hop_transpose_rule)


def hop_lowering(prim):

  def rule(ctx, *args, backend, name, call_jaxpr, **_params):
    return mlir._call_lowering(  # pylint: disable=protected-access
        name, name, call_jaxpr, backend,
        ctx.module_context, ctx.avals_in, ctx.avals_out, *args)

  mlir.register_lowering(prim, rule)
  return rule

register_hop_transformation_rule('mlir', hop_lowering)


def batch_fun(fun: lu.WrappedFun, axis_data, in_dims):
  tag = jax_core.TraceTag()
  in_dims = in_dims() if callable(in_dims) else in_dims
  batched, out_dims = batching.batch_subtrace(fun, tag, axis_data, in_dims)
  return batched, out_dims


class FlatPrimitive(jax_core.Primitive):
  """Contains default implementations of transformations."""

  def __init__(self, name):
    super(FlatPrimitive, self).__init__(name)
    self.multiple_results = True

    def _abstract(*flat_avals, **params):
      return pe.abstract_eval_fun(self.impl, *flat_avals, **params)

    self.def_abstract_eval(_abstract)

    def _jvp(primals, tangents, **params):
      primals_out, tangents_out = ad.jvp(
          lu.wrap_init(self.impl, params)).call_wrapped(primals, tangents)

      return primals_out, tangents_out

    ad.primitive_jvps[self] = _jvp

    def _batch(axis_data, args, dims, **params):
      batched, out_dims = batch_fun(
          lu.wrap_init(self.impl, params), axis_data, dims)
      return batched.call_wrapped(*args), out_dims()

    batching.fancy_primitive_batchers[self] = _batch

    def _mlir(c, *mlir_args, **params):
      lowering = mlir.lower_fun(self.impl, multiple_results=True)
      return lowering(c, *mlir_args, **params)

    mlir.register_lowering(self, _mlir)


def call_bind(prim, **params):
  """Binds a primitive to a function call."""

  def bind(f):
    """Wraps a function to be bound to a primitive, keeping track of Pytree information."""

    def wrapped(*args, **kwargs):
      """Runs a function and binds it to a call primitive."""
      fun = lu.wrap_init(f)
      flat_args, in_tree = tree_util.tree_flatten((args, kwargs))
      flat_fun, out_tree = api_util.flatten_fun(fun, in_tree)
      out_tree_dest = None
      out = prim.bind(
          flat_fun,
          *flat_args,
          num_args=len(flat_args),
          name=f.__name__,
          in_tree=in_tree,
          out_tree=lambda: out_tree_dest,
          **params)
      out_tree_dest = out_tree()
      return tree_util.tree_unflatten(out_tree_dest, out)

    return wrapped

  return bind


def initial_style_bind(prim, **params):
  """Binds a primitive to a function call."""

  def bind(f):
    """Wraps a function to be bound to a primitive, keeping track of Pytree information."""

    def wrapped(*args, **kwargs):
      """Runs a function and binds it to a call primitive."""
      jaxpr, (in_tree, out_tree) = trace_util.stage(
          f, dynamic=True)(*args, **kwargs)
      flat_args = tree_util.tree_leaves(args)
      outs = prim.bind(
          *it.chain(jaxpr.literals, flat_args),
          jaxpr=jaxpr.jaxpr,
          in_tree=in_tree,
          out_tree=out_tree,
          num_consts=len(jaxpr.literals),
          **params)
      return tree_util.tree_unflatten(out_tree, outs)

    return wrapped

  return bind


class InitialStylePrimitive(FlatPrimitive):
  """Contains default implementations of transformations."""

  def __init__(self, name):
    super().__init__(name)

    def fun_impl(*args, **params):
      consts, args = jax_util.split_list(args, [params['num_consts']])
      return jax_core.eval_jaxpr(params['jaxpr'], consts, *args)

    self.def_impl(fun_impl)
    for register_func in initial_transformation_rules.values():
      register_func(self)

  def subcall(self, name):
    return InitialStylePrimitive(f'{self.name}/{name}')


tie_all_p = jax_core.Primitive('tie_all')
tie_all_p.multiple_results = True
tie_all_p.def_impl(lambda *args: args)
tie_all_p.def_abstract_eval(lambda *args: safe_map(  # pylint: disable=g-long-lambda
    jax_core.raise_to_shaped, args))

mlir.register_lowering(tie_all_p, lambda c, *args: args)


def _tie_all_batch_rule(batched_args, batch_dims):
  return batched_args, batch_dims


def _tie_all_transpose(cts_in, *args, **params):
  del args, params
  return cts_in


def _tie_all_jvp(primals, tangents, **params):
  del params
  return primals, tangents


ad.primitive_jvps[tie_all_p] = _tie_all_jvp
ad.primitive_transposes[tie_all_p] = _tie_all_transpose
batching.primitive_batchers[tie_all_p] = _tie_all_batch_rule


def tie_all(*args):
  """An identity function that ties arguments together in a JAX trace."""
  flat_args, in_tree = tree_util.tree_flatten(args)
  if len(flat_args) <= 1:
    return args
  out = tie_all_p.bind(*flat_args)
  return tree_util.tree_unflatten(in_tree, out)


def tie_in(x, y):
  """A reimplementation of `jax.tie_in` that handles pytrees."""
  return tie_all(x, y)[1]
