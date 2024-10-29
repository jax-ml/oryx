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

"""Module for the harvest transformation.

This module contains a general-purpose set of tools for transforming
functions with a specific side-effect mechanism into pure functions. The names
of the transformations in this module are inspired by the Sow/Reap mechanism in
Mathematica.

The harvest module exposes two main functions: `sow` and `harvest`. `sow` is
used to tag values and `harvest` can inject values into functions or pull out
tagged values.

`harvest` is a very general purpose transformation purely focused on converting
functions that have special side-effects (defined using `sow`) and
"functionalizing" them. Specifically, a function
`f :: (x: X) -> Y` has a set of defined intermediates, or `Sows`. This set
can be divided into intermediates you are "collecting" and intermediates you are
"injecting", or `Reaps` and `Plants` respectively. Functionalizing
`f` now gives you `harvest(f) :: (plants: Plants, x: X) -> Tuple[Y, Reaps]`.
Generally, most users will not need to use `harvest` directly, but will use
wrappers around it.

## `sow`

`sow` is the function used to tag values in a function. It takes in a single
positional argument, `value`, which is returned as an output, so `sow` outside
of a tracing context behaves like the identity function, i.e.
`sow(x, ...) == x`. It also takes in two mandatory keyword arguments,
`tag` and `name`. `tag` is a string used to namespace intermediate values in a
function. For example, some intermediates may be useful for probabilistic
programming (samples), and others may be useful to logging (summaries). The tag
enables `harvest` to interact with only one set of intermediates at a time.
The `name` is a string that describes the value you are `sow`-ing. Eventually,
when calling `harvest` on a function, the `name` is used as the identifier
for the intermediate value.

Finally, `sow` takes in an optional string keyword argument `mode`, which is by
default set to `'strict'`. The `mode` of a `sow` describes how it behaves when
the same name appears multiple times. In "strict" mode, `sow` will error if the
same `(tag, name)` appears more than once. Another option is `'append'`, in
which all sows of the same name will be appended into a growing array. Finally,
there is `'clobber'`, where only the final sown value for a given `(tag, name)`
will be returned. The final optional argument for `sow` is `key`, which will
automatically be tied-in to the output of `sow` to introduce a fake
data-dependence. By default, it is `None`.

## `sow_cond`

`sow_cond` is a variant of `sow`, that takes an additional positional argument,
`pred`. It supports a single `mode` `'cond_clobber'`, which is like `clobber`,
but sows values conditionally on `pred`, falling back on zeros if no sow took
place. This allows reaping values from loop iterations besides the final one.

## `harvest`

`harvest` is a function transformation that augments the behaviors of `sow`s
in the function body. Recall, that by default, `sow`s act as identity functions
and do not affect the semantics of a function. Harvesting `f` produces a
function that can take advantage of `sow`s present in its execution. `harvest`
is a function that takes in a function `f` and a string `tag`. `harvest` will
only interact with `sow`s whose tag matches the input `tag`. The returned
function can interact with the `sow`s in the function body in either of two
ways. The first is via "injection", where intermediate values in the function
values can be overridden. `harvest(f)` takes in an additional initial argument,
`plants`, a dictionary mapping names to values. Each name in `plants` should
correspond to a `sow` in `f`, and while running `harvest(f)` rather than using
the value at runtime for the `sow`, we substitute in the value from the `plants`
dictionary. The other way in which `harvest(f)` interacts with `sow`s is that
if it encounters a `sow` whose tag matches and whose name is *not* in
`plants`, it will add the output of the `sow` to a dictionary mapping the sow
name to its output, called `reaps`. The `reaps` dictionary, at the end of
`harvest(f)`'s execution, will contain the outputs of all `sow`s whose values
were not injected, or "planted."

The general convention is that, for any given execution of
`harvest(f, tag=tag)`, there will be *no more remaining sows* of the given tag
if the function were to be reharvested, i.e. if we were to nest harvests with
the same tag `harvest(harvest(f, tag='some_tag'), tag='some_tag')`, the outer
harvest would have nothing to plant or to reap.

## Examples:

#### Using `sow` and `harvest`
```python
def f(x):
  y = sow(x + 1., tag='intermediate', name='y')
  return y + 1.

# Injecting, or "planting" a value for `y`.
harvest(f, tag='intermediate')({'y': 0.}, 1.)  # ==> (1., {})
harvest(f, tag='intermediate')({'y': 0.}, 5.)  # ==> (1., {})

# Collecting , or "reaping" the value of `y`.
harvest(f, tag='intermediate')({}, 1.)  # ==> (3., {'y': 2.})
harvest(f, tag='intermediate')({}, 5.)  # ==> (7., {'y': 6.})
```

#### Using `reap` and `plant`.
`reap` and `plant` are simple wrappers around `harvest`. `reap` only pulls
intermediate values without injecting, and `plant` only injects values without
collecting intermediate values.

```python
def f(x):
  y = sow(x + 1., tag='intermediate', name='y')
  return y + 1.

# Injecting, or "planting" a value for `y`.
plant(f, tag='intermediate')({'y': 0.}, 1.)  # ==> 1.
plant(f, tag='intermediate')({'y': 0.}, 5.)  # ==> 1.

# Collecting , or "reaping" the value of `y`.
reap(f, tag='intermediate')(1.)  # ==> {'y': 2.}
reap(f, tag='intermediate')(5.)  # ==> {'y': 6.}
```

#### Sharp edges

* `harvest` has undefined semantics under autodifferentiation. If a function
  you're taking the gradient of has a `sow`, it might produce unintuitive
  results when harvested. To better control gradient semantics, you can use
  `jax.custom_jvp` or `jax.custom_vjp`. The current implementation sows primals
  and tangents in the JVP but ignore cotangents in the VJP. These particular
  semantics are subject to change.
* Planting values into a `pmap` is partially working. Harvest tries to plant all
  the values, assuming they have a leading map dimension.
"""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import functools
from typing import Any, Callable, Dict, FrozenSet, Hashable, Iterable, List, Optional, Tuple, Union

from jax import api_util
from jax import lax
from jax import tree_util
from jax import util as jax_util
from jax._src import ad_checkpoint
from jax._src import core as jax_core
from jax._src import effects
from jax._src import pjit
from jax._src import sharding_impls
from jax._src.lax import control_flow as lcf
from jax.experimental import shard_map
import jax.extend.linear_util as lu
from jax.interpreters import ad
from jax.interpreters import batching
from jax.interpreters import mlir
from jax.interpreters import partial_eval as pe
import jax.numpy as jnp


__all__ = [
    'HarvestTrace',
    'call_and_reap',
    'harvest',
    'nest',
    'plant',
    'reap',
    'sow',
]

Value = Any

sow_p = jax_core.Primitive('sow')
sow_p.multiple_results = True


class SowEffect(effects.Effect):
  __repr__ = lambda _: 'Sow'


sow_effect = SowEffect()

effects.remat_allowed_effects.add_type(SowEffect)
effects.control_flow_allowed_effects.add_type(SowEffect)
effects.lowerable_effects.add_type(SowEffect)


@sow_p.def_impl
def _sow_impl(*args, **_):
  return args


@sow_p.def_effectful_abstract_eval
def _sow_abstract_eval(*avals, **_):
  return avals, {sow_effect}


def _sow_jvp(primals, tangents, **kwargs):
  out_primals = sow_p.bind(*primals, **kwargs)
  return out_primals, tangents


ad.primitive_jvps[sow_p] = _sow_jvp


def _sow_transpose(cts_in, *args, **kwargs):
  del args, kwargs
  return cts_in


ad.primitive_transposes[sow_p] = _sow_transpose


def _sow_batch_rule(batched_args, batch_dims, **params):
  outs = sow_p.bind(*batched_args, **params)
  return outs, batch_dims


batching.primitive_batchers[sow_p] = _sow_batch_rule
mlir.register_lowering(sow_p, lambda c, *args, **kw: args)


def sow(value, *, tag: Hashable, name: str, mode: str = 'strict', key=None):
  """Marks a value with a name and a tag.

  Args:
    value: A JAX value to be tagged and named.
    tag: a string representing the tag of the sown value.
    name: a string representing the name to sow the value with.
    mode: The mode by which to sow the value. There are three options: 1.
      `'strict'` - if another value is sown with the same name and tag in the
      same context, harvest will throw an error. 2. `'clobber'` - if another is
      value is sown with the same name and tag, it will replace this value 3.
      `'append'` - sown values of the same name and tag are appended to a
      growing list. Append mode assumes some ordering on the values being sown
      defined by data-dependence.
    key: an optional JAX value that will be tied into the sown value.

  Returns:
    The original `value` that was passed in, or a planted value.
  """
  if mode == 'cond_clobber':
    raise ValueError("For 'cond_clobber' mode, use `sow_cond`.'")
  with jax_core.take_current_trace() as trace:
    return _sow(trace, value, tag=tag, name=name, mode=mode, key=key)


def sow_cond(
    value,
    pred,
    *,
    tag: Hashable,
    name: str,
    mode: str = 'cond_clobber',
    key=None,
):
  """Marks a value, alongside a predicate, with a name and a tag.

  The predicate determines whether the value is to be clobbered in this loop
  iteration -- if it's reaped but never clobbered, the value will be full of
  zeros.

  Args:
    value: A JAX value to be tagged and named.
    pred: Whether to sow the value.
    tag: a string representing the tag of the sown value.
    name: a string representing the name to sow the value with.
    mode: The mode by which to sow the value. There are three options: 1.
      `'strict'` - if another value is sown with the same name and tag in the
      same context, harvest will throw an error. 2. `'clobber'` - if another is
      value is sown with the same name and tag, it will replace this value 3.
      `'append'` - sown values of the same name and tag are appended to a
      growing list. Append mode assumes some ordering on the values being sown
      defined by data-dependence.
    key: an optional JAX value that will be tied into the sown value.

  Returns:
    The original `value` that was passed in, or a planted value.
  """
  if mode != 'cond_clobber':
    raise ValueError("`sow_cond` only supports 'cond_clobber' mode.")
  with jax_core.take_current_trace() as trace:
    return _sow(trace, value, tag=tag, name=name,
                mode=mode, key=key, pred=pred)[0]


def _sow(trace, value, *, tag, name, mode, key=None, pred=None):
  del key
  assert (pred is not None) == (mode == 'cond_clobber')
  if pred is not None:
    value = value, pred
  flat_args, in_tree = tree_util.tree_flatten(value)
  out_flat = sow_p.bind_with_trace(
      trace, flat_args,
      dict(name=name, tag=tag, mode=mode, tree=in_tree))
  return tree_util.tree_unflatten(in_tree, out_flat)


nest_p = jax_core.CallPrimitive('nest')


def _nest_impl(f, *args, **_):
  return f.call_wrapped(*args)


nest_p.def_impl(_nest_impl)


def _nest_lowering(ctx, *args, name, call_jaxpr, scope, **_):
  return mlir.core_call_lowering(
      ctx,
      *args,
      name=jax_util.wrap_name(name, f'nest[{scope}]'),
      call_jaxpr=call_jaxpr)


mlir.register_lowering(nest_p, _nest_lowering)


def _nest_transpose_rule(*args, **kwargs):
  return ad.call_transpose(nest_p, *args, **kwargs)


ad.primitive_transposes[nest_p] = _nest_transpose_rule


def nest(f, *, scope: str):
  """Wraps a function to create a new scope for harvested values.

  Harvested values live in one dynamic name scope (for a particular tag),
  and in strict mode, values with the same name cannot be collected or injected
  more than once. `nest(f, scope=[name])` will take all tagged values in `f` and
  put them into a nested dictionary with key `[name]`. This enables having
  duplicate names in one namespace provided they are in different scopes. This
  is different from using a separate tag to namespace, as it enables creating
  nested/hierarchical structure within a single tag's namespace.

  Example:
  ```python
  def foo(x):
    return sow(x, tag='test', name='x')
  harvest(foo, tag='test')({}, 1.)  # (1., {'x': 1.})
  harvest(nest(foo, scope='a'), tag='test')({}, 1.)  # (1., {'a': {'x': 1.}})
  ```

  Args:
    f: a function to be transformed
    scope: a string that will act as the parent scope of all values tagged in
      `f`.

  Returns:
    A semantically identical function to `f`, but when harvested, uses nested
    values according to the input scope.
  """

  def wrapped(*args, **kwargs):
    fun = lu.wrap_init(f, kwargs)
    flat_args, in_tree = tree_util.tree_flatten(args)
    flat_fun, out_tree = api_util.flatten_fun_nokwargs(fun, in_tree)
    out_flat = nest_p.bind(
        flat_fun,
        *flat_args,
        scope=scope,
        name=getattr(f, '__name__', '<no name>'))
    return tree_util.tree_unflatten(out_tree(), out_flat)

  return wrapped


class HarvestTrace(jax_core.Trace):
  """An evaluating trace that dispatches to a dynamic context."""

  def __init__(self, parent_trace, context):
    self.parent_trace = parent_trace
    self.context = context

  def process_primitive(
      self, primitive: jax_core.Primitive, vals: List[Any],
      params: Dict[str, Any]) -> Union[Any, List[Any]]:
    custom_rule = self.context.get_custom_rule(primitive)
    if custom_rule:
      return custom_rule(self, *vals, **params)
    return self.default_process_primitive(primitive, vals, params)

  def default_process_primitive(
      self, primitive: jax_core.Primitive, vals: List[Any],
      params: Dict[str, Any]) -> Union[Any, List[Any]]:
    if primitive is sow_p:
      with jax_core.set_current_trace(self.parent_trace):
        return self.context.process_sow(*vals, **params)
    outvals = primitive.bind_with_trace(self.parent_trace, vals, params)
    if not primitive.multiple_results:
      outvals = [outvals]
    if primitive.multiple_results:
      return outvals
    return outvals[0]

  def process_call(self, call_primitive: jax_core.Primitive, f: Any,
                   vals: List[Any], params: Dict[str, Any]):
    context = self.context
    if call_primitive is nest_p:
      return context.process_nest(self, f, *vals, **params)
    return context.process_higher_order_primitive(self, call_primitive, f,
                                                  vals, params, False)

  def process_map(self, call_primitive: jax_core.Primitive, f: Any,
                  vals: List[Any], params: Dict[str, Any]):
    return self.context.process_higher_order_primitive(
        self, call_primitive, f, vals, params, True)

  def process_custom_jvp_call(self, primitive, fun, jvp, vals, *,
                              symbolic_zeros):
    return self.context.process_custom_jvp_call(
        self, primitive, fun, jvp, vals, symbolic_zeros=symbolic_zeros)

  def process_shard_map(self, primitive, f, vals, **params):
    return primitive.bind_with_trace(self.parent_trace, (f, *vals), params)

  def process_custom_vjp_call(self, primitive, fun, fwd, bwd, vals,
                              out_trees, symbolic_zeros):
    return self.context.process_custom_vjp_call(
        self, primitive, fun, fwd, bwd, vals, out_trees, symbolic_zeros)


@dataclasses.dataclass(frozen=True)
class HarvestSettings:
  """Contains the settings for a HarvestTrace."""
  tag: Hashable
  blocklist: FrozenSet[str]
  allowlist: Union[FrozenSet[str], None]
  exclusive: bool


@dataclasses.dataclass
class HarvestContext:
  """A context that handles `sow`s and `nest`s in a `HarvestTrace`."""
  settings: HarvestSettings

  def process_sow(self, *values, name, tag, mode, tree):
    """Handles a `sow` primitive in a `HarvestTrace`."""
    if mode not in {'strict', 'append', 'clobber', 'cond_clobber'}:
      raise ValueError(f'Invalid mode: {mode}')
    if tag != self.settings.tag:
      if self.settings.exclusive:
        return values
      return sow_p.bind(*values, name=name, tag=tag, tree=tree, mode=mode)
    if (self.settings.allowlist is not None and
        name not in self.settings.allowlist):
      return values
    if name in self.settings.blocklist:
      return values
    return self.handle_sow(*values, name=name, tag=tag, tree=tree, mode=mode)

  def get_custom_rule(self, primitive):
    raise NotImplementedError

  def handle_sow(self, *values, name, tag, mode, tree):
    raise NotImplementedError

  def process_nest(self, trace, f, *vals, scope, name):
    raise NotImplementedError

  def process_higher_order_primitive(self, trace: HarvestTrace,
                                     call_primitive: jax_core.Primitive, f: Any,
                                     vals: List[Any],
                                     params: Dict[str, Any], is_map: bool):
    raise NotImplementedError

  def process_custom_jvp_call(self, trace, primitive, fun, jvp, vals, *,
                              symbolic_zeros):
    raise NotImplementedError

  def process_custom_vjp_call(self, trace, primitive, fun, fwd, bwd, vals,
                              out_trees, symbolic_zeros):
    raise NotImplementedError


reap_custom_rules = {}


@dataclasses.dataclass
class Reap:
  value: Any
  pred: Any
  metadata: Dict[str, Any]


@dataclasses.dataclass
class ReapContext(HarvestContext):
  """Contains the settings and storage for the current trace in the stack."""
  settings: HarvestSettings
  reaps: Dict[str, Reap]

  def get_custom_rule(self, primitive):
    return reap_custom_rules.get(primitive)

  def handle_sow(self, *values, name, tag, tree, mode):
    """Stores a sow in the reaps dictionary."""
    del tag
    if prev_reap := self.reaps.get(name):
      if {mode, prev_reap.metadata['mode']} - {'clobber', 'cond_clobber'}:
        raise ValueError(f'Variable has already been reaped: {name}')
    avals = tree_util.tree_unflatten(
        tree,
        [jax_core.raise_to_shaped(jax_core.get_aval(v)) for v in values])
    vals = tree_util.tree_unflatten(tree, values)
    pred = None
    if mode == 'cond_clobber':
      avals, _ = avals
      vals, pred = vals
      if prev_reap:
        if prev_reap.metadata['mode'] == 'clobber':
          vals = lax.cond(pred, lambda: vals, lambda: prev_reap.value)
        elif prev_reap.metadata['mode'] == 'cond_clobber':
          vals = lax.cond(pred, lambda: vals, lambda: prev_reap.value)
          pred = pred | prev_reap.pred
    metadata = dict(mode=mode, aval=avals)
    self.reaps[name] = Reap(vals, pred, metadata)
    return values

  def reap_higher_order_primitive(self, trace, call_primitive, f, vals,
                                  params, is_map):
    """Wraps the inner function with a reap trace."""
    name = jax_util.wrap_name(params.pop('name', f.__name__), 'reap')
    f, aux = reap_eval(f, self.settings)

    if is_map:
      out_axes_thunk = params['out_axes_thunk']

      @jax_util.as_hashable_function(closure=('harvest', out_axes_thunk))
      def new_out_axes_thunk():
        out_axes = out_axes_thunk()
        assert all(out_axis == 0 for out_axis in out_axes)
        out_tree, _ = aux()
        return (0,) * out_tree.num_leaves

      params = dict(params, out_axes_thunk=new_out_axes_thunk)
    out_flat = call_primitive.bind_with_trace(
        trace.parent_trace, (f, *vals), dict(params, name=name))
    out_tree, metadata = aux()
    out_vals, reaps, preds = tree_util.tree_unflatten(out_tree, out_flat)
    return out_vals, reaps, preds, metadata

  def process_nest(self, trace, f, *vals, scope, name, **params):
    out_tracers, reap_tracers, _, _ = self.reap_higher_order_primitive(
        trace, nest_p, f, vals, dict(params, name=name, scope=scope), False)
    tag = self.settings.tag
    if reap_tracers:
      flat_reap_tracers, reap_tree = tree_util.tree_flatten(reap_tracers)
      trace.process_primitive(
          sow_p, flat_reap_tracers,
          dict(name=scope, tag=tag, tree=reap_tree, mode='strict'))
    return out_tracers

  def process_higher_order_primitive(self, trace, call_primitive, f, vals,
                                     params, is_map):
    out_tracers, reap_tracers, pred_tracers, metadata = (
        self.reap_higher_order_primitive(
            trace, call_primitive, f, vals, params, is_map
        )
    )
    tag = self.settings.tag
    for k, v in reap_tracers.items():
      if metadata[k]['mode'] == 'cond_clobber':
        v = (v, pred_tracers[k])
      flat_reap_tracers, reap_tree = tree_util.tree_flatten(v)
      trace.process_primitive(
          sow_p, flat_reap_tracers,
          dict(name=k, tag=tag, tree=reap_tree, mode=metadata[k]['mode']))
    return out_tracers

  def process_custom_jvp_call(self, trace, primitive, fun, jvp, vals, *,
                              symbolic_zeros):
    fun, aux1 = reap_eval(fun, self.settings)

    @lu.transformation_with_aux
    def _jvp_subtrace(context, *args):
      with harvest_trace(context):
        outs = yield args, {}
        yield outs, (None, None)

    jvp, aux2 = _jvp_subtrace(jvp, self)
    out_flat = primitive.bind_with_trace(
        trace.parent_trace, (fun, jvp, *vals),
        dict(symbolic_zeros=symbolic_zeros))
    fst, (out_tree, metadata) = lu.merge_linear_aux(aux1, aux2)
    if fst:
      out, reaps, preds = tree_util.tree_unflatten(out_tree, out_flat)
      tag = self.settings.tag
      for k, v in reaps.items():
        if metadata[k]['mode'] == 'cond_clobber':
          v = (v, preds[k])
        flat_reap_tracers, reap_tree = tree_util.tree_flatten(v)
        trace.process_primitive(
            sow_p, flat_reap_tracers,
            dict(name=k, tag=tag, tree=reap_tree, mode=metadata[k]['mode']))
    else:
      out = out_flat
    return out

  def process_custom_vjp_call(self, trace, primitive, fun, fwd, bwd, vals,
                              out_trees, symbolic_zeros):
    fun, aux1 = reap_eval(fun, self.settings)

    @lu.transformation_with_aux
    def _fwd_subtrace(context, *args):
      with harvest_trace(context):
        outs = yield args, {}
        yield outs, (None, None)

    fwd, aux2 = _fwd_subtrace(fwd, self)
    bwd_ = reap_function(lu.wrap_init(bwd), self.settings, True)
    bwd = reap_wrapper_drop_aux(bwd_).call_wrapped
    out_flat = primitive.bind_with_trace(
        trace.parent_trace, (fun, fwd, bwd, *vals),
        dict(out_trees=out_trees, symbolic_zeros=symbolic_zeros))
    fst, (out_tree, metadata) = lu.merge_linear_aux(aux1, aux2)
    if fst:
      out, reaps, preds = tree_util.tree_unflatten(out_tree, out_flat)
      tag = self.settings.tag
      for k, v in reaps.items():
        if metadata[k]['mode'] == 'cond_clobber':
          v = (v, preds[k])
        flat_reap_tracers, reap_tree = tree_util.tree_flatten(v)
        trace.process_primitive(
            sow_p, flat_reap_tracers,
            dict(name=k, tag=tag, tree=reap_tree, mode=metadata[k]['mode']))
    else:
      out = out_flat
    return out


@lu.transformation
def reap_function(settings: HarvestSettings,
                  return_metadata: bool, args: Iterable[Any]):
  """A function transformation that returns reap values and predicates."""
  context = ReapContext(settings, {})
  with harvest_trace(context):
    out_values = yield args, {}
    reap_values = tree_util.tree_map(lambda x: x.value, context.reaps)
    pred_values = tree_util.tree_map(lambda x: x.pred, context.reaps)
    reap_metadata = tree_util.tree_map(lambda x: x.metadata, context.reaps)
  if return_metadata:
    out = (out_values, reap_values, pred_values, reap_metadata)
  else:
    out = (out_values, reap_values, pred_values)
  yield out


def reap_eval(
    f: lu.WrappedFun,
    settings: HarvestSettings) -> Tuple[lu.WrappedFun, Callable[[], Any]]:
  f = reap_function(f, settings, True)
  return reap_wrapper(f)


@lu.transformation_with_aux
def reap_wrapper(*args):
  out, reaps, preds, metadata = yield (args,), {}
  out_flat, out_tree = tree_util.tree_flatten((out, reaps, preds))
  yield out_flat, (out_tree, metadata)


@lu.transformation
def reap_wrapper_drop_aux(*args):
  out, reaps, preds, _ = yield (args,), {}
  out_flat, _ = tree_util.tree_flatten((out, reaps, preds))
  yield out_flat


def call_and_reap(f,
                  *,
                  tag: Hashable,
                  allowlist: Optional[Iterable[str]] = None,
                  blocklist: Iterable[str] = frozenset(),
                  exclusive: bool = False):
  """Transforms a function into one that additionally returns its sown values.

  Args:
    f: a function to be transformed.
    tag: a string tag; only sown values with `tag` will be reaped.
    allowlist: an optional sequence of string names, which if provided will
      enforce that only sows with names in the allowlist will be reaped.
    blocklist: an optional sequence of string names, which if provided will
      enforce that only no sows with names in the blocklist will be reaped.
    exclusive: determines whether or not to execute in "exclusive" mode where
      other tags are removed during execution.

  Returns:
    A new function that executes the original and returns its sown values as
    an additional return value.
  """

  def wrapped(*args, **kwargs):
    out, reaps, preds = _call_and_reap(
        f,
        tag=tag,
        allowlist=allowlist,
        blocklist=blocklist,
        exclusive=exclusive,
    )(*args, **kwargs)

    def select_from_pred(pred, value):
      if pred is None:
        return value
      else:
        return tree_util.tree_map(
            lambda leaf: lax.select(pred, leaf, lax.zeros_like_array(leaf)),
            value,
        )

    return out, tree_util.tree_map(
        select_from_pred, preds, reaps, is_leaf=lambda x: x is None
    )

  return wrapped


def _call_and_reap(f, *, tag, allowlist, blocklist, exclusive):
  """Like call_and_reap but including predicates."""
  blocklist = frozenset(blocklist)
  if allowlist is not None:
    allowlist = frozenset(allowlist)
  settings = HarvestSettings(tag, blocklist, allowlist, exclusive)

  def wrapped(*args, **kwargs):
    fun = lu.wrap_init(f, kwargs)
    flat_args, in_tree = tree_util.tree_flatten(args)
    flat_fun, out_tree = api_util.flatten_fun_nokwargs(fun, in_tree)
    flat_fun = reap_function(flat_fun, settings, False)
    out_flat, reaps, preds = flat_fun.call_wrapped(flat_args)
    return tree_util.tree_unflatten(out_tree(), out_flat), reaps, preds

  return wrapped


def reap(f,
         *,
         tag: Hashable,
         allowlist: Optional[Iterable[str]] = None,
         blocklist: Iterable[str] = frozenset(),
         exclusive: bool = False):
  """Transforms a function into one that returns its sown values.

  Args:
    f: a function to be transformed.
    tag: a string tag; only sown values with `tag` will be reaped.
    allowlist: an optional sequence of string names, which if provided will
      enforce that only sows with names in the allowlist will be reaped.
    blocklist: an optional sequence of string names, which if provided will
      enforce that only no sows with names in the blocklist will be reaped.
    exclusive: determines whether or not to execute in "exclusive" mode where
      other tags are removed during execution.

  Returns:
    A new function that executes the original and returns its sown values.
  """

  def wrapped(*args, **kwargs):
    return call_and_reap(
        f,
        tag=tag,
        allowlist=allowlist,
        blocklist=blocklist,
        exclusive=exclusive)(*args, **kwargs)[1]

  return wrapped


@lu.transformation_with_aux
def _reap_metadata_wrapper(*args):
  out, reaps, preds, metadata = yield (args,), {}
  yield (out, reaps, preds), metadata


def _get_harvest_metadata(closed_jaxpr, settings, *args):
  """Probes a jaxpr for metadata like its sown values."""
  fun = lu.wrap_init(jax_core.jaxpr_as_fun(closed_jaxpr))

  settings = HarvestSettings(settings.tag, settings.blocklist,
                             settings.allowlist, True)
  fun = reap_function(fun, settings, True)
  fun, aux = _reap_metadata_wrapper(fun)
  flat_args, in_tree = tree_util.tree_flatten(args)
  flat_fun, out_tree = api_util.flatten_fun_nokwargs(fun, in_tree)
  in_avals = jax_util.safe_map(
      lambda a: jax_core.raise_to_shaped(jax_core.get_aval(a)),
      flat_args)
  pe.trace_to_jaxpr_dynamic(flat_fun, in_avals)
  metadata = aux()
  out_tree()
  return metadata


def _update_clobber_carry(carry_reaps, carry_preds, name, val, preds, mode):
  if mode == 'cond_clobber':
    carry_reaps[name], carry_preds[name] = lax.cond(
        preds[name],
        lambda val=val: (val, True),
        lambda name=name: (carry_reaps[name], carry_preds[name]),
    )
  else:
    carry_reaps[name] = val


def _reap_scan_rule(trace: HarvestTrace, *vals, length, reverse, jaxpr,
                    num_consts, num_carry, linear, unroll, _split_transpose):
  """Reaps the body of a scan to pull out `clobber` and `append` sows."""

  const_vals, carry_vals, xs_vals = jax_util.split_list(
      vals, [num_consts, num_carry])
  _, carry_avals, xs_avals = tree_util.tree_map(
      lambda x: jax_core.get_aval(x), (const_vals, carry_vals, xs_vals))  # pylint: disable=unnecessary-lambda
  settings = trace.context.settings
  with jax_core.set_current_trace(trace.parent_trace):
    x_vals = [t[0] if hasattr(jax_core.get_aval(t), '_getitem') else t
              for t in xs_vals]
  x_avals = [jax_core.get_aval(t) for t in x_vals]
  metadata = _get_harvest_metadata(jaxpr, settings,
                                   *(const_vals + carry_vals + x_vals))

  reap_modes = collections.defaultdict(set)
  reap_carry_avals = {}
  cond_carry_avals = {}
  for name, meta in metadata.items():
    mode = meta['mode']
    aval = meta['aval']
    if mode == 'strict':
      raise ValueError(f'Cannot use strict mode for \'{name}\' inside `scan`.')
    reap_modes[mode].add(name)
    if mode == 'clobber':
      reap_carry_avals[name] = aval
      cond_carry_avals[name] = None
    if mode == 'cond_clobber':
      reap_carry_avals[name] = aval
      cond_carry_avals[name] = jax_core.raise_to_shaped(jax_core.get_aval(True))

  body_fun = jax_core.jaxpr_as_fun(jaxpr)

  reap_carry_flat_avals = tree_util.tree_leaves(
      (reap_carry_avals, cond_carry_avals)
  )

  reap_carry_in_tree = tree_util.tree_structure(
      ((carry_avals, reap_carry_avals, cond_carry_avals), xs_avals))

  def new_body(carry, x):
    carry, carry_reaps, carry_preds = carry
    all_values = const_vals + tree_util.tree_leaves((carry, x))
    out, reaps, preds = _call_and_reap(
        body_fun,
        tag=settings.tag,
        allowlist=settings.allowlist,
        blocklist=settings.blocklist,
        exclusive=settings.exclusive)(*all_values)
    carry_out, y = jax_util.split_list(out, [num_carry])
    clobber_reap_modes = reap_modes['clobber'] | reap_modes['cond_clobber']
    for name, val in reaps.items():
      if name in clobber_reap_modes:
        _update_clobber_carry(carry_reaps, carry_preds, name, val, preds, mode)
    xs_reaps = {
        name: val for name, val in reaps.items() if name in reap_modes['append']
    }
    return (carry_out, carry_reaps, carry_preds), (y, xs_reaps)

  new_body_jaxpr, consts, out_tree = lcf._initial_style_jaxpr(  # pylint: disable=protected-access
      new_body, reap_carry_in_tree,
      tuple(carry_avals + reap_carry_flat_avals + x_avals))

  with jax_core.set_current_trace(trace.parent_trace):
    dummy_reap_carry_vals = tree_util.tree_map(
        lambda x: jnp.zeros(x.shape, x.dtype),
        reap_carry_flat_avals,
    )
  out = lax.scan_p.bind_with_trace(
      trace.parent_trace,
      (consts + carry_vals + dummy_reap_carry_vals + xs_vals),
      dict(reverse=reverse,
           length=length,
           jaxpr=new_body_jaxpr,
           num_consts=len(consts),
           num_carry=len(carry_vals + dummy_reap_carry_vals),
           linear=(
               linear[:len(consts)] + (False,) * len(dummy_reap_carry_vals) +
               linear[len(consts):]),
           unroll=unroll,
           _split_transpose=_split_transpose))
  (carry_out, carry_reaps, carry_preds), (ys, ys_reaps) = (
      tree_util.tree_unflatten(out_tree, out)
  )
  for k, v in carry_reaps.items():
    mode = metadata[k]['mode']
    _sow(trace, v, tag=settings.tag, mode=mode, name=k, pred=carry_preds[k])
  for k, v in ys_reaps.items():
    mode = metadata[k]['mode']
    _sow(trace, v, tag=settings.tag, mode=mode, name=k)
  return carry_out + ys


reap_custom_rules[lcf.scan_p] = _reap_scan_rule


def _reap_while_rule(trace: HarvestTrace, *tracers, cond_jaxpr, body_jaxpr,
                     cond_nconsts, body_nconsts):
  """Reaps the body of a while loop to get the reaps of `clobber` sows."""
  cond_const_vals, body_const_vals, init_vals = jax_util.split_list(
      tracers, [cond_nconsts, body_nconsts])
  _, init_avals = tree_util.tree_map(lambda x: jax_core.get_aval(x),  # pylint: disable=unnecessary-lambda
                                     (body_const_vals, init_vals))
  settings = trace.context.settings
  body_metadata = _get_harvest_metadata(body_jaxpr, settings,
                                        *(body_const_vals + init_vals))
  reap_avals = {}
  cond_avals = collections.defaultdict(lambda: None)
  for k, meta in body_metadata.items():
    mode = meta['mode']
    if mode not in ['clobber', 'cond_clobber']:
      raise ValueError(
          f"Must use clobber or cond_clobber mode for '{k}' inside of a"
          ' `while_loop`.'
      )
    reap_avals[k] = meta['aval']
    if mode == 'cond_clobber':
      cond_avals[k] = jax_core.raise_to_shaped(jax_core.get_aval(True))

  cond_fun = jax_core.jaxpr_as_fun(cond_jaxpr)
  body_fun = jax_core.jaxpr_as_fun(body_jaxpr)
  reap_settings = dict(
      tag=settings.tag,
      allowlist=settings.allowlist,
      blocklist=settings.blocklist,
      exclusive=settings.exclusive)

  def new_cond(carry, *_):
    return cond_fun(*(cond_const_vals + carry))

  def new_body(carry, carry_reaps, carry_preds):
    carry, reaps, preds = _call_and_reap(body_fun, **reap_settings)(
        *(body_const_vals + carry)
    )
    for name, val in reaps.items():
      mode = body_metadata[name]['mode']
      _update_clobber_carry(carry_reaps, carry_preds, name, val, preds, mode)
    return (carry, carry_reaps, carry_preds)

  new_in_avals, new_in_tree = tree_util.tree_flatten(
      (init_avals, reap_avals, cond_avals)
  )
  new_cond_jaxpr, cond_consts, _ = lcf._initial_style_jaxpr(  # pylint: disable=protected-access
      new_cond, new_in_tree, tuple(new_in_avals))
  new_body_jaxpr, body_consts, out_tree = lcf._initial_style_jaxpr(  # pylint: disable=protected-access
      new_body, new_in_tree, tuple(new_in_avals))
  dummy_reap_vals = tree_util.tree_map(lambda x: jnp.zeros(x.shape, x.dtype),
                                       (reap_avals, cond_avals))
  new_in_vals = tree_util.tree_leaves((init_vals, dummy_reap_vals))
  out = lax.while_p.bind_with_trace(
      trace.parent_trace,
      (cond_consts + body_consts + new_in_vals),
      dict(cond_nconsts=len(cond_consts),
           body_nconsts=len(body_consts),
           cond_jaxpr=new_cond_jaxpr,
           body_jaxpr=new_body_jaxpr))
  out, reaps, preds = tree_util.tree_unflatten(out_tree, out)
  for k, v in reaps.items():
    mode = body_metadata[k]['mode']
    _sow(trace, v, name=k, tag=settings.tag, mode=mode, pred=preds[k])
  return out


reap_custom_rules[lcf.while_p] = _reap_while_rule


def _check_branch_metadata(branch_metadatas):
  """Checks that a set of harvest metadata are consistent with each other."""
  first_branch_meta = branch_metadatas[0]
  for branch_metadata in branch_metadatas[1:]:
    if len(branch_metadata) != len(first_branch_meta):
      raise ValueError('Mismatching number of `sow`s between branches.')
    for name, meta in branch_metadata.items():
      if name not in first_branch_meta:
        raise ValueError(f'Missing sow in branch: \'{name}\'.')
      first_meta_aval = first_branch_meta[name]['aval']
      if meta['aval'].shape != first_meta_aval.shape:
        raise ValueError(f'Mismatched shape between branches: \'{name}\'.')
      if meta['aval'].dtype != first_meta_aval.dtype:
        raise ValueError(f'Mismatched dtype between branches: \'{name}\'.')


def _reap_cond_rule(trace, *tracers, branches, linear=None):
  """Reaps each path of the `cond`."""
  index_val, ops_vals = tracers[0], tracers[1:]
  _, ops_avals = tree_util.tree_map(lambda x: jax_core.get_aval(x),  # pylint: disable=unnecessary-lambda
                                    (index_val, ops_vals))
  settings = trace.context.settings
  reap_settings = dict(
      tag=settings.tag,
      allowlist=settings.allowlist,
      blocklist=settings.blocklist,
      exclusive=settings.exclusive)
  branch_metadatas = tuple(
      _get_harvest_metadata(branch, settings, *ops_vals)
      for branch in branches)
  _check_branch_metadata(branch_metadatas)
  branch_funs = tuple(map(jax_core.jaxpr_as_fun, branches))
  reaped_branches = tuple(
      _call_and_reap(f, **reap_settings) for f in branch_funs)
  in_tree = tree_util.tree_structure(ops_avals)
  new_branch_jaxprs, consts, out_trees = (
      lcf._initial_style_jaxprs_with_common_consts(  # pylint: disable=protected-access
          reaped_branches, in_tree, ops_avals, lax.cond_p.name))
  if linear is None:
    out = lax.cond_p.bind_with_trace(
        trace.parent_trace,
        (index_val, *consts, *ops_vals),
        dict(branches=tuple(new_branch_jaxprs)))
  else:
    out = lax.cond_p.bind_with_trace(
        trace.parent_trace,
        (index_val, *consts, *ops_vals),
        dict(branches=tuple(new_branch_jaxprs),
             linear=(False,) * len(tuple(consts) + linear)))
  out, reaps, preds = tree_util.tree_unflatten(out_trees[0], out)
  for k, v in reaps.items():
    mode = branch_metadatas[0][k]['mode']
    _sow(trace, v, name=k, tag=settings.tag, mode=mode, pred=preds[k])
  return out


reap_custom_rules[lcf.cond_p] = _reap_cond_rule


def _reap_checkpoint_rule(trace, *invals, jaxpr, policy, prevent_cse,
                          differentiated):
  """Reap checkpoint rule."""
  settings = trace.context.settings
  reap_settings = dict(
      tag=settings.tag,
      allowlist=settings.allowlist,
      blocklist=settings.blocklist,
      exclusive=settings.exclusive)
  closed_jaxpr = jax_core.ClosedJaxpr(jaxpr, ())
  reap_metadata = _get_harvest_metadata(closed_jaxpr, settings, *invals)
  remat_fun = jax_core.jaxpr_as_fun(closed_jaxpr)
  reaped_remat_fun = _call_and_reap(remat_fun, **reap_settings)
  reap_jaxpr, consts, out_tree = lcf._initial_style_jaxpr(  # pylint: disable=protected-access
      reaped_remat_fun, tree_util.tree_structure(invals),
      tuple(jax_core.get_aval(t) for t in invals))
  outvals = ad_checkpoint.remat_p.bind_with_trace(
      trace.parent_trace,
      (*consts, *invals),
      dict(jaxpr=reap_jaxpr.jaxpr,
           policy=policy,
           prevent_cse=prevent_cse,
           differentiated=differentiated))
  out, reaps, preds = tree_util.tree_unflatten(out_tree, outvals)
  for k, v in reaps.items():
    mode = reap_metadata[k]['mode']
    _sow(trace, v, name=k, tag=settings.tag, mode=mode, pred=preds[k])
  return out


reap_custom_rules[ad_checkpoint.remat_p] = _reap_checkpoint_rule


@lu.cache
def _oryx_pjit_jaxpr(flat_fun, in_avals):
  jaxpr, out_avals, consts = pe.trace_to_jaxpr_dynamic(flat_fun, in_avals)
  if any(isinstance(c, jax_core.Tracer) for c in consts):
    jaxpr = pe.convert_constvars_jaxpr(jaxpr)
    jaxpr = pe.close_jaxpr(jaxpr)
    final_consts = consts
  else:
    jaxpr = jax_core.ClosedJaxpr(jaxpr, consts)
    final_consts = []

  return jaxpr, final_consts, out_avals


def _calc_extra_inps(num_consts, params):
  in_shardings = (
      sharding_impls.UNSPECIFIED,) * num_consts + params['in_shardings']
  donated_invars = (False,) * num_consts + params['donated_invars']
  in_layouts = (None,) * num_consts + params['in_layouts']
  return in_shardings, donated_invars, in_layouts


def _reap_pjit_rule(trace, *invals, **params):
  """Reap pjit rule."""
  if params['in_shardings'] and not any(
      isinstance(i, sharding_impls.UnspecifiedValue) for i in params['in_shardings']
  ):
    raise ValueError(
        'oryx only supports pjit which has no in_axis_resources '
        f'specified. Got {params["in_shardings"]}'
    )
  if params['out_shardings'] and not any(
      isinstance(o, sharding_impls.UnspecifiedValue) for o in params['out_shardings']
  ):
    raise ValueError(
        'oryx only supports pjit which has no out_axis_resources '
        f'specified. Got {params["out_shardings"]}'
    )

  settings = trace.context.settings
  reap_settings = dict(
      tag=settings.tag,
      allowlist=settings.allowlist,
      blocklist=settings.blocklist,
      exclusive=settings.exclusive)
  closed_jaxpr = params['jaxpr']
  reap_metadata = _get_harvest_metadata(closed_jaxpr, settings, *invals)
  pjit_fun = jax_core.jaxpr_as_fun(closed_jaxpr)
  reaped_pjit_fun = lu.wrap_init(_call_and_reap(pjit_fun, **reap_settings))
  in_tree = tree_util.tree_structure(invals)
  flat_fun, out_tree = api_util.flatten_fun_nokwargs(reaped_pjit_fun, in_tree)

  reap_jaxpr, final_consts, out_avals = _oryx_pjit_jaxpr(
      flat_fun, tuple(jax_core.get_aval(t) for t in invals))
  in_shardings, donated_invars, in_layouts = _calc_extra_inps(
      len(final_consts), params)

  new_params = {
      **params,
      'jaxpr': reap_jaxpr,
      'out_shardings': (sharding_impls.UNSPECIFIED,) * len(out_avals),
      'in_shardings': in_shardings,
      'donated_invars': donated_invars,
      'in_layouts': in_layouts,
      'out_layouts': (None,) * len(out_avals)
  }
  outvals = pjit.pjit_p.bind_with_trace(
      trace.parent_trace, (*final_consts, *invals), new_params)

  out, reaps, preds = tree_util.tree_unflatten(out_tree(), outvals)
  for k, v in reaps.items():
    mode = reap_metadata[k]['mode']
    _sow(trace, v, name=k, tag=settings.tag, mode=mode, pred=preds[k])
  return out


reap_custom_rules[pjit.pjit_p] = _reap_pjit_rule


plant_custom_rules = {}


@dataclasses.dataclass
class PlantContext(HarvestContext):
  """Contains the settings and storage for the current trace in the stack."""
  settings: HarvestSettings
  plants: Dict[str, Any]

  def __post_init__(self):
    self._already_planted = set()

  def get_custom_rule(self, primitive):
    return plant_custom_rules.get(primitive)

  def handle_sow(self, *values, name, tag, tree, mode):
    """Returns the value stored in the plants dictionary."""
    if name in self._already_planted:
      raise ValueError(f'Variable has already been planted: {name}')
    if name in self.plants:
      if not mode.endswith('clobber'):
        self._already_planted.add(name)
      if mode == 'cond_clobber':
        return tree_util.tree_leaves((self.plants[name], True))
      else:
        return tree_util.tree_leaves(self.plants[name])
    return sow_p.bind(*values, name=name, tag=tag, mode=mode, tree=tree)

  def process_nest(self, trace, f, *tracers, scope, name, **params):
    return self.process_higher_order_primitive(
        trace, nest_p, f, tracers, dict(params, name=name, scope=scope), False)

  def process_higher_order_primitive(self, trace, call_primitive, f, vals,
                                     params, is_map):
    del is_map
    name = jax_util.wrap_name(params.pop('name', f.__name__), 'reap')
    plants = trace.context.plants
    if 'in_axes' in params:
      # TODO(b/199459308): figure out if invars are mapped or unmapped
      params = dict(
          params,
          in_axes=(0,) * len(tree_util.tree_leaves(plants)) + params['in_axes'])
    if 'donated_invars' in params:
      params = dict(params)
      params['donated_invars'] = (
          (False,) * len(tree_util.tree_leaves(plants)) +
          params['donated_invars'])
    elif call_primitive is nest_p:
      plants = plants.get(params['scope'], {})
    all_vals, all_tree = tree_util.tree_flatten((plants, vals))
    f = plant_eval(f, self.settings, all_tree)
    return call_primitive.bind_with_trace(
        trace.parent_trace, (f, *all_vals), dict(name=name, **params))

  def process_custom_jvp_call(self, trace, primitive, fun, jvp, vals, *,
                              symbolic_zeros):
    fun = _subtrace(fun, trace.context)
    jvp = _subtrace(jvp, trace.context)
    out_flat = primitive.bind_with_trace(
        trace.parent_trace,
        (fun, jvp) + tuple(vals),
        dict(symbolic_zeros=symbolic_zeros))
    return out_flat

  def process_custom_vjp_call(self, trace, primitive, fun, fwd, bwd, vals,
                              out_trees, symbolic_zeros):
    fun = _subtrace(fun, trace.context)
    fwd = _subtrace(fwd, trace.context)
    # We don't need to subtrace the `bwd` since it's triggered in another trace.
    out_flat = primitive.bind_with_trace(
        trace.parent_trace,
        (fun, fwd, bwd) + tuple(vals),
        dict(out_trees=out_trees, symbolic_zeros=symbolic_zeros))
    return out_flat


@contextlib.contextmanager
def harvest_trace(context: HarvestContext):
  with jax_core.take_current_trace() as parent_trace:
    trace = HarvestTrace(parent_trace, context)
    with jax_core.set_current_trace(trace):
      yield


@lu.transformation
def _subtrace(context: HarvestContext, *args: Iterable[Any]):
  with harvest_trace(context):
    outs = yield args, {}
    yield outs


@lu.transformation
def plant_function(settings: HarvestSettings,
                   in_tree: Any, args: Iterable[Any]):
  """A function transformation that injects values in place of sows."""
  plants, args = tree_util.tree_unflatten(in_tree, args)
  context = PlantContext(settings, plants)
  with harvest_trace(context):
    ans = yield args, {}
  yield ans


def plant_eval(f: lu.WrappedFun, settings: HarvestSettings,
               all_tree: Any) -> Tuple[lu.WrappedFun, Callable[[], Any]]:
  f = plant_function(f, settings, all_tree)
  return plant_wrapper(f)


@lu.transformation
def plant_wrapper(*args):
  out = yield (args,), {}
  yield out


def plant(f,
          *,
          tag: Hashable,
          allowlist: Optional[Iterable[str]] = None,
          blocklist: Iterable[str] = frozenset(),
          exclusive: bool = False):
  """Transforms a function into one that injects values in place of sown ones.

  Args:
    f: a function to be transformed.
    tag: a string tag; only sown values with `tag` will be planted.
    allowlist: an optional sequence of string names, which if provided will
      enforce that only sows with names in the allowlist will be planted.
    blocklist: an optional sequence of string names, which if provided will
      enforce that only no sows with names in the blocklist will be planted.
    exclusive: determines whether or not to execute in "exclusive" mode where
      other tags are removed during execution.

  Returns:
    A new function that takes in a dictionary of planted values in addition to
    the original function's inputs, and injects the planted values in place of
    sown values.
  """

  blocklist = frozenset(blocklist)
  if allowlist is not None:
    allowlist = frozenset(allowlist)
  settings = HarvestSettings(tag, blocklist, allowlist, exclusive)

  def wrapped(plants, *args, **kwargs):
    fun = lu.wrap_init(f, kwargs)
    flat_args, in_tree = tree_util.tree_flatten(args)
    flat_fun, out_tree = api_util.flatten_fun_nokwargs(fun, in_tree)
    all_args, all_tree = tree_util.tree_flatten((plants, flat_args))
    flat_fun = plant_function(flat_fun, settings, all_tree)
    out_flat = flat_fun.call_wrapped(all_args)
    return tree_util.tree_unflatten(out_tree(), out_flat)

  return wrapped


def _plant_scan_rule(trace: HarvestTrace, *tracers, length, reverse, jaxpr,
                     num_consts, num_carry, linear, unroll, _split_transpose):
  """Injects values into a scan according to their sow mode."""

  const_vals, carry_vals, xs_vals = jax_util.split_list(
      tracers, [num_consts, num_carry])
  carry_avals, xs_avals = tree_util.tree_map(lambda x: jax_core.get_aval(x),  # pylint: disable=unnecessary-lambda
                                             (carry_vals, xs_vals))
  settings = trace.context.settings

  with jax_core.set_current_trace(trace.parent_trace):
    x_vals = [t[0] if hasattr(jax_core.get_aval(t), '_getitem') else t
              for t in xs_vals]
  x_avals = [t.aval for t in x_vals]
  metadata = _get_harvest_metadata(jaxpr, settings,
                                   *(const_vals + carry_vals + x_vals))

  plants = trace.context.plants
  plant_modes = collections.defaultdict(set)
  plant_xs_avals = {}
  for name, meta in metadata.items():
    mode = meta['mode']
    aval = meta['aval']
    if mode == 'strict':
      raise ValueError(f'Cannot use strict mode for \'{name}\' inside `scan`.')
    plant_modes[mode].add(name)
    if mode == 'append' and name in plants:
      plant_xs_avals[name] = aval
  body_fun = jax_core.jaxpr_as_fun(jaxpr)
  all_clobber_plants = {
      name: value
      for name, value in plants.items()
      if name in plant_modes['clobber'] | plant_modes['cond_clobber']
  }
  append_plants = {
      name: value
      for name, value in plants.items()
      if name in plant_modes['append']
  }

  plant_xs_flat_avals, _ = tree_util.tree_flatten(plant_xs_avals)

  plant_xs_in_tree = tree_util.tree_structure(
      (carry_avals, (xs_avals, plant_xs_avals)))

  def new_body(carry, x):
    x, plants = x
    all_plants = {**plants, **all_clobber_plants}
    all_values = const_vals + tree_util.tree_leaves((carry, x))
    out = plant(
        body_fun,
        tag=settings.tag,
        allowlist=settings.allowlist,
        blocklist=settings.blocklist,
        exclusive=settings.exclusive)(all_plants, *all_values)
    carry_out, y = jax_util.split_list(out, [num_carry])
    return carry_out, y

  new_body_jaxpr, consts, _ = lcf._initial_style_jaxpr(  # pylint: disable=protected-access
      new_body, plant_xs_in_tree,
      tuple(carry_avals + x_avals + plant_xs_flat_avals))
  plant_vals = tree_util.tree_leaves(append_plants)
  out = lcf.scan_p.bind_with_trace(
      trace.parent_trace,
      (consts + carry_vals + xs_vals + plant_vals),
      dict(reverse=reverse,
           length=length,
           jaxpr=new_body_jaxpr,
           num_consts=len(consts),
           num_carry=num_carry,
           linear=linear + (False,) * len(plant_vals),
           unroll=unroll,
           _split_transpose=_split_transpose))
  return out


plant_custom_rules[lcf.scan_p] = _plant_scan_rule


def _plant_while_rule(trace: HarvestTrace, *tracers, cond_jaxpr, body_jaxpr,
                      cond_nconsts, body_nconsts):
  """Injects values into a while loop, overriding values for all iterations."""
  cond_const_vals, body_const_vals, init_vals = jax_util.split_list(
      tracers, [cond_nconsts, body_nconsts])
  init_avals = tree_util.tree_map(lambda x: jax_core.get_aval(x), init_vals)  # pylint: disable=unnecessary-lambda
  settings = trace.context.settings
  body_metadata = _get_harvest_metadata(body_jaxpr, settings,
                                        *(body_const_vals + init_vals))
  for k, meta in body_metadata.items():
    mode = meta['mode']
    if mode not in ['clobber', 'cond_clobber']:
      raise ValueError(
          f"Must use clobber or cond_clobber mode for '{k}' inside of a"
          ' `while_loop`.'
      )

  body_fun = jax_core.jaxpr_as_fun(body_jaxpr)
  plant_settings = dict(
      tag=settings.tag,
      allowlist=settings.allowlist,
      blocklist=settings.blocklist,
      exclusive=settings.exclusive)
  plants = trace.context.plants

  def new_body(*carry):
    carry = plant(body_fun, **plant_settings)(plants,
                                              *(tuple(body_const_vals) + carry))
    return carry

  in_tree = tree_util.tree_structure(init_avals)
  new_body_jaxpr, new_body_consts, _ = lcf._initial_style_jaxpr(  # pylint: disable=protected-access
      new_body, in_tree, tuple(init_avals))
  out = lcf.while_p.bind_with_trace(
      trace.parent_trace,
      (cond_const_vals + new_body_consts + init_vals),
      dict(cond_nconsts=len(cond_const_vals),
           body_nconsts=len(new_body_consts),
           cond_jaxpr=cond_jaxpr,
           body_jaxpr=new_body_jaxpr))
  return out


plant_custom_rules[lcf.while_p] = _plant_while_rule


def _plant_cond_rule(trace, *tracers, branches, linear=None):
  """Injects the same values into both branches of a conditional."""
  index_val, ops_vals = tracers[0], tracers[1:]
  ops_avals = tree_util.tree_map(lambda x: jax_core.get_aval(x), ops_vals)  # pylint: disable=unnecessary-lambda
  settings = trace.context.settings
  plant_settings = dict(
      tag=settings.tag,
      allowlist=settings.allowlist,
      blocklist=settings.blocklist,
      exclusive=settings.exclusive)
  branch_metadatas = tuple(
      _get_harvest_metadata(branch, settings, *ops_vals)
      for branch in branches)
  _check_branch_metadata(branch_metadatas)
  plants = trace.context.plants
  branch_funs = tuple(map(jax_core.jaxpr_as_fun, branches))
  planted_branches = tuple(
      functools.partial(plant(f, **plant_settings), plants)
      for f in branch_funs)
  in_tree = tree_util.tree_structure(ops_avals)
  new_branch_jaxprs, consts, _ = (
      lcf._initial_style_jaxprs_with_common_consts(  # pylint: disable=protected-access
          planted_branches, in_tree, ops_avals, lax.cond_p.name))
  if linear is None:
    out = lax.cond_p.bind_with_trace(
        trace.parent_trace,
        (index_val, *consts, *ops_vals),
        dict(branches=tuple(new_branch_jaxprs)))
  else:
    out = lax.cond_p.bind_with_trace(
        trace.parent_trace,
        (index_val, *consts, *ops_vals),
        dict(branches=tuple(new_branch_jaxprs),
             linear=(False,) * len(tuple(consts) + linear)))
  return out


plant_custom_rules[lcf.cond_p] = _plant_cond_rule


def _plant_checkpoint_rule(trace, *invals, jaxpr, policy, prevent_cse,
                           differentiated):
  """Plant checkpoint rule."""
  settings = trace.context.settings
  plant_settings = dict(
      tag=settings.tag,
      allowlist=settings.allowlist,
      blocklist=settings.blocklist,
      exclusive=settings.exclusive)
  closed_jaxpr = jax_core.ClosedJaxpr(jaxpr, ())
  plants = trace.context.plants
  remat_fun = jax_core.jaxpr_as_fun(closed_jaxpr)
  planted_remat_fun = functools.partial(
      plant(remat_fun, **plant_settings), plants)
  plant_jaxpr, consts, _ = lcf._initial_style_jaxpr(  # pylint: disable=protected-access
      planted_remat_fun, tree_util.tree_structure(invals),
      tuple(jax_core.get_aval(t) for t in invals))
  return ad_checkpoint.remat_p.bind_with_trace(
      trace.parent_trace,
      (*consts, *invals),
      dict(jaxpr=plant_jaxpr.jaxpr,
           policy=policy,
           prevent_cse=prevent_cse,
           differentiated=differentiated))


plant_custom_rules[ad_checkpoint.remat_p] = _plant_checkpoint_rule


def _plant_pjit_rule(trace, *invals, **params):
  """Plant pjit rule."""
  if params['in_shardings'] and not any(
      isinstance(i, sharding_impls.UnspecifiedValue) for i in params['in_shardings']
  ):
    raise ValueError(
        'oryx only supports pjit which has no in_axis_resources '
        f'specified. Got {params["in_shardings"]}'
    )
  if params['out_shardings'] and not any(
      isinstance(o, sharding_impls.UnspecifiedValue) for o in params['out_shardings']
  ):
    raise ValueError(
        'oryx only supports pjit which has no out_axis_resources '
        f'specified. Got {params["out_shardings"]}'
    )

  settings = trace.context.settings
  plant_settings = dict(
      tag=settings.tag,
      allowlist=settings.allowlist,
      blocklist=settings.blocklist,
      exclusive=settings.exclusive)
  closed_jaxpr = params['jaxpr']
  plants = trace.context.plants

  pjit_fun = jax_core.jaxpr_as_fun(closed_jaxpr)
  planted_pjit_fun = lu.wrap_init(functools.partial(
      plant(pjit_fun, **plant_settings), plants))
  in_tree = tree_util.tree_structure(invals)
  flat_fun, _ = api_util.flatten_fun_nokwargs(planted_pjit_fun, in_tree)

  planted_jaxpr, final_consts, out_avals = _oryx_pjit_jaxpr(
      flat_fun, tuple(jax_core.get_aval(t) for t in invals))
  in_shardings, donated_invars, in_layouts = _calc_extra_inps(
      len(final_consts), params)

  new_params = {
      **params,
      'jaxpr': planted_jaxpr,
      'out_shardings': (sharding_impls.UNSPECIFIED,) * len(out_avals),
      'in_shardings': in_shardings,
      'donated_invars': donated_invars,
      'in_layouts': in_layouts,
      'out_layouts': (None,) * len(out_avals),
  }
  outvals = pjit.pjit_p.bind_with_trace(
      trace.parent_trace, (*final_consts, *invals), new_params)

  return outvals


plant_custom_rules[pjit.pjit_p] = _plant_pjit_rule


def harvest(f,
            *,
            tag: Hashable,
            allowlist: Optional[Iterable[str]] = None,
            blocklist: Iterable[str] = frozenset(),
            exclusive: bool = False):
  kwargs = dict(
      tag=tag, allowlist=allowlist, blocklist=blocklist, exclusive=exclusive)
  return call_and_reap(plant(f, **kwargs), **kwargs)


# Handle shard_map
@shard_map.register_check(sow_p)
def _sow_check(mesh, *in_rep, name, tag, mode, tree):
  del mesh, name, tag, mode, tree
  return in_rep[0]  # TODO(conmy): does this limit use to one output only?


@shard_map.register_rewrite(sow_p)
def _sow_rewrite(mesh, in_rep, *args, name, tag, mode, tree):
  raise ValueError(
      'Detected sow calls inside a shard_map. This is not currently supported.'
  )
