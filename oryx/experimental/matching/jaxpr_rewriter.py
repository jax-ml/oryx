from __future__ import annotations

import dataclasses

from typing import Any, List, Tuple

from jax import core as jax_core
import jax.numpy as jnp

from oryx.experimental.matching import matcher
from oryx.experimental.matching import jax_rewrite as jr

Expr = matcher.Expr
Bindings = matcher.Bindings
Continuation = matcher.Continuation
Success = matcher.Success

@dataclasses.dataclass(frozen=True)
class JaxprVar(jr.JaxExpression):
  _shape: Tuple[int, ...]
  _dtype: jnp.dtype

  def match(self, expr, bindings, succeed):
    if not isinstance(expr, JaxprVar):
      return
    yield from matcher.matcher((self._shape, self.dtype))(
        (expr._shape, expr._dtype), bindings, succeed)

  def tree_map(self, fn):
    return self

  def tree_children(self):
    yield from []

  @property
  def dtype(self):
    return self._dtype

  @property
  def shape(self):
    return self._shape

  def evaluate(self, env):
    return env[self]

@dataclasses.dataclass(frozen=True)
class Literal(jr.JaxExpression):
  value: Any
  _dtype: jnp.dtype

  def match(self, expr, bindings, succeed):
    if not isinstance(expr, Literal):
      return
    yield from matcher.matcher((self.value, self.dtype))(
        (expr.value, expr._dtype), bindings, succeed)

  def tree_map(self, fn):
    return self

  def tree_children(self):
    yield from []

  @property
  def dtype(self):
    return self._dtype

  @property
  def shape(self):
    return ()

  def __hash__(self):
    return object.__hash__(self)

  def evaluate(self, _):
    return self.value

@dataclasses.dataclass(frozen=True)
class Jaxpr:
  constvars: Tuple[JaxprVar]
  invars: Tuple[JaxprVar]
  eqns: Tuple[JaxprEqn]
  outvars: Tuple[JaxprVar]

  def match(self, pattern):
    for eqn in self.eqns:
      pass
    

  @classmethod
  def from_jaxpr(cls, jaxpr: jax_core.Jaxpr) -> Jaxpr:
    var_mapping = {}
    for var in jaxpr.invars + jaxpr.constvars:
      new_invar = JaxprVar(var.aval.shape, var.aval.dtype)
      var_mapping[var] = new_invar
    eqns = []
    for eqn in jaxpr.eqns:
      invars = []
      for var in eqn.invars:
        if isinstance(var, jax_core.Literal):
          invars.append(Literal(var.val, var.aval.dtype))
        else:
          invars.append(var_mapping[var])
      new_outvars = []
      for var in eqn.outvars:
        new_outvar = JaxprVar(var.aval.shape, var.aval.dtype)
        var_mapping[var] = new_outvar
        new_outvars.append(new_outvar)
      eqns.append(JaxprEqn(tuple(invars), tuple(new_outvars), eqn.primitive,
        jr.Params(eqn.params)))
    invars = tuple(var_mapping[v] for v in jaxpr.invars)
    constvars = tuple(var_mapping[v] for v in jaxpr.constvars)
    outvars = tuple(var_mapping[v] for v in jaxpr.outvars)
    eqns = tuple(eqns)
    return Jaxpr(constvars, invars, eqns, outvars)

  def to_jaxpr(self) -> jax_core.Jaxpr:
    gen = jax_core.gensym()
    var_mapping = {}
    for var in self.invars + self.constvars:
      var_mapping[var] = gen(jax_core.ShapedArray(var._shape, var._dtype))
    eqns = []
    for eqn in self.eqns:
      invars = []
      for var in eqn.invars:
        if isinstance(var, Literal):
          invars.append(jax_core.Literal(var.value, jax_core.ShapedArray((), var._dtype)))
        else:
          invars.append(var_mapping[var])
      for var in eqn.outvars:
        var_mapping[var] = gen(jax_core.ShapedArray(var._shape, var._dtype))
      outvars = [var_mapping[var] for var in eqn.outvars]
      eqns.append(jax_core.JaxprEqn(invars, outvars, eqn.primitive,
        dict(eqn.params), jax_core.no_effects, None))
    constvars = [var_mapping[v] for v in self.constvars]
    invars = [var_mapping[v] for v in self.invars]
    outvars = [var_mapping[v] for v in self.outvars]
    return jax_core.Jaxpr(
        constvars, invars, outvars, eqns, jax_core.no_effects)


@dataclasses.dataclass(frozen=True)
class JaxprEqn(jr.JaxExpression):
  invars: Tuple[jax_core.Atom]
  outvars: Tuple[jax_core.JaxprVar]
  primitive: jax_core.Primitive
  params: jr.Params

  @property
  def dtype(self):
    return [o.aval.dtype for o in self.outvars]
  
  @property
  def shape(self):
    return [o.aval.shape for o in self.outvars]

  def match(self, expr: Expr, bindings: Bindings,
            succeed: Continuation) -> Success:
    if not isinstance(expr, JaxprEqn):
      return
    yield from matcher.matcher(
        (self.invars, self.outvars, self.primitive, self.params))(
        (expr.invars, expr.outvars, expr.primitive, expr.params),
        bindings, succeed)

  def tree_map(self, fn):
    return self

  def tree_children(self):
    yield from self.invars
    yield from self.outvars
    yield self.primitive
    yield self.params

  def evaluate(self, env):
    invals = [jr.evaluate(v, env) for v in self.invars]
    return self.primitive.bind(*invals, **self.params)
