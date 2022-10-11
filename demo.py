import jax
import jax.numpy as jnp
from jax import core as jax_core

from oryx.experimental.matching import matcher
from oryx.experimental.matching import rules
from oryx.experimental.matching import jax_rewrite as jr
from oryx.experimental.matching import jaxpr_rewriter

Var = matcher.Var

def f(w, x):
  # x = x + 1.
  return w.dot(x) + 2.
jaxpr = jax.make_jaxpr(f)(jnp.ones((5, 2)), jnp.ones((2, 3))).jaxpr
print(jaxpr)

jaxpr_graph = jaxpr_rewriter.JaxprGraph.from_jaxpr(jaxpr)
node = jaxpr_graph.outvars[0]
pattern = jaxpr_rewriter.Eqn(
    jax.lax.add_p, jr.Params(),
      [jaxpr_rewriter.Eqn(
          jax.lax.dot_general_p, jr.Params(dimension_numbers=(((1,), (0,)), ((), ())),
                                           precision=matcher.Dot,
                                           preferred_element_type=matcher.Dot),
          [Var('x'), Var('y')],
          Var('shape'), Var('dtype')), Var('z')],
      Var('shape'), Var('dtype'))

def _handler(x, y, z,shape, dtype):
  return jaxpr_rewriter.Eqn(
      jax.lax.mul_p, jr.Params(), [x, y], shape, dtype)
print(jaxpr_graph.to_jaxpr())
jaxpr_graph.rewrite_subgraph(pattern, _handler)
print(jaxpr_graph.to_jaxpr())


# eqns = [
#     jaxpr_rewriter.JaxprEqn(
#     [matcher.Var('x'), jaxpr_rewriter.Literal(matcher.Var('y',
#       restrictions=[lambda x: x < 5]), jnp.float32)],
#     matcher.Var('outvars'),
#     jax.lax.add_p,
#     matcher.Dot)
#     ]
# jaxpr_pattern = jaxpr_rewriter.Jaxpr(
#     [],
#     [matcher.Var('x'), matcher.Var('y')],
#     eqns,
#     matcher.Var('outvars'))
    
# print(jaxpr_pattern)
# print(jaxpr.match(jaxpr_pattern))
