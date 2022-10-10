import jax
import jax.numpy as jnp
from jax import core as jax_core

from oryx.experimental.matching import matcher
from oryx.experimental.matching import rules
from oryx.experimental.matching import jax_rewrite as jr
from oryx.experimental.matching import jaxpr_rewriter


def f(w, x):
  x = x + 1.
  return w.dot(x) + 2.
jaxpr = jax.make_jaxpr(f)(jnp.ones((5, 2)), jnp.ones((2, 3))).jaxpr
print(jaxpr)

jaxpr = jaxpr_rewriter.Jaxpr.from_jaxpr(jaxpr)
print(jaxpr)

add_one = jaxpr_rewriter.JaxprEqn(
    [matcher.Var('x'), jaxpr_rewriter.Literal(matcher.Var('y',
      restrictions=[lambda x: x < 5]), jnp.float32)],
    matcher.Var('outvars'),
    jax.lax.add_p,
    matcher.Dot)

def _add_one_handler(x, y, outvars):
  return jaxpr_rewriter.JaxprEqn(
      (x, jaxpr_rewriter.Literal(y + 1., jnp.float32)),
      outvars,
      jax.lax.mul_p,
      jr.Params())

add_one_rule = rules.term_rewriter(
    rules.make_rule(add_one, _add_one_handler),
    )

print(add_one_rule(jaxpr).to_jaxpr())
