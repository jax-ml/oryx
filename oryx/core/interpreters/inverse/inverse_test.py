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

"""Tests for oryx.core.interpreters.inverse."""
import os
import unittest

from absl.testing import absltest
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

from oryx.core.interpreters import harvest
from oryx.core.interpreters.inverse import core
from oryx.core.interpreters.inverse import rules
from oryx.internal import test_util
from tensorflow_probability.substrates import jax as tfp

tfb = tfp.bijectors
tfd = tfp.distributions

del rules  # needed for registration only


class InverseTest(test_util.TestCase):

  def test_trivial_inverse(self):

    def f(x):
      return x

    f_inv = core.inverse(f)
    np.testing.assert_allclose(f_inv(1.0), 1.0)

    def f2(x, y):
      return x, y

    f2_inv = core.inverse(f2)
    np.testing.assert_allclose(f2_inv(1.0, 2.0), (1.0, 2.0))

  def test_mul_inverse(self):

    def f(x):
      return x * 2.

    f_inv = core.inverse(f)
    np.testing.assert_allclose(f_inv(1.0), 0.5)

    def f2(x):
      return 2. * x

    f2_inv = core.inverse(f2)
    np.testing.assert_allclose(f2_inv(1.0), 0.5)

  def test_div_inverse(self):

    def f(x):
      return x / 2.

    f_inv = core.inverse(f)
    np.testing.assert_allclose(f_inv(1.0), 2.)

    def f2(x):
      return 2. / x

    f2_inv = core.inverse(f2)
    np.testing.assert_allclose(f2_inv(1.0), 2.)

  def test_trivial_noninvertible(self):

    def f(x):
      del x
      return 1.

    with self.assertRaises(ValueError):
      core.inverse(f)(1.)

  def test_noninvertible(self):

    def f(x, y):
      return x + y, x + y

    with self.assertRaises(ValueError):
      core.inverse(f)(1., 2.)

  def test_simple_inverse(self):

    def f(x):
      return jnp.exp(x)

    f_inv = core.inverse(f, 0.1)
    np.testing.assert_allclose(f_inv(1.0), 0.)

    def f2(x):
      return jnp.exp(x)

    f2_inv = core.inverse(f2, jnp.zeros(2))
    np.testing.assert_allclose(f2_inv(jnp.ones(2)), jnp.zeros(2))

  def test_conditional_inverse(self):

    def f(x, y):
      return x + 1., jnp.exp(x + 1.) + y

    f_inv = core.inverse(f, 0., 2.)
    np.testing.assert_allclose(f_inv(0., 2.), (-1., 1.))

  def test_simple_ildj(self):

    def f(x):
      return jnp.exp(x)

    f_inv = core.ildj(f, 0.1)
    np.testing.assert_allclose(f_inv(2.0), -jnp.log(2.))

    def f2(x):
      return jnp.exp(x)

    f2_inv = core.ildj(f2, jnp.zeros(2))
    np.testing.assert_allclose(f2_inv(2 * jnp.ones(2)), -2 * jnp.log(2.))

  def test_advanced_inverse_two(self):

    def f(x, y):
      return jnp.exp(x), x**2 + y

    f_inv = core.inverse(f, 0.1, 0.2)
    np.testing.assert_allclose(
        f_inv(2.0, 2.0), (jnp.log(2.), 2 - jnp.log(2.)**2))

  def test_advanced_inverse_three(self):

    def f(x, y, z):
      return jnp.exp(x), x**2 + y, jnp.exp(z + y)

    f_inv = core.inverse(f, 0., 0., 0.)
    np.testing.assert_allclose(
        f_inv(2.0, 2.0, 2.0),
        (jnp.log(2.), 2 - jnp.log(2.)**2, jnp.log(2.0) - (2 - jnp.log(2.)**2)))

  def test_mul_inverse_ildj(self):

    def f(x):
      return x * 2

    f_inv = core.inverse_and_ildj(f, 1.)
    x, ildj_ = f_inv(2.)
    np.testing.assert_allclose(x, 1.)
    np.testing.assert_allclose(
        ildj_, -jnp.log(jnp.abs(jax.jacrev(f)(1.))), atol=1e-6, rtol=1e-6)

    def f2(x):
      return 2 * x

    f2_inv = core.inverse_and_ildj(f2, 1.)
    x, ildj_ = f2_inv(2.)
    np.testing.assert_allclose(x, 1.)
    np.testing.assert_allclose(
        ildj_, -jnp.log(jnp.abs(jax.jacrev(f)(1.))), atol=1e-6, rtol=1e-6)

  def test_lower_triangular_jacobian(self):

    def f(x, y):
      return x + 2., jnp.exp(x) + y

    def f_vec(x):
      return jnp.array([x[0] + 2., jnp.exp(x[0]) + x[1]])

    f_inv = core.inverse_and_ildj(f, 0., 0.)
    x, ildj_ = f_inv(3., jnp.exp(1.) + 1.)
    np.testing.assert_allclose(x, (1., 1.))
    np.testing.assert_allclose(
        ildj_,
        -jnp.log(
            jnp.abs(jnp.linalg.slogdet(jax.jacrev(f_vec)(jnp.ones(2)))[0])),
        atol=1e-6,
        rtol=1e-6)

  def test_div_inverse_ildj(self):

    def f(x):
      return x / 2

    f_inv = core.inverse_and_ildj(f, 2.)
    x, ildj_ = f_inv(2.)
    np.testing.assert_allclose(x, 4.)
    np.testing.assert_allclose(
        ildj_, -jnp.log(jnp.abs(jax.jacrev(f)(4.))), atol=1e-6, rtol=1e-6)

    def f2(x):
      return 3. / x

    f2_inv = core.inverse_and_ildj(f2, 2.)
    x, ildj_ = f2_inv(2.)
    np.testing.assert_allclose(x, 1.5)
    np.testing.assert_allclose(
        ildj_, -jnp.log(jnp.abs(jax.jacrev(f2)(1.5))), atol=1e-6, rtol=1e-6)

  def test_inverse_of_jit(self):

    def f(x):
      x = jax.jit(lambda x: x)(x)
      return x / 2.

    f_inv = core.inverse_and_ildj(f, 2.)
    x, ildj_ = f_inv(2.)
    np.testing.assert_allclose(x, 4.)
    np.testing.assert_allclose(
        ildj_, -jnp.log(jnp.abs(jax.jacrev(f)(4.))), atol=1e-6, rtol=1e-6)

    def f2(x):
      return jax.jit(lambda x: 3. / x)(x)

    f2_inv = core.inverse_and_ildj(f2, 2.)
    x, ildj_ = f2_inv(2.)
    np.testing.assert_allclose(x, 1.5)
    np.testing.assert_allclose(
        ildj_, -jnp.log(jnp.abs(jax.jacrev(f2)(1.5))), atol=1e-6, rtol=1e-6)

  def test_inverse_of_pmap(self):
    x = jnp.ones(jax.local_device_count())

    def f(x):
      return jax.pmap(lambda x: jnp.exp(x) + 2.)(x)

    f_inv = core.inverse_and_ildj(f, x * 4)
    x_, ildj_ = f_inv(x * 4)
    np.testing.assert_allclose(x_, jnp.log(2.) * x)
    np.testing.assert_allclose(
        ildj_,
        -jnp.log(jnp.abs(jnp.sum(jax.jacrev(f)(jnp.log(2.) * x)))),
        atol=1e-6,
        rtol=1e-6)

  def test_pmap_forward(self):
    if jax.local_device_count() < 2:
      raise unittest.SkipTest('Not enough devices for test.')

    def f(x, y):
      z = jax.pmap(jnp.exp)(x)
      return x + 2., z + y

    def f_vec(x):
      return jnp.array([x[0] + 2., jnp.exp(x[0]) + x[1]])

    f_inv = core.inverse_and_ildj(f, jnp.ones(2), jnp.ones(2))
    x, ildj_ = f_inv(2 * jnp.ones(2), jnp.ones(2))
    np.testing.assert_allclose(x, (jnp.zeros(2), jnp.zeros(2)))
    np.testing.assert_allclose(
        ildj_,
        -jnp.log(
            jnp.abs(jnp.linalg.slogdet(jax.jacrev(f_vec)(jnp.ones(2)))[0])),
        atol=1e-6,
        rtol=1e-6)

  def test_inverse_of_sow_is_identity(self):

    def f(x):
      return harvest.sow(x, name='x', tag='foo')

    x, ildj_ = core.inverse_and_ildj(f, 1.)(1.)
    self.assertEqual(x, 1.)
    self.assertEqual(ildj_, 0.)

  def test_sow_happens_in_forward_pass(self):

    def f(x, y):
      return x, harvest.sow(x, name='x', tag='foo') * y

    vals = harvest.reap(core.inverse(f), tag='foo')(1., 1.)
    self.assertDictEqual(vals, dict(x=1.))

  def test_inverse_of_nest(self):

    def f(x):
      x = harvest.nest(lambda x: x, scope='foo')(x)
      return x / 2.

    f_inv = core.inverse_and_ildj(f, 2.)
    x, ildj_ = f_inv(2.)
    np.testing.assert_allclose(x, 4.)
    np.testing.assert_allclose(
        ildj_, -jnp.log(jnp.abs(jax.jacrev(f)(4.))), atol=1e-6, rtol=1e-6)

  def test_inverse_of_split(self):

    def f(x):
      return jnp.split(x, 2)

    f_inv = core.inverse_and_ildj(f, jnp.ones(4))
    x, ildj_ = f_inv([jnp.ones(2), jnp.ones(2)])
    np.testing.assert_allclose(x, jnp.ones(4))
    np.testing.assert_allclose(ildj_, 0., atol=1e-6, rtol=1e-6)

  def test_inverse_of_concatenate(self):

    def f(x, y):
      return jnp.concatenate([x, y])

    f_inv = core.inverse_and_ildj(f, jnp.ones(2), jnp.ones(2))
    (x, y), ildj_ = f_inv(jnp.ones(4))
    np.testing.assert_allclose(x, jnp.ones(2))
    np.testing.assert_allclose(y, jnp.ones(2))
    np.testing.assert_allclose(ildj_, 0., atol=1e-6, rtol=1e-6)

  def test_inverse_of_reshape(self):

    def f(x):
      return jnp.reshape(x, (4,))

    f_inv = core.inverse_and_ildj(f, jnp.ones((2, 2)))
    x, ildj_ = f_inv(jnp.ones(4))
    np.testing.assert_allclose(x, jnp.ones((2, 2)))
    np.testing.assert_allclose(ildj_, 0.)

  def test_softplus_inverse_ildj(self):
    softplus_inv = core.inverse_and_ildj(jax.nn.softplus)
    softplus_bij = tfb.Softplus()
    x, ildj = softplus_inv(0.1)
    np.testing.assert_allclose(x,
                               softplus_bij.inverse(0.1))
    np.testing.assert_allclose(ildj,
                               softplus_bij.inverse_log_det_jacobian(0.1))

  def test_sigmoid_ildj(self):

    def naive_sigmoid(x):
      # This is the default JAX implementation of sigmoid.
      return 1. / (1 + jnp.exp(-x))

    naive_inv = core.inverse(naive_sigmoid)
    naive_ildj = core.ildj(naive_sigmoid)

    with self.assertRaises(AssertionError):
      np.testing.assert_allclose(
          naive_inv(0.9999), jax.scipy.special.logit(0.9999))
    with self.assertRaises(AssertionError):
      np.testing.assert_allclose(
          naive_ildj(0.9999),
          tfb.Sigmoid().inverse_log_det_jacobian(0.9999, 0))

    f_inv = core.inverse(jax.nn.sigmoid)
    f_ildj = core.ildj(jax.nn.sigmoid)
    np.testing.assert_allclose(f_inv(0.9999), jax.scipy.special.logit(0.9999))
    np.testing.assert_allclose(
        f_ildj(0.9999),
        tfb.Sigmoid().inverse_log_det_jacobian(0.9999, 0))

  def test_logit_ildj(self):

    def naive_logit(x):
      # This is the default JAX implementation of logit.
      return jnp.log(x / (1. - x))

    naive_inv = core.inverse(naive_logit)
    naive_ildj = core.ildj(naive_logit)
    with self.assertRaises(ValueError):
      naive_inv(-100.)
    with self.assertRaises(ValueError):
      naive_ildj(-100.)
    f_inv = core.inverse(jax.scipy.special.logit)
    f_ildj = core.ildj(jax.scipy.special.logit)
    np.testing.assert_allclose(f_inv(-100.), jax.scipy.special.expit(-100.))
    np.testing.assert_allclose(
        f_ildj(-100.),
        tfb.Sigmoid().forward_log_det_jacobian(-100., 0))

  def test_integer_pow_inverse(self):

    def f(x):
      return lax.integer_pow(x, 2)

    f_inv = core.inverse(f)
    np.testing.assert_allclose(f_inv(2.), jnp.sqrt(2.))

    def f2(x):
      return lax.integer_pow(x, 3)

    f2_inv = core.inverse(f2)
    np.testing.assert_allclose(f2_inv(2.), np.cbrt(2.))

  def test_integer_pow_ildj(self):

    def f(x):
      return lax.integer_pow(x, 2)

    f_ildj = core.ildj(f)
    np.testing.assert_allclose(
        f_ildj(2.),
        tfb.Power(2.).inverse_log_det_jacobian(2.))

    def f2(x):
      return lax.integer_pow(x, 3)

    f2_ildj = core.ildj(f2)
    np.testing.assert_allclose(
        f2_ildj(2.),
        tfb.Power(3.).inverse_log_det_jacobian(2.))

  def test_reciprocal_inverse(self):

    def f(x):
      return jnp.reciprocal(x)

    f_inv = core.inverse(f)
    np.testing.assert_allclose(f_inv(2.), 0.5)

  def test_reciprocal_ildj(self):

    def f(x):
      return jnp.reciprocal(x)

    f_ildj = core.ildj(f)
    np.testing.assert_allclose(f_ildj(2.), np.log(1 / 4.))

  def test_pow_inverse(self):

    def f(x, y):
      return lax.pow(x, y)

    f_x_inv = core.inverse(lambda x: f(x, 2.))
    np.testing.assert_allclose(f_x_inv(2.), jnp.sqrt(2.))
    f_y_inv = core.inverse(lambda y: f(2., y))
    np.testing.assert_allclose(f_y_inv(3.), jnp.log(3.) / jnp.log(2.))

  def test_pow_ildj(self):

    def f(x, y):
      return lax.pow(x, y)

    f_x_ildj = core.ildj(lambda x: f(x, 2.))
    np.testing.assert_allclose(
        f_x_ildj(3.),
        tfb.Power(2.).inverse_log_det_jacobian(3.))
    f_y_ildj = core.ildj(lambda y: f(2., y))
    f_y_inv = core.inverse(lambda y: f(2., y))
    y = f_y_inv(3.)
    np.testing.assert_allclose(
        f_y_ildj(3.), -jnp.log(jnp.abs(jax.grad(lambda y: f(2., y))(y))))
    np.testing.assert_allclose(
        f_y_ildj(3.), jnp.log(jnp.abs(jax.grad(f_y_inv)(3.))))

  def test_sqrt_inverse(self):

    def f(x):
      return jnp.sqrt(x)

    f_inv = core.inverse(f)
    np.testing.assert_allclose(f_inv(2.), 4.)

  def test_sqrt_ildj(self):

    def f(x):
      return jnp.sqrt(x)

    f_ildj = core.ildj(f)
    np.testing.assert_allclose(f_ildj(3.), jnp.log(2.) + jnp.log(3.))

  def test_joint_inverse(self):
    dist = tfd.JointDistributionNamed(
        {'a': tfd.JointDistributionNamed({'b': tfd.HalfNormal(10.)})})
    bij = dist.experimental_default_event_space_bijector()

    # This used to throw an exception due to patching the bijector.
    bij(dist.sample(seed=jax.random.PRNGKey(0)))


if __name__ == '__main__':
  os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=2'
  absltest.main()
