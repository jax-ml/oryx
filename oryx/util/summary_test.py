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

"""Tests for oryx.util.util."""

from absl.testing import absltest

from jax import lax
import jax.numpy as jnp
import numpy as np

from oryx import util
from oryx.internal import test_util


class SummaryTest(test_util.TestCase):

  def test_can_pull_out_summarized_values_in_strict_mode(self):
    def f(x):
      return util.summary(x, name='x')
    _, summaries = util.get_summaries(f)(1.)
    self.assertDictEqual(dict(x=1.), summaries)

  def test_can_pull_out_non_dependent_values(self):
    def f(x):
      util.summary(x ** 2, name='y')
      return x
    _, summaries = util.get_summaries(f)(2.)
    self.assertDictEqual(dict(y=4.), summaries)

  def test_duplicate_names_error_in_strict_mode(self):
    def f(x):
      util.summary(x, name='x')
      util.summary(x, name='x')
      return x
    with self.assertRaisesRegex(ValueError, 'has already been reaped: x'):
      util.get_summaries(f)(2.)

  def test_can_pull_summaries_out_of_scan_in_append_mode(self):
    def f(x):
      def body(x, _):
        util.summary(x, name='x', mode='append')
        return x + 1, ()
      return lax.scan(body, x, jnp.arange(10.))[0]
    value, summaries = util.get_summaries(f)(0.)
    self.assertEqual(value, 10.)
    np.testing.assert_allclose(summaries['x'], np.arange(10.))


if __name__ == '__main__':
  absltest.main()
