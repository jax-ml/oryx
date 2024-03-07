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

"""Tests for oryx.core.primitive."""

from absl.testing import absltest
import jax

from oryx.core import primitive
from oryx.internal import test_util


class PrimitiveTest(test_util.TestCase):

  def test_tie_in_grad(self):

    def f(x, key):
      key = primitive.tie_in(x, key)
      jax.random.split(key)
      return x

    jax.grad(f)(0.0, jax.random.PRNGKey(0))


if __name__ == '__main__':
  absltest.main()
