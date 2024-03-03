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

"""Module for probability bijectors and related functions."""
import inspect

from tensorflow_probability.substrates import jax as tfp  # pylint: disable=g-importing-member

tfb = tfp.bijectors

__all__ = tfb.__all__
for name in __all__:
  bij = getattr(tfb, name)
  locals()[name] = bij

del tfb
