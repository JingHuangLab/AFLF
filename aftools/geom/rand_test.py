r"""Test cases for `aftools.geom.rand.py`."""
# Authors: Zilin Song.


import os
import sys
sys.dont_write_bytecode=True
sys.path.insert(0, os.getcwd())

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from aftools.geom.rand import dropout


class Test_dropout(unittest.TestCase):
  r"""Test for geom.rand.dropout()."""

  def setUp(self):
    self.tensor = (np.random.random_sample((10, 3))-.5)*np.random.randint(5, 100)
    self.key    = jax.random.PRNGKey(seed=np.random.randint(low=-99999, high=100000))

  def tearDown(self):
    del self.tensor
    del self.key

  def test_dropout_haiku(self):
    r"""Test dropout against haiku dropout reference."""
    # prep.
    rate = np.random.random_sample()
    # ref.
    import haiku as hk
    t_dropout_ref = hk.dropout(rng=self.key, rate=rate, x=self.tensor)
    # test.
    t_dropout = dropout(tensor=self.tensor, p=rate, key=self.key)
    np.testing.assert_equal(bool((t_dropout!=self.tensor).any()), True) # ensures dropout applied.
    np.testing.assert_allclose(t_dropout, t_dropout_ref, rtol=0., atol=1e-4)

  def test_dropout_rate_ge1(self):
    r"""Test dropout with rate >= 1 zeros the tensor."""
    # test > 1.
    rate = np.random.random_sample() + 1. + 1e-5
    t_dropout = dropout(tensor=self.tensor, p=rate, key=self.key)
    np.testing.assert_array_equal(t_dropout, jnp.zeros_like(self.tensor))
    # test = 1.
    t_dropout = dropout(tensor=self.tensor, p=1., key=self.key)
    np.testing.assert_array_equal(t_dropout, jnp.zeros_like(self.tensor))

  def test_dropout_rate_le0(self):
    r"""Test dropout with rate <= 0 preserves the tensor."""
    # test < 0.
    rate = - np.random.random_sample() - 1e-5
    t_dropout = dropout(tensor=self.tensor, p=rate, key=self.key)
    np.testing.assert_allclose(t_dropout, self.tensor, rtol=0., atol=1e-5)
    # test = 0.
    t_dropout = dropout(tensor=self.tensor, p=0., key=self.key)
    np.testing.assert_allclose(t_dropout, self.tensor, rtol=0., atol=1e-5)
