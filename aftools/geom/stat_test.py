r"""Test cases for `aftools.geom.stat.py`."""
# Authors: Zilin Song.


import os
import sys
sys.dont_write_bytecode=True
sys.path.insert(0, os.getcwd())

import unittest

import jax.numpy as jnp
import     numpy as  np

from aftools.geom.stat import update, reduce


class Test_welford(unittest.TestCase):

  def setUp(self):
    N, P, Q = np.random.randint(10, 100), 10, 20
    self.shape = (P, Q)
    self.vecs = jnp.asarray([(np.random.random_sample((P, Q))-.5) for _ in range(N)])

  def tearDown(self):
    del self.vecs

  def test_reduce_one_sample(self):
    r"""Test reducing a single sample produces the sample as mean and zero variance."""
    # test.
    stat_dict = reduce(stat=update(x=self.vecs[0, :, :]))
    np.testing.assert_array_equal(stat_dict['mean'], self.vecs[0, :, :])
    np.testing.assert_array_equal(stat_dict['var' ], 0.)
    np.testing.assert_array_equal(stat_dict['std' ], 0.)
    np.testing.assert_array_equal(stat_dict['cv'  ], 0.)

  def test_reduce_many_samples(self):
    r"""Test Welford reduction matches numpy mean/var/std/cv over many samples."""
    # ref x stats.
    x_mean_ref = jnp.mean(self.vecs, axis=0)
    x_var_ref  = jnp.var (self.vecs, axis=0, ddof=1)
    x_std_ref  = jnp.std (self.vecs, axis=0, ddof=1)
    x_cv_ref   = x_std_ref / jnp.abs(x_mean_ref)
    # test.
    stat = None
    for _ in range(self.vecs.shape[0]):
      stat = update(x=self.vecs[_, :, :], stat=stat)
    stat_dict = reduce(stat=stat)
    np.testing.assert_allclose(stat_dict['mean'], x_mean_ref, rtol=0.,   atol=1e-6)
    np.testing.assert_allclose(stat_dict['var' ], x_var_ref,  rtol=0.,   atol=1e-6)
    np.testing.assert_allclose(stat_dict['std' ], x_std_ref,  rtol=0.,   atol=1e-6)
    np.testing.assert_allclose(stat_dict['cv'  ], x_cv_ref,   rtol=1e-2, atol=0.)
