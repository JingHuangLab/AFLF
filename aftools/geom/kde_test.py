r"""Test cases for `aftools.geom.kde.py`."""
# Authors: Zilin Song.


import os
import sys
sys.dont_write_bytecode=True
sys.path.insert(0, os.getcwd())

import unittest

import jax
import jax.numpy as jnp
import     numpy as  np

from aftools.geom.kde import bin_breaks, uniform_binning, gaussian_binning


class Test_bin_breaks(unittest.TestCase):
  r"""Test for geom.kde.bin_breaks()."""

  def test_init_sanity(self):
    r"""Test edge cases and sanity of bin_breaks."""
    with self.assertRaisesRegex(AssertionError, f"Illegal edge range: `init=12.0` and `last=3.0`."):
      bin_breaks(12., 3., num_bins=30)
    np.testing.assert_array_equal(bin_breaks(0., 2., num_bins=3), jnp.asarray([0., 2.]))
    np.testing.assert_array_equal(bin_breaks(0., 0., num_bins=2), jnp.asarray([0.    ]))

class Test_uniform_binning(unittest.TestCase):
  r"""Test for geom.kde.uniform_binning()."""

  def setUp(self):
    self.breaks = jnp.arange(1, 10)*.1

  def tearDown(self):
    del self.breaks

  def test_onehot(self):
    r"""Test that logits at bin centers produce one-hot bin assignments."""
    logits = .1 * jnp.arange(self.breaks.shape[0]+1) + 1e-2
    binned = uniform_binning(logits=logits, breaks=self.breaks)
    np.testing.assert_array_equal(binned, jnp.eye(logits.shape[0]))  # one item per bin.

  def test_input(self):
    r"""Test uniform_binning with empty, scalar, open-bin, and deterministic inputs."""
    # empty.
    logits = jnp.array([])
    binned = uniform_binning(logits=logits, breaks=self.breaks)
    self.assertEqual(binned.shape, (0, self.breaks.shape[0]+1))
    # scalar.
    logits = jnp.array([.4])+1e-2
    binned = uniform_binning(logits=logits, breaks=self.breaks)
    self.assertEqual(binned.shape, (1, self.breaks.shape[0]+1))
    self.assertEqual(binned[0, 4], 1.)
    # open bins.
    logits = jnp.array([-1e6, .01, .43, .45, .65, 1e6])
    binned = uniform_binning(logits=logits, breaks=self.breaks)
    np.testing.assert_array_equal(binned[:, 0], jnp.array([1., 1., 0., 0., 0., 0.]))
    np.testing.assert_array_equal(binned[:, 4], jnp.array([0., 0., 1., 1., 0., 0.]))
    np.testing.assert_array_equal(binned[:, 6], jnp.array([0., 0., 0., 0., 1., 0.]))
    np.testing.assert_array_equal(binned[:, 9], jnp.array([0., 0., 0., 0., 0., 1.]))
    # deterministic.
    logits = np.random.random_sample(100)
    binned0 = uniform_binning(logits=logits, breaks=self.breaks)
    binned1 = uniform_binning(logits=logits, breaks=self.breaks)
    np.testing.assert_array_equal(binned0, binned1)


class Test_gaussian_binning(unittest.TestCase):
  r"""Test for geom.kde.gaussian_binning()."""

  def setUp(self):
    self.breaks = jnp.arange(1, 10)*.1
    self.rwidth = np.random.random_sample()

  def tearDown(self):
    del self.breaks
    del self.rwidth

  def test_probs(self):
    r"""Test gaussian_binning probability properties, extremes, and determinism."""
    # probs.
    logits = .1 * jnp.arange(self.breaks.shape[0]+1) + 5e-2
    binned = gaussian_binning(logits=logits, breaks=self.breaks, rwidth=self.rwidth)
    self.assertEqual(binned.shape, (logits.shape[0], logits.shape[0]))
    np.testing.assert_array_compare(lambda x, y: x<=y, binned, jnp.ones (binned.shape))
    np.testing.assert_array_compare(lambda x, y: x>=y, binned, jnp.zeros(binned.shape))
    np.testing.assert_allclose(jnp.sum(binned, axis=-1), jnp.ones(logits.shape), rtol=0., atol=1e-6)
    # logits at bin centers are symmetric. `binned==binned[::-1, ::-1]`.
    np.testing.assert_allclose(binned, jnp.flip(binned, axis=(0,1)), rtol=0., atol=1e-6)
    # extreme values.
    logits = jnp.array([-1e6, 1e6])
    binned = gaussian_binning(logits=logits, breaks=self.breaks, rwidth=self.rwidth)
    self.assertGreater(binned[0,  0], 0.99) # -1e6 -> almost all mass in first bin
    self.assertGreater(binned[1, -1], 0.99) #  1e6 -> almost all mass in  last bin
    # empty.
    binned = gaussian_binning(jnp.array([]), breaks=self.breaks, rwidth=self.rwidth)
    self.assertEqual(binned.shape, (0, self.breaks.shape[0]+1))
    # deterministic.
    logits = np.random.random_sample(100)
    binned0 = gaussian_binning(logits=logits, breaks=self.breaks, rwidth=self.rwidth)
    binned1 = gaussian_binning(logits=logits, breaks=self.breaks, rwidth=self.rwidth)
    np.testing.assert_allclose(binned0, binned1, rtol=0., atol=1e-6)

  def test_grad(self):
    r"""Test that gaussian_binning is differentiable w.r.t. logits."""
    # probs.
    logits = .1 * jnp.arange(self.breaks.shape[0]+1) + 5e-2
    def loss(x: jnp.ndarray) -> jnp.ndarray:
      return jnp.sum(gaussian_binning(logits=x, breaks=self.breaks, rwidth=self.rwidth))
    grad = jax.grad(loss)(logits)
    self.assertEqual(grad.shape, logits.shape)
    self.assertTrue (jnp.all(jnp.isfinite(grad)))
