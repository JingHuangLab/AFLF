r"""Test cases for `aftools.geom.spatial.py`."""
# Authors: Zilin Song


import os
import sys
sys.dont_write_bytecode=True
sys.path.insert(0, os.getcwd())

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from aftools.geom.spatial import kabsch, kabsch_rotr


class Test_kabsch(unittest.TestCase):
  r"""Tests for geom.kabsch()."""

  def setUp(self):
    self.to_refer = (np.random.random_sample((100, 3))-.5)*np.random.randint(5, 100)
    self.to_align = (np.random.random_sample((100, 3))-.5)*np.random.randint(5, 100)
    mask = np.zeros((100, ))
    mask[:np.random.randint(10, 90)] = 1
    self.mask = np.random.permutation(mask)
    self.to_refer_masked = self.to_refer[self.mask==1]
    self.to_align_masked = self.to_align[self.mask==1]

  def tearDown(self):
    del self.to_refer
    del self.to_align
    del self.mask
    del self.to_refer_masked
    del self.to_align_masked

  def test_kabsch(self):
    r"""Test kabsch and kabsch_rotr on a rigidly transformed reference."""
    # prep.
    V, _, W = np.linalg.svd(np.random.random_sample((3, 3)))
    V[:, -1] *= np.sign((np.linalg.det(V @ W)))
    rot_true = V @ W
    trans_true = np.random.random_sample((3, ))
    mob: np.ndarray = self.to_refer @ rot_true + trans_true
    # test.
    rot, trans = kabsch     (to_refer=self.to_refer, to_align=mob, mask=self.mask)
    aligned    = kabsch_rotr(to_refer=self.to_refer, to_align=mob, mask=self.mask)
    # NOTE (ZS):
    # Since:
    #   aligned =  to_align @ rot + trans
    #           = (to_refer @ rot_true + trans_true) @ rot + trans
    #           =  to_refer @ rot_true @ rot + trans_true @ rot + trans
    #           =  to_refer
    # we have:
    #   to_refer = aligned
    #   rot_true @ rot == np.eye(3)
    #   trans_true @ rot + trans == 0
    np.testing.assert_array_less(np.abs(self.to_refer - np.asarray(aligned)        ), 1e-4)
    np.testing.assert_array_less(np.abs(rot_true @ np.asarray(rot) - np.eye(3)), 1e-4)
    np.testing.assert_array_less(np.abs(trans_true @ rot + trans              ), 1e-4)

  def test_kabsch_scipy(self):
    r"""Test kabsch against scipy.spatial.transform.Rotation alignment."""
    # prep.
    from scipy.spatial.transform import Rotation
    ref_mask_origin = self.to_refer_masked - np.mean(self.to_refer_masked, axis=0)
    mob_mask_origin = self.to_align_masked - np.mean(self.to_align_masked, axis=0)
    mob_origin      = self.to_align      - np.mean(self.to_align_masked, axis=0)
    rot_scipy, trans = Rotation.align_vectors(a=ref_mask_origin, b=mob_mask_origin)
    aligned_scipy = rot_scipy.apply(mob_origin) + np.mean(self.to_refer_masked, axis=0)
    # test.
    rot, trans = kabsch(to_refer=self.to_refer, to_align=self.to_align, mask=self.mask)
    aligned = jnp.dot(jnp.asarray(self.to_align), rot) + trans
    np.testing.assert_array_less(np.abs(rot_scipy.as_matrix()@np.asarray(rot) - np.eye(3)), 1e-4)
    np.testing.assert_array_less(np.abs( np.asarray(aligned) - aligned_scipy),              1e-4)

  def test_grad_kabsch(self):
    r"""Test kabsch outputs have zero gradient w.r.t. input coordinates."""
    # test: no `rot` gradient backprop. to `to_refer` and `to_align`.
    ro_f = lambda r, a, m: jnp.sum(kabsch(to_refer=r, to_align=a, mask=m)[0])
    g_to_refer = jax.grad(ro_f, argnums=0)(self.to_refer, self.to_align, self.mask)
    g_to_align = jax.grad(ro_f, argnums=1)(self.to_refer, self.to_align, self.mask)
    self.assertEqual(jnp.sum(g_to_align), jnp.sum(g_to_refer))
    self.assertEqual(jnp.sum(g_to_align), 0.)
    # test: no `trans` gradient backprop. to `to_refer` and `to_align`.
    tr_f = lambda r, a, m: jnp.sum(kabsch(to_refer=r, to_align=a, mask=m)[1])
    g_to_refer = jax.grad(tr_f, argnums=0)(self.to_refer, self.to_align, self.mask)
    g_to_align = jax.grad(tr_f, argnums=1)(self.to_refer, self.to_align, self.mask)
    self.assertEqual(jnp.sum(g_to_align), jnp.sum(g_to_refer))
    self.assertEqual(jnp.sum(g_to_align), 0.)
