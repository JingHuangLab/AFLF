r"""Test cases for `aftools.lora._lora.py`."""
# Authors: Zilin Song


import os
import sys
sys.dont_write_bytecode=True
sys.path.insert(0, os.getcwd())
import typing

import copy
import json
import unittest
import parameterized

import jax
import jax.numpy as jnp
import     numpy as  np

from aftools.lora._lora import (_mask, _mask_maximum,
                                _init,
                                _offset,
                                _lora, _lora_structmod, _lora_evoformer,
                                LoraCheckpointPreset, )


def ref_lora_masks(x: jnp.ndarray, topk: int) -> jnp.ndarray:
  # generate reference mask - argsort() sorts from small to large so we take the negative.
  inds = jnp.argsort(-x, axis=None, kind='stable')[:topk] # = jnp.argsort(x)[::-1][:2]
  mask = jnp.ones((x.size, )).at[inds].set(0)
  return jnp.asarray(mask.reshape(*x.shape), dtype=int)


class Test_mask(unittest.TestCase):
  r"""Tests for _lora._mask()."""

  def setUp(self):
    k0         = jax.random.PRNGKey(seed=np.random.randint(low=-99999, high=100000))
    k0, k1, k2 = jax.random.split(key=k0, num=3)
    d0, d1, d2 = tuple(np.random.randint(low=12, high=34) for _ in range(3))
    self.x = jax.random.uniform(key=k0, shape=(d0, d1, d2), minval=-100., maxval=100.)
    self.reprs = {
      'repr_single': jax.random.uniform(key=k0, shape=(d0, d1    ), minval=-100, maxval=100.),
      'repr_msa':    jax.random.uniform(key=k1, shape=(d0, d1, d2), minval=-100, maxval=100.),
      'repr_pair':   jax.random.uniform(key=k2, shape=(d0, d1, d2), minval=-100, maxval=100.), }

  def tearDown(self):
    del self.x
    del self.reprs

  def test_init_sanity(self):
    r"""Test _mask_maximum raises on illegal k values."""
    with self.assertRaisesRegex(AssertionError,  "Illegal `k=None`."): # k is None.
      _mask_maximum(x=self.x, k='None')
    with self.assertRaisesRegex(AssertionError,  "Illegal `k=-1`."):   # k>=0.
      _mask_maximum(x=self.x, k=-1)
    with self.assertRaisesRegex(AssertionError, f"Illegal `k={self.x.size+123}`."): # k<=x.size.
      _mask_maximum(x=self.x, k=self.x.size+123)
    with self.assertRaisesRegex(AssertionError,  "Illegal `k=45.0`."): # k is int.
      _mask_maximum(x=self.x, k=45.)
    with self.assertRaisesRegex(AssertionError,  "Illegal `k=-6`."): # k > 0.
      _mask_maximum(x=self.x, k=-6)
    with self.assertRaisesRegex(AssertionError, f"Illegal `k={self.x.size+789}`."): # k < x.shape.
      _mask_maximum(x=self.x, k=self.x.size+789)

  def test_mask_maximum_none(self):
    r"""Test _mask_maximum with k=None or 0 returns an all-ones mask."""
    # test.
    mask = _mask_maximum(x=self.x, k=None)
    np.testing.assert_array_equal(mask, jnp.ones(self.x.shape))
    mask = _mask_maximum(x=self.x, k=0)
    np.testing.assert_array_equal(mask, jnp.ones(self.x.shape))

  def test_mask_maximum_int(self):
    r"""Test _mask_maximum against a reference argsort-based top-k mask."""
    # prep.
    thresh = np.random.randint(low=1, high=self.x.size)
    # ref.
    mask_ref = ref_lora_masks(x=self.x, topk=thresh)
    # test.
    mask = _mask_maximum(x=self.x, k=thresh)
    np.testing.assert_array_equal(mask, mask_ref)

  @parameterized.parameterized.expand([('single',  None,  None),
                                       ('single', 'int',  None),
                                       ('single',  None, 'int'),
                                       ('single', 'int', 'int'),
                                       (   'msa',  None,  None),
                                       (   'msa', 'int',  None),
                                       (   'msa',  None, 'int'),
                                       (   'msa', 'int', 'int'),
                                       (  'pair',  None,  None),
                                       (  'pair', 'int',  None),
                                       (  'pair',  None, 'int'),
                                       (  'pair', 'int', 'int'), ])
  def test_mask_to_str(self,
                       mask_to:      typing.Literal['single', 'msa', 'pair'],
                       mask_thresh0: typing.Literal[None, 'int'],
                       mask_thresh1: typing.Literal[None, 'int'], ):
    r"""Test _mask for a single representation with min/max masks."""
    # prep.
    x = self.reprs[f'repr_{mask_to}']
    thresh0 = np.random.randint(low=5, high=x.size) if mask_thresh0=='int' else None
    thresh1 = np.random.randint(low=5, high=x.size) if mask_thresh1=='int' else None
    # test.
    lora_masks = _mask(reprs=self.reprs, mask_to=mask_to, mask_mink=thresh0, mask_maxk=thresh1)
    self.assertEqual(len(lora_masks.keys()), 1)
    self.assertIn(f'lora_mask_{mask_to}', lora_masks.keys())
    mask = lora_masks[f'lora_mask_{mask_to}']
    self.assertEqual(mask.shape, x.shape)
    # ref.
    mask_ref0 = jnp.ones(x.shape) if thresh0 is None else ref_lora_masks(x=-x, topk=thresh0)
    mask_ref1 = jnp.ones(x.shape) if thresh1 is None else ref_lora_masks(x= x, topk=thresh1)
    mask_ref  = ((mask_ref0+mask_ref1)>1.5)  # residual-stable sum==2.
    # test.
    np.testing.assert_allclose(mask, mask_ref, rtol=0., atol=1e-5)

  @parameterized.parameterized.expand([(['single', 'pair'], ),
                                       (['msa',    'pair'], ), ])
  def test_mask_to_list(self, mask_to: list[str]):
    r"""Test _mask for a list of representations returns separate masks."""
    # prep.
    mask_to_spec = 'single' if 'single' in mask_to else 'msa' if 'msa' in mask_to else None
    self.assertTrue(not mask_to_spec is None)
    spec_size = self.reprs[f'repr_{mask_to_spec}'].size
    pair_size = self.reprs[f'repr_pair'          ].size
    mask_mink = [(None, np.random.randint(low=1, high=spec_size))[np.random.randint(low=0,high=2)],
                 (None, np.random.randint(low=1, high=pair_size))[np.random.randint(low=0,high=2)],]
    mask_maxk = [(None, np.random.randint(low=1, high=spec_size))[np.random.randint(low=0,high=2)],
                 (None, np.random.randint(low=1, high=pair_size))[np.random.randint(low=0,high=2)],]
    # test.
    lora_masks = _mask(reprs=self.reprs, mask_to=mask_to, mask_mink=mask_mink, mask_maxk=mask_maxk)
    self.assertEqual(len(lora_masks.keys()), 2)
    self.assertIn(f'lora_mask_{mask_to_spec}', lora_masks.keys())
    self.assertIn(f'lora_mask_pair',           lora_masks.keys())
    self.assertEqual(lora_masks[f'lora_mask_{mask_to_spec}'].shape,
                     self.reprs[f'repr_{mask_to_spec}'     ].shape, )
    self.assertEqual(lora_masks[f'lora_mask_pair'          ].shape,
                     self.reprs[f'repr_pair'               ].shape, )


class Test_init(unittest.TestCase):
  r"""Test for _lora._init()."""

  def setUp(self):
    k0         = jax.random.PRNGKey(seed=np.random.randint(low=-99999, high=100000))
    k0, k1, k2 = jax.random.split(key=k0, num=3)
    d0, d1, d2 = tuple(np.random.randint(low=12, high=34) for _ in range(3))
    self.reprs = {
      'repr_single': jax.random.uniform(key=k0, shape=(d0, d1    ), minval=-100, maxval=100.),
      'repr_msa':    jax.random.uniform(key=k1, shape=(d0, d1, d2), minval=-100, maxval=100.),
      'repr_pair':   jax.random.uniform(key=k2, shape=(d0, d1, d2), minval=-100, maxval=100.), }
    self.lora_feats = dict()
    self.init_rank  = np.random.randint(low=1, high=11)
    self.init_scale = np.random.uniform(low=0., high=1.)
    self.key = jax.random.PRNGKey(seed=np.random.randint(low=-99999, high=100000))

  def tearDown(self):
    del self.reprs
    del self.lora_feats
    del self.init_rank
    del self.init_scale
    del self.key

  def test_init_sanity(self):
    r"""Test _init raises on illegal init_to, init_from, init_rank, or init_scale."""
    # sane inputs.
    init_to   = ['single', 'msa', 'pair'][np.random.randint(low=0, high=3)]
    init_from = ['normal', 'uniform'    ][np.random.randint(low=0, high=2)]
    init_rank = np.random.randint(low=1 , high=10)
    init_scale= np.random.uniform(low=0., high=1.)
    with self.assertRaisesRegex(AssertionError, f"Illegal `init_to={init_to.capitalize()}`."):
      _init(reprs=self.reprs,
            init_to   =init_to.capitalize(),
            init_from =init_from,
            init_rank =init_rank,
            init_scale=init_scale,
            key=self.key, )
    with self.assertRaisesRegex(AssertionError, f"Illegal `init_from={init_from.capitalize()}`."):
      _init(reprs=self.reprs,
            init_to   =init_to,
            init_from =init_from.capitalize(),
            init_rank =init_rank,
            init_scale=init_scale,
            key=self.key, )
    with self.assertRaisesRegex(AssertionError, f"Illegal `init_rank=10.0`."):
      _init(reprs=self.reprs,
            init_to   =init_to,
            init_from =init_from,
            init_rank =10.,
            init_scale=init_scale,
            key=self.key, )
    with self.assertRaisesRegex(AssertionError, f"Illegal `init_rank=-3`."):
      _init(reprs=self.reprs,
            init_to   =init_to,
            init_from =init_from,
            init_rank =-3,
            init_scale=init_scale,
            key=self.key, )
    with self.assertRaisesRegex(AssertionError, f"Illegal `init_scale=3`."):
      _init(reprs=self.reprs,
            init_to   =init_to,
            init_from =init_from,
            init_rank =init_rank,
            init_scale=3,
            key=self.key, )
    with self.assertRaisesRegex(AssertionError, f"Illegal `init_scale=-0.5`."):
      _init(reprs=self.reprs,
            init_to   =init_to,
            init_from =init_from,
            init_rank =init_rank,
            init_scale=-.5,
            key=self.key, )

  @parameterized.parameterized.expand([('single', 'uniform'), ('single',  'normal'),
                                       (   'msa', 'uniform'), (   'msa',  'normal'),
                                       (  'pair', 'uniform'), (  'pair',  'normal'), ] )
  def test_init_to_str(self,
                       init_to:   typing.Literal['single', 'msa', 'pair'],
                       init_from: typing.Literal['uniform', 'normal'], ):
    r"""Test _init returns LoRA tensors of expected keys and shapes."""
    # test.
    lora_feats = _init(reprs=self.reprs,
                       init_to  =init_to,
                       init_from=init_from,
                       init_rank=self.init_rank,
                       init_scale=self.init_scale,
                       key=self.key, )
    lora_feat = lora_feats[f'lora_feat_{init_to}']
    self.assertIn(f'lora_feat_{init_to}', lora_feats                        .keys())
    self.assertIn('A',                    lora_feats[f'lora_feat_{init_to}'].keys())
    self.assertIn('B',                    lora_feats[f'lora_feat_{init_to}'].keys())
    if init_to in ['single']:
      self.assertEqual(len(lora_feats[f'lora_feat_{init_to}'].keys()), 2)
    if init_to in ['pair', 'msa']:
      self.assertIn('C', lora_feats[f'lora_feat_{init_to}'].keys())
      self.assertEqual(len(lora_feats[f'lora_feat_{init_to}'].keys()), 3)
    # ref.
    num_vecs = 2 if init_to in ['single'] else 3 if init_to in ['pair', 'msa'] else None
    self.assertTrue(not num_vecs is None)
    shape = self.reprs[f'repr_{init_to}'].shape
    # test.
    self.assertEqual(len(lora_feats[f'lora_feat_{init_to}'].items()), num_vecs)
    self.assertEqual(  (                shape[0], self.init_rank), lora_feat['A'].shape)
    if init_to in ['single', ]:
      self.assertEqual((self.init_rank, shape[1],               ), lora_feat['B'].shape)
    if init_to in ['pair', 'msa']:
      self.assertEqual((self.init_rank, shape[1], self.init_rank), lora_feat['B'].shape)
      self.assertEqual((self.init_rank, shape[2]                ), lora_feat['C'].shape)

  @parameterized.parameterized.expand([(['single', 'pair'], ),
                                       (['msa',    'pair'], ), ])
  def test_init_to_list(self, init_to: list[str]):
    r"""Test _init for a list of representations."""
    # test.
    lora_feats = _init(reprs=self.reprs,
                       init_to  =init_to,
                       init_from=['uniform', 'normal'][np.random.randint(low=0, high=2)],
                       init_rank=self.init_rank,
                       init_scale=self.init_scale,
                       key=self.key, )
    for act_to in init_to:
      self.assertEqual(len(lora_feats.keys()), 2)
      self.assertIn(f'lora_feat_{act_to}', lora_feats.keys())
      # check keys in each lora feature.
      lora_feat = lora_feats[f'lora_feat_{act_to}']
      self.assertIn('A', lora_feat.keys())
      self.assertIn('B', lora_feat.keys())
      if act_to in ['single']:
        self.assertEqual(len(lora_feat.keys()), 2)
      if act_to in ['msa', 'pair']:
        self.assertIn('C', lora_feat.keys())
        self.assertEqual(len(lora_feat.keys()), 3)
      # check shape in each lora feature.
      shape = self.reprs[f'repr_{act_to}'].shape
      self.assertEqual(  (                shape[0], self.init_rank), lora_feat['A'].shape)
      if   act_to in ['single', ]:
        self.assertEqual((self.init_rank, shape[1],               ), lora_feat['B'].shape)
      elif act_to in ['msa', 'pair']:
        self.assertEqual((self.init_rank, shape[1], self.init_rank), lora_feat['B'].shape)
        self.assertEqual((self.init_rank, shape[2]                ), lora_feat['C'].shape)
      else:
        self.assertTrue(False, msg=f'Illegal `{act_to} in init_to`.')

class Test_offset(unittest.TestCase):
  r"""Test of _lora._offset()."""

  def setUp(self):
    k0 = jax.random.PRNGKey(seed=np.random.randint(low=-99999, high=100000))
    k0, k1, k2, k3 = jax.random.split(key=k0, num=4)
    d0, d1, d2 = tuple(np.random.randint(low=12, high=34) for _ in range(3))
    self.reprs = {
      'repr_single': jax.random.uniform(key=k0, shape=(d0, d1    ), minval=-100, maxval=100.),
      'repr_msa':    jax.random.uniform(key=k1, shape=(d0, d1, d2), minval=-100, maxval=100.),
      'repr_pair':   jax.random.uniform(key=k2, shape=(d0, d1, d2), minval=-100, maxval=100.), }
    self.lora_feats = _init(reprs=self.reprs,
                            init_to   =['single', 'msa', 'pair'],
                            init_from =['uniform', 'normal'][np.random.randint(low=0, high=2)],
                            init_rank =np.random.randint(low=1, high=11),
                            init_scale=np.random.uniform(low=0., high=1.),
                            key=k3, )
    s_size = self.reprs[f'repr_single'].size
    m_size = self.reprs[f'repr_msa'   ].size
    p_size = self.reprs[f'repr_pair'  ].size
    mask_mink = [(None, np.random.randint(low=1, high=s_size))[np.random.randint(low=0,high=2)],
                 (None, np.random.randint(low=1, high=m_size))[np.random.randint(low=0,high=2)],
                 (None, np.random.randint(low=1, high=p_size))[np.random.randint(low=0,high=2)], ]
    mask_maxk = [(None, np.random.randint(low=1, high=s_size))[np.random.randint(low=0,high=2)],
                 (None, np.random.randint(low=1, high=m_size))[np.random.randint(low=0,high=2)],
                 (None, np.random.randint(low=1, high=p_size))[np.random.randint(low=0,high=2)], ]
    self.lora_masks = _mask(reprs=self.reprs,
                            mask_to  =['single', 'msa', 'pair'],
                            mask_mink=mask_mink,
                            mask_maxk=mask_maxk, )

  def tearDown(self):
    del self.reprs
    del self.lora_feats
    del self.lora_masks

  def test_init_sanity(self):
    r"""Test _offset raises on illegal offset_to."""
    offset_to = ['single', 'msa', 'pair'][np.random.randint(low=0, high=3)]
    with self.assertRaisesRegex(AssertionError, f"Illegal `offset_to={offset_to.capitalize()}`."):
      _offset(reprs=self.reprs,
              lora_feats=self.lora_feats,
              lora_masks=self.lora_masks,
              offset_to=offset_to.capitalize(), )

  @parameterized.parameterized.expand([('single', 'ai,ib->ab'     ),
                                       (   'msa', 'ai,ibj,jc->abc'),
                                       (  'pair', 'ai,ibj,jc->abc'), ])
  def test_offset_to_str(self,
                         offset_to:   typing.Literal['single', 'msa', 'pair'],
                         einsum_expr: typing.Literal['ai,ib->ab', 'ai,ibj,jc->abc'], ):
    r"""Test _offset for a single representation inverts the LoRA product."""
    # ref.
    repr_ref = self.reprs[f'repr_{offset_to}']
    # test.
    reprs = _offset(reprs     =copy.deepcopy(self.reprs),
                    lora_feats=self.lora_feats,
                    lora_masks=self.lora_masks,
                    offset_to=offset_to, )
    lora_v = self.lora_feats[f'lora_feat_{offset_to}'].values()
    lora_m = self.lora_masks[f'lora_mask_{offset_to}']
    repr_v = reprs[f'repr_{offset_to}'] + jnp.einsum(einsum_expr, *lora_v) * lora_m
    np.testing.assert_allclose(repr_v, repr_ref, rtol=0., atol=1e-5)

  @parameterized.parameterized.expand([(['single', 'pair'], ),
                                       (['msa',    'pair'], ), ])
  def test_offset_to_list(self, offset_to: list[str]):
    r"""Test _offset for a list of representations."""
    # test.
    reprs = _offset(reprs     =copy.deepcopy(self.reprs),
                    lora_feats=self.lora_feats,
                    lora_masks=self.lora_masks,
                    offset_to=offset_to, )
    for act_to in offset_to:
      ## ref.
      repr_ref = self.reprs[f'repr_{act_to}']
      ## test.
      offset_expr = ('ai,ib->ab'      if act_to in ['single',    ] else
                     'ai,ibj,jc->abc' if act_to in ['msa', 'pair'] else 'Wrong expr.')
      lora_v = self.lora_feats[f'lora_feat_{act_to}'].values()
      lora_m = self.lora_masks[f'lora_mask_{act_to}']
      repr_v = reprs[f'repr_{act_to}'] + jnp.einsum(offset_expr, *lora_v) * lora_m
      np.testing.assert_allclose(repr_v, repr_ref, rtol=0., atol=1e-5)


class Test_lora(unittest.TestCase):
  r"""Test of _lora._lora()."""

  def setUp(self):
    k0 = jax.random.PRNGKey(seed=np.random.randint(low=-99999, high=100000))
    k0, k1, k2, k3 = jax.random.split(key=k0, num=4)
    d0, d1, d2 = tuple(np.random.randint(low=12, high=34) for _ in range(3))
    self.reprs = {
      'repr_single': jax.random.uniform(key=k0, shape=(d0, d1    ), minval=-100, maxval=100.),
      'repr_msa':    jax.random.uniform(key=k1, shape=(d0, d1, d2), minval=-100, maxval=100.),
      'repr_pair':   jax.random.uniform(key=k2, shape=(d0, d1, d2), minval=-100, maxval=100.), }
    self.lora_feats = _init(reprs=self.reprs,
                            init_to   =['single', 'msa', 'pair'],
                            init_from =['uniform', 'normal'][np.random.randint(low=0, high=2)],
                            init_rank =np.random.randint(low=1, high=11),
                            init_scale=np.random.uniform(low=0., high=1.),
                            key=k3, )
    s_size = self.reprs[f'repr_single'].size
    m_size = self.reprs[f'repr_msa'   ].size
    p_size = self.reprs[f'repr_pair'  ].size
    mask_mink = [(None, np.random.randint(low=1, high=s_size))[np.random.randint(low=0,high=2)],
                 (None, np.random.randint(low=1, high=m_size))[np.random.randint(low=0,high=2)],
                 (None, np.random.randint(low=1, high=p_size))[np.random.randint(low=0,high=2)], ]
    mask_maxk = [(None, np.random.randint(low=1, high=s_size))[np.random.randint(low=0,high=2)],
                 (None, np.random.randint(low=1, high=m_size))[np.random.randint(low=0,high=2)],
                 (None, np.random.randint(low=1, high=p_size))[np.random.randint(low=0,high=2)], ]
    self.lora_masks = _mask(reprs=self.reprs,
                            mask_to  =['single', 'msa', 'pair'],
                            mask_mink=mask_mink,
                            mask_maxk=mask_maxk, )

  def tearDown(self):
    del self.reprs
    del self.lora_feats
    del self.lora_masks

  @parameterized.parameterized.expand([('single', 'ai,ib->ab'     ),
                                       (   'msa', 'ai,ibj,jc->abc'),
                                       (  'pair', 'ai,ibj,jc->abc'), ])
  def test_lora_to_str(self,
                       lora_to:     typing.Literal['single', 'msa', 'pair'],
                       einsum_expr: typing.Literal['ai,ib->ab', 'ai,ibj,jc->abc'], ):
    r"""Test _lora for a single representation adds the LoRA product."""
    # ref.
    repr_ref = self.reprs[f'repr_{lora_to}']
    # test.
    reprs = _lora(lora_feats=self.lora_feats,
                  lora_masks=self.lora_masks,
                  reprs     =copy.deepcopy(self.reprs),
                  lora_to=lora_to, )
    lora_v = self.lora_feats[f'lora_feat_{lora_to}'].values()
    lora_m = self.lora_masks[f'lora_mask_{lora_to}']
    repr_v = reprs[f'repr_{lora_to}'] - jnp.einsum(einsum_expr, *lora_v) * lora_m
    np.testing.assert_allclose(repr_v, repr_ref, rtol=0., atol=1e-5)

  @parameterized.parameterized.expand([('structmod', ),
                                       ('evoformer', ), ])
  def test_lora_to_both(self, lora_to: typing.Literal['structmod', 'evoformer']):
    r"""Test _lora_structmod and _lora_evoformer apply LoRA to both representations."""
    # prep.
    lora_to_spec = 'single' if lora_to=='structmod' else 'msa' if lora_to=='evoformer' else None
    self.assertTrue(not lora_to_spec is None)
    f_lora = {'structmod': _lora_structmod, 'evoformer': _lora_evoformer}[lora_to]
    # test.
    reprs = f_lora(lora_feats=self.lora_feats,
                   lora_masks=self.lora_masks,
                   reprs     =copy.deepcopy(self.reprs), )
    for act_to in [lora_to_spec, 'pair']:
      ## ref.
      repr_ref = self.reprs[f'repr_{act_to}']
      ## test.
      exec_expr = ('ai,ib->ab'      if act_to in ['single',    ] else
                   'ai,ibj,jc->abc' if act_to in ['msa', 'pair'] else 'Wrong expr.')
      lora_v = self.lora_feats[f'lora_feat_{act_to}'].values()
      lora_m = self.lora_masks[f'lora_mask_{act_to}']
      repr_v = reprs[f'repr_{act_to}'] - jnp.einsum(exec_expr, *lora_v) * lora_m
      np.testing.assert_allclose(repr_v, repr_ref, rtol=0., atol=1e-5)
