r"""Test cases for `aftools.utils.selection`."""
# Authors: Zilin Song.


import os
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.getcwd())

import unittest

import numpy as np

import aftools.utils as _u
from aftools.utils.selection import AtomIndexerByAtomType, AtomIndexerGroup


# helpers.
_pdb_dir = os.path.join(os.path.dirname(__file__), "_dummy", "p20.pdb")


def _pdbfile() -> _u.mm.mm_app.PDBFile:
  r"""Return a dummy OpenMM PDBFile for a 20-mer peptide."""
  pdb_string = _u.io.txt.load(file=_pdb_dir)
  pdb_struct = _u.mm.mm_pdbstructure.PdbStructure(input_stream=pdb_string.split('\n'))
  pdb_file   = _u.mm.mm_app.PDBFile(file=pdb_struct)
  return pdb_file


def _mm_info() -> list[tuple[int, str, str, int]]:
  r"""Return a dummy list of `(atom_index, atom_name, residue_name, residue_index)` tuples."""
  mm_list = []
  for line in _u.io.txt.load(file=_pdb_dir).split('\n'):
    if line[:4]=='ATOM':
      words = line.split()
      mm_list.append((int(words[1])-1, str(words[2]), str(words[3]), int(words[4])-1 ))
  return mm_list


def _mm_indexes(residue_indexes: list[int], atom_type_names: list[str]) -> np.ndarray:
  r"""Return reference OpenMM atom indexes for the given selections."""
  indexes = [_[0] for _ in _mm_info() if _[1] in atom_type_names and _[3] in residue_indexes]
  return np.asarray(indexes)


def _af_batch() -> _u.af.TAFFeatures:
  r"""Return a dummy AF batch feature for a 20-mer peptide."""
  from alphafold.common.protein import residue_constants as _af_res
  res3 = list(res.name for res in _pdbfile().getTopology().residues())
  res1 = list(_af_res.restype_3to1[_] for _ in res3)
  aatype = np.asarray([_af_res.restype_order_with_x[_] for _ in res1])
  seq_mask = np.ones(aatype.shape)
  # make atom37_atom_exists.
  from alphafold.model.all_atom_multimer import get_atom37_mask
  atom37_atom_exists = get_atom37_mask(aatype=aatype)
  return dict(aatype=aatype, seq_mask=seq_mask, atom37_atom_exists=atom37_atom_exists)


def _af_info() -> list[tuple[int, str, str, int]]:
  r"""Return a dummy list of `(atom_index, atom_name, residue_name, residue_index)` tuples for AF."""
  from alphafold.common.protein import residue_constants as _af_res
  batch = _af_batch()
  ind_res3 = {v: _af_res.restype_1to3[k] for k, v in _af_res.restype_order.items()}
  nres = batch['aatype'].shape[0]
  _res_ids   = np.repeat(np.arange(nres), repeats=len(_u.af.AFAtomTypes)).tolist()
  _res_names = np.repeat(batch['aatype'], repeats=len(_u.af.AFAtomTypes))
  _res_names = [ind_res3[_] for _ in _res_names]
  _atm_names = np.repeat(np.arange(len(_u.af.AFAtomTypes))[None, :], repeats=nres, axis=0).flatten()
  _atm_names = [_u.af.AFIndexAtomType[_] for _ in _atm_names]
  _atm_ids   = np.arange(len(_atm_names)).tolist()
  return list(zip(_atm_ids, _atm_names, _res_names, _res_ids))


class Test_AtomIndexerByAtomType(unittest.TestCase):
  r"""Tests for `AtomIndexerByAtomType`."""

  def setUp(self):
    r"""Set up random residue and atom-type selections."""
    num_res, num_atm = np.random.randint(3, 20), np.random.randint(3, 37)
    self.residue_indexes = [np.random.randint(20)        for _ in range(num_res)]
    self.atom_type_names = [_u.af.AFIndexAtomType.get(_) for _ in range(num_atm)]

  def tearDown(self):
    r"""Clean up test fixtures."""
    del self.residue_indexes
    del self.atom_type_names

  def test_init_sanity(self):
    r"""Test that invalid inputs raise assertion errors."""
    with self.assertRaisesRegex(AssertionError, "Illegal residue index: `1.0`."):
      AtomIndexerByAtomType(residue_indexes=[1.], atom_type_names=['CA', 'N'])
    with self.assertRaisesRegex(AssertionError, "Illegal residue index: `-1`."):
      AtomIndexerByAtomType(residue_indexes=[-1], atom_type_names=['CA', 'N'])
    with self.assertRaisesRegex(AssertionError, "Illegal atom type name: `123`."):
      AtomIndexerByAtomType(residue_indexes=[0,1,2], atom_type_names=[123, 'O', 'CB'])
    with self.assertRaisesRegex(AssertionError, "Illegal atom type name: `XXX`."):
      AtomIndexerByAtomType(residue_indexes=[3,4,5], atom_type_names=['XXX', 'N'])

  def test_mm_indexes(self):
    r"""Test OpenMM atom indexes against reference selections."""
    ind = AtomIndexerByAtomType(residue_indexes=self.residue_indexes,
                                atom_type_names=self.atom_type_names, )
    # ref.
    indexes = _mm_indexes(residue_indexes=self.residue_indexes,
                          atom_type_names=self.atom_type_names, )
    self.assertEqual(indexes.shape, np.unique(indexes).shape)
    # test - indexes.
    indexes_selected = ind.mm_indexes(top=_pdbfile().getTopology())
    self.assertEqual(indexes_selected.shape, np.unique(indexes_selected).shape)
    np.testing.assert_array_equal(np.unique(indexes), np.unique(indexes_selected))

  def test_af_indexes(self):
    r"""Test AF atom indexes against OpenMM selections for monomer mode."""
    ind = AtomIndexerByAtomType(residue_indexes=self.residue_indexes,
                                atom_type_names=self.atom_type_names, )
    # take mm selections from index.
    mm_info          = _mm_info()
    mm_info_selected = [mm_info[_] for _ in ind.mm_indexes(top=_pdbfile().getTopology())]
    # take af selections from index.
    af_info          = _af_info()
    af_info_selected = [af_info[_] for _ in ind.af_indexes(batch=_af_batch(), is_multimer=False)]
    # test - indexes and ordering.
    self.assertEqual(len(mm_info_selected), len(af_info_selected))
    for (_, mm1, mm2, mm3), (_, af1, af2, af3) in list(zip(mm_info_selected, af_info_selected)):
      # atom indexes are in-equal b/w mm and af.
      self.assertEqual(mm1, af1)
      self.assertEqual(mm2, af2)
      self.assertEqual(mm3, af3)

  def test_af_onehots(self):
    r"""Test AF one-hot masks against AF index selections for monomer mode."""
    ind = AtomIndexerByAtomType(residue_indexes=self.residue_indexes,
                                atom_type_names=self.atom_type_names, )
    af_info = _af_info()
    # take af info from onehots.
    onehots = ind.af_onehots(batch=_af_batch(), is_multimer=False)
    af_info_onehots = [af_info[_] for _ in range(onehots.shape[0]) if onehots[_]==1]
    # take af info from indexes.
    indexes = ind.af_indexes(batch=_af_batch(), is_multimer=False)
    af_info_indexes = [af_info[_] for _ in indexes]
    # test - indexes and ordering.
    self.assertEqual(len(af_info_onehots), len(af_info_indexes))
    for (_, o1, o2, o3), (_, i1, i2, i3) in list(zip(af_info_onehots, af_info_indexes)):
      # atom indexes are in-equal b/w mm and af.
      self.assertEqual(o1, i1)
      self.assertEqual(o2, i2)
      self.assertEqual(o3, i3)

  def test_af_multimer_indexes(self):
    r"""Test AF atom indexes against OpenMM selections for multimer mode."""
    ind = AtomIndexerByAtomType(residue_indexes=self.residue_indexes,
                                atom_type_names=self.atom_type_names, )
    # take mm selections from index.
    mm_info          = _mm_info()
    mm_info_selected = [mm_info[_] for _ in ind.mm_indexes(top=_pdbfile().getTopology())]
    # take af selections from index.
    af_info          = _af_info()
    af_info_selected = [af_info[_] for _ in ind.af_indexes(batch=_af_batch(), is_multimer=True)]
    # test - indexes and ordering.
    self.assertEqual(len(mm_info_selected), len(af_info_selected))
    for (_, mm1, mm2, mm3), (_, af1, af2, af3) in list(zip(mm_info_selected, af_info_selected)):
      # atom indexes are in-equal b/w mm and af.
      self.assertEqual(mm1, af1)
      self.assertEqual(mm2, af2)
      self.assertEqual(mm3, af3)

  def test_af_multimer_onehots(self):
    r"""Test AF one-hot masks against AF index selections for multimer mode."""
    ind = AtomIndexerByAtomType(residue_indexes=self.residue_indexes,
                                atom_type_names=self.atom_type_names, )
    af_info = _af_info()
    # take af info from onehots.
    onehots = ind.af_onehots(batch=_af_batch(), is_multimer=True)
    af_info_onehots = [af_info[_] for _ in range(onehots.shape[0]) if onehots[_]==1]
    # take af info from indexes.
    indexes = ind.af_indexes(batch=_af_batch(), is_multimer=True)
    af_info_indexes = [af_info[_] for _ in indexes]
    # test - indexes and ordering.
    self.assertEqual(len(af_info_onehots), len(af_info_indexes))
    for (_, o1, o2, o3), (_, i1, i2, i3) in list(zip(af_info_onehots, af_info_indexes)):
      # atom indexes are in-equal b/w mm and af.
      self.assertEqual(o1, i1)
      self.assertEqual(o2, i2)
      self.assertEqual(o3, i3)


class Test_AtomIndexerGroup(unittest.TestCase):
  r"""Tests for `AtomIndexerGroup`."""

  def setUp(self):
    r"""Set up random atom indexers and groups."""
    n_atom_indexes = np.random.randint(2, 8)
    self.residue_indexes = [[np.random.randint(20) for _ in range(np.random.randint(3, 20))
                             ] for __ in range(n_atom_indexes)]
    self.atom_type_names = [[_u.af.AFIndexAtomType.get(_) for _ in range(np.random.randint(3, 37))
                             ] for __ in range(n_atom_indexes)]
    r1 = [       np.random.randint(20) for _ in range(np.random.randint(3, 20))]
    a1 = [_u.af.AFIndexAtomType.get(_) for _ in range(np.random.randint(3, 37))]
    self.indexer0 = AtomIndexerByAtomType(residue_indexes=r1, atom_type_names=a1)
    r2 = [       np.random.randint(20) for _ in range(np.random.randint(3, 20))]
    a2 = [_u.af.AFIndexAtomType.get(_) for _ in range(np.random.randint(3, 37))]
    self.indexer1 = AtomIndexerByAtomType(residue_indexes=r2, atom_type_names=a2)

  def tearDown(self):
    r"""Clean up test fixtures."""
    del self.residue_indexes
    del self.atom_type_names
    del self.indexer0
    del self.indexer1

  def test_append(self):
    r"""Test appending indexers to a group and rejecting invalid inputs."""
    g = AtomIndexerGroup()
    with self.assertRaisesRegex(AssertionError, "Illegal `indexer` type `<class 'str'>`."):
      g = g.append("not_an_indexer")
    g = g.append(self.indexer0)
    self.assertEqual(len(g), 1)
    self.assertIs(g[0], self.indexer0)

  def test_extend(self):
    r"""Test extending a group with multiple indexers."""
    g = AtomIndexerGroup().extend([self.indexer0, self.indexer1])
    self.assertIs(g[0], self.indexer0)
    self.assertIs(g[1], self.indexer1)
    self.assertEqual(list(g), [self.indexer0, self.indexer1])

  def test_af_onehots(self):
    r"""Test stacked one-hot masks for empty and non-empty groups."""
    # empty.
    g = AtomIndexerGroup()
    np.testing.assert_array_equal(g.af_onehots(batch=_af_batch(), is_multimer=False), np.empty(0))
    np.testing.assert_array_equal(g.af_onehots(batch=_af_batch(), is_multimer=True ), np.empty(0))
    # concat.
    indexers = [AtomIndexerByAtomType(residue_indexes=_[0], atom_type_names=_[1]
                            ) for _ in list(zip(self.residue_indexes, self.atom_type_names))]
    onehots_ref = np.asarray([i.af_onehots(batch=_af_batch(), is_multimer=False) for i in indexers])
    self.assertEqual(onehots_ref.shape, (len(indexers), _af_batch()['aatype'].shape[0]*37))
    g = g.extend(indexers=indexers)
    onehots_g = g.af_onehots(batch=_af_batch(), is_multimer=False)
    np.testing.assert_array_equal(onehots_ref, onehots_g)
