r"""Test cases for `aftools.utils.serialization`."""
# Authors: Zilin Song.


import typing
import json
import unittest

import jax.numpy as jnp
import numpy as np

from aftools.utils.serialization import JSONable


# helpers.
def gen_arr() -> np.ndarray:
  r"""Return a random float32 array for testing serialization."""
  return np.random.random_sample((100, 20)).astype(np.float32)


class InnerNamedTuple(typing.NamedTuple):
  r"""Named tuple used in nested serialization tests."""
  x = 'x'
  y = {'in_namedtuple': list((1, 2, 3)), }
  z = gen_arr()


class Inner(JSONable):
  r"""Inner JSONable class used in nested serialization tests."""
  _json_attr_ = ['jnp_arr', 'np_arr',
                 's', 'i', 'b', 'f', 'none',
                 'dct', 'lst', 'tpl', 'st', 'namedtpl', ]

  def __init__(self):
    self.s, self.i, self.b, self.f, self.none = "{txt}", 7, False, 3.14, None
    self.dct, self.lst, self.tpl, self.st = {'in': 1}, [2, 3], (4, 5), {6, 7}
    self.namedtpl = InnerNamedTuple()
    self.np_arr  = np.asarray(gen_arr())
    self.jnp_arr = jnp.asarray(gen_arr())


class OuterNamedTuple(typing.NamedTuple):
  r"""Outer named tuple used in nested serialization tests."""
  inner_named_tuple = InnerNamedTuple()
  inner_obj = Inner()


class Outer(JSONable):
  r"""Outer JSONable class used in round-trip serialization tests."""
  _json_attr_ = ['jnp_arr', 'np_arr',
                 's', 'i', 'b', 'f', 'none',
                 'dct', 'lst', 'tpl', 'st', 'namedtpl',
                 'inner0', 'inner1', ]

  def __init__(self):
    self.s, self.i, self.b, self.f, self.none = "{out}", 9, False, 2.71, None
    self.dct, self.lst, self.tpl, self.st = {'out': 8}, [9, 10], (11, 12), {13, 14}
    self.np_arr = np.asarray(gen_arr())
    self.jnp_arr = jnp.asarray(gen_arr())
    self.namedtpl = OuterNamedTuple()
    self.inner0, self.inner1 = Inner(), Inner()


def gen_jsonables() -> tuple[type[Inner], type[Outer]]:
  r"""Return the Inner and Outer JSONable classes used in tests."""
  return Inner, Outer


class Test_JSONable(unittest.TestCase):
  r"""Tests for `JSONable` serialization and deserialization."""

  def setUp(self):
    r"""Set up the Inner and Outer classes for each test."""
    self.Inner, self.Outer = gen_jsonables()

  def tearDown(self):
    r"""Clean up test fixtures."""
    del self.Inner
    del self.Outer

  def test_primitives(self):
    r"""Test round-trip of primitive attributes."""
    oo         = self.Outer()
    oo2: Outer = JSONable.from_json(oo.to_json())
    for attr in ['s', 'i', 'b', 'f', 'none']:
      self.assertEqual(getattr(oo, attr), getattr(oo2, attr))

  def test_np_array(self):
    r"""Test round-trip of a NumPy array attribute."""
    oo         = self.Outer()
    oo2: Outer = JSONable.from_json(oo.to_json())
    np.testing.assert_array_equal(oo.np_arr, oo2.np_arr)

  def test_jnp_array(self):
    r"""Test round-trip of a JAX array attribute."""
    oo         = self.Outer()
    oo2: Outer = JSONable.from_json(oo.to_json())
    self.assertTrue(jnp.array_equal(oo.jnp_arr, oo2.jnp_arr))

  def test_jsonable_nested(self):
    r"""Test round-trip of nested JSONable objects and their contents."""
    oo         = self.Outer()
    oo2: Outer = JSONable.from_json(oo.to_json())
    self.assertIsInstance(oo2.namedtpl, OuterNamedTuple)
    self.assertIsInstance(oo2.inner0.namedtpl, InnerNamedTuple)
    self.assertIsInstance(oo2.inner1.namedtpl, InnerNamedTuple)
    # primitives.
    for attr in ['s', 'i', 'b', 'f', 'none']:
      self.assertEqual(getattr(oo.inner0, attr), getattr(oo2.inner0, attr))
      self.assertEqual(getattr(oo.inner1, attr), getattr(oo2.inner1, attr))
    # containers.
    for attr in ['dct', 'lst', 'tpl', 'st', 'namedtpl']:
      self.assertEqual(getattr(oo.inner0, attr), getattr(oo2.inner0, attr))
      self.assertEqual(getattr(oo.inner1, attr), getattr(oo2.inner1, attr))
    # np.ndarray.
    np.testing.assert_array_equal(oo.inner0.np_arr, oo2.inner0.np_arr)
    np.testing.assert_array_equal(oo.inner1.np_arr, oo2.inner1.np_arr)
    # jnp.ndarray.
    np.testing.assert_array_equal(oo.inner0.jnp_arr, oo2.inner0.jnp_arr)
    np.testing.assert_array_equal(oo.inner1.jnp_arr, oo2.inner1.jnp_arr)

  def test_containers(self):
    r"""Test round-trip of container and named-tuple attributes."""
    oo         = self.Outer()
    oo2: Outer = JSONable.from_json(oo.to_json())
    self.assertIsInstance(oo2.namedtpl, OuterNamedTuple)
    self.assertIsInstance(oo2.inner0.namedtpl, InnerNamedTuple)
    self.assertIsInstance(oo2.inner1.namedtpl, InnerNamedTuple)
    for attr in ['dct', 'lst', 'tpl', 'st', 'namedtpl', ]:
      self.assertEqual(getattr(oo, attr), getattr(oo2, attr))

  def test_full_roundtrip(self):
    r"""Test round-trip through the concrete `Outer.from_json` classmethod."""
    oo         = self.Outer()
    oo2: Outer = Outer.from_json(oo.to_json())
    self.assertIsInstance(oo2, self.Outer)

  def test_valid_json_string(self):
    r"""Test that `to_json()` produces a valid JSON string."""
    oo = self.Outer()
    json_str = oo.to_json()
    parsed = json.loads(json_str) # should not raise.
    self.assertIsInstance(parsed, dict)
