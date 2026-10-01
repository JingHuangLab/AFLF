r"""AFTools utilities for JSON serialization."""
# Authors: Zilin Song.


import json
import base64
import typing

import jax.numpy as jnp
import numpy as np


class JSONable:
  r"""Base class for JSON-(de)serializable objects."""

  _json_attr_: tuple[str, ...]
  r"""Tuple of attribute names to serialize to JSON."""

  def to_dict(self) -> dict:
    r"""Return a dictionary of the JSON-izable attributes."""
    return {_: getattr(self, _) for _ in self._json_attr_}

  def to_json(self) -> str:
    r"""Return a JSON string representation of this object."""
    return json.dumps(encode(obj=self.to_dict()), indent=2)

  @classmethod
  def from_dict(cls, d: dict) -> "JSONable":
    r"""Create an instance from a decoded dictionary.

      Args:
        d (dict): The decoded dictionary.

      Returns:
        obj (JSONable): An instance of this class.
    """
    o = cls.__new__(cls)
    for k, v in d.items():
      setattr(o, k, v)
    return o

  @classmethod
  def from_json(cls, s: str) -> "JSONable":
    r"""Create an instance from a JSON string.

      Args:
        s (str): The JSON string.

      Returns:
        obj (JSONable): An instance of this class.
    """
    d = decode(obj=json.loads(s))
    return cls.from_dict(d=d)


# JSON serialization implementations.
def is_primitive(obj: typing.Any) -> bool:
  r"""Return whether `obj` is a JSON primitive (bool, int, float, str, or None)."""
  return isinstance(obj, (bool, int, float, str)) or obj is None


def is_namedtuple(obj: typing.Any) -> bool:
  r"""Return whether `obj` is a named tuple."""
  return isinstance(obj, tuple) and hasattr(obj, '_asdict') and hasattr(obj, '_fields')


def auto_class(fully_qualified: str) -> typing.Type:
  r"""Return the class object for a fully-qualified name.

    Args:
      fully_qualified (str): The fully-qualified class name, e.g. `package.module.Class`.

    Returns:
      cls (type): The class object.
  """
  mod_name, cls_name = fully_qualified.rsplit('.', 1)
  mod = __import__(mod_name, fromlist=[cls_name])
  return getattr(mod, cls_name)


def encode(obj: typing.Union[bool, int, float, str, None,
                             np.ndarray, jnp.ndarray, JSONable,
                             list, tuple, set, typing.NamedTuple,
                             dict, ],
           ) -> typing.Union[bool, int, float, str, dict]:
  r"""Convert `obj` to a JSON-serializable representation.

    Args:
      obj (Any): The object to encode. Supports primitives, NumPy/JAX arrays,
        `JSONable` instances, containers, named tuples, and dicts.

    Returns:
      encoded (Union[bool, int, float, str, dict]): A JSON-serializable object.
  """
  if is_primitive(obj=obj):
    return obj
  if isinstance(obj, np.ndarray):
    return {'__np_array__' : True,
            '__np_dtype__' : str(obj.dtype),
            '__np_shape__' :     obj.shape ,
            '__meta_data__': base64.b64encode(obj.tobytes()).decode("ascii"), }
  if isinstance(obj, jnp.ndarray):
    return {'__jnp_array__': True,
            '__jnp_dtype__': str(obj.dtype),
            '__jnp_shape__':     obj.shape ,
            '__meta_data__': base64.b64encode(obj.tobytes()).decode("ascii"), }
  if isinstance(obj, JSONable):
    return {'__jsonable__' : True,
            '__json_name__': f"{obj.__class__.__module__}.{obj.__class__.__qualname__}",
            '__meta_data__': encode(obj.to_dict()), }
  if isinstance(obj, (list, tuple, set)):
    if is_namedtuple(obj=obj):            # NamedTuple -> tuple.
      return {'__namedtuple__': True,
              '__json_name__' : f"{obj.__class__.__module__}.{obj.__class__.__qualname__}",
              '__meta_data__' : encode({_: getattr(obj, _) for _ in obj._fields}), }
    assert type(obj).__name__ in ('list', 'tuple', 'set'), f"Illegal type `{type(obj)}: {obj}`."
    return {'__container__': type(obj).__name__,
            '__meta_data__': list(encode(obj=_) for _ in obj), }
  if isinstance(obj, dict):
    return {k: encode(v) for (k, v) in obj.items()}
  raise TypeError(f"Illegal `obj` of type `{type(obj)}` not json-able.")


def decode(obj: typing.Union[bool, int, float, str, None, dict]) -> typing.Any:
  r"""Reverse `encode()`: convert a JSON-decoded object back to the original types.

    Args:
      obj (Union[bool, int, float, str, None, dict]): The JSON-decoded object.

    Returns:
      decoded (Any): The reconstructed object.
  """
  if not isinstance(obj, dict):
    assert is_primitive(obj=obj), f"Illegal `obj` of type `{type(obj)}` not `decode()`-able."
    return obj  # primitives if not dict.
  if obj.get('__np_array__'):
    return np.frombuffer(buffer=base64.b64decode(obj['__meta_data__'].encode("ascii")),
                         dtype =np.dtype(obj['__np_dtype__']), ).reshape(obj['__np_shape__'])
  if obj.get('__jnp_array__'):
    return jnp.frombuffer(buffer=base64.b64decode(obj['__meta_data__'].encode("ascii")),
                          dtype =jnp.dtype(obj['__jnp_dtype__']), ).reshape(obj['__jnp_shape__'])
  if obj.get('__jsonable__'):
    return auto_class(obj['__json_name__']).from_dict(d=decode(obj=obj['__meta_data__']))
  if obj.get('__namedtuple__'):
    return auto_class(obj['__json_name__'])(**decode(obj=obj['__meta_data__']))
  if obj.get('__container__'):
    cls_container = {'list': list, 'tuple': tuple, 'set': set}[obj['__container__']]
    return cls_container(decode(_) for _ in obj['__meta_data__'])
  return {k: decode(obj=v) for (k, v) in obj.items()}
