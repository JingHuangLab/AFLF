r"""Distances."""
# Authors: Zilin Song.


import jax.numpy as jnp


def dist_pair(cor0: jnp.ndarray, cor1: jnp.ndarray) -> jnp.ndarray:
  r"""Computes the pairwise 2-norm distances in `cor0` (N, 3) and `cor1` (M, 3).

    Args:
      cor0 (jnp.ndarray): The `cor0` coordinates set (N, 3).
      cor1 (jnp.ndarray): The `cor1` coordinates set (M, 3).

    Returns:
      dist (jnp.ndarray): The pairwise distances (N, M).
  """
  diff = jnp.expand_dims(cor0, axis=1) - jnp.expand_dims(cor1, axis=0)  # (N, M, 3)
  diff = jnp.sum(diff**2, axis=2)                                       # (N, M)
  # NOTE (ZS): Autograd to jnp.sqrt at zero yields `jnp.nan`.
  dist = jnp.sqrt(diff + (diff==0.)) - (diff==0.)
  return dist


def dist_self(cor: jnp.ndarray) -> jnp.ndarray:
  r"""Computes the self-pairwise 2-norm distances in `cor` (N, 3).

    Args:
      cor (jnp.ndarray): The `cor` coordinates set (N, 3)

    Returns:
      dist (jnp.ndarray): The self-pairwise distances (N, N).
  """
  return dist_pair(cor0=cor, cor1=cor)
