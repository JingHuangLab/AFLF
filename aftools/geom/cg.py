r"""Coordinates coarse-graining."""
# Authors: Zilin Song.


import jax
import jax.numpy as jnp


def centroid(cor: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
  r"""JAX-differentiable implementation of computing the centroid of the one-hot selected atoms.

    Args:
      cor  (jnp.ndarray): The atom coordinates (N, 3).
      mask (jnp.ndarray): The one-hot mask that selects the subset of atoms (N, ).

    Returns:
      centroid (jnp.ndarray): The centroid of selected atom coordinates (3, ).
  """
  nele = jnp.sum(mask)
  cent = jnp.einsum('w,wx->x', mask, cor) / (nele + (nele == 0))
  return cent


def centroid_batched(cor: jnp.ndarray, masks: jnp.ndarray) -> jnp.ndarray:
  r"""JAX-differentiable implementation of computing the centroids of the one-hot selected atoms on
    the batched one-hot flags.

    Args:
      cor   (jnp.ndarray): The atom coordinates (N, 3).
      masks (jnp.ndarray): The batched one-hot masks that select the subset of atoms (Ngroups, N).

    Returns:
      centroids (jnp.ndarray): The centroids of selected atom coordinates (Ngroups, 3).
  """
  return jax.vmap(fun=centroid, in_axes=(None, 0))(cor, masks)
