r"""Spatial transforms."""
# Authors: Zilin Song.


import jax
import jax.numpy as jnp

import aftools.geom as _g


def kabsch(to_refer: jnp.ndarray,
           to_align: jnp.ndarray,
           mask:     jnp.ndarray,
           ) -> tuple[jnp.ndarray, jnp.ndarray]:
  r"""Computes the rotational and translational operators using the Kabsch algorithm that aligns the
    `mask`-ed atoms in the `to_align` set onto the `to_refer` set of coordinates. This module stops
    gradients for all inputs.

    Args:
      to_refer (jnp.ndarray): The coordinates to be referred (N, 3).
      to_align (jnp.ndarray): The coordinates to be  aligned (N, 3).
      mask     (jnp.ndarray): The one-hot mask that selects the subset of atoms (N, ).

    Returns:
      rot   (jnp.ndarray): The    rotation operator (3, 3).
      trans (jnp.ndarray): The translation operator (3, ).
  """
  to_refer, to_align = jax.lax.stop_gradient(to_refer), jax.lax.stop_gradient(to_align)
  # computes the subset centroids.
  to_refer_centroid = _g.cg.centroid(cor=to_refer, mask=mask)
  to_align_centroid = _g.cg.centroid(cor=to_align, mask=mask)
  # translate to origin and compute the correlation matrix.
  to_refer_origin = (to_refer - to_refer_centroid) * mask.reshape(-1, 1)
  to_align_origin = (to_align - to_align_centroid) * mask.reshape(-1, 1)
  # Kabsch.
  V, _, W = jnp.linalg.svd( jnp.dot(jnp.transpose(to_align_origin), to_refer_origin) )
  V = V.at[:, -1].set( V[:, -1]*jnp.sign(jnp.linalg.det(jnp.dot(V, W))) )     # handles reflection.
  # pack results.
  rot   = jnp.dot(V, W)
  trans = jnp.dot(-to_align_centroid, rot) + to_refer_centroid
  return rot, trans


def rotr(cor: jnp.ndarray, rot: jnp.ndarray, trans: jnp.ndarray) -> jnp.ndarray:
  r"""Rotates and translates the `to_align` set.

    Args:
      cor   (jnp.ndarray): The coordinates to be rotated and translated (N, 3).
      rot   (jnp.ndarray): The    rotation operator (3, 3).
      trans (jnp.ndarray): The translation operator (3, ).

    Returns:
      aligned (jnp.ndarray): The rotated and translated coordinates (N, 3).
  """
  return jnp.dot(cor, rot) + trans


def kabsch_rotr(to_refer: jnp.ndarray, to_align: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
  r"""Chains the `kabschs` and the `rotrs` operations. This module stops gradients for all inputs.

    Args:
      to_refer (jnp.ndarray): The coordinates to be referred (N, 3).
      to_align (jnp.ndarray): The coordinates to be  aligned (N, 3).
      mask     (jnp.ndarray): The one-hot mask that selects the subset of atoms (N, ).

    Returns:
      aligned (jnp.ndarray): The coordinates of `to_align` aligned onto `to_refer` (N, 3).
  """
  rot, trans = kabsch(to_refer=to_refer, to_align=to_align, mask=mask)
  aligned = rotr(cor=to_align, rot=rot, trans=trans)
  return aligned
