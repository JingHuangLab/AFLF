r"""AFTools AF Random Enumerator (AFRE) losses - rigid body term."""
# Authors: Zilin Song.


import typing

import jax
import jax.numpy as jnp
import     haiku as  hk

from aftools import BasePreset
import aftools.geom as _g


def _loss_per_segment(cor: jnp.ndarray, ref: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
  r"""Computes the rigid body loss per segment."""
  ref_aligned = _g.spatial.kabsch_rotr(to_refer=cor, to_align=ref, mask=mask) # (Nres*37, 3)
  loss = jnp.sum((cor - ref_aligned)**2, axis=-1)                             # (Nres*37, )
  loss*= mask
  return loss


def loss(cor: jnp.ndarray, spec: dict[str, typing.Union[float, jnp.ndarray]]) -> jnp.ndarray:
  r"""Computes the rigid body loss.

    Args:
      cor  (jnp.ndarray): The predicted atom coordinates (Nres*37, 3).
      spec (dict[str, Union[float, jnp.ndarray]]): The rigid body loss specs with keys:
        `masks`     (jnp.ndarray): The masks for segment divisions (Nseg, Nres*37).
        `ref`       (jnp.ndarray): The reference atom coordinates (Nres*37, 3).
        `weight`    (float):       The loss weight.
        `dropout_p` (float):       The loss dropout rate.

    Returns:
      loss (jnp.ndarray): The rigid body loss.
  """
  # loss impl (Nseg, Nres*37).
  loss = jax.vmap(_loss_per_segment, in_axes=(None, None, 0))(cor, spec['ref'], spec['masks'])
  loss*= spec['weight']
  # reduction.
  loss = _g.rand.dropout(tensor=loss, p=spec['dropout_p'], key=hk.next_rng_key())
  loss = jnp.sum(loss, axis=-1) # (Nseg, )
  return jnp.mean(loss)


class preset(BasePreset):
  r"""The rigid body loss spec preset."""

  def __init__(self):
    r"""Create a rigid body loss spec preset."""
    self.weight: float = 1.
    r"""The loss weight, default: `1.`."""
    self.dropout_p: float = 0.
    r"""The loss dropout rate, default: `0.`."""

  def instantiate(self,
                  ref:   jnp.ndarray,
                  masks: jnp.ndarray,
                  ) -> dict[str, typing.Union[float, jnp.ndarray]]:
    r"""Create a rigid body loss spec from this preset.

      Args:
        ref   (jnp.ndarray): The reference atom coordinates (Nres*37, 3).
        masks (jnp.ndarray): The masks for segment divisions (Nseg, Nres*37).

      Returns:
        spec (dict[str, Union[float, jnp.ndarray]]): The rigid body loss spec.
    """
    return {'ref':    jnp.copy(     ref      ),
            'masks':  jnp.copy(     masks    ),
            'weight':    float(self.weight   ),
            'dropout_p': float(self.dropout_p), }
