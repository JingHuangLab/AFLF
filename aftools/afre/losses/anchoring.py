r"""AFTools AF Random Enumerator (AFRE) losses - anchoring term."""
# Authors: Zilin Song.


import typing

import jax
import jax.numpy as jnp
import     haiku as  hk

from aftools import BasePreset
import aftools.geom as _g


def _huber_loss(loss: jnp.ndarray, delta: float) -> jnp.ndarray:
  r"""The Huber loss function."""
  return jnp.where(loss > delta, delta * (loss - 0.5*delta), 0.5 * loss**2)


def loss(cor: jnp.ndarray, spec: dict[str, typing.Union[float, jnp.ndarray]]) -> jnp.ndarray:
  r"""Computes the anchoring loss.

    Args:
      cor  (jnp.ndarray): The predicted atom coordinates (Nres*37, 3).
      spec (dict[str, Union[float, jnp.ndarray]]): The anchoring loss specs with keys:
        `cent_masks`   (jnp.ndarray): The masks for centroid divisions (Ncent, Nres*37).
        `dist_ref`     (jnp.ndarray): The reference centroid distances (Ncent, Ncent).
        `thresh_lower` (jnp.ndarray): The lower bound of the tolerated distance differences.
        `thresh_upper` (jnp.ndarray): The upper bound of the tolerated distance differences.
        `switch_delta` (float):       The delta param of the Huber distance differences loss.
        `mask`         (jnp.ndarray): The loss mask (Ncent, Ncent).
        `weight`       (float):       The loss weight.
        `dropout_p`    (float):       The loss dropout rate.

    Returns:
      loss (jnp.ndarray): The anchoring loss.
  """
  dist_cor = _g.dist.dist_self(cor=_g.cg.centroid_batched(cor=cor, masks=spec['cent_masks']))
  # loss impl (Ncent, Ncent).
  dist_diff = dist_cor - spec['dist_ref']
  loss = jax.nn.relu(spec['thresh_lower']-dist_diff) + jax.nn.relu(dist_diff-spec['thresh_upper'])
  loss = _huber_loss(loss=loss, delta=spec['switch_delta'])
  loss*= spec['mask']
  loss*= spec['weight']
  # reduction.
  loss = _g.rand.dropout(tensor=loss, p=spec['dropout_p'], key=hk.next_rng_key())
  loss = jnp.sum(loss, axis=-1) # (Ncent, )
  return jnp.mean(loss)


class preset(BasePreset):
  r"""The anchoring loss spec preset."""

  def __init__(self):
    r"""Create an anchoring loss spec preset."""
    self.thresh_lower: typing.Union[float, jnp.ndarray] = -1.
    r"""The lower bound of the tolerated distance differences, default: `-1.`."""
    self.thresh_upper: typing.Union[float, jnp.ndarray] =  1.
    r"""The upper bound of the tolerated distance differences, default:  `1.`."""
    self.switch_delta: float = 0.1
    r"""The delta param of the Huber distance differences loss, default: `0.1`."""
    self.mask: typing.Union[int, jnp.ndarray] = 1
    r"""The loss mask (Ncent, Ncent), default: `no mask`."""
    self.weight: float = 1.
    r"""The loss weight, default: `1.`."""
    self.dropout_p: float = 0.
    r"""The loss dropout rate, default: `0.`."""

  def instantiate(self,
                  ref:        jnp.ndarray,
                  cent_masks: jnp.ndarray,
                  ) -> dict[str, typing.Union[float, jnp.ndarray]]:
    r"""Create an anchoring loss spec from this preset.

      Args:
        ref        (jnp.ndarray): The reference atom coordinates (Nres*37, 3).
        cent_masks (jnp.ndarray): The masks for centroid divisions (Ncent, Nres*37).

      Returns:
        spec (dict[str, Union[float, jnp.ndarray]]): The anchoring loss spec.
    """
    dist_ref = _g.dist.dist_self(cor=_g.cg.centroid_batched(cor=ref, masks=cent_masks))
    mask = (self.mask==1) * jnp.ones_like(dist_ref)  # (Ncent, Ncent)
    assert (jnp.sum(mask, axis=-1)!=0.).all(), f"Illegal `mask` with all zero row."
    thresh_lower = self.thresh_lower * jnp.ones_like(dist_ref)  # (Ncent, Ncent)
    thresh_upper = self.thresh_upper * jnp.ones_like(dist_ref)  # (Ncent, Ncent)
    assert (thresh_lower<=thresh_upper).all(), f"Illegal `thresh_lower > thresh_upper`."
    return {'cent_masks':   jnp.copy(     cent_masks  ),
            'dist_ref':     jnp.copy(     dist_ref    ),
            'thresh_lower': jnp.copy(     thresh_lower),
            'thresh_upper': jnp.copy(     thresh_upper),
            'switch_delta':    float(self.switch_delta),
            'mask':         jnp.copy(     mask        ),
            'weight':          float(self.weight      ),
            'dropout_p':       float(self.dropout_p   ), }
