r"""AFTools AF Random Enumerator (AFRE) losses - repelling term."""
# Authors: Zilin Song.


import typing


import jax.numpy as jnp
import     haiku as  hk

from aftools import BasePreset
import aftools.geom as _g


def loss(cor: jnp.ndarray, spec: dict[str, typing.Union[float, jnp.ndarray]]) -> jnp.ndarray:
  r"""Computes the repelling loss.

    Args:
      cor  (jnp.ndarray): The predicted atom coordinates (Nres*37, 3).
      spec (dict[str, Union[float, jnp.ndarray]]): The repelling loss specs with keys:
        `cent_masks`   (jnp.ndarray): The masks for centroid divisions (Ncent, Nres*37).
        `dist_refs`    (jnp.ndarray): The reference centroid distances (Nrep, Ncent, Ncent).
        `gauss_rwidth` (float):       The repulsive Gaussian reciprocal width.
        `diver_weight` (jnp.ndarray): The diversity weights per distances (Ncent, Ncent).
        `mask`         (jnp.ndarray): The loss mask (Ncent, Ncent).
        `weights`      (jnp.ndarray): The loss weights      per repellent (Nrep, ).
        `weights_mask` (jnp.ndarray): The loss weights mask per repellent (Nrep, ).
        `dropout_p`    (float):       The loss dropout rate.

    Returns:
      loss (jnp.ndarray): The repelling loss.
  """
  dist_cor = _g.dist.dist_self(cor=_g.cg.centroid_batched(cor=cor, masks=spec['cent_masks']))
  # loss impl (Nrep, Ncent, Ncent).
  dist_cor = jnp.expand_dims(dist_cor, axis=0)
  expo = - ((dist_cor - spec['dist_refs']) * spec['gauss_rwidth'] / jnp.sqrt(2.)) ** 2.
  loss = jnp.exp(expo) - (expo==0.) # exp(0.)-(0.==0.) = 0.
  loss*= jnp.expand_dims(spec['diver_weight']                , axis= 0    )
  loss*= jnp.expand_dims(spec['mask']                        , axis= 0    )
  loss*= jnp.expand_dims(spec['weights']*spec['weights_mask'], axis=(1, 2))
  # reduction.
  loss = _g.rand.dropout(tensor=loss, p=spec['dropout_p'], key=hk.next_rng_key())
  loss = jnp.sum(jnp.sum(loss, axis=0), axis=-1)
  return jnp.mean(loss)


class preset(BasePreset):
  r"""The repelling loss spec preset."""

  def __init__(self):
    r"""Create a repelling loss spec preset."""
    self.gauss_rwidth: float = 1.
    r"""The repulsive Gaussian reciprocal width, default: `1.`."""
    self.mask: typing.Union[int, jnp.ndarray] = 1
    r"""The loss mask (Ncent, Ncent), default: `no mask`."""
    self.weights: jnp.ndarray = .1 * jnp.ones(10)
    r"""The loss weights (Nrep, ), default: `0.1 * jnp.ones(10)`."""
    self.dropout_p: float = 0.
    r"""The loss dropout rate, default: `0.`."""

  def instantiate(self,
                  ref:        jnp.ndarray,
                  cent_masks: jnp.ndarray,
                  ) -> dict[str, typing.Union[float, jnp.ndarray]]:
    r"""Create a repelling loss spec from this preset.

      Args:
        ref        (jnp.ndarray): The reference atom coordinates (Nres*37, 3).
        cent_masks (jnp.ndarray): The masks for centroid divisions (Ncent, Nres*37).

      Returns:
        spec (dict[str, Union[float, jnp.ndarray]]): The repelling loss spec.
    """
    dist_ref  = _g.dist.dist_self(cor=_g.cg.centroid_batched(cor=ref, masks=cent_masks))
    dist_refs = jnp.stack([dist_ref for _ in range(self.weights.size)], axis=0)
    mask      = (self.mask==1) * jnp.ones_like(dist_ref)  # (Ncent, Ncent)
    assert (jnp.sum(mask, axis=-1)!=0.).all(), f"Illegal `mask` with all zero row."
    diver_weight = jnp.ones_like(dist_ref)
    weights_mask = jnp.ones_like(self.weights)
    return {'cent_masks':   jnp.copy(     cent_masks  ),
            'dist_refs':    jnp.copy(     dist_refs   ),
            'gauss_rwidth':    float(self.gauss_rwidth),
            'diver_weight': jnp.copy(     diver_weight),
            'mask':         jnp.copy(     mask        ),
            'weights':      jnp.copy(self.weights     ),
            'weights_mask': jnp.copy(     weights_mask),
            'dropout_p':       float(self.dropout_p   ), }
