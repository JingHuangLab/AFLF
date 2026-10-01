r"""AFTools AF Random Enumerator (AFRE) losses - centroid histogram term."""
# Authors: Zilin Song.


import typing


import jax.numpy as jnp
import     haiku as  hk

from aftools import BasePreset
import aftools.geom as _g


def loss(cor: jnp.ndarray, spec: dict[str, typing.Union[float, jnp.ndarray]]) -> jnp.ndarray:
  r"""Computes the centroid histogram loss.

    Args:
      cor  (jnp.ndarray): The predicted atom coordinates (Nres*37, 3).
      spec (dict[str, Union[float, jnp.ndarray]]): The centroid histogram loss specs with keys:
        `cent_masks`   (jnp.ndarray): The masks for centroid divisions (Ncent, Nres*37).
        `prob_ref`     (jnp.ndarray): The reference centroid distance density (Ncent^2, Nbin).
        `gauss_breaks` (jnp.ndarray): The Gaussian KDE histogram bin edges (Nbin-1, ).
        `gauss_rwidth` (float):       The Gaussian KDE reciprocal smearing width.
        `mask`         (jnp.ndarray): The loss mask (Ncent, Ncent).
        `weight`       (float):       The loss weight.
        `dropout_p`    (float):       The loss dropout rate.

    Returns:
      loss (jnp.ndarray): The centroid histogram loss.
  """
  dist_cor = _g.dist.dist_self(cor=_g.cg.centroid_batched(cor=cor, masks=spec['cent_masks']))
  # loss impl (Ncent^2, Nbin).
  prob_cor = _g.kde.gaussian_binning(logits=dist_cor.ravel(),
                                     breaks=spec['gauss_breaks'],
                                     rwidth=spec['gauss_rwidth'], )
  loss = - spec['prob_ref'] * jnp.log(prob_cor + (prob_cor==0.))  # log(0.+(0.==0.)) = 0.
  loss*= jnp.expand_dims(spec['mask'].ravel(), axis=-1)
  loss*=                 spec['weight']
  # reduction.
  loss = _g.rand.dropout(tensor=loss, p=spec['dropout_p'], key=hk.next_rng_key())
  loss = jnp.sum(loss, axis=-1) # (Ncent^2, )
  return jnp.mean(loss)


class preset(BasePreset):
  r"""The centroid histogram loss spec preset."""

  def __init__(self):
    r"""Create a centroid histogram loss spec preset."""
    self.gauss_breaks: jnp.ndarray = jnp.linspace(2, 64, 64-1)
    r"""The Gaussian KDE histogram bin edges (Nbin-1, ), default: `jnp.linspace(2, 64, 64-1)`."""
    self.gauss_rwidth: float = 1.
    r"""The Gaussian KDE reciprocal smearing width, default: `1.`."""
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
    r"""Create a centroid histogram loss spec from this preset.

      Args:
        ref        (jnp.ndarray): The reference atom coordinates (Nres*37, 3).
        cent_masks (jnp.ndarray): The masks for centroid divisions (Ncent, Nres*37).

      Returns:
        spec (dict[str, Union[float, jnp.ndarray]]): The centroid histogram loss spec.
    """
    dist_ref = _g.dist.dist_self(cor=_g.cg.centroid_batched(cor=ref, masks=cent_masks))
    prob_ref = _g.kde.gaussian_binning(logits=dist_ref.ravel(),
                                       breaks=self.gauss_breaks,
                                       rwidth=self.gauss_rwidth, )
    mask     = (self.mask==1) * jnp.ones_like(dist_ref)  # (Ncent, Ncent)
    assert (jnp.sum(mask, axis=-1)!=0.).all(), f"Illegal `mask` with all zero row."
    return {'cent_masks':   jnp.copy(     cent_masks  ),
            'prob_ref':     jnp.copy(     prob_ref    ),
            'gauss_breaks': jnp.copy(self.gauss_breaks),
            'gauss_rwidth':    float(self.gauss_rwidth),
            'mask':         jnp.copy(     mask        ),
            'weight':          float(self.weight      ),
            'dropout_p':       float(self.dropout_p   ), }
