r"""Kernel density estimations."""
# Authors: Zilin Song.


import jax
import jax.numpy as jnp
import jax.scipy as jsp


# histogram binning.
def bin_breaks(init: float, last: float, num_bins: int) -> jnp.ndarray:
  r"""Compute the locations of bin edges shared by adjacent bins (N_bins-1, ).

    Args:
      init     (float): The right edge of the first bin covering (-inf, `init`].
      last     (float): The  left edge of the  last bin covering [`last`,  inf).
      num_bins (float): The total number of bins (N_bins).

    Returns:
      breaks (jnp.ndarray): The locations of edges shared by adjacent bins (N_bins-1, ).
  """
  assert last>=init>=0., f"Illegal edge range: `init={init}` and `last={last}`."
  return jnp.linspace(init, last, num_bins-1)


def gaussian_binning(logits: jnp.ndarray, breaks: jnp.ndarray, rwidth: jnp.ndarray) -> jnp.ndarray:
  r"""**Differentiable** Gaussian kernel density estimation of the logits into the bins specified by
    the bin edges at `breaks`. Each logit is Gaussian-smeared with the reciprocal width `rwidth` via
    the `erf` function and the density per logit per bin is computed by integrating the Gaussian CDF
    between the bin edges.

    Args:
      logits (jnp.ndarray): The logits to be soft binned (N, ).
      breaks (jnp.ndarray): The locations of edges shared by adjacent bins (N_bins-1, ).
      rwidth (float):       The reciprocal Gaussian width for logits smearing.

    Returns:
      binned (jnp.ndarray): The histogram probabilities per logit per bin (N, N_bins).
  """
  # integration bounds (N_bins, ).
  cdf_lower_bound = jnp.concatenate((jnp.expand_dims(-jnp.inf, axis=0), breaks))  # [-inf, last]
  cdf_upper_bound = jnp.concatenate((breaks, jnp.expand_dims( jnp.inf, axis=0)))  # [init,  inf]
  # integrands (N, N_bins).
  denom = rwidth / jnp.sqrt(2.)  # the denominator.
  cdf_lower = .5 * (1. + jsp.special.erf((cdf_lower_bound - logits[:, jnp.newaxis]) * denom))
  cdf_upper = .5 * (1. + jsp.special.erf((cdf_upper_bound - logits[:, jnp.newaxis]) * denom))
  # NOTE (ZS): The JAX numerics here sometimes produces `(0.-0.)<0.`, thus the relu op.
  probs = jax.nn.relu(cdf_upper - cdf_lower)
  return probs / jnp.expand_dims(jnp.sum(probs, axis=-1), axis=1) # (N, N_bins), ensure normalized.


def uniform_binning(logits: jnp.ndarray, breaks: jnp.ndarray) -> jnp.ndarray:
  r"""**Non-differentiable** uniform kernel density estimation of the logits into the bins specified
    by the bin edges at `breaks`.

    Args:
      logits (jnp.ndarray): The logits to be hard binned (N, ).
      breaks (jnp.ndarray): The locations of edges shared by adjacent bins (N_bins-1, ).

    Returns:
      binned (jnp.ndarray): The one-hot encoded bin indexes per logit (N, N_bins).
  """
  logits = jnp.expand_dims(a=jax.lax.stop_gradient(logits), axis=-1)  # (N, 1)
  logits_bin_indexes = jnp.sum(logits > breaks, axis=-1)              # (N, ), ints in [0, N_bins-1]
  onehots = jax.nn.one_hot(logits_bin_indexes, num_classes=breaks.shape[0]+1) # (N, N_bins)
  return onehots
