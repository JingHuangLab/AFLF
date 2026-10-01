r"""Randomized."""
# Authors: Zilin Song.


import jax
import jax.numpy as jnp


def dropout(tensor: jnp.ndarray, p: float, key: jax.random.KeyArray) -> jnp.ndarray:
  r"""Applies dropout to `tensor`.

    Args:
      tensor (jnp.ndarray):         The tensor to apply dropout.
      p      (float):               The dropout rate [0., 1.].
      key    (jax.random.KeyArray): The JAX PRNG key.

    Returns:
      tensor (jnp.ndarray): The dropped-out `tensor`.
  """
  keep_prob = 1. - jax.nn.relu(p) # rate >= 0.
  keep_mask = jax.random.bernoulli(key=key, p=keep_prob, shape=tensor.shape)
  return keep_mask * tensor / (keep_prob + (keep_prob == 0.)) # denom = 1. if prob==0 else prob


def uniform(key: jax.random.KeyArray, shape: tuple[int, ...]) -> jnp.ndarray:
  r"""Sample from the uniform distribution U(-1, 1)."""
  return jax.random.uniform(key=key, shape=shape, minval=-1., maxval=1.)


def normal (key: jax.random.KeyArray, shape: tuple[int, ...]) -> jnp.ndarray:
  r"""Sample from the normal distribution N(0, 1)."""
  return jax.random.normal(key=key, shape=shape)
