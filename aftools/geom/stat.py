r"""Statistics."""
# Authors: Zilin Song.


import jax.numpy as jnp


def update(x: jnp.ndarray, stat: dict[str, jnp.ndarray] = None) -> dict[str, jnp.ndarray]:
  r"""Performs one step of Welford online update, returns the updated Welford state."""
  if stat is None:
    stat = {'n': 0., 'M': 0., 'S': 0.}
  n, M, S = stat['n'], stat['M'], stat['S']
  n = n + 1             ; delta  = x - M
  M = M + delta / n     ; delta_ = x - M
  S = S + delta * delta_
  return {'n': n, 'M': M, 'S': S}


def reduce(stat: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
  r"""Decode the Welford statistics for the mean/std/var and the coefficient of variation."""
  n, M, S = stat['n'], stat['M'], stat['S']
  mean = M
  var  = S / (n - 1. + (n==1.))
  std  = jnp.sqrt(var + (var==0.)) - (var==0.)
  cv   = std / (jnp.abs(M) + (M==0.))
  return {'mean': mean, 'var': var, 'std': std, 'cv': cv}
