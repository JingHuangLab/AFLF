r"""AFTools Checkpoint models for AF."""
# Authors: Zilin Song.


import typing
import functools

import jax
import haiku as hk

from aftools import BaseModel
import aftools.utils as _u
import aftools.chkp  as _c


def _infer_for_checkpoint(batch: _u.af.TAFFeatures, config: _u.af.TAFConfig) -> _u.af.TAFResults:
  mod = _c.modules.InferForCheckpoint(config=config.model)
  results = mod(batch=batch, 
                is_training=False, 
                compute_loss=False, 
                ensemble_representations=False, 
                return_representations  =False, )
  return results


class InferForCheckpointModel(BaseModel):
  r"""AF Checkpoint model that infers and tracks the checkpoints."""

  def __init__(self, config: _u.af.TAFConfig, params: _u.af.TAFParams, key: jax.random.KeyArray):
    r"""Create an AF Checkpoint model that infers and tracks the checkpoints.

      Args:
        config (TAFConfig): The AF configurations.
        params (TAFParams): The AF parameters.
        key (jax.random.KeyArray): The JAX PRNG key.
    """
    super().__init__(config=config, params=params, key=key, multimer_mode=False)

  def compile(self, layer_index: int, jit_pred: bool) -> None:
    r"""Compile the model for inference.

      Args:
        layer_index (int): The Evoformer layer index in [0, 48] whose latent state is retrieved.
        jit_pred (bool): Whether to JIT-compile the predict function.
    """
    # config.
    self.config.update_from_flattened_dict({'model.global_config.aftools': {}, })
    self.config.update_from_flattened_dict({
      'model.global_config.aftools.last_recycle': False,
      'model.global_config.aftools.chkp_layer_i': int(layer_index), })
    # params and pred function.
    prefix = _c.modules.MODULE_SCOPE['InferForCheckpoint']
    self.params = _u.af.prep_params(params=self.params, prefix=prefix, layer_index=0)
    self.f_pred = hk.transform(f=functools.partial(_infer_for_checkpoint, config=self.config)).apply
    if jit_pred: self.f_pred = jax.jit(self.f_pred)

  def predict(self, features_pkl: _u.af.TAFFeatures) -> _u.af.TAFResults:
    r"""Predict with the model.

      Args:
        features_pkl (TAFFeatures): From `alphafold.model.features.np_example_to_features()`.

      Returns:
        results (TAFResults): The AF outputs.
    """
    results = self.f_pred(self.params, self.key, features_pkl)
    return results


def _infer_from_structmod(reprs: _u.af.TAFFeatures,
                          batch: _u.af.TAFFeatures,
                          config: _u.af.TAFConfig,
                          ) -> _u.af.TAFResults:
  mod            = _c.modules.InferFromStructmod(config=config)
  mod_confidence = _c.AFConfidenceModule(config=config)
  results            = mod(reprs=reprs, batch=batch, is_training=False)
  results_confidence = mod_confidence(results=results, batch=batch)
  results.update({'confidence': results_confidence})
  return results


def _infer_from_evoformer(reprs: _u.af.TAFFeatures,
                          batch: _u.af.TAFFeatures,
                          config: _u.af.TAFConfig,
                          ) -> _u.af.TAFResults:
  mod            = _c.modules.InferFromEvoformer(config=config)
  mod_confidence = _c.AFConfidenceModule(config=config)
  results            = mod(reprs=reprs, batch=batch, is_training=False)
  results_confidence = mod_confidence(results=results, batch=batch)
  results.update({'confidence': results_confidence})
  return results


def _infer_from_evoformer_bf16(reprs: _u.af.TAFFeatures,
                               batch: _u.af.TAFFeatures,
                               config: _u.af.TAFConfig,
                               ) -> _u.af.TAFResults:
  mod            = _c.modules.InferFromEvoformerBF16(config=config)
  mod_confidence = _c.AFConfidenceModule(config=config)
  results            = mod(reprs=reprs, batch=batch, is_training=False)
  results_confidence = mod_confidence(results=results, batch=batch)
  results.update({'confidence': results_confidence})
  return results


_TFunc = typing.Callable[[_u.af.TAFFeatures, _u.af.TAFFeatures, _u.af.TAFConfig], _u.af.TAFResults]
def infer_func(layer_index: int, use_bf16: bool) -> _TFunc:
  r"""Get the AF checkpoint inference function.

    Args:
      layer_index (int): The Evoformer layer index in [0, 48] whose latent state is retrieved.
      use_bf16 (bool): Whether the bfloat16 runtime is requested.

    Returns:
      f_infer (Callable):
        The AF checkpoint inference function.
        Input:
          reprs  (TAFFeatures): The AF checkpoint representations.
          batch  (TAFFeatures): The AF batch features.
          config (TAFConfig):   The AF configurations.
        Output:
          results (TAFResults): The AF outputs.
  """
  if layer_index == 48:                                return _infer_from_structmod
  if layer_index in tuple(range(48)) and not use_bf16: return _infer_from_evoformer
  if layer_index in tuple(range(48)) and     use_bf16: return _infer_from_evoformer_bf16
  raise ValueError(f"Illegal input: `layer_index={layer_index}`.")


def infer_scope(layer_index: int, use_bf16: bool) -> str:
  r"""Get the AF checkpoint inference module parameter scope.

    Args:
      layer_index (int): The Evoformer layer index in [0, 48] whose latent state is retrieved.
      use_bf16 (bool): Whether the bfloat16 runtime is requested.

    Returns:
      prefix (str): The prefix to the parameter scope.
  """
  module_scope_key = None
  if layer_index == 48:                                module_scope_key = 'InferFromStructmod'
  if layer_index in tuple(range(48)) and not use_bf16: module_scope_key = 'InferFromEvoformer'
  if layer_index in tuple(range(48)) and     use_bf16: module_scope_key = 'InferFromEvoformer_bf16'
  if module_scope_key is not None: return _c.modules.MODULE_SCOPE[module_scope_key]
  raise ValueError(f"Illegal input: `layer_index={layer_index}` and `use_bf16={use_bf16}`.")


class InferFromCheckpointModel(BaseModel):
  r"""AF Checkpoint model that infers from the checkpoints."""

  def __init__(self, config: _u.af.TAFConfig, params: _u.af.TAFParams, key: jax.random.KeyArray):
    r"""Create an AF Checkpoint model that infers from the checkpoints.

      Args:
        config (TAFConfig): The AF configurations.
        params (TAFParams): The AF parameters.
        key (jax.random.KeyArray): The JAX PRNG key.
    """
    super().__init__(config=config, params=params, key=key, multimer_mode=False)

  def compile(self, batch: _u.af.TAFFeatures, use_bf16: bool, jit_pred: bool) -> None:
    r"""Compile the model for inference.

      Args:
        batch (TAFFeatures): The AF batch features.
        use_bf16 (bool): Whether the bfloat16 runtime is requested.
        jit_pred (bool): Whether the `predict` function is JIT-compiled.
    """
    layer_index = batch['aftools_layer_index']
    # params.
    prefix = infer_scope(layer_index=layer_index, use_bf16=use_bf16)
    self.params = _u.af.prep_params(params=self.params, prefix=prefix, layer_index=layer_index)
    # pred function.
    f_pred = infer_func(layer_index=layer_index, use_bf16=use_bf16)
    self.f_pred = hk.transform(f=functools.partial(f_pred, batch=batch, config=self.config)).apply
    if jit_pred: self.f_pred = jax.jit(self.f_pred)

  def predict(self, reprs: _u.af.TAFFeatures) -> _u.af.TAFResults:
    r"""Predict with the model.

      Args:
        reprs (TAFFeatures): The AF checkpoint representations.

      Returns:
        results (TAFResults): The AF outputs.
    """
    results = self.f_pred(self.params, self.key, reprs)
    return results
