r"""AFTools Low-Rank Adaptation (LoRA) models for AF."""
# Authors: Zilin Song.


import typing

import aftools.utils as _u
import aftools.chkp  as _c
import aftools.lora  as _l


def lora_infer(lora_feats: _u.af.TAFFeatures,
               lora_masks: _u.af.TAFFeatures,
               reprs:      _u.af.TAFFeatures,
               batch:      _u.af.TAFFeatures,
               config:     _u.af.TAFConfig,
               layer_index: int,
               apply_reprs: typing.Literal['single', 'msa', 'pair', 'both'],
               use_bf16: bool, 
               ) -> _u.af.TAFResults:
  r"""Infer AF with LoRA.

    Args:
      lora_feats (TAFFeatures): The LoRA features.
      lora_masks (TAFFeatures): The LoRA masks.
      reprs (TAFFeatures): The AF checkpoint representations.
      batch (TAFFeatures): The AF batch features.
      config (TAFConfig):  The AF configurations.
      layer_index (int): The Evoformer layer index in [0, 48] whose latent state is retrieved.
      apply_reprs (str): The representations to apply LoRA: `single`, `msa`, `pair`, `both`.
      use_bf16 (bool): Whether the bfloat16 runtime is requested.

    Returns:
      results (TAFResults): The AF output results.
  """
  # callables.
  f_lora  = _l.        lora_func(layer_index=layer_index, apply_reprs=apply_reprs)
  f_infer = _c.models.infer_func(layer_index=layer_index, use_bf16=use_bf16)
  # inference.
  out_lora  = f_lora (lora_feats=lora_feats, lora_masks=lora_masks, reprs=reprs)
  out_infer = f_infer(reprs=out_lora, batch=batch, config=config)
  return out_infer
