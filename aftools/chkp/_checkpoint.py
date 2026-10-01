r"""AlphaFoldTools Checkpoint modules."""
# Authors: Zilin Song.


import typing


def get_action_to(infer_from: typing.Literal['structmod', 'evoformer'], 
                  apply_lora: typing.Literal['single', 'msa', 'pair', 'both'], 
                  ) -> typing.Union[str, list[str]]:
  r"""Determine the value of the `xxxx_to` keywords for the functions.

    Args:
      infer_from (str): The checkpoint to infer from: `structmod`, `evoformer`.
      apply_lora (str): The representation to apply LoRA: `single`, `msa`, `pair`, `both`.
    
    Returns:
      act_to (Union[str, list[str]]):
        The value for the keyword `xxxx_to`.
        One of: `single`, `msa`, `pair`, `['single', 'pair']`, `['msa', 'pair']`.
  """
  actions_to = {'structmod': {'single': 'single', 'pair': 'pair', 'both': ['single', 'pair']}, 
                'evoformer': {   'msa':    'msa', 'pair': 'pair', 'both': [   'msa', 'pair']}, }
  assert infer_from in actions_to            .keys(), f"Illegal `infer_from={infer_from}`."
  assert apply_lora in actions_to[infer_from].keys(), f"Illegal `apply_lora={apply_lora}`."
  return actions_to[infer_from][apply_lora]