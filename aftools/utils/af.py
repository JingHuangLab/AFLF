r"""AFTools utilities for AlphaFold."""
# Authors: Zilin Song.


import os
import io
import copy
import typing

import ml_collections
import numpy as np
import jax.numpy as jnp

import alphafold.model.config   as _af_config
import alphafold.model.utils    as _af_utils
import alphafold.common.protein as _af_prot


# type hints.
TAFConfig   = ml_collections.ConfigDict
TAFParams   = dict[str, np.ndarray]
TAFFeatures = dict[str, typing.Union[jnp.ndarray, dict[str,
                        typing.Union[jnp.ndarray, dict[str, jnp.ndarray]], ]]]
TAFResults  = dict[str, dict[str, jnp.ndarray]]
TAFProtein  = _af_prot.Protein

TAFMonomerName = typing.Literal[
  'model_1', 'model_1_ptm',
  'model_2', 'model_2_ptm',
  'model_3', 'model_3_ptm',
  'model_4', 'model_4_ptm',
  'model_5', 'model_5_ptm', ]
TAFMultimerName = typing.Literal[
  'model_1_multimer_v3',
  'model_2_multimer_v3',
  'model_3_multimer_v3',
  'model_4_multimer_v3',
  'model_5_multimer_v3', ]
TAFModelsName = typing.Union[TAFMonomerName, TAFMultimerName]


# cast funcs.
def cast_jnp_to_np(dict_jnp: TAFResults) -> dict[str, dict[str, np.ndarray]]:
  r"""Recursively cast JAX arrays in `dict_jnp` to NumPy arrays."""
  for k, v in dict_jnp.items():
    if isinstance(v, dict):
      dict_jnp[k] = cast_jnp_to_np(v)
    elif isinstance(v, jnp.ndarray):
      if v.dtype == jnp.bfloat16: # custom XLA/JAX dtype.
        v = v.astype(jnp.float32)
      dict_jnp[k] = np.asarray(v)
  return dict_jnp


def cast_af_prediction_to_pdb_string(batch: TAFFeatures, results: TAFResults) -> str:
  r"""Convert the AF prediction to the output-ready PDB strings.

    Args:
      batch (TAFFeatures): The AF batch features.
      results (TAFResults): The AF output results.

    Returns:
      pdb_string (str): The content of PDB file as a string.
  """
  # NOTE (ZS):
  #   The `remove_leading_feature_dimension` option is used for casting correct shapes for the batch
  #   features `features.pkl` in the preprocessed AF inputs. In AlphaFoldTools, a copy of the batch
  #   feature during the inference is returned which does not require this reduction.
  prot = _af_prot.from_prediction(features=batch,
                                  result=results,
                                  remove_leading_feature_dimension=False, )
  pdb_string = _af_prot.to_pdb(prot=prot)
  return pdb_string


# constants.
AFAtomTypes: list[str] = _af_prot.residue_constants.atom_types
r"""The AF atom types (total 37):
'N',   'CA',  'C',   'CB',  'O',   'CG',  'CG1', 'CG2', 'OG',  'OG1', 'SG',  'CD',  'CD1', 'CD2',
'ND1', 'ND2', 'OD1', 'OD2', 'SD',  'CE',  'CE1', 'CE2', 'CE3', 'NE',  'NE1', 'NE2', 'OE1', 'OE2',
'CH2', 'NH1', 'NH2', 'OH',  'CZ',  'CZ2', 'CZ3', 'NZ',  'OXT'.
"""


AFAtomTypeIndex: dict[str, int] = {atom_type: index for index, atom_type in enumerate(AFAtomTypes)}
r"""Mapping from AF atom type to its index."""

AFIndexAtomType: dict[int, str] = {index: atom_type for index, atom_type in enumerate(AFAtomTypes)}
r"""Mapping from index to the AF atom type."""


# helpers.
def get_residue_indexes(batch: TAFFeatures) -> list[int]:
  r"""Return the residue indexes from the AF batch features."""
  return list(_ for _ in range(batch['aatype'].shape[0]))


# configs.
def AFConfig(model_name: TAFMonomerName,
             msa_clusters:  int  =   508,
             num_recycling: int  =     3,
             use_templates: bool = False,
             use_dropout:   bool = False,
             use_remat:     bool =  True,
             ) -> TAFConfig:
  r"""The AF configurations.

    Args:
      model_name (TAFMonomerName): The AF model name, `model_{1, 2, 3, 4, 5}(_ptm)`.
      msa_clusters (int): The number of MSA clusters. Default: 508.
      num_recycling (int): The number of recycling iterations. Default: 3.
      use_templates (bool): Whether to embed templates. Default: False.
      use_dropout (bool): Whether to enable dropout. Default: False.
      use_remat (bool): Whether to use re-materialization for Evoformers. Default: True.

    Returns:
      config (TAFConfig): The AF configurations.
  """
  assert (   model_name in _af_config.MODEL_PRESETS.get('monomer')
          or model_name in _af_config.MODEL_PRESETS.get('monomer_ptm')
          ), f"Illegal `model_name={model_name}`."
  config = copy.deepcopy(_af_config.CONFIG)
  config.update_from_flattened_dict(_af_config.CONFIG_DIFFS[model_name])
  # NOTE (ZS):
  #   AF preprocesses features in alphafold.model.RunModel().process_features() so that config.data
  #   should be modified as well.
  # Turn off multimer_model.
  config.model.global_config.multimer_mode = False
  # Single ensemble.
  config.data.eval.num_ensemble = 1
  # Maybe reduce `num_msa_clusters` for efficient memory scaling.
  config.data.eval.max_msa_clusters = msa_clusters
  # If use recycling.
  config.data.common.num_recycle = num_recycling
  config.model.num_recycle       = num_recycling
  config.data.common.resample_msa_in_recycling           = True if num_recycling>0 else False
  config.model.resample_msa_in_recycling                 = True if num_recycling>0 else False
  config.model.embeddings_and_evoformer.recycle_features = True if num_recycling>0 else False
  config.model.embeddings_and_evoformer.recycle_pos      = True if num_recycling>0 else False
  # If use template.
  config.data.eval.max_templates   = 4 if use_templates else 0
  config.data.common.use_templates = use_templates
  config.data.common.reduce_msa_clusters_by_max_templates = use_templates
  config.model.embeddings_and_evoformer.template.enabled  = use_templates
  # If use dropout.
  config.model.global_config.deterministic = not use_dropout
  # If use re-materialization (reverse-mode autograd where necessary).
  config.model.global_config.use_remat = use_remat
  return config


def AFConfig_multimer(model_name: TAFMultimerName,
                      msa_clusters:  int = 508,
                      num_recycling: int = 20,
                      use_templates: bool = False,
                      use_dropout:   bool = False,
                      use_remat:     bool =  True,
                      ) -> TAFConfig:
  r"""The AF-Multimer configurations.

    Args:
      model_name (TAFMultimerName): The AF-Multimer model name, `model_{1, 2, 3, 4, 5}_multimer_v3`.
      msa_clusters (int): The number of MSA clusters. Default: 508.
      num_recycling (int): The number of recycling iterations. Default: 20.
      use_templates (bool): Whether to embed templates. Default: False.
      use_dropout (bool): Whether to enable dropout. Default: False.
      use_remat (bool): Whether to use re-materialization for Evoformers. Default: True.

    Returns:
      config (TAFConfig): The AF-Multimer configurations.
  """
  # Load presets.
  assert (model_name in _af_config.MODEL_PRESETS.get('multimer')
          ), f"Illegal `model_name={model_name}`."
  config = copy.deepcopy(_af_config.CONFIG_MULTIMER)
  config.update_from_flattened_dict(_af_config.CONFIG_DIFFS[model_name])
  # Turn on multimer_mode.
  config.model.global_config.multimer_mode = True
  # Single ensemble.
  config.model.num_ensemble_eval = 1
  # Maybe reduce `num_msa_clusters` for efficient memory scaling.
  config.model.embeddings_and_evoformer.num_msa = msa_clusters
  # If use recycling.
  config.model.num_recycle                               = num_recycling
  config.model.resample_msa_in_recycling                 = True if num_recycling>0 else False
  config.model.embeddings_and_evoformer.recycle_pos      = True if num_recycling>0 else False
  config.model.embeddings_and_evoformer.recycle_features = True if num_recycling>0 else False
  # If use template.
  config.model.embeddings_and_evoformer.template.enabled = use_templates
  # If use dropout.
  config.model.global_config.deterministic = not use_dropout
  # If use re-materialization (reverse-mode autograd where necessary).
  config.model.global_config.use_remat = use_remat
  return config


# parameters.
def AFParams(model_name: TAFMonomerName, params_dir: str) -> TAFParams:
  r"""The AF parameters.

    Args:
      model_name (TAFMonomerName): The AF model name, `model_{1, 2, 3, 4, 5}(_ptm)`.
      params_dir (str): The directory containing the `params_*.npz` files.

    Returns:
      params (TAFParams): The AF monomer parameters.
  """
  assert (   model_name in _af_config.MODEL_PRESETS.get('monomer')
          or model_name in _af_config.MODEL_PRESETS.get('monomer_ptm')
          ), f"Illegal `model_name={model_name}`."
  return load_params(model_name=model_name, params_dir=params_dir)


def AFParams_multimer(model_name: TAFMultimerName, params_dir: str) -> TAFParams:
  r"""The AF-Multimer parameters.

    Args:
      model_name (TAFMultimerName): The AF-Multimer model name, `model_{1, 2, 3, 4, 5}_multimer_v3`.
      params_dir (str): The directory containing the `params_*.npz` files.

    Returns:
      params (TAFParams): The AF-Multimer parameters.
  """
  assert (model_name in _af_config.MODEL_PRESETS.get('multimer')
          ), f"Illegal `model_name={model_name}`."
  return load_params(model_name=model_name, params_dir=params_dir)


def load_params(model_name: TAFModelsName, params_dir: str) -> TAFParams:
  r"""Load the AlphaFold model parameters from disk.

    Args:
      model_name (TAFModelsName): The AF model name.
      params_dir (str): The directory containing the `params_*.npz` files.

    Returns:
      params (TAFParams): The loaded AF parameters.
  """
  with open(os.path.join(params_dir, f'params_{model_name}.npz'), 'rb') as f:
    params = np.load(io.BytesIO(f.read()), allow_pickle=False)
  params = _af_utils.flat_params_to_haiku(params=params)
  return params


def prep_params(params: TAFParams, prefix: str, layer_index: int) -> TAFParams:
  r"""Prepare AF parameters for inference from a checkpoint.

    Args:
      params (TAFParams): The input AF parameters.
      prefix (str): The Haiku parameter scope to prepend to each key.
      layer_index (int): The zero-based Evoformer input layer index in [0, 48].

    Returns:
      new_params (TAFParams): The prefixed and trimmed AF parameters.
  """
  layer_index = 0 if layer_index == 48 else layer_index
  assert layer_index in list(range(48)), f"Illegal `layer_index={layer_index}`"
  new_params = {}
  for (k, v) in params.items():
    # remove and replace prefix.
    new_k = k.removeprefix("alphafold/alphafold_iteration/")
    new_k = f"{prefix}/{new_k}"
    # trims away unused evoformer layers.
    new_v = v
    if k.startswith("alphafold/alphafold_iteration/evoformer/evoformer_iteration/"):
      new_v = {kk: vv[layer_index:] for (kk, vv) in v.items()}
    new_params.update({new_k: new_v})
  return new_params
