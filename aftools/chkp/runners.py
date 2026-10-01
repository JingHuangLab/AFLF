r"""AFTools Checkpoint runners for AF."""
# Authors: Zilin Song.


import os
import typing

import numpy as np

import alphafold.model.features as _af_features

from aftools import BaseRunner, Runtime
import aftools.utils as _u
import aftools.chkp  as _c


class InferForCheckpointRunner(BaseRunner):
  r"""AF Checkpoint runner that infers and tracks the checkpoints."""

  def __init__(self, project_dir: str = os.getcwd(), runtime_tag: str = ''):
    r"""Create an AF Checkpoint runner that infers and tracks the checkpoints.

      Args:
        project_dir (str): The project directory for the runner I/O.
        runtime_tag (str): The suffix tag to the runtime directory.
    """
    super().__init__(project_dir=project_dir,
                     runtime_tag=runtime_tag,
                     runner_name=r'chkp.infer_for_checkpoint', )

  def execute(self,
              config: _u.af.TAFConfig,
              params: _u.af.TAFParams,
              checkpoint_at: typing.Union[int, typing.Literal['evoformer', 'structmod']],
              features_seed: typing.Union[int, None],
              features_pkl: _u.af.TAFFeatures,
              ) -> None:
    r"""Execute the runner.

      Args:
        config (TAFConfig): The AF configurations.
        params (TAFParams): The AF parameters.
        checkpoint_at (Union[int, Literal['evoformer', 'structmod']]): The Evoformer layer index in
          [0, 48] or an alias string (`evoformer` = 0, `structmod` = 48).
        features_seed (Union[None, int]): The random seed for featurizing `features_pkl`.
        features_pkl (TAFFeatures): From `np.load('features.pkl', allow_pickle=True)`.
    """
    layer_index = -1
    if checkpoint_at == 'evoformer':   layer_index = 0
    if checkpoint_at == 'structmod':   layer_index = 48
    if isinstance(checkpoint_at, int): layer_index = int(checkpoint_at)
    assert layer_index in tuple(range(49)), f"Illegal `checkpoint_at={checkpoint_at}`."
    if features_seed is None: features_seed = np.random.randint(low=-99999, high=100000)
    assert isinstance(features_seed, int), f"Illegal `features_seed={features_seed}`."
    with Runtime(whoami=self.runner_name, working_dir=self.working_dir) as RT:
      # kernels.
      mod = _c.models.InferForCheckpointModel(config=config, params=params, key=self.key)
      mod.compile(layer_index=layer_index, jit_pred=False)  # Do not jit.
      features_pkl = _af_features.np_example_to_features(features_pkl, mod.config, features_seed)
      results = mod.predict(features_pkl=features_pkl)
      results = _u.af.cast_jnp_to_np(dict_jnp=results)
      # outputs.
      out_checkpoint = {
        'batch': {'seq_mask':                results.pop('aftools_batch_seq_mask'               ),
                  'aatype':                  results.pop('aftools_batch_aatype'                 ),
                  'residue_index':           results.pop('aftools_batch_residue_index'          ),
                  'atom14_atom_exists':      results.pop('aftools_batch_atom14_atom_exists'     ),
                  'atom37_atom_exists':      results.pop('aftools_batch_atom37_atom_exists'     ),
                  'residx_atom37_to_atom14': results.pop('aftools_batch_residx_atom37_to_atom14'),
                  'aftools_layer_index': int(layer_index), }, }
      latent_evoformer = {'repr_msa':    results.pop('aftools_evoformer_repr_msa'),
                          'repr_pair':   results.pop('aftools_evoformer_repr_pair'),
                          'mask_msa':    results.pop('aftools_evoformer_mask_msa'),
                          'mask_pair':   results.pop('aftools_evoformer_mask_pair'), }
      latent_structmod = {'repr_single': results.pop('aftools_structmod_repr_single'),
                          'repr_pair':   results.pop('aftools_structmod_repr_pair'), }
      if layer_index in tuple(range(48)): out_checkpoint.update({'latent_reprs': latent_evoformer})
      if layer_index == 48:               out_checkpoint.update({'latent_reprs': latent_structmod})
      RT.logger.info("Output results ...")
      RT.io.pdb.dump_af(file=RT.todir('result.pdb'), results=results, batch=out_checkpoint['batch'])
      RT.io.pkl.dump(file=RT.todir('result.pkl'), to_dump=results)
      RT.io.pkl.dump(file=RT.todir('checkpoint.pkl'), to_dump=out_checkpoint)


class InferFromCheckpointRunner(BaseRunner):
  r"""AF Checkpoint runner that infers from the checkpoints."""

  def __init__(self, project_dir: str = os.getcwd(), runtime_tag: str = r''):
    r"""Create an AF Checkpoint runner that infers from the checkpoints.

      Args:
        project_dir (str): The project base directory for the runner I/O.
        runtime_tag (str): The suffix tag to the runtime directory.
    """
    super().__init__(project_dir=project_dir,
                     runtime_tag=runtime_tag,
                     runner_name=r'chkp.infer_from_checkpoint', )

  def execute(self,
              config: _u.af.TAFConfig,
              params: _u.af.TAFParams,
              use_bf16: bool,
              checkpoint_pkl: _u.af.TAFFeatures,
              ) -> None:
    r"""Execute the runner.

      Args:
        config (TAFConfig): The AF configurations.
        params (TAFParams): The AF parameters.
        checkpoint_pkl (TAFFeatures): The checkpoint pickle file, `checkpoint.pkl`.
    """
    batch = checkpoint_pkl['batch']
    layer_index = checkpoint_pkl['batch']['aftools_layer_index']
    with Runtime(whoami=self.runner_name, working_dir=self.working_dir) as RT:
      # kernels.
      mod = _c.models.InferFromCheckpointModel(config=config, params=params, key=self.key)
      mod.compile(batch=batch, use_bf16=use_bf16, jit_pred=False)
      results = mod.predict(reprs=checkpoint_pkl['latent_reprs'])
      results = _u.af.cast_jnp_to_np(dict_jnp=results)
      # outputs.
      RT.logger.info("Output results ...")
      outname = f'latent_{layer_index}'
      RT.io.pdb.dump_af(file=RT.todir(f'{outname}_result.pdb'), results=results, batch=batch)
      RT.io.pkl.dump   (file=RT.todir(f'{outname}_result.pkl'), to_dump=results)
