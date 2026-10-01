r"""AFTools AF random enumerator (AFRE) runners."""
# Authors: Zilin Song.


import os
import time
import typing
import functools

import jax
import jax.numpy as jnp
import optax

from aftools import BasePreset, BaseRunner, Runtime
import aftools.utils as _u
import aftools.chkp  as _c
import aftools.lora  as _l
import aftools.geom  as _g
import aftools.afre  as _re


class AFRERunnerPreset(BasePreset):
  r"""The AFRE Runner preset."""

  def __init__(self):
    r"""Create an AFRE Runner preset."""
    # runner configs.
    self.lora = _l.LoraCheckpointPreset()
    r"""The LoRA preset."""
    self.af_confidence_loss_spec = _c .AFConfidenceLossSpecPreset()
    r"""The AF Confidence loss spec preset."""
    self.afre_geometry_loss_spec = _re.AFREGeometryLossSpecPreset()
    r"""The AFRE Geometry loss spec preset."""
    self.optax_optimizer = optax.adam(learning_rate=0.002)
    r"""The optax optimizer, default: `optax.adam(learning_rate=0.002)`."""
    # AFRE runtime configs.
    self.num_steps: int = 10000
    r"""The number of integration steps, default: `10000`."""
    self.traj_freq: int = 1
    r"""The AFRE frequency of saving the AFRE trajectories, default: `1`."""
    self.save_freq: int = 1000
    r"""The AFRE frequency of saving the AFRE runner states if DEBUG=True, default: `1000`."""
    self.repelling_pushstack_freq: int = 10
    r"""The AFRE frequency of pushing a new repellent state, default: `10`."""
    self.repelling_pushstack_pert: float = 0.1
    r"""The AFRE scaling to the pair distances of the new repellent state, default: 0.1."""
    self.repelling_diversify_freq: int = 10
    r"""The AFRE frequency of updating the diversifying weight, default: `1`."""
    self.repelling_diversify_rtau: float = 0.
    r"""The AFRE reciprocal temperature of the diversifying weight, default: `0.0`."""
    self.repelling_diversify_apply_to: typing.Literal['dist', 'cent'] = 'dist'
    r"""The AFRE target to which the diversifying weights are taken, default: `dist`."""
    self.DEBUG: bool = False
    r"""True causes AFRERunner to output many of its internal files (can be huge in file size)."""

  def instantiate(self, project_dir: str = os.getcwd(), runtime_tag: str = r'') -> "AFRERunner":
    r"""Create an AFRE runner from this preset.

      Args:
        project_dir (str): The project base directory for the runner I/O.
        runtime_tag (str): The suffix tag to the runtime directory.
    """
    runner = AFRERunner(project_dir=project_dir, runtime_tag=runtime_tag)
    runner.preset = self
    return runner


def _runner_initialized(func: typing.Callable) -> typing.Callable:
  r"""Decorator that ensures the method is called after `AFRERunner().execute_init()`."""
  @functools.wraps(func)
  def wrapper(inst: "AFRERunner", *args, **kwargs) -> typing.Any:
    if None not in [inst.afre_model, inst.afre_state, inst.optimizer, inst.istep]:
      return func(inst, *args, **kwargs)
    raise RuntimeError("Illegal AFRERunner uninitialized.")
  return wrapper


class AFRERunner(BaseRunner):
  r"""The AFRE Runner."""

  def __init__(self, project_dir: str = os.getcwd(), runtime_tag: str = r''):
    r"""Create an AFRE runner.

      Args:
        project_dir (str): The project base directory for the runner I/O.
        runtime_tag (str): The suffix tag to the runtime directory.
    """
    super().__init__(project_dir=project_dir, runtime_tag=runtime_tag, runner_name=r'afre.runner')
    self._preset = AFRERunnerPreset()
    self._afre_model: _re.models.AFREModel = None
    self._afre_state: _re.models.AFREState = None
    self._optimizer: optax.GradientTransformation = None
    self._istep: int = None

  @property
  def preset(self) -> AFRERunnerPreset:
    return self._preset
  @preset.setter
  def preset(self, val: AFRERunnerPreset) -> None:
    assert isinstance(val, AFRERunnerPreset), f"Illegal `type(preset)={type(val)}`."
    self._preset = val

  @property
  def afre_model(self) -> _re.models.AFREModel:
    return self._afre_model
  @afre_model.setter
  def afre_model(self, val: _re.models.AFREModel) -> None:
    assert isinstance(val, _re.models.AFREModel), f"Illegal `type(val)={type(val)}`."
    self._afre_model = val

  @property
  def afre_state(self) -> _re.models.AFREState:
    return self._afre_state
  @afre_state.setter
  def afre_state(self, val: _re.models.AFREState) -> None:
    assert isinstance(val, _re.models.AFREState), f"Illegal `type(optim)={type(val)}`."
    self._afre_state = val

  @property
  def optimizer(self) -> optax.GradientTransformation:
    return self._optimizer
  @optimizer.setter
  def optimizer(self, val: optax.GradientTransformation) -> None:
    assert isinstance(val, optax.GradientTransformation), f"Illegal `type(optim)={type(val)}`."
    self._optimizer = val

  @property
  def istep(self) -> int:
    return self._istep
  @istep.setter
  def istep(self, val: int) -> None:
    assert isinstance(val, int), f"Illegal `type(istep)={type(val)}`."
    self._istep = int(val)

  def exec_init(self, RT: Runtime,
                config: _u.af.TAFConfig,
                params: _u.af.TAFParams,
                apply_reprs: typing.Literal['single', 'msa', 'pair', 'both'],
                use_bf16: bool,
                checkpoint_pkl:  _u.af.TAFFeatures,
                anchoring_indexers: _u.selection.AtomIndexerGroup,
                cistogram_indexers: _u.selection.AtomIndexerGroup,
                repelling_indexers: _u.selection.AtomIndexerGroup,
                rigidbody_indexers: _u.selection.AtomIndexerGroup,
                ) -> _u.af.TAFFeatures:
    r"""Execute the Runner initialization: (1) Create the LoRA features; (2) Compile the AFRE model;
      (3) Performs initial inference; (4) Initialize the AFRE optimizer, and; (5) the AFRE state.

      Args:
        RT (Runtime): The Runtime context manager.
        config (TAFConfig): The AF configurations.
        params (TAFParams): The AF parameters.
        apply_lora (str): The representation to apply LoRA: `single`, `msa`, `pair`, `both`.
        use_bf16 (bool): Whether the bfloat16 runtime is requested.
        checkpoint_pkl (TAFFeatures): The checkpoint pickle file, `checkpoint.pkl`.
        anchoring_indexers (AtomIndexerGroup): The anchoring atom indexer group.
        cistogram_indexers (AtomIndexerGroup): The cistogram atom indexer group.
        repelling_indexers (AtomIndexerGroup): The repelling atom indexer group.
        rigidbody_indexers (AtomIndexerGroup): The rigidbody atom indexer group.

      Returns:
        lora_checkpoint_pkl (TAFFeatures): The LoRA checkpoint pickle file, `lora_checkpoint.pkl`.
    """
    key_lora, key_afre, self.key = jax.random.split(self.key, num=3)
    layer_index = checkpoint_pkl['batch']['aftools_layer_index']
    # LoRA.
    RT.logger.info(f"Initializing the LoRA features ... `layer_index={layer_index}`.")
    _lora_pkl = self.preset.lora.instantiate(reprs=checkpoint_pkl['latent_reprs'],
                                             batch=checkpoint_pkl['batch'],
                                             layer_index=layer_index,
                                             apply_reprs=apply_reprs,
                                             key=key_lora, )
    if self.preset.DEBUG:
      RT.io.pkl.dump(file=RT.todir('lora_checkpoint.pkl'), to_dump=_u.af.cast_jnp_to_np(_lora_pkl))
    _lora_feats = _lora_pkl['latent_lora_feats']
    # AFRE model.
    RT.logger.info("Initializing the AFRE model ...")
    self.afre_model = _re.models.AFREModel(config=config, params=params, key=key_afre)
    self.afre_model.compile(lora_masks=_lora_pkl['latent_lora_masks'],
                            reprs     =_lora_pkl['latent_reprs'],
                            batch     =_lora_pkl['batch'],
                            apply_reprs=apply_reprs,
                            use_bf16=use_bf16,
                            jit_pred=False,
                            jit_grad=True, )
    RT.logger.info( "Initializing the AFRE first inference ...")
    _t = time.perf_counter()
    _results = self.afre_model.predict(lora_feats=_lora_feats)
    RT.logger.info(f"Initializing the AFRE first inference ... {time.perf_counter() - _t:6.2f}s")
    RT.io.pdb.dump_af(file=RT.todir('init_results.pdb'), results=_results, batch=_lora_pkl['batch'])
    if self.preset.DEBUG:
      RT.io.pkl.dump (file=RT.todir('init_results.pkl'), to_dump=_results)
    # Optimizer.
    RT.logger.info(f"Initializing the AFRE optimizer ...")
    self.optimizer = self.preset.optax_optimizer
    _optax_opt_state = self.optimizer.init(params=_lora_feats)
    # AFRE state.
    RT.logger.info(f"Initializing the AFRE state ...")
    _af_confidence_loss_spec = self.preset.af_confidence_loss_spec.instantiate()
    _afre_geometry_loss_spec = self.preset.afre_geometry_loss_spec.instantiate(
                                                            batch=_lora_pkl['batch'],
                                                            results=_results,
                                                            anchoring_indexers=anchoring_indexers,
                                                            cistogram_indexers=cistogram_indexers,
                                                            repelling_indexers=repelling_indexers,
                                                            rigidbody_indexers=rigidbody_indexers, )
    self.afre_state = _re.models.AFREState(af_confidence_loss_spec=_af_confidence_loss_spec,
                                           afre_geometry_loss_spec=_afre_geometry_loss_spec,
                                           optax_opt_state=_optax_opt_state, )
    self.afre_state.update_stat(results=_results)
    if self.preset.DEBUG:
      self.afre_state.record_hist(lora_feats=_lora_feats)
    # Step counter.
    self.istep = 0
    # Renew initial state.
    self.exec_repelling_pushstack(RT=RT, results=_results)
    self.exec_repelling_diversify(RT=RT)
    if self.preset.DEBUG:
      RT.io.txt.dump(file=RT.todir('afre_state_init.json'), to_dump=self.afre_state.to_json())
    return _lora_pkl

  @_runner_initialized
  def exec_step(self, lora_feats: _u.af.TAFFeatures) -> tuple[_u.af.TAFFeatures, _u.af.TAFResults]:
    r"""Execute the AFRE step.

      Args:
        lora_feats (TAFFeatures): The LoRA features.

      Returns:
        lora_feats (TAFFeatures): The updated LoRA features.
        results    (TAFResults):  The AF output results.
    """
    # Optimizer step.
    _t = time.perf_counter()
    _grads, results = self.afre_model.autograd(lora_feats=lora_feats, afre_state=self.afre_state)
    _updts, opt_state = self.optimizer.update(updates=_grads, state=self.afre_state.optax_opt_state)
    lora_feats = optax.apply_updates(params=lora_feats, updates=_updts)
    results.update({'step_time': time.perf_counter()-_t})
    # AFRE state update.
    self.afre_state.optax_opt_state = opt_state
    self.afre_state.update_stat(results=results)
    if self.preset.DEBUG:
      self.afre_state.record_hist(lora_feats=lora_feats)
    self.istep += 1
    return lora_feats, results

  @_runner_initialized
  def exec_log(self, RT: Runtime, results: _u.af.TAFResults) -> None:
    r"""Execute the AFRE logging of the log.

      Args:
        RT      (Runtime):     The Runtime context manager.
        batch   (TAFFeatures): The AF batch features.
        results (TAFFeatures): The AF output results.
    """
    _logstr = f"AFRE iteration step {self.istep:>4} - {results['step_time']:6.2f}s "
    _logstr+= _re.models.afre_log(results=results)
    RT.logger.info(_logstr)
    RT.io.log.write(file=RT.todir('afre_traj.log'), log=f"{_logstr}\n")

  @_runner_initialized
  def exec_repelling_pushstack(self, RT: Runtime, results: _u.af.TAFResults) -> None:
    r"""Execute the AFRE protocol: Perturbs the new repellent state; Prepends the repellent state to
      the stack of repellents, and; Pops the rightmost repellent from the queue of repellents.

      Args:
        RT (Runtime): The Runtime context manager.
        results (TAFFeatures): The AF output results.
    """
    _logstr = f"AFRE iteration step {self.istep:>4} - updating AFRE repellent state ..."
    RT.logger.info(_logstr)
    RT.io.log.write(file=RT.todir('afre_traj.log'), log=f"{_logstr}\n")
    # unpack.
    ref = results['structure_module']['final_atom_positions'].reshape(-1, 3)  # (Nres*37, 3)
    cent_masks   = self.afre_state.afre_geometry_loss_spec['repelling']['cent_masks']
    dist_refs    = self.afre_state.afre_geometry_loss_spec['repelling']['dist_refs']
    weights_mask = self.afre_state.afre_geometry_loss_spec['repelling']['weights_mask']
    key, self.key = jax.random.split(key=self.key, num=2)
    # perturb.
    dist_ref = _g.dist.dist_self(cor=_g.cg.centroid_batched(cor=ref, masks=cent_masks))
    dist_ref+= _g.rand.uniform(key=key, shape=dist_ref.shape) * self.preset.repelling_pushstack_pert
    # prepend & pop the rightmost.
    dist_refs = jnp.vstack((jnp.expand_dims(dist_ref, axis=0), dist_refs)).at[:-1, :, :].get()
    # offset rightward the weights_mask to 1 and update to the new weights_mask.
    weights_mask = weights_mask.at[int(jnp.sum(weights_mask))].set(1)
    # make the new spec and update to state.
    new_spec = {'dist_refs': jnp.copy(dist_refs), 'weights_mask': jnp.copy(weights_mask)}
    self.afre_state.afre_geometry_loss_spec['repelling'].update(new_spec)

  @_runner_initialized
  def exec_repelling_diversify(self, RT: Runtime) -> None:
    r"""Execute the AFRE update of the repelling diversity weights.

      Args:
        RT (Runtime): The Runtime context manager.
    """
    _logstr = f"AFRE iteration step {self.istep:>4} - updating AFRE diversify weight ..."
    RT.logger.info(_logstr)
    RT.io.log.write(file=RT.todir('afre_traj.log'), log=f"{_logstr}\n")
    # unpack.
    mask = self.afre_state.afre_geometry_loss_spec['repelling']['mask']
    rtau = self.preset.repelling_diversify_rtau
    cv   = _g.stat.reduce(stat=self.afre_state.stat_repelling_dist_cor)['cv']
    # compute the diversifying weights (from the coefficient of variation).
    if self.preset.repelling_diversify_apply_to == 'dist':
      expo = jnp.where(mask==0., jnp.inf, cv*rtau).reshape(-1)      # (Ncent^2, )
      nele = jnp.sum(mask)
    if self.preset.repelling_diversify_apply_to == 'cent':
      expo = jnp.sum(cv*rtau*mask, axis=-1)                         # (Ncent, )
      expo = jnp.where(jnp.sum(mask, axis=-1)==0., jnp.inf, expo)   # (Ncent, )
      nele = jnp.sum(mask) / jnp.sum(mask, axis=-1, keepdims=True)  # (Ncent, 1) per-row.
    wraw = jax.nn.softmax(-expo).reshape(mask.shape[0], -1)   # (Ncent, Ncent) or (Ncent, 1), sum=1
    diver_weight = wraw * nele * mask                         # (Ncent, Ncent), mean=1, sum=nele
    new_spec = {'diver_weight': jnp.copy(diver_weight)}
    self.afre_state.afre_geometry_loss_spec['repelling'].update(new_spec)
    # outputs to file.
    _logstr = (f"step {self.istep}:\n"
               f"mean = {jnp.sum(diver_weight)/jnp.count_nonzero(diver_weight)}\n"
               f"{str(diver_weight)}")
    if self.preset.DEBUG:
      RT.io.txt.dump(file=RT.todir(f'diver_weight.log'), to_dump=_logstr)

  @_runner_initialized
  def exec_save(self, RT: Runtime) -> None:
    r"""Execute the AFRE state saving if in DEBUG mode.

      Args:
        RT (Runtime): The Runtime context manager.
    """
    if self.preset.DEBUG:
      RT.logger.info(f"AFRE iteration step {self.istep:>4} - saving runner states ...")
      RT.io.txt.dump(file=RT.todir('afre_state.json'), to_dump=self.afre_state.to_json())

  def execute(self,
              config: _u.af.TAFConfig,
              params: _u.af.TAFParams,
              apply_reprs: typing.Literal['single', 'msa', 'pair', 'both'],
              use_bf16: bool,
              checkpoint_pkl:  _u.af.TAFFeatures,
              anchoring_indexers: _u.selection.AtomIndexerGroup,
              cistogram_indexers: _u.selection.AtomIndexerGroup,
              repelling_indexers: _u.selection.AtomIndexerGroup,
              rigidbody_indexers: _u.selection.AtomIndexerGroup,
              ) -> None:
    r"""Execute the Runner.

      Args:
        RT (Runtime): The Runtime context manager.
        config (TAFConfig): The AF configurations.
        params (TAFParams): The AF parameters.
        apply_lora (str): The representation to apply LoRA: `single`, `msa`, `pair`, `both`.
        use_bf16 (bool): Whether the bfloat16 runtime is requested.
        checkpoint_pkl (TAFFeatures): The checkpoint pickle file, `checkpoint.pkl`.
        anchoring_indexers (AtomIndexerGroup): The anchoring atom indexer group.
        cistogram_indexers (AtomIndexerGroup): The cistogram atom indexer group.
        repelling_indexers (AtomIndexerGroup): The repelling atom indexer group.
        rigidbody_indexers (AtomIndexerGroup): The rigidbody atom indexer group.
    """
    with Runtime(whoami=self.runner_name, working_dir=self.working_dir) as RT:
      # init.
      RT.logger.info("AFRE initializations ...")
      lora_pkl = self.exec_init(RT=RT,
                                config=config,
                                params=params,
                                apply_reprs=apply_reprs,
                                use_bf16=use_bf16,
                                checkpoint_pkl=checkpoint_pkl,
                                anchoring_indexers=anchoring_indexers,
                                cistogram_indexers=cistogram_indexers,
                                repelling_indexers=repelling_indexers,
                                rigidbody_indexers=rigidbody_indexers, )
      lora_feats, batch = lora_pkl['latent_lora_feats'], lora_pkl['batch']
      # iter.
      RT.logger.info("AFRE iterations ...")

      for _ in range(self.preset.num_steps):
        lora_feats, results = self.exec_step(lora_feats=lora_feats)

        self.exec_log(RT=RT, results=results)

        if (self.istep % self.preset.traj_freq) == 0:
          RT.io.dcd.write_af(file=RT.todir('afre_traj.dcd'), batch=batch, results=results)

        if (self.istep % self.preset.repelling_pushstack_freq) == 0:
          self.exec_repelling_pushstack(RT=RT, results=results)

        if (self.istep % self.preset.repelling_diversify_freq) == 0:
          self.exec_repelling_diversify(RT=RT)

        if (self.istep % self.preset.save_freq) == 0:
          self.exec_save(RT=RT)

      self.exec_save(RT=RT)
