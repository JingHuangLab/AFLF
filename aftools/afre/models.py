r"""AFTools AF random enumerator (AFRE) models."""
# Authors: Zilin Song.


import copy
import typing
import functools

import jax
import jax.numpy as jnp
import     haiku as  hk
import optax

from aftools import BaseModel
import aftools.utils as _u
import aftools.chkp  as _c
import aftools.lora  as _l
import aftools.geom  as _g
import aftools.afre  as _re


def afre_infer(lora_feats: _u.af.TAFFeatures,
               lora_masks: _u.af.TAFFeatures,
               reprs:      _u.af.TAFFeatures,
               batch:      _u.af.TAFFeatures,
               config:     _u.af.TAFConfig,
               layer_index: int,
               apply_reprs: typing.Literal['single', 'msa', 'pair', 'both'],
               use_bf16: bool, 
               af_confidence_loss_spec: dict[str, float],
               afre_geometry_loss_spec: dict[str, dict[str, typing.Union[float, jnp.ndarray]]],
               ) -> tuple[jnp.ndarray, _u.af.TAFResults]:
  r"""The AFRE inference function.

    Args:
      lora_feats (TAFFeatures): The LoRA features.
      lora_masks (TAFFeatures): The LoRA masks.
      reprs (TAFFeatures): The AF checkpoint representations.
      batch (TAFFeatures): The AF batch features.
      config (TAFConfig):  The AF configurations.
      layer_index (int): The Evoformer layer index in [0, 48] whose latent state is retrieved.
      apply_reprs (str): The representations to apply LoRA: `single`, `msa`, `pair`, `both`.
      use_bf16 (bool): Whether the bfloat16 runtime is requested.
      af_confidence_loss_spec (dict[str, float]):
        The AF confidence loss spec.
      afre_geometry_loss_spec (dict[str, dict[str, typing.Union[float, jnp.ndarray]]]):
        The AFRE Geometry loss spec.

    Returns:
      loss (jnp.ndarray): AFRE geometry loss.
      results (TAFResults): Auxiliary outputs.
  """
  results = _l.models.lora_infer(lora_feats=lora_feats,
                                 lora_masks=lora_masks,
                                 reprs=reprs,
                                 batch=batch,
                                 config=config,
                                 layer_index=layer_index,
                                 apply_reprs=apply_reprs,
                                 use_bf16=use_bf16, )
  cnfd, out_cnfd = _c. af_confidence_loss(results=results, spec=af_confidence_loss_spec)
  afre, out_afre = _re.afre_geometry_loss(results=results, spec=afre_geometry_loss_spec)
  loss = cnfd + afre
  results.update({'af_confidence_loss': out_cnfd, 'afre_geometry_loss': out_afre})
  return loss, results


def afre_log(results: _u.af.TAFResults) -> str:
  r"""The log string from the inferred results."""
  return ( "| af_confidence_loss "
          f"- plddt {results['af_confidence_loss']['plddt']:>8.4f} "
          f"- ptm {  results['af_confidence_loss']['ptm'  ]:>8.4f} "
           "| afre_geometry_loss "
          f"- anchoring {results['afre_geometry_loss']['anchoring']:>12.8f} "
          f"- cistogram {results['afre_geometry_loss']['cistogram']:> 8.4f} "
          f"- repelling {results['afre_geometry_loss']['repelling']:> 8.4f} "
          f"({           results['afre_geometry_loss']['repelling_num_repellents']}) "
          f"- rigidbody {results['afre_geometry_loss']['rigidbody']:> 8.4f} ")


class AFREState(_u.serialization.JSONable):
  r"""The AFRE state."""

  _json_attr_ = ['af_confidence_loss_spec',
                 'afre_geometry_loss_spec',
                 'optax_opt_state',
                 'hist_lora_feats',
                 'stat_repelling_dist_cor', ]

  def __init__(self,
               af_confidence_loss_spec: dict[str, float],
               afre_geometry_loss_spec: dict[str, dict[str, typing.Union[float, jnp.ndarray]]],
               optax_opt_state: optax.OptState, ):
    r"""Create an AFRE state.

      Args:
        af_confidence_loss_spec (AFConfidenceLossSpec): The AF Confidence loss spec.
        afre_geometry_loss_spec (AFREGeometryLossSpec): The AFRE Geometry loss spec.
        optax_opt_state         (optax.OptState):       The internal state of optax optimizer.
    """
    self.af_confidence_loss_spec = af_confidence_loss_spec
    self.afre_geometry_loss_spec = afre_geometry_loss_spec
    self.optax_opt_state = optax_opt_state
    self.hist_lora_feats: list[_u.af.TAFFeatures] = list()  # history tracker for `lora_feats`.
    self.stat_repelling_dist_cor = None # dist_cor statistics for diversifying weight.

  def record_hist(self, lora_feats: _u.af.TAFFeatures) -> None:
    r"""Record the `lora_feats` input to the AFRE model."""
    self.hist_lora_feats.append(copy.deepcopy(lora_feats))

  def update_stat(self, results: _u.af.TAFResults) -> None:
    r"""Update all running statistics."""
    # unpack.
    cor = results['structure_module']['final_atom_positions'].reshape(-1, 3)  # (Nres*37, 3)
    cent_masks = self.afre_geometry_loss_spec['repelling']['cent_masks']
    dist_cor = _g.dist.dist_self(cor=_g.cg.centroid_batched(cor=cor, masks=cent_masks))
    # update.
    self.stat_repelling_dist_cor = _g.stat.update(x=dist_cor, stat=self.stat_repelling_dist_cor)


class AFREModel(BaseModel):
  r"""The AFRE model."""

  def __init__(self, config: _u.af.TAFConfig, params: _u.af.TAFParams, key: jax.random.KeyArray):
    r"""Create an AFRE model.

      Args:
        config (TAFConfig): The AF configurations.
        params (TAFParams): The AF parameters.
        key (jax.random.KeyArray): The JAX PRNG Key.
    """
    super().__init__(config=config, params=params, key=key, multimer_mode=False)

  def compile(self,
              lora_masks: _u.af.TAFFeatures,
              reprs:      _u.af.TAFFeatures,
              batch:      _u.af.TAFFeatures,
              apply_reprs: typing.Literal['single', 'msa', 'pair', 'both'],
              use_bf16: bool,
              jit_pred: bool,
              jit_grad: bool,
              ) -> None:
    r"""Compile the model for inference.

      Args:
        lora_masks (TAFFeatures): The LoRA masks.
        reprs      (TAFFeatures): The AF checkpoint representations.
        batch      (TAFFeatures): The AF batch features.
        apply_reprs (str): The representations to apply LoRA: `single`, `msa`, `pair`, `both`.
        use_bf16 (bool): Whether the bfloat16 runtime is requested.
        jit_pred (bool): Whether the `predict` function is JIT-compiled.
        jit_auto (bool): Whether the `autograd` function is JIT-compiled.
    """
    # checkpoint flag.
    layer_index = batch['aftools_layer_index']
    # params.
    prefix = _c.models.infer_scope(layer_index=layer_index, use_bf16=use_bf16)
    self.params = _u.af.prep_params(params=self.params, prefix=prefix, layer_index=layer_index)
    # pred function.
    f_pred = functools.partial(_l.models.lora_infer,
                               lora_masks =lora_masks,
                               reprs      =reprs,
                               batch      =batch,
                               config     =self.config,
                               layer_index=layer_index,
                               apply_reprs=apply_reprs,
                               use_bf16=use_bf16, )
    self.f_pred = hk.transform(f=f_pred).apply
    if jit_pred: self.f_pred = jax.jit(self.f_pred)
    # grad function.
    f_grad = functools.partial(afre_infer,
                               lora_masks =lora_masks,
                               reprs      =reprs,
                               batch      =batch,
                               config     =self.config,
                               layer_index=layer_index,
                               apply_reprs=apply_reprs,
                               use_bf16=use_bf16, )
    self.f_grad = jax.grad(hk.transform(f=f_grad).apply, argnums=2, has_aux=True)
    if jit_grad: self.f_grad = jax.jit(self.f_grad)

  def predict(self, lora_feats: _u.af.TAFFeatures) -> _u.af.TAFResults:
    r"""Predict with the model.

      Args:
        lora_feats (TAFFeatures): The LoRA features.

      Returns:
        results (TAFResults): The AF output results.
    """
    results = self.f_pred(self.params, self.key, lora_feats)
    return results

  def autograd(self,
               lora_feats: _u.af.TAFFeatures,
               afre_state: AFREState,
               ) -> tuple[jnp.ndarray, _u.af.TAFResults]:
    r"""Auto-differentiate the model w.r.t. `lora_feats`.

      Args:
        lora_feats (TAFFeatures): The LoRA features.
        afre_state (AFREState):   The AFRE state.

      Returns:
        lora_grads (TAFFeatures): The gradient tensors to `lora_feats`.
        results    (TAFResults):  The AF output results.
    """
    key, self.key = jax.random.split(key=self.key, num=2)
    lora_grads, results = self.f_grad(self.params, key, lora_feats,
                                      af_confidence_loss_spec=afre_state.af_confidence_loss_spec,
                                      afre_geometry_loss_spec=afre_state.afre_geometry_loss_spec, )
    return lora_grads, results