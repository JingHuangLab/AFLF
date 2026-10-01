r"""AFTools checkpoint confidence computations."""
# Authors: Zilin Song.


import typing
import functools

import jax
import jax.numpy as jnp

from aftools import BasePreset
import aftools.utils as _u


# pLDDT.
def _plddt(logits: jnp.ndarray, num_bins: int) -> jnp.ndarray:
  r"""JAX-differentiable implementation of the AF pLDDT scores.

    Args:
      logits   (jnp.ndarray): From `AFResults['predicted_lddt']['logits']`.
      num_bins (int):         From `config.model.heads.predicted_lddt.num_bins`.

    Returns:
      plddt (jnp.ndarray): The pLDDT scores per residue.
  """
  bin_width = 1. / float(num_bins)
  bin_centers = jnp.arange(start=.5*bin_width, stop=1., step=bin_width)
  # From: alphafold/common/confidence.py, line 34-36.
  return jnp.sum(jax.nn.softmax(logits, axis=-1) * bin_centers[None, :], axis=-1) # [Nres, ]


# pTM and ipTM.
def _bin_tm(num_res: int, num_bins: int, max_error_bin: float) -> jnp.ndarray:
  r"""Compute the bin TM scores for the pTM computation.

    Args:
      num_res       (int):   From `af_batch['aatype'].shape[0]`.
      num_bins      (int):   From `af_config.model.heads.predicted_aligned_error.num_bins`.
      max_error_bin (float): From `af_config.model.heads.predicted_aligned_error.max_error_bin`.
  """
  # compute d0.
  ## From: alphafold/common/confidence.py, line 142-147.
  num_res = int(num_res) if num_res >= 19. else 19
  d0      = 1.24 * (num_res - 15) ** (1./3.) - 1.8
  # compute bin_centers.
  ## From: alphafold/model/modules.py, line 1160-1162.
  breaks = jnp.linspace(0., float(max_error_bin), num_bins-1)
  ## From: alphafold/common/confidence.py, line 50-54.
  step = breaks[1] - breaks[0]
  bin_centers = jnp.asarray(breaks + step / 2.)
  bin_centers = jnp.concatenate((bin_centers, jnp.asarray([bin_centers[-1] + step])), axis=0)
  ## From: alphafold/common/confidence.py, line 152-153.
  bin_tm = 1. / (1. + jnp.square(bin_centers) / jnp.square(d0))
  return bin_tm


def _ptm(logits: jnp.ndarray, bin_tm: jnp.ndarray) -> jnp.ndarray:
  r"""JAX-differentiable implementation of the AF pTM scores.

    Args:
      logits (jnp.ndarray): From `AFResults['predicted_aligned_error']['logits']`.
      bin_tm (jnp.ndarray): From `aftools.chkp.modules._ptm_bin_tm()`.

    Returns:
      ptm (jnp.ndarray): The pTM scores per residue.
  """
  # From: alphafold/common/confidence.py, line 149-155.
  return jnp.mean(jnp.sum(jax.nn.softmax(logits, axis=-1)*bin_tm, axis=-1), axis=-1)


def _iptm(logits: jnp.ndarray, bin_tm: jnp.ndarray, asym_id: jnp.ndarray) -> jnp.ndarray:
  r"""JAX-differentiable implementation of the AF-Multimer ipTM scores.

    Args:
      logits  (jnp.ndarray): From `AFResults['predicted_aligned_error']['logits']`.
      bin_tm  (jnp.ndarray): From `aftools.chkp.modules._ptm_bin_tm()`.
      asym_id (jnp.ndarray): From `AFResults['predicted_aligned_error']['asym_id']`.

    Returns:
      iptm (jnp.ndarray): The ipTM scores per residue.
  """
  # From: alphafold/common/confidence.py, line 149-159.
  iptm = jnp.sum(jax.nn.softmax(logits, axis=-1) * bin_tm, axis=-1)
  # From: alphafold/common/confidence.py, line 165-167.
  pair_mask = (asym_id[:, None]!=asym_id[None, :])
  pair_mask = pair_mask / jnp.sum(pair_mask, axis=-1, keepdims=True)
  return jnp.sum(iptm * pair_mask, axis=-1)


# confidence modules.
def af_confidence(results: _u.af.TAFResults,
                  batch:   _u.af.TAFFeatures,
                  plddt_num_bins:    int,
                  ptm_num_bins:      int,
                  ptm_max_error_bin: float,
                  ) -> dict[str, jnp.ndarray]:
  r"""JAX-differentiable implementation of the AF confidence scores.

    Args:
      results (TAFResults):  The AF output results.
      batch   (TAFFeatures): The AF batch features.
      plddt_num_bins    (int):   From `config.model.heads.predicted_lddt.num_bins`.
      ptm_num_bins      (int):   From `config.model.heads.predicted_aligned_error.num_bins`.
      ptm_max_error_bin (float): From `config.model.heads.predicted_aligned_error.max_error_bin`.

    Returns:
      results (dict[str, jnp.ndarray]): The AF confidence score as a `dict()` with keys:
        `plddt` (jnp.ndarray): The pLDDT scores per residue;
        `ptm`   (jnp.ndarray): The pTM   scores per residue, AF takes the max value.
  """
  ptm_bin_tm = _bin_tm(num_res=batch['aatype'].shape[0],
                       num_bins=ptm_num_bins,
                       max_error_bin=ptm_max_error_bin, )
  # scores.
  out_plddt = _plddt(logits=results['predicted_lddt']['logits'], num_bins=plddt_num_bins)
  out_ptm   = _ptm  (logits=results['predicted_aligned_error']['logits'], bin_tm=ptm_bin_tm)
  return dict(plddt=out_plddt, ptm=out_ptm)


def AFConfidenceModule(config: _u.af.TAFConfig,
                       ) -> typing.Callable[[_u.af.TAFResults,
                                             _u.af.TAFFeatures, ], dict[str, jnp.ndarray]]:
  r"""Create an AF confidence module.

    Args:
      config (TAFConfig): The AF model configuration.

    Returns:
      AFConfidence_f (Callable):
        The JAX-differentiable implementation of the AF confidence scores.
        Input:
          results (TAFResults):  The AF output results.
          batch   (TAFFeatures): The AF batch features.
        Output:
          results (TAFResults) the AF confidence score as a `dict()` with keys:
            `plddt` (jnp.ndarray): The pLDDT scores per residue, metrics take the mean value.
            `ptm`   (jnp.ndarray): The pTM   scores per residue, metrics take the max  value.
  """
  m = functools.partial(af_confidence,
                        plddt_num_bins   =config.model.heads.predicted_lddt.num_bins,
                        ptm_num_bins     =config.model.heads.predicted_aligned_error.num_bins,
                        ptm_max_error_bin=config.model.heads.predicted_aligned_error.max_error_bin,)
  return m


def af_multimer_confidence(results: _u.af.TAFResults,
                           batch:   _u.af.TAFFeatures,
                           plddt_num_bins:    int,
                           ptm_num_bins:      int,
                           ptm_max_error_bin: float,
                           ) -> dict[str, jnp.ndarray]:
  r"""JAX-differentiable implementation of the AF-Multimer confidence scores.

    Args:
      results (TAFResults):  The AF-Multimer output results.
      batch   (TAFFeatures): The AF-Multimer batch features.
      plddt_num_bins    (int):   From `config.model.heads.predicted_lddt.num_bins`.
      ptm_num_bins      (int):   From `config.model.heads.predicted_aligned_error.num_bins`.
      ptm_max_error_bin (float): From `config.model.heads.predicted_aligned_error.max_error_bin`.

    Returns:
      results (dict[str, jnp.ndarray]): The AF-Multimer confidence score as a `dict()` with keys:
        `plddt` (jnp.ndarray): The  pLDDT scores per residue;
        `ptm`   (jnp.ndarray): The  pTM   scores per residue, AF takes the max value.
        `iptm`  (jnp.ndarray): The ipTM   scores per residue, AF takes the max value.
  """
  ptm_btm = _bin_tm(num_res      =batch['aatype'].shape[0],
                    num_bins     =ptm_num_bins,
                    max_error_bin=ptm_max_error_bin, )
  # scores.
  out_plddt = _plddt(logits =results['predicted_lddt']['logits'], num_bins=plddt_num_bins)
  out_ptm   = _ptm  (logits =results['predicted_aligned_error']['logits'], bin_tm=ptm_btm)
  out_iptm  =_iptm  (logits =results['predicted_aligned_error']['logits'], bin_tm=ptm_btm,
                     asym_id=results['predicted_aligned_error']['asym_id'], )
  return dict(plddt=out_plddt, ptm=out_ptm, iptm=out_iptm)


def AFMultimerConfidenceModule(config: _u.af.TAFConfig,
                               ) -> typing.Callable[[_u.af.TAFResults,
                                                     _u.af.TAFFeatures, ], dict[str, jnp.ndarray]]:
  r"""Create an AF-Multimer confidence module.

    Args:
      config (TAFConfig): The AF-Multimer model configuration.

    Returns:
      AFConfidence_f (Callable):
        The JAX-differentiable implementation of the AF confidence scores.
        Input:
          results (TAFResults):  The AF output results.
          batch   (TAFFeatures): The AF batch features.
        Output:
          results (TAFResults): the AF confidence score as a `dict()` with keys:
            `plddt` (jnp.ndarray): The  pLDDT scores per residue, metrics take the mean value.
            `ptm`   (jnp.ndarray): The  pTM   scores per residue, metrics take the max  value.
            `iptm`  (jnp.ndarray): The ipTM   scores per residue, metrics take the max  value.
  """
  m = functools.partial(af_multimer_confidence,
                        plddt_num_bins   =config.model.heads.predicted_lddt.num_bins,
                        ptm_num_bins     =config.model.heads.predicted_aligned_error.num_bins,
                        ptm_max_error_bin=config.model.heads.predicted_aligned_error.max_error_bin,)
  return m


# confidence losses.
def af_confidence_loss(results: _u.af.TAFResults,
                       spec: dict[str, float],
                       ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
  r"""Compute the AF confidence loss.

    Args:
      results (TAFResults): The AF output results.
      spec    (dict[str, float]):
        The AF confidence loss spec with keys:
        `weight_plddt` (float): The pLDDT loss weight.
        `weight_ptm`   (float): The pTM   loss weight.

    Returns:
      loss    (jnp.ndarray): The mean-reduced AF confidence loss.
      results (TAFResults):  The AF confidence loss results: `sum`, `plddt`, `ptm`.
  """
  val_plddt, val_ptm = results['confidence']['plddt'], results['confidence']['ptm']
  loss_plddt = spec['weight_plddt'] * jnp.mean(1.-val_plddt)
  loss_ptm   = spec['weight_ptm'  ] * jnp.mean(1.-val_ptm  )
  # pack.
  loss = loss_plddt + loss_ptm
  aux = {'sum':   jax.lax.stop_gradient(loss      ),
         'plddt': jax.lax.stop_gradient(loss_plddt),
         'ptm':   jax.lax.stop_gradient(loss_ptm  ), }
  return loss, aux


class AFConfidenceLossSpecPreset(BasePreset):
  r"""The AF Confidence loss spec preset."""

  def __init__(self):
    r"""Create an AF confidence loss spec preset."""
    self.weight_plddt: float = 1.
    r"""The pLDDT loss weight, default: 1."""
    self.weight_ptm:   float = 0.
    r"""The pTM loss weight, default: 0."""

  def instantiate(self) -> dict[str, float]:
    r"""Create the AF confidence loss spec from this preset.

      Returns:
        loss_spec (dict[str, float]):
          The AF Confidence loss spec with keys:
            `weight_plddt` (float): The pLDDT loss weight.
            `weight_ptm`   (float): The pTM   loss weight.
    """
    return {'weight_plddt': float(self.weight_plddt), 'weight_ptm': float(self.weight_ptm)}
