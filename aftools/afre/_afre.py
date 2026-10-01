r"""AFTools AF Random Enumerator (AFRE)."""
# Authors: Zilin Song.


import typing

import jax
import jax.numpy as jnp

from aftools import BasePreset
import aftools.utils as _u
import aftools.afre  as _re


def afre_geometry_loss(results: _u.af.TAFResults,
                       spec:    dict[str, dict[str, typing.Union[float, jnp.ndarray]]],
                       ) -> tuple[jnp.ndarray, _u.af.TAFResults]:
  r"""Computes the AFRE Geometry loss.

    Args:
      results (TAFResults): The AF output results.
      spec (dict[str, dict[str, Union[float, jnp.ndarray]]]): The AFRE Geometry loss spec with keys:
        'anchoring' (dict[str, Union[float, jnp.ndarray]]): The anchoring loss specs with keys:
          `cent_masks`   (jnp.ndarray): The masks for centroid divisions (Ncent, Nres*37).
          `dist_ref`     (jnp.ndarray): The reference centroid distances (Ncent, Ncent).
          `thresh_lower` (jnp.ndarray): The lower bound of the tolerated distance differences.
          `thresh_upper` (jnp.ndarray): The upper bound of the tolerated distance differences.
          `switch_delta` (float):       The delta param of the Huber distance differences loss.
          `mask`         (jnp.ndarray): The loss mask (Ncent, Ncent).
          `weight`       (float):       The loss weight.
          `dropout_p`    (float):       The loss dropout rate.
        'cistogram' (dict[str, Union[float, jnp.ndarray]]): The cistogram loss specs with keys:
          `cent_masks`   (jnp.ndarray): The masks for centroid divisions (Ncent, Nres*37).
          `prob_ref`     (jnp.ndarray): The reference centroid distance density (Ncent^2, Nbin).
          `gauss_breaks` (jnp.ndarray): The Gaussian KDE histogram bin edges (Nbin-1, ).
          `gauss_rwidth` (float):       The Gaussian KDE reciprocal smearing width.
          `mask`         (jnp.ndarray): The loss mask (Ncent, Ncent).
          `weight`       (float):       The loss weight.
          `dropout_p`    (float):       The loss dropout rate.
        'repelling' (dict[str, Union[float, jnp.ndarray]]): The repelling loss specs with keys:
          `cent_masks`   (jnp.ndarray): The masks for centroid divisions (Ncent, Nres*37).
          `dist_refs`    (jnp.ndarray): The reference centroid distances (Nrep, Ncent, Ncent).
          `gauss_rwidth` (float):       The repulsive Gaussian reciprocal width.
          `diver_weight` (jnp.ndarray): The diversity weights per distances (Ncent, Ncent).
          `mask`         (jnp.ndarray): The loss mask (Ncent, Ncent).
          `weights`      (jnp.ndarray): The loss weights      per repellent (Nrep, ).
          `weights_mask` (jnp.ndarray): The loss weights mask per repellent (Nrep, ).
          `dropout_p`    (float):       The loss dropout rate.
        'rigidbody' (dict[str, Union[float, jnp.ndarray]]): The rigidbody loss specs with keys:
          `masks`     (jnp.ndarray): The masks for segment divisions (Nseg, Nres*37).
          `ref`       (jnp.ndarray): The reference atom coordinates (Nres*37, 3).
          `weight`    (float):       The loss weight.
          `dropout_p` (float):       The loss dropout rate.

    Returns:
      loss    (jnp.ndarray): The AFRE Geometry loss.
      results (TAFResults):  The AFRE Geometry results with keys:
        `sum`                      (jnp.ndarray): The AFRE     total loss.
        `anchoring`                (jnp.ndarray): The AFRE anchoring loss.
        `cistogram`                (jnp.ndarray): The AFRE cistogram loss.
        `repelling`                (jnp.ndarray): The AFRE repelling loss.
        `rigidbody`                (jnp.ndarray): The AFRE rigidbody loss.
        `repelling_num_repellents` (jnp.ndarray): The number of AFRE repellent states.
  """
  # unpack.
  cor = results['structure_module']['final_atom_positions'].reshape(-1, 3)  # (Nres*37, 3)
  # losses.
  loss_anchoring = _re.losses.anchoring.loss(cor=cor, spec=spec['anchoring'])
  loss_cistogram = _re.losses.cistogram.loss(cor=cor, spec=spec['cistogram'])
  loss_repelling = _re.losses.repelling.loss(cor=cor, spec=spec['repelling'])
  loss_rigidbody = _re.losses.rigidbody.loss(cor=cor, spec=spec['rigidbody'])
  loss = loss_anchoring + loss_cistogram + loss_repelling + loss_rigidbody
  aux = {'sum':       jax.lax.stop_gradient(loss          ),
         'anchoring': jax.lax.stop_gradient(loss_anchoring),
         'cistogram': jax.lax.stop_gradient(loss_cistogram),
         'repelling': jax.lax.stop_gradient(loss_repelling),
         'repelling_num_repellents': jnp.count_nonzero(spec['repelling']['weights_mask']),
         'rigidbody': jax.lax.stop_gradient(loss_rigidbody), }
  return loss, aux


class AFREGeometryLossSpecPreset(BasePreset):
  r"""The AFRE Geometry loss spec preset."""

  def __init__(self):
    r"""Create an AFRE Geometry loss spec preset."""
    self.anchoring = _re.losses.anchoring.preset()
    r"""The anchoring loss spec preset."""
    self.cistogram = _re.losses.cistogram.preset()
    r"""The cistogram loss spec preset."""
    self.repelling = _re.losses.repelling.preset()
    r"""The repelling loss spec preset."""
    self.rigidbody = _re.losses.rigidbody.preset()
    r"""The rigidbody loss spec preset."""

  def instantiate(self,
                  batch:   _u.af.TAFFeatures,
                  results: _u.af.TAFResults,
                  anchoring_indexers: _u.selection.AtomIndexerGroup,
                  cistogram_indexers: _u.selection.AtomIndexerGroup,
                  repelling_indexers: _u.selection.AtomIndexerGroup,
                  rigidbody_indexers: _u.selection.AtomIndexerGroup,
                  ) -> dict[str, dict[str, typing.Union[float, jnp.ndarray]]]:
    r"""Create an AFRE Geometry loss spec from this preset.

      Args:
        batch              (TAFFeatures):      The AF batch features.
        results            (TAFResults):       The AF output results.
        anchoring_indexers (AtomIndexerGroup): The anchoring atom indexer group.
        cistogram_indexers (AtomIndexerGroup): The cistogram atom indexer group.
        repelling_indexers (AtomIndexerGroup): The repelling atom indexer group.
        rigidbody_indexers (AtomIndexerGroup): The rigidbody atom indexer group.

      Returns:
        afre_geometry_loss_spec (dict[str, dict[str, Union[float, jnp.ndarray]]]):
          The AFRE Geometry loss spec with keys:
          'anchoring' (dict[str, Union[float, jnp.ndarray]]): The anchoring loss specs with keys:
            `cent_masks`   (jnp.ndarray): The masks for centroid divisions (Ncent, Nres*37).
            `dist_ref`     (jnp.ndarray): The reference centroid distances (Ncent, Ncent).
            `thresh_lower` (jnp.ndarray): The lower bound of the tolerated distance differences.
            `thresh_upper` (jnp.ndarray): The upper bound of the tolerated distance differences.
            `switch_delta` (float):       The delta param of the Huber distance differences loss.
            `mask`         (jnp.ndarray): The loss mask (Ncent, Ncent).
            `weight`       (float):       The loss weight.
            `dropout_p`    (float):       The loss dropout rate.
          'cistogram' (dict[str, Union[float, jnp.ndarray]]): The cistogram loss specs with keys:
            `cent_masks`   (jnp.ndarray): The masks for centroid divisions (Ncent, Nres*37).
            `prob_ref`     (jnp.ndarray): The reference centroid distance density (Ncent^2, Nbin).
            `gauss_breaks` (jnp.ndarray): The Gaussian KDE histogram bin edges (Nbin-1, ).
            `gauss_rwidth` (float):       The Gaussian KDE reciprocal smearing width.
            `mask`         (jnp.ndarray): The loss mask (Ncent, Ncent).
            `weight`       (float):       The loss weight.
            `dropout_p`    (float):       The loss dropout rate.
          'repelling' (dict[str, Union[float, jnp.ndarray]]): The repelling loss specs with keys:
            `cent_masks`   (jnp.ndarray): The masks for centroid divisions (Ncent, Nres*37).
            `dist_refs`    (jnp.ndarray): The reference centroid distances (Nrep, Ncent, Ncent).
            `gauss_rwidth` (float):       The repulsive Gaussian reciprocal width.
            `diver_weight` (jnp.ndarray): The diversity weights per distances (Ncent, Ncent).
            `mask`         (jnp.ndarray): The loss mask (Ncent, Ncent).
            `weights`      (jnp.ndarray): The loss weights      per repellent (Nrep, ).
            `weights_mask` (jnp.ndarray): The loss weights mask per repellent (Nrep, ).
            `dropout_p`    (float):       The loss dropout rate.
          'rigidbody' (dict[str, Union[float, jnp.ndarray]]): The rigidbody loss specs with keys:
            `masks`     (jnp.ndarray): The masks for segment divisions (Nseg, Nres*37).
            `ref`       (jnp.ndarray): The reference atom coordinates (Nres*37, 3).
            `weight`    (float):       The loss weight.
            `dropout_p` (float):       The loss dropout rate.
    """
    # unpack.
    ref = results['structure_module']['final_atom_positions'].reshape(-1, 3) # (Nres*37, 3)
    # anchoring loss spec.
    anchoring_cent_masks = anchoring_indexers.af_onehots(batch=batch, is_multimer=False)
    cistogram_cent_masks = cistogram_indexers.af_onehots(batch=batch, is_multimer=False)
    repelling_cent_masks = repelling_indexers.af_onehots(batch=batch, is_multimer=False)
    rigidbody_masks      = rigidbody_indexers.af_onehots(batch=batch, is_multimer=False)
    anchoring_spec = self.anchoring.instantiate(ref=ref, cent_masks=anchoring_cent_masks)
    cistogram_spec = self.cistogram.instantiate(ref=ref, cent_masks=cistogram_cent_masks)
    repelling_spec = self.repelling.instantiate(ref=ref, cent_masks=repelling_cent_masks)
    rigidbody_spec = self.rigidbody.instantiate(ref=ref,      masks=rigidbody_masks     )
    return {'anchoring': anchoring_spec,
            'cistogram': cistogram_spec,
            'repelling': repelling_spec,
            'rigidbody': rigidbody_spec, }