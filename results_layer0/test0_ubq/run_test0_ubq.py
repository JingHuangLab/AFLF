r"""AFRE test for ubq."""
# Authors: Zilin Song.

import optax
import jax.numpy as jnp
jnp.set_printoptions(precision=6, threshold=jnp.inf)

import sys
sys.path.insert(0, "../../")
import aftools.utils as _u
import aftools.afre  as _re

dest = '.'

config = _u.af.AFConfig(model_name='model_1_ptm')
params = _u.af.AFParams(model_name='model_1_ptm', params_dir="/u/songzl/3.alphafoldtools/params")
checkpoint_pkl = _u.io.pkl.load(f'./{dest}/aftools_runtime/checkpoint/infer_for_checkpoint/checkpoint.pkl')

# Region selections. ===============================================================================
prep = _u.io.pkl.load(file=f'./{dest}/prep.pkl')
nres = checkpoint_pkl['batch']['aatype'].shape[0] # num. residues.

# rigidbody. ---------------------------------------------------------------------------------------
rigidbody_rids = [[], ]  # first one is for all 'E'.
for (s, r) in prep['segments']:
  if s=='H': rigidbody_rids   .append(r)
  if s=='E': rigidbody_rids[0].extend(r)
rigidbody_indexers0 = [_u.selection.AtomIndexerByAtomType(_, ['N', 'CA', 'C', 'O', 'CB']) for _ in rigidbody_rids]
rigidbody_indexers1 = [_u.selection.AtomIndexerByProlineRing(pro=_) for _ in [18, 36, 37]]
rigidbody_indexers2 = [_u.selection.AtomIndexerByPeptideBond(residue_index=_) for _ in range(nres-1)]
rigidbody_indexers  =  _u.selection.AtomIndexerGroup().extend(rigidbody_indexers0
                                                     ).extend(rigidbody_indexers1
                                                     ).extend(rigidbody_indexers2)

# cistogram, anchoring, and repelling. -------------------------------------------------------------
_mask = jnp.zeros((nres, nres))
for (i, j) in prep['contacts']:
  _mask = _mask.at[i, j].set(1 if abs(i-j)>4 else 0)                # remove 4-nn pairs.

contact_flag = (jnp.sum(_mask, axis=-1) != 0)                                     # flags only residues with contact.
contact_rids = jnp.arange(nres, dtype=int)[contact_flag].reshape(-1, 1).tolist()  # discards non-contacting residues
contact_mask = jnp.copy(_mask[contact_flag, :][:, contact_flag])                  # discards non-contacting residues masks

cistogram_mask = jnp.copy(contact_mask)
anchoring_mask = jnp.copy(contact_mask)
repelling_mask = jnp.copy(contact_mask)
cistogram_indexers = _u.selection.AtomIndexerGroup().extend([_u.selection.AtomIndexerByAtomType(_, _u.af.AFAtomTypes) for _ in contact_rids])
anchoring_indexers = _u.selection.AtomIndexerGroup().extend([_u.selection.AtomIndexerByAtomType(_, _u.af.AFAtomTypes) for _ in contact_rids])
repelling_indexers = _u.selection.AtomIndexerGroup().extend([_u.selection.AtomIndexerByAtomType(_, _u.af.AFAtomTypes) for _ in contact_rids])


# presets. =========================================================================================
preset = _re.runners.AFRERunnerPreset()

# loss weights.
preset.af_confidence_loss_spec.weight_plddt      = 0.1
preset.af_confidence_loss_spec.weight_ptm        = 0.

preset.afre_geometry_loss_spec.rigidbody.weight  = 0.5
preset.afre_geometry_loss_spec.cistogram.weight  = 3.0
preset.afre_geometry_loss_spec.anchoring.weight  = 1.0
preset.afre_geometry_loss_spec.repelling.weights = 0.05 * jnp.ones(10)

# loss dist_mask.
preset.afre_geometry_loss_spec.cistogram.mask = cistogram_mask
preset.afre_geometry_loss_spec.anchoring.mask = anchoring_mask
preset.afre_geometry_loss_spec.repelling.mask = repelling_mask

# loss dropouts.
preset.afre_geometry_loss_spec.rigidbody.dropout_p = 0.2
preset.afre_geometry_loss_spec.cistogram.dropout_p = 0.2
preset.afre_geometry_loss_spec.anchoring.dropout_p = 0.0
preset.afre_geometry_loss_spec.repelling.dropout_p = 0.5

# anchoring specs.
preset.afre_geometry_loss_spec.anchoring.thresh_lower = -3.*jnp.ones_like(anchoring_mask)
preset.afre_geometry_loss_spec.anchoring.thresh_upper =  3.*jnp.ones_like(anchoring_mask)

# cistogram breaks.
preset.afre_geometry_loss_spec.cistogram.gauss_breaks = jnp.linspace(2, 20, 10)
preset.afre_geometry_loss_spec.cistogram.gauss_rwidth = 4.

# repelling specs.
preset.afre_geometry_loss_spec.repelling.gauss_rwidth = 4.

# runtime.
preset.num_steps = 20000    # total number of steps, trajectories are save per step.
preset.save_freq = 1000     # saves the restart file.
preset.repelling_pushstack_freq = 10          # pushes a new repellent state.
preset.repelling_pushstack_pert = 0.1         # the distant perturbation to pushed repellent states.
preset.repelling_diversify_freq = 20          # updates the diversifying weights.
preset.repelling_diversify_rtau = 1.          # diversity reciprocal tempering.
preset.repelling_diversify_apply_to = 'dist'  # diversify per distance.

# optimizer.
preset.optax_optimizer = optax.adam(learning_rate=.002)

runner = preset.instantiate(project_dir=f'./{dest}')
runner.execute(config=config,
               params=params,
               apply_reprs='msa',
               use_bf16=True,
               checkpoint_pkl=checkpoint_pkl, 
               anchoring_indexers=anchoring_indexers,
               cistogram_indexers=cistogram_indexers,
               repelling_indexers=repelling_indexers,
               rigidbody_indexers=rigidbody_indexers, )