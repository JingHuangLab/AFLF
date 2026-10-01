r"""AFRE test for ndka."""
# Authors: Zilin Song.

import sys
sys.path.insert(0, "../../")
import optax
import jax.numpy as jnp
jnp.set_printoptions(precision=6, threshold=jnp.inf)

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
rigidbody_rids = []
for (s, r) in prep['segments']:
  rigidbody_rids.append(r)
rigidbody_indexers0 = [_u.selection.AtomIndexerByAtomType(_, ['N', 'CA', 'C', 'O', 'CB']) for _ in rigidbody_rids]
rigidbody_indexers1 = [_u.selection.AtomIndexerByProlineRing(pro=_) for _ in [10, 12, 21, 30, 31, 39, 43, 63, 85, 120, 155]]
rigidbody_indexers2 = [_u.selection.AtomIndexerByPeptideBond(residue_index=_) for _ in range(nres-1)]
rigidbody_indexers  =  _u.selection.AtomIndexerGroup().extend(rigidbody_indexers0
                                                     ).extend(rigidbody_indexers1
                                                     ).extend(rigidbody_indexers2)
# rigidbody_indexers.af_onehots(batch=checkpoint_pkl['batch'], is_multimer=False); exit()

# cistogram. ---------------------------------------------------------------------------------------
contact_resids = jnp.arange(nres, dtype=int).reshape(-1, 1).tolist()
contact_mask = jnp.zeros((nres, nres))

for i in range(nres):
  _mask = [1 if int(abs(_-i)>4) else 0 for _ in range(nres)]
  contact_mask = contact_mask.at[i].set(jnp.copy(_mask))

cistogram_mask = jnp.copy(contact_mask)
cistogram_indexers = _u.selection.AtomIndexerGroup().extend([_u.selection.AtomIndexerByAtomType(_, _u.af.AFAtomTypes) for _ in contact_resids])

# anchoring and repelling. -------------------------------------------------------------------------
# unpack domain selections.
contact_resids = []
for (s, r) in prep['segments']:
  if 168 in r: continue
  contact_resids.append(r)  # this seg has no interactions.

# make interaction masks.
contact_mask = jnp.zeros((len(contact_resids), len(contact_resids)))
for (i, j) in prep['contacts']:
  # Mask segment pairs to 0 if their are neighboring segments in the chain.
  _flag_not_neighbor = 0
  for   a_seg_idx, (_, a_resids) in enumerate(prep['segments']):
    for b_seg_idx, (_, b_resids) in enumerate(prep['segments']):
      if i in a_resids and j in b_resids:
        _flag_not_neighbor = int(abs(a_seg_idx - b_seg_idx) > 1)
  for a_idx, a_resids in enumerate(contact_resids):
    for b_idx, b_resids in enumerate(contact_resids):
      if i in a_resids and j in b_resids and a_idx != b_idx and _flag_not_neighbor:
        contact_mask = contact_mask.at[a_idx, b_idx].set(1)

anchoring_mask     = jnp.copy(contact_mask)
repelling_mask     = jnp.copy(contact_mask)
anchoring_indexers =  _u.selection.AtomIndexerGroup().extend([_u.selection.AtomIndexerByAtomType(_, ['N', 'CA', 'C', 'O', 'CB']) for _ in contact_resids])
repelling_indexers =  _u.selection.AtomIndexerGroup().extend([_u.selection.AtomIndexerByAtomType(_, ['N', 'CA', 'C', 'O', 'CB']) for _ in contact_resids])


# presets. =========================================================================================
preset = _re.runners.AFRERunnerPreset()

# loss weights.
preset.af_confidence_loss_spec.weight_plddt      = 0.1
preset.af_confidence_loss_spec.weight_ptm        = 0.

preset.afre_geometry_loss_spec.rigidbody.weight  = 0.5
preset.afre_geometry_loss_spec.cistogram.weight  = 0.5
preset.afre_geometry_loss_spec.anchoring.weight  = 3.0
preset.afre_geometry_loss_spec.repelling.weights = 0.1 * jnp.ones(40)

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
preset.afre_geometry_loss_spec.anchoring.thresh_lower = - 1.*jnp.ones_like(anchoring_mask)
preset.afre_geometry_loss_spec.anchoring.thresh_upper =  10.*jnp.ones_like(anchoring_mask)

# cistogram breaks.
preset.afre_geometry_loss_spec.cistogram.gauss_breaks = jnp.linspace(2, 64, 32)
preset.afre_geometry_loss_spec.cistogram.gauss_rwidth = 4.

# repelling specs.
preset.afre_geometry_loss_spec.repelling.gauss_rwidth = 4.

# runtime.
preset.num_steps = 10000    # total number of steps, trajectories are save per step.
preset.save_freq = 1000     # saves the restart file.
preset.repelling_pushstack_freq = 10          # pushes a new repellent state.
preset.repelling_pushstack_pert = 0.1         # the distant perturbation to pushed repellent states.
preset.repelling_diversify_freq = 20          # updates the diversifying weights.
preset.repelling_diversify_rtau = 10.         # diversity reciprocal tempering.
preset.repelling_diversify_apply_to = 'cent'  # diversify per centroid.

# optimizer.
preset.optax_optimizer = optax.adam(learning_rate=.003)

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