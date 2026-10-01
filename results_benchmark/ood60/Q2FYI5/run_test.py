r"""AFRE test for tem1."""
# Authors: Runtong Qian.


import optax
import jax.numpy as jnp
jnp.set_printoptions(precision=6, threshold=jnp.inf)

import aftools.utils as _u
import aftools.afre  as _re

from argparse import ArgumentParser
parser = ArgumentParser(description="Script for running aflf domain motion.")
parser.add_argument("version", type=int, help="version number")
args = parser.parse_args()

dest = f'./v{args.version}'

config = _u.af.AFConfig(model_name='model_1_ptm')
params = _u.af.AFParams(model_name='model_1_ptm', params_dir="/u/songzl/3.alphafoldtools/params")
checkpoint_pkl = _u.io.pkl.load(f'{dest}/aftools_runtime/checkpoint/infer_for_checkpoint/checkpoint.pkl')

# Region selections. ===============================================================================
prep = _u.io.pkl.load(file=f'{dest}/prep.pkl')
nres = checkpoint_pkl['batch']['aatype'].shape[0] # num. residues.
beta_sheet_resids = []
for beta_sheet_seg_idxs in prep['beta_sheets']:
  beta_contact_resids = []
  for seg_idx in beta_sheet_seg_idxs:
    beta_contact_resids.extend(prep['segments'][seg_idx][1])
  beta_sheet_resids.append(beta_contact_resids)

# domain selection
domain_resids = [list(range(27,41))+list(range(9,15))]
if args.version == 1:
  domain_resids = [list(range(28,42))+list(range(10,16))]
resid_domain = [-1] * nres
for i, rids in enumerate(domain_resids):
  for rid in rids:
    resid_domain[rid] = i  

# rigidbody. ---------------------------------------------------------------------------------------
rigidbody_rids = [list(_) for _ in beta_sheet_resids]  
for (s, r) in prep['segments']:
  if s=='H': rigidbody_rids   .append(r)
rigidbody_indexers0 = [_u.selection.AtomIndexerByAtomType(_, ['N', 'CA', 'C', 'O', 'CB']) for _ in rigidbody_rids]
rigidbody_indexers1 = [_u.selection.AtomIndexerByPeptideBond(residue_index=_) for _ in range(nres-1)]
rigidbody_indexers  =  _u.selection.AtomIndexerGroup().extend(rigidbody_indexers0
                                                     ).extend(rigidbody_indexers1
                                                     )
if prep['prolines']:
  rigidbody_indexers2 = [_u.selection.AtomIndexerByProlineRing(pro=int(_)) for _ in prep['prolines']]
  rigidbody_indexers.extend(rigidbody_indexers2)
if prep['ss_bonds']:
  rigidbody_indexers3 = [_u.selection.AtomIndexerByCystineBond(cys0=int(cys0), cys1=int(cys1)) for (cys0, cys1) in prep['ss_bonds']]
  rigidbody_indexers.extend(rigidbody_indexers3)


# cistogram. ---------------------------------------------------------------------------------------
contact_mask = jnp.zeros((nres, nres))

for i in range(nres):
  _mask = [1 if resid_domain[j] != -1 and resid_domain[j] == resid_domain[i] and int(abs(j-i)>4) else 0 for j in range(nres)]
  contact_mask = contact_mask.at[i].set(jnp.asarray(_mask))

cistogram_flag = jnp.sum(contact_mask, axis=-1) != 0
contact_resids = jnp.arange(nres, dtype=int)[cistogram_flag].reshape(-1, 1).tolist()
cistogram_mask = contact_mask[cistogram_flag, :][:, cistogram_flag]
cistogram_indexers = _u.selection.AtomIndexerGroup().extend([_u.selection.AtomIndexerByAtomType(_, _u.af.AFAtomTypes) for _ in contact_resids])

# anchoring and repelling. -------------------------------------------------------------------------
# unpack domain selections.
contact_resids = []
for beta_sheet_seg_idxs in prep['beta_sheets']:
  if any(prep['segment_has_contact'][_] for _ in beta_sheet_seg_idxs):
    beta_contact_resids = []
    for seg_idx in beta_sheet_seg_idxs:
      beta_contact_resids.extend(prep['segments'][seg_idx][1])
    contact_resids.append(beta_contact_resids)
for seg_idx, (s, r) in enumerate(prep['segments']):
  if s in ['H', '-'] and prep['segment_has_contact'][seg_idx]:
    contact_resids.append(r)


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

subdomain_resids = [list(range(27, 41)),
                    list(range(9, 15)),
                    list(range(27, 31)),
                    list(range(61, 69))]
if args.version == 1:
  subdomain_resids = [list(range(28, 42)),
                      list(range(10, 16)),
                      list(range(28, 32)),
                      list(range(62, 70))]

contact_resids.extend(subdomain_resids)
contact_mask = jnp.pad(contact_mask, ((0, len(subdomain_resids)), (0, len(subdomain_resids))))
contact_mask = contact_mask.at[[-1, -2, -3, -4], [-2, -1, -4, -3]].set(1)

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
preset.afre_geometry_loss_spec.anchoring.weight  = 1.0
preset.afre_geometry_loss_spec.repelling.weights = 0.05 * jnp.ones(50)

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

preset.afre_geometry_loss_spec.anchoring.thresh_lower =  preset.afre_geometry_loss_spec.anchoring.thresh_lower.at[[-1, -2], [-2, -1]].set(-8.)
preset.afre_geometry_loss_spec.anchoring.thresh_upper =  preset.afre_geometry_loss_spec.anchoring.thresh_upper.at[[-1, -2], [-2, -1]].set(1.)

preset.afre_geometry_loss_spec.anchoring.thresh_lower =  preset.afre_geometry_loss_spec.anchoring.thresh_lower.at[[-3, -4], [-4, -3]].set(-3.)
preset.afre_geometry_loss_spec.anchoring.thresh_upper =  preset.afre_geometry_loss_spec.anchoring.thresh_upper.at[[-3, -4], [-4, -3]].set(1.)

# cistogram breaks.
preset.afre_geometry_loss_spec.cistogram.gauss_breaks = jnp.linspace(2, 64, 32)
preset.afre_geometry_loss_spec.cistogram.gauss_rwidth = 4.

# repelling specs.
preset.afre_geometry_loss_spec.repelling.gauss_rwidth = 4.

# runtime.
preset.num_steps = 5000     # total number of steps, trajectories are save per step.
preset.save_freq = 10000     # saves the restart file.
preset.repelling_pushstack_freq = 10          # pushes a new repellent state.
preset.repelling_pushstack_pert = 0.1         # the distant perturbation to pushed repellent states.
preset.repelling_diversify_freq = 10          # updates the diversifying weights.
preset.repelling_diversify_rtau = 10.         # diversity reciprocal tempering.
preset.repelling_diversify_apply_to = 'cent'  # diversify per centroid.

# optimizer.
preset.optax_optimizer = optax.adam(learning_rate=.002)

for i in range(5):
  runner = preset.instantiate(project_dir=f'{dest}', runtime_tag=f'_domain_both{i}')
  runner.execute(config=config,
                params=params,
                apply_reprs='both',
                use_bf16=True,
                checkpoint_pkl=checkpoint_pkl,
                anchoring_indexers=anchoring_indexers,
                cistogram_indexers=cistogram_indexers,
                repelling_indexers=repelling_indexers,
                rigidbody_indexers=rigidbody_indexers, )