r"""AFRE test for adk."""
# Authors: Runtong Qian.


import optax
import jax.numpy as jnp
jnp.set_printoptions(precision=6, threshold=jnp.inf)

import aftools.utils as _u
import aftools.afre  as _re

from argparse import ArgumentParser
parser = ArgumentParser(description="Script for running aflf ablation with configurable options.")
parser.add_argument("--disable_plddt", action="store_false", dest="enable_plddt", help="Disable pLDDT loss")
parser.add_argument("--disable_rigidbody", action="store_false", dest="enable_rigidbody", help="Disable rigidbody loss")
parser.add_argument("--disable_cistogram", action="store_false", dest="enable_cistogram", help="Disable cistogram loss")
parser.add_argument("--disable_anchoring", action="store_false", dest="enable_anchoring", help="Disable anchoring loss")
parser.add_argument("--disable_repelling", action="store_false", dest="enable_repelling", help="Disable repelling loss")
parser.add_argument("--weight_plddt", type=float, default=0.1, help="Weight for pLDDT loss (default: 0.1)")

parser.add_argument("--length_repel", type=int, default=50, help="Weight for pLDDT loss (default: 0.1)")
parser.add_argument("--weight_repel", type=float, default=0.1, help="Weight for pLDDT loss (default: 0.1)")
parser.add_argument("--bin_num_cisto", type=int, default=32, help="Weight for pLDDT loss (default: 0.1)")
parser.add_argument("--cisto_weight", type=float, default=0.5, help="Weight for pLDDT loss (default: 0.1)")

parser.add_argument("--anchor_weight", type=float, default=1.0, help="Weight for pLDDT loss (default: 0.1)")


parser.add_argument("--runtime_tag", type=str, required=True, help="Runtime tag for the script")
args = parser.parse_args()

dest = '.'

config = _u.af.AFConfig(model_name='model_1_ptm')
params = _u.af.AFParams(model_name='model_1_ptm', params_dir="/u/songzl/3.alphafoldtools/params")
checkpoint_pkl = _u.io.pkl.load(f'{dest}/aftools_runtime/checkpoint/infer_for_checkpoint/checkpoint.pkl')

# Region selections. ===============================================================================
prep = _u.io.pkl.load(file=f'{dest}/prep.pkl')
nres = checkpoint_pkl['batch']['aatype'].shape[0] # num. residues.

# rigidbody. ---------------------------------------------------------------------------------------
rigidbody_rids = [[], ]  # first one is for all 'E'.
for (s, r) in prep['segments']:
  if s=='H': rigidbody_rids   .append(r)
  if s=='E': rigidbody_rids[0].extend(r)
rigidbody_indexers0 = [_u.selection.AtomIndexerByAtomType(_, ['N', 'CA', 'C', 'O', 'CB']) for _ in rigidbody_rids]
rigidbody_indexers1 = [_u.selection.AtomIndexerByProlineRing(pro=_) for _ in [8, 26, 86, 90, 111, 127, 138, 139, 176, 200, ]]
rigidbody_indexers2 = [_u.selection.AtomIndexerByPeptideBond(residue_index=_) for _ in range(nres-1)]
rigidbody_indexers  =  _u.selection.AtomIndexerGroup().extend(rigidbody_indexers0
                                                     ).extend(rigidbody_indexers1
                                                     ).extend(rigidbody_indexers2)

# cistogram. ---------------------------------------------------------------------------------------
contact_resids = jnp.arange(nres, dtype=int).reshape(-1, 1).tolist()
contact_mask = jnp.zeros((nres, nres))
for i in range(nres):
  _same_domain_resids = None
  if   i in list(range( 30,  60)): _same_domain_resids = list(range( 30,  60))
  elif i in list(range(121, 160)): _same_domain_resids = list(range(121, 160))
  else:                            _same_domain_resids = [_ for _ in range(nres) if _ not in (list(range(30, 60))+list(range(121, 160)))]
  _mask = [1 if (_ in _same_domain_resids) and int(abs(_-i)>4) else 0 for _ in range(nres)]
  contact_mask = contact_mask.at[i].set(jnp.copy(_mask))

cistogram_mask = jnp.copy(contact_mask)
cistogram_indexers = _u.selection.AtomIndexerGroup().extend([_u.selection.AtomIndexerByAtomType(_, _u.af.AFAtomTypes) for _ in contact_resids])

# anchoring and repelling. -------------------------------------------------------------------------
# unpack domain selections.
contact_resids  = [[], ]  # first one is for all 'E'
contact_domains = ['', ]  # first one is for all 'E'
for (s, r) in prep['segments']:
  if s in ['H', '-']:
    contact_resids .append(r)
    contact_domains.append('NMP' if (35 in r or 48 in r) else 'LID' if (144 in r) else 'CORE')
  else:
    contact_resids [0].extend(r)
    contact_domains[0] += ('NMP' if (35 in r or 48 in r) else 'LID' if (144 in r) else 'CORE')
# check that all E-strands are 'CORE' domain.
contact_domains[0] = 'CORE' if contact_domains[0] == 'CORECORECORE' else None
assert contact_domains[0] == 'CORE' 

# make interaction masks.
contact_mask = jnp.zeros((len(contact_resids), len(contact_resids)))
lid_nmp_mask = jnp.zeros((len(contact_resids), len(contact_resids)))
for (i, j) in prep['contacts']:
  # Mask segment pairs to 0 if their are neighboring segments in the chain.
  _flag_not_neighbor = 0
  for   a_seg_idx, (_, a_resids) in enumerate(prep['segments']):
    for b_seg_idx, (_, b_resids) in enumerate(prep['segments']):
      if i in a_resids and j in b_resids:
        _flag_not_neighbor = int(abs(a_seg_idx-b_seg_idx) > 1)
  # Mask segment pairs to 0 if their are inter-domain pairs
  for   a_idx, (a_resids, a_domain) in enumerate(zip(contact_resids, contact_domains)):
    for b_idx, (b_resids, b_domain) in enumerate(zip(contact_resids, contact_domains)):
      if i in a_resids and j in b_resids and a_idx != b_idx:
        _flag_same_domain = int( a_domain==b_domain ) * _flag_not_neighbor
        _flag_pair_NMPLID = int((a_domain, b_domain) in [('NMP', 'LID'), ('LID', 'NMP')])
        # contact if seg is in contact with at least 1 E-seg, even if other E-segs are neighbors.
        if (_flag_same_domain + _flag_pair_NMPLID) == 1:
          contact_mask = contact_mask.at[a_idx, b_idx].set(1)
        # exclusively selects NMP - LID contacts (for relaxing anchoring thresholds).
        if _flag_pair_NMPLID == 1:
          lid_nmp_mask = lid_nmp_mask.at[a_idx, b_idx].set(1)

anchoring_mask = jnp.copy(contact_mask)
repelling_mask = jnp.copy(contact_mask)
anchoring_indexers = _u.selection.AtomIndexerGroup().extend([_u.selection.AtomIndexerByAtomType(_, _u.af.AFAtomTypes) for _ in contact_resids])
repelling_indexers = _u.selection.AtomIndexerGroup().extend([_u.selection.AtomIndexerByAtomType(_, _u.af.AFAtomTypes) for _ in contact_resids])


# presets. =========================================================================================
preset = _re.runners.AFRERunnerPreset()

# loss weights.
preset.af_confidence_loss_spec.weight_plddt      = args.weight_plddt * args.enable_plddt
preset.af_confidence_loss_spec.weight_ptm        = 0.

preset.afre_geometry_loss_spec.rigidbody.weight  = 0.5 * args.enable_rigidbody
preset.afre_geometry_loss_spec.cistogram.weight  = 0.5 * args.enable_cistogram
preset.afre_geometry_loss_spec.anchoring.weight  = 1.0 * args.enable_anchoring
preset.afre_geometry_loss_spec.repelling.weights = 0.1 * args.enable_repelling * jnp.ones(50)

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
preset.afre_geometry_loss_spec.anchoring.thresh_lower = -1.*jnp.ones_like(anchoring_mask)
preset.afre_geometry_loss_spec.anchoring.thresh_upper =  1.*jnp.ones_like(anchoring_mask) + 24 * lid_nmp_mask

# cistogram breaks.
# preset.afre_geometry_loss_spec.cistogram.gauss_breaks = jnp.linspace(1, 64, 64) # 1: sudden jumps.
preset.afre_geometry_loss_spec.cistogram.gauss_breaks = jnp.linspace(2, 64, 32)
preset.afre_geometry_loss_spec.cistogram.gauss_rwidth = 4.

# repelling specs.
preset.afre_geometry_loss_spec.repelling.gauss_rwidth = 4.

# runtime.
preset.num_steps = 10000     # total number of steps, trajectories are save per step.
preset.save_freq = 1000     # saves the restart file.
preset.repelling_pushstack_freq = 10          # pushes a new repellent state.
preset.repelling_pushstack_pert = 0.1         # the distant perturbation to pushed repellent states.
preset.repelling_diversify_freq = 10          # updates the diversifying weights.
preset.repelling_diversify_rtau = 0.          # diversity reciprocal tempering.
preset.repelling_diversify_apply_to = 'cent'  # diversify per centroid.

# optimizer.
preset.optax_optimizer = optax.adam(learning_rate=.002)

runner = preset.instantiate(project_dir=f'{dest}', runtime_tag=args.runtime_tag)
runner.execute(config=config,
               params=params,
               apply_reprs='msa',
               use_bf16=True,
               checkpoint_pkl=checkpoint_pkl,
               anchoring_indexers=anchoring_indexers,
               cistogram_indexers=cistogram_indexers,
               repelling_indexers=repelling_indexers,
               rigidbody_indexers=rigidbody_indexers, )
