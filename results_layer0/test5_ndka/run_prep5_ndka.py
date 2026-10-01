r"""Helper script for setting up the AFRE atom groups and masks."""
# Authors: Zilin Song.

from itertools import groupby

import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.dssp      import DSSP
from MDAnalysis.analysis.distances import distance_array


def get_secondary_structure_segments(u: mda.Universe) -> list[tuple[str, list[int]]]:
  r"""Return a list of (ss, resid) tuples."""
  # 1. SS string by MDAnalysis DSSP with visual relabeling.
  print(f"SS string - Raw:\n{''.join(DSSP(u).run().results.dssp[0])}")
  #     ---EEEEEEEE-HHHHH---HHHHHHHHHHH--EE-EEEEE---HHHHHHH-HHH-----HHHHHHHH----EEEEEEEEE-HHHHHHHHH----HHH-----HHHHH---HHH--EEE---HHHHHHHHHHH--HHH--------HHHH--
  ss = '---EEEEEEEE-HHHHH---HHHHHHHHHHH--EEEEEEEE---HHHHHHH---------HHHHHHHH----EEEEEEEEE-HHHHHHHHH------------HHHHH--------EEE---HHHHHHHHHHH-------------------'
  print(f"SS string - Ref:\n{ss}")
  # 2. Split the SS str and the resids into segments.
  segs = list(''.join(list(_[1])) for _ in groupby(ss))
  resids = np.split(np.arange(len(ss)), indices_or_sections=np.cumsum([len(_) for _ in segs]))
  # 3. Combine the ss-flag and the resids for each segment.
  seg_resids = list()
  for (s, r) in list(zip(segs, resids)):
    if s[0] in ['E', 'H']:                    # helices and e-strands entirely form single segments.
      seg_resids.append((s[0], r.tolist()))
    if s[0] == '-' and len(r.tolist()) >= 7:  # loops longer than 7 takes the mid-7-residues as single segments.
      mid = len(r.tolist())//2
      seg_resids.append((s[0], r.tolist()[mid-3:mid+4]))
  print(f"Segmentation Results - Number of segments: {len(seg_resids)}.")
  for (s, r) in seg_resids:
    print(f"{s} {r}")
  return seg_resids

def get_contacting_residue_pairs(u: mda.Universe, dist_thresh: float) -> list[tuple[int, int, int]]:
  r"""Return a list of (i, j) contacting residue pairs by inter-atomic distance thresholds."""
  atom_groups = [u.select_atoms(f"resid {_+1}:{_+2}") for _ in range(len(u.residues))] # residues.
  mask = list()
  for i, i_ag in enumerate(atom_groups):
    for j, j_ag in enumerate(atom_groups):
      if (distance_array(i_ag, j_ag) <= dist_thresh).any():
        mask.append((i, j)) # contact pairs.
  return mask


if __name__ == '__main__':
  pdb = f"./aftools_runtime/checkpoint/infer_from_checkpoint/latent_0_result.pdb"
  u = mda.Universe(pdb, format='pdb')
  segments = get_secondary_structure_segments(u=u)
  contacts = get_contacting_residue_pairs(u=u, dist_thresh=4.5)
  import pickle as pkl
  pkl.dump({'segments': segments, 'contacts': contacts}, open(f"./prep.pkl", 'wb'))