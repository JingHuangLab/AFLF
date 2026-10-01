r"""Helper script for setting up the AFRE atom groups and masks."""
# Authors: Runtong Qian.

from itertools import groupby

import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.dssp      import DSSP
from MDAnalysis.analysis.distances import distance_array
from MDAnalysis.lib.distances import self_capped_distance



def get_secondary_structure_segments(u: mda.Universe) -> list[tuple[str, list[int]]]:
  r"""Return a list of (ss, resid) tuples."""
  # 1. SS string by MDAnalysis DSSP with visual relabeling.
  print(f"SS string - Raw:\n{''.join(DSSP(u).run().results.dssp[0])}")
  #     ------------HHHHHHHHHHHHHHHH----HHHHHHHHHHHHHHHH-------HHHHH------HHHHHHHHHHHHH---HHHHHHHH-HHHHHHHHH-------HHHHHHHHH-----HHHHHH-HHH--HHHHHHHH-HHHH---EEHHHHHHHHHHHHHHHHHH----EEEEE-HHHH---EE--EEEEEE-----------HHHHHHHHHHHHH---E-EE-EE---EEEEEE------HHH----EEEEEEE--HHHHHHHHHHHH--HHHHHHHHHHHHH--EEE----EEEE-----E--E-----HHHHHHH-------HHH---
  ss = '------------HHHHHHHHHHHHHHHH----HHHHHHHHHHHHHHHH-------HHHHH------HHHHHHHHHHHHH---HHHHHHHH-HHHHHHHHH-------HHHHHHHHH-----HHHHHH-HHH--HHHHHHHH-HHHH---EEHHHHHHHHHHHHHHHHHH----EEEEE-HHHH---EE--EEEEEE-----------HHHHHHHHHHHHH---E-EE-EE---EEEEEE------HHH----EEEEEEE--HHHHHHHHHHHH--HHHHHHHHHHHHH--EEE----EEEE-----E--E-----HHHHHHH-------HHH---'
  print(f"SS string - Ref:\n{ss}")
  # 2. Split the SS str and the resids into segments.
  segs = list(''.join(list(_[1])) for _ in groupby(ss))
  resids = np.split(np.arange(len(ss)), indices_or_sections=np.cumsum([len(_) for _ in segs]))
  # 3. Combine the ss-flag and the resids for each segment.
  seg_resids = list()
  for (s, r) in list(zip(segs, resids)):
    if s[0] in ['E', 'H']:                  # helices and e-strands entirely form single segments.
      seg_resids.append((s[0], r.tolist()))
    if s[0] == '-' and len(r.tolist()) > 7: # loops longer than 7 takes the mid-7-residues as single segments.
      # if 62 in r.tolist() or 46 in r.tolist():                 #   except: LID domain entirely forms single segment.
      #   r_loop = r.tolist()
      # else:
      mid = len(r.tolist())//2
      r_loop = r.tolist()[mid-3:mid+4]
      seg_resids.append((s[0], r_loop))
      
  print(f"Segmentation Results - Number of segments: {len(seg_resids)}.")
  for (s, r) in seg_resids:
    print(f"{s} {r}")

  return seg_resids

def get_contacting_residue_pairs(u: mda.Universe, dist_thresh: float, select='all') -> list[tuple[int, int, int]]:
  r"""Return a list of (i, j) contacting residue pairs by inter-atomic distance thresholds."""
  atom_groups = [u.select_atoms(f"resid {_+1}:{_+2} and ({select})") for _ in range(len(u.residues))] # residues.
  mask = list()
  for i, i_ag in enumerate(atom_groups):
    for j, j_ag in enumerate(atom_groups):
      if (distance_array(i_ag, j_ag) <= dist_thresh).any():
        mask.append((i, j)) # contact pairs.
  # mask.append((1, 143))
  return mask

def get_segment_contact_flags(segments: list[tuple[str, list[int]]],
                              contacts: list[tuple[int, int]],
                              beta_sheets: list[list[int]],
                              ) -> list[bool]:
  r"""Return whether each segment has any non-neighbor, inter-block contact."""
  resid_to_seg_idx = {r: seg_idx
                      for seg_idx, (_, resids) in enumerate(segments)
                      for r in resids}
  seg_to_block_idx = {}
  for block_idx, beta_sheet in enumerate(beta_sheets):
    for seg_idx in beta_sheet:
      seg_to_block_idx[seg_idx] = block_idx
  for seg_idx, (s, _) in enumerate(segments):
    if s in ['H', '-']:
      seg_to_block_idx[seg_idx] = len(seg_to_block_idx)

  segment_has_contact = [False for _ in segments]
  for (i, j) in contacts:
    i_seg_idx = resid_to_seg_idx.get(i, None)
    j_seg_idx = resid_to_seg_idx.get(j, None)
    if i_seg_idx is None or j_seg_idx is None or abs(i_seg_idx-j_seg_idx) <= 1:
      continue
    if seg_to_block_idx.get(i_seg_idx, None) == seg_to_block_idx.get(j_seg_idx, None):
      continue
    segment_has_contact[i_seg_idx] = True
    segment_has_contact[j_seg_idx] = True

  print(f"Segment contact flags:") 
  print(f"- {segment_has_contact}")
  return segment_has_contact

def group_beta_sheets(u: mda.Universe,
                      segments: list[tuple[str, list[int]]],
                      min_ladder_len: int = 2,
                      min_ladder_ratio: float = 0.5,
                      hbond_dist_thresh: float = 3.5,
                      ) -> list[list[int]]:
  r"""Group beta strands into sheets by residue contacts.

  Returns grouped indexes of `segments`, e.g. [[0, 3, 4], [6, 7, 10]].
  """
  e_seg_idxs = [i for i, (s, _) in enumerate(segments) if s=='E']
  if len(e_seg_idxs) == 0:
    return []

  # Pre-index backbone N/O atoms. Avoid repeated selection-string parsing in
  # the strand-pair loops below.
  no_atoms_by_residue = []
  for residue in u.residues:
    atoms = {}
    for atom in residue.atoms:
      if atom.name in ['N', 'O'] and atom.name not in atoms:
        atoms[atom.name] = atom
    no_atoms_by_residue.append(atoms)

  graph = {i: set() for i in e_seg_idxs}
  for i, i_seg in enumerate(e_seg_idxs):
    for j_seg in e_seg_idxs[i+1:]:
      contact_directions = {}
      for i_pos, i_resid in enumerate(segments[i_seg][1]):
        for j_pos, j_resid in enumerate(segments[j_seg][1]):
          i_atoms = no_atoms_by_residue[i_resid]
          j_atoms = no_atoms_by_residue[j_resid]
          directions = set()
          if 'O' in i_atoms and 'N' in j_atoms:
            if np.linalg.norm(i_atoms['O'].position - j_atoms['N'].position) <= hbond_dist_thresh:
              directions.add('iO_jN')
          if 'N' in i_atoms and 'O' in j_atoms:
            if np.linalg.norm(i_atoms['N'].position - j_atoms['O'].position) <= hbond_dist_thresh:
              directions.add('iN_jO')
          if directions:
            contact_directions[(i_pos, j_pos)] = directions

      if len(contact_directions)==0:
        continue
      contact_positions = set(contact_directions)

      ladder_len = 0
      for direction in [1, -1]:
        for start in contact_positions:
          if any((start[0]-step, start[1]-direction*step) in contact_positions
                 for step in [1, 2]):
            continue
          length, pos = 1, start
          while True:
            next_positions = [(pos[0]+step, pos[1]+direction*step) for step in [1, 2]]
            hits = [next_pos for next_pos in next_positions if next_pos in contact_positions]
            if len(hits)==0:
              break
            length += 1
            pos = hits[0]
          ladder_len = max(ladder_len, length)

      ladder_ratio = ladder_len / min(len(segments[i_seg][1]), len(segments[j_seg][1]))

      has_bridge = False
      for (i_pos, j_pos), directions in contact_directions.items():
        if 'iO_jN' in directions and 'iN_jO' in directions:
          has_bridge = True
          break
        for next_pos in [(i_pos+1, j_pos), (i_pos+2, j_pos),
                         (i_pos, j_pos+1), (i_pos, j_pos+2)]:
          next_directions = contact_directions.get(next_pos, set())
          if 'iO_jN' in directions and 'iN_jO' in next_directions:
            has_bridge = True
          if 'iN_jO' in directions and 'iO_jN' in next_directions:
            has_bridge = True
        if has_bridge:
          break

      if ladder_len >= min_ladder_len or ladder_ratio >= min_ladder_ratio or has_bridge:
        graph[i_seg].add(j_seg)
        graph[j_seg].add(i_seg)

  beta_sheet_seg_indices, visited = [], set()
  for root in e_seg_idxs:
    if root in visited:
      continue
    group, stack = [], [root]
    visited.add(root)
    while stack:
      i = stack.pop()
      group.append(i)
      for j in sorted(graph[i]):
        if j not in visited:
          visited.add(j)
          stack.append(j)
    beta_sheet_seg_indices.append(sorted(group))

  print(f"Beta sheet segment:")
  for group in beta_sheet_seg_indices:
    print(f'- {group}')
  
  return beta_sheet_seg_indices


def get_proline_res_idxs(u: mda.Universe) -> list[int]:
  r"""Return a list of proline residue indices."""
  prolines = u.select_atoms("resname PRO").residues
  print(f"Prolines: {list(prolines.resindices)}")
  return prolines.resindices.tolist()

def get_ssbonds(u: mda.Universe, max_dist_thresh: float) -> list[tuple[int, int]]:
  sg_atoms = u.select_atoms("resname CYS and name SG")
  if len(sg_atoms) < 2:
    return []
  
  pairs = self_capped_distance(
    sg_atoms.positions,
    max_cutoff=max_dist_thresh,
    min_cutoff=1.0,
  )
  if len(pairs[0]) == 0:
    return []
  
  res_pairs = []
  for (i, j) in pairs[0]:
    res_pairs.append((int(sg_atoms[i].resindex), int(sg_atoms[j].resindex)))

  print(f"SSBonds: {res_pairs}")
  return res_pairs
  

if __name__ == '__main__':
  import sys
  system = sys.argv[1]
  version = sys.argv[2]
  dest = f"{system}/v{version}"

  pdb = f"{dest}/aftools_runtime/checkpoint/infer_from_checkpoint/latent_40_result.pdb"
  u = mda.Universe(pdb, format='pdb')
  segments = get_secondary_structure_segments(u=u)
  contacts = get_contacting_residue_pairs(u=u, dist_thresh=4.5)
  beta_sheet_seg_indices = group_beta_sheets(u=u, segments=segments)
  segment_has_contact = get_segment_contact_flags(segments=segments,
                                                  contacts=contacts,
                                                  beta_sheets=beta_sheet_seg_indices)
  prolines = get_proline_res_idxs(u=u)
  ss_bonds = get_ssbonds(u=u, max_dist_thresh=4.0)

  import pickle as pkl
  pkl.dump({'segments': segments, 'contacts': contacts, 'prolines': prolines, 'ss_bonds': ss_bonds, 'beta_sheets': beta_sheet_seg_indices, 'segment_has_contact': segment_has_contact
  }, open(f"{dest}/prep.pkl", 'wb'))