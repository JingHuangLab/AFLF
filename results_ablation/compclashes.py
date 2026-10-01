# Compute the positional clashes from ablation.
# Authors: Zilin Song
# 

# Prevent BLAS oversubscription.
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

from itertools import combinations

import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import capped_distance

sysnames = ['0_ubq', '1_adk', '2_tem1']

cfgnames = [
            '-plddt',
  '-anchoring-plddt',
  '-cistogram-plddt',
  '-rigidbody-plddt',
  # 'plddt1only',
  # 'plddt5only',
  # 'plddt10only',
  # '-repel_plddt1',
  # '-repel_plddt5',
  # '-repel_plddt10',
  # 'anchor_only',
  # 'repel_only',
  # 'full',
  ]
CLASH_CUTOFF = 2.0

dest = "/u/qianrt/backup/projects/aftools/6_ablation"

def load_traj(sysname: str, cfgname: str):
  assert sysname in sysnames
  assert cfgname in cfgnames
  subdir = f"{dest}/{sysname}/aftools_runtime/afre_{cfgname}"
  topo = f"{subdir}/init_results.pdb"
  traj = f"{subdir}/afre_traj.dcd"
  return mda.Universe(topo, traj, topology_format='pdb', format='dcd')

def make_excl(u: mda.Universe, heavy: mda.AtomGroup) -> set:
  """Excludes, at a 2 A cutoff, everything that is close by construction:
    1) all intra-residue pairs;
    2) all pairs between consecutive residues within a chain;
    3) disulfides: SG-SG pairs < SS_CUTOFF on the reference frame.
  """
  excl = set()
  # 1. intra-residue pairs.
  for res in heavy.residues:
    members = res.atoms.intersection(heavy).indices
    excl.update(combinations(sorted(members), 2))
  # 2. all pairs between consecutive residues.
  residues = heavy.residues
  for k in range(len(residues) - 1):
    r0, r1 = residues[k], residues[k + 1]
    if r0.segid != r1.segid:
      continue
    i0 = r0.atoms.intersection(heavy).indices
    i1 = r1.atoms.intersection(heavy).indices
    for i in i0:
      for j in i1:
        excl.add((min(i, j), max(i, j)))
  # 3. disulfides
  sg = heavy[heavy.names == 'SG'].indices
  excl.update(combinations(sorted(sg), 2))
  return excl

def count_clashes(u: mda.Universe, heavy: mda.AtomGroup, excl: set) -> tuple[np.ndarray, np.ndarray]:
  frames, counts = list(), list()

  for ts in u.trajectory:
    pairs, _ = capped_distance(heavy.positions, heavy.positions, max_cutoff=CLASH_CUTOFF)
    gi = heavy.indices[pairs[:, 0]]
    gj = heavy.indices[pairs[:, 1]]
    n = 0
    for i, j in zip(gi, gj):
      if i >= j:  # dedup.
        continue
      if (i, j) not in excl:
        n += 1
    frames.append(ts.frame)
    counts.append(n)
  return np.asarray(frames, dtype=int), np.asarray(counts, dtype=int)

def run_proc(sysname: str, cfgname: str):
  print(f"Processing: {sysname:<10}, {cfgname}", flush=True)
  u = load_traj(sysname=sysname, cfgname=cfgname)
  heavy = u.select_atoms("not (name H* or name 1H* or name 2H* or name 3H*)")
  frames, counts = count_clashes(u, heavy, make_excl(u, heavy))
  with open(f"./zcsvs/clashes_{sysname}_{cfgname}.csv", 'w') as f:
    for fr, s in zip(frames, counts):
      f.write(f"{fr},{s}\n")
  return f"{sysname} {cfgname}: {len(frames)} frames, max {counts.max()}"


from concurrent.futures import ProcessPoolExecutor

if __name__ == '__main__':
  os.makedirs('./zcsvs', exist_ok=True)

  n_workers = min(len(sysnames) * len(cfgnames), os.cpu_count() or 1)
  with ProcessPoolExecutor(max_workers=n_workers) as pool:
    results = pool.map(run_proc,
                       [s for s in sysnames for _ in cfgnames],
                       [c for _ in sysnames for c in cfgnames])
    for msg in results:
      print(msg, flush=True)