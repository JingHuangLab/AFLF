# Compute the RMSD (to first frame / to previous frame) from ablation.
# Authors: Zilin Song
# 

# Prevent BLAS oversubscription.
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import rms

sysnames = ['0_ubq', '1_adk', '2_tem1']

cfgnames = [
            '-plddt',
  # '-anchoring-plddt',
  # '-cistogram-plddt',
  # '-rigidbody-plddt',
  # 'plddt1only',
  # 'plddt5only',
  # 'plddt10only',
  '-repel_plddt1',
  '-repel_plddt5',
  '-repel_plddt10',
  # 'anchor_only',
  # 'repel_only',
  'full',
  ]

RMSD_SEL = "backbone"

dest = "/u/qianrt/backup/projects/aftools/6_ablation"


def load_traj(sysname: str, cfgname: str):
  assert sysname in sysnames
  assert cfgname in cfgnames
  subdir = f"{dest}/{sysname}/aftools_runtime/afre_{cfgname}"
  topo = f"{subdir}/init_results.pdb"
  traj = f"{subdir}/afre_traj.dcd"
  return mda.Universe(topo, traj, topology_format='pdb', format='dcd')


def rmsd_first_and_prev(u: mda.Universe):
  """Per-frame RMSD to frame 0 and to the previous frame (Angstrom)."""
  ag = u.select_atoms(RMSD_SEL)
  assert len(ag) > 0, f"selection {RMSD_SEL!r} is empty"

  u.trajectory[0]
  ref0 = ag.positions.copy()
  prev = ag.positions.copy()

  frames     = [0]
  rmsd_first = [0.0]
  rmsd_prev  = [0.0]

  for ts in u.trajectory[1:]:
    curr = ag.positions
    rmsd_first.append(rms.rmsd(curr, ref0, superposition=True))
    rmsd_prev.append(rms.rmsd(curr, prev, superposition=True))
    prev = curr.copy()
    frames.append(ts.frame)

  return (np.asarray(frames, dtype=int),
          np.asarray(rmsd_first),
          np.asarray(rmsd_prev))


def run_proc(sysname: str, cfgname: str):
  print(f"Processing: {sysname:<10}, {cfgname}", flush=True)
  u = load_traj(sysname=sysname, cfgname=cfgname)
  frames, rmsd_first, rmsd_prev = rmsd_first_and_prev(u=u)
  with open(f"./zcsvs/rmsd_{sysname}_{cfgname}.csv", 'w') as f:
    for fr, r0, rp in zip(frames, rmsd_first, rmsd_prev):
      f.write(f"{fr},{r0:.4f},{rp:.4f}\n")
  return (f"{sysname} {cfgname}: {len(frames)} frames, "
          f"rmsd_first max {rmsd_first.max():.3f} A, "
          f"rmsd_prev mean {rmsd_prev[1:].mean():.4f} A")


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