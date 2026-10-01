# Compute the B-factors from AFLF, AFSample2, and BioEmu.
# Authors: Rui Zhan, Zilin Song
# 

# Prevent BLAS oversubscription.
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import numpy as np
import scipy.stats as ss
import MDAnalysis as mda
from MDAnalysis.analysis import align, rms

dest = "/u/qianrt/backup/projects/aftools/5_benchmark/x_2release/1_structural_fluctuation"

res3to1 = {'Ala': 'A', 'Cys': 'C', 'Asp': 'D', 'Glu': 'E', 'Phe': 'F', 'Gly': 'G', 'His': 'H', 'Hsd': 'H', 'Hse': 'H',
           'Ile': 'I', 'Lys': 'K', 'Leu': 'L', 'Met': 'M', 'Asn': 'N', 'Pro': 'P', 'Gln': 'Q', 
           'Arg': 'R', 'Ser': 'S', 'Thr': 'T', 'Val': 'V', 'Trp': 'W', 'Tyr': 'Y', }
res3to1 = {k.upper(): v for (k, v) in res3to1.items()}

sysnames = {   "BPTI":   "5PTI.pdb",
               "P00644": "1EY0.pdb",
               "P00648": "1A2P.pdb",
               "P00698": "1AKI.pdb",
               "P01854": "1I1B.pdb",
               "P06654": "1PGB.pdb",
               "P61823": "1KF5.pdb",
               "P62937": "3K0N.pdb",
               "PDZ2":   "3LNX.pdb",
               "UBQ":    "1UBQ.pdb", 
            }

appnames = {"aftools":  ("afre_msa",   "init_results.pdb", "afre_traj.dcd"),
            "afsample2":("afsample2_", "topology.pdb",     "trajectory.xtc"), 
            "bioemu":   ("bioemu_",    "topology.pdb",     "samples.xtc"), 
            } # app: (subdir, top, traj)

runids = ['0', '1', '2', '3']


def load_traj(sysname: str, appname: str, runid: str) -> mda.Universe:
  r"""Load the trajectory"""
  assert sysname in sysnames.keys()
  assert appname in appnames.keys()
  assert runid in runids or runid=='xx'
  subdir, topo, traj = appnames[appname]
  subdir = f"{dest}/{sysname}/v0/{appname}_runtime/{subdir}"
  trajdirs = [f"{subdir}{_}/{traj}" for _ in runids if _ != runid]; print(trajdirs)
  u = mda.Universe( f"{subdir}0/{topo}", trajdirs, topology_format='pdb', format=traj[-3:])
  return u

def get_sim_bfactor(sysname: str, appname: str, runid: str) -> tuple[np.ndarray, str, np.ndarray]:
  c_alphas = 'protein and name CA'
  u = load_traj(sysname=sysname, appname=appname, runid=runid)
  resids = u.select_atoms(c_alphas).resids
  resnames = ''.join([res3to1[_] for _ in u.select_atoms(c_alphas).resnames])
  ref = align.AverageStructure(u, u, select=c_alphas, ref_frame=0).run().results.universe
  align.AlignTraj(u, ref, select=c_alphas, in_memory=True).run()
  rmsf = rms.RMSF(u.select_atoms(c_alphas)).run()
  bfactors = 8. * (np.pi**2) / 3. * (np.asarray(rmsf.results.rmsf)**2)
  return resids, resnames, bfactors, 

def get_exp_bfactor(sysname: str) -> tuple[np.ndarray, str, np.ndarray]:
  r"""Get the PDB b-factors."""
  assert sysname in sysnames.keys()
  resids, resnames,  bfactors = list(), list(), list()
  u = mda.Universe(f"{dest}/{sysname}/v0/{sysnames[sysname]}")
  for a in u.atoms:
    if a.name=='CA' and a.altLoc in ['', 'A']:
      resids.append(a.resid)
      resnames.append(res3to1[a.resname])
      bfactors.append(float(a.tempfactor))
  resids = np.asarray(resids)
  resnames = ''.join(resnames)
  bfactors = [float(f"{_:.2f}") for _ in bfactors]
  return resids, resnames, bfactors

def run_proc(sysname: str) -> list[str]:
  "Compute mean and std for each system."
  exp_resids, exp_resnames, exp_bfactors = get_exp_bfactor(sysname=sysname)
  tau_lines = []
  resids = {k: None for k in appnames.keys()} 
  means  = {k: None for k in appnames.keys()}
  stds   = {k: None for k in appnames.keys()}
  for a in appnames.keys():
    sim_resids, sim_resnames, sim_bfactors = get_sim_bfactor(sysname=sysname, appname=a, runid='xx')
    # sanity check (identical sequences.)
    assert exp_resnames == ''.join(np.asarray([_ for _ in sim_resnames])[exp_resids - 1])
    # compute tau.
    tau = ss.kendalltau(exp_bfactors, sim_bfactors[exp_resids-1])[0]
    tau_lines.append(f"{sysname},{a},{tau}\n")
    # compute resids.
    resids[a] = np.copy(sim_resids[exp_resids-1])
    # compute means.
    means[a] = np.copy(sim_bfactors[exp_resids-1])
    # compute stds.
    r_sim_bfactors = []
    for r in runids:
      sim_resids, sim_resnames, sim_bfactors = get_sim_bfactor(sysname=sysname, appname=a, runid=r)
      # sanity check (identical sequences.)
      assert exp_resnames == ''.join(np.asarray([_ for _ in sim_resnames])[exp_resids - 1])
      r_sim_bfactors.append(sim_bfactors[exp_resids-1])
    r_sim_bfactors = np.asarray(r_sim_bfactors) # (3, N)
    stds[a] = np.std(r_sim_bfactors, axis=0, ddof=1)
  
  # sanity.
  assert resids['aftools'].ndim==means['afsample2'].ndim==stds['bioemu'].ndim==1
  for a in appnames:
    assert resids['aftools'].shape[0]==resids[a].shape[0]
    assert means['aftools'].shape[0]==means[a].shape[0]
    assert stds['aftools'].shape[0]==stds[a].shape[0]
  with open(f"./zcsvs/{sysname}_bfactors.csv", 'w') as f:
    f.write("resid,exp,aftools_mean,aftools_std,afsample2_mean,afsample2_std,bioemu_mean,bioemu_std\n")
    for _ in range(resids['aftools'].shape[0]):
      f.write(f"{resids['aftools'][_]},{exp_bfactors[_]},"
              f"{means['aftools'][_]},{stds['aftools'][_]},"
              f"{means['afsample2'][_]},{stds['afsample2'][_]},"
              f"{means['bioemu'][_]},{stds['bioemu'][_]}\n")

  return tau_lines

from concurrent.futures import ProcessPoolExecutor, as_completed

import multiprocessing as mp

if __name__ == "__main__":

    ctx = mp.get_context("spawn")
    with open("./zcsvs/taus.csv", "w") as f, ProcessPoolExecutor(max_workers=4, mp_context=ctx) as ex:
        futures = {ex.submit(run_proc, s): s for s in sysnames.keys()}
        for fut in as_completed(futures):
            s = futures[fut]
            try:
                f.writelines(fut.result())
            except Exception as e:
                f.write(f"{s:<6} - FAILED: {e}\n")
            f.flush()