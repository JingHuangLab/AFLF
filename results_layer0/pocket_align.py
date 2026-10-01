r"""Align the AFRE trajectory to the open state and take the closest frame to open/closed state."""
# Authors: Zilin Song.

import MDAnalysis as mda, numpy as np
from MDAnalysis.analysis import align, rms


def sanity_check(ref:  mda.Universe, select_ref:  str,
                 traj: mda.Universe, select_traj: str, ) -> None:
  atoms_ref  = ref .select_atoms(select_ref )
  atoms_traj = traj.select_atoms(select_traj)
  for _ in range(len(atoms_ref)):
    print(f"{atoms_ref [_].resname}{atoms_ref [_].resid:>3} {atoms_ref [_].name}",
          f"{atoms_traj[_].resname}{atoms_traj[_].resid:>3} {atoms_traj[_].name}",
          f" -- {atoms_ref [_].resname==atoms_traj[_].resname}")
  assert len(atoms_ref )==len(atoms_traj), f"{len(atoms_ref )} != {len(atoms_traj)}"
  print("#####################################", flush=True)


def dcd_rotr(ref: mda.Universe, traj: mda.Universe, select: dict[str, str], filename: str) -> None:
  r"""Output the aligned dcd files."""
  aligned = align.AlignTraj(traj, ref, select=select, filename=filename).run()


def dcd_rmsd(ref:  mda.Universe, select_ref:  str,
             traj: mda.Universe, select_traj: str, filename: str) -> int:
  r"""Output and return the indexes of the minimal RMSDs."""
  rmsd = np.asarray([rms.rmsd(a=ref .select_atoms(select_ref ).positions,
                              b=traj.select_atoms(select_traj).positions,
                              weights=None, center=True, superposition=True
                              ) for _ in traj.trajectory[::1]])
  with open(filename, 'w') as f:
    for _ in range(rmsd.shape[0]):
      f.write(f"{_+1:<5} {rmsd[_]}\n")
    f.write(f"min - {np.argmin(rmsd)+1} {rmsd[np.argmin(rmsd)]}")
  print(rmsd[np.argmin(rmsd)])
  return np.argmin(rmsd)


def pdb_extract(traj: mda.Universe, index: int, filename: str) -> None:
  with mda.Writer(filename) as w:
    traj.trajectory[index]
    w.write(traj.select_atoms("all"))


# dest            = "./test2_tem1/fpocket_h11"
# dest_ref_opened = f"{dest}/../1pzo.pdb"
# dest_ref_closed = f"{dest}/../1jwp.pdb"
# sel_rmsd_opened = "name CA and (resnum 212:229 or resnum 272:289)" # to get the most similar frame.
# sel_rmsd_closed = "name CA and (resnum 212:229 or resnum 272:289)" # to get the most similar frame.
# sel_rmsd_traj   = "name CA and (resnum 187:204 or resnum 245:262)" # to get the most similar frame.
# sel_align       = {'reference': "name CA",
#                    'mobile'   : "name CA", }

# dest            =  "./test3_bclxl/fpocket"
# dest_ref_opened = f"{dest}/../2yxj.pdb"
# dest_ref_closed = f"{dest}/../3fdl.pdb"
# sel_rmsd_opened = "name CA and (resnum  85:111 or resnum 119:130 or resnum 137:156 or resnum 187:192) and not altloc B"
# sel_rmsd_closed = "name CA and (resnum  85:111 or resnum 119:130 or resnum 137:156 or resnum 187:192) and not altloc B"
# sel_rmsd_traj   = "name CA and (resnum  34:60  or resnum  68:79  or resnum  86:105 or resnum 136:141)"
# sel_align = {'reference': "name CA and resnum 85:194 and not altloc B",
#              'mobile'   : "name CA and resnum 34:143", }

# dest            =  "./test4_blg/fpocket"
# dest_ref_opened = f"{dest}/../1gx8.pdb"
# dest_ref_closed = f"{dest}/../1bsq.pdb"
# sel_rmsd_opened = "name CA and (resnum  27:40  or resnum  83:91) and not altloc B"
# sel_rmsd_closed = "name CA and (resnum  27:40  or resnum  83:91) and not altloc B"
# sel_rmsd_traj   = "name CA and (resnum  27:40  or resnum  83:91)"
# sel_align = {'reference': "name CA and resnum 2:162 and not altloc A",
#              'mobile'   : "name CA and resnum 2:162",}

# dest            =  "./test5_ndka/fpocket"
# dest_ref_opened = f"{dest}/../2hvd.pdb"
# dest_ref_closed = f"{dest}/../3l7u.pdb"
# sel_rmsd_opened = "name CA and (resnum  52:60  or resnum  91:116) and not altloc B"
# sel_rmsd_closed = "name CA and (resnum  52:60  or resnum  91:116) and not altloc B"
# sel_rmsd_traj   = "name CA and (resnum  52:60  or resnum  91:116)"
# sel_align = {'reference': "name CA and resnum 6:162 and not altloc B",
#              'mobile'   : "name CA and resnum 6:162", }

# dest            =  "./test6_gltp/fpocket"
# dest_ref_opened = f"{dest}/../2eum.pdb"
# dest_ref_closed = f"{dest}/../1swx.pdb"
# sel_rmsd_opened = "name CA and (resnum  40:62  or resnum 142:154) and not altloc B"
# sel_rmsd_closed = "name CA and (resnum  40:62  or resnum 142:154) and not altloc B"
# sel_rmsd_traj   = "name CA and (resnum  40:62  or resnum 142:154)"
# sel_align = {'reference': "name CA and (resnum 8:167 or resnum 172:209) and not altloc B",
#              'mobile'   : "name CA and (resnum 8:167 or resnum 172:209)", }


# if __name__ == '__main__':
#   dest_top = f"{dest}/../aftools_runtime/afre/init_results.pdb"
#   dest_dcd = f"{dest}/../aftools_runtime/afre/afre_traj.dcd"
#   ref_opened = mda.Universe(dest_ref_opened)
#   ref_closed = mda.Universe(dest_ref_closed)
#   traj       = mda.Universe(dest_top, dest_dcd, topology_format='pdb')

#   sanity_check(ref=ref_opened, select_ref=sel_rmsd_opened, traj=traj, select_traj=sel_rmsd_traj)
#   sanity_check(ref=ref_closed, select_ref=sel_rmsd_closed, traj=traj, select_traj=sel_rmsd_traj)
#   sanity_check(ref=ref_opened, select_ref=sel_align["reference"], traj=traj, select_traj=sel_align["mobile"])
#   sanity_check(ref=ref_closed, select_ref=sel_align["reference"], traj=traj, select_traj=sel_align["mobile"])

#   dcd_rmsd(ref=ref_opened, select_ref=sel_align.get('reference'), traj=traj, select_traj=sel_align.get('mobile'), filename=f"{dest}/afre_rmsd_opened.log")
#   dcd_rmsd(ref=ref_closed, select_ref=sel_align.get('reference'), traj=traj, select_traj=sel_align.get('mobile'), filename=f"{dest}/afre_rmsd_closed.log")

#   f_opened = dcd_rmsd(ref=ref_opened, select_ref=sel_rmsd_opened, traj=traj, select_traj=sel_rmsd_traj, filename=f"{dest}/afre_rmsd_pocket_opened.log")
#   f_closed = dcd_rmsd(ref=ref_closed, select_ref=sel_rmsd_closed, traj=traj, select_traj=sel_rmsd_traj, filename=f"{dest}/afre_rmsd_pocket_closed.log")
#   f_closed=5188

#   dcd_rotr(ref=ref_opened, traj=traj, select=sel_align, filename=f"{dest}/afre_align.dcd")
#   pdb_extract(traj=mda.Universe(dest_top, f"{dest}/afre_align.dcd", topology_format='pdb'), index=f_opened, filename=f"{dest}/afre_align_opened_{f_opened+1}.pdb")
#   pdb_extract(traj=mda.Universe(dest_top, f"{dest}/afre_align.dcd", topology_format='pdb'), index=f_closed, filename=f"{dest}/afre_align_closed_{f_closed+1}.pdb")

if __name__ == '__main__':
  dest            = "./test2_tem1/fpocket_omega"
  dest_ref_closed = f"{dest}/../1jwp.pdb"
  sel_rmsd_closed = "name CA and (resnum 59:70 or resnum 272:289)" # to get the most similar frame.
  sel_rmsd_traj   = "name CA and (resnum 34:45 or resnum 245:262)" # to get the most similar frame.
  sel_align       = {'reference': "name CA",
                    'mobile'   : "name CA", }

  dest_top = f"{dest}/../aftools_runtime/afre/init_results.pdb"
  dest_dcd = f"{dest}/../aftools_runtime/afre/afre_traj.dcd"
  ref_closed = mda.Universe(dest_ref_closed)
  traj       = mda.Universe(dest_top, dest_dcd, topology_format='pdb')

  sanity_check(ref=ref_closed, select_ref=sel_rmsd_closed, traj=traj, select_traj=sel_rmsd_traj)
  sanity_check(ref=ref_closed, select_ref=sel_align["reference"], traj=traj, select_traj=sel_align["mobile"])

  dcd_rmsd(ref=ref_closed, select_ref=sel_align.get('reference'), traj=traj, select_traj=sel_align.get('mobile'), filename=f"{dest}/afre_rmsd_closed.log")

  f_closed = dcd_rmsd(ref=ref_closed, select_ref=sel_rmsd_closed, traj=traj, select_traj=sel_rmsd_traj, filename=f"{dest}/afre_rmsd_pocket_closed.log")
  f_closed=5188

  dcd_rotr(ref=ref_closed, traj=traj, select=sel_align, filename=f"{dest}/afre_align.dcd")
  pdb_extract(traj=mda.Universe(dest_top, f"{dest}/afre_align.dcd", topology_format='pdb'), index=f_closed, filename=f"{dest}/afre_align_closed_{f_closed+1}.pdb")