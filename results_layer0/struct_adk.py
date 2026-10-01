"""Extract the structural features of AdK along the trajectory."""
# Authors: Zilin Song.

import numpy as np
import MDAnalysis as mda


def dist_AA(a, b) -> np.ndarray:
  return np.linalg.norm(a-b)

def angle_deg(a, v, b) -> np.ndarray:
  u1, u2 = a - v, b - v
  cos = np.dot(u1, u2) / (np.linalg.norm(u1) * np.linalg.norm(u2))
  return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


if __name__ == '__main__':

  u = mda.Universe('./test1_adk/aftools_runtime/afre/init_results.pdb',
                   './test1_adk/aftools_runtime/afre/afre_traj.dcd', 
                   topology_format='pdb', format='dcd', )

  dist_nmp_lid = {'nmp':      u.select_atoms(f"backbone and (resid 30:59)"),
                  'lid':      u.select_atoms(f"backbone and (resid 122:159)"), }
  angl_nmp_core = {'corelid': u.select_atoms(f"(backbone or name CB) and (resid 115:125)"),
                   'core':    u.select_atoms(f"(backbone or name CB) and (resid 90:100)"),
                   'nmp':     u.select_atoms(f"(backbone or name CB) and (resid 35:55)"), }
  angl_lid_core = {'core':    u.select_atoms(f"(backbone or name CB) and (resid 179:185)"),
                   'corelid': u.select_atoms(f"(backbone or name CB) and (resid 115:125)"),
                   'lid':     u.select_atoms(f"(backbone or name CB) and (resid 125:153)"), }
  
  dists, thetas = list(), list()
  for ts in u.trajectory:
    dists.append((ts.frame, dist_AA(a=dist_nmp_lid['nmp'].center_of_geometry(),
                                    b=dist_nmp_lid['lid'].center_of_geometry(), )))
    thetas.append((ts.frame, 
                   angle_deg(a=angl_nmp_core['corelid'].center_of_geometry(),
                             v=angl_nmp_core['core'   ].center_of_geometry(),
                             b=angl_nmp_core['nmp'    ].center_of_geometry(), ),
                   angle_deg(a=angl_lid_core['core'   ].center_of_geometry(),
                             v=angl_lid_core['corelid'].center_of_geometry(),
                             b=angl_lid_core['lid'    ].center_of_geometry(), )))

  with open(f"./zpdbs/struct_adk/adk_dist.txt", 'w') as f:
    f.write("# Time(ps)	Dist_LID_NMP(Angstrom)\n")
    for (i, d) in dists:
      f.write(f"{i:.3f}\t{d:.3f}\n")

  with open(f"./zpdbs/struct_adk/adk_angles.txt", 'w') as f:
    f.write("# Time(ps)	Angle_NMP(deg)	Angle_LID(deg)\n")
    for (i, nmp, lid) in thetas:
      f.write(f"{i:.3f}\t{nmp:.3f}\t{lid:.3f}\n")
  
  # extract frames.
  to_extract = {'open':   [2840, 7040], 
                'closed': [5560, 8900], 
                'angles_open':   [2848,], 
                'angles_closed': [9524,], }

  for state, frames in to_extract.items():
    for i in frames:
      u.trajectory[i]
      u.atoms.write(f'./zpdbs/struct_adk/adk_{state}_f{i}.pdb')