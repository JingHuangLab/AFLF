r"""AlphaFoldTools utilities."""
# Authors: Zilin Song.


import os


# directories.
DIR_AFPARAMS = '/u/songzl/3.alphafoldtools/params'  # TODO: put to configurable.


DIR_HIERARCHY: dict[str, str] = {
  r'base_runner': None, 
  r'chkp.infer_for_checkpoint':    os.path.join('checkpoint',          'infer_for_checkpoint'), 
  r'chkp.infer_from_checkpoint':   os.path.join('checkpoint',          'infer_from_checkpoint'), 
  r'chkp_m.infer_for_checkpoint':  os.path.join('checkpoint_multimer', 'infer_for_checkpoint'), 
  r'chkp_m.infer_from_checkpoint': os.path.join('checkpoint_multimer', 'infer_from_checkpoint'), 
  r'afre.runner': os.path.join('afre', ), }
r"""The AFTools runner names to the runtime directory hierarchies."""