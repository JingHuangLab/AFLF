r"""AFTools utilities."""
# Authors: Zilin Song.


import os


DIR_HIERARCHY: dict[str, str] = {
  r'base_runner': None,
  r'chkp.infer_for_checkpoint':    os.path.join('checkpoint',          'infer_for_checkpoint'),
  r'chkp.infer_from_checkpoint':   os.path.join('checkpoint',          'infer_from_checkpoint'),
  r'chkp_m.infer_for_checkpoint':  os.path.join('checkpoint_multimer', 'infer_for_checkpoint'),
  r'chkp_m.infer_from_checkpoint': os.path.join('checkpoint_multimer', 'infer_from_checkpoint'),
  r'afre.runner': os.path.join('afre', ), }
r"""Mapping from runner names to runtime subdirectory paths."""


from aftools.utils import serialization
from aftools.utils import af
from aftools.utils import mm
from aftools.utils import io         # must be imported after af and mm.
from aftools.utils import selection  # must be imported after af and mm.
