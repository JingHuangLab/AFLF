r"""AFTools utilities for OpenMM."""
# Authors: Zilin Song.


import simtk.openmm     as mm
import simtk.openmm.app as mm_app
import simtk.unit       as mm_unit
import simtk.openmm.app.internal.pdbstructure as mm_pdbstructure


TMMCor = mm_unit.Quantity
r"""OpenMM coordinate quantity type."""
TMMTop = mm_app.Topology
r"""OpenMM topology type."""
