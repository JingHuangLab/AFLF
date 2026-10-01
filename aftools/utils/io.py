r"""AFTools utilities for I/O."""
# Authors: Zilin Song.


import abc
import pickle

import numpy as np

import aftools.utils as _u


# abstract classes.
class IOWriter(abc.ABC):
  r"""Abstract persistent file writer."""

  def __init__(self, name: str, file: str):
    r"""Create an abstract persistent file writer.

      Args:
        name (str): The writer name.
        file (str): The output file path.
    """
    self._name = name
    self._file = file

  @property
  def name(self) -> str:
    r"""Return the writer name."""
    return self._name

  @property
  def file(self) -> str:
    r"""Return the output file path."""
    return self._file

  @abc.abstractmethod
  def write(self, **kwargs) -> None:
    r"""Write data to the file."""
    raise NotImplementedError(r"Abstract method should be implemented by sub-classes.")

  @abc.abstractmethod
  def close(self) -> None:
    r"""Close the file and release resources."""
    raise NotImplementedError(r"Abstract method should be implemented by sub-classes.")


class IOWriterQueue(abc.ABC):
  r"""Abstract queue of persistent file writers."""

  def __init__(self):
    r"""Create an empty writer queue."""
    self.queue: list[IOWriter] = list()
    r"""The queued writers."""

  def close(self) -> None:
    r"""Close all writers in the queue."""
    for w in self.queue:
      w.close()
    del self


class IOStaticClass:
  r"""Static class that wraps filesystem operations."""

  def __init__(self):
    raise RuntimeError(r"This class is static and cannot be instantiated.")

  def __new__(self):
    raise RuntimeError(r"This class is static and cannot be instantiated.")


# dcd.
class DCDWriter(IOWriter):
  r"""Persistent DCD trajectory writer."""

  def __init__(self, file: str, top: _u.mm.TMMTop):
    r"""Create a persistent DCD writer.

      Args:
        file (str): The output DCD file path.
        top (TMMTop): The OpenMM topology.
    """
    super().__init__(name=file, file=file)
    self._dcd = _u.mm.mm_app.DCDFile(file=open(file=self.file, mode='wb'), topology=top, dt=.1)

  def write(self, cor: _u.mm.TMMCor) -> None:
    r"""Write a coordinate frame.

      Args:
        cor (TMMCor): The OpenMM coordinates.
    """
    self._dcd.writeModel(positions=cor)

  def close(self) -> None:
    r"""Close the underlying DCD file."""
    self._dcd._file.close()
    del self._dcd
    del self


class DCDWriterQueue(IOWriterQueue):
  r"""Queue of persistent DCD writers."""

  def __init__(self):
    r"""Create an empty DCD writer queue."""
    super().__init__()
    self.queue: list[DCDWriter]

  def write_mm(self, file: str, cor: _u.mm.TMMCor, top: _u.mm.TMMTop = None) -> None:
    r"""Write OpenMM coordinates to a DCD file, creating the writer on first use.

      Args:
        file (str): The output DCD file path.
        cor (TMMCor): The OpenMM coordinates.
        top (TMMTop): The OpenMM topology, required when creating a new writer.
    """
    if file not in [w.name for w in self.queue]:
      assert top is not None, f"Illegal `top` is None for new DCDWriter named `{file}`."
      self.queue.append(DCDWriter(file=file, top=top))
    for w in self.queue:
      if file == w.name: w.write(cor=cor); return
    raise RuntimeError(f"`{file}` not found in DCDWriterQueue: `{[w.name for w in self.queue]}`.")

  def write_af(self, file: str, batch: _u.af.TAFFeatures, results: _u.af.TAFResults) -> None:
    r"""Write an AlphaFold prediction as a DCD frame, creating the writer on first use.

      Args:
        file (str): The output DCD file path.
        batch (TAFFeatures): The AF batch features.
        results (TAFResults): The AF output results.
    """
    if file not in [w.name for w in self.queue]:
      pdb_string = _u.af.cast_af_prediction_to_pdb_string(batch=batch, results=results)
      pdb_struct = _u.mm.mm_pdbstructure.PdbStructure(input_stream=pdb_string.split('\n'))
      pdb_file   = _u.mm.mm_app.PDBFile(file=pdb_struct)
      self.queue.append(DCDWriter(file=file, top=pdb_file.getTopology()))
    for w in self.queue:
      if file == w.name:
        af_positions = np.asarray(results['structure_module']['final_atom_positions'])
        af_mask      = np.asarray(results['structure_module']['final_atom_mask'])
        w.write(cor=_u.mm.mm_unit.Quantity(af_positions[af_mask >= 0.5], _u.mm.mm_unit.angstrom))
        return
    raise RuntimeError(f"`{file}` not found in DCDWriterQueue: `{[w.name for w in self.queue]}`.")

# numpy.
class npy(IOStaticClass):
  r"""Static class for NumPy NPY file operations."""

  @staticmethod
  def dump(file: str, to_dump: np.ndarray) -> None:
    r"""Write a NumPy array to `file`.

      Args:
        file (str): The output NPY file path.
        to_dump (np.ndarray): The array to write.
    """
    assert isinstance(to_dump, np.ndarray), f"Illegal `to_dump` not an `np.ndarray`."
    np.save(file=file, arr=to_dump)

  @staticmethod
  def load(file: str) -> np.ndarray:
    r"""Load a NumPy array from `file`.

      Args:
        file (str): The input NPY file path.

      Returns:
        arr (np.ndarray): The loaded array.
    """
    return np.load(file=file)


# pdb.
class pdb(IOStaticClass):
  r"""Static class for PDB file operations."""

  @staticmethod
  def dump_af(file: str, results: _u.af.TAFResults, batch: _u.af.TAFFeatures) -> None:
    r"""Write an AlphaFold prediction as a PDB file.

      Args:
        file (str): The output PDB file path.
        results (TAFResults): The AF output results.
        batch (TAFFeatures): The AF batch features.
    """
    pdb_string = _u.af.cast_af_prediction_to_pdb_string(batch=batch, results=results)
    with open(file=file, mode='w') as str_file: str_file.write(pdb_string)

  @staticmethod
  def dump_mm(file: str, cor: _u.mm.TMMCor, top: _u.mm.TMMTop, ff_resnames: bool) -> None:
    r"""Write OpenMM coordinates as a PDB file.

      Args:
        file (str): The output PDB file path.
        cor (TMMCor): The OpenMM coordinates.
        top (TMMTop): The OpenMM topology.
        ff_resnames (bool): Whether to use force-field residue names from `top` in the output.
    """
    # PDBFile maintains a _residueNameReplacements dict to convert the std. PDB resnames from the FF
    # resnames, e.g., HSD -> HIS. The `ff_resnames=False` replaces the resname in the topology file
    # with the std. PDB resnames for the PDB output.
    _u.mm.mm_app.PDBFile._loadNameReplacementTables()
    if ff_resnames:
      _u.mm.mm_app.PDBFile._residueNameReplacements = {}
      _u.mm.mm_app.PDBFile.   _atomNameReplacements = {}
    with open(file=file, mode='w') as pdb_file:
      _u.mm.mm_app.PDBFile.writeModel(topology=top, positions=cor, file=pdb_file, keepIds=True)

  @staticmethod
  def load_mm(file: str) -> _u.mm.mm_app.PDBFile:
    r"""Load a PDB file.

      Args:
        file (str): The input PDB file path.

      Returns:
        pdb_file (PDBFile): The loaded OpenMM PDB file object.
    """
    return _u.mm.mm_app.PDBFile(file=file)


# pickle-able.
class pkl(IOStaticClass):
  r"""Static class for Python pickle file operations."""

  @staticmethod
  def dump(file: str, to_dump: object) -> None:
    r"""Pickle an object to `file`.

      Args:
        file (str): The output pickle file path.
        to_dump (object): The object to pickle.
    """
    with open(file=file, mode='wb') as pkl_file: pickle.dump(obj=to_dump, file=pkl_file)

  @staticmethod
  def load(file: str) -> object:
    r"""Load a pickled object from `file`.

      Args:
        file (str): The input pickle file path.

      Returns:
        obj (object): The unpickled object.
    """
    with open(file=file, mode='rb') as pkl_file: obj = pickle.load(pkl_file)
    return obj


# log.
class LOGWriter(IOWriter):
  r"""Persistent LOG file writer."""

  def __init__(self, file: str):
    r"""Create a persistent LOG writer.

      Args:
        file (str): The output LOG file path.
    """
    super().__init__(name=file, file=file)
    self._log = open(file=file, mode='w')

  def write(self, log: str) -> None:
    r"""Write a log line and flush.

      Args:
        log (str): The log content to write.
    """
    self._log.write(log)
    self._log.flush()

  def close(self) -> None:
    r"""Close the underlying LOG file."""
    self._log.close()
    del self


class LOGWriterQueue(IOWriterQueue):
  r"""Queue of persistent LOG writers."""

  def __init__(self):
    r"""Create an empty LOG writer queue."""
    super().__init__()
    self.queue: list[LOGWriter]

  def write(self, file: str, log: str) -> None:
    r"""Write log content to a file, creating the writer on first use.

      Args:
        file (str): The output LOG file path.
        log (str): The log content to write.
    """
    if file not in [w.name for w in self.queue]:
      self.queue.append(LOGWriter(file=file))
    for w in self.queue:
      if file == w.name: w.write(log=log); return
    raise RuntimeError(f"`{file}` not found in LOGWriterQueue: `{[w.name for w in self.queue]}`.")


class txt(IOStaticClass):
  r"""Static class for plain-text file operations."""

  @staticmethod
  def dump(file: str, to_dump: str) -> None:
    r"""Write a string to `file`.

      Args:
        file (str): The output text file path.
        to_dump (str): The string to write.
    """
    with open(file=file, mode='w') as f: f.write(to_dump)

  @staticmethod
  def load(file: str) -> str:
    r"""Load the contents of `file` as a string.

      Args:
        file (str): The input text file path.

      Returns:
        text (str): The file contents.
    """
    with open(file=file, mode='r') as f: lines = f.read()
    return lines


# General IO manager.
class IOManager:
  r"""Manager that bundles all AFTools filesystem operations."""

  def __init__(self):
    r"""Create an I/O manager."""
    self.dcd = DCDWriterQueue()
    self.npy = npy
    self.pdb = pdb
    self.pkl = pkl
    self.log = LOGWriterQueue()
    self.txt = txt

  def close(self):
    r"""Close all managed writers."""
    self.dcd.close()
    self.log.close()
