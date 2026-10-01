r"""AFTools base abstractions."""
# Authors: Zilin Song.


import os
import abc
import typing
import logging

import jax
import numpy as np

import aftools.utils as _u


class BasePreset(abc.ABC):
  r"""Abstract factory preset."""

  @abc.abstractmethod
  def instantiate(self, **kwargs) -> object:
    r"""Build and return the configured object."""
    raise NotImplementedError(r"Abstract method must be implemented by sub-classes.")


class BaseModel(abc.ABC):
  r"""Abstract AFTools model."""

  def __init__(self,
               config: _u.af.TAFConfig,
               params: _u.af.TAFParams,
               key: jax.random.KeyArray,
               multimer_mode: bool, ):
    r"""Constructor for the abstract Model class for AFTools.

      Args:
        config (TAFConfig): The AF configurations, `aftools.utils.af.AFConfig()`.
        params (TAFParams): The AF parameters    , `aftools.utils.af.AFParams()`.
        key (jax.random.KeyArray): The JAX PRNG key.
        multimer_mode (bool): If the model is for AF-Multimer.
    """
    _msg = 'AF-Multimer' if multimer_mode else 'AF' if multimer_mode is False else None
    assert _msg is not None, f"Illegal `multimer_mode={multimer_mode}`."
    assert config.model.global_config.multimer_mode == multimer_mode, f"Illegal non-{_msg} config."
    self.config = config
    self.params = params
    self.key = key

  @abc.abstractmethod
  def compile(self, **kwargs) -> None:
    r"""Compile the model for inference."""
    raise NotImplementedError(r"Abstract method must be implemented by sub-classes.")

  @abc.abstractmethod
  def predict(self, **kwargs) -> None:
    r"""Predict with the model."""
    raise NotImplementedError(r"Abstract method must be implemented by sub-classes.")


class BaseRunner(abc.ABC):
  r"""Abstract AFTools runner."""

  def __init__(self,
               project_dir: str = os.getcwd(),
               runtime_tag: str = r'', 
               runner_name: str = r'base_runner',):
    r"""Constructor of the abstract Runner class for AFTools.

      Args:
        project_dir (str): The project directory for the runner I/O.
        runtime_tag (str): The suffix tag to the runtime directory. 
        runner_name (str): The identity string for the Runner.
    """
    self._project_dir = os.path.abspath(project_dir)
    runtime_dir = _u.DIR_HIERARCHY.get(str(runner_name), None)
    assert runtime_dir is not None, f"Illegal `runner_name={runner_name}`."
    runtime_dir = f'{runtime_dir}{str(runtime_tag)}'
    self._working_dir = os.path.join(self.project_dir, 'aftools_runtime', runtime_dir)
    self._runner_name = f"aftools.{runner_name}"
    self._key = jax.random.PRNGKey(seed=np.random.randint(low=-99999, high=100000))

  @property
  def project_dir(self) -> str:
    r"""Return the project root directory."""
    return self._project_dir

  @property
  def working_dir(self) -> str:
    r"""Return the runtime working directory."""
    return self._working_dir

  @property
  def runner_name(self) -> str:
    r"""Return the runner identity string."""
    return self._runner_name

  @property
  def key(self) -> jax.random.KeyArray:
    r"""Return the JAX PRNG key."""
    return self._key

  @key.setter
  def key(self, val: jax.random.KeyArray) -> None:
    assert isinstance(val, jax.Array)
    assert val.shape == (2,)
    self._key = val

  @abc.abstractmethod
  def execute(self, **kwargs) -> typing.Union[None, typing.Any]:
    r"""Execute the Runner."""
    raise NotImplementedError(r"Abstract method must be implemented by sub-classes.")


class Runtime:
  r"""Runtime context manager for AFTools runners."""

  def __init__(self, whoami: str, working_dir: str) -> None:
    r"""Create a runtime context.

      Args:
        whoami      (str): The identity string for the Runner.
        working_dir (str): The directory for the Runner I/O.
    """
    assert isinstance(whoami, str), f"Illegal `whoami` type: {type(whoami)}."
    assert isinstance(working_dir, str), f"Illegal `working_dir` type: {type(working_dir)}."
    self._whoami = whoami
    self._logger = logging.getLogger(name=self._whoami)
    self._logger.setLevel('INFO')
    self._working_dir = working_dir
    self._io = _u.io.IOManager()

  @property
  def whoami(self) -> str:
    r"""Return the runner identity string."""
    return self._whoami

  @property
  def working_dir(self) -> str:
    r"""Return the working directory."""
    return self._working_dir

  @property
  def logger(self) -> logging.Logger:
    r"""Return the logger instance."""
    return self._logger

  @property
  def io(self) -> _u.io.IOManager:
    r"""Return the I/O manager."""
    return self._io

  def __enter__(self) -> "Runtime":
    self.mkdir(directory=self.working_dir)
    self.logger.info(f"Executing at working directory: {self.working_dir}.")
    self.logger.info("Executing kernel ...")
    return self

  def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
    self.io.close()
    if exc_type is not None:
      self.logger.info("Abnormal exit: Exception raised during Runtime !!!")
    else:
      self.logger.info("Normal exit: Done.")

  def mkdir(self, directory: str) -> None:
    r"""Create a directory if it does not exist."""
    if os.path.exists(directory):
      return
    os.makedirs(directory)

  def todir(self, *files: str) -> str:
    r"""Return a path inside the working directory."""
    return os.path.join(self.working_dir, *files)
