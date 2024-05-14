from ._version import get_versions

from pylammpsmpi.wrapper.extended import LammpsLibrary
from pylammpsmpi.wrapper.concurrent import LammpsConcurrent
from pylammpsmpi.wrapper.base import LammpsBase


__all__ = [LammpsLibrary, LammpsConcurrent, LammpsBase]
__version__ = get_versions()["version"]


try:
    from pylammpsmpi.wrapper.ase import LammpsASELibrary
    __all__ += [LammpsASELibrary]
except ImportError:
    pass
