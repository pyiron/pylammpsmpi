from ._version import get_versions

from pylammpsmpi.wrapper.extended import LammpsLibrary
from pylammpsmpi.wrapper.concurrent import LammpsConcurrent
from pylammpsmpi.wrapper.base import LammpsBase

try:
    from pylammpsmpi.wrapper.ase import LammpsASELibrary
except ImportError:
    pass


__version__ = get_versions()["version"]
