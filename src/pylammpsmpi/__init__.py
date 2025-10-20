import pylammpsmpi._version
from pylammpsmpi.wrapper.base import LammpsBase
from pylammpsmpi.wrapper.concurrent import LammpsConcurrent, init_function
from pylammpsmpi.wrapper.extended import LammpsLibrary

__all__ = ["LammpsLibrary", "LammpsConcurrent", "LammpsBase", "init_function"]
__version__ = pylammpsmpi._version.__version__


try:
    from pylammpsmpi.wrapper.ase import LammpsASELibrary

    __all__ += ["LammpsASELibrary"]
except ImportError:
    pass
