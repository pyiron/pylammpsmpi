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

try:
    from pylammpsmpi.wrapper.pyiron import lammps_file_interface_function

    __all__ += ["lammps_file_interface_function"]
except ImportError:
    pass
