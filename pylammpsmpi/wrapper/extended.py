# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from typing import Any, List, Optional

from pylammpsmpi.wrapper.base import LammpsConcurrent

__author__ = "Sarath Menon, Jan Janssen"
__copyright__ = (
    "Copyright 2020, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Jan Janssen"
__email__ = "janssen@mpie.de"
__status__ = "production"
__date__ = "Feb 28, 2020"


func_list = [
    "file",
    "extract_global",
    "extract_box",
    "extract_atom",
    "extract_fix",
    "extract_variable",
    "get_natoms",
    "set_variable",
    "reset_box",
    "generate_atoms",
    "set_fix_external_callback",
    "get_neighlist",
    "find_pair_neighlist",
    "find_fix_neighlist",
    "find_compute_neighlist",
    "get_neighlist_size",
    "get_neighlist_element_neighbors",
    "command",
    "gather_atoms",
    "scatter_atoms",
    "get_thermo",
    "extract_compute",
    "create_atoms",
]


prop_list = [
    "version",
    "natoms",
    "has_exceptions",
    "has_gzip_support",
    "has_png_support",
    "has_jpeg_support",
    "has_ffmpeg_support",
    "installed_packages",
]


command_list = [
    "angle_coeff",
    "angle_style",
    "atom_modify",
    "atom_style",
    "atom_style",
    "balance",
    "bond_coeff",
    "box",
    "bond_style",
    "boundary",
    "change_box",
    "clear",
    "comm_modify",
    "compute",
    "compute_modify",
    "create_atoms",
    "create_bonds",
    "create_box",
    "delete_atoms",
    "delete_bonds",
    "dielectric",
    "dihedral_coeff",
    "dihedral_style",
    "dimension",
    "displace_atoms",
    "dump",
    "dynamical_matrix",
    "fix",
    "fix_modify",
    "echo",
    "group2ndx",
    "ndx2group",
    "group",
    "hyper",
    "improper_coeff",
    "improper_style",
    "include",
    "info",
    "jump",
    "kim_init",
    "kim_interactions",
    "kim_query",
    "kim_param",
    "kspace_modify",
    "kspace_style",
    "label",
    "log",
    "message",
    "min_modify",
    "lattice",
    "mass",
    "minimize",
    "minimize/kk",
    "min_style",
    "molecule",
    "neb",
    "neb/spin",
    "next",
    "neighbor",
    "neigh_modify",
    "newton",
    "nthreads",
    "package",
    "pair_coeff",
    "pair_modify",
    "pair_style",
    "pair_write",
    "partition",
    "prd",
    "print",
    "python",
    "processors",
    "read",
    "read_data",
    "read_dump",
    "read_restart",
    "region",
    "replicate",
    "rerun",
    "reset_ids",
    "reset_timestep",
    "restart",
    "run",
    "run_style",
    "server",
    "set",
    "shell",
    "special_bonds",
    "suffix",
    "tad",
    "temper",
    "thermo",
    "thermo_modify",
    "thermo_style",
    "third_order",
    "timer",
    "timestep",
    "uncompute",
    "undump",
    "unfix",
    "units",
    "variable",
    "velocity",
    "write_coeff",
    "write_data",
    "write_dump",
    "write_restart",
]


thermo_list = [
    "step",
    "elapsed",
    "elaplong",
    "dt",
    "time",
    "cpu",
    "tpcpu",
    "spcpu",
    "cpuremain",
    "part",
    "timeremain",
    "atoms",
    "temp",
    "press",
    "pe",
    "ke",
    "etotal",
    "enthalpy",
    "evdwl",
    "ecoul",
    "epair",
    "ebond",
    "eangle",
    "edihed",
    "eimp",
    "emol",
    "elong",
    "etail",
    "vol",
    "density",
    "lx",
    "ly",
    "lz",
    "xlo",
    "xhi",
    "ylo",
    "yhi",
    "zlo",
    "zhi",
    "xy",
    "xz",
    "yz",
    "xlat",
    "ylat",
    "zlat",
    "bonds",
    "angles",
    "dihedrals",
    "impropers",
    "pxx",
    "pyy",
    "pzz",
    "pxy",
    "pxz",
    "pyz",
    "fmax",
    "fnorm",
    "nbuild",
    "ndanger",
    "cella",
    "cellb",
    "cellc",
    "cellalpha",
    "cellbeta",
    "cellgamma",
]


class LammpsLibrary:
    """
    Top level class which manages the lammps library provided by LammpsBase

    Args:
        cores (int): Number of CPU cores to use for Lammps simulation (default: 1)
        oversubscribe (bool): Whether to oversubscribe CPU cores (default: False)
        working_directory (str): Path to the working directory (default: ".")
        client: Client object for distributed computing (default: None)
        mode (str): Mode of operation (default: "local")
        cmdargs: Additional command line arguments for Lammps (default: None)
    """

    def __init__(
        self,
        cores: int = 1,
        oversubscribe: bool = False,
        working_directory: str = ".",
        client: Any = None,
        mode: str = "local",
        cmdargs: Optional[List[str]] = None,
    ) -> None:
        self.cores = cores
        self.working_directory = working_directory
        self.oversubscribe = oversubscribe
        self.client = client
        self.mode = mode
        self.lmp = LammpsConcurrent(
            cores=self.cores,
            oversubscribe=self.oversubscribe,
            working_directory=self.working_directory,
            cmdargs=cmdargs,
        )

    def __getattr__(self, name: str) -> Any:
        """
        Try to run input as a lammps command

        Args:
            name (str): Name of the lammps command

        Returns:
            Any: Result of the lammps command
        """
        if name in func_list:

            def func_wrapper(*args, **kwargs) -> Any:
                func = getattr(self.lmp, name)
                fut = func(*args, **kwargs)
                return fut.result()

            return func_wrapper

        elif name in thermo_list:
            fut = self.lmp.get_thermo(name)
            return fut.result()

        elif name in command_list:

            def command_wrapper(*args) -> Any:
                args = [name] + list(args)
                cmd = " ".join([str(x) for x in args])
                fut = self.lmp.command(cmd)
                return fut.result()

            return command_wrapper

        elif name in prop_list:
            fut = getattr(self.lmp, name)
            return fut.result()

        else:
            raise AttributeError(name)

    def close(self) -> None:
        """
        Close the Lammps simulation
        """
        self.lmp.close()

    def __dir__(self) -> List[str]:
        """
        Get the list of attributes and methods of the LammpsLibrary object

        Returns:
            List[str]: List of attributes and methods
        """
        return (
            list(super().__dir__()) + func_list + thermo_list + command_list + prop_list
        )
