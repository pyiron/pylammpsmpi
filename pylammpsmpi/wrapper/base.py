# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from typing import List, Union

from pylammpsmpi.wrapper.concurrent import LammpsConcurrent

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


class LammpsBase(LammpsConcurrent):
    @property
    def version(self) -> str:
        """
        Get the version of lammps

        Returns:
        version: str
            version string of lammps
        """
        return super().version.result()

    def file(self, inputfile: str) -> None:
        """
        Read script from an input file

        Parameters:
        inputfile: str
            name of inputfile

        Returns:
        None
        """
        _ = super().file(inputfile=inputfile).result()

    # TODO
    def extract_setting(self, *args) -> Union[int, float, str]:
        """
        Extract a setting value

        Parameters:
        *args: tuple
            arguments to specify the setting to extract

        Returns:
        value: int, float, or str
            extracted setting value
        """
        return super().extract_setting(*args).result()

    def extract_global(self, name: str) -> Union[int, float, str]:
        """
        Extract value of global simulation parameters

        Parameters:
        name : str
            name of the global parameter to extract

        Returns:
        value: int, float, or str
            extracted value of the global parameter
        """
        return super().extract_global(name=name).result()

    def extract_box(self) -> List[Union[float, List[float], List[int]]]:
        """
        Get the simulation box

        Returns:
        box : list
            list containing the simulation box information
            [boxlo, boxhi, xy, yz, xz, periodicity, box_change]
            where boxlo and boxhi are lower and upper bounds of the box in three dimensions,
            xy, yz, xz are the box tilts, periodicity is an array which shows if
            the box is periodic in three dimensions, and box_change is a list of booleans
            indicating if the box dimensions have changed
        """
        return super().extract_box().result()

    def extract_atom(self, name: str) -> Union[List[int], List[float]]:
        """
        Extract a property of the atoms

        Parameters:
        name : str
            the property of atom to be extracted

        Returns:
        val : list of int or float
            If the requested name has multiple dimensions, output
            will be a multi-dimensional list.
        """
        return super().extract_atom(name=name).result()

    def extract_fix(self, *args) -> Union[int, float, List[Union[int, float]]]:
        """
        Extract a fix value

        Parameters:
        *args: tuple
            arguments to specify the fix value to extract

        Returns:
        value: int, float, or list of int or float
            extracted fix value corresponding to the requested dimensions
        """
        return super().extract_fix(*args).result()

    def extract_variable(self, *args) -> Union[int, float, List[Union[int, float]]]:
        """
        Extract the value of a variable

        Parameters:
        *args: tuple
            arguments to specify the variable to extract

        Returns:
        data: int, float, or list of int or float
            value of the variable depending on the requested dimension
        """
        return super().extract_variable(*args).result()

    @property
    def natoms(self) -> int:
        """
        Get the number of atoms

        Returns:
        natoms : int
            number of atoms
        """
        return self.get_natoms()

    def get_natoms(self) -> int:
        """
        Get the number of atoms

        Returns:
        natoms : int
            number of atoms
        """
        return super().get_natoms().result()

    def set_variable(self, *args) -> int:
        """
        Set the value of a string style variable

        Parameters:
        *args: tuple
            arguments to specify the variable and its value

        Returns:
        flag : int
            0 if successful, -1 otherwise
        """
        return super().set_variable(*args).result()

    def reset_box(self, *args) -> None:
        """
        Reset the simulation box

        Parameters:
        *args: tuple
            arguments to specify the new box dimensions

        Returns:
        None
        """
        _ = super().reset_box(*args).result()

    def generate_atoms(
        self,
        ids: List[int] = None,
        type: List[int] = None,
        x: List[float] = None,
        v: List[float] = None,
        image: List[int] = None,
        shrinkexceed: bool = False,
    ) -> None:
        """
        Create atoms on all procs

        Parameters:
        ids : list of ints, optional
            ids of N atoms that need to be created
            if not specified, ids from 1 to N are assigned

        type : list of atom types, optional
            type of N atoms, if not specified, all atoms are assigned as type 1

        x: list of positions
            list of the type `[posx, posy, posz]` for N atoms

        v: list of velocities
            list of the type `[vx, vy, vz]` for N atoms

        image: list of ints, optional
            if not specified a list of 0s will be used.

        shrinkexceed: bool, optional
            default False

        Returns:
        None
        """
        _ = (
            super()
            .generate_atoms(
                ids=ids, type=type, x=x, v=v, image=image, shrinkexceed=shrinkexceed
            )
            .result()
        )

    def create_atoms(
        self,
        n: int,
        id: List[int],
        type: List[int],
        x: List[float],
        v: List[float] = None,
        image: List[int] = None,
        shrinkexceed: bool = False,
    ) -> None:
        """
        Create atoms on all procs

        Parameters:
        n : int
            number of atoms

        id : list of ints, optional
            ids of N atoms that need to be created
            if not specified, ids from 1 to N are assigned

        type : list of atom types, optional
            type of N atoms, if not specified, all atoms are assigned as type 1

        x: list of positions
            list of the type `[posx, posy, posz]` for N atoms

        v: list of velocities
            list of the type `[vx, vy, vz]` for N atoms

        image: list of ints, optional
            if not specified a list of 0s will be used.

        shrinkexceed: bool, optional
            default False

        Returns:
        None
        """
        _ = (
            super()
            .create_atoms(
                n=n, id=id, type=type, x=x, v=v, image=image, shrinkexceed=shrinkexceed
            )
            .result()
        )

    @property
    def has_exceptions(self) -> bool:
        """Return whether the LAMMPS shared library was compiled with C++ exceptions handling enabled"""
        return super().has_exceptions.result()

    @property
    def has_gzip_support(self) -> bool:
        return super().has_gzip_support.result()

    @property
    def has_png_support(self) -> bool:
        return super().has_png_support.result()

    @property
    def has_jpeg_support(self) -> bool:
        return super().has_jpeg_support.result()

    @property
    def has_ffmpeg_support(self) -> bool:
        return super().has_ffmpeg_support.result()

    @property
    def installed_packages(self) -> List[str]:
        return super().installed_packages.result()

    def set_fix_external_callback(self, *args) -> None:
        _ = super().set_fix_external_callback(*args).result()

    def get_neighlist(self, *args):
        """Returns an instance of :class:`NeighList` which wraps access to the neighbor list with the given index
        :param idx: index of neighbor list
        :type  idx: int
        :return: an instance of :class:`NeighList` wrapping access to neighbor list data
        :rtype:  NeighList
        """
        return super().get_neighlist(*args).result()

    def find_pair_neighlist(self, style: str) -> int:
        """Find neighbor list index of pair style neighbor list
        Try finding pair instance that matches style. If exact is set, the pair must
        match style exactly. If exact is 0, style must only be contained. If pair is
        of style pair/hybrid, style is instead matched the nsub-th hybrid sub-style.
        Once the pair instance has been identified, multiple neighbor list requests
        may be found. Every neighbor list is uniquely identified by its request
        index. Thus, providing this request index ensures that the correct neighbor
        list index is returned.
        :param style: name of pair style that should be searched for
        :type  style: string
        :param exact: controls whether style should match exactly or only must be contained in pair style name, defaults to True
        :type  exact: bool, optional
        :param nsub:  match nsub-th hybrid sub-style, defaults to 0
        :type  nsub:  int, optional
        :param request:   index of neighbor list request, in case there are more than one, defaults to 0
        :type  request:   int, optional
        :return: neighbor list index if found, otherwise -1
        :rtype:  int
        """
        return super().find_pair_neighlist(*args).result()

    def find_fix_neighlist(self, *args):
        """Find neighbor list index of fix neighbor list
        :param fixid: name of fix
        :type  fixid: string
        :param request:   index of neighbor list request, in case there are more than one, defaults to 0
        :type  request:   int, optional
        :return: neighbor list index if found, otherwise -1
        :rtype:  int
        """
        return super().find_fix_neighlist(*args).result()

    def find_compute_neighlist(self, *args):
        """Find neighbor list index of compute neighbor list
        :param computeid: name of compute
        :type  computeid: string
        :param request:   index of neighbor list request, in case there are more than one, defaults to 0
        :type  request:   int, optional
        :return: neighbor list index if found, otherwise -1
        :rtype:  int
        """
        return super().find_compute_neighlist(*args).result()

    def get_neighlist_size(self, *args):
        """Return the number of elements in neighbor list with the given index
        :param idx: neighbor list index
        :type  idx: int
        :return: number of elements in neighbor list with index idx
        :rtype:  int
        """
        return super().get_neighlist_size(*args).result()

    def get_neighlist_element_neighbors(self, *args):
        return super().get_neighlist_element_neighbors(*args).result()

    def command(self, cmd):
        """
        Send a command to the lammps object

        Parameters
        ----------
        cmd : string, list of strings
            command to be sent

        Returns
        -------
        None
        """
        _ = super().command(cmd=cmd).result()

    def gather_atoms(self, *args, concat=False, ids=None):
        """
        Gather atom properties

        Parameters
        ----------
        name : {'x', 'mass', 'id', 'type', 'mask', 'v', 'f',
                'molecule', 'q', 'mu', 'omega', 'angmom', 'torque', 'radius'}
            the property of atom to be extracted

        concat : bool, optional. Default False
            If True, gather information from all processors,
            but not sorted according to Atom ids

        ids : list, optional. Default None
            If a list of ids are provided, the required information
            for only those atoms are returned

        Returns
        -------
        val : array of length n_atoms sorted by atom ids
            If the requested name has multiple dimensions, output
            will be a multi-dimensional array.

        Notes
        -----
        This method gathers information from all processors.

        See Also
        --------
        extract_atoms
        """
        return super().gather_atoms(*args, concat=concat, ids=ids).result()

    def scatter_atoms(self, *args, ids=None):
        """
        Scatter atoms for the lammps library

        Args:
            *args:
        """
        _ = super().scatter_atoms(*args, ids=ids).result()

    def get_thermo(self, *args):
        """
        Return current value of thermo keyword

        Parameters
        ----------
        name : string
            name of the thermo keyword

        Returns
        -------
        val
            value of the thermo keyword

        """
        return super().get_thermo(*args).result()

    # TODO
    def extract_compute(self, id, style, type, length=0, width=0):
        """
        Extract compute value from the lammps library

        Parameters
        ----------
        id : string
            id of the compute

        style: {0, 1}
            0 - global data
            1 - per atom data

        type: {0, 1, 2}
            0 - scalar
            1 - vector
            2 - array

        length: int, optional. Default 0
            if `style` is 0 and `type` is 1 or 2, then `length` is the length
            of vector.

        width: int, optional. Default 0
            if `type` is 2, then `width` is the number of elements in each
            element along length.

        Returns
        -------
        val
            data computed by the fix depending on the chosen inputs

        """
        return (
            super()
            .extract_compute(id=id, style=style, type=type, length=length, width=width)
            .result()
        )
