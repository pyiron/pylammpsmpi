# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

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
    def version(self):
        """
        Get the version of lammps

        Parameters
        ----------
        None

        Returns
        -------
        version: string
            version string of lammps
        """
        return super().version.result()

    def file(self, inputfile):
        """
        Read script from an input file

        Parameters
        ----------
        inputfile: string
            name of inputfile

        Returns
        -------
        None
        """
        _ = super().file(inputfile=inputfile).result()

    # TODO
    def extract_setting(self, *args):
        return super().extract_setting(*args).result()

    def extract_global(self, name):
        """
        Extract value of global simulation parameters

        Parameters
        ----------
        name : string
            see notes for a set of possible options

        Notes
        -----
        The possible options for `name` are-
        "dt", "boxlo", "boxhi", "boxxlo", "boxxhi",
        "boxylo", "boxyhi", "boxzlo", "boxzhi", "periodicity",
        "xy", "xz", "yz", "natoms", "nbonds", "nangles",
        "ndihedrals", "nimpropers", "nlocal", "nghost",
        "nmax", "ntypes", "ntimestep", "units", "triclinic",
        "q_flag", "atime", "atimestep"

        Also global constants defined by units can be accessed-
        "boltz", "hplanck", "mvv2e", "ftm2v", "mv2d",
        "nktv2p", "qqr2e", "qe2f", "vxmu2f", "xxt2kmu",
        "dielectric", "qqr2e", "e_mass", "hhmrr2e",
        "mvh2r", "angstrom", "femtosecond", "qelectron"

        """
        return super().extract_global(name=name).result()

    def extract_box(self):
        """
        Get the simulation box

        Parameters
        ----------
        None

        Returns
        -------
        box : list
            of the form `[boxlo,boxhi,xy,yz,xz,periodicity,box_change]` where
            `boxlo` and `boxhi` are lower and upper bounds of the box in three dimensions,
            `xy, yz, xz` are the box tilts, `periodicity` is an array which shows if
            the box is periodic in three dimensions.
        """
        return super().extract_box().result()

    def extract_atom(self, name):
        """
        Extract a property of the atoms

        Parameters
        ----------
        name : {'x', 'mass', 'id', 'type', 'mask', 'v', 'f',
                'molecule', 'q', 'mu', 'omega', 'angmom', 'torque', 'radius'}
            the property of atom to be extracted

        Returns
        -------
        val : array of length n_atoms
            If the requested name has multiple dimensions, output
            will be a multi-dimensional array.

        Notes
        -----
        This method only gathers information from the current processor.
        Rest of the values would be zero.

        See Also
        --------
        scatter_atoms
        """
        return super().extract_atom(name=name).result()

    def extract_fix(self, *args):
        """
        Extract a fix value

        Parameters
        ----------
        id: string
            id of the fix

        style: {0, 1, 2}
            0 - global data
            1 - per-atom data
            2 - local data

        type: {0, 1, 2}
            0 - scalar
            1 - vector
            2 - array

        i: int, optional
            index to select fix output

        j: int, optional
            index to select fix output

        Returns
        -------
        value
            Fix data corresponding to the requested dimensions
        """
        return super().extract_fix(*args).result()

    def extract_variable(self, *args):
        """
        Extract the value of a variable

        Parameters
        ----------
        name: string
            name of the variable

        group: string
            group id (ignored for equal style variables)

        flag: {0, 1}
            0 - equal style variable
            1 - atom style variable

        Returns
        -------
        data
            value of variable depending on the requested dimension

        Notes
        -----
        Currently only returns the information provided on a single processor

        """
        return super().extract_variable(*args).result()

    @property
    def natoms(self):
        return self.get_natoms()

    def get_natoms(self):
        """
        Get the number of atoms

        Parameters
        ----------
        None

        Returns
        -------
        natoms : int
            number of atoms
        """
        return super().get_natoms().result()

    def set_variable(self, *args):
        """
        Set the value of a string style variable

        Parameters
        ----------
        name: string
            name of the variable

        value: string
            value of the variable

        Returns
        -------
        flag : int
            0 if successfull, -1 otherwise
        """
        return super().set_variable(*args).result()

    def reset_box(self, *args):
        """
        Reset the simulation box

        Parameters
        ----------
        boxlo: array of floats
            lower bound of box in three dimensions

        boxhi: array of floats
            upper bound of box in three dimensions

        xy, yz, xz : floats
            box tilts
        """
        _ = super().reset_box(*args).result()

    def generate_atoms(
        self, ids=None, type=None, x=None, v=None, image=None, shrinkexceed=False
    ):
        """
        Create atoms on all procs

        Parameters
        ----------
        ids : list of ints, optional
            ids of N atoms that need to be created
            if not specified, ids from 1 to N are assigned

        type : list of atom types, optional
            type of N atoms, if not specied, all atoms are assigned as type 1

        x: list of positions
            list of the type `[posx, posy, posz]` for N atoms

        v: list of velocities
            list of the type `[vx, vy, vz]` for N atoms

        image: list of ints, optional
            if not specified a list of 0s will be used.

        shrinkexceed: bool, optional
            default False

        Returns
        -------
        None

        """
        _ = (
            super()
            .generate_atoms(
                ids=ids, type=type, x=x, v=v, image=image, shrinkexceed=shrinkexceed
            )
            .result()
        )

    def create_atoms(self, n, id, type, x, v=None, image=None, shrinkexceed=False):
        """
        Create atoms on all procs

        Parameters
        ----------
        n : int
            number of atoms

        id : list of ints, optional
            ids of N atoms that need to be created
            if not specified, ids from 1 to N are assigned

        type : list of atom types, optional
            type of N atoms, if not specied, all atoms are assigned as type 1

        x: list of positions
            list of the type `[posx, posy, posz]` for N atoms

        v: list of velocities
            list of the type `[vx, vy, vz]` for N atoms

        image: list of ints, optional
            if not specified a list of 0s will be used.

        shrinkexceed: bool, optional
            default False

        Returns
        -------
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
    def has_exceptions(self):
        """Return whether the LAMMPS shared library was compiled with C++ exceptions handling enabled"""
        return super().has_exceptions.result()

    @property
    def has_gzip_support(self):
        return super().has_gzip_support.result()

    @property
    def has_png_support(self):
        return super().has_png_support.result()

    @property
    def has_jpeg_support(self):
        return super().has_jpeg_support.result()

    @property
    def has_ffmpeg_support(self):
        return super().has_ffmpeg_support.result()

    @property
    def installed_packages(self):
        return super().installed_packages.result()

    def set_fix_external_callback(self, *args):
        _ = super().set_fix_external_callback(*args).result()

    def get_neighlist(self, *args):
        """Returns an instance of :class:`NeighList` which wraps access to the neighbor list with the given index
        :param idx: index of neighbor list
        :type  idx: int
        :return: an instance of :class:`NeighList` wrapping access to neighbor list data
        :rtype:  NeighList
        """
        return super().get_neighlist(*args).result()

    def find_pair_neighlist(self, *args):
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
