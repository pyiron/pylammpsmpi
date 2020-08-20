# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import os
import pickle
import subprocess


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


class LammpsBase:
    def __init__(self, cores=8, working_directory="."):
        self.cores = cores
        self.working_directory = working_directory
        self._process = None

    def start_process(self):
        executable = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "mpi", "lmpmpi.py"
        )
        self._process = subprocess.Popen(
            ["mpiexec", "--oversubscribe", "-n", str(self.cores), "python", executable],
            stdout=subprocess.PIPE,
            stderr=None,
            stdin=subprocess.PIPE,
            cwd=self.working_directory,
        )

    def _send(self, command, data=None):
        """
        Send a command to the Lammps Library executable

        Parameters
        ----------
        command : string
            command to be send to the

        data: optional, default None
            data to be sent to the command

        Returns
        -------
        None
        """
        pickle.dump({"c": command, "d": data}, self._process.stdin)
        self._process.stdin.flush()

    def _receive(self):
        """
        Receive data from the Lammps library

        Parameters
        ----------
        None

        Returns
        -------
        data : string
            data from the command
        """
        output = pickle.load(self._process.stdout)
        return output

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
        self._send(command="get_version", data=[])
        return self._receive()

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
        if not os.path.exists(inputfile):
            raise FileNotFoundError("Input file does not exist")
        self._send(command="get_file", data=[inputfile])
        _ = self._receive()

    #TODO
    def extract_setting(self, *args):
        self._send(command="extract_setting", data=list(args))
        return self._receive()

    def extract_global(self, name, type):
        """
        Extract value of global simulation parameters

        Parameters
        ----------
        name : string
            see notes for a set of possible options

        type : {0, 1}
            0 if output value is integer
            1 if output value is float

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
        self._send(command="extract_global", data=[name, type])
        return self._receive()

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
        self._send(command="extract_box", data=[])
        return self._receive()

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
        self._send(command="extract_atom", data=list([name]))
        return self._receive()

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

        self._send(command="extract_fix", data=list(args))
        return self._receive()

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
        self._send(command="extract_variable", data=list(args))
        return self._receive()

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

        self._send(command="get_natoms", data=[])
        return self._receive()

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
        self._send(command="set_variable", data=list(args))
        return self._receive()

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
        self._send(command="reset_box", data=list(args))
        _ = self._receive()

    def generate_atoms(self, ids=None, type=None, x=None, v=None, image=None, shrinkexceed=False):
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

        if x is not None:
            natoms = len(x)
            if type is None:
                type = [1]*natoms
            if ids is None:
                ids = list(range(1, natoms+1))
            if image is None:
                image = [0]*natoms

            funct_args = [natoms, ids, type, x, v, image, shrinkexceed]
            self._send(command="create_atoms", data=funct_args)
            _ = self._receive()
        else:
            raise TypeError("Value of x cannot be None")

    @property
    def has_exceptions(self):
        """ Return whether the LAMMPS shared library was compiled with C++ exceptions handling enabled """
        self._send(command="has_exceptions", data=[])
        return self._receive()

    @property
    def has_gzip_support(self):
        self._send(command="has_gzip_support", data=[])
        return self._receive()

    @property
    def has_png_support(self):
        self._send(command="has_png_support", data=[])
        return self._receive()

    @property
    def has_jpeg_support(self):
        self._send(command="has_jpeg_support", data=[])
        return self._receive()

    @property
    def has_ffmpeg_support(self):
        self._send(command="has_ffmpeg_support", data=[])
        return self._receive()

    @property
    def installed_packages(self):
        self._send(command="get_installed_packages", data=[])
        return self._receive()

    def set_fix_external_callback(self, *args):
        self._send(command="set_fix_external_callback", data=list(args))

    def get_neighlist(self, *args):
        """Returns an instance of :class:`NeighList` which wraps access to the neighbor list with the given index
        :param idx: index of neighbor list
        :type  idx: int
        :return: an instance of :class:`NeighList` wrapping access to neighbor list data
        :rtype:  NeighList
        """
        self._send(command="get_neighlist", data=list(args))
        return self._receive()

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
        self._send(command="find_pair_neighlist", data=list(args))
        return self._receive()

    def find_fix_neighlist(self, *args):
        """Find neighbor list index of fix neighbor list
        :param fixid: name of fix
        :type  fixid: string
        :param request:   index of neighbor list request, in case there are more than one, defaults to 0
        :type  request:   int, optional
        :return: neighbor list index if found, otherwise -1
        :rtype:  int
         """
        self._send(command="find_fix_neighlist", data=list(args))
        return self._receive()

    def find_compute_neighlist(self, *args):
        """Find neighbor list index of compute neighbor list
        :param computeid: name of compute
        :type  computeid: string
        :param request:   index of neighbor list request, in case there are more than one, defaults to 0
        :type  request:   int, optional
        :return: neighbor list index if found, otherwise -1
        :rtype:  int
         """
        self._send(command="find_compute_neighlist", data=list(args))
        return self._receive()

    def get_neighlist_size(self, *args):
        """Return the number of elements in neighbor list with the given index
        :param idx: neighbor list index
        :type  idx: int
        :return: number of elements in neighbor list with index idx
        :rtype:  int
         """
        self._send(command="get_neighlist_size", data=list(args))
        return self._receive()

    def get_neighlist_element_neighbors(self, *args):
        self._send(command="get_neighlist_element_neighbors", data=list(args))
        return self._receive()

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
        if isinstance(cmd, list):
            for c in cmd:
                self._send(command="command", data=c)
                _ = self._receive()
        elif len(cmd.split('\n')) > 1:
            for c in cmd.split('\n'):
                self._send(command="command", data=c)
                _ = self._receive()
        else:
            self._send(command="command", data=cmd)
            _ = self._receive()

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
        if concat:
            self._send(command="gather_atoms_concat", data=list(args))
        elif ids is not None:
            lenids = len(ids)
            args = list(args)
            args.append(len(ids))
            args.append(ids)
            self._send(command="gather_atoms_subset", data=args)
        else:
            self._send(command="gather_atoms", data=list(args))
        return self._receive()

    def scatter_atoms(self, *args, ids=None):
        """
        Scatter atoms for the lammps library

        Args:
            *args:
        """
        if ids is not None:
            lenids = len(ids)
            args = list(args)
            args.append(len(ids))
            args.append(ids)
            self._send(command="scatter_atoms_subset", data=args)
            _ = self._receive()
        else:
            self._send(command="scatter_atoms", data=list(args))
            _ = self._receive()

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
        self._send(command="get_thermo", data=list(args))
        return self._receive()

    #TODO
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
        args = [id, style, type, length, width]
        self._send(command="extract_compute", data=args)
        return self._receive()

    def close(self):
        """
        Close the current lammps object

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._send(command="close")
        try:
            self._process.kill()
        except AttributeError:
            pass
        self._process = None

    #TODO
    def __del__(self):
        if self._process is not None:
            self.close()
