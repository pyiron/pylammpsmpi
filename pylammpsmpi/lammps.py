# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import os
import pickle
import subprocess



__author__ = "Jan Janssen"
__copyright__ = (
    "Copyright 2020, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Jan Janssen"
__email__ = "janssen@mpie.de"
__status__ = "production"
__date__ = "Feb 28, 2020"


class LammpsLibrary(object):
    def __init__(self, cores=1, working_directory="."):
        executable = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "mpi", "lmpmpi.py"
        )
        self._process = subprocess.Popen(
            ["mpiexec", "-n", str(cores), "python", executable],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            cwd=working_directory,
        )

    def _send(self, command, data=None):
        """
        Send a command to the Lammps Library executable

        Args:
            command (str): command to be send to the
            data:
        """
        pickle.dump({"c": command, "d": data}, self._process.stdin)
        self._process.stdin.flush()

    def _receive(self):
        """
        Receive data from the Lammps library

        Returns:
            data
        """
        output = pickle.load(self._process.stdout)
        return output

    def version(self):
        self._send(command="get_version", data=[])
        return self._receive()

    def file(self, *args):
        self._send(command="get_file", data=list(args))

    def commands_list(self, *args):
        self._send(command="commands_list", data=list(args))

    def commands_string(self, *args):
        self._send(command="commands_string", data=list(args))

    def extract_setting(self, *args):
        self._send(command="extract_setting", data=list(args))
        return self._receive()

    def extract_global(self, *args):
        self._send(command="extract_global", data=list(args))
        return self._receive()

    def extract_box(self):
        self._send(command="extract_box", data=[])
        return self._receive()

    def extract_atom(self, name=None):
        if ((name is not None)):
            self._send(command="extract_atom", data=list([name]))
            return self._receive()
        else:
            raise ValueError("name cannot be None")

    def extract_fix(self, *args):
        self._send(command="extract_fix", data=list(args))
        return self._receive()

    def extract_variable(self, *args):
        self._send(command="extract_variable", data=list(args))
        return self._receive()

    def get_natoms(self):
        self._send(command="get_natoms", data=[])
        return self._receive()

    def set_variable(self, *args):
        self._send(command="set_variable", data=list(args))
        return self._receive()

    def reset_box(self, *args):
        self._send(command="reset_box", data=list(args))


    def scatter_atoms_subset(self, *args):
        self._send(command="scatter_atoms_subset", data=list(args))

    def create_atoms(self, *args):
        self._send(command="create_atoms", data=list(args))

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
        Send a command to the lammps library

        Args:
            cmd (str):
        """
        self._send(command="command", data=cmd)


    def gather_atoms(self, *args, concat=False, ids=None):
        """
        Gather atoms from the lammps library

        Args:
            *args:

        Returns:
            np.array
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


    def scatter_atoms(self, *args):
        """
        Scatter atoms for the lammps library

        Args:
            *args:
        """
        self._send(command="scatter_atoms", data=list(args))

    def get_thermo(self, *args):
        """
        Get thermo from the lammps library

        Args:
            *args:

        Returns:

        """
        self._send(command="get_thermo", data=list(args))
        return self._receive()

    def extract_compute(self, *args):
        """
        Extract compute from the lammps library

        Args:
            *args:

        Returns:

        """
        self._send(command="extract_compute", data=list(args))
        return self._receive()

    def close(self):
        self._send(command="close")
        self._process.kill()
        self._process = None

    def __del__(self):
        if self._process is not None:
            self.close()
