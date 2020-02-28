# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

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
            os.path.dirname(os.path.abspath(__file__)), "lmpmpi.py"
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

    def command(self, command):
        """
        Send a command to the lammps library

        Args:
            command (str):
        """
        self._send(command="command", data=command)

    def gather_atoms(self, *args):
        """
        Gather atoms from the lammps library

        Args:
            *args:

        Returns:
            np.array
        """
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
