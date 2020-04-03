import os
import pickle
import subprocess
import sys
from pylammpsmpi.commands import command_list, thermo_list


class DaskLammps:
    """
    Dask lammps implementation
    """
    def __init__(self, cores=8, working_directory="."):
        self.cores = cores
        self.working_directory = working_directory

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
