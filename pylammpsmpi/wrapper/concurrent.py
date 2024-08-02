# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import os
import sys
from concurrent.futures import Future
from queue import Queue
from typing import Any, List, Optional

from executorlib.shared import (
    MpiExecInterface,
    RaisingThread,
    cancel_items_in_queue,
    interface_bootup,
)

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


def execute_async(
    future_queue: Any,
    cmdargs: Optional[List[str]] = None,
    cores: int = 1,
    oversubscribe: bool = False,
    cwd: Optional[str] = None,
) -> None:
    """
    Asynchronously executes a command using MPI.

    Args:
        future_queue (Any): A queue to receive task dictionaries.
        cmdargs (Optional[List[str]], optional): Additional command-line arguments. Defaults to None.
        cores (int, optional): Number of CPU cores to use. Defaults to 1.
        oversubscribe (bool, optional): Whether to oversubscribe the CPU cores. Defaults to False.
        cwd (Optional[str], optional): Current working directory. Defaults to None.

    Returns:
        None
    """
    executable = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "mpi", "lmpmpi.py"
    )
    cmds = [sys.executable, executable]
    if cmdargs is not None:
        cmds.extend(cmdargs)
    interface = interface_bootup(
        command_lst=cmds,
        connections=MpiExecInterface(
            cwd=cwd,
            cores=cores,
            oversubscribe=oversubscribe,
        ),
    )
    while True:
        task_dict = future_queue.get()
        if "shutdown" in task_dict.keys() and task_dict["shutdown"]:
            interface.shutdown(wait=task_dict["wait"])
            break
        elif "command" in task_dict.keys() and "future" in task_dict.keys():
            f = task_dict.pop("future")
            if f.set_running_or_notify_cancel():
                f.set_result(interface.send_and_receive_dict(input_dict=task_dict))


class LammpsConcurrent:
    def __init__(
        self,
        cores: int = 8,
        oversubscribe: bool = False,
        working_directory: str = ".",
        cmdargs: list = None,
    ):
        """
        Initialize the LammpsConcurrent object.

        Parameters
        ----------
        cores : int, optional
            Number of cores to use for parallel execution (default is 8).
        oversubscribe : bool, optional
            Whether to oversubscribe the cores (default is False).
        working_directory : str, optional
            Working directory for Lammps execution (default is current directory).
        cmdargs : list, optional
            Additional command line arguments for Lammps (default is None).

        Returns
        -------
        None
        """
        self.cores = cores
        self.working_directory = working_directory
        self._future_queue = Queue()
        self._process = None
        self._oversubscribe = oversubscribe
        self._cmdargs = cmdargs
        self._start_process()

    def _start_process(self):
        self._process = RaisingThread(
            target=execute_async,
            kwargs={
                "future_queue": self._future_queue,
                "cmdargs": self._cmdargs,
                "cores": self.cores,
                "oversubscribe": self._oversubscribe,
                "cwd": self.working_directory,
            },
        )
        self._process.start()

    def _send_and_receive_dict(self, command, data=None):
        f = Future()
        self._future_queue.put({"command": command, "args": data, "future": f})
        return f

    @property
    def version(self) -> Future:
        """
        Get the version of Lammps.

        Returns
        -------
        version : Future
            A future object representing the version string of Lammps.
        """
        return self._send_and_receive_dict(command="get_version", data=[])

    def file(self, inputfile: str) -> Future:
        """
        Read script from an input file.

        Parameters
        ----------
        inputfile : str
            Name of the input file.

        Returns
        -------
        None
        """
        if not os.path.exists(inputfile):
            raise FileNotFoundError("Input file does not exist")
        return self._send_and_receive_dict(command="get_file", data=[inputfile])

    # TODO
    def extract_setting(self, *args):
        return self._send_and_receive_dict(command="extract_setting", data=list(args))

    def extract_global(self, name: str) -> Future:
        """
        Extract value of global simulation parameters.

        Parameters
        ----------
        name : str
            Name of the global simulation parameter.

        Returns
        -------
        value : Future
            A future object representing the value of the requested global parameter.
        """
        return self._send_and_receive_dict(command="extract_global", data=[name])

    def extract_box(self) -> Future:
        """
        Get the simulation box.

        Returns
        -------
        box : Future
            A future object representing the simulation box.
        """
        return self._send_and_receive_dict(command="extract_box", data=[])

    def extract_atom(self, name: str) -> Future:
        """
        Extract a property of the atoms.

        Parameters
        ----------
        name : str
            The property of the atom to be extracted.

        Returns
        -------
        val : Future
            A future object representing the extracted property of the atoms.
        """
        return self._send_and_receive_dict(command="extract_atom", data=list([name]))

    def extract_fix(self, *args) -> Future:
        """
        Extract a fix value.

        Parameters
        ----------
        args : tuple
            Variable number of arguments specifying the fix value to extract.

        Returns
        -------
        value : Future
            A future object representing the extracted fix value.
        """
        return self._send_and_receive_dict(command="extract_fix", data=list(args))

    def extract_variable(self, *args) -> Future:
        """
        Extract the value of a variable.

        Parameters
        ----------
        args : tuple
            Variable number of arguments specifying the variable to extract.

        Returns
        -------
        data : Future
            A future object representing the value of the variable.
        """
        return self._send_and_receive_dict(command="extract_variable", data=list(args))

    @property
    def natoms(self) -> Future:
        """
        Get the number of atoms.

        Returns
        -------
        natoms : Future
            A future object representing the number of atoms.
        """
        return self.get_natoms()

    def get_natoms(self) -> Future:
        """
        Get the number of atoms.

        Returns
        -------
        natoms : Future
            A future object representing the number of atoms.
        """
        return self._send_and_receive_dict(command="get_natoms", data=[])

    def set_variable(self, *args) -> Future:
        """
        Set the value of a string style variable.

        Parameters
        ----------
        args : tuple
            Variable number of arguments specifying the variable name and value.

        Returns
        -------
        flag : Future
            A future object representing the success flag (0 if successful, -1 otherwise).
        """
        return self._send_and_receive_dict(command="set_variable", data=list(args))

    def reset_box(self, *args) -> Future:
        """
        Reset the simulation box.

        Parameters
        ----------
        args : tuple
            Variable number of arguments specifying the box parameters.

        Returns
        -------
        None
        """
        return self._send_and_receive_dict(command="reset_box", data=list(args))

    def generate_atoms(
        self, ids=None, type=None, x=None, v=None, image=None, shrinkexceed=False
    ) -> Future:
        """
        Create atoms on all processors.

        Parameters
        ----------
        ids : list of ints, optional
            IDs of N atoms that need to be created.
        type : list of atom types, optional
            Type of N atoms.
        x : list of positions
            List of positions for N atoms.
        v : list of velocities, optional
            List of velocities for N atoms.
        image : list of ints, optional
            List of image flags for N atoms.
        shrinkexceed : bool, optional
            Whether to shrink the box if the atoms exceed the box boundaries.

        Returns
        -------
        None
        """
        return self.create_atoms(
            ids=ids, type=type, x=x, v=v, image=image, shrinkexceed=shrinkexceed
        )

    def create_atoms(
        self, n, id, type, x, v=None, image=None, shrinkexceed=False
    ) -> Future:
        """
        Create atoms on all processors.

        Parameters
        ----------
        n : int
            Number of atoms.
        id : list of ints, optional
            IDs of N atoms that need to be created.
        type : list of atom types, optional
            Type of N atoms.
        x : list of positions
            List of positions for N atoms.
        v : list of velocities, optional
            List of velocities for N atoms.
        image : list of ints, optional
            List of image flags for N atoms.
        shrinkexceed : bool, optional
            Whether to shrink the box if the atoms exceed the box boundaries.

        Returns
        -------
        None
        """
        if x is not None:
            funct_args = [n, id, type, x, v, image, shrinkexceed]
            return self._send_and_receive_dict(command="create_atoms", data=funct_args)
        else:
            raise TypeError("Value of x cannot be None")

    @property
    def has_exceptions(self) -> Future:
        """
        Return whether the Lammps shared library was compiled with C++ exceptions handling enabled.

        Returns
        -------
        has_exceptions : Future
            A future object representing whether the Lammps library has exceptions handling enabled.
        """
        return self._send_and_receive_dict(command="has_exceptions", data=[])

    @property
    def has_gzip_support(self) -> Future:
        """
        Return whether the Lammps shared library has gzip support.

        Returns
        -------
        has_gzip_support : Future
            A future object representing whether the Lammps library has gzip support.
        """
        return self._send_and_receive_dict(command="has_gzip_support", data=[])

    @property
    def has_png_support(self) -> Future:
        """
        Return whether the Lammps shared library has PNG support.

        Returns
        -------
        has_png_support : Future
            A future object representing whether the Lammps library has PNG support.
        """
        return self._send_and_receive_dict(command="has_png_support", data=[])

    @property
    def has_jpeg_support(self) -> Future:
        """
        Return whether the Lammps shared library has JPEG support.

        Returns
        -------
        has_jpeg_support : Future
            A future object representing whether the Lammps library has JPEG support.
        """
        return self._send_and_receive_dict(command="has_jpeg_support", data=[])
        return self._send_and_receive_dict(command="has_jpeg_support", data=[])

    @property
    def has_ffmpeg_support(self):
        return self._send_and_receive_dict(command="has_ffmpeg_support", data=[])

    @property
    def installed_packages(self):
        return self._send_and_receive_dict(command="get_installed_packages", data=[])

    def set_fix_external_callback(self, *args):
        return self._send_and_receive_dict(
            command="set_fix_external_callback", data=list(args)
        )

    def get_neighlist(self, *args):
        """Returns an instance of :class:`NeighList` which wraps access to the neighbor list with the given index
        :param idx: index of neighbor list
        :type  idx: int
        :return: an instance of :class:`NeighList` wrapping access to neighbor list data
        :rtype:  NeighList
        """
        return self._send_and_receive_dict(command="get_neighlist", data=list(args))

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
        return self._send_and_receive_dict(
            command="find_pair_neighlist", data=list(args)
        )

    def find_fix_neighlist(self, *args):
        """Find neighbor list index of fix neighbor list
        :param fixid: name of fix
        :type  fixid: string
        :param request:   index of neighbor list request, in case there are more than one, defaults to 0
        :type  request:   int, optional
        :return: neighbor list index if found, otherwise -1
        :rtype:  int
        """
        return self._send_and_receive_dict(
            command="find_fix_neighlist", data=list(args)
        )

    def find_compute_neighlist(self, *args):
        """Find neighbor list index of compute neighbor list
        :param computeid: name of compute
        :type  computeid: string
        :param request:   index of neighbor list request, in case there are more than one, defaults to 0
        :type  request:   int, optional
        :return: neighbor list index if found, otherwise -1
        :rtype:  int
        """
        return self._send_and_receive_dict(
            command="find_compute_neighlist", data=list(args)
        )

    def get_neighlist_size(self, *args):
        """Return the number of elements in neighbor list with the given index
        :param idx: neighbor list index
        :type  idx: int
        :return: number of elements in neighbor list with index idx
        :rtype:  int
        """
        return self._send_and_receive_dict(
            command="get_neighlist_size", data=list(args)
        )

    def get_neighlist_element_neighbors(self, *args):
        return self._send_and_receive_dict(
            command="get_neighlist_element_neighbors", data=list(args)
        )

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
            return self._send_and_receive_dict(
                command="commands_string", data="\n".join(cmd)
            )
        elif len(cmd.split("\n")) > 1:
            return self._send_and_receive_dict(command="commands_string", data=cmd)
        else:
            return self._send_and_receive_dict(command="command", data=cmd)

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
            return self._send_and_receive_dict(
                command="gather_atoms_concat", data=list(args)
            )
        elif ids is not None:
            args = list(args)
            args.append(len(ids))
            args.append(ids)
            return self._send_and_receive_dict(command="gather_atoms_subset", data=args)
        else:
            return self._send_and_receive_dict(command="gather_atoms", data=list(args))

    def scatter_atoms(self, *args, ids=None):
        """
        Scatter atoms for the lammps library

        Args:
            *args:
        """
        if ids is not None:
            args = list(args)
            args.append(len(ids))
            args.append(ids)
            return self._send_and_receive_dict(
                command="scatter_atoms_subset", data=args
            )
        else:
            return self._send_and_receive_dict(command="scatter_atoms", data=list(args))

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
        return self._send_and_receive_dict(command="get_thermo", data=list(args))

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
        args = [id, style, type, length, width]
        return self._send_and_receive_dict(command="extract_compute", data=args)

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
        cancel_items_in_queue(que=self._future_queue)
        self._future_queue.put({"shutdown": True, "wait": True})
        self._process.join()
        self._process = None

    # TODO
    def __del__(self):
        if self._process is not None:
            self.close()
