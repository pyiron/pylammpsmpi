# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pylammpsmpi.utils.commands import command_list, thermo_list, func_list, prop_list
from pylammpsmpi.utils.lammps import LammpsBase

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


class LammpsLibrary:
    """
    Top level class which manages the lammps library provided by LammpsBase
    """
    def __init__(self, cores=1, working_directory=".", client=None, mode='local'):
        self.cores = cores
        self.working_directory = working_directory
        self.client = client
        self.mode = mode

        if self.mode == 'dask':
            fut = self.client.submit(LammpsBase, cores=self.cores, working_directory=self.working_directory, actor=True)
            self.lmp = fut.result()

            fut = self.lmp.start_process()
            _ = fut.result()

        elif self.mode == 'local':
            self.lmp = LammpsBase(cores=self.cores, working_directory=self.working_directory)
            self.lmp.start_process()

        else:
            raise ValueError("mode should be either dask or local")

    def __getattr__(self, name):
        """
        Try to run input as a lammps command
        """
        if name in func_list:
            if self.mode == 'dask':
                def func_wrapper(*args, **kwargs):
                    func = getattr(self.lmp, name)
                    fut = func(*args, **kwargs)
                    return fut.result()
            else:
                def func_wrapper(*args, **kwargs):
                    func = getattr(self.lmp, name)
                    fut = func(*args, **kwargs)
                    return fut

            return func_wrapper

        elif name in thermo_list:
            if self.mode == 'dask':
                fut = self.lmp.get_thermo(name)
                return fut.result()
            else:
                fut = self.lmp.get_thermo(name)
                return fut

        elif name in command_list:
            if self.mode == 'dask':
                def command_wrapper(*args):
                    args = [name] + list(args)
                    cmd = " ".join([str(x) for x in args])
                    fut = self.lmp.command(cmd)
                    return fut.result()
            else:
                def command_wrapper(*args):
                    args = [name] + list(args)
                    cmd = " ".join([str(x) for x in args])
                    fut = self.lmp.command(cmd)
                    return fut
            return command_wrapper

        elif name in prop_list:
            fut = getattr(self.lmp, name)
            return fut

        else:
            raise AttributeError(name)

    def close(self):
        fut = self.lmp.close()
        if fut is not None:
            return fut.result()
