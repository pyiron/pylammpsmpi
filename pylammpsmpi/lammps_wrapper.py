# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pylammpsmpi.utils.commands import command_list, thermo_list, func_list, prop_list
from pylammpsmpi.utils.lammps import LammpsBase, LammpsConcurrent

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

    def __init__(
        self,
        cores=1,
        oversubscribe=False,
        working_directory=".",
        client=None,
        mode="local",
        cmdargs=None,
    ):
        self.cores = cores
        self.working_directory = working_directory
        self.oversubscribe = oversubscribe
        self.client = client
        self.mode = mode

        if self.mode == "dask":
            fut = self.client.submit(
                LammpsBase,
                cores=self.cores,
                oversubscribe=self.oversubscribe,
                working_directory=self.working_directory,
                cmdargs=cmdargs,
                actor=True,
            )
            self.lmp = fut.result()

            fut = self.lmp.start_process()
            _ = fut.result()

        elif self.mode == "local":
            self.lmp = LammpsConcurrent(
                cores=self.cores,
                oversubscribe=self.oversubscribe,
                working_directory=self.working_directory,
                cmdargs=cmdargs,
            )
            self.lmp.start_process()

        else:
            raise ValueError("mode should be either dask or local")

    def __getattr__(self, name):
        """
        Try to run input as a lammps command
        """
        if name in func_list:
            def func_wrapper(*args, **kwargs):
                func = getattr(self.lmp, name)
                fut = func(*args, **kwargs)
                return fut.result()

            return func_wrapper

        elif name in thermo_list:
            fut = self.lmp.get_thermo(name)
            return fut.result()

        elif name in command_list:
            def command_wrapper(*args):
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

    def close(self):
        fut = self.lmp.close()
        if fut is not None:
            return fut.result()

    def __dir__(self):
        return (
            list(super().__dir__()) + func_list + thermo_list + command_list + prop_list
        )
