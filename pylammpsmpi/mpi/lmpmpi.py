# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import sys
from ctypes import c_double, c_int

import numpy as np
from executorlib.shared import (
    interface_connect,
    interface_receive,
    interface_send,
    interface_shutdown,
)
from lammps import lammps
from mpi4py import MPI

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

# dict for extract atom methods
atom_properties = {
    "x": {"type": 3, "gtype": 1, "dim": 3},
    "mass": {"type": 2, "gtype": 1, "dim": 1},
    "id": {"type": 0, "gtype": 0, "dim": 1},
    "type": {"type": 0, "gtype": 0, "dim": 1},
    "mask": {"type": 0, "gtype": 0, "dim": 1},
    "v": {"type": 3, "gtype": 1, "dim": 3},
    "f": {"type": 3, "gtype": 1, "dim": 3},
    "molecule": {"type": 0, "gtype": 0, "dim": 1},
    "q": {"type": 2, "gtype": 1, "dim": 1},
    "mu": {"type": 3, "gtype": 1, "dim": 3},
    "omega": {"type": 3, "gtype": 1, "dim": 3},
    "angmom": {"type": 3, "gtype": 1, "dim": 3},
    "torque": {"type": 3, "gtype": 1, "dim": 3},
    "radius": {"type": 2, "gtype": 1, "dim": 1},
    "image": {"type": 2, "gtype": 0, "dim": 3},
    # we can add more quantities as needed
    # taken directly from atom.cpp -> extract()
}


def extract_compute(job, funct_args):
    def convert_data(val, type, length, width):
        data = []
        if type == 2:
            for i in range(length):
                dummy = []
                for j in range(width):
                    dummy.append(val[i][j])
                data.append(np.array(dummy))
            data = np.array(data)
        elif type == 1:
            for i in range(length):
                data.append(val[i])
            data = np.array(data)
        else:
            data = val
        return data

    id = funct_args[0]
    style = funct_args[1]
    type = funct_args[2]
    length = funct_args[3]
    width = funct_args[4]

    filtered_args = [id, style, type]
    if style == 0:
        val = job.extract_compute(*filtered_args)
        return convert_data(val=val, type=type, length=length, width=width)
    elif style == 1:  # per atom property
        val = _gather_data_from_all_processors(
            data=job.numpy.extract_compute(*filtered_args)
        )
        if MPI.COMM_WORLD.rank == 0:
            length = job.get_natoms()
            return convert_data(val=val, type=type, length=length, width=width)
    else:  # Todo
        raise ValueError("Local style is currently not supported")


def get_version(job, funct_args):
    if MPI.COMM_WORLD.rank == 0:
        return job.version()


def get_file(job, funct_args):
    job.file(*funct_args)
    return 1


def commands_list(job, funct_args):
    job.commands_list(*funct_args)
    return 1


def commands_string(job, funct_args):
    job.commands_string(*funct_args)
    return 1


def extract_setting(job, funct_args):
    if MPI.COMM_WORLD.rank == 0:
        return job.extract_setting(*funct_args)


def extract_global(job, funct_args):
    if MPI.COMM_WORLD.rank == 0:
        return job.extract_global(*funct_args)


def extract_box(job, funct_args):
    if MPI.COMM_WORLD.rank == 0:
        return job.extract_box(*funct_args)


def extract_atom(job, funct_args):
    if MPI.COMM_WORLD.rank == 0:
        # extract atoms return an internal data type
        # this has to be reformatted
        name = str(funct_args[0])
        if name not in atom_properties.keys():
            return []

        # this block prevents error when trying to access values
        # that do not exist
        try:
            val = job.extract_atom(name, atom_properties[name]["type"])
        except ValueError:
            return []
        # this is per atom quantity - so get
        # number of atoms - first dimension
        natoms = job.get_natoms()
        # second dim is from dict
        dim = atom_properties[name]["dim"]
        data = []
        if dim > 1:
            for i in range(int(natoms)):
                dummy = [val[i][x] for x in range(dim)]
                data.append(dummy)
        else:
            data = [val[x] for x in range(int(natoms))]

        return np.array(data)


def extract_fix(job, funct_args):
    if MPI.COMM_WORLD.rank == 0:
        return job.extract_fix(*funct_args)


def extract_variable(job, funct_args):
    # in the args - if the third one,
    # which is the type is 1 - a lammps array is returned
    if funct_args[2] == 1:
        data = _gather_data_from_all_processors(
            data=job.numpy.extract_variable(*funct_args)
        )
        if MPI.COMM_WORLD.rank == 0:
            return np.array(data)
    else:
        if MPI.COMM_WORLD.rank == 0:
            # if type is 1 - reformat file
            try:
                data = job.extract_variable(*funct_args)
            except ValueError:
                return []
            return data


def get_natoms(job, funct_args):
    if MPI.COMM_WORLD.rank == 0:
        return job.get_natoms()


def set_variable(job, funct_args):
    return job.set_variable(*funct_args)


def reset_box(job, funct_args):
    job.reset_box(*funct_args)
    return 1


def gather_atoms(job, funct_args):
    # extract atoms return an internal data type
    # this has to be reformatted
    name = str(funct_args[0])
    if name not in atom_properties.keys():
        return []

    # this block prevents error when trying to access values
    # that do not exist
    try:
        val = job.gather_atoms(
            name, atom_properties[name]["gtype"], atom_properties[name]["dim"]
        )
    except ValueError:
        return []
    # this is per atom quantity - so get
    # number of atoms - first dimension
    val = list(val)
    dim = atom_properties[name]["dim"]
    if dim > 1:
        data = [val[x : x + dim] for x in range(0, len(val), dim)]
    else:
        data = list(val)
    return np.array(data)


def gather_atoms_concat(job, funct_args):
    # extract atoms return an internal data type
    # this has to be reformatted
    name = str(funct_args[0])
    if name not in atom_properties.keys():
        return []

    # this block prevents error when trying to access values
    # that do not exist
    try:
        val = job.gather_atoms_concat(
            name, atom_properties[name]["gtype"], atom_properties[name]["dim"]
        )
    except ValueError:
        return []
    # this is per atom quantity - so get
    # number of atoms - first dimension
    val = list(val)
    dim = atom_properties[name]["dim"]
    if dim > 1:
        data = [val[x : x + dim] for x in range(0, len(val), dim)]
    else:
        data = list(val)
    return np.array(data)


def gather_atoms_subset(job, funct_args):
    # convert to ctypes
    name = str(funct_args[0])
    lenids = int(funct_args[1])
    ids = funct_args[2]

    # prep ids
    cids = (lenids * c_int)()
    for i in range(lenids):
        cids[i] = ids[i]

    if name not in atom_properties.keys():
        return []

    # this block prevents error when trying to access values
    # that do not exist
    try:
        val = job.gather_atoms_subset(
            name,
            atom_properties[name]["gtype"],
            atom_properties[name]["dim"],
            lenids,
            cids,
        )
    except ValueError:
        return []
    # this is per atom quantity - so get
    # number of atoms - first dimension
    val = list(val)
    dim = atom_properties[name]["dim"]
    if dim > 1:
        data = [val[x : x + dim] for x in range(0, len(val), dim)]
    else:
        data = list(val)
    return np.array(data)


def create_atoms(job, funct_args):
    job.create_atoms(*funct_args)
    return 1


def has_exceptions(job, funct_args):
    return job.has_exceptions


def has_gzip_support(job, funct_args):
    return job.has_gzip_support


def has_png_support(job, funct_args):
    return job.has_png_support


def has_jpeg_support(job, funct_args):
    return job.has_jpeg_support


def has_ffmpeg_support(job, funct_args):
    return job.has_ffmpeg_support


def installed_packages(job, funct_args):
    return job.installed_packages


def set_fix_external_callback(job, funct_args):
    job.set_fix_external_callback(*funct_args)
    return 1


def get_neighlist(job, funct_args):
    if MPI.COMM_WORLD.rank == 0:
        return job.get_neighlist(*funct_args)


def find_pair_neighlist(job, funct_args):
    if MPI.COMM_WORLD.rank == 0:
        return job.find_pair_neighlist(*funct_args)


def find_fix_neighlist(job, funct_args):
    if MPI.COMM_WORLD.rank == 0:
        return job.find_fix_neighlist(*funct_args)


def find_compute_neighlist(job, funct_args):
    if MPI.COMM_WORLD.rank == 0:
        return job.find_compute_neighlist(*funct_args)


def get_neighlist_size(job, funct_args):
    if MPI.COMM_WORLD.rank == 0:
        return job.get_neighlist_size(*funct_args)


def get_neighlist_element_neighbors(job, funct_args):
    if MPI.COMM_WORLD.rank == 0:
        return job.get_neighlist_element_neighbors(*funct_args)


def get_thermo(job, funct_args):
    if MPI.COMM_WORLD.rank == 0:
        return np.array(job.get_thermo(*funct_args))


def scatter_atoms(job, funct_args):
    name = str(funct_args[0])
    py_vector = funct_args[1]
    # now see if its an integer or double type- but before flatten
    py_vector = np.array(py_vector).flatten()

    if atom_properties[name]["gtype"] == 0:
        c_vector = (len(py_vector) * c_int)(*py_vector)
    else:
        c_vector = (len(py_vector) * c_double)(*py_vector)

    job.scatter_atoms(
        name, atom_properties[name]["gtype"], atom_properties[name]["dim"], c_vector
    )
    return 1


def scatter_atoms_subset(job, funct_args):
    name = str(funct_args[0])
    lenids = int(funct_args[2])
    ids = funct_args[3]

    # prep ids
    cids = (lenids * c_int)()
    for i in range(lenids):
        cids[i] = ids[i]

    py_vector = funct_args[1]
    # now see if its an integer or double type- but before flatten
    py_vector = np.array(py_vector).flatten()

    if atom_properties[name]["gtype"] == 0:
        c_vector = (len(py_vector) * c_int)(*py_vector)
    else:
        c_vector = (len(py_vector) * c_double)(*py_vector)

    job.scatter_atoms_subset(
        name,
        atom_properties[name]["gtype"],
        atom_properties[name]["dim"],
        lenids,
        cids,
        c_vector,
    )
    return 1


def command(job, funct_args):
    job.command(funct_args)
    return 1


def select_cmd(argument):
    """
    Select a lammps command

    Args:
        argument (str): [close, extract_compute, get_thermo, scatter_atoms, command, gather_atoms]

    Returns:
        function: the selected function
    """
    switcher = {
        f.__name__: f
        for f in [
            extract_compute,
            get_version,
            get_file,
            commands_list,
            commands_string,
            extract_setting,
            extract_global,
            extract_box,
            extract_atom,
            extract_fix,
            extract_variable,
            get_natoms,
            set_variable,
            reset_box,
            gather_atoms_concat,
            gather_atoms_subset,
            scatter_atoms_subset,
            create_atoms,
            has_exceptions,
            has_gzip_support,
            has_png_support,
            has_jpeg_support,
            has_ffmpeg_support,
            installed_packages,
            set_fix_external_callback,
            get_neighlist,
            find_pair_neighlist,
            find_fix_neighlist,
            find_compute_neighlist,
            get_neighlist_size,
            get_neighlist_element_neighbors,
            get_thermo,
            scatter_atoms,
            command,
            gather_atoms,
        ]
    }
    return switcher.get(argument)


def _gather_data_from_all_processors(data):
    data_gather = MPI.COMM_WORLD.gather(data, root=0)
    if MPI.COMM_WORLD.rank == 0:
        data = []
        for vl in data_gather:
            for v in vl:
                data.append(v)
        return data


def _run_lammps_mpi(argument_lst):
    index_selected = argument_lst.index("--zmqport")
    port_selected = argument_lst[index_selected + 1]
    if "--host" in argument_lst:
        index_selected = argument_lst.index("--host")
        host = argument_lst[index_selected + 1]
    else:
        host = "localhost"
    argument_red_lst = argument_lst[:index_selected]
    if MPI.COMM_WORLD.rank == 0:
        context, socket = interface_connect(host=host, port=port_selected)
    else:
        context, socket = None, None
    # Lammps executable
    args = ["-screen", "none"]
    if len(argument_red_lst) > 1:
        args.extend(argument_red_lst[1:])
    job = lammps(cmdargs=args)
    while True:
        if MPI.COMM_WORLD.rank == 0:
            input_dict = interface_receive(socket=socket)
        else:
            input_dict = None
        input_dict = MPI.COMM_WORLD.bcast(input_dict, root=0)
        if "shutdown" in input_dict.keys() and input_dict["shutdown"]:
            job.close()
            if MPI.COMM_WORLD.rank == 0:
                interface_send(socket=socket, result_dict={"result": True})
                interface_shutdown(socket=socket, context=context)
            break
        output = select_cmd(input_dict["command"])(
            job=job, funct_args=input_dict["args"]
        )
        if MPI.COMM_WORLD.rank == 0 and output is not None:
            interface_send(socket=socket, result_dict={"result": output})


if __name__ == "__main__":
    _run_lammps_mpi(argument_lst=sys.argv)
