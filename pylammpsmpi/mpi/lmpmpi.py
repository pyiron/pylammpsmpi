# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from ctypes import c_double, c_int
from mpi4py import MPI
import numpy as np
import pickle
import sys
from lammps import lammps

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
    # we can add more quantities as needed
    # taken directly from atom.cpp -> extract()
}

# Lammps executable
job = lammps(cmdargs=["-screen", "none"])


def extract_compute(funct_args):
    # if MPI.COMM_WORLD.rank == 0:
    id = funct_args[0]
    style =  funct_args[1]
    type = funct_args[2]
    length = funct_args[3]
    width = funct_args[4]
    filtered_args = [id, style, type]

    val = job.extract_compute(*filtered_args)

    # now process
    # length should be set
    data = []
    if style == 1:
        length = job.get_natoms()
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


def get_version(funct_args):
    if MPI.COMM_WORLD.rank == 0:
        return job.version()


def get_file(funct_args):
    job.file(*funct_args)
    return 1


def commands_list(funct_args):
    job.commands_list(*funct_args)
    return 1


def commands_string(funct_args):
    job.commands_string(*funct_args)
    return 1


def extract_setting(funct_args):
    if MPI.COMM_WORLD.rank == 0:
        return job.extract_setting(*funct_args)


def extract_global(funct_args):
    if MPI.COMM_WORLD.rank == 0:
        return job.extract_global(*funct_args)


def extract_box(funct_args):
    if MPI.COMM_WORLD.rank == 0:
        return job.extract_box(*funct_args)


def extract_atom(funct_args):
    if MPI.COMM_WORLD.rank == 0:
        # extract atoms return an internal data type
        # this has to be reformatted
        name = str(funct_args[0])
        if not name in atom_properties.keys():
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


def extract_fix(funct_args):
    if MPI.COMM_WORLD.rank == 0:
        return job.extract_fix(*funct_args)


def extract_variable(funct_args):
    # in the args - if the third one,
    # which is the type is 1 - a lammps array is returned
    if MPI.COMM_WORLD.rank == 0:
        # if type is 1 - reformat file
        try:
            data = job.extract_variable(*funct_args)
        except ValueError:
            return []
        if funct_args[2] == 1:
            data = np.array(data)
        return data


def get_natoms(funct_args):
    if MPI.COMM_WORLD.rank == 0:
        return job.get_natoms()


def set_variable(funct_args):
    return job.set_variable(*funct_args)


def reset_box(funct_args):
    job.reset_box(*funct_args)
    return 1


def gather_atoms(funct_args):
    # extract atoms return an internal data type
    # this has to be reformatted
    name = str(funct_args[0])
    if name not in atom_properties.keys():
        return []

    # this block prevents error when trying to access values
    # that do not exist
    try:
        val = job.gather_atoms(name, atom_properties[name]["gtype"], atom_properties[name]["dim"])
    except ValueError:
        return []
    # this is per atom quantity - so get
    # number of atoms - first dimension
    val = list(val)
    dim = atom_properties[name]["dim"]
    if dim > 1:
        data = [val[x:x+dim] for x in range(0, len(val), dim)]
    else:
        data = list(val)
    return np.array(data)


def gather_atoms_concat(funct_args):
    # extract atoms return an internal data type
    # this has to be reformatted
    name = str(funct_args[0])
    if not name in atom_properties.keys():
        return []

    # this block prevents error when trying to access values
    # that do not exist
    try:
        val = job.gather_atoms_concat(name, atom_properties[name]["gtype"], atom_properties[name]["dim"])
    except ValueError:
        return []
    # this is per atom quantity - so get
    # number of atoms - first dimension
    val = list(val)
    dim = atom_properties[name]["dim"]
    if dim > 1:
        data = [val[x:x+dim] for x in range(0, len(val), dim)]
    else:
        data = list(val)
    return np.array(data)


def gather_atoms_subset(funct_args):
    # convert to ctypes
    name = str(funct_args[0])
    lenids = int(funct_args[1])
    ids = funct_args[2]

    # prep ids
    cids = (lenids*c_int)()
    for i in range(lenids):
        cids[i] = ids[i]

    if not name in atom_properties.keys():
        return []

    # this block prevents error when trying to access values
    # that do not exist
    try:
        val = job.gather_atoms_subset(name, atom_properties[name]["gtype"], atom_properties[name]["dim"], lenids, cids)
    except ValueError:
        return []
    # this is per atom quantity - so get
    # number of atoms - first dimension
    val = list(val)
    dim = atom_properties[name]["dim"]
    if dim>1:
        data = [val[x:x+dim] for x in range(0, len(val), dim)]
    else:
        data = list(val)
    return np.array(data)


def create_atoms(funct_args):
    # we have to process the input items
    # args are natoms, ids, type, x, v, image, shrinkexceed
    natoms = funct_args[0]
    ids = funct_args[1]
    type = funct_args[2]
    x = funct_args[3]
    v = funct_args[4]
    image = funct_args[5]
    shrinkexceed = funct_args[6]

    id_lmp = (c_int * natoms)()
    id_lmp[:] = ids

    type_lmp = (c_int * natoms)()
    type_lmp[:] = type

    image_lmp = (c_int * natoms)()
    image_lmp[:] = image

    args = [natoms, id_lmp, type_lmp, x, v, image_lmp, shrinkexceed]
    job.create_atoms(*args)
    return 1


def has_exceptions(funct_args):
    return job.has_exceptions


def has_gzip_support(funct_args):
    return job.has_gzip_support


def has_png_support(funct_args):
    return job.has_png_support


def has_jpeg_support(funct_args):
    return job.has_jpeg_support


def has_ffmpeg_support(funct_args):
    return job.has_ffmpeg_support


def installed_packages(funct_args):
    return job.installed_packages


def set_fix_external_callback(funct_args):
    job.set_fix_external_callback(*funct_args)


def get_neighlist(funct_args):
    if MPI.COMM_WORLD.rank == 0:
        return job.get_neighlist(*funct_args)


def find_pair_neighlist(funct_args):
    if MPI.COMM_WORLD.rank == 0:
        return job.find_pair_neighlist(*funct_args)


def find_fix_neighlist(funct_args):
    if MPI.COMM_WORLD.rank == 0:
        return job.find_fix_neighlist(*funct_args)


def find_compute_neighlist(funct_args):
    if MPI.COMM_WORLD.rank == 0:
        return job.find_compute_neighlist(*funct_args)


def get_neighlist_size(funct_args):
    if MPI.COMM_WORLD.rank == 0:
        return job.get_neighlist_size(*funct_args)


def get_neighlist_element_neighbors(funct_args):
    if MPI.COMM_WORLD.rank == 0:
        return job.get_neighlist_element_neighbors(*funct_args)


def get_thermo(funct_args):
    if MPI.COMM_WORLD.rank == 0:
        return np.array(job.get_thermo(*funct_args))


def scatter_atoms(funct_args):
    name = str(funct_args[0])
    py_vector = funct_args[1]
    # now see if its an integer or double type- but before flatten
    py_vector = np.array(py_vector).flatten()

    if atom_properties[name]["gtype"] == 0:
        c_vector = (len(py_vector) * c_int)(*py_vector)
    else:
        c_vector = (len(py_vector) * c_double)(*py_vector)

    job.scatter_atoms(name, atom_properties[name]["gtype"], atom_properties[name]["dim"], c_vector)
    return 1


def scatter_atoms_subset(funct_args):
    name = str(funct_args[0])
    lenids = int(funct_args[2])
    ids = funct_args[3]

    # prep ids
    cids = (lenids*c_int)()
    for i in range(lenids):
        cids[i] = ids[i]

    py_vector = funct_args[1]
    # now see if its an integer or double type- but before flatten
    py_vector = np.array(py_vector).flatten()

    if atom_properties[name]["gtype"] == 0:
        c_vector = (len(py_vector) * c_int)(*py_vector)
    else:
        c_vector = (len(py_vector) * c_double)(*py_vector)

    job.scatter_atoms_subset(name, atom_properties[name]["gtype"], atom_properties[name]["dim"], lenids, cids, c_vector)
    return 1


def command(funct_args):
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
            extract_compute, get_version, get_file, commands_list, commands_string,
            extract_setting, extract_global, extract_box, extract_atom, extract_fix, extract_variable,
            get_natoms, set_variable, reset_box, gather_atoms_concat, gather_atoms_subset,
            scatter_atoms_subset, create_atoms, has_exceptions, has_gzip_support, has_png_support,
            has_jpeg_support, has_ffmpeg_support, installed_packages, set_fix_external_callback,
            get_neighlist, find_pair_neighlist, find_fix_neighlist, find_compute_neighlist, get_neighlist_size,
            get_neighlist_element_neighbors, get_thermo, scatter_atoms, command, gather_atoms
        ]
    }
    return switcher.get(argument)


if __name__ == "__main__":
    while True:
        if MPI.COMM_WORLD.rank == 0:
            input_dict = pickle.load(sys.stdin.buffer)
            # with open('process.txt', 'a') as file:
            #     print('Input:', input_dict, file=file)
        else:
            input_dict = None
        input_dict = MPI.COMM_WORLD.bcast(input_dict, root=0)
        if input_dict["c"] == "close":
            job.close()
            break
        output = select_cmd(input_dict["c"])(input_dict["d"])
        if MPI.COMM_WORLD.rank == 0 and output is not None:
            # with open('process.txt', 'a') as file:
            #     print('Output:', output, file=file)
            pickle.dump(output, sys.stdout.buffer)
            sys.stdout.flush()
