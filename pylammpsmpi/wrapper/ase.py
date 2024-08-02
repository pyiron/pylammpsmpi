import importlib
import os
import warnings
from ctypes import c_double, c_int
from typing import List, Optional

import numpy as np
from ase.atoms import Atoms
from ase.calculators.lammps import Prism
from ase.data import atomic_masses, atomic_numbers
from scipy import constants

from pylammpsmpi.wrapper.base import LammpsBase


class LammpsASELibrary:
    """
    A class that provides an interface to interact with LAMMPS using ASE.

    Args:
        working_directory (str, optional): The working directory. Defaults to None.
        cores (int, optional): The number of cores to use. Defaults to 1.
        comm (object, optional): The MPI communicator. Defaults to None.
        logger (object, optional): The logger object. Defaults to None.
        log_file (str, optional): The log file path. Defaults to None.
        library (object, optional): The LAMMPS library object. Defaults to None.
        disable_log_file (bool, optional): Whether to disable the log file. Defaults to True.
    """

    def __init__(
        self,
        working_directory: Optional[str] = None,
        cores: int = 1,
        comm: Optional[object] = None,
        logger: Optional[object] = None,
        log_file: Optional[str] = None,
        library: Optional[object] = None,
        disable_log_file: bool = True,
    ):
        self._logger = logger
        self._prism = None
        self._structure = None
        self._cores = cores
        if library is not None:
            self._interactive_library = library
            self._cores = library.cores
        elif self._cores == 1:
            lammps = getattr(importlib.import_module("lammps"), "lammps")
            if disable_log_file:
                self._interactive_library = lammps(
                    cmdargs=["-screen", "none", "-log", "none"],
                    comm=comm,
                )
            else:
                if log_file is None:
                    log_file = os.path.join(working_directory, "log.lammps")
                self._interactive_library = lammps(
                    cmdargs=["-screen", "none", "-log", log_file],
                    comm=comm,
                )
        else:
            self._interactive_library = LammpsBase(
                cores=self._cores, working_directory=working_directory
            )

    def interactive_lib_command(self, command: str) -> None:
        """
        Execute a LAMMPS command using the interactive library.

        Args:
            command (str): The LAMMPS command to execute.
        """
        if self._logger is not None:
            self._logger.debug("Lammps library: " + command)
        self._interactive_library.command(command)

    def interactive_positions_getter(self) -> np.ndarray:
        """
        Get the positions of atoms from the interactive library.

        Returns:
            np.ndarray: The positions of atoms.
        """
        positions = np.reshape(
            np.array(self._interactive_library.gather_atoms("x", 1, 3)),
            (len(self._structure), 3),
        )
        if not _check_ortho_prism(prism=self._prism):
            positions = self._prism.vector_to_ase(positions)
        return positions

    def interactive_positions_setter(self, positions: List[List[float]]) -> None:
        """
        Set the positions of atoms in the interactive library.

        Args:
            positions (List[List[float]]): The positions of atoms.
        """
        if not _check_ortho_prism(prism=self._prism):
            positions = self._prism.vector_to_lammps(positions)
        positions = np.array(positions).flatten()
        if self._cores == 1:
            self._interactive_library.scatter_atoms(
                "x", 1, 3, (len(positions) * c_double)(*positions)
            )
        else:
            self._interactive_library.scatter_atoms("x", positions)
        self.interactive_lib_command(command="change_box all remap")

    def interactive_cells_getter(self) -> np.ndarray:
        """
        Get the cell vectors from the interactive library.

        Returns:
            np.ndarray: The cell vectors.
        """
        cc = np.array(
            [
                [self._interactive_library.get_thermo("lx"), 0, 0],
                [
                    self._interactive_library.get_thermo("xy"),
                    self._interactive_library.get_thermo("ly"),
                    0,
                ],
                [
                    self._interactive_library.get_thermo("xz"),
                    self._interactive_library.get_thermo("yz"),
                    self._interactive_library.get_thermo("lz"),
                ],
            ]
        )
        return self._prism.vector_to_ase(cc)

    def interactive_cells_setter(self, cell: np.ndarray) -> None:
        """
        Set the cell vectors in the interactive library.

        Args:
            cell (np.ndarray): The cell vectors.
        """
        self._prism = Prism(cell)
        lx, ly, lz, xy, xz, yz = self._prism.get_lammps_prism()
        if not _check_ortho_prism(prism=self._prism):
            warnings.warn(
                "Warning: setting upper trangular matrix might slow down the calculation"
            )

        is_skewed = cell_is_skewed(cell=cell, tolerance=1.0e-8)
        was_skewed = cell_is_skewed(cell=self._structure.cell, tolerance=1.0e-8)

        if is_skewed:
            if not was_skewed:
                self.interactive_lib_command(command="change_box all triclinic")
            self.interactive_lib_command(
                command="change_box all x final 0 %f y final 0 %f z final 0 %f  xy final %f xz final %f yz final %f remap units box"
                % (lx, ly, lz, xy, xz, yz),
            )
        elif was_skewed:
            self.interactive_lib_command(
                command="change_box all x final 0 %f y final 0 %f z final 0 %f xy final %f xz final %f yz final %f remap units box"
                % (lx, ly, lz, 0.0, 0.0, 0.0),
            )
            self.interactive_lib_command(command="change_box all ortho")
        else:
            self.interactive_lib_command(
                command="change_box all x final 0 %f y final 0 %f z final 0 %f remap units box"
                % (lx, ly, lz),
            )

    def interactive_volume_getter(self) -> float:
        """
        Get the volume of the system from the interactive library.

        Returns:
            float: The volume of the system.
        """
        return self._interactive_library.get_thermo("vol")

    def interactive_forces_getter(self) -> np.ndarray:
        """
        Get the forces on atoms from the interactive library.

        Returns:
            np.ndarray: The forces on atoms.
        """
        ff = np.reshape(
            np.array(self._interactive_library.gather_atoms("f", 1, 3)),
            (len(self._structure), 3),
        )
        if not _check_ortho_prism(prism=self._prism):
            ff = self._prism.vector_to_ase(ff)
        return ff

    def interactive_structure_setter(
        self,
        structure: Atoms,
        units: str,
        dimension: int,
        boundary: str,
        atom_style: str,
        el_eam_lst: List[str],
        calc_md: bool = True,
    ) -> None:
        """
        Set the structure in the interactive library.

        Args:
            structure (ase.Atoms): The structure.
            units (str): The units of the simulation.
            dimension (int): The dimension of the simulation.
            boundary (str): The boundary conditions.
            atom_style (str): The atom style.
            el_eam_lst (List[str]): The list of element symbols.
            calc_md (bool, optional): Whether to calculate molecular dynamics. Defaults to True.
        """
        self.interactive_lib_command(command="clear")
        control_dict = set_selective_dynamics(structure=structure, calc_md=calc_md)
        self.interactive_lib_command(command="units " + units)
        self.interactive_lib_command(command="dimension " + str(dimension))
        self.interactive_lib_command(command="boundary " + boundary)
        self.interactive_lib_command(command="atom_style " + atom_style)

        self.interactive_lib_command(command="atom_modify map array")
        self._prism = Prism(structure.cell)
        if not _check_ortho_prism(prism=self._prism):
            warnings.warn(
                "Warning: setting upper trangular matrix might slow down the calculation"
            )
        xhi, yhi, zhi, xy, xz, yz = self._prism.get_lammps_prism()
        if self._prism.is_skewed():
            self.interactive_lib_command(
                command="region 1 prism"
                + " 0.0 "
                + str(xhi)
                + " 0.0 "
                + str(yhi)
                + " 0.0 "
                + str(zhi)
                + " "
                + str(xy)
                + " "
                + str(xz)
                + " "
                + str(yz)
                + " units box",
            )
        else:
            self.interactive_lib_command(
                command="region 1 block"
                + " 0.0 "
                + str(xhi)
                + " 0.0 "
                + str(yhi)
                + " 0.0 "
                + str(zhi)
                + " units box",
            )
        el_struct_lst = get_species_symbols(structure)
        if atom_style == "full":
            self.interactive_lib_command(
                command="create_box "
                + str(len(el_eam_lst))
                + " 1 "
                + "bond/types 1 "
                + "angle/types 1 "
                + "extra/bond/per/atom 2 "
                + "extra/angle/per/atom 2 ",
            )
        else:
            self.interactive_lib_command(
                command="create_box " + str(len(el_eam_lst)) + " 1"
            )
        el_dict = {}
        for id_eam, el_eam in enumerate(el_eam_lst):
            if el_eam in el_struct_lst:
                self.interactive_lib_command(
                    command="mass {0:3d} {1:f}".format(
                        id_eam + 1, atomic_masses[atomic_numbers[el_eam]]
                    ),
                )
            else:
                self.interactive_lib_command(
                    command="mass {0:3d} {1:f}".format(id_eam + 1, 1.00),
                )
        if not _check_ortho_prism(prism=self._prism):
            positions = self._prism.vector_to_lammps(structure.positions).flatten()
        else:
            positions = structure.positions.flatten()
        try:
            elem_all = get_lammps_indicies_from_ase_structure(
                structure=structure, el_eam_lst=el_eam_lst
            )
        except KeyError:
            missing = set(get_species_symbols(structure)).difference(el_dict.keys())
            missing = ", ".join([el.Abbreviation for el in missing])
            raise ValueError(
                f"Structure contains elements [{missing}], that are not present in the potential!"
            )
        if self._cores == 1:
            self._interactive_library.create_atoms(
                n=len(structure),
                id=None,
                type=elem_all,
                x=positions,
                v=None,
                image=None,
                shrinkexceed=False,
            )
        else:
            self._interactive_library.create_atoms(
                n=len(structure),
                id=range(1, len(structure) + 1),
                type=elem_all,
                x=positions,
                v=None,
                image=None,
                shrinkexceed=False,
            )
        self.interactive_lib_command(command="change_box all remap")
        for key, value in control_dict.items():
            self.interactive_lib_command(command=key + " " + value)
        self._structure = structure

    def interactive_indices_getter(self) -> np.ndarray:
        """
        Get the indices of atoms from the interactive library.

        Returns:
            np.ndarray: The indices of atoms.
        """
        return np.array(self._interactive_library.gather_atoms("type", 0, 1))

    def interactive_energy_pot_getter(self) -> float:
        """
        Get the potential energy from the interactive library.

        Returns:
            float: The potential energy.
        """
        return self._interactive_library.get_thermo("pe")

    def interactive_energy_tot_getter(self) -> float:
        """
        Get the total energy from the interactive library.

        Returns:
            float: The total energy.
        """
        return self._interactive_library.get_thermo("etotal")

    def interactive_steps_getter(self) -> int:
        """
        Get the number of steps from the interactive library.

        Returns:
            int: The number of steps.
        """
        return self._interactive_library.get_thermo("step")

    def interactive_temperatures_getter(self) -> float:
        """
        Get the temperature from the interactive library.

        Returns:
            float: The temperature.
        """
        return self._interactive_library.get_thermo("temp")

    def interactive_pressures_getter(self):
        pp = np.array(
            [
                [
                    self._interactive_library.get_thermo("pxx"),
                    self._interactive_library.get_thermo("pxy"),
                    self._interactive_library.get_thermo("pxz"),
                ],
                [
                    self._interactive_library.get_thermo("pxy"),
                    self._interactive_library.get_thermo("pyy"),
                    self._interactive_library.get_thermo("pyz"),
                ],
                [
                    self._interactive_library.get_thermo("pxz"),
                    self._interactive_library.get_thermo("pyz"),
                    self._interactive_library.get_thermo("pzz"),
                ],
            ]
        )
        if not _check_ortho_prism(prism=self._prism):
            pp = self._prism.tensor2_to_ase(pp)
        return pp

    def interactive_indices_setter(self, indices, el_eam_lst):
        elem_all = get_lammps_indicies_from_ase_indices(
            indices=indices, structure=self._structure, el_eam_lst=el_eam_lst
        )
        if self._cores == 1:
            self._interactive_library.scatter_atoms(
                "type", 0, 1, (len(elem_all) * c_int)(*elem_all)
            )
        else:
            self._interactive_library.scatter_atoms("type", elem_all)

    def interactive_stress_getter(self, enable_stress_computation=True):
        """
        This gives back an Nx3x3 array of stress/atom defined in http://lammps.sandia.gov/doc/compute_stress_atom.html
        Keep in mind that it is stress*volume in eV. Further discussion can be found on the website above.

        Returns:
            numpy.array: Nx3x3 np array of stress/atom
        """
        if enable_stress_computation:
            self.interactive_lib_command("compute st all stress/atom NULL")
            self.interactive_lib_command("run 0")
        id_lst = self._interactive_library.extract_atom("id")
        id_lst = np.array([id_lst[i] for i in range(len(self._structure))]) - 1
        id_lst = np.arange(len(id_lst))[np.argsort(id_lst)]
        ind = np.array([0, 3, 4, 3, 1, 5, 4, 5, 2])
        ss = self._interactive_library.extract_compute("st", 1, 2)
        ss = np.array(
            [ss[i][j] for i in range(len(self._structure)) for j in range(6)]
        ).reshape(-1, 6)[id_lst]
        ss = (
            ss[:, ind].reshape(len(self._structure), 3, 3)
            / constants.eV
            * constants.bar
            * constants.angstrom**3
        )
        if not _check_ortho_prism(prism=self._prism):
            ss = np.einsum("ij,njk->nik", self._prism.rot_mat, ss)
            ss = np.einsum("nij,kj->nik", ss, self._prism.rot_mat)
        return ss

    def interactive_velocities_getter(self):
        velocity = np.reshape(
            np.array(self._interactive_library.gather_atoms("v", 1, 3)),
            (len(self._structure), 3),
        )
        if not _check_ortho_prism(prism=self._prism):
            velocity = self._prism.vector_to_ase(velocity)
        return velocity

    def close(self):
        if self._interactive_library is not None:
            self._interactive_library.close()

    def set_fix_external_callback(self, fix_id, callback, caller=None):
        self._interactive_library.set_fix_external_callback(
            fix_id=fix_id, callback=callback, caller=caller
        )

    def __enter__(self):
        """
        Compatibility function for the with statement
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Compatibility function for the with statement
        """
        self.close()


def cell_is_skewed(cell, tolerance=1.0e-8):
    """
    Check whether the simulation box is skewed/sheared. The algorithm compares the box volume
    and the product of the box length in each direction. If these numbers do not match, the box
    is considered to be skewed and the function returns True

    Args:
        tolerance (float): Relative tolerance above which the structure is considered as skewed

    Returns:
        (bool): Whether the box is skewed or not.
    """
    volume = np.abs(np.linalg.det(cell))
    prod = np.linalg.norm(cell, axis=-1).prod()
    if volume > 0:
        if abs(volume - prod) / volume < tolerance:
            return False
    return True


def _check_ortho_prism(prism, rtol=0.0, atol=1e-08):
    """
    Check if the rotation matrix of the UnfoldingPrism object is sufficiently close to a unit matrix

    Args:
        prism (ase.calculators.lammps.Prism): Prism object to check
        rtol (float): relative precision for numpy.isclose()
        atol (float): absolute precision for numpy.isclose()

    Returns:
        boolean: True or False
    """
    return np.isclose(prism.rot_mat, np.eye(3), rtol=rtol, atol=atol).all()


def get_species_symbols(structure):
    return np.array(sorted(structure.symbols.indices().keys()))


def get_species_indices_dict(structure):
    return {el: i for i, el in enumerate(sorted(structure.symbols.indices().keys()))}


def get_structure_indices(structure):
    element_indices_dict = get_species_indices_dict(structure=structure)
    elements = np.array(structure.get_chemical_symbols())
    indices = elements.copy()
    for k, v in element_indices_dict.items():
        indices[elements == k] = v
    return indices.astype(int)


def get_lammps_indicies_from_ase_indices(indices, structure, el_eam_lst):
    el_struct_lst = get_species_symbols(structure=structure)
    el_pot_dict = {
        el_eam: id_eam + 1
        for id_eam, el_eam in enumerate(el_eam_lst)
        if el_eam in el_struct_lst
    }
    ind_translate_dict = {
        i: el_pot_dict[el]
        for i, el in enumerate(sorted(structure.symbols.indices().keys()))
    }
    elem_all = indices.copy()
    for k, v in ind_translate_dict.items():
        elem_all[indices == k] = v
    return elem_all.astype(int)


def get_lammps_indicies_from_ase_structure(structure, el_eam_lst):
    el_struct_lst = get_species_symbols(structure=structure)
    el_pot_dict = {
        el_eam: id_eam + 1
        for id_eam, el_eam in enumerate(el_eam_lst)
        if el_eam in el_struct_lst
    }
    symbols_lst = np.array(structure.get_chemical_symbols())
    elem_all = symbols_lst.copy()
    for k, v in el_pot_dict.items():
        elem_all[symbols_lst == k] = v
    return elem_all.astype(int)


def get_fixed_atom_boolean_vector(structure):
    fixed_atom_vector = np.array([[False, False, False]] * len(structure))
    for c in structure.constraints:
        c_dict = c.todict()
        if c_dict["name"] == "FixAtoms":
            fixed_atom_vector[c_dict["kwargs"]["indices"]] = [True, True, True]
        elif c_dict["name"] == "FixedPlane":
            if all(np.isin(c_dict["kwargs"]["direction"], [0, 1])):
                if "indices" in c_dict["kwargs"].keys():
                    fixed_atom_vector[c_dict["kwargs"]["indices"]] = np.array(
                        c_dict["kwargs"]["direction"]
                    ).astype(bool)
                elif "a" in c_dict["kwargs"].keys():
                    fixed_atom_vector[c_dict["kwargs"]["a"]] = np.array(
                        c_dict["kwargs"]["direction"]
                    ).astype(bool)
            else:
                raise ValueError(
                    "Currently the directions are limited to [1, 0, 0], [1, 1, 0], [1, 1, 1] and its permutations."
                )
        else:
            raise ValueError("Only FixAtoms and FixedPlane are currently supported. ")
    return fixed_atom_vector


def set_selective_dynamics(structure, calc_md):
    control_dict = {}
    if len(structure.constraints) > 0:
        sel_dyn = get_fixed_atom_boolean_vector(structure=structure)
        # Enter loop only if constraints present
        if len(np.argwhere(np.any(sel_dyn, axis=1)).flatten()) != 0:
            all_indices = np.arange(len(structure), dtype=int)
            constraint_xyz = np.argwhere(np.all(sel_dyn, axis=1)).flatten()
            not_constrained_xyz = np.setdiff1d(all_indices, constraint_xyz)
            # LAMMPS starts counting from 1
            constraint_xyz += 1
            ind_x = np.argwhere(sel_dyn[not_constrained_xyz, 0]).flatten()
            ind_y = np.argwhere(sel_dyn[not_constrained_xyz, 1]).flatten()
            ind_z = np.argwhere(sel_dyn[not_constrained_xyz, 2]).flatten()
            constraint_xy = not_constrained_xyz[np.intersect1d(ind_x, ind_y)] + 1
            constraint_yz = not_constrained_xyz[np.intersect1d(ind_y, ind_z)] + 1
            constraint_zx = not_constrained_xyz[np.intersect1d(ind_z, ind_x)] + 1
            constraint_x = (
                not_constrained_xyz[np.setdiff1d(np.setdiff1d(ind_x, ind_y), ind_z)] + 1
            )
            constraint_y = (
                not_constrained_xyz[np.setdiff1d(np.setdiff1d(ind_y, ind_z), ind_x)] + 1
            )
            constraint_z = (
                not_constrained_xyz[np.setdiff1d(np.setdiff1d(ind_z, ind_x), ind_y)] + 1
            )
            control_dict = {}
            if len(constraint_xyz) > 0:
                control_dict["group constraintxyz"] = "id " + " ".join(
                    [str(ind) for ind in constraint_xyz]
                )
                control_dict["fix constraintxyz"] = "constraintxyz setforce 0.0 0.0 0.0"
                if calc_md:
                    control_dict["velocity constraintxyz"] = "set 0.0 0.0 0.0"
            if len(constraint_xy) > 0:
                control_dict["group constraintxy"] = "id " + " ".join(
                    [str(ind) for ind in constraint_xy]
                )
                control_dict["fix constraintxy"] = "constraintxy setforce 0.0 0.0 NULL"
                if calc_md:
                    control_dict["velocity constraintxy"] = "set 0.0 0.0 NULL"
            if len(constraint_yz) > 0:
                control_dict["group constraintyz"] = "id " + " ".join(
                    [str(ind) for ind in constraint_yz]
                )
                control_dict["fix constraintyz"] = "constraintyz setforce NULL 0.0 0.0"
                if calc_md:
                    control_dict["velocity constraintyz"] = "set NULL 0.0 0.0"
            if len(constraint_zx) > 0:
                control_dict["group constraintxz"] = "id " + " ".join(
                    [str(ind) for ind in constraint_zx]
                )
                control_dict["fix constraintxz"] = "constraintxz setforce 0.0 NULL 0.0"
                if calc_md:
                    control_dict["velocity constraintxz"] = "set 0.0 NULL 0.0"
            if len(constraint_x) > 0:
                control_dict["group constraintx"] = "id " + " ".join(
                    [str(ind) for ind in constraint_x]
                )
                control_dict["fix constraintx"] = "constraintx setforce 0.0 NULL NULL"
                if calc_md:
                    control_dict["velocity constraintx"] = "set 0.0 NULL NULL"
            if len(constraint_y) > 0:
                control_dict["group constrainty"] = "id " + " ".join(
                    [str(ind) for ind in constraint_y]
                )
                control_dict["fix constrainty"] = "constrainty setforce NULL 0.0 NULL"
                if calc_md:
                    control_dict["velocity constrainty"] = "set NULL 0.0 NULL"
            if len(constraint_z) > 0:
                control_dict["group constraintz"] = "id " + " ".join(
                    [str(ind) for ind in constraint_z]
                )
                control_dict["fix constraintz"] = "constraintz setforce NULL NULL 0.0"
                if calc_md:
                    control_dict["velocity constraintz"] = "set NULL NULL 0.0"
    return control_dict
