import logging
import unittest

from ase.atoms import Atoms
from ase.build import bulk
from ase.calculators.lammps.coordinatetransform import Prism
from ase.constraints import FixAtoms, FixedPlane, FixCom
import numpy as np

from pylammpsmpi import LammpsASELibrary, LammpsLibrary
from pylammpsmpi.wrapper.ase import (
    cell_is_skewed,
    get_species_symbols,
    get_structure_indices,
    get_lammps_indicies_from_ase_structure,
    set_selective_dynamics,
)


class TestLammpsASELibrary(unittest.TestCase):
    def test_static(self):
        lmp = LammpsASELibrary(
            working_directory=None,
            cores=1,
            comm=None,
            logger=logging.getLogger("TestStaticLogger"),
            log_file=None,
            library=LammpsLibrary(cores=2, mode="local"),
            disable_log_file=True,
        )
        structure = bulk("Al", cubic=True).repeat([2, 2, 2])
        lmp.interactive_structure_setter(
            structure=structure,
            units="lj",
            dimension=3,
            boundary=" ".join(["p" if coord else "f" for coord in structure.pbc]),
            atom_style="atomic",
            el_eam_lst=["Al"],
            calc_md=False,
        )
        lmp.interactive_lib_command("pair_style lj/cut 6.0")
        lmp.interactive_lib_command("pair_coeff 1 1 1.0 1.0 4.04")
        lmp.interactive_lib_command(
            command="thermo_style custom step temp pe etotal pxx pxy pxz pyy pyz pzz vol"
        )
        lmp.interactive_lib_command(command="thermo_modify format float %20.15g")
        lmp.interactive_lib_command("run 0")
        self.assertTrue(
            np.all(np.isclose(lmp.interactive_cells_getter(), structure.cell.array))
        )
        self.assertTrue(
            np.isclose(lmp.interactive_energy_pot_getter(), -0.04342932384411341)
        )
        self.assertTrue(
            np.isclose(lmp.interactive_energy_tot_getter(), -0.04342932384411341)
        )
        self.assertTrue(np.isclose(np.sum(lmp.interactive_forces_getter()), 0.0))
        self.assertTrue(np.isclose(lmp.interactive_volume_getter(), 531.4409999999999))
        self.assertTrue(
            np.all(lmp.interactive_indices_getter() == [1] * len(structure))
        )
        self.assertEqual(lmp.interactive_steps_getter(), 0)
        self.assertEqual(lmp.interactive_temperatures_getter(), 0)
        self.assertTrue(
            np.isclose(
                np.sum(lmp.interactive_pressures_getter()), -0.015661731917941832
            )
        )
        self.assertEqual(np.sum(lmp.interactive_velocities_getter()), 0.0)
        self.assertTrue(np.isclose(np.sum(lmp.interactive_positions_getter()), 291.6))
        lmp.interactive_cells_setter(cell=1.01 * structure.cell.array)
        lmp.interactive_lib_command("run 0")
        self.assertTrue(
            np.all(
                np.isclose(lmp.interactive_cells_getter(), 1.01 * structure.cell.array)
            )
        )
        lmp.close()

    def test_small_displacement(self):
        lmp = LammpsASELibrary(
            working_directory=None,
            cores=1,
            comm=None,
            logger=logging.getLogger("TestStaticLogger"),
            log_file=None,
            library=LammpsLibrary(cores=2, mode="local"),
            disable_log_file=True,
        )
        structure = bulk("Al", cubic=True)
        lmp.interactive_structure_setter(
            structure=structure,
            units="lj",
            dimension=3,
            boundary=" ".join(["p" if coord else "f" for coord in structure.pbc]),
            atom_style="atomic",
            el_eam_lst=["Al"],
            calc_md=False,
        )
        lmp.interactive_lib_command("pair_style lj/cut 6.0")
        lmp.interactive_lib_command("pair_coeff 1 1 1.0 1.0 4.04")
        lmp.interactive_lib_command(
            command="thermo_style custom step temp pe etotal pxx pxy pxz pyy pyz pzz vol"
        )
        lmp.interactive_lib_command(command="thermo_modify format float %20.15g")
        lmp.interactive_lib_command("run 0")
        self.assertTrue(
            np.all(np.isclose(lmp.interactive_positions_getter(), structure.positions))
        )
        self.assertTrue(
            np.isclose(lmp.interactive_energy_pot_getter(), -0.04342932384411341)
        )
        positions = structure.positions.copy()
        positions[0] += np.array([0.1, 0.1, 0.1])
        lmp.interactive_positions_setter(positions=positions)
        lmp.interactive_lib_command("run 0")
        self.assertTrue(
            np.isclose(lmp.interactive_energy_pot_getter(), -0.043829529274767506)
        )
        self.assertTrue(
            np.all(np.isclose(lmp.interactive_positions_getter(), positions))
        )

    def test_small_displacement_skewed(self):
        lmp = LammpsASELibrary(
            working_directory=None,
            cores=1,
            comm=None,
            logger=logging.getLogger("TestStaticLogger"),
            log_file=None,
            library=LammpsLibrary(cores=2, mode="local"),
            disable_log_file=True,
        )
        structure = bulk("Al").repeat([2, 2, 2])
        lmp.interactive_structure_setter(
            structure=structure,
            units="lj",
            dimension=3,
            boundary=" ".join(["p" if coord else "f" for coord in structure.pbc]),
            atom_style="atomic",
            el_eam_lst=["Al"],
            calc_md=False,
        )
        lmp.interactive_lib_command("pair_style lj/cut 6.0")
        lmp.interactive_lib_command("pair_coeff 1 1 1.0 1.0 4.04")
        lmp.interactive_lib_command(
            command="thermo_style custom step temp pe etotal pxx pxy pxz pyy pyz pzz vol"
        )
        lmp.interactive_lib_command(command="thermo_modify format float %20.15g")
        lmp.interactive_lib_command("run 0")
        self.assertTrue(
            np.all(np.isclose(lmp.interactive_positions_getter(), structure.positions))
        )
        self.assertTrue(
            np.isclose(lmp.interactive_energy_pot_getter(), -0.04342932384411341)
        )
        positions = structure.positions.copy()
        positions[0] += np.array([0.1, 0.1, 0.1])
        lmp.interactive_positions_setter(positions=positions)
        lmp.interactive_lib_command("run 0")
        self.assertTrue(
            np.isclose(lmp.interactive_energy_pot_getter(), -0.0440227281853079)
        )
        self.assertTrue(
            np.all(np.isclose(lmp.interactive_positions_getter(), positions))
        )

    def test_static_with_statement(self):
        structure = bulk("Al").repeat([2, 2, 2])
        with LammpsASELibrary(
            working_directory=None,
            cores=2,
            comm=None,
            logger=None,
            log_file=None,
            library=None,
            disable_log_file=True,
        ) as lmp:
            lmp.interactive_structure_setter(
                structure=structure,
                units="lj",
                dimension=3,
                boundary=" ".join(["p" if coord else "f" for coord in structure.pbc]),
                atom_style="atomic",
                el_eam_lst=["Al"],
                calc_md=False,
            )
            lmp.interactive_lib_command("pair_style lj/cut 6.0")
            lmp.interactive_lib_command("pair_coeff 1 1 1.0 1.0 4.04")
            lmp.interactive_lib_command(
                command="thermo_style custom step temp pe etotal pxx pxy pxz pyy pyz pzz vol"
            )
            lmp.interactive_lib_command(command="thermo_modify format float %20.15g")
            lmp.interactive_lib_command("run 0")
            self.assertTrue(
                np.isclose(lmp.interactive_energy_pot_getter(), -0.04342932384411333)
            )
            self.assertTrue(
                np.isclose(lmp.interactive_energy_tot_getter(), -0.04342932384411333)
            )
            self.assertTrue(np.isclose(np.sum(lmp.interactive_forces_getter()), 0.0))
            self.assertTrue(
                np.isclose(lmp.interactive_volume_getter(), 132.86024999999998)
            )
            self.assertTrue(
                np.all(lmp.interactive_indices_getter() == [1] * len(structure))
            )
            self.assertEqual(lmp.interactive_steps_getter(), 0)
            self.assertEqual(lmp.interactive_temperatures_getter(), 0)
            self.assertTrue(
                np.isclose(
                    np.sum(lmp.interactive_pressures_getter()), -0.015661731917941825
                )
            )
            self.assertEqual(np.sum(lmp.interactive_velocities_getter()), 0.0)


class TestASEHelperFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.structure_skewed = bulk("Al").repeat([2, 2, 2])
        cls.structure_cubic = bulk("Al", cubic=True).repeat([2, 2, 2])
        cls.structure_skewed_mix = Atoms(
            symbols=["Al", "Al", "Al", "Al", "Au", "Au"],
            positions=[
                [-1.25120847, 3.13757959, 1.09837453],
                [3.23128782, 0.98516899, 3.33835018],
                [1.77212851, 1.69333354, 1.10604864],
                [0.20795084, 2.42941504, 3.33067607],
                [4.6655973, 0.31115683, 1.11420958],
                [-2.68551794, 3.81159175, 3.32251513],
            ],
            cell=[
                [5.3322323737169475, -0.0016906521100747, 8.4534718925e-06],
                [-3.3521530195865985, 4.14672704356616, -0.0207341535846293],
                [0.0, -0.0222878092490574, 4.457450407979247],
            ],
        )

    def test_get_species_symbols(self):
        self.assertEqual(get_species_symbols(structure=self.structure_cubic), ["Al"])

    def test_get_structure_indices(self):
        indicies = get_structure_indices(structure=self.structure_cubic)
        self.assertEqual(len(indicies), len(self.structure_cubic))
        self.assertEqual(len(set(indicies)), 1)
        self.assertEqual(set(indicies), {0})

    def test_cell_is_skewed(self):
        self.assertTrue(cell_is_skewed(cell=self.structure_skewed.cell))
        self.assertFalse(cell_is_skewed(cell=self.structure_cubic.cell))

    def test_get_lammps_indicies_from_ase_structure(self):
        indicies = get_lammps_indicies_from_ase_structure(
            structure=self.structure_cubic, el_eam_lst=["Al", "H"]
        )
        self.assertEqual(len(indicies), len(self.structure_cubic))
        self.assertEqual(len(set(indicies)), 1)
        self.assertEqual(set(indicies), {1})

    def test_unfolding_prism_cubic(self):
        prism = Prism(self.structure_cubic.cell.array)
        self.assertTrue(
            np.all(np.isclose(prism.get_lammps_prism(), [8.1, 8.1, 8.1, 0.0, 0.0, 0.0]))
        )
        self.assertTrue(
            np.all(
                np.isclose(
                    prism.vector_to_lammps([[1.0, 1.0, 1.0]]),
                    np.array([1.0, 1.0, 1.0]),
                )
            )
        )

    def test_unfolding_prism_skewed(self):
        prism = Prism(self.structure_skewed.cell.array)
        self.assertTrue(
            np.all(
                np.isclose(
                    prism.get_lammps_prism(),
                    [
                        5.7275649276,
                        4.9602167291,
                        4.6765371804,
                        2.8637824638,
                        2.8637824638,
                        1.6534055764,
                    ],
                )
            )
        )
        self.assertTrue(
            np.all(
                np.isclose(
                    prism.vector_to_lammps([[1.0, 1.0, 1.0]]),
                    np.array([1.41421356, 0.81649658, 0.57735027]),
                )
            )
        )

    def test_folding(self):
        prism = Prism(self.structure_skewed_mix.cell.array)
        xhi, yhi, zhi, xy, xz, yz = prism.get_lammps_prism()
        cell_new = [[xhi, 0, 0], [xy, yhi, 0], [xz, yz, zhi]]
        cell_old = prism.vector_to_ase(cell_new)
        self.assertTrue(
            np.all(np.isclose(self.structure_skewed_mix.cell.array, cell_old))
        )


class TestConstraints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        structure = bulk("Cu", cubic=True)
        structure.symbols[2:] = "Al"
        cls.structure = structure

    def test_selective_dynamics_mixed_calcmd(self):
        atoms = self.structure.copy()
        c1 = FixAtoms(indices=[atom.index for atom in atoms if atom.symbol == "Cu"])
        c2 = FixedPlane(
            [atom.index for atom in atoms if atom.symbol == "Al"],
            [1, 0, 0],
        )
        atoms.set_constraint([c1, c2])
        control_dict = set_selective_dynamics(structure=atoms, calc_md=True)
        self.assertEqual(len(control_dict), 6)
        self.assertTrue(control_dict["group constraintxyz"], "id 1 2")
        self.assertTrue(
            control_dict["fix constraintxyz"], "constraintxyz setforce 0.0 0.0 0.0"
        )
        self.assertTrue(control_dict["velocity constraintxyz"], "set 0.0 0.0 0.0")
        self.assertTrue(control_dict["group constraintx"], "id 3 4")
        self.assertTrue(
            control_dict["fix constraintx"], "constraintx setforce 0.0 NULL NULL"
        )
        self.assertTrue(control_dict["velocity constraintx"], "set 0.0 NULL NULL")

    def test_selective_dynamics_mixed(self):
        atoms = self.structure.copy()
        c1 = FixAtoms(indices=[atom.index for atom in atoms if atom.symbol == "Cu"])
        c2 = FixedPlane(
            [atom.index for atom in atoms if atom.symbol == "Al"],
            [1, 0, 0],
        )
        atoms.set_constraint([c1, c2])
        control_dict = set_selective_dynamics(structure=atoms, calc_md=False)
        self.assertEqual(len(control_dict), 4)
        self.assertTrue(control_dict["group constraintxyz"], "id 1 2")
        self.assertTrue(
            control_dict["fix constraintxyz"], "constraintxyz setforce 0.0 0.0 0.0"
        )
        self.assertTrue(control_dict["group constraintx"], "id 3 4")
        self.assertTrue(
            control_dict["fix constraintx"], "constraintx setforce 0.0 NULL NULL"
        )

    def test_selective_dynamics_single_fix(self):
        atoms = self.structure.copy()
        c1 = FixAtoms(indices=[atom.index for atom in atoms if atom.symbol == "Cu"])
        atoms.set_constraint(c1)
        control_dict = set_selective_dynamics(structure=atoms, calc_md=False)
        self.assertEqual(len(control_dict), 2)
        self.assertTrue(control_dict["group constraintxyz"], "id 1 2")
        self.assertTrue(
            control_dict["fix constraintxyz"], "constraintxyz setforce 0.0 0.0 0.0"
        )

    def test_selective_dynamics_errors(self):
        atoms = self.structure.copy()
        atoms.set_constraint(FixCom())
        with self.assertRaises(ValueError):
            set_selective_dynamics(structure=atoms, calc_md=False)

    def test_selective_dynamics_wrong_plane(self):
        atoms = self.structure.copy()
        atoms.set_constraint(
            FixedPlane(
                [atom.index for atom in atoms if atom.symbol == "Al"],
                [2, 1, 0],
            )
        )
        with self.assertRaises(ValueError):
            set_selective_dynamics(structure=atoms, calc_md=False)


class TestBinary(unittest.TestCase):
    def setUp(self):
        self.result = [-0.04678345, -0.96708929, -2.625]
        bulk_al = bulk("Al", a=4.0, cubic=True).repeat([2, 2, 2])
        bulk_au = bulk("Au", a=4.0, cubic=True).repeat([2, 2, 2])
        bulk_mix = bulk("Al", a=4.0, cubic=True).repeat([2, 2, 2])
        chemical_symbol_lst = np.array(bulk_mix.get_chemical_symbols())
        chemical_symbol_lst[: int(len(chemical_symbol_lst) / 2) - 1] = "Au"
        bulk_mix.set_chemical_symbols(chemical_symbol_lst)
        self.structure_lst = [bulk_al, bulk_mix, bulk_au]

    @staticmethod
    def setup_job(lmp_instance):
        lmp_instance.interactive_lib_command("pair_style lj/cut 6.0")
        lmp_instance.interactive_lib_command("pair_coeff 1 1 1.0 1.0 4.0")
        lmp_instance.interactive_lib_command("pair_coeff 1 2 1.0 1.5 4.0")
        lmp_instance.interactive_lib_command("pair_coeff 2 2 1.0 2.0 4.0")
        lmp_instance.interactive_lib_command(
            command="thermo_style custom step temp pe etotal pxx pxy pxz pyy pyz pzz vol"
        )
        lmp_instance.interactive_lib_command(
            command="thermo_modify format float %20.15g"
        )

    def test_individual_calculation(self):
        energy_lst = []
        for structure in self.structure_lst:
            lmp = LammpsASELibrary(
                working_directory=None,
                cores=1,
                comm=None,
                logger=logging.getLogger("TestStaticLogger"),
                log_file=None,
                library=LammpsLibrary(cores=2, mode="local"),
                disable_log_file=True,
            )
            lmp.interactive_structure_setter(
                structure=structure,
                units="lj",
                dimension=3,
                boundary=" ".join(["p" if coord else "f" for coord in structure.pbc]),
                atom_style="atomic",
                el_eam_lst=["Al", "Au"],
                calc_md=False,
            )
            self.setup_job(lmp_instance=lmp)
            lmp.interactive_lib_command("run 0")
            energy_lst.append(lmp.interactive_energy_pot_getter())
        self.assertTrue(np.allclose(self.result, energy_lst))

    def test_interactive_calculation(self):
        energy_lst = []
        lmp = LammpsASELibrary(
            working_directory=None,
            cores=1,
            comm=None,
            logger=logging.getLogger("TestStaticLogger"),
            log_file=None,
            library=LammpsLibrary(cores=2, mode="local"),
            disable_log_file=True,
        )
        for structure in self.structure_lst:
            lmp.interactive_structure_setter(
                structure=structure,
                units="lj",
                dimension=3,
                boundary=" ".join(["p" if coord else "f" for coord in structure.pbc]),
                atom_style="atomic",
                el_eam_lst=["Al", "Au"],
                calc_md=False,
            )
            self.setup_job(lmp_instance=lmp)
            lmp.interactive_lib_command("run 0")
            energy_lst.append(lmp.interactive_energy_pot_getter())
        self.assertTrue(np.allclose(self.result, energy_lst))

    def test_interactive_calculation_inverse(self):
        energy_lst = []
        lmp = LammpsASELibrary(
            working_directory=None,
            cores=1,
            comm=None,
            logger=logging.getLogger("TestStaticLogger"),
            log_file=None,
            library=LammpsLibrary(cores=2, mode="local"),
            disable_log_file=True,
        )
        for structure in self.structure_lst[::-1]:
            lmp.interactive_structure_setter(
                structure=structure,
                units="lj",
                dimension=3,
                boundary=" ".join(["p" if coord else "f" for coord in structure.pbc]),
                atom_style="atomic",
                el_eam_lst=["Al", "Au"],
                calc_md=False,
            )
            self.setup_job(lmp_instance=lmp)
            lmp.interactive_lib_command("run 0")
            energy_lst.append(lmp.interactive_energy_pot_getter())
        self.assertTrue(np.allclose(self.result, energy_lst[::-1]))
