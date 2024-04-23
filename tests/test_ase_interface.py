import unittest

from ase.build import bulk
import numpy as np

from pylammpsmpi import LammpsASELibrary, LammpsLibrary


class TestLammpsASELibrary(unittest.TestCase):
    def test_static(self):
        lmp = LammpsASELibrary(
            working_directory=None,
            cores=1,
            comm=None,
            logger=None,
            log_file=None,
            library=LammpsLibrary(cores=2, mode='local'),
            diable_log_file=True,
        )
        structure = bulk("Al", cubic=True).repeat([2, 2, 2])
        lmp.interactive_lib_command(command="units	lj")
        lmp.interactive_lib_command(command="atom_style atomic")
        lmp.interactive_lib_command(command="atom_modify map array")
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
        lmp.interactive_lib_command(command="thermo_style custom step temp pe etotal pxx pxy pxz pyy pyz pzz vol")
        lmp.interactive_lib_command(command="thermo_modify format float %20.15g")
        lmp.interactive_lib_command("run 0")
        self.assertTrue(np.all(np.isclose(lmp.interactive_cells_getter(), structure.cell.array)))
        self.assertTrue(np.isclose(lmp.interactive_energy_pot_getter(), -0.04342932384411341))
        self.assertTrue(np.isclose(lmp.interactive_energy_tot_getter(), -0.04342932384411341))
        self.assertTrue(np.isclose(np.sum(lmp.interactive_forces_getter()), 0.0))
        self.assertTrue(np.isclose(lmp.interactive_volume_getter(), 531.4409999999999))
        self.assertTrue(np.all(lmp.interactive_indices_getter() == [1] * len(structure)))
        self.assertEqual(lmp.interactive_steps_getter(), 0)
        self.assertEqual(lmp.interactive_temperatures_getter(), 0)
        self.assertTrue(np.isclose(np.sum(lmp.interactive_pressures_getter()), -0.015661731917941832))
        lmp.close()

    def test_static_with_statement(self):
        structure = bulk("Al").repeat([2, 2, 2])
        with LammpsASELibrary(
            working_directory=None,
            cores=2,
            comm=None,
            logger=None,
            log_file=None,
            library=None,
            diable_log_file=True,
        ) as lmp:
            lmp.interactive_lib_command(command="units	lj")
            lmp.interactive_lib_command(command="atom_style atomic")
            lmp.interactive_lib_command(command="atom_modify map array")
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
            lmp.interactive_lib_command(command="thermo_style custom step temp pe etotal pxx pxy pxz pyy pyz pzz vol")
            lmp.interactive_lib_command(command="thermo_modify format float %20.15g")
            lmp.interactive_lib_command("run 0")
            self.assertTrue(np.isclose(lmp.interactive_energy_pot_getter(), -0.3083820387630098))
            self.assertTrue(np.isclose(lmp.interactive_energy_tot_getter(), -0.3083820387630098))
            self.assertTrue(np.isclose(np.sum(lmp.interactive_forces_getter()), 0.0))
            self.assertTrue(np.isclose(lmp.interactive_volume_getter(), 132.86024999999998))
            self.assertTrue(np.all(lmp.interactive_indices_getter() == [1] * len(structure)))
            self.assertEqual(lmp.interactive_steps_getter(), 0)
            self.assertEqual(lmp.interactive_temperatures_getter(), 0)
            self.assertTrue(np.isclose(np.sum(lmp.interactive_pressures_getter()), -0.00937227406237915))
