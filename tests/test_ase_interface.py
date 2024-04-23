import unittest

from ase.build import bulk
import numpy as np

from pylammpsmpi import LammpsASELibrary, LammpsLibrary


class TestLammpsASELibrary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lmp = LammpsASELibrary(
            working_directory=None,
            cores=1,
            comm=None,
            logger=None,
            log_file=None,
            library=LammpsLibrary(cores=2, mode='local'),
            diable_log_file=True,
        )

    @classmethod
    def tearDownClass(cls):
        cls.lmp.close()

    def test_static(self):
        structure = bulk("Al", cubic=True).repeat([2, 2, 2])
        self.lmp.interactive_lib_command(command="units	lj")
        self.lmp.interactive_lib_command(command="atom_style atomic")
        self.lmp.interactive_lib_command(command="atom_modify map array")
        self.lmp.interactive_structure_setter(
            structure=structure,
            units="lj",
            dimension=3,
            boundary=" ".join(["p" if coord else "f" for coord in structure.pbc]),
            atom_style="atomic",
            el_eam_lst=["Al"],
            calc_md=False,
        )
        self.lmp.interactive_lib_command("pair_style lj/cut 6.0")
        self.lmp.interactive_lib_command("pair_coeff 1 1 1.0 1.0 4.04")
        self.lmp.interactive_lib_command("run 0")
        self.assertTrue(np.all(np.isclose(self.lmp.interactive_cells_getter(), structure.cell.array)))
        self.assertEqual(float(self.lmp.interactive_energy_pot_getter()), -0.04342932384411341)
        self.assertEqual(float(self.lmp.interactive_energy_tot_getter()), -0.04342932384411341)
        self.assertTrue(np.isclose(np.sum(self.lmp.interactive_forces_getter()), 0.0))
