import unittest
from lammps import Lammps
from pylammpsmpi.mpi.lmpmpi import select_cmd


class TestMpiBackend(unittest.TestCase):
    def test_select_cmd(self):
        lmp = Lammps()
        self.assertTrue(
            select_cmd("get_version")(job=lmp, funct_args=[])
            in [20220623, 20230802, 20231121, 20240207, 20240627, 20240829]
        )
        lmp.close()