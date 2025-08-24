import unittest
import os
import numpy as np
from lammps import lammps
from pylammpsmpi.mpi.lmpmpi import select_cmd


class TestMpiBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.execution_path = os.path.dirname(os.path.abspath(__file__))
        cls.lammps_file = os.path.join(cls.execution_path, "in.simple")
        cls.lmp = lammps()
        cls.lmp.file(cls.lammps_file)

    @classmethod
    def tearDownClass(cls):
        cls.lmp.close()

    def test_extract_atom(self):
        f = select_cmd("extract_atom")(job=self.lmp, funct_args=["f"])
        self.assertEqual(len(f), 256)
        self.assertEqual(np.round(f[0][0], 2), -0.26)

        ids = select_cmd("extract_atom")(job=self.lmp, funct_args=["id"])
        self.assertEqual(len(ids), 256)

    def test_extract_compute_global(self):
        compute_temp = select_cmd("extract_compute")(
            job=self.lmp, funct_args=["1", 0, 0, 0, 0]
        )
        self.assertIsInstance(compute_temp, float)

    def test_extract_compute_per_atom(self):
        compute_ke_atom = select_cmd("extract_compute")(
            job=self.lmp, funct_args=["ke", 1, 1, 256, 0]
        )
        self.assertEqual(len(compute_ke_atom), 256)

    def test_gather_atoms(self):
        f = select_cmd("gather_atoms")(job=self.lmp, funct_args=["f"])
        self.assertEqual(len(f), 256)
        # this checks if info was gathered from
        # all processors
        self.assertEqual(np.round(f[-22][0], 2), 0.31)

        ids = select_cmd("extract_atom")(job=self.lmp, funct_args=["id"])
        self.assertEqual(len(ids), 256)
        self.assertEqual(select_cmd("get_natoms")(job=self.lmp, funct_args=[]), 256)

    def test_extract_fix(self):
        x = select_cmd("extract_fix")(job=self.lmp, funct_args=["2", 0, 1, 1])
        self.assertEqual(np.round(x, 2), -2.61)

    def test_extract_variable(self):
        x = select_cmd("extract_variable")(job=self.lmp, funct_args=["tt", "all", 0])
        self.assertEqual(np.round(x, 2), 1.13)
        x = select_cmd("extract_variable")(job=self.lmp, funct_args=["fx", "all", 1])
        self.assertEqual(len(x), 256)
        self.assertEqual(np.round(x[0], 2), -0.26)

    def test_scatter_atoms(self):
        f = select_cmd("gather_atoms")(job=self.lmp, funct_args=["f"])
        val = np.random.randint(0, 100)
        f[1][0] = val
        select_cmd("scatter_atoms")(job=self.lmp, funct_args=["f", f])
        f1 = select_cmd("gather_atoms")(job=self.lmp, funct_args=["f"])
        self.assertEqual(f1[1][0], val)

        f = select_cmd("gather_atoms")(job=self.lmp, funct_args=["f", 2, [1, 2]])
        val = np.random.randint(0, 100)
        f[1][1] = val
        select_cmd("scatter_atoms")(job=self.lmp, funct_args=["f", f, 2, [1, 2]])
        f1 = select_cmd("gather_atoms")(job=self.lmp, funct_args=["f", 2, [1, 2]])
        self.assertEqual(f1[1][1], val)

    def test_extract_box(self):
        box = select_cmd("extract_box")(job=self.lmp, funct_args=[])
        self.assertEqual(len(box), 7)

        self.assertEqual(box[0][0], 0.0)
        self.assertEqual(np.round(box[1][0], 2), 6.72)

    def test_version(self):
        self.assertTrue(
            select_cmd("get_version")(job=self.lmp, funct_args=[])
            in [20220623, 20230802, 20231121, 20240207, 20240627, 20240829]
        )

    def test_extract_global(self):
        self.assertEqual(
            select_cmd("extract_global")(job=self.lmp, funct_args=["boxhi"]),
            [6.718384765530029, 6.718384765530029, 6.718384765530029],
        )
        self.assertEqual(
            select_cmd("extract_global")(job=self.lmp, funct_args=["boxlo"]),
            [0.0, 0.0, 0.0],
        )

    def test_properties(self):
        self.assertEqual(
            select_cmd("has_exceptions")(job=self.lmp, funct_args=[]), True
        )
        self.assertEqual(
            select_cmd("has_gzip_support")(job=self.lmp, funct_args=[]), True
        )
        self.assertEqual(
            select_cmd("has_png_support")(job=self.lmp, funct_args=[]), True
        )
        self.assertEqual(
            select_cmd("has_jpeg_support")(job=self.lmp, funct_args=[]), True
        )
        self.assertEqual(
            select_cmd("has_ffmpeg_support")(job=self.lmp, funct_args=[]), False
        )

    def test_get_thermo(self):
        self.assertEqual(
            float(select_cmd("get_thermo")(job=self.lmp, funct_args=["temp"])),
            1.1298532212880312,
        )
        select_cmd("command")(job=self.lmp, funct_args="run 0")
        self.assertEqual(
            float(select_cmd("get_thermo")(job=self.lmp, funct_args=["temp"])),
            1.129853221288031,
        )

    def test_installed_packages(self):
        packages = select_cmd("installed_packages")(job=self.lmp, funct_args=[])
        self.assertIsInstance(packages, list)
        self.assertIn("MANYBODY", packages)
        self.assertIn("KSPACE", packages)
        self.assertIn("MC", packages)


if __name__ == "__main__":
    unittest.main()
