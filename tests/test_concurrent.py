import unittest
import numpy as np
import os
from pylammpsmpi import LammpsConcurrent


class TestLammpsConcurrent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.execution_path = os.path.dirname(os.path.abspath(__file__))
        cls.citation_file = os.path.join(cls.execution_path, "citations.txt")
        cls.lammps_file = os.path.join(cls.execution_path, "in.simple")
        cls.lmp = LammpsConcurrent(cores=1, cmdargs=["-cite", cls.citation_file])
        cls.lmp.file(cls.lammps_file).result()

    @classmethod
    def tearDownClass(cls):
        cls.lmp.close()

    def test_extract_atom(self):
        f = self.lmp.extract_atom("f").result()
        self.assertEqual(len(f), 256)
        self.assertEqual(np.round(f[0][0], 2), -0.26)

        ids = self.lmp.extract_atom("id").result()
        self.assertEqual(len(ids), 256)

    def test_extract_compute_global(self):
        compute_temp = self.lmp.extract_compute("1", 0, 0).result()
        self.assertIsInstance(compute_temp, float)

    def test_extract_compute_per_atom(self):
        compute_ke_atom = self.lmp.extract_compute("ke", 1, 1).result()
        self.assertEqual(len(compute_ke_atom), 256)

    def test_gather_atoms(self):
        f = self.lmp.gather_atoms("f").result()
        self.assertEqual(len(f), 256)
        # this checks if info was gathered from
        # all processors
        self.assertEqual(np.round(f[-22][0], 2), 0.31)

        ids = self.lmp.extract_atom("id").result()
        self.assertEqual(len(ids), 256)

    def test_extract_fix(self):
        x = self.lmp.extract_fix("2", 0, 1, 1).result()
        self.assertEqual(np.round(x, 2), -2.61)

    def test_extract_variable(self):
        x = self.lmp.extract_variable("tt", "all", 0).result()
        self.assertEqual(np.round(x, 2), 1.13)
        x = self.lmp.extract_variable("fx", "all", 1).result()
        self.assertEqual(len(x), 256)
        self.assertEqual(np.round(x[0], 2), -0.26)

    def test_scatter_atoms(self):
        f = self.lmp.gather_atoms("f").result()
        val = np.random.randint(0, 100)
        f[1][0] = val
        self.lmp.scatter_atoms("f", f).result()
        f1 = self.lmp.gather_atoms("f").result()
        self.assertEqual(f1[1][0], val)

        f = self.lmp.gather_atoms("f", ids=[1, 2]).result()
        val = np.random.randint(0, 100)
        f[1][1] = val
        self.lmp.scatter_atoms("f", f, ids=[1, 2]).result()
        f1 = self.lmp.gather_atoms("f", ids=[1, 2]).result()
        self.assertEqual(f1[1][1], val)

    def test_extract_box(self):
        box = self.lmp.extract_box().result()
        self.assertEqual(len(box), 7)

        self.assertEqual(box[0][0], 0.0)
        self.assertEqual(np.round(box[1][0], 2), 6.72)

        self.lmp.delete_atoms("group", "all").result()
        self.lmp.reset_box([0.0, 0.0, 0.0], [8.0, 8.0, 8.0], 0.0, 0.0, 0.0).result()
        box = self.lmp.extract_box().result()
        self.assertEqual(box[0][0], 0.0)
        self.assertEqual(np.round(box[1][0], 2), 8.0)
        self.lmp.clear()
        self.lmp.file(self.lammps_file)

    def test_cmdarg_options(self):
        self.assertTrue(os.path.isfile(self.citation_file))


if __name__ == "__main__":
    unittest.main()
