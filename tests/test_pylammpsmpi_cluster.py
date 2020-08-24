# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import unittest
import os
import numpy as np
from pylammpsmpi import LammpsLibrary
from dask.distributed import Client, LocalCluster


class TestLocalLammpsLibrary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        execution_path = os.path.dirname(os.path.abspath(__file__))
        cluster = LocalCluster(n_workers=1, threads_per_worker=2)
        client = Client(cluster)
        cls.lmp = LammpsLibrary(cores=2, mode='dask', client=client)
        cls.lmp.file(os.path.join(execution_path, "in.simple"))

    @classmethod
    def tearDownClass(cls):
        cls.lmp.close()

    def test_extract_atom(self):
        f = self.lmp.extract_atom("f")
        self.assertEqual(len(f), 256)
        self.assertEqual(np.round(f[0][0], 2), -0.26)

        ids = self.lmp.extract_atom("id")
        self.assertEqual(len(ids), 256)

    def test_gather_atoms(self):
        f = self.lmp.gather_atoms("f")
        self.assertEqual(len(f), 256)
        # this checks if info was gathered from
        # all processors
        self.assertEqual(np.round(f[-22][0], 2), 0.31)

        ids = self.lmp.extract_atom("id")
        self.assertEqual(len(ids), 256)

    def test_extract_box(self):
        box = self.lmp.extract_box()
        self.assertEqual(len(box), 7)

        self.assertEqual(box[0][0], 0.0)
        self.assertEqual(np.round(box[1][0], 2), 6.72)

        self.lmp.reset_box([0.0,0.0,0.0], [8.0,8.0,8.0], 0.0,0.0,0.0)
        box = self.lmp.extract_box()
        self.assertEqual(box[0][0], 0.0)
        self.assertEqual(np.round(box[1][0], 2), 8.0)

    def test_extract_fix(self):
        x = self.lmp.extract_fix("2", 0, 1, 1)
        self.assertEqual(np.round(x, 2), -2.61)

    def test_extract_variable(self):
        x = self.lmp.extract_variable("tt", "all", 0)
        self.assertEqual(np.round(x, 2), 1.13)

        x = self.lmp.extract_variable("fx", "all", 1)
        self.assertEqual(len(x), 128)
        self.assertEqual(np.round(x[0], 2), -0.26)

    def test_scatter_atoms(self):
        f = self.lmp.gather_atoms("f")
        val = np.random.randint(0, 100)
        f[1][0] = val
        self.lmp.scatter_atoms("f", f)
        f1 = self.lmp.gather_atoms("f")
        self.assertEqual(f1[1][0], val)

        f = self.lmp.gather_atoms("f", ids=[1,2])
        val = np.random.randint(0, 100)
        f[1][1] = val
        self.lmp.scatter_atoms("f", f, ids=[1,2])
        f1 = self.lmp.gather_atoms("f", ids=[1,2])
        self.assertEqual(f1[1][1], val)


if __name__ == "__main__":
    unittest.main()

