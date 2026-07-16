# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import os
import sys
import unittest

import numpy as np

from pylammpsmpi import LammpsBase

try:
    import lammps.mliap  # noqa: F401

    if sys.platform == "darwin":
        _HAS_MLIAP = False
    else:
        _HAS_MLIAP = True
except ImportError:
    _HAS_MLIAP = False

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class TestLammpsBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.execution_path = os.path.dirname(os.path.abspath(__file__))
        cls.citation_file = os.path.join(cls.execution_path, "citations.txt")
        cls.lammps_file = os.path.join(cls.execution_path, "in.simple")
        cls.lmp = LammpsBase(
            cores=1,
            oversubscribe=False,
            working_directory=".",
            hostname_localhost=True,
            cmdargs=["-cite", cls.citation_file],
        )
        cls.lmp.file(cls.lammps_file)

    @classmethod
    def tearDownClass(cls):
        cls.lmp.close()

    def test_file_not_found(self):
        lmp = LammpsBase(
            cores=1,
            oversubscribe=False,
            working_directory=".",
            hostname_localhost=True,
            cmdargs=["-cite", self.citation_file],
        )
        with self.assertRaises(FileNotFoundError):
            lmp.file("file_does_not_exist.txt")
        lmp.close()

    def test_extract_atom(self):
        f = self.lmp.extract_atom("f")
        self.assertEqual(len(f), 256)
        self.assertEqual(np.round(f[0][0], 2), -0.26)

        ids = self.lmp.extract_atom("id")
        self.assertEqual(len(ids), 256)

    def test_extract_compute_global(self):
        compute_temp = self.lmp.extract_compute("1", 0, 0)
        self.assertIsInstance(compute_temp, float)

    def test_extract_compute_per_atom(self):
        compute_ke_atom = self.lmp.extract_compute("ke", 1, 1)
        self.assertEqual(len(compute_ke_atom), 256)

    def test_gather_atoms(self):
        f = self.lmp.gather_atoms("f")
        self.assertEqual(len(f), 256)
        # this checks if info was gathered from
        # all processors
        self.assertEqual(np.round(f[-22][0], 2), 0.31)

        ids = self.lmp.extract_atom("id")
        self.assertEqual(len(ids), 256)
        self.assertEqual(self.lmp.get_natoms(), 256)
        self.assertEqual(self.lmp.natoms, 256)

    def test_extract_fix(self):
        x = self.lmp.extract_fix("2", 0, 1, 1)
        self.assertEqual(np.round(x, 2), -2.61)

    def test_extract_variable(self):
        x = self.lmp.extract_variable("tt", "all", 0)
        self.assertEqual(np.round(x, 2), 1.13)
        x = self.lmp.extract_variable("fx", "all", 1)
        self.assertEqual(len(x), 256)
        self.assertEqual(np.round(x[0], 2), -0.26)

    def test_scatter_atoms(self):
        f = self.lmp.gather_atoms("f")
        val = np.random.randint(0, 100)
        f[1][0] = val
        self.lmp.scatter_atoms("f", f)
        f1 = self.lmp.gather_atoms("f")
        self.assertEqual(f1[1][0], val)

        f = self.lmp.gather_atoms("f", ids=[1, 2])
        val = np.random.randint(0, 100)
        f[1][1] = val
        self.lmp.scatter_atoms("f", f, ids=[1, 2])
        f1 = self.lmp.gather_atoms("f", ids=[1, 2])
        self.assertEqual(f1[1][1], val)

    def test_extract_box(self):
        box = self.lmp.extract_box()
        self.assertEqual(len(box), 7)

        self.assertEqual(box[0][0], 0.0)
        self.assertEqual(np.round(box[1][0], 2), 6.72)

        self.lmp.command("delete_atoms group all")
        self.lmp.reset_box([0.0, 0.0, 0.0], [8.0, 8.0, 8.0], 0.0, 0.0, 0.0)
        box = self.lmp.extract_box()
        self.assertEqual(box[0][0], 0.0)
        self.assertEqual(np.round(box[1][0], 2), 8.0)
        self.lmp.command("clear")
        self.lmp.file(self.lammps_file)

    def test_cmdarg_options(self):
        self.assertTrue(os.path.isfile(self.citation_file))

    def test_version(self):
        self.assertTrue(
            self.lmp.version
            in [20220623, 20230802, 20231121, 20240207, 20240627, 20240829, 20250722]
        )

    def test_extract_global(self):
        self.assertEqual(
            self.lmp.extract_global(name="boxhi"),
            [6.718384765530029, 6.718384765530029, 6.718384765530029],
        )
        self.assertEqual(self.lmp.extract_global(name="boxlo"), [0.0, 0.0, 0.0])

    def test_properties(self):
        self.assertEqual(self.lmp.has_exceptions, True)
        self.assertEqual(self.lmp.has_gzip_support, True)
        self.assertEqual(self.lmp.has_png_support, True)
        self.assertEqual(self.lmp.has_jpeg_support, True)
        self.assertEqual(self.lmp.has_ffmpeg_support, False)

    def test_get_thermo(self):
        self.assertEqual(float(self.lmp.get_thermo("temp")), 1.1298532212880312)

    def test_extract_setting(self):
        self.assertEqual(self.lmp.extract_setting("nlocal"), 256)
        self.assertEqual(self.lmp.extract_setting("ntypes"), 1)

    def test_command_list(self):
        self.lmp.command(["variable cmdtest equal 1", "variable cmdtest delete"])
        self.assertEqual(self.lmp.get_natoms(), 256)

    def test_command_multiline_string(self):
        self.lmp.command("variable cmdtest2 equal 2\nvariable cmdtest2 delete")
        self.assertEqual(self.lmp.get_natoms(), 256)

    def test_neighlist(self):
        idx = self.lmp.find_pair_neighlist("lj/cut")
        self.assertIsInstance(idx, int)
        self.assertGreaterEqual(idx, 0)
        size = self.lmp.get_neighlist_size(idx)
        self.assertEqual(size, 256)
        self.assertIsInstance(self.lmp.find_fix_neighlist("2"), int)
        self.assertIsInstance(self.lmp.find_compute_neighlist("ke"), int)

    @unittest.skipUnless(_HAS_MLIAP, "lammps.mliap not available")
    def test_activate_mliappy(self):
        result = self.lmp.activate_mliappy()
        self.assertIsNone(result)

    @unittest.skipUnless(
        _HAS_MLIAP and _HAS_TORCH, "lammps.mliap or torch not available"
    )
    def test_mliappy_pytorch_workflow(self):
        # Follows lammps/lammps examples/mliap/mliap_pytorch_Ta06A.py: load a
        # PyTorch SNAP model through the ML-IAP python coupling and run a few
        # steps of MD. Ta06A.mliap.descriptor/.pt are vendored from that
        # LAMMPS example (GPL-licensed, used here only as test fixtures).
        descriptor_file = os.path.join(self.execution_path, "Ta06A.mliap.descriptor")
        model_file = os.path.join(self.execution_path, "Ta06A.mliap.pytorch.model.pt")

        self.lmp.command("clear")
        try:
            self.lmp.activate_mliappy()
            self.lmp.command(
                f"""
units           metal
boundary        p p p
lattice         bcc 3.316
region          box block 0 4 0 4 0 4
create_box      1 box
create_atoms    1 box
mass 1 180.88
pair_style hybrid/overlay zbl 4 4.8 mliap model mliappy LATER descriptor sna {descriptor_file}
pair_coeff 1 1 zbl 73 73
pair_coeff * * mliap Ta
"""
            )

            model = torch.load(model_file, weights_only=False)
            self.lmp.mliappy_load_model(model)

            self.lmp.command(
                """
compute  eatom all pe/atom
compute  energy all reduce sum c_eatom
thermo_style    custom step temp epair c_energy etotal press
thermo          5
thermo_modify norm yes
timestep 0.5e-3
neighbor 1.0 bin
neigh_modify once no every 1 delay 0 check yes
velocity all create 300.0 4928459 loop geom
fix 1 all nve
run 0
"""
            )

            self.assertEqual(self.lmp.get_natoms(), 128)
            self.assertAlmostEqual(self.lmp.get_thermo("temp"), 300.0, places=4)
            self.assertAlmostEqual(self.lmp.get_thermo("pe"), -11.85157, places=4)

            self.lmp.command("run 10")
            final_temp = self.lmp.get_thermo("temp")
            self.assertTrue(np.isfinite(final_temp))
            self.assertLess(final_temp, 300.0)
        finally:
            self.lmp.command("clear")
            self.lmp.file(self.lammps_file)


if __name__ == "__main__":
    unittest.main()
