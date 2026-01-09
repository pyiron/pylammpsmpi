import unittest
import os
import numpy as np
from lammps import lammps
from pylammpsmpi.mpi.lmpmpi import select_cmd


class TestMpiBackendExtended(unittest.TestCase):
    def setUp(self):
        self.execution_path = os.path.dirname(os.path.abspath(__file__))
        self.lammps_file = os.path.join(self.execution_path, "in.simple")
        self.lmp = lammps()
        select_cmd("get_file")(job=self.lmp, funct_args=[self.lammps_file])

    def tearDown(self):
        self.lmp.close()

    def test_gather_atoms_invalid(self):
        f = select_cmd("gather_atoms")(job=self.lmp, funct_args=["error"])
        self.assertEqual(len(f), 0)

    def test_extract_compute_style_error(self):
        with self.assertRaises(ValueError):
            select_cmd("extract_compute")(job=self.lmp, funct_args=["1", 2, 0, 0, 0])

    def test_get_file(self):
        lmp = lammps()
        ret = select_cmd("get_file")(job=lmp, funct_args=[self.lammps_file])
        self.assertEqual(ret, 1)
        lmp.close()

    def test_commands(self):
        ret = select_cmd("commands_list")(job=self.lmp, funct_args=["run 0", "run 1"])
        self.assertEqual(ret, 1)
        ret = select_cmd("commands_string")(job=self.lmp, funct_args="run 0\nrun 1")
        self.assertEqual(ret, 1)

    def test_set_variable_deprecated(self):
        ret = select_cmd("set_variable")(job=self.lmp, funct_args=["tt", 1.0])
        self.assertEqual(ret, -1)

    def test_variable_setting_with_command(self):
        select_cmd("command")(job=self.lmp, funct_args="variable my_var equal 1.0")
        select_cmd("command")(job=self.lmp, funct_args="variable my_var equal 2.0")
        x = select_cmd("extract_variable")(
            job=self.lmp, funct_args=["my_var", "all", 0]
        )
        self.assertEqual(x, 2.0)

    def test_reset_box(self):
        lmp = lammps()
        select_cmd("command")(job=lmp, funct_args="units lj")
        select_cmd("command")(job=lmp, funct_args="dimension 3")
        select_cmd("command")(job=lmp, funct_args="boundary p p p")
        select_cmd("command")(job=lmp, funct_args="atom_style atomic")
        select_cmd("command")(job=lmp, funct_args="region mybox block 0 10 0 10 0 10")
        select_cmd("command")(job=lmp, funct_args="create_box 1 mybox")
        box = select_cmd("extract_box")(job=lmp, funct_args=[])
        box[0][0] = -1.0
        box[1][0] = 7.0
        ret = select_cmd("reset_box")(
            job=lmp, funct_args=[box[0], box[1], box[2], box[3], box[4]]
        )
        self.assertEqual(ret, 1)
        new_box = select_cmd("extract_box")(job=lmp, funct_args=[])
        self.assertEqual(new_box[0][0], -1.0)
        self.assertEqual(new_box[1][0], 7.0)
        lmp.close()

    def test_command(self):
        ret = select_cmd("command")(job=self.lmp, funct_args="run 0")
        self.assertEqual(ret, 1)

    def test_extract_variable_error(self):
        v = select_cmd("extract_variable")(
            job=self.lmp, funct_args=["var_not_exist", "all", 0]
        )
        self.assertIsNone(v)

    def test_gather_atoms_concat(self):
        f = select_cmd("gather_atoms_concat")(job=self.lmp, funct_args=["f"])
        self.assertEqual(len(f), 256)
        self.assertEqual(np.round(f[-22][0], 2), 0.31)
        f = select_cmd("gather_atoms_concat")(job=self.lmp, funct_args=["error"])
        self.assertEqual(len(f), 0)

    def test_gather_atoms_subset(self):
        ids = [1, 5, 10]
        f = select_cmd("gather_atoms_subset")(
            job=self.lmp, funct_args=["f", len(ids), ids]
        )
        self.assertEqual(len(f), len(ids))
        f = select_cmd("gather_atoms_subset")(job=self.lmp, funct_args=["error", 0, []])
        self.assertEqual(len(f), 0)

    def test_scatter_atoms_subset(self):
        ids = [1, 5, 10]
        f = select_cmd("gather_atoms_subset")(
            job=self.lmp, funct_args=["f", len(ids), ids]
        )
        val = np.random.randint(0, 100)
        f[1][0] = val
        ret = select_cmd("scatter_atoms_subset")(
            job=self.lmp, funct_args=["f", f, len(ids), ids]
        )
        self.assertEqual(ret, 1)
        f1 = select_cmd("gather_atoms_subset")(
            job=self.lmp, funct_args=["f", len(ids), ids]
        )
        self.assertEqual(f1[1][0], val)

    def test_scatter_atoms_integer(self):
        types = select_cmd("gather_atoms")(job=self.lmp, funct_args=["type"])
        val = 2
        types[10] = val
        ret = select_cmd("scatter_atoms")(
            job=self.lmp, funct_args=["type", types.astype(int)]
        )
        self.assertEqual(ret, 1)
        types_new = select_cmd("gather_atoms")(job=self.lmp, funct_args=["type"])
        self.assertEqual(types_new[10], val)

    def test_neighlist(self):
        nl = select_cmd("get_neighlist")(job=self.lmp, funct_args=[0])
        self.assertIsNotNone(nl)
        pnl = select_cmd("find_pair_neighlist")(job=self.lmp, funct_args=["lj/cut"])
        self.assertIsInstance(pnl, int)
        fnl = select_cmd("find_fix_neighlist")(job=self.lmp, funct_args=["2"])
        self.assertIsInstance(fnl, int)
        cnl = select_cmd("find_compute_neighlist")(job=self.lmp, funct_args=["ke"])
        self.assertIsInstance(cnl, int)
        size = select_cmd("get_neighlist_size")(job=self.lmp, funct_args=[0])
        self.assertIsInstance(size, int)
        elem = select_cmd("get_neighlist_element_neighbors")(
            job=self.lmp, funct_args=[0, 0]
        )
        self.assertIsInstance(elem, object)

    def test_extract_atom_error(self):
        f = select_cmd("extract_atom")(job=self.lmp, funct_args=["error"])
        self.assertEqual(len(f), 0)

    def test_create_atoms(self):
        lmp = lammps()
        select_cmd("command")(job=lmp, funct_args="units lj")
        select_cmd("command")(job=lmp, funct_args="dimension 3")
        select_cmd("command")(job=lmp, funct_args="boundary p p p")
        select_cmd("command")(job=lmp, funct_args="atom_style atomic")
        select_cmd("command")(job=lmp, funct_args="region box block 0 10 0 10 0 10")
        select_cmd("command")(job=lmp, funct_args="create_box 1 box")
        ret = select_cmd("create_atoms")(
            job=lmp, funct_args=[1, [1], [1], [[1.0, 1.0, 1.0]], [0], 0]
        )
        self.assertEqual(ret, 1)
        lmp.close()

    def test_set_fix_external_callback(self):
        select_cmd("command")(
            job=self.lmp, funct_args="fix cb all external pf/callback 1 1"
        )
        callback = lambda *a, **k: None
        args_none = ["cb", callback]
        ret_none = select_cmd("set_fix_external_callback")(
            job=self.lmp, funct_args=args_none
        )
        self.assertEqual(ret_none, 1)
        args_list = ["cb", callback, ["pylammpsmpi.lammps.reference", 42, "other"]]
        ret_list = select_cmd("set_fix_external_callback")(
            job=self.lmp, funct_args=args_list
        )
        self.assertEqual(ret_list, 1)
        args_dict = ["cb", callback, {"lmp": "pylammpsmpi.lammps.reference", "val": 42}]
        ret_dict = select_cmd("set_fix_external_callback")(
            job=self.lmp, funct_args=args_dict
        )
        self.assertEqual(ret_dict, 1)
        args_lammps = ["cb", callback, "pylammpsmpi.lammps.reference"]
        ret_lammps = select_cmd("set_fix_external_callback")(
            job=self.lmp, funct_args=args_lammps
        )
        self.assertEqual(ret_lammps, 1)


if __name__ == "__main__":
    unittest.main()
