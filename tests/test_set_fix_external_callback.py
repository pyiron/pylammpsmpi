# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH
# Distributed under the terms of "New BSD License", see the LICENSE file.

import os
import unittest

from pylammpsmpi import LammpsLibrary
from pylammpsmpi.helpers.callbacks import HelperClass, external_callback


class TestSetFixExternalCallback(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.execution_path = os.path.dirname(os.path.abspath(__file__))
        cls.lammps_file = os.path.join(cls.execution_path, "in.simple")
        cls.lmp = LammpsLibrary(
            cores=2,
            oversubscribe=False,
            working_directory=cls.execution_path,
            cmdargs=None,
        )
        cls.lmp.file(cls.lammps_file)

    @classmethod
    def tearDownClass(cls):
        cls.lmp.close()

    def test_set_fix_external_callback(self):
        helper = HelperClass(token=2648)
        self.lmp.fix("cb all external pf/callback 1 1")
        self.lmp.set_fix_external_callback("cb", external_callback, [self.lmp, helper])
        self.lmp.run(0)


if __name__ == "__main__":
    unittest.main()
