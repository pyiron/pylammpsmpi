# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH
# Distributed under the terms of "New BSD License", see the LICENSE file.

import os
import unittest
import numpy as np

from pylammpsmpi import LammpsLibrary


class HelperClass:
    """
    Helper object passed through to the external callback.
    """

    def __init__(self, token: int):
        self.token = token


def external_callback(caller, ntimestep, nlocal, tag, x, f):
    lmp, helper = caller
    assert isinstance(helper, HelperClass), "helper is not a HelperClass instance"
    assert isinstance(helper.token, int), "helper.token is not an int"
    assert isinstance(ntimestep, int), "ntimestep is not int"
    assert isinstance(nlocal, int), "nlocal is not int"
    assert isinstance(tag[0], (int, np.integer)), "tag[0] not int-like"
    assert len(x[0]) == 3, "x[0] must have length 3"
    assert len(f[0]) == 3, "f[0] must have length 3"
    pe = lmp.get_thermo("pe")
    assert isinstance(float(pe), float), "Potential energy should be float-like"


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
