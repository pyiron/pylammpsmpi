# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH
# Distributed under the terms of "New BSD License", see the LICENSE file.

import numpy as np


def external_callback_without_caller(caller, ntimestep, nlocal, tag, x, f):
    assert caller is None, "caller is not None"
    assert isinstance(ntimestep, int), "ntimestep is not int"
    assert isinstance(nlocal, int), "nlocal is not int"
    assert isinstance(tag[0], (int, np.integer)), "tag[0] not int-like"
    assert len(x[0]) == 3, "x[0] must have length 3"
    assert len(f[0]) == 3, "f[0] must have length 3"


def external_callback_with_caller(caller, ntimestep, nlocal, tag, x, f):
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


class HelperClass:
    """
    Helper object passed through to the external callback.
    """

    def __init__(self, token: int):
        self.token = token
