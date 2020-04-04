import pytest
import os
import numpy as np
from pylammpsmpi import LammpsLibrary

lmp = LammpsLibrary(cores=2, mode='local')
lmp.file("tests/in.simple")

def test_extract_atom():

    f = lmp.extract_atom("f")
    assert len(f) == 256
    assert np.round(f[0][0], 2) == -0.26

    ids = lmp.extract_atom("id")
    assert len(ids) == 256

def test_gather_atoms():

    f = lmp.gather_atoms("f")
    assert len(f) == 256
    #this checks if info was gathered from
    #all processors
    assert np.round(f[-22][0], 2) == 0.31

    ids = lmp.extract_atom("id")
    assert len(ids) == 256

def test_extract_box():

    box = lmp.extract_box()
    assert len(box) == 7

    assert box[0][0] == 0.0
    assert np.round(box[1][0], 2) == 6.72

    lmp.reset_box([0.0,0.0,0.0], [8.0,8.0,8.0], 0.0,0.0,0.0)
    box = lmp.extract_box()
    assert box[0][0] == 0.0
    assert np.round(box[1][0], 2) == 8.0


def test_extract_fix():

    x = lmp.extract_fix("2", 0, 1, 1)
    assert np.round(x, 2) == -2.61

def test_extract_variable():

    x = lmp.extract_variable("tt", "all", 0)
    assert np.round(x, 2) == 1.13

    x = lmp.extract_variable("fx", "all", 1)
    assert len(x) == 128
    assert np.round(x[0], 2) == -0.26

def test_scatter_atoms():

    f = lmp.gather_atoms("f")
    val = np.random.randint(0, 100)
    f[1][0] = val
    lmp.scatter_atoms("f", f)
    f1 = lmp.gather_atoms("f")
    assert f1[1][0] == val

    f = lmp.gather_atoms("f", ids=[1,2])
    val = np.random.randint(0, 100)
    f[1][1] = val
    lmp.scatter_atoms("f", f, ids=[1,2])
    f1 = lmp.gather_atoms("f", ids=[1,2])
    assert f1[1][1] == val
