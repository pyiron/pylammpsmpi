import pytest
import os
import numpy as np
from pylammpsmpi.lammps import LammpsLibrary

lmp = LammpsLibrary(cores=2)
lmp.file("in.simple")

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
b
    
