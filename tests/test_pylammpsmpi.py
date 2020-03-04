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
    
    ids = lmp.extract_atom("idb")

def test_extract_atom():
    f = lmp.extract_atom("f")
    assert len(f) == 256
    
    assert np.round(f[0][0], 2) == -0.26

    
