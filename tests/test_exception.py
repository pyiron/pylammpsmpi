import os.path
from unittest import TestCase
from pylammpsmpi import LammpsLibrary


lmp_str = """\
# 3d Lennard-Jones melt

units		lj
atom_style	atomic
atom_modify	map array

lattice		fcc 0.8442
region		box block 0 4 0 4 0 4
create_box	1 box
create_atoms	1 box
mass		1 1.0

velocity	all create 100.44 87287 loop geom

pair_style	lj/cut 2.5
pair_coeff	1 1 1.0 1.0 2.5

neighbor	0.3 bin
neigh_modify	delay 0 every 20 check no

compute         1 all temp
compute         2 all pressure 1
compute         ke all ke/atom
compute         msd all msd
compute         v all property/atom vx vy vz 

variable        fx atom fx
variable        tt equal temp
variable        test string "25"
fix		           1 all nve
fix              2 all ave/time 10 1 10 c_1 c_2



run		1000
"""


class TestException(TestCase):
    def test_overlapping_atoms(self):
        with open("in.error", "w") as f:
            f.writelines(lmp_str)
        with self.assertRaises(Exception):
            lmp = LammpsLibrary(cores=2)
            lmp.file("in.error")

    def tearDown(self):
        for f in ["in.error", "log.lammps"]:
            if os.path.exists(f):
                os.remove(f)
