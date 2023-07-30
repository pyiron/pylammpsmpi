import unittest

try:
    from ase.build import bulk
    from ase.constraints import FixAtoms
    from ase.constraints import FixedPlane
    from ase.constraints import FixCom
    from pylammpsmpi.wrapper.ase import set_selective_dynamics
    skip_ase_test = False
except ImportError:
    skip_ase_test = True


@unittest.skipIf(skip_ase_test, "ase is not installed, so the LAMMPS ASE tests are skipped.")
class TestConstraints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        structure = bulk("Cu", cubic=True)
        structure.symbols[2:] = "Al"
        cls.structure = structure

    def test_selective_dynamics_mixed_calcmd(self):
        atoms = self.structure.copy()
        c1 = FixAtoms(indices=[atom.index for atom in atoms if atom.symbol == 'Cu'])
        c2 = FixedPlane(
            [atom.index for atom in atoms if atom.symbol == 'Al'],
            [1, 0, 0],
        )
        atoms.set_constraint([c1, c2])
        control_dict = set_selective_dynamics(structure=atoms, calc_md=True)
        self.assertEqual(len(control_dict), 6)
        self.assertTrue(control_dict['group constraintxyz'], 'id 1 2')
        self.assertTrue(control_dict['fix constraintxyz'], 'constraintxyz setforce 0.0 0.0 0.0')
        self.assertTrue(control_dict['velocity constraintxyz'], 'set 0.0 0.0 0.0')
        self.assertTrue(control_dict['group constraintx'], 'id 3 4')
        self.assertTrue(control_dict['fix constraintx'], 'constraintx setforce 0.0 NULL NULL')
        self.assertTrue(control_dict['velocity constraintx'], 'set 0.0 NULL NULL')

    def test_selective_dynamics_mixed(self):
        atoms = self.structure.copy()
        c1 = FixAtoms(indices=[atom.index for atom in atoms if atom.symbol == 'Cu'])
        c2 = FixedPlane(
            [atom.index for atom in atoms if atom.symbol == 'Al'],
            [1, 0, 0],
        )
        atoms.set_constraint([c1, c2])
        control_dict = set_selective_dynamics(structure=atoms, calc_md=False)
        self.assertEqual(len(control_dict), 4)
        self.assertTrue(control_dict['group constraintxyz'], 'id 1 2')
        self.assertTrue(control_dict['fix constraintxyz'], 'constraintxyz setforce 0.0 0.0 0.0')
        self.assertTrue(control_dict['group constraintx'], 'id 3 4')
        self.assertTrue(control_dict['fix constraintx'], 'constraintx setforce 0.0 NULL NULL')

    def test_selective_dynamics_single_fix(self):
        atoms = self.structure.copy()
        c1 = FixAtoms(indices=[atom.index for atom in atoms if atom.symbol == 'Cu'])
        atoms.set_constraint(c1)
        control_dict = set_selective_dynamics(structure=atoms, calc_md=False)
        self.assertEqual(len(control_dict), 2)
        self.assertTrue(control_dict['group constraintxyz'], 'id 1 2')
        self.assertTrue(control_dict['fix constraintxyz'], 'constraintxyz setforce 0.0 0.0 0.0')

    def test_selective_dynamics_errors(self):
        atoms = self.structure.copy()
        atoms.set_constraint(FixCom())
        with self.assertRaises(ValueError):
            set_selective_dynamics(structure=atoms, calc_md=False)

    def test_selective_dynamics_wrong_plane(self):
        atoms = self.structure.copy()
        atoms.set_constraint(FixedPlane(
            [atom.index for atom in atoms if atom.symbol == 'Al'],
            [2, 1, 0],
        ))
        with self.assertRaises(ValueError):
            set_selective_dynamics(structure=atoms, calc_md=False)