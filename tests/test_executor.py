import unittest
from ase.build import bulk
from executorlib import Executor
from executorlib.standalone.serialize import cloudpickle_register
from pylammpsmpi import LammpsASELibrary


def calc_lmp(structure):
    from mpi4py import MPI

    lmp = LammpsASELibrary(
        working_directory=None,
        cores=1,
        comm=MPI.COMM_SELF,
        logger=None,
        log_file=None,
        library=None,
        disable_log_file=True,
    )
    lmp.interactive_structure_setter(
        structure=structure,
        units="lj",
        dimension=3,
        boundary=" ".join(["p" if coord else "f" for coord in structure.pbc]),
        atom_style="atomic",
        el_eam_lst=["Al"],
        calc_md=False,
    )
    lmp.interactive_lib_command("pair_style lj/cut 6.0")
    lmp.interactive_lib_command("pair_coeff 1 1 1.0 1.0 4.04")
    lmp.interactive_lib_command(
        command="thermo_style custom step temp pe etotal pxx pxy pxz pyy pyz pzz vol"
    )
    lmp.interactive_lib_command(command="thermo_modify format float %20.15g")
    lmp.interactive_lib_command("run 0")
    return lmp.interactive_energy_pot_getter()


class TestWithExecutor(unittest.TestCase):
    def test_executor(self):
        with Executor(max_cores=2, backend="local", hostname_localhost=True) as exe:
            cloudpickle_register(ind=1)
            future = exe.submit(calc_lmp, bulk("Al", cubic=True).repeat([2, 2, 2]))
            energy = future.result()
        self.assertAlmostEqual(energy, -0.04342932384411344)
