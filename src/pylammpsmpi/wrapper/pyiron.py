import os
from ase.atoms import Atoms
import numpy as np
from pyiron_lammps import calc_md, calc_static, calc_minimize, get_potential_by_name
from typing import Optional

from pylammpsmpi.wrapper.ase import LammpsASELibrary


def _get_status(lmp, data_dict):
    data_dict['steps'].append(lmp.interactive_steps_getter())
    data_dict['volume'].append(lmp.interactive_volume_getter())
    data_dict['natoms'].append(len(lmp.interactive_indices_getter()))
    data_dict['cells'].append(lmp.interactive_cells_getter())
    data_dict['indices'].append(lmp.interactive_indices_getter()-1)
    data_dict['forces'].append(lmp.interactive_forces_getter())
    data_dict['velocities'].append(lmp.interactive_velocities_getter() / 1000)
    data_dict['positions'].append(lmp.interactive_positions_getter())
    data_dict['temperature'].append(lmp.interactive_temperatures_getter())
    data_dict['energy_pot'].append(lmp.interactive_energy_pot_getter())
    data_dict['energy_tot'].append(lmp.interactive_energy_tot_getter())
    data_dict['pressures'].append(lmp.interactive_pressures_getter() * 10000)


def lammps_file_interface_function(
    structure: Atoms,
    potential: str,
    working_directory: Optional[str] = None,
    calc_mode: str = "static",
    calc_kwargs: Optional[dict] = None,
    units: str = "metal",
    resource_path: Optional[str] = None,
):
    """
    A single function to execute a LAMMPS calculation based on the LAMMPS job implemented in pyiron

    Examples:

    >>> import os
    >>> from ase.build import bulk
    >>> from pyiron_atomistics.lammps.lammps import lammps_function
    >>>
    >>> shell_output, parsed_output, job_crashed = lammps_function(
    ...     working_directory=os.path.abspath("lmp_working_directory"),
    ...     structure=bulk("Al", cubic=True),
    ...     potential='2009--Mendelev-M-I--Al-Mg--LAMMPS--ipr1',
    ...     calc_mode="md",
    ...     calc_kwargs={"temperature": 500.0, "pressure": 0.0, "n_ionic_steps": 1000, "n_print": 100},
    ...     cutoff_radius=None,
    ...     units="metal",
    ...     bonds_kwargs={},
    ...      enable_h5md=False,
    ... )

    Args:
        working_directory (str): directory in which the LAMMPS calculation is executed
        structure (Atoms): ase.atoms.Atoms - atomistic structure
        potential (str): Name of the LAMMPS potential based on the NIST database and the OpenKIM database
        calc_mode (str): select mode of calculation ["static", "md", "minimize", "vcsgc"]
        calc_kwargs (dict): key-word arguments for the calculate function, the input parameters depend on the calc_mode:
          "static": No parameters
          "md": "temperature", "pressure", "n_ionic_steps", "time_step", "n_print", "temperature_damping_timescale",
                "pressure_damping_timescale", "seed", "tloop", "initial_temperature", "langevin", "delta_temp",
                "delta_press", job_name", "rotation_matrix"
          "minimize": "ionic_energy_tolerance", "ionic_force_tolerance", "max_iter", "pressure", "n_print", "style",
                      "rotation_matrix"
          "vcsgc": "mu", "ordered_element_list", "target_concentration", "kappa", "mc_step_interval", "swap_fraction",
                   "temperature_mc", "window_size", "window_moves", "temperature", "pressure", "n_ionic_steps",
                   "time_step", "n_print", "temperature_damping_timescale", "pressure_damping_timescale", "seed",
                   "initial_temperature", "langevin", "job_name", "rotation_matrix"
        cutoff_radius (float): cut-off radius for the interatomic potential
        units (str): Units for LAMMPS
        bonds_kwargs (dict): key-word arguments to create atomistic bonds:
          "species", "element_list", "cutoff_list", "max_bond_list", "bond_type_list", "angle_type_list",
        server_kwargs (dict): key-word arguments to create server object - the available parameters are:
          "user", "host", "run_mode", "queue", "qid", "cores", "threads", "new_h5", "structure_id", "run_time",
          "memory_limit", "accept_crash", "additional_arguments", "gpus", "conda_environment_name",
          "conda_environment_path"
        enable_h5md (bool): activate h5md mode for LAMMPS
        write_restart_file (bool): enable writing the LAMMPS restart file
        read_restart_file (bool): enable loading the LAMMPS restart file
        restart_file (str): file name of the LAMMPS restart file to copy
        executable_version (str): LAMMPS version to for the execution
        executable_path (str): path to the LAMMPS executable
        input_control_file (str|list|dict): Option to modify the LAMMPS input file directly

    Returns:
        str, dict, bool: Tuple consisting of the shell output (str), the parsed output (dict) and a boolean flag if
                         the execution raised an accepted error.
    """
    if calc_kwargs is None:
        calc_kwargs = {}

    dimension = 3
    boundary = "p p p"
    atom_style = "atomic"

    potential_dataframe = get_potential_by_name(
        potential_name=potential, resource_path=resource_path
    )

    data_dict = {
        'steps': [],
        'natoms': [],
        'cells': [],
        'indices': [],
        'forces': [],
        'velocities': [],
        'positions': [],
        'temperature': [],
        'energy_pot': [],
        'energy_tot': [],
        'volume': [],
        'pressures': []
    }

    if working_directory is not None:
        os.makedirs(working_directory, exist_ok=True)
    lmp = LammpsASELibrary(working_directory=working_directory)
    lmp.interactive_structure_setter(
        structure=structure,
        units=units,
        dimension=dimension,
        boundary=boundary,
        atom_style=atom_style,
        el_eam_lst=potential_dataframe.Species,
    )
    for l in potential_dataframe.Config:
        lmp.interactive_lib_command(l)

    if calc_mode == "static":
        for l in calc_static():
            lmp.interactive_lib_command(l)
    elif calc_mode == "md":
        if "n_ionic_steps" in calc_kwargs.keys():
            n_ionic_steps = calc_kwargs.pop("n_ionic_steps")
        else:
            n_ionic_steps = 1
        if "n_print" in calc_kwargs.keys():
            n_print = calc_kwargs.pop("n_print")
        else:
            n_print = 1
        for l in calc_md(**calc_kwargs, n_print=1) + ["run 0"]:
            lmp.interactive_lib_command(l)
        for s in range(n_ionic_steps // n_print):
            _get_status(lmp=lmp, data_dict=data_dict)
            lmp.interactive_lib_command("run " + str(n_print))
    elif calc_mode == "minimize":
        for l in calc_minimize(**calc_kwargs):
            lmp.interactive_lib_command(l)

    else:
        raise ValueError(
            f"calc_mode must be one of: static, md or minimize, not {calc_mode}"
        )
    _get_status(lmp=lmp, data_dict=data_dict)
    lmp.close()
    return None, {"generic": {k: np.array(v) for k, v in data_dict.items()}}, False
