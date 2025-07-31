# Interfaces
## LammpsBase
The most basic interface is the `LammpsBase`, it implements the same commands like the default `lammps.lammps` interface
and returns the same datatypes. With this API compatibility to the standard interface, this interface is commonly the 
easiest way to accelerate a serial LAMMPS based workflow by leveraging `mpi4py` parallel LAMMPS instances.

Import the `LammpsBase` class as:
```python
from pylammpsmpi import LammpsBase
```

Then an instance of this class can be instantiated using:
```python
lmp = LammpsBase(
    cores=8,
    oversubscribe=False,
    working_directory=".",
    cmdargs=None,
)
```
In this example the `mpi4py` parallel LAMMPS instances is set to be executed with 8 CPU cores `cores=8`. For OpenMPI 
based installations of the LAMMPS library oversubscription can be enabled by the `oversubscribe` flag. In the same way
the `enable_flux_backend` flag enables the use of the flux framework as parallelization backend rather than using MPI 
directly. This is primarily used in large allocation to distribute a number of tasks over the available resources. 
Furthermore by providing a working directory `working_directory` the output of the LAMMPS simulation can be stored in
a predefined folder and with `cmdargs` additional command line arguments can be attached to the LAMMPS libary call. For
more information on the available command line arguments for the LAMMPS library, please refer to the official 
documentation. In addition, the parameters `queue_adapter` and `queue_adapter_kwargs` provide an interface to 
[pysqa](https://pysqa.readthedocs.org) the simple queue adapter for python. The `queue_adapter` can be set as 
`pysqa.queueadapter.QueueAdapter` object and the `queue_adapter_kwargs` parameter represents a dictionary of input 
arguments for the `submit_job()` function of the queue adapter. 

Read an input file named `in.simple` located in the `tests` folder of this repository:
```python
lmp.file("tests/in.simple")
```

In analogy to the `file()` function which executes a whole file with input commands the `command()` function can either 
execute a single command provided as string:
```python
lmp.command("run 1")
```

Or alternatively execute a list of commands, when called with a list of strings as argument:
```python
lmp.command(["run 1", "run 1"])
```

Other commands implemented by the `LammpsBase` class are:

* `lmp.version`: to get the version of the LAMMPS library 
* `lmp.extract_setting()`: to extract settings from the LAMMPS library
* `lmp.extract_global()`: to extract global variables from the LAMMPS library
* `lmp.extract_box()`: to extract the box size from the LAMMPS library 
* `lmp.extract_atom()`: to extract an individual atom from the LAMMPS library 
* `lmp.extract_fix()`: to extract a fix from the LAMMPS library 
* `lmp.extract_variable()`: to extract a variable from the LAMMPS library
* `lmp.natoms`: to extract the total number of atoms from the LAMMPS library 
* `lmp.get_natoms()`: to extract the total number of atoms from the LAMMPS library 
* `lmp.set_variable()`: to set a variable for the LAMMPS library 
* `lmp.reset_box()`: to reset the simulation box for the LAMMPS library 
* `lmp.generate_atoms()`: to generate atoms inside the simulation box for the LAMMPS library 
* `lmp.create_atoms()`: to create atoms inside the simulation box for the LAMMPS library
* `lmp.has_exceptions`: check if the LAMMPS library was compiled with the flag for exception handling
* `lmp.has_gzip_support`: check if the LAMMPS library was compiled with the flag for gzip compression
* `lmp.has_png_support`: check if the LAMMPS library was compiled with the flag for png graphics output
* `lmp.has_jpeg_support`: check if the LAMMPS library was compiled with the flag for jpeg graphics output
* `lmp.has_ffmpeg_support`: check if the LAMMPS library was compiled with the flag for ffmepg video output
* `lmp.installed_packages`: list packages installed in the LAMMPS library 
* `lmp.set_fix_external_callback()`: set fix for external call back for the LAMMPS library 
* `lmp.get_neighlist()`: get the neighbor list from the LAMMPS library
* `lmp.find_pair_neighlist()`: get the neighbor list pairs from LAMMPS library
* `lmp.find_fix_neighlist()`: find the list index of the fix neighbor list from the LAMMPS library
* `lmp.find_compute_neighlist()`: find the list index of the compute neighbor list from the LAMMPS library 
* `lmp.get_neighlist_size()`: get the size of the neighbor list from the LAMMPS libary 
* `lmp.get_neighlist_element_neighbors()`: get the neighbors of one specific element of the neighbor list from the 
  LAMMPS library.
* `lmp.gather_atoms()`: Gather information of the atoms from the LAMMPS library
* `lmp.scatter_atoms()`: Scatter information of the atoms to the LAMMPS library 
* `lmp.get_thermo()`: Get a thermo entry from the LAMMPS library 
* `lmp.extract_compute()`: Extract a compute from the LAMMPS library

For more details about the individual commands of the `LammpsBase` class please refer to the LAMMPS python library 
documentation. 

## LammpsConcurrent
Inspired by the `concurrent.futures` module in the standard python library the `pylammpsmpi.LammpsConcurrent` interface 
implements the same API as the `pylammpsmpi.LammpsBase` class but rather than holding the controlling process until the
`mpi4py` parallel LAMMPS instance finishes the execution of a given set of commands, the `pylammpsmpi.LammpsConcurrent` 
interface returns a   `concurrent.futures.Future` object. This enables the development of asynchronous / concurrent 
workflows.

Import the `LammpsConcurrent` class as:
```python
from pylammpsmpi import LammpsConcurrent
```

Then an instance of this class can be instantiated using:
```python
lmp = LammpsConcurrent(
    cores=8,
    oversubscribe=False,
    working_directory=".",
    cmdargs=None,
)
```
In this example the `mpi4py` parallel LAMMPS instances is set to be executed with 8 CPU cores `cores=8`. For OpenMPI 
based installations of the LAMMPS library oversubscription can be enabled by the `oversubscribe` flag. In the same way
the `enable_flux_backend` flag enables the use of the flux framework as parallelization backend rather than using MPI 
directly. This is primarily used in large allocation to distribute a number of tasks over the available resources. 
Furthermore by providing a working directory `working_directory` the output of the LAMMPS simulation can be stored in
a predefined folder and with `cmdargs` additional command line arguments can be attached to the LAMMPS libary call. For
more information on the available command line arguments for the LAMMPS library, please refer to the official 
documentation. In addition, the parameters `queue_adapter` and `queue_adapter_kwargs` provide an interface to 
[pysqa](https://pysqa.readthedocs.org) the simple queue adapter for python. The `queue_adapter` can be set as 
`pysqa.queueadapter.QueueAdapter` object and the `queue_adapter_kwargs` parameter represents a dictionary of input 
arguments for the `submit_job()` function of the queue adapter. 

Read an input file named `in.simple` located in the `tests` folder of this repository:
```python
fs = lmp.file("tests/in.simple")
fs.result()

>>> True
```
In contrast to the `LammpsBase` instance the `file()` function as well as any other function returns a 
`concurrent.futures.Future` object. The status object can be checked with the `done()` function. Alternatively the 
main process controlling the `mpi4py` parallel LAMMPS instances can be hold until the execution is completed using the 
`result()` function. If the underlying LAMMPS library does not return any output then the `LammpsConcurrent` class
returns `True` to indicate the execution finished successful. 

In analogy to the `file()` function which executes a whole file with input commands the `command()` function can either 
execute a single command provided as string:
```python
fs = lmp.command("run 1")
fs.result()

>>> True
```

Or alternatively execute a list of commands, when called with a list of strings as argument:
```python
fs = lmp.command(["run 1", "run 1"])
fs.result()

>>> True
```
Again in both cases the `command()` function returns again a `concurrent.futures.Future` object to track the status of 
the execution and continue interacting with the main process while the corresponding commands are executed in the 
`mpi4py` parallel LAMMPS instance.

Other commands implemented by the `LammpsConcurrent` class are:

* `lmp.version`: to get the version of the LAMMPS library 
* `lmp.extract_setting()`: to extract settings from the LAMMPS library
* `lmp.extract_global()`: to extract global variables from the LAMMPS library
* `lmp.extract_box()`: to extract the box size from the LAMMPS library 
* `lmp.extract_atom()`: to extract an individual atom from the LAMMPS library 
* `lmp.extract_fix()`: to extract a fix from the LAMMPS library 
* `lmp.extract_variable()`: to extract a variable from the LAMMPS library
* `lmp.natoms`: to extract the total number of atoms from the LAMMPS library 
* `lmp.get_natoms()`: to extract the total number of atoms from the LAMMPS library 
* `lmp.set_variable()`: to set a variable for the LAMMPS library 
* `lmp.reset_box()`: to reset the simulation box for the LAMMPS library 
* `lmp.generate_atoms()`: to generate atoms inside the simulation box for the LAMMPS library 
* `lmp.create_atoms()`: to create atoms inside the simulation box for the LAMMPS library
* `lmp.has_exceptions`: check if the LAMMPS library was compiled with the flag for exception handling
* `lmp.has_gzip_support`: check if the LAMMPS library was compiled with the flag for gzip compression
* `lmp.has_png_support`: check if the LAMMPS library was compiled with the flag for png graphics output
* `lmp.has_jpeg_support`: check if the LAMMPS library was compiled with the flag for jpeg graphics output
* `lmp.has_ffmpeg_support`: check if the LAMMPS library was compiled with the flag for ffmepg video output
* `lmp.installed_packages`: list packages installed in the LAMMPS library 
* `lmp.set_fix_external_callback()`: set fix for external call back for the LAMMPS library 
* `lmp.get_neighlist()`: get the neighbor list from the LAMMPS library
* `lmp.find_pair_neighlist()`: get the neighbor list pairs from LAMMPS library
* `lmp.find_fix_neighlist()`: find the list index of the fix neighbor list from the LAMMPS library
* `lmp.find_compute_neighlist()`: find the list index of the compute neighbor list from the LAMMPS library 
* `lmp.get_neighlist_size()`: get the size of the neighbor list from the LAMMPS libary 
* `lmp.get_neighlist_element_neighbors()`: get the neighbors of one specific element of the neighbor list from the 
  LAMMPS library.
* `lmp.gather_atoms()`: Gather information of the atoms from the LAMMPS library
* `lmp.scatter_atoms()`: Scatter information of the atoms to the LAMMPS library 
* `lmp.get_thermo()`: Get a thermo entry from the LAMMPS library 
* `lmp.extract_compute()`: Extract a compute from the LAMMPS library

For more details about the individual commands of the `LammpsBase` class please refer to the LAMMPS python library 
documentation. 

## LammpsLibrary
Finally, the :code:`pylammpsmpi.LammpsLibrary` interface adds a higher level interface on top of the default 
`lammps.lammps` interface. This higher level interface provides direct access to the commands and thermodynamic 
properties used in the LAMMPS input files. Especially for experienced LAMMPS users who   are familiar with the LAMMPS 
input files this interface simplifies switching from file based input to using the python interface.

Import the `LammpsLibrary` class as:
```python
from pylammpsmpi import LammpsLibrary
```

Then an instance of this class can be instantiated using:
```python
lmp = LammpsLibrary(
    cores=8,
    oversubscribe=False,
    working_directory=".",
    cmdargs=None,
)
```
In this example the `mpi4py` parallel LAMMPS instances is set to be executed with 8 CPU cores `cores=8`. For OpenMPI 
based installations of the LAMMPS library oversubscription can be enabled by the `oversubscribe` flag. In the same way
the `enable_flux_backend` flag enables the use of the flux framework as parallelization backend rather than using MPI 
directly. This is primarily used in large allocation to distribute a number of tasks over the available resources. 
Furthermore by providing a working directory `working_directory` the output of the LAMMPS simulation can be stored in
a predefined folder and with `cmdargs` additional command line arguments can be attached to the LAMMPS libary call. For
more information on the available command line arguments for the LAMMPS library, please refer to the official 
documentation. In addition, the parameters `queue_adapter` and `queue_adapter_kwargs` provide an interface to 
[pysqa](https://pysqa.readthedocs.org) the simple queue adapter for python. The `queue_adapter` can be set as 
`pysqa.queueadapter.QueueAdapter` object and the `queue_adapter_kwargs` parameter represents a dictionary of input 
arguments for the `submit_job()` function of the queue adapter. 

Read an input file named `in.simple` located in the `tests` folder of this repository:
```python
lmp.file("tests/in.simple")
```

In analogy to the `file()` function which executes a whole file with input commands the `command()` function can either 
execute a single command provided as string:
```python
lmp.command("run 1")
```

Or alternatively execute a list of commands, when called with a list of strings as argument:
```python
lmp.command(["run 1", "run 1"])
```

Beyond the commands defined in the `LammpsBase` class, the `LammpsLibrary` implements a wide range of commands commonly
used in the LAMMPS input files.