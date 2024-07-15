# pylammpsmpi - Parallel Lammps Python interface
[![Coverage Status](https://coveralls.io/repos/github/pyiron/pylammpsmpi/badge.svg?branch=main)](https://coveralls.io/github/pyiron/pylammpsmpi?branch=main)
[![Python package](https://github.com/pyiron/pylammpsmpi/workflows/Python%20package/badge.svg)](https://github.com/pyiron/pylammpsmpi/actions)

With `pylammpsmpi` you can control a `mpi4py` parallel LAMMPS instance from a serial python process or a Jupyter 
notebook. Internally `pylammpsmpi` leverages the `executorlib` communication interface to connect the serial python 
process the user interacts with, with the `mpi4py` parallel LAMMPS instance. The advantage of separating the `mpi4py` 
parallel LAMMPS instance from the rest of the workflow is that the workflow can be written as serial python code, while
still benefiting from the parallel performance of LAMMPS. Still this comes at the cost of additional data transfer, as 
the LAMMPS pointers cannot be transferred this way and the linked data has to be copied instead. So copying large 
atomistic structures can decrease the performance of the `pylammpsmpi` interface in comparison to writing your own fully
`mpi4py` parallel LAMMPS workflows.

# Interfaces
The `pylammpsmpi` module implements three different interfaces for different use cases:

* `pylammpsmpi.LammpsBase`: The most basic interface is the `LammpsBase`, it implements the same commands like the 
  default `lammps.lammps` interface and returns the same datatypes. With this API compatibility to the standard 
  interface, this interface is commonly the easiest way to accelerate a serial LAMMPS based workflow by leveraging 
  `mpi4py` parallel LAMMPS instances.
* `pylammpsmpi.LammpsConcurrent`: Inspired by the `concurrent.futures` module in the standard python library the 
  `pylammpsmpi.LammpsConcurrent` interface implements the same API as the `pylammpsmpi.LammpsBase` class but rather than 
  holding the controlling process until the `mpi4py` parallel LAMMPS instance finishes the execution of a given set of
  commands, the `pylammpsmpi.LammpsConcurrent` interface returns a `concurrent.futures.Future` object. This enables the 
  development of asynchronous / concurrent workflows.
* `pylammpsmpi.LammpsLibrary`: Finally, the `pylammpsmpi.LammpsLibrary` interface adds a higher level interface on top 
  of the default `lammps.lammps` interface. This higher level interface provides direct access to the commands and 
  thermodynamic properties used in the LAMMPS input files. Especially for experienced LAMMPS users who are familiar with
  the LAMMPS input files this interface simplifies switching from file based input to using the python interface.

The choice of interface depends on the users background, experience and the simulation protocol the user wants to
implement. Still internally all three interfaces are based on the `pylammpsmpi.LammpsConcurrent` interface, so they use 
an additional thread to connect the `mpi4py` parallel LAMMPS instance to the serial python process or Jupyter notebook.

# Documentation
* [Installation](https://pylammpsmpi.readthedocs.io/en/latest/installation.html)
  * [conda-based installation](https://pylammpsmpi.readthedocs.io/en/latest/installation.html#conda-based-installation)
  * [pypi-based installation](https://pylammpsmpi.readthedocs.io/en/latest/installation.html#pypi-based-installation)
* [Interfaces](https://pylammpsmpi.readthedocs.io/en/latest/interfaces.html) 
  * [LammpsBase](https://pylammpsmpi.readthedocs.io/en/latest/interfaces.html#lammpsbase)
  * [LammpsConcurrent](https://pylammpsmpi.readthedocs.io/en/latest/interfaces.html#lammpsconcurrent)
  * [LammpsLibrary](https://pylammpsmpi.readthedocs.io/en/latest/interfaces.html#lammpslibrary)
* [Development](https://pylammpsmpi.readthedocs.io/en/latest/development.html) 

# License
`pylammpsmpi` is released under the BSD license https://github.com/pyiron/pylammpsmpi/blob/main/LICENSE . It is a 
spin-off of the `pyiron` project https://github.com/pyiron/pyiron therefore if you use `pylammpsmpi` for calculation 
which result in a scientific publication, please cite: 

    @article{pyiron-paper,
      title = {pyiron: An integrated development environment for computational materials science},
      journal = {Computational Materials Science},
      volume = {163},
      pages = {24 - 36},
      year = {2019},
      issn = {0927-0256},
      doi = {https://doi.org/10.1016/j.commatsci.2018.07.043},
      url = {http://www.sciencedirect.com/science/article/pii/S0927025618304786},
      author = {Jan Janssen and Sudarsan Surendralal and Yury Lysogorskiy and Mira Todorova and Tilmann Hickel and Ralf Drautz and Jörg Neugebauer},
      keywords = {Modelling workflow, Integrated development environment, Complex simulation protocols},
    }
