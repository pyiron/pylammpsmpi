.. pylammpsmpi documentation master file, created by
   sphinx-quickstart on Sun Mar  8 11:28:34 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Parallel Lammps Python interface
================================
With :code:`pylammpsmpi` you can control a :code:`mpi4py` parallel LAMMPS instance from a serial python process or a
Jupyter notebook. Internally :code:`pylammpsmpi` leverages the :code:`pympipool` communication interface to connect the
serial python process the user interacts with, with the :code:`mpi4py` parallel LAMMPS instance. The advantage of
separating the :code:`mpi4py` parallel LAMMPS instance from the rest of the workflow is that the workflow can be written
as serial python code, while still benefiting from the parallel performance of LAMMPS. Still this comes at the cost of
additional data transfer, as the LAMMPS pointers cannot be transferred this way and the linked data has to be copied
instead. So copying large atomistic structures can decrease the performance of the :code:`pylammpsmpi` interface in
comparison to writing your own fully :code:`mpi4py` parallel LAMMPS workflows.

Interfaces
----------
The :code:`pylammpsmpi` module implements three different interfaces for different use cases:

* :code:`pylammpsmpi.LammpsBase`: The most basic interface is the :code:`LammpsBase`, it implements the same commands
  like the default :code:`lammps.lammps` interface and returns the same datatypes. With this API compatibility to the
  standard interface, this interface is commonly the easiest way to accelerate a serial LAMMPS based workflow by
  leveraging :code:`mpi4py` parallel LAMMPS instances.
* :code:`pylammpsmpi.LammpsConcurrent`: Inspired by the :code:`concurrent.futures` module in the standard python library
  the :code:`pylammpsmpi.LammpsConcurrent` interface implements the same API as the :code:`pylammpsmpi.LammpsBase` class
  but rather than holding the controlling process until the :code:`mpi4py` parallel LAMMPS instance finishes the
  execution of a given set of commands, the :code:`pylammpsmpi.LammpsConcurrent` interface returns a
  :code:`concurrent.futures.Future` object. This enables the development of asynchronous / concurrent workflows.
* :code:`pylammpsmpi.LammpsLibrary`: Finally, the :code:`pylammpsmpi.LammpsLibrary` interface adds a higher level
  interface on top of the default :code:`lammps.lammps` interface. This higher level interface provides direct access to
  the commands and thermodynamic properties used in the LAMMPS input files. Especially for experienced LAMMPS users who
  are familiar with the LAMMPS input files this interface simplifies switching from file based input to using the python
  interface.

The choice of interface depends on the users background, experience and the simulation protocol the user wants to
implement. Still internally all three interfaces are based on the :code:`pylammpsmpi.LammpsConcurrent` interface, so
they use an additional thread to connect the :code:`mpi4py` parallel LAMMPS instance to the serial python process or
Jupyter notebook.

Documentation
-------------

.. toctree::

    installation
    interfaces
    development