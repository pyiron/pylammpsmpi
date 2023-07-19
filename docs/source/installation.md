# Installation
The `pylammpsmpi` package can be installed either via `pip` or `conda`. While the package itself is written in pure 
python and is therefore operation system independent, compiling the LAMMPS simulation code as library can be challenging
in particular on Windows. To simplify the installation for Mac OS and Linux users, the LAMMPS simulation code is 
provided as `conda-forge` package. Consequently, the conda-based installation is recommended for `pylammpsmpi`.

## conda-based installation 
The `conda` package combines all dependencies in one package including the LAMMPS simulation code: 
```
conda install -c conda-forge pylammpsmpi
```
When resolving the dependencies with `conda` gets slow it is recommended to use `mamba` instead of `conda`. So you can 
also install `pylammpsmpi` using: 
```
mamba install -c conda-forge pylammpsmpi
```

## pypi-based installation
Alternatively, `pylammpsmpi` can also be installed from the python package index (pypi) using the following command: 
```
pip install pylammpsmpi
```
This pypi package only contains the `pylammpsmpi` and the python dependencies it is build on, but it does not contain
the LAMMPS simulation code. This installation is recommended for advanced users who want to optimize their installation
for optimal performance, by compiling their own version of LAMMPS and `mpi4py` to match the MPI library on their 
computer of high performance computing (HPC) cluster. 
