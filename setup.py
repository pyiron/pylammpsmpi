from setuptools import setup, find_packages
import versioneer

setup(
    name="pylammpsmpi",
    version=versioneer.get_version(),
    description="Parallel Lammps Python interface",
    long_description="PylammpsMPI couples a serial python process to an MPI parallel LAMMPS libary.",
    url='https://github.com/pyiron/pylammpsmpi',
    author='Jan Janssen',
    author_email='janssen@mpie.de',
    license='BSD',

    classifiers=['Development Status :: 5 - Production/Stable',
                 'Topic :: Scientific/Engineering :: Physics',
                 'License :: OSI Approved :: BSD License',
                 'Intended Audience :: Science/Research',
                 'Operating System :: OS Independent',
                 'Programming Language :: Python :: 3.7',
                 'Programming Language :: Python :: 3.8',
                 'Programming Language :: Python :: 3.9',
                 'Programming Language :: Python :: 3.10',
                 'Programming Language :: Python :: 3.11'
                ],

    keywords='lammps, mpi4py',
    packages=find_packages(exclude=["*tests*"]),
    install_requires=[
        "cloudpickle==2.2.1", "mpi4py==3.1.4", "pympipool==0.5.0", "pyzmq==25.1.0",
    ],
    cmdclass=versioneer.get_cmdclass(),
)
