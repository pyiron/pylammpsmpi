from setuptools import setup, find_packages
import versioneer

setup(
    name="pylammpsmpi",
    version=versioneer.get_version(),
    description="Parallel Lammps Python interface",
    url='https://github.com/pyiron/pylammpsmpi',
    author='Jan Janssen',
    author_email='janssen@mpie.de',
    license='BSD',

    classifiers=['Development Status :: 5 - Production/Stable',
                 'Topic :: Scientific/Engineering :: Physics',
                 'License :: OSI Approved :: BSD License',
                 'Intended Audience :: Science/Research',
                 'Operating System :: OS Independent',
                 'Programming Language :: Python :: 3',
                 'Programming Language :: Python :: 3.4',
                 'Programming Language :: Python :: 3.5',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7'],

    keywords='lammps, mpi4py',
    packages=find_packages(exclude=["*tests*"]),
    install_requires=['mpi4py'],
    cmdclass=versioneer.get_cmdclass(),
)
