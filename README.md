# `PyLammpsMPI`

## Example usage

### Setting up a simulation


```python
from pylammpsmpi.lammps import LammpsLibrary
```

Set up a job which runs on 2 cores


```python
lmp = LammpsLibrary(cores=4)
```

Read an input file


```python
lmp.file("tests/in.simple")
```

Check version of lammps


```python
lmp.version
```




    20190807



Check number of atoms


```python
lmp.natoms
```




    256



### Run commands


```python
lmp.command("run 1")
```


```python
lmp.command(["run 1", "run 1"])
```

Extract a global property


```python
lmp.extract_global("boxxhi", 1)
```




    6.718384765530029



### Accessing simulation box


```python
lmp.extract_box()
```




    ([0.0, 0.0, 0.0],
     [6.718384765530029, 6.718384765530029, 6.718384765530029],
     0.0,
     0.0,
     0.0,
     [1, 1, 1],
     0)



### Accessing and changing atom properties

Get individual atom properties, for example force on each atoms


```python
ff = lmp.gather_atoms("f")
print(type(ff))
print(len(ff))
```

    <class 'numpy.ndarray'>
    256


Get atom properties by their ids


```python
ids = lmp.gather_atoms("id")
```


```python
ff = lmp.gather_atoms("f", ids=ids[:10])
len(ff)
```




    10



Change atom properties


```python
ff = ff*0.5
lmp.scatter_atoms("f", ff, ids=ids[:10])
```

### Access value of variables


```python
temp = lmp.extract_variable("tt", "all", 0)
temp
```




    0.8846341461467611



### Access value of computes


```python
ke = lmp.extract_compute("ke", 1, 1)
len(ke)
```




    256




```python
v = lmp.extract_compute("v", 1, 2, width=3)
v.shape
```




    (256, 3)




```python
lmp.extract_compute("1", 0, 0)
```




    0.8846341461467611




```python
msd = lmp.extract_compute("msd", 0, 1, length=4)
msd[0]
```




    0.005507481618069701



### Access values from fix


```python
x = lmp.extract_fix("2", 0, 1, 1)
x
```




    -2.605817524153117



### Change the simulation box


```python
lmp.reset_box([0.0,0.0,0.0], [8.0,8.0,8.0], 0.0,0.0,0.0)
```
