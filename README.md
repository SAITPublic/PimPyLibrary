# PimPyLib

This repository provides an python interface for PIMLibrary
## Download submodule
use submodule update for pybind11 submodule once for downloading it.
```
git submodule update --init
```
## How to Build and Install
Below command builds PimPyLib interface and installs python package
```
export ROCM_PATH=<path to rocm installation>
pip3 install --trusted-host 'pypi.org' ./PimPyLib
```

## Tests
`tests` folder contains python test scripts for testing python bindings
### example
`python3 elt_add.py`