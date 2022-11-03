# PimPyLibrary

This repository provides an python interface for PIMLibrary.
## Download submodule
Use submodule update for pybind11 submodule once for downloading it.
```
cd PimPyLibrary
git submodule update --init
```
## How to build and install PIM custom ops
The command below is how to build the PimPyLibrary interface and install the python package.
### Install PimPyLibrary interface
```
export ROCM_PATH=<path to rocm installation>
pip3 install --trusted-host 'pypi.org' .
```

### Install Pytroch custom ops
```
pip3 install --trusted-host 'pypi.org' custom-ops/pytorch
```

## How to test PIM custom ops
PIM custom ops can be run with numpy and pytorch.
### numpy examples
To run all numpy tests
```
python3 -m unittest examples/numpy/test_*.py
```

### Pytorch examples 
To run all Pytorch tests
```
python3 -m unittest examples/pytorch/test_*.py 
```

