# PimPyLibrary

This repository provides an python interface for PIMLibrary
## Download submodule
use submodule update for pybind11 submodule once for downloading it.
```
git submodule update --init
```
## How to Build and Install
Below command builds PimPyLibrary interface and installs python package
```
export ROCM_PATH=<path to rocm installation>
pip3 install --trusted-host 'pypi.org' ./PimPyLibrary
```

## Tests
`tests` folder contains python test scripts for testing python bindings
### example
For all tests
`python3 -m unittest unittests/test_*`

For Individual tests
`python3 -m unittest unittests/test_elt_add.py`


## Custom ops
Custom ops are supported for pytorch and tensorflow
### Pytorch custom ops
To install
```
pip3 install --trusted-host 'pypi.org' ./PimPyLibrary/custom-ops/pytorch
```
To run tests in pytorch/unit-tests
```
python3 ./PimPyLibrary/custom-ops/pytorch/unit-tests/test_eltwise_add.py 
```

### Tensorflow custom ops
To install
```
pip3 install --trusted-host 'pypi.org' ./PimPyLibrary/custom-ops/tensorflow -v
```
To run tests in tensorflow/unit-tests
```
python3 ./PimPyLibrary/custom-ops/tensorflow/unit-tests/test_eltwise_add.py
