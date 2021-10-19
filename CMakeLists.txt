cmake_minimum_required(VERSION 3.4...3.18)
project(PimPyLib)

add_subdirectory(external/pybind11)
include_directories($ENV{ROCM_PATH}/include)
link_directories($ENV{ROCM_PATH}/lib)

add_library(pim_api MODULE pim-py-bind/pim_py_bind.cpp)
target_link_libraries(pim_api PRIVATE pybind11::module pybind11::lto PimRuntime)
pybind11_extension(pim_api)
pybind11_strip(pim_api)
