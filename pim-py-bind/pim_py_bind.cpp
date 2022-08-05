#include <pim_runtime_api.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include "half.hpp"

namespace py = pybind11;

PimBo* PyWrapperPimCreateBoNCHW(int n, int c, int h, int w, PimPrecision prec, PimMemType mem, uintptr_t usr_ptr)
{
    void* user = (usr_ptr == 0) ? nullptr : (void*)usr_ptr;
    return PimCreateBo(n, c, h, w, prec, mem, user);
}

PimBo* PyWrapperPimCreateBoDesc(PimDesc* desc, PimMemType mem, PimMemFlag mflag, uintptr_t usr_ptr)
{
    void* user = (usr_ptr == 0) ? nullptr : (void*)usr_ptr;
    return PimCreateBo(desc, mem, mflag, user);
}

PimBo* PyWrapperPimCreateBoGemmDesc(PimGemmDesc* desc, PimMemType mem, PimMemFlag mflag, uintptr_t usr_ptr)
{
    void* user = (usr_ptr == 0) ? nullptr : (void*)usr_ptr;
    return PimCreateBo(desc, mem, mflag, user);
}

int PyWrapperPimAllocMemory(uintptr_t usr_ptr, size_t size, PimMemType mem)
{
    void* user = (usr_ptr == 0) ? nullptr : (void*)usr_ptr;
    return PimAllocMemory(&user, size, mem);
}

PYBIND11_MODULE(pim_api, api_interface)
{
    api_interface.doc() = "pybind11 binding for PimLibrary";
    py::enum_<PimRuntimeType>(api_interface, "PimRuntimeType")
        .value("RT_TYPE_HIP", RT_TYPE_HIP)
        .value("RT_TYPE_OPENCL", RT_TYPE_OPENCL)
        .export_values();

    py::enum_<PimMemType>(api_interface, "PimMemType")
        .value("MEM_TYPE_HOST", MEM_TYPE_HOST)
        .value("MEM_TYPE_DEVICE", MEM_TYPE_DEVICE)
        .value("MEM_TYPE_PIM", MEM_TYPE_PIM)
        .export_values();

    py::enum_<PimMemFlag>(api_interface, "PimMemFlag")
        .value("ELT_OP", ELT_OP)
        .value("GEMV_INPUT", GEMV_INPUT)
        .value("GEMV_WEIGHT", GEMV_WEIGHT)
        .value("GEMV_OUTPUT", GEMV_OUTPUT)
        .value("GEMM_INPUT", GEMM_INPUT)
        .value("GEMM_WEIGHT", GEMM_WEIGHT)
        .value("GEMM_BIAS", GEMM_BIAS)
        .value("GEMM_OUTPUT", GEMM_OUTPUT)
        .export_values();

    py::enum_<PimActFunc>(api_interface, "PimActFunc")
        .value("NONE", NONE)
        .value("ACT_RELU", ACT_RELU)
        .export_values();

    py::enum_<PimMemCpyType>(api_interface, "PimMemCpyType")
        .value("HOST_TO_HOST", HOST_TO_HOST)
        .value("HOST_TO_DEVICE", HOST_TO_DEVICE)
        .value("HOST_TO_PIM", HOST_TO_PIM)
        .value("DEVICE_TO_HOST", DEVICE_TO_HOST)
        .value("DEVICE_TO_DEVICE", DEVICE_TO_DEVICE)
        .value("DEVICE_TO_PIM", DEVICE_TO_PIM)
        .value("PIM_TO_HOST", PIM_TO_HOST)
        .value("PIM_TO_DEVICE", PIM_TO_DEVICE)
        .value("PIM_TO_PIM", PIM_TO_PIM)
        .export_values();

    py::enum_<PimOpType>(api_interface, "PimOpType")
        .value("OP_GEMV", OP_GEMV)
        .value("OP_GEMM", OP_GEMM)
        .value("OP_ELT_ADD", OP_ELT_ADD)
        .value("OP_ELT_MUL", OP_ELT_MUL)
        .value("OP_RELU", OP_RELU)
        .value("OP_BN", OP_BN)
        .value("OP_DUMMY", OP_DUMMY)
        .export_values();

    py::enum_<PimPrecision>(api_interface, "PimPrecision")
        .value("PIM_FP16", PIM_FP16)
        .value("PIM_INT8", PIM_INT8)
        .export_values();

    py::class_<PimBShape>(api_interface, "PimBShape")
        .def(py::init<>())
        .def_readwrite("n", &PimBShape::n)
        .def_readwrite("c", &PimBShape::c)
        .def_readwrite("h", &PimBShape::h)
        .def_readwrite("w", &PimBShape::w);

    py::class_<PimBo>(api_interface, "PimBo", py::buffer_protocol()).def_buffer([](PimBo& bo) -> py::buffer_info {
        py::capsule FreePimBo(bo.data, [](void* py_usr_ptr) {});
        return py::buffer_info(
            bo.data,                                                              /* Pointer to buffer */
            sizeof(half_float::half),                                             /* Size of one scalar */
            "e",                                                                  /* Python struct-style format descriptor */
            4,                                                                    /* Number of dimensions */
            { bo.bshape.n, bo.bshape.c, bo.bshape.h, bo.bshape.w },               /* Buffer dimensions */
            {
              sizeof(half_float::half) * bo.bshape.c * bo.bshape.h * bo.bshape.w, /* Strides (in bytes) for each index */
              sizeof(half_float::half) * bo.bshape.h * bo.bshape.w,
              sizeof(half_float::half) * bo.bshape.w,
              sizeof(half_float::half)
            });
    });

    py::class_<PimDesc>(api_interface, "PimDesc")
        .def(py::init<>())
        .def_readwrite("bshape", &PimDesc::bshape)
        .def_readwrite("precision", &PimDesc::precision)
        .def_readwrite("op_type", &PimDesc::op_type)
        .def_readonly("bshape_r", &PimDesc::bshape_r);

    py::class_<PimGemmDesc>(api_interface, "PimGemmDesc")
        .def(py::init<>())
        .def_readwrite("in_bshape", &PimGemmDesc::in_bshape)
        .def_readwrite("in_bshape_r", &PimGemmDesc::in_bshape_r)
        .def_readwrite("wei_bshape", &PimGemmDesc::wei_bshape)
        .def_readwrite("wei_bshape_r", &PimGemmDesc::wei_bshape_r)
        .def_readwrite("bias_bshape", &PimGemmDesc::bias_bshape)
        .def_readwrite("bias_bshape_r", &PimGemmDesc::bias_bshape_r)
        .def_readwrite("out_bshape", &PimGemmDesc::out_bshape)
        .def_readwrite("out_bshape_r", &PimGemmDesc::out_bshape_r)
	.def_readwrite("precision", &PimGemmDesc::precision);

    api_interface.def("PimInitialize", &PimInitialize, "For initialization of pim data",
                      py::arg("rt_type") = RT_TYPE_HIP, py::arg("PimPrecision") = PIM_FP16);
    api_interface.def("PimDeinitialize", &PimDeinitialize, "For de initialization of pim data");
    api_interface.def("PimCreateBo", &PyWrapperPimCreateBoNCHW,
		      py::return_value_policy::reference, "For Creating PimBo memory object using nchw values");
    api_interface.def("PimCreateBo", &PyWrapperPimCreateBoDesc,
                      py::return_value_policy::reference, "For Creating PimBo memory object", py::arg("desc"),
                      py::arg("mem"), py::arg("mflag") = ELT_OP, py::arg("usr_ptr") = 0);
    api_interface.def("PimCreateBo", &PyWrapperPimCreateBoGemmDesc,
                      py::return_value_policy::reference, "For Creating PimBo memory object", py::arg("desc"),
                      py::arg("mem"), py::arg("mflag"), py::arg("usr_ptr") = 0);
    api_interface.def("PimDestroyBo", static_cast<int (*)(PimBo*)>(&PimDestroyBo));
    api_interface.def("PimCreateDesc", &PimCreateDesc, py::return_value_policy::reference);
    api_interface.def("PimCreateGemmDesc", &PimCreateGemmDesc, py::return_value_policy::reference);
    api_interface.def("PimDestroyDesc", &PimDestroyDesc);
    api_interface.def("PimDestroyGemmDesc", &PimDestroyGemmDesc);
    api_interface.def("PimAllocMemory", &PyWrapperPimAllocMemory);
    api_interface.def("PimAllocMemory", static_cast<int (*)(PimBo*)>(&PimAllocMemory));
    api_interface.def("PimFreeMemory", static_cast<int (*)(void*, PimMemType)>(&PimFreeMemory));
    api_interface.def("PimFreeMemory", static_cast<int (*)(PimBo*)>(&PimFreeMemory));
    api_interface.def("PimCopyMemory", static_cast<int (*)(void*, void*, size_t, PimMemCpyType)>(&PimCopyMemory));
    api_interface.def("PimCopyMemory", static_cast<int (*)(PimBo*, PimBo*, PimMemCpyType)>(&PimCopyMemory));
    api_interface.def("PimExecuteAdd", static_cast<int (*)(PimBo*, PimBo*, PimBo*, void*, bool)>(&PimExecuteAdd));
    api_interface.def("PimExecuteAdd", static_cast<int (*)(PimBo*, void*, PimBo*, void*, bool)>(&PimExecuteAdd));
    api_interface.def("PimExecuteMul", static_cast<int (*)(PimBo*, PimBo*, PimBo*, void*, bool)>(&PimExecuteMul));
    api_interface.def("PimExecuteMul", static_cast<int (*)(PimBo*, void*, PimBo*, void*, bool)>(&PimExecuteMul));
    api_interface.def("PimExecuteRelu", static_cast<int (*)(PimBo*, PimBo*, void*, bool)>(&PimExecuteRelu));
    api_interface.def("PimExecuteGemv", static_cast<int (*)(PimBo*, PimBo*, PimBo*, void*, bool)>(&PimExecuteGemv));
    api_interface.def("PimExecuteGemvAdd",
                      static_cast<int (*)(PimBo*, PimBo*, PimBo*, void*, bool)>(&PimExecuteGemvAdd));
    api_interface.def("PimExecuteGemvAdd",
                      static_cast<int (*)(PimBo*, PimBo*, PimBo*, PimBo*, bool ,void*, bool)>(&PimExecuteGemvAdd));
    api_interface.def("PimExecuteGemvList",
		      static_cast<int (*)(PimBo*, PimBo*, PimBo*, void*, bool)>(&PimExecuteGemvList));
    api_interface.def("PimExecuteGemm",
		      static_cast<int (*)(PimBo*, PimBo*, PimBo*, PimBo*, PimActFunc, bool, void*, bool)>(&PimExecuteGemm));
    api_interface.def("PimSetDevice", static_cast<int (*)(unsigned int)>(&PimSetDevice));
    api_interface.def("PimGetDevice", [](py::array_t<unsigned int> buffer){
                      py::buffer_info info = buffer.request();
                      PimGetDevice(static_cast<unsigned int *>(info.ptr));});
    api_interface.def("PimSynchronize", &PimSynchronize);
    api_interface.def("PimExecuteDummy", &PimExecuteDummy);
    api_interface.def("createStream", static_cast<void* (*)(PimRuntimeType)>(&createStream));
}
