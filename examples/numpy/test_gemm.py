import unittest
import numpy as np
import pim_api

class TestGemm(unittest.TestCase):
    def setUp(self):
        self.n = 1
        self.c = 1
        self.in_h = 1
        self.in_w = 1024
        self.out_h = 1
        self.out_w = 4096
        pim_api.PimInitialize(pim_api.RT_TYPE_HIP, pim_api.PIM_FP16)

    def test_gemm(self):
        gemm_order = pim_api.I_X_W
        transposed = False
        input = np.random.uniform(0.01, 0.05, (self.n, self.c, self.in_h, self.in_w)).astype(np.float16)
        self.weights = np.random.uniform(0.01, 0.05, (self.n, self.c, self.in_w, self.out_w)).astype(np.float16)
        golden = np.matmul(input, self.weights)
        output = np.zeros_like(golden, dtype=np.float16)

        desc = pim_api.PimCreateGemmDesc(self.n, self.c, self.in_h, self.in_w, self.out_h, self.out_w, pim_api.PIM_FP16, gemm_order)
        host_in = pim_api.PimCreateBo(desc, pim_api.MEM_TYPE_HOST, pim_api.GEMM_INPUT, input.__array_interface__['data'][0] , transposed);
        host_weight = pim_api.PimCreateBo(desc, pim_api.MEM_TYPE_HOST, pim_api.GEMM_WEIGHT, self.weights.__array_interface__['data'][0] , transposed);
        host_out = pim_api.PimCreateBo(desc, pim_api.MEM_TYPE_HOST, pim_api.GEMM_OUTPUT, output.__array_interface__['data'][0] , transposed)
        pim_in = pim_api.PimCreateBo(desc, pim_api.MEM_TYPE_DEVICE, pim_api.GEMM_INPUT, 0, transposed)
        pim_out = pim_api.PimCreateBo(desc, pim_api.MEM_TYPE_DEVICE, pim_api.GEMM_OUTPUT, 0, transposed)

        pim_api.PimCopyMemory(pim_in, host_in, pim_api.HOST_TO_DEVICE)
        pim_api.PimExecuteGemm(pim_out, pim_in, host_weight, None, pim_api.NONE, gemm_order, None, True)
        pim_api.PimCopyMemory(host_out, pim_out, pim_api.DEVICE_TO_HOST)

        self.assertEqual(np.allclose(golden, pim_out, rtol=0, atol=1e-01, equal_nan=False), True, "All Valaues should be equal")

    def test_gemm_channel(self):
        self.c = 4
        gemm_order = pim_api.I_X_W
        transposed = False
        input = np.random.uniform(0.01, 0.05, (self.n, self.c, self.in_h, self.in_w)).astype(np.float16)
        self.weights = np.random.uniform(0.01, 0.05, (self.n, self.c, self.in_w, self.out_w)).astype(np.float16)
        golden = np.matmul(input, self.weights)
        output = np.zeros_like(golden, dtype=np.float16)

        desc = pim_api.PimCreateGemmDesc(self.n, self.c, self.in_h, self.in_w, self.out_h, self.out_w, pim_api.PIM_FP16, gemm_order)
        host_in = pim_api.PimCreateBo(desc, pim_api.MEM_TYPE_HOST, pim_api.GEMM_INPUT, input.__array_interface__['data'][0] , transposed);
        host_weight = pim_api.PimCreateBo(desc, pim_api.MEM_TYPE_HOST, pim_api.GEMM_WEIGHT, self.weights.__array_interface__['data'][0] , transposed);
        host_out = pim_api.PimCreateBo(desc, pim_api.MEM_TYPE_HOST, pim_api.GEMM_OUTPUT, output.__array_interface__['data'][0] , transposed)
        pim_in = pim_api.PimCreateBo(desc, pim_api.MEM_TYPE_DEVICE, pim_api.GEMM_INPUT, 0, transposed)
        pim_out = pim_api.PimCreateBo(desc, pim_api.MEM_TYPE_DEVICE, pim_api.GEMM_OUTPUT, 0, transposed)

        pim_api.PimCopyMemory(pim_in, host_in, pim_api.HOST_TO_DEVICE)
        pim_api.PimExecuteGemm(pim_out, pim_in, host_weight, None, pim_api.NONE, gemm_order, None, True)
        pim_api.PimCopyMemory(host_out, pim_out, pim_api.DEVICE_TO_HOST)

        self.assertEqual(np.allclose(golden, pim_out, rtol=0, atol=1e-01, equal_nan=False), True, "All Valaues should be equal")

    def tearDown(self):
        pim_api.PimDeinitialize()

if __name__ == '__main__':
    unittest.main()
