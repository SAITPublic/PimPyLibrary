import unittest
import numpy as np
import pim_api

class TestGemv(unittest.TestCase):
    def setUp(self):
        self.input_len = 256
        self.output_len = 4096
        self.weights = np.random.uniform(0.01, 0.05, (self.input_len, self.output_len)).astype(np.float16)
        pim_api.PimInitialize(pim_api.RT_TYPE_HIP, pim_api.PIM_FP16)

    def test_gemv(self):
        input = np.random.uniform(0.01, 0.05, (1, 1, 1, self.input_len)).astype(np.float16)
        golden = np.matmul(input, self.weights)
        pim_out = np.zeros_like(golden, dtype=np.float16)

        host_input = pim_api.PimCreateBo(self.input_len, 1, 1, 1, pim_api.PIM_FP16, pim_api.MEM_TYPE_HOST, input.__array_interface__['data'][0])
        host_weight = pim_api.PimCreateBo(self.input_len, self.output_len, 1, 1, pim_api.PIM_FP16, pim_api.MEM_TYPE_HOST, self.weights.__array_interface__['data'][0])
        host_output = pim_api.PimCreateBo(self.output_len, 1, 1, 1, pim_api.PIM_FP16, pim_api.MEM_TYPE_HOST, pim_out.__array_interface__['data'][0])
        pim_input = pim_api.PimCreateBo(self.input_len, 1, 1, 1, pim_api.PIM_FP16, pim_api.MEM_TYPE_DEVICE,0)
        pim_output = pim_api.PimCreateBo(self.output_len, 1, 1, 1, pim_api.PIM_FP16, pim_api.MEM_TYPE_DEVICE,0)

        pim_api.PimCopyMemory(pim_input, host_input, pim_api.HOST_TO_DEVICE)
        pim_api.PimExecuteGemv(pim_output, pim_input, host_weight, None, 1)
        pim_api.PimCopyMemory(host_output, pim_output, pim_api.DEVICE_TO_HOST)

        self.assertEqual(np.allclose(golden, pim_out, rtol=0, atol=1e-01, equal_nan=False), True, "All Valaues should be equal")

    def test_gemv_batch(self):
        batch_dim = 4
        input = np.random.uniform(0.01, 0.05, (batch_dim, 1, 1, self.input_len)).astype(np.float16)
        golden = np.matmul(input, self.weights)
        pim_out = np.zeros_like(golden, dtype=np.float16)

        host_input = pim_api.PimCreateBo(self.input_len, 1, 1, batch_dim, pim_api.PIM_FP16, pim_api.MEM_TYPE_HOST, input.__array_interface__['data'][0])
        host_weight = pim_api.PimCreateBo(self.input_len, self.output_len, 1, 1, pim_api.PIM_FP16, pim_api.MEM_TYPE_HOST, self.weights.__array_interface__['data'][0])
        host_output = pim_api.PimCreateBo(self.output_len, 1, 1, batch_dim, pim_api.PIM_FP16, pim_api.MEM_TYPE_HOST, pim_out.__array_interface__['data'][0])

        pim_input = pim_api.PimCreateBo(self.input_len, 1, 1, batch_dim, pim_api.PIM_FP16, pim_api.MEM_TYPE_DEVICE,0)
        pim_output = pim_api.PimCreateBo(self.output_len, 1, 1, batch_dim, pim_api.PIM_FP16, pim_api.MEM_TYPE_DEVICE,0)

        pim_api.PimCopyMemory(pim_input, host_input, pim_api.HOST_TO_DEVICE)
        pim_api.PimExecuteGemv(pim_output, pim_input, host_weight, None, 1)
        pim_api.PimCopyMemory(host_output, pim_output, pim_api.DEVICE_TO_HOST)

        self.assertEqual(np.allclose(golden, pim_out, rtol=0, atol=1e-01, equal_nan=False), True, "All Valaues should be equal")

if __name__ == '__main__':
    unittest.main()
