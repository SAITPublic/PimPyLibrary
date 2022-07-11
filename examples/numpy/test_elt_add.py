import unittest
import numpy as np
import pim_api

class TestEltAdd(unittest.TestCase):
    def setUp(self):
        self.length = 128 * 1024
        self.input1 = np.random.normal(0, 0.05, self.length).astype(np.float16)
        self.input2 = np.random.normal(0, 0.05, self.length).astype(np.float16)
        self.golden = self.input1 + self.input2
        self.pim_out = np.zeros_like(self.golden, dtype=np.float16)

        pim_api.PimInitialize(pim_api.RT_TYPE_HIP, pim_api.PIM_FP16)

    def test_elt_add(self):
        host_input1 = pim_api.PimCreateBo(self.length, 1, 1, 1, pim_api.PIM_FP16, pim_api.MEM_TYPE_HOST, self.input1.__array_interface__['data'][0])
        host_input2 = pim_api.PimCreateBo(self.length, 1, 1, 1, pim_api.PIM_FP16, pim_api.MEM_TYPE_HOST, self.input2.__array_interface__['data'][0])
        host_output = pim_api.PimCreateBo(self.length, 1, 1, 1, pim_api.PIM_FP16, pim_api.MEM_TYPE_HOST, self.pim_out.__array_interface__['data'][0])

        pim_input1 = pim_api.PimCreateBo(self.length, 1, 1, 1, pim_api.PIM_FP16, pim_api.MEM_TYPE_PIM,0)
        pim_input2 = pim_api.PimCreateBo(self.length, 1, 1, 1, pim_api.PIM_FP16, pim_api.MEM_TYPE_PIM,0)
        pim_output = pim_api.PimCreateBo(self.length, 1, 1, 1, pim_api.PIM_FP16, pim_api.MEM_TYPE_PIM,0)

        pim_api.PimCopyMemory(pim_input1, host_input1, pim_api.HOST_TO_PIM)
        pim_api.PimCopyMemory(pim_input2, host_input2, pim_api.HOST_TO_PIM)

        pim_api.PimExecuteAdd(pim_output, pim_input1, pim_input2, None, 1)
        pim_api.PimCopyMemory(host_output, pim_output, pim_api.PIM_TO_HOST)
        equal_arrays = np.allclose(self.golden, self.pim_out, atol=1e-3)

        self.assertEqual(equal_arrays, True, "All Valaues should be equal")

    def tearDown(self):
        pim_api.PimDeinitialize()

if __name__ == '__main__':
    unittest.main()
