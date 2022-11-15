# Copyright (C) 2021 Samsung Electronics Co. LTD

# This software is a property of Samsung Electronics.
# No part of this software, either material or conceptual may be copied or distributed, transmitted,
# transcribed, stored in a retrieval system, or translated into any human or computer language in any form by any means,
# electronic, mechanical, manual or otherwise, or disclosed
# to third parties without the express written permission of Samsung Electronics.
# (Use of the Software is restricted to non-commercial, personal or academic, research purpose only)

import unittest
import numpy as np
import pim_api

class TestRelu(unittest.TestCase):
    def test_relu(self):
        length = 128 * 1024
        input1 = np.random.randn(length).astype(np.float16)
        golden = np.maximum(input1,0.0)
        pim_out = np.zeros_like(golden, dtype=np.float16)

        pim_api.PimInitialize(pim_api.RT_TYPE_HIP, pim_api.PIM_FP16)
        host_input1 = pim_api.PimCreateBo(length, 1, 1, 1, pim_api.PIM_FP16, pim_api.MEM_TYPE_HOST, input1.__array_interface__['data'][0])
        host_output = pim_api.PimCreateBo(length, 1, 1, 1, pim_api.PIM_FP16, pim_api.MEM_TYPE_HOST, pim_out.__array_interface__['data'][0])

        pim_input1 = pim_api.PimCreateBo(length, 1, 1, 1, pim_api.PIM_FP16, pim_api.MEM_TYPE_PIM,0)
        pim_output = pim_api.PimCreateBo(length, 1, 1, 1, pim_api.PIM_FP16, pim_api.MEM_TYPE_PIM,0)

        pim_api.PimCopyMemory(pim_input1, host_input1, pim_api.HOST_TO_PIM)

        pim_api.PimExecuteRelu(pim_output, pim_input1, None, 1)
        pim_api.PimCopyMemory(host_output, pim_output, pim_api.PIM_TO_HOST)

        comparison = pim_out == golden
        equal_arrays = comparison.all()

        self.assertEqual(equal_arrays, True, "All Values should be equal")

if __name__ == '__main__':
    unittest.main()
