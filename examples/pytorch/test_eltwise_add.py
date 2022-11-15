# Copyright (C) 2021 Samsung Electronics Co. LTD

# This software is a property of Samsung Electronics.
# No part of this software, either material or conceptual may be copied or distributed, transmitted,
# transcribed, stored in a retrieval system, or translated into any human or computer language in any form by any means,
# electronic, mechanical, manual or otherwise, or disclosed
# to third parties without the express written permission of Samsung Electronics.
# (Use of the Software is restricted to non-commercial, personal or academic, research purpose only)

import unittest
import torch
import torch.nn as nn
from torch.autograd import Function
import pim_api
from pim_pytorch.pim_eltwise import PimEltwiseFunction as pim_elt
from pim_pytorch.pim_eltwise import PimEltwise


class PyEltAddTest(unittest.TestCase):
    def test_add_func(self):
        pim_api.PimInitialize(pim_api.RT_TYPE_HIP, pim_api.PIM_FP16)
        with torch.no_grad():
            gpu0 = torch.device(0)
            input0 = torch.rand((128 * 1024), dtype=torch.float16, device=gpu0)
            input1 = torch.rand((128 * 1024), dtype=torch.float16, device=gpu0)
            add = torch.tensor([0], dtype=torch.int32, device=gpu0)

            pim_result = pim_elt.apply(input0, input1, add)
            true_result = input0 + input1
            # print(true_result)
            # print(result)
            self.assertTrue(torch.allclose(pim_result, true_result, atol=0.01))

    def test_add_layer(self):
        pim_api.PimInitialize(pim_api.RT_TYPE_HIP, pim_api.PIM_FP16)
        with torch.no_grad():
            gpu0 = torch.device(0)
            input0 = torch.rand((128, 1024), dtype=torch.float16, device=gpu0)
            input1 = torch.rand(1024, dtype=torch.float16, device=gpu0)
            add = torch.tensor([0], dtype=torch.int32, device=gpu0)
            pim_eltwise_layer = PimEltwise(0)  # 0 for 1 , 1 for mul
            pim_result = pim_eltwise_layer(input0, input1)
            true_result = input0 + input1
            # print(true_result)
            # print(result)
            self.assertTrue(torch.allclose(pim_result, true_result, atol=0.01))


if __name__ == "__main__":
    unittest.main()
