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
import numpy as np

class PyApiTest(unittest.TestCase):

    def test_device_set(self):
        dev_cnt = torch.cuda.device_count()
        pim_api.PimInitialize(pim_api.RT_TYPE_HIP, pim_api.PIM_FP16)
        old_dev = np.array([1],dtype=np.uint32)
        ret = pim_api.PimGetDevice(old_dev)

        for i in range(dev_cnt):
          pim_api.PimSetDevice(i)
          dev = np.array([1],dtype=np.uint32)
          ret = pim_api.PimGetDevice(dev)
          self.assertTrue(dev[0] == i)
        pim_api.PimSetDevice(old_dev) #reset back

    def test_buffer_destroy(self):
        pim_api.PimInitialize(pim_api.RT_TYPE_HIP, pim_api.PIM_FP16)
        length = 1024
        input = torch.ones(length,dtype=torch.half)
        dev_input = pim_api.PimCreateBo(
            length, 1, 1, 1, pim_api.PIM_FP16, pim_api.MEM_TYPE_DEVICE, input.data_ptr(), True)
        pim_api.PimDestroyBo(dev_input)

if __name__ == "__main__":
    unittest.main()
