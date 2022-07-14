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
            length, 1, 1, 1, pim_api.PIM_FP16, pim_api.MEM_TYPE_DEVICE, input.data_ptr())
        pim_api.PimDestroyBo(dev_input)


    def test_gemvlist(self):
         pim_api.PimInitialize(pim_api.RT_TYPE_HIP, pim_api.PIM_FP16)
         in_size = 256
         out_size = 512
         moe_cnt = 8
         block = True

         dev_input = pim_api.PimCreateBo(in_size, 1, 1, moe_cnt, pim_api.PIM_FP16, pim_api.MEM_TYPE_DEVICE, 0);
         dev_weight = pim_api.PimCreateBo(in_size, out_size, 1, moe_cnt, pim_api.PIM_FP16, pim_api.MEM_TYPE_DEVICE, 0);
         dev_output = pim_api.PimCreateBo(out_size, 1, 1, moe_cnt, pim_api.PIM_FP16, pim_api.MEM_TYPE_DEVICE, 0);

         pim_api.PimExecuteGemvList(dev_output, dev_input, dev_weight, None, block);
         pim_api.PimSynchronize(None);

         pim_api.PimDestroyBo(dev_input)
         pim_api.PimDestroyBo(dev_weight)
         pim_api.PimDestroyBo(dev_output)

if __name__ == "__main__":
    unittest.main()
