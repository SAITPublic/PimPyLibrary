import unittest
import torch
import torch.nn as nn
from torch.autograd import Function
import pim_api
from pim_pytorch.pim_relu import PimReluFunction as pim_relu #function
from pim_pytorch.pim_relu import PimRelu #layer
import torch.nn.functional as F

class PyReluTest(unittest.TestCase):
    def test_relu_func(self):
        pim_api.PimInitialize(pim_api.RT_TYPE_HIP, pim_api.PIM_FP16)
        with torch.no_grad():
            gpu0 = torch.device(0)
            input0 = torch.rand((128 * 1024), dtype=torch.float16, device=gpu0)

            pim_result = pim_relu.apply(input0)
            true_result = F.relu(input0)
            self.assertTrue(torch.allclose(pim_result, true_result, atol=0.01))

    def test_relu_layer(self):
        pim_api.PimInitialize(pim_api.RT_TYPE_HIP, pim_api.PIM_FP16)
        with torch.no_grad():
            gpu0 = torch.device(0)
            input0 = torch.rand((128, 1024), dtype=torch.float16, device=gpu0)
            pim_eltwise_layer = PimRelu()

            pim_result = pim_eltwise_layer(input0)
            true_result = F.relu(input0)
            self.assertTrue(torch.allclose(pim_result, true_result, atol=0.01))


if __name__ == "__main__":
    unittest.main()
