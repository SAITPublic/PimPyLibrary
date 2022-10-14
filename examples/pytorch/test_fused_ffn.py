import unittest
import torch
import torch.nn as nn
from torch.autograd import Function
import pim_api
from pim_pytorch.pim_fused_ffn import PimFusedFFNFunction as pim_fused_ffn
from pim_pytorch.pim_fused_ffn import PimFusedFFN


class PyFusedFFNTest(unittest.TestCase):
    def config_test(self, batch, channel, inout_h, in_w1, out_w1, in_w2, out_w2):
        relu = nn.ReLU()
        device = torch.device('cuda')

        input = torch.rand(size=(batch, channel, inout_h, in_w1),
                           dtype=torch.float16, device=device)
        w1 = torch.rand(size=(batch, channel, in_w1, out_w1),
                        dtype=torch.float16, device=device)
        b1 = torch.rand(size=(batch, channel, inout_h, out_w1),
                        dtype=torch.float16, device=device)

        w2 = torch.rand(size=(batch, channel, in_w2, out_w2),
                              dtype=torch.float16, device=device)
        b2 = torch.rand(size=(batch, channel, inout_h, out_w2),
                              dtype=torch.float16, device=device)

        #scale and uniform[ -1 to 1 ]
        sc = 0.2
        input = sc * input - sc / 2.0
        w1 = sc * w1 - sc / 2.0
        w2 = sc * w2 - sc / 2.0

        #ffn1
        output = b1.clone()
        output += torch.matmul(input, w1)
        output = relu(output)
        #print('Pytorch ffn1:', output)

        #ffn2
        pytorch_result = b2.clone()
        pytorch_result += torch.matmul(output, w2)

        pim_result = pim_fused_ffn.apply(input, w1, b1, w2, b2, pim_api.I_X_W)
        return pim_result, pytorch_result


    def test_fused_ffn_1x4x1x1024_1x4x1024x4096_1x4x4096x1024(self):
        pim_api.PimInitialize(pim_api.RT_TYPE_HIP, pim_api.PIM_FP16)
        batch = 1
        channel = 4
        inout_h = 1
        in_w1 = 1024
        out_w1 = 4096
        in_w2 = 4096
        out_w2 = 1024
        pim_result, pytorch_result = self.config_test(batch, channel, inout_h, in_w1, out_w1, in_w2, out_w2)
        self.assertTrue(torch.allclose(pim_result, pytorch_result, atol=0.5))
        pim_api.PimDeinitialize()

    def test_fused_ffn_1x8x1x1024_1x8x1024x4096_1x8x4096x1024(self):
        pim_api.PimInitialize(pim_api.RT_TYPE_HIP, pim_api.PIM_FP16)
        batch = 1
        channel = 8
        inout_h = 1
        in_w1 = 1024
        out_w1 = 4096
        in_w2 = 4096
        out_w2 = 1024
        pim_result, pytorch_result = self.config_test(batch, channel, inout_h, in_w1, out_w1, in_w2, out_w2)
        self.assertTrue(torch.allclose(pim_result, pytorch_result, atol=1.0))
        pim_api.PimDeinitialize()

    #to be verified
    def _test_fused_ffn_1x4x8x1024_1x4x1024x4096_1x4x4096x1024(self):
        pim_api.PimInitialize(pim_api.RT_TYPE_HIP, pim_api.PIM_FP16)
        batch = 1
        channel = 4
        inout_h = 8
        in_w1 = 1024
        out_w1 = 4096
        in_w2 = 4096
        out_w2 = 1024
        pim_result, pytorch_result = self.config_test(batch, channel, inout_h, in_w1, out_w1, in_w2, out_w2)
        self.assertTrue(torch.allclose(pim_result, pytorch_result, atol=0.5))
        pim_api.PimDeinitialize()

if __name__ == "__main__":
    torch.manual_seed(2)
    unittest.main()
