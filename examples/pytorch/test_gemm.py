import unittest
import torch
import torch.nn as nn
from torch.autograd import Function
import pim_api
from pim_pytorch.pim_gemm import PimGemmFunction as pim_gemm
from pim_pytorch.pim_gemm import PimGemm


class PyGemmTest(unittest.TestCase):
    #just for debugging ,to be removed
    def get_weight(self, weight, rows, cols):
        weight = torch.zeros_like(weight)
        channel = weight.size()[0]
        for c in range(channel):
            for i in range(5):
                for j in range(5):
                    weight[c][i][j] = i
        return weight

    def config_test(self, batch, channel, inout_h, in_w, out_w, block):

      with torch.no_grad():
        device = torch.device('cuda')
        relu = nn.ReLU()
        input = torch.rand(size=(batch, channel, inout_h, in_w),
                           dtype=torch.float16, device=device)
        weight = torch.rand(size=(batch, channel, in_w, out_w),
                             dtype=torch.float16, device=device)
        bias = torch.rand(size=(batch, channel, inout_h, out_w),
                          dtype=torch.float16, device=device)

        #scale and uniform[ -1 to 1 ]
        sc = 2.0
        input = sc * input - sc / 2.0
        weight = sc * weight - sc / 2.0
        assert(input.is_contiguous())
        assert(weight.is_contiguous())
        assert(bias.is_contiguous())
        output = torch.matmul(input, weight)
        output += bias
        output = relu(output)
        pim_result = pim_gemm.apply(input, weight, bias, pim_api.ACT_RELU, pim_api.I_X_W, block)
        return pim_result, output

    def testGemm_2x1x1x1024_2x1x1024x4096(self):
        pim_api.PimInitialize(pim_api.RT_TYPE_HIP, pim_api.PIM_FP16)
        batch = 2
        channel = 1
        inout_h = 1
        in_w = 1024
        out_w = 4096
        pim_result, pytorch_result = self.config_test(batch, channel, inout_h, in_w, out_w, True)
        self.assertTrue(torch.allclose(pim_result, pytorch_result, atol=0.1))
        pim_api.PimDeinitialize()

    def testGemm_1x1x1x1024_1x1x1024x4096(self):
        pim_api.PimInitialize(pim_api.RT_TYPE_HIP, pim_api.PIM_FP16)
        batch = 1
        channel = 1
        inout_h = 1
        in_w = 1024
        out_w = 4096
        pim_result, pytorch_result = self.config_test(batch, channel, inout_h, in_w, out_w, True)
        self.assertTrue(torch.allclose(pim_result, pytorch_result, atol=0.1))
        pim_api.PimDeinitialize()

    def testGemm_1x1x8x1024_1x1x1024x4096(self):
        pim_api.PimInitialize(pim_api.RT_TYPE_HIP, pim_api.PIM_FP16)
        batch = 1
        channel = 1
        inout_h = 8
        in_w = 1024
        out_w = 4096
        pim_result, pytorch_result = self.config_test(batch, channel, inout_h, in_w, out_w, True)
        self.assertTrue(torch.allclose(pim_result, pytorch_result, atol=0.1))
        pim_api.PimDeinitialize()

    def testGemm_1x8x1x4096_1x8x4096x1024(self):
        pim_api.PimInitialize(pim_api.RT_TYPE_HIP, pim_api.PIM_FP16)
        batch = 1
        channel = 8
        inout_h = 1
        in_w = 4096
        out_w = 1024
        pim_result, pytorch_result = self.config_test(batch, channel, inout_h, in_w, out_w, True)
        self.assertTrue(torch.allclose(pim_result, pytorch_result, atol=1.5))
        pim_api.PimDeinitialize()


    def testGemm_1x4x8x1024_1x4x1024x4096(self):
        pim_api.PimInitialize(pim_api.RT_TYPE_HIP, pim_api.PIM_FP16)
        batch = 1
        channel = 4
        inout_h = 8
        in_w = 1024
        out_w = 4096
        pim_result, pytorch_result = self.config_test(batch, channel, inout_h, in_w, out_w, True)
        self.assertTrue(torch.allclose(pim_result, pytorch_result, atol=0.5))
        pim_api.PimDeinitialize()

    def testGemm_1x4x1x4096_1x4x4096x1024(self):
        pim_api.PimInitialize(pim_api.RT_TYPE_HIP, pim_api.PIM_FP16)
        batch = 1
        channel = 4
        inout_h = 1
        in_w = 4096
        out_w = 1024
        pim_result, pytorch_result = self.config_test(batch, channel, inout_h, in_w, out_w, True)
        self.assertTrue(torch.allclose(pim_result, pytorch_result, atol=0.5))
        pim_api.PimDeinitialize()

    def testGemm_1x64x1x256_1x64x256x64(self):
        pim_api.PimInitialize(pim_api.RT_TYPE_HIP, pim_api.PIM_FP16)
        batch = 1
        channel = 64
        inout_h = 1
        in_w = 256
        out_w = 64
        pim_result, pytorch_result = self.config_test(batch, channel, inout_h, in_w, out_w, True)
        self.assertTrue(torch.allclose(pim_result, pytorch_result, atol=0.1))
        pim_api.PimDeinitialize()

    def testGemm_1x64x1x1024_1x64x1024x64(self):
        pim_api.PimInitialize(pim_api.RT_TYPE_HIP, pim_api.PIM_FP16)
        batch = 1
        channel = 64
        inout_h = 1
        in_w = 1024
        out_w = 64
        pim_result, pytorch_result = self.config_test(batch, channel, inout_h, in_w, out_w, True)
        self.assertTrue(torch.allclose(pim_result, pytorch_result, atol=0.1))
        pim_api.PimDeinitialize()
# fail TODO:check
    def _testGemm_1x4x8x4096_1x4x4096x1024(self):
        pim_api.PimInitialize(pim_api.RT_TYPE_HIP, pim_api.PIM_FP16)
        batch = 1
        channel = 4
        inout_h = 8
        in_w = 4096
        out_w = 1024
        pim_result, pytorch_result = self.config_test(batch, channel, inout_h, in_w, out_w, True)
        self.assertTrue(torch.allclose(pim_result, pytorch_result, atol=0.1))
        pim_api.PimDeinitialize()

if __name__ == "__main__":
    torch.set_printoptions(edgeitems=10)
    torch.manual_seed(2)
    unittest.main()

