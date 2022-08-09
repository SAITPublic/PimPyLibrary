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
        n_channels = weight.size()[0]
        for c in range(n_channels):
            for i in range(5):
                for j in range(5):
                    weight[c][i][j] = i
        return weight

    def config_test(self, n_channels, inout_h, in_w, out_w, block, make_input_4D=False):

      with torch.no_grad():
        device = torch.device('cuda')
        relu = nn.ReLU()
        input = torch.rand(size=(n_channels, inout_h, in_w),
                           dtype=torch.float16, device=device)
        weights = torch.rand(size=(n_channels, in_w, out_w),
                             dtype=torch.float16, device=device)
        bias = torch.rand(size=(n_channels, inout_h, out_w),
                          dtype=torch.float16, device=device)

        if make_input_4D == True:
            input = torch.unsqueeze(input,0)
            bias = torch.unsqueeze(bias,0)

        #scale and uniform[ -1 to 1 ]
        sc = 2.0
        input = sc * input - sc/2.0
        weights = sc * weights - sc/2.0
        assert(input.is_contiguous())
        assert(weights.is_contiguous())
        assert(bias.is_contiguous())
        output = torch.matmul(input, weights)
        output += bias
        output = relu(output)
        pim_result = pim_gemm.apply(input, weights, bias, pim_api.ACT_RELU, block)
        return pim_result, output

    def testGemm_1x1024_1024x4096_4Dinput(self):
        pim_api.PimInitialize(pim_api.RT_TYPE_HIP, pim_api.PIM_FP16)
        n_channels = 1
        inout_h = 1
        in_w = 1024
        out_w = 4096
        pim_result, pytorch_result = self.config_test(n_channels, inout_h, in_w, out_w, True, True)
        self.assertTrue(torch.allclose(pim_result, pytorch_result, atol=0.1))
        pim_api.PimDeinitialize()

    def testGemm_1x1024_1024x4096(self):
        pim_api.PimInitialize(pim_api.RT_TYPE_HIP, pim_api.PIM_FP16)
        n_channels = 1
        inout_h = 1
        in_w = 1024
        out_w = 4096
        pim_result, pytorch_result = self.config_test(n_channels, inout_h, in_w, out_w, True)
        self.assertTrue(torch.allclose(pim_result, pytorch_result, atol=0.1))
        pim_api.PimDeinitialize()

    def testGemm_8x1024_1024x4096(self):
        pim_api.PimInitialize(pim_api.RT_TYPE_HIP, pim_api.PIM_FP16)
        n_channels = 1
        inout_h = 8
        in_w = 1024
        out_w = 4096
        pim_result, pytorch_result = self.config_test(n_channels, inout_h, in_w, out_w, True)
        self.assertTrue(torch.allclose(pim_result, pytorch_result, atol=0.1))
        pim_api.PimDeinitialize()

    def testGemm_8x1x4096_4096x1024(self):
        pim_api.PimInitialize(pim_api.RT_TYPE_HIP, pim_api.PIM_FP16)
        n_channels = 8
        inout_h = 1
        in_w = 4096
        out_w = 1024
        pim_result, pytorch_result = self.config_test(n_channels, inout_h, in_w, out_w, True)
        self.assertTrue(torch.allclose(pim_result, pytorch_result, atol=1.5))
        pim_api.PimDeinitialize()


    def testGemm_4x8x1024_4x1024x4096(self):
        pim_api.PimInitialize(pim_api.RT_TYPE_HIP, pim_api.PIM_FP16)
        n_channels = 4
        inout_h = 8
        in_w = 1024
        out_w = 4096
        pim_result, pytorch_result = self.config_test(n_channels, inout_h, in_w, out_w, True)
        self.assertTrue(torch.allclose(pim_result, pytorch_result, atol=0.5))
        pim_api.PimDeinitialize()

    def testGemm_4x1x4096_4x4096x1024(self):
        pim_api.PimInitialize(pim_api.RT_TYPE_HIP, pim_api.PIM_FP16)
        n_channels = 4
        inout_h = 1
        in_w = 4096
        out_w = 1024
        pim_result, pytorch_result = self.config_test(n_channels, inout_h, in_w, out_w, True)
        self.assertTrue(torch.allclose(pim_result, pytorch_result, atol=0.5))
        pim_api.PimDeinitialize()

    def testGemm_64x1x256_64x256x64(self):
        pim_api.PimInitialize(pim_api.RT_TYPE_HIP, pim_api.PIM_FP16)
        n_channels = 64
        inout_h = 1
        in_w = 256
        out_w = 64
        pim_result, pytorch_result = self.config_test(n_channels, inout_h, in_w, out_w, True)
        self.assertTrue(torch.allclose(pim_result, pytorch_result, atol=0.1))
        pim_api.PimDeinitialize()

    def testGemm_64x1x1024_64x1024x64(self):
        pim_api.PimInitialize(pim_api.RT_TYPE_HIP, pim_api.PIM_FP16)
        n_channels = 64
        inout_h = 1
        in_w = 1024
        out_w = 64
        pim_result, pytorch_result = self.config_test(n_channels, inout_h, in_w, out_w, True)
        self.assertTrue(torch.allclose(pim_result, pytorch_result, atol=0.1))
        pim_api.PimDeinitialize()
# fail TODO:check
    def _testGemm_4x8x4096_4x4096x1024(self):
        pim_api.PimInitialize(pim_api.RT_TYPE_HIP, pim_api.PIM_FP16)
        n_channels = 4
        inout_h = 8
        in_w = 4096
        out_w = 1024
        pim_result, pytorch_result = self.config_test(n_channels, inout_h, in_w, out_w, True)
        self.assertTrue(torch.allclose(pim_result, pytorch_result, atol=0.1))
        pim_api.PimDeinitialize()

if __name__ == "__main__":
    torch.set_printoptions(edgeitems=10)
    torch.manual_seed(2)
    unittest.main()

