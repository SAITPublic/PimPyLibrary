import unittest
import torch
import torch.nn as nn
from torch.autograd import Function
import pim_api
from pim_pytorch.pim_fused_ffn import PimFusedFFNFunction as pim_fused_ffn
from pim_pytorch.pim_fused_ffn import PimFusedFFN


class PyFusedFFNTest(unittest.TestCase):


    def config_test(self, n_channels, inout_h, in_w1, out_w1, in_w2, out_w2):
        relu = nn.ReLU()
        device = torch.device('cuda')

        input = torch.rand(size=(n_channels, inout_h, in_w1),
                           dtype=torch.float16, device=device)
        w1 = torch.rand(size=(n_channels, in_w1, out_w1),
                        dtype=torch.float16, device=device)
        b1 = torch.rand(size=(n_channels, inout_h, out_w1),
                        dtype=torch.float16, device=device)

        w2 = torch.rand(size=(n_channels, in_w2, out_w2),
                            dtype=torch.float16, device=device)
        b2 = torch.rand(size=(n_channels, inout_h, out_w2),
                        dtype=torch.float16, device=device)

        # Todo: scale and uniform[ -1 to 1 ]
        sc = 0.2
        input = sc * input - sc/2.0
        w1 = sc * w1 - sc/2.0
        w2 = sc * w2 - sc/2.0

        #ffn1
        output = b1.clone()
        for i in range(n_channels):
            output[i] += torch.matmul(input[i], w1[i])
        output = relu(output)
        #print('Pytorch ffn1:', output)

        #ffn2
        pytorch_result = b2.clone()
        for i in range(n_channels):
            pytorch_result[i] += torch.matmul(output[i], w2[i])

        _w1 = w1.clone()
        _w1 = _w1.permute(0,2,1).contiguous()
        _w2 = w2.clone()
        _w2 = _w2.permute(0,2,1).contiguous()

        pim_result = pim_fused_ffn.apply(input, _w1, b1, _w2, b2, True)
        #print("Pytorch Result:", pytorch_result.shape, pytorch_result)
        #print("PIM Result:", pim_result)
        #for i in range(n_channels):
           #print(torch.max(pim_result[i][0] - pytorch_result[i][0]))
        return pim_result , pytorch_result


    def test_fused_ffn_4x1x1024_4x1024x4096_4x4096x1024(self):

        pim_api.PimInitialize(pim_api.RT_TYPE_HIP, pim_api.PIM_FP16)

        n_batch = 1
        n_channels = 4
        inout_h = 1
        in_w1 = 1024
        out_w1 = 4096
        in_w2 = 4096
        out_w2 = 1024

        pim_result , pytorch_result = self.config_test(n_channels, inout_h, in_w1, out_w1, in_w2, out_w2)
        self.assertTrue(torch.allclose(pim_result, pytorch_result, atol=0.5))
        pim_api.PimDeinitialize()

    def test_fused_ffn_4x8x1024_4x1024x4096_4x4096x1024(self):

        pim_api.PimInitialize(pim_api.RT_TYPE_HIP, pim_api.PIM_FP16)

        n_batch = 1
        n_channels = 4
        inout_h = 8
        in_w1 = 1024
        out_w1 = 4096
        in_w2 = 4096
        out_w2 = 1024

        pim_result , pytorch_result = self.config_test(n_channels, inout_h, in_w1, out_w1, in_w2, out_w2)
        self.assertTrue(torch.allclose(pim_result, pytorch_result, atol=0.5))
        pim_api.PimDeinitialize()

if __name__ == "__main__":
    torch.manual_seed(2)
    unittest.main()
