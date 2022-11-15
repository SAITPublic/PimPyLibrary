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
from pim_pytorch.pim_dense import PimDenseFunction as pim_dense
from pim_pytorch.pim_dense import PimDense


class PyDenseTest(unittest.TestCase):

    def getTranspose(self, weights):
        weights_t = weights.clone().detach()
        weights_t = torch.transpose(weights_t, 0, 1).contiguous()
        return weights_t


    def testDense(self):
        with torch.no_grad():
            pim_api.PimInitialize(pim_api.RT_TYPE_HIP, pim_api.PIM_FP16)
            n_batch = 4
            in_size = 1024
            out_size = 4096
            device = torch.device('cuda')

            input = torch.rand(size=(n_batch, in_size), dtype=torch.float16)
            input = input.to(device)

            dense = nn.Linear(in_size, out_size, bias=False)
            dense = dense.to(device)
            dense.half()
            #dense.weight.fill_(0.1)

            # Pass the tensor to pytorch model and obtain output
            pytorch_result = None
            pytorch_result = dense(input)
            weights_t = self.getTranspose(dense.weight)
            bias = None
            bias = torch.zeros_like(pytorch_result)
            pim_result = pim_dense.apply(input, weights_t, bias)  # Obtain PIM output

            #print(torch.max(torch.abs(pim_result - pytorch_result)))
            #print("Pytorch Result:", pytorch_result)
            #print("PIM Result:", pim_result)
            #print("Weights:", weights_t)
            self.assertTrue(torch.allclose(pim_result, pytorch_result, atol=0.5))


    def testDense2(self):
        pim_api.PimInitialize(pim_api.RT_TYPE_HIP, pim_api.PIM_FP16)
        in_batch = 1
        in_size = 1024
        out_size = 4096

        with torch.no_grad():
            device = torch.device('cuda')
            input = torch.rand(size=(in_batch, in_size), dtype=torch.float16)
            input = input.to(device)

            pim_dense_layer = PimDense(in_size, out_size)
            pim_dense_layer.to(device)
            pim_dense_layer.half()

            dense = nn.Linear(in_size, out_size)
            dense = dense.to(device)
            dense.half()
            #dense.weight.fill_(0.0)

            # Pass the tensor to pytorch model and obtain output
            pytorch_result = dense(input)
            bias = dense.bias.clone().detach()
            weights_t = self.getTranspose(dense.weight)  # Weight copy to be used in PIM computation.
            pim_dense_layer.weight.copy_(weights_t)
            pim_dense_layer.bias.copy_(bias)

            pim_result = pim_dense_layer(input)  # Obtain PIM output
            #print("Pytorch Result:", pytorch_result, pytorch_result.shape)
            #print("PIM Result:", pim_result, pim_result.shape)
            self.assertTrue(torch.allclose(pim_result, pytorch_result, atol=0.5))


if __name__ == "__main__":
    torch.manual_seed(2)
    unittest.main()
