# Copyright (C) 2021 Samsung Electronics Co. LTD

# This software is a property of Samsung Electronics.
# No part of this software, either material or conceptual may be copied or distributed, transmitted,
# transcribed, stored in a retrieval system, or translated into any human or computer language in any form by any means,
# electronic, mechanical, manual or otherwise, or disclosed
# to third parties without the express written permission of Samsung Electronics.
# (Use of the Software is restricted to non-commercial, personal or academic, research purpose only)

import torch
import torch.nn as nn
from torch.autograd import Function
import pim_api

class PimDenseFunction(Function):
    @staticmethod
    def forward(ctx, inputs, weights, bias, gemm_order=pim_api.I_X_W, block=True):

        if inputs.ndim  not in [2,3]:
            print('Input dimension not supported in Dense')
            return

        transposed = False
        num_batch = 1
        num_channels = 1

        #weight is always 2D in dense
        in_w = weights.size()[0]
        out_w = weights.size()[1]

        if inputs.ndim == 2:
           inout_h = inputs.size()[0]
           out_tensor = torch.zeros(
                (inout_h, out_w), dtype=torch.float16, device=inputs.device)

        if inputs.ndim == 3:
           num_batch = inputs.size()[0]
           inout_h = inputs.size()[1]
           out_tensor = torch.zeros(
                (num_channels, inout_h, out_w), dtype=torch.float16, device=inputs.device)

        #print(num_batch, num_channels, inout_h, in_w, out_w)
        bias_data = 0
        if bias is not None:
            bias_data = bias.data_ptr()

        pim_gemm_desc = pim_api.PimCreateGemmDesc(num_batch, num_channels, inout_h, in_w, inout_h, out_w, pim_api.PIM_FP16, gemm_order)
        device_input = pim_api.PimCreateBo(pim_gemm_desc, pim_api.MEM_TYPE_DEVICE, pim_api.GEMM_INPUT, inputs.data_ptr(), transposed)
        device_weight = pim_api.PimCreateBo(pim_gemm_desc, pim_api.MEM_TYPE_DEVICE, pim_api.GEMM_WEIGHT, weights.data_ptr(), transposed)
        device_bias = pim_api.PimCreateBo(pim_gemm_desc, pim_api.MEM_TYPE_DEVICE, pim_api.GEMM_BIAS, bias_data,transposed )
        device_output = pim_api.PimCreateBo(pim_gemm_desc, pim_api.MEM_TYPE_DEVICE, pim_api.GEMM_OUTPUT, out_tensor.data_ptr(), transposed)
        pim_api.PimExecuteGemm(device_output, device_input, device_weight, device_bias, pim_api.NONE, gemm_order, None, block)

        pim_api.PimDestroyBo(device_input)
        pim_api.PimDestroyBo(device_weight)
        pim_api.PimDestroyBo(device_bias)
        pim_api.PimDestroyBo(device_output)
        pim_api.PimDestroyGemmDesc(pim_gemm_desc)

        return out_tensor

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError

class PimDense(nn.Module):
    """A nn.module wrapper for py_pim_dense function.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                  device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(PimDense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((in_features,out_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self) -> None:
        pass

    def __repr__(self):
        return "PIM dense layer"

    def forward(self, inputs):
        return PimDenseFunction.apply(inputs, self.weight, self.bias, pim_api.I_X_W, True)