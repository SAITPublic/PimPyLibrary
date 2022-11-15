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

class PimGemmFunction(Function):
    @staticmethod
    def forward(ctx, inputs, weights, bias, act, gemm_order=pim_api.I_X_W, block=True):

        transposed = False
        if inputs.ndim not in [4]:
            print("Input dimension not supported in Gemm")
            return

        if weights.ndim not in [4]:
            print("Weight dimension not supported in Gemm")
            return

        batch = inputs.size()[0]
        channel = inputs.size()[1]
        inout_h = inputs.size()[2]
        in_w = inputs.size()[3]
        out_w = weights.size()[3]

        out_tensor = torch.empty(
            (batch, channel, inout_h, out_w), dtype=torch.float16, device=inputs.device)

        #print('Custom op pimgemm descriptor (n, c, inout_h, in_w, out_w)', batch, channel, inout_h, in_w, out_w)
        pim_gemm_desc = pim_api.PimCreateGemmDesc(batch, channel, inout_h, in_w, inout_h, out_w, pim_api.PIM_FP16, gemm_order)
        device_input = pim_api.PimCreateBo(pim_gemm_desc, pim_api.MEM_TYPE_DEVICE, pim_api.GEMM_INPUT, inputs.data_ptr(), transposed)
        device_weight = pim_api.PimCreateBo(pim_gemm_desc, pim_api.MEM_TYPE_DEVICE, pim_api.GEMM_WEIGHT, weights.data_ptr(), transposed)
        device_bias = pim_api.PimCreateBo(pim_gemm_desc, pim_api.MEM_TYPE_DEVICE, pim_api.GEMM_BIAS, bias.data_ptr(), transposed)
        device_output = pim_api.PimCreateBo(pim_gemm_desc, pim_api.MEM_TYPE_DEVICE, pim_api.GEMM_OUTPUT, out_tensor.data_ptr(), transposed)
        pim_api.PimExecuteGemm(device_output, device_input, device_weight, device_bias, act, gemm_order, None, block)

        pim_api.PimDestroyBo(device_input)
        pim_api.PimDestroyBo(device_weight)
        pim_api.PimDestroyBo(device_bias)
        pim_api.PimDestroyBo(device_output)
        pim_api.PimDestroyGemmDesc(pim_gemm_desc)

        return out_tensor

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError

class PimGemm(nn.Module):

    def __init__(self,device=None, dtype=None) -> None:
        super(PimGemm, self).__init__()

    def reset_parameters(self) -> None:
        super(PimGemm, self).reset_parameters()

    def __repr__(self):
        return "PIM Gemm layer"

    def forward(self, inputs, weight, bias, act, gemm_order=pim_api.I_X_W, block=True):
        return PimPimGemmFunction.apply(inputs, weight, bias, act, gemm_order, block)
