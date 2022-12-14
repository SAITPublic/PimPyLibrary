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


class PimFusedFFNFunction(Function):
    @staticmethod
    def forward(ctx, inputs, fc1_w, fc1_bias, fc2_w, fc2_bias, gemm_order=pim_api.I_X_W, block=True):

        transposed = False
        input_dims = inputs.ndim
        if inputs.ndim not in [4]:
            print("Input dimension not supported in Gemm")
            return

        if fc1_w.ndim not in [4] or fc2_w.ndim not in [4]:
            print("Weight dimension not supported in Gemm")
            return

        #--first ffn-------------
        batch = inputs.size()[0]
        channel = inputs.size()[1]
        inout_h = inputs.size()[2]
        in_w = inputs.size()[3]
        out_w = fc1_w.size()[3]
        out_tensor = torch.empty(
                (batch, channel, inout_h, out_w), dtype=torch.float16, device=inputs.device)

        pim_gemm_desc = pim_api.PimCreateGemmDesc(batch, channel, inout_h, in_w, inout_h, out_w, pim_api.PIM_FP16, gemm_order)
        device_input = pim_api.PimCreateBo(pim_gemm_desc, pim_api.MEM_TYPE_DEVICE, pim_api.GEMM_INPUT, inputs.data_ptr(), transposed)
        device_weight = pim_api.PimCreateBo(pim_gemm_desc, pim_api.MEM_TYPE_DEVICE, pim_api.GEMM_WEIGHT, fc1_w.data_ptr(), transposed)
        device_bias = pim_api.PimCreateBo(pim_gemm_desc, pim_api.MEM_TYPE_DEVICE, pim_api.GEMM_BIAS, fc1_bias.data_ptr(), transposed)
        device_output = pim_api.PimCreateBo(pim_gemm_desc, pim_api.MEM_TYPE_DEVICE, pim_api.GEMM_OUTPUT, out_tensor.data_ptr(), transposed)
        pim_api.PimExecuteGemm(device_output, device_input, device_weight, device_bias, pim_api.ACT_RELU, gemm_order, None, block)

        pim_api.PimDestroyBo(device_weight)
        pim_api.PimDestroyBo(device_input)
        pim_api.PimDestroyBo(device_bias)
        pim_api.PimDestroyBo(device_output)
        pim_api.PimDestroyGemmDesc(pim_gemm_desc)

        #--second ffn-------------
        in_w = fc1_w.size()[3]
        out_w = fc2_w.size()[3]
        o2 = torch.empty(
            (batch, channel, inout_h, out_w), dtype=torch.float16, device=inputs.device)

        pim_gemm_desc = pim_api.PimCreateGemmDesc(batch, channel, inout_h, in_w, inout_h, out_w, pim_api.PIM_FP16, gemm_order);
        device_input = pim_api.PimCreateBo(pim_gemm_desc, pim_api.MEM_TYPE_DEVICE, pim_api.GEMM_INPUT, out_tensor.data_ptr(), transposed);
        device_weight = pim_api.PimCreateBo(pim_gemm_desc, pim_api.MEM_TYPE_DEVICE, pim_api.GEMM_WEIGHT, fc2_w.data_ptr(), transposed);
        device_bias = pim_api.PimCreateBo(pim_gemm_desc, pim_api.MEM_TYPE_DEVICE, pim_api.GEMM_BIAS, fc2_bias.data_ptr(), transposed);
        device_output = pim_api.PimCreateBo(pim_gemm_desc, pim_api.MEM_TYPE_DEVICE, pim_api.GEMM_OUTPUT, o2.data_ptr(), transposed);
        pim_api.PimExecuteGemm(device_output, device_input, device_weight, device_bias, pim_api.NONE, gemm_order, None, block);

        pim_api.PimDestroyBo(device_weight)
        pim_api.PimDestroyBo(device_input)
        pim_api.PimDestroyBo(device_bias)
        pim_api.PimDestroyBo(device_output)
        pim_api.PimDestroyGemmDesc(pim_gemm_desc)

        return o2

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError


class PimFusedFFN(nn.Linear):
    """A nn.module wrapper for py_pim_dense function.
    """

    def __init__(self, device=None, dtype=None) -> None:
        super(PimFusedFFN, self).__init__()

    def reset_parameters(self) -> None:
        super(PimFusedFFN, self).reset_parameters()

    def __repr__(self):
        return "PIM Fused FFN layer"

    def forward(self, x, batched_fc1_w, batched_fc1_bias, batched_fc2_w, batched_fc2_bias, gemm_order=pim_api.I_X_W, block=True):
        return PimFusedFFNFunction.apply(x, batched_fc1_w, batched_fc1_bias, batched_fc2_w, batched_fc2_bias, gemm_order, block)
