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



class PimReluFunction(Function):
    @staticmethod
    def forward(ctx, input):
        length = torch.numel(input)
        out_tensor = torch.empty(
            input.size(), dtype=torch.float16, device=input.device)

        dev_input = pim_api.PimCreateBo(
            1, 1, 1, length, pim_api.PIM_FP16, pim_api.MEM_TYPE_DEVICE, input.data_ptr(), False)
        dev_output = pim_api.PimCreateBo(
            1, 1, 1, length, pim_api.PIM_FP16, pim_api.MEM_TYPE_DEVICE, out_tensor.data_ptr(), False)

        pim_input = pim_api.PimCreateBo(
            1, 1, 1, length, pim_api.PIM_FP16, pim_api.MEM_TYPE_PIM, 0, False)
        pim_output = pim_api.PimCreateBo(
            1, 1, 1, length, pim_api.PIM_FP16, pim_api.MEM_TYPE_PIM, 0, False)

        pim_api.PimCopyMemory(pim_input, dev_input, pim_api.DEVICE_TO_PIM)

        pim_api.PimExecuteRelu(pim_output, pim_input, None, 1)
        pim_api.PimCopyMemory(dev_output, pim_output, pim_api.PIM_TO_DEVICE)


        pim_api.PimDestroyBo(dev_input)
        pim_api.PimDestroyBo(dev_output)
        pim_api.PimDestroyBo(pim_input)
        pim_api.PimDestroyBo(pim_output)

        return out_tensor

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError


class PimRelu(nn.Module):
    """A nn.module wrapper for py_pim_eltwise function.
    """

    def __init__(self, operation=0):
        super(PimRelu, self).__init__()

    def __repr__(self):
        return "Pim Relu Layer"

    def forward(self, input):
        return PimReluFunction.apply(input)
