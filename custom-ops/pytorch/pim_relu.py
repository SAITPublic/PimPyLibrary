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
            length, 1, 1, 1, pim_api.PIM_FP16, pim_api.MEM_TYPE_DEVICE, input.data_ptr())
        dev_output = pim_api.PimCreateBo(
            length, 1, 1, 1, pim_api.PIM_FP16, pim_api.MEM_TYPE_DEVICE, out_tensor.data_ptr())

        pim_input = pim_api.PimCreateBo(
            length, 1, 1, 1, pim_api.PIM_FP16, pim_api.MEM_TYPE_PIM, 0)
        pim_output = pim_api.PimCreateBo(
            length, 1, 1, 1, pim_api.PIM_FP16, pim_api.MEM_TYPE_PIM, 0)

        pim_api.PimCopyMemory(pim_input, dev_input, pim_api.DEVICE_TO_PIM)

        pim_api.PimExecuteRelu(pim_output, pim_input, None, 1)
        pim_api.PimCopyMemory(dev_output, pim_output, pim_api.PIM_TO_DEVICE)

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
