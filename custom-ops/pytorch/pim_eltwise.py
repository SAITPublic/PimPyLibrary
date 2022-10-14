import torch
import torch.nn as nn
from torch.autograd import Function
import pim_api

# Todo , broadcasting logic


class PimEltwiseFunction(Function):
    @staticmethod
    def forward(ctx, input1, input2, operation):

        if input1.size() != input2.size():
            if operation == 0:
                return torch.add(input1, input2)
            if operation == 1:
                return torch.mul(input1, input2)

        length = torch.numel(input1)
        out_tensor = torch.empty(
            input1.size(), dtype=torch.float16, device=input1.device)

        dev_input1 = pim_api.PimCreateBo(
            1, 1, 1, length, pim_api.PIM_FP16, pim_api.MEM_TYPE_DEVICE, input1.data_ptr(), False)
        dev_input2 = pim_api.PimCreateBo(
            1, 1, 1, length, pim_api.PIM_FP16, pim_api.MEM_TYPE_DEVICE, input2.data_ptr(), False)
        dev_output = pim_api.PimCreateBo(
            1, 1, 1, length, pim_api.PIM_FP16, pim_api.MEM_TYPE_DEVICE, out_tensor.data_ptr(), False)

        pim_input1 = pim_api.PimCreateBo(
            1, 1, 1, length, pim_api.PIM_FP16, pim_api.MEM_TYPE_PIM, 0, False)
        pim_input2 = pim_api.PimCreateBo(
            1, 1, 1, length, pim_api.PIM_FP16, pim_api.MEM_TYPE_PIM, 0, False)
        pim_output = pim_api.PimCreateBo(
            1, 1, 1, length, pim_api.PIM_FP16, pim_api.MEM_TYPE_PIM, 0, False)

        pim_api.PimCopyMemory(pim_input1, dev_input1, pim_api.DEVICE_TO_PIM)
        pim_api.PimCopyMemory(pim_input2, dev_input2, pim_api.DEVICE_TO_PIM)

        if operation == 0:
            pim_api.PimExecuteAdd(pim_output, pim_input1, pim_input2, None, 1)
        else:
            pim_api.PimExecuteMul(pim_output, pim_input1, pim_input2, None, 1)

        pim_api.PimCopyMemory(dev_output, pim_output, pim_api.PIM_TO_DEVICE)
        pim_api.PimDestroyBo(dev_input1)
        pim_api.PimDestroyBo(dev_input2)
        pim_api.PimDestroyBo(dev_output)
        pim_api.PimDestroyBo(pim_input1)
        pim_api.PimDestroyBo(pim_input2)
        pim_api.PimDestroyBo(pim_output)
        return out_tensor

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError


class PimEltwise(nn.Module):
    """A nn.module wrapper for py_pim_eltwise function.
    """

    def __init__(self, operation=0):
        super(PimEltwise, self).__init__()
        self.operation = operation
        if operation:
            self.op_t = torch.tensor([1], dtype=torch.int32)  # mul
        else:
            self.op_t = torch.tensor([0], dtype=torch.int32)  # add

    def __repr__(self):
        if self.operation:
            return "Pim Eltwise Mul Layer"
        else:
            return "Pim Eltwise Add Layer"

    def forward(self, input1, input2):
        return PimEltwiseFunction.apply(input1, input2, self.op_t)
