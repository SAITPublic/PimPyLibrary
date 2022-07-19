import torch
import torch.nn as nn
from torch.autograd import Function
import pim_api


class PimDenseFunction(Function):
    @staticmethod
    def forward(ctx, inputs, weights, bias):

        num_iters = 1
        num_batch = inputs.size()[0]
        num_rows = weights.size()[1]
        num_cols = weights.size()[0]
        print(num_rows, num_cols)

        if inputs.ndim > 3:
            print("More than 3 dimensional input not supported in PimDense")
            return

        if inputs.ndim == 3:
            num_iters = inputs.size()[1]
            out_tensor = torch.zeros(
                (num_batch, num_iters, num_cols), dtype=torch.float16, device=inputs.device)
        else:
            out_tensor = torch.zeros(
                (num_batch, num_cols), dtype=torch.float16, device=inputs.device)

        in_len = torch.numel(inputs)
        out_len = torch.numel(out_tensor)
        pim_desc = pim_api.PimCreateDesc(
            num_batch, 1, num_rows, num_cols, pim_api.PIM_FP16, pim_api.OP_GEMV)
        dev_weight = pim_api.PimCreateBo(
            pim_desc, pim_api.MEM_TYPE_DEVICE, pim_api.GEMV_WEIGHT, weights.data_ptr())

        offset_row = 0
        offset_col = 0

        for i in range(num_iters):
            dev_in = pim_api.PimCreateBo(
                pim_desc, pim_api.MEM_TYPE_DEVICE, pim_api.GEMV_INPUT, inputs.data_ptr() + offset_col)
            dev_out = pim_api.PimCreateBo(
                pim_desc, pim_api.MEM_TYPE_DEVICE, pim_api.GEMV_OUTPUT, out_tensor.data_ptr() + offset_row)
            pim_api.PimExecuteGemv(dev_out, dev_in, dev_weight, None, 0)
            offset_row += (num_rows * num_batch)*2  # sizeof(half)
            offset_col += (num_cols * num_batch)*2
            pim_api.PimDestroyBo(dev_in)
            pim_api.PimDestroyBo(dev_out)

        if bias is not None:
            # print(bias.shape)
            # print(out_tensor.shape)
            out_tensor = torch.add(bias, out_tensor)

        pim_api.PimDestroyBo(dev_weight)
        pim_api.PimDestroyDesc(pim_desc)
        return out_tensor

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError


class PimDense(nn.Linear):
    """A nn.module wrapper for py_pim_dense function.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                  device=None, dtype=None) -> None:
        super(PimDense, self).__init__(in_features, out_features, bias)

    def reset_parameters(self) -> None:
        super(PimDense, self).reset_parameters()

    def __repr__(self):
        return "PIM dense layer"

    def forward(self, inputs):
        return PimDenseFunction.apply(inputs, self.weight, self.bias)
