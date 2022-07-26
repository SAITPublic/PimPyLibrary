import torch
import torch.nn as nn
from torch.autograd import Function
import pim_api


class PimGemmFunction(Function):
    @staticmethod
    def forward(ctx, inputs, weights, bias, act, block):

        if inputs.ndim > 3:
            print("More than 3 dimensional input not supported in Gemm")
            return

        num_batch = 1
        num_channels = inputs.size()[0]
        inout_h  = inputs.size()[1]
        in_w = weights.size()[2]
        out_w = weights.size()[1]

        out_tensor = torch.empty(
                (num_channels, inout_h, out_w), dtype=torch.float16, device=inputs.device)


        pim_gemm_desc = pim_api.PimCreateGemmDesc(num_batch, num_channels, inout_h, in_w, out_w, pim_api.PIM_FP16)
        device_input = pim_api.PimCreateBo(pim_gemm_desc, pim_api.MEM_TYPE_DEVICE, pim_api.GEMM_INPUT, inputs.data_ptr())
        device_weight = pim_api.PimCreateBo(pim_gemm_desc, pim_api.MEM_TYPE_DEVICE, pim_api.GEMM_WEIGHT, weights.data_ptr())
        device_bias = pim_api.PimCreateBo(pim_gemm_desc, pim_api.MEM_TYPE_DEVICE, pim_api.GEMM_BIAS, bias.data_ptr())
        device_output = pim_api.PimCreateBo(pim_gemm_desc, pim_api.MEM_TYPE_DEVICE, pim_api.GEMM_OUTPUT, out_tensor.data_ptr())
        pim_api.PimExecuteGemm(device_output, device_input, device_weight, device_bias, act, None, block)
        if (block == False):
            pim_api.PimSynchronize(None)

        pim_api.PimDestroyBo(device_input)
        pim_api.PimDestroyBo(device_weight)
        pim_api.PimDestroyBo(device_bias)
        pim_api.PimDestroyBo(device_output)
        pim_api.PimDestroyGemmDesc(pim_gemm_desc)
        return out_tensor

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError


class PimGemm(nn.Linear):
    """A nn.module wrapper for py_pim_dense function.
    """

    def __init__(self,device=None, dtype=None) -> None:
        super(PimGemm, self).__init__()

    def reset_parameters(self) -> None:
        super(PimGemm, self).reset_parameters()

    def __repr__(self):
        return "PIM dense layer"

    def forward(self, inputs, weight, bias, act):
        return PimPimGemmFunction.apply(inputs, weight, bias, act, block=True)
