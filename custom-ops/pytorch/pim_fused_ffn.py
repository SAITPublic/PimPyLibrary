import torch
import torch.nn as nn
from torch.autograd import Function
import pim_api


class PimFusedFFNFunction(Function):
    @staticmethod
    def forward(ctx, inputs, fc1_w, fc1_bias, fc2_w, fc2_bias, block):

        #--first ffn-------------
        num_batch = 1
        num_channels = inputs.size()[0]
        inout_h  = inputs.size()[1]
        in_w = fc1_w.size()[2]
        out_w = fc1_w.size()[1]
        out_tensor = torch.empty(
                (num_channels, inout_h,  out_w), dtype=torch.float16, device=inputs.device)


        pim_gemm_desc = pim_api.PimCreateGemmDesc(num_batch, num_channels, inout_h, in_w, out_w, pim_api.PIM_FP16)
        device_input = pim_api.PimCreateBo(pim_gemm_desc, pim_api.MEM_TYPE_DEVICE, pim_api.GEMM_INPUT, inputs.data_ptr())
        device_weight = pim_api.PimCreateBo(pim_gemm_desc, pim_api.MEM_TYPE_DEVICE, pim_api.GEMM_WEIGHT, fc1_w.data_ptr())
        device_bias = pim_api.PimCreateBo(pim_gemm_desc, pim_api.MEM_TYPE_DEVICE, pim_api.GEMM_BIAS, fc1_bias.data_ptr())
        device_output = pim_api.PimCreateBo(pim_gemm_desc, pim_api.MEM_TYPE_DEVICE, pim_api.GEMM_OUTPUT, out_tensor.data_ptr())
        pim_api.PimExecuteGemm(device_output, device_input, device_weight, device_bias, pim_api.ACT_RELU, None, 0)
        if block==False:
            pim_api.PimSynchronize(None)

        pim_api.PimDestroyBo(device_weight)
        pim_api.PimDestroyBo(device_input)
        pim_api.PimDestroyBo(device_bias)
        pim_api.PimDestroyBo(device_output)
        pim_api.PimDestroyGemmDesc(pim_gemm_desc)

        #--second ffn-------------
        num_channels = out_tensor.size()[0]
        in_w = fc2_w.size()[2]
        out_w = fc2_w.size()[1]
        o2 = torch.empty(
                (num_channels, inout_h,  out_w), dtype=torch.float16, device=inputs.device)


        pim_gemm_desc = pim_api.PimCreateGemmDesc(num_batch, num_channels, inout_h, in_w, out_w, pim_api.PIM_FP16);
        device_input = pim_api.PimCreateBo(pim_gemm_desc, pim_api.MEM_TYPE_DEVICE, pim_api.GEMM_INPUT, out_tensor.data_ptr());
        device_weight = pim_api.PimCreateBo(pim_gemm_desc, pim_api.MEM_TYPE_DEVICE, pim_api.GEMM_WEIGHT, fc2_w.data_ptr());
        device_bias = pim_api.PimCreateBo(pim_gemm_desc, pim_api.MEM_TYPE_DEVICE, pim_api.GEMM_BIAS, fc2_bias.data_ptr());
        device_output = pim_api.PimCreateBo(pim_gemm_desc, pim_api.MEM_TYPE_DEVICE, pim_api.GEMM_OUTPUT, o2.data_ptr());
        pim_api.PimExecuteGemm(device_output, device_input, device_weight, device_bias, pim_api.NONE, None, 0);
        if block==False:
            pim_api.PimSynchronize(None)

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

    def forward(self, x, batched_fc1_w, batched_fc1_bias, batched_fc2_w, batched_fc2_bias, block=True):
        return PimFusedFFNFunction.apply(x, batched_fc1_w, batched_fc1_bias, batched_fc2_w, batched_fc2_bias, block)
