import torch
import torch.nn as nn
from torch.autograd import Function
import pim_api
from py_pim_ops.pim_dense import PimDenseFunction as pim_dense
from py_pim_ops.pim_dense import PimDense


def testDense(self):
    pim_api.PimInitialize(pim_api.RT_TYPE_HIP, pim_api.PIM_FP16)
    n_batch = 2
    in_size = 1024
    out_size = 1024
    device = torch.device('cuda')

    input = torch.rand(size=(n_batch, in_size), dtype=torch.float16)
    input = input.to(device)

    dense = nn.Linear(in_size, out_size, bias=False)
    dense = dense.to(device)
    dense.half()

    # Pass the tensor to pytorch model and obtain output
    pytorch_result = None
    pytorch_result = dense(input)
    weights = dense.weight  # Weight copy to be used in PIM computation.
    weights_t = weights.clone().detach()
    weights_t = torch.transpose(weights_t, 0, 1).contiguous()
    #bias = dense.bias.clone().detach()
    bias = None
    pim_result = pim_dense.apply(input, weights, bias)  # Obtain PIM output

    print("Pytorch Result:", pytorch_result)
    print("PIM Result:", pim_result)
    #print("Weights:", weights)


def testDense2(self):
    pim_api.PimInitialize(pim_api.RT_TYPE_HIP, pim_api.PIM_FP16)
    in_batch = 2
    in_iters = 1
    in_size = 1024
    out_size = 1024

    with torch.no_grad():
      device = torch.device('cuda')
      input = torch.ones(size=(in_batch, in_size), dtype=torch.float16)
      input = input.to(device)

      pim_dense_layer = PimDense(in_size, out_size, bias=False)
      pim_dense_layer.to(device)
      pim_dense_layer.half()

      dense = nn.Linear(in_size, out_size, bias=False)
      dense = dense.to(device)
      dense.half()

      # Pass the tensor to pytorch model and obtain output
      pytorch_result = dense(input)
      weights = dense.weight  # Weight copy to be used in PIM computation.
      pim_dense_layer.weight.copy_(weights)
      pim_dense_layer.bias = None

      #bias = dense.bias.clone().detach()
      pim_result = pim_dense_layer(input)  # Obtain PIM output
      print("Pytorch Result:", pytorch_result)
      print("PIM Result:", pim_result)


torch.manual_seed(2)
testDense(1)
testDense2(1)
