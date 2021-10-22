import numpy as np
import pim_api

length = 128 * 1024

pim_api.PimInitialize(pim_api.RT_TYPE_HIP, pim_api.PIM_FP16)
host_input1 = pim_api.PimCreateBo(length, 1, 1, 1, pim_api.PIM_FP16, pim_api.MEM_TYPE_HOST, None)
host_input2 = pim_api.PimCreateBo(length, 1, 1, 1, pim_api.PIM_FP16, pim_api.MEM_TYPE_HOST, None)
host_output = pim_api.PimCreateBo(length, 1, 1, 1, pim_api.PIM_FP16, pim_api.MEM_TYPE_HOST, None)

pim_input1 = pim_api.PimCreateBo(length, 1, 1, 1, pim_api.PIM_FP16, pim_api.MEM_TYPE_PIM,None)
pim_input2 = pim_api.PimCreateBo(length, 1, 1, 1, pim_api.PIM_FP16, pim_api.MEM_TYPE_PIM,None)
pim_output = pim_api.PimCreateBo(length, 1, 1, 1, pim_api.PIM_FP16, pim_api.MEM_TYPE_PIM,None)

input1 = np.random.randn(length).astype(np.float16)
input2 = np.random.randn(length).astype(np.float16)
golden = input1 + input2
pim_out = np.zeros_like(golden, dtype=np.float16)

np.copyto(np.frombuffer(host_input1, dtype=np.float16), input1)
np.copyto(np.frombuffer(host_input2, dtype=np.float16), input2)

pim_api.PimCopyMemory(pim_input1, host_input1, pim_api.HOST_TO_PIM)
pim_api.PimCopyMemory(pim_input2, host_input2, pim_api.HOST_TO_PIM)

pim_api.PimExecuteAdd(pim_output, pim_input1, pim_input2, None, 1)
pim_api.PimCopyMemory(host_output, pim_output, pim_api.HOST_TO_PIM)

np.copyto(pim_out, np.frombuffer(host_output, dtype=np.float16))

comparison = pim_out == golden
equal_arrays = comparison.all()

print("Pim output matches with golden : ", equal_arrays)
