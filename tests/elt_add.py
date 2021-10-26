import numpy as np
import pim_api

length = 128 * 1024
input1 = np.random.randn(length).astype(np.float16)
input2 = np.random.randn(length).astype(np.float16)
golden = input1 + input2
pim_out = np.zeros_like(golden, dtype=np.float16)

pim_api.PimInitialize(pim_api.RT_TYPE_HIP, pim_api.PIM_FP16)
host_input1 = pim_api.PimCreateBo(length, 1, 1, 1, pim_api.PIM_FP16, pim_api.MEM_TYPE_HOST, input1.__array_interface__['data'][0])
host_input2 = pim_api.PimCreateBo(length, 1, 1, 1, pim_api.PIM_FP16, pim_api.MEM_TYPE_HOST, input2.__array_interface__['data'][0])
host_output = pim_api.PimCreateBo(length, 1, 1, 1, pim_api.PIM_FP16, pim_api.MEM_TYPE_HOST, pim_out.__array_interface__['data'][0])

pim_input1 = pim_api.PimCreateBo(length, 1, 1, 1, pim_api.PIM_FP16, pim_api.MEM_TYPE_PIM,0)
pim_input2 = pim_api.PimCreateBo(length, 1, 1, 1, pim_api.PIM_FP16, pim_api.MEM_TYPE_PIM,0)
pim_output = pim_api.PimCreateBo(length, 1, 1, 1, pim_api.PIM_FP16, pim_api.MEM_TYPE_PIM,0)

pim_api.PimCopyMemory(pim_input1, host_input1, pim_api.HOST_TO_PIM)
pim_api.PimCopyMemory(pim_input2, host_input2, pim_api.HOST_TO_PIM)

pim_api.PimExecuteAdd(pim_output, pim_input1, pim_input2, None, 1)
pim_api.PimCopyMemory(host_output, pim_output, pim_api.PIM_TO_HOST)

comparison = pim_out == golden
equal_arrays = comparison.all()

print("Pim output matches with golden : ", equal_arrays)
