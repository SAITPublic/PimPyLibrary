import unittest
import torch
import torch.nn as nn
from torch.autograd import Function
import pim_api
from pim_eltwise import PimEltwiseFunction as pim_elt

class PyEltTest(unittest.TestCase):
  def test_vec(self):
    pim_api.PimInitialize(pim_api.RT_TYPE_HIP, pim_api.PIM_FP16)
    with torch.no_grad():
        gpu0 =  torch.device(0)
        input0 = torch.rand(128 * 1024, dtype=torch.float16,device=gpu0)
        input1 = torch.rand(128 * 1024, dtype=torch.float16,device=gpu0)
        add = torch.tensor([0], dtype=torch.int32,device=gpu0)

        pim_result = pim_elt.apply(input0, input1, add)
        true_result = input0 + input1
        #print(true_result)
        #print(result)
        self.assertTrue(torch.allclose(pim_result, true_result, atol=0.01))

if __name__ == "__main__":
    unittest.main()
