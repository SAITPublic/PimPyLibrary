import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
import pim_api
from pim_pytorch.pim_dense import PimDenseFunction as pim_dense
from pim_pytorch.pim_dense import PimDense


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


inp_size = 1024
out_size = 32

class Net(nn.Module):

    def getTranspose(self, weights):
        weights_t = weights.clone().detach()
        weights_t = torch.transpose(weights_t, 0, 1).contiguous()
        return weights_t
        
    def __init__(self):
        super(Net, self).__init__()
        self.net1 = nn.Linear(inp_size, out_size, bias=False)
        self.pim_net1 = PimDense(inp_size, out_size, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x, use_pim=0):
        #print('Using Pim:', use_pim)
        if use_pim == 0:
            return self.net1(x)
        else:
            return self.pim_net1(x)

    def set_pim_weights(self):
        weights = self.net1.weight  # Weight copy to be used in PIM computation.
        self.pim_net1.weight.copy_(self.getTranspose(weights))
        self.pim_net1.bias = None


def demo_basic(rank, world_size, model):
    #print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    with torch.no_grad():
        # create model and move it to GPU with id rank
        pim_api.PimInitialize(pim_api.RT_TYPE_HIP, pim_api.PIM_FP16)
        input = torch.rand(1, inp_size).half().to(rank)
        model = model.to(rank)
        golden = model(input)

        #print('setting pim device',rank)
        pim_api.PimSetDevice(rank)

        ddp_model = DDP(model, device_ids=[rank])

        outputs = ddp_model(input,1)
        #outputs = model(input,1)
        #Todo check outputs in re-ordered format
        print(torch.allclose(outputs,golden,atol=2.0))

        cleanup()

def run_demo(demo_fn, world_size):
    input = torch.ones(1, inp_size).half().to(0)
    model = Net()
    model = model.half().to(0)
    golden = model(input)

    with torch.no_grad():
        model.net1.weight.fill_(0.2)
        model.set_pim_weights() #Todo where to set weights ??

    mp.spawn(demo_fn,
             args=(world_size,model,),
             nprocs=world_size,
             join=True)

if __name__ == '__main__':
   run_demo(demo_basic,2)

