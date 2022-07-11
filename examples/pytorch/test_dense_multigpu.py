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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net1 = nn.Linear(inp_size, inp_size, bias=False)
        self.pim_net1 = PimDense(inp_size, inp_size, bias=False)

        self.relu = nn.ReLU()
        self.net2 = nn.Linear(inp_size,32, bias=False)
        self.pim_net2 = PimDense(inp_size, 32, bias=False)

    def forward(self, x, use_pim=0):
        #print('Using Pim:', use_pim)
        if use_pim == 0:
            return self.net2(self.relu(self.net1(x)))
        else:
            return self.pim_net2(self.relu(self.pim_net1(x)))

    def set_pim_weights(self):
        weights = self.net1.weight  # Weight copy to be used in PIM computation.
        self.pim_net1.weight.copy_(weights)
        weights = self.net2.weight  # Weight copy to be used in PIM computation.
        self.pim_net2.weight.copy_(weights)
        #pim_dense_layer.bias = None


def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    with torch.no_grad():
        # create model and move it to GPU with id rank
        input = torch.randn(inp_size, inp_size).half().to(rank)
        model = Net()
        model = model.half()
        model.set_pim_weights() #Todo where to set weights ??
        model = model.to(rank)
        golden = model(input)

        print('setting pim device',rank)
        pim_api.PimInitialize(pim_api.RT_TYPE_HIP, pim_api.PIM_FP16)
        pim_api.PimSetDevice(rank)

        ddp_model = DDP(model, device_ids=[rank])

        outputs = ddp_model(input,0)
        print(torch.allclose(outputs,golden))

        cleanup()

def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)

if __name__ == '__main__':
   run_demo(demo_basic,4)

