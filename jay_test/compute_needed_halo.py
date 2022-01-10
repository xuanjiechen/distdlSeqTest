import numpy as np
import torch
from mpi4py import MPI

import distdl
from distdl.backends.mpi.partition import MPIPartition
from distdl.nn.conv_feature import DistributedFeatureConv2d
from distdl.utilities.slicing import compute_subshape
from distdl.utilities.torch import zero_volume_tensor
import distdl.nn.distributedSequential as distSeq
from distdl.nn.mixins.halo_mixin import HaloMixin

# test case to check the gloabl tensor out shape
def compute_needed_halo():
    P_world = distdl.backend.backend.Partition(MPI.COMM_WORLD)
    P_world._comm.Barrier()

    num_procs = 4

    # base partition parameter indicates how many processors we want
    P_base = P_world.create_partition_inclusive(np.arange(num_procs))

    # check to see if the base partition is active or not
    if not P_base.active:
        return
    
    # layer partition parameter:
    # the last two values' product equals to number of processors we request
    # the second value is the number of channels
    # based on the assumption that all layers are using the same partition
    # so P_net is applicable to all layers
    P_net = P_base.create_cartesian_topology_partition([1,1,2,2])

    torch.manual_seed(1)
    in_data = torch.rand(5, 1, 5, 3)
    # this means the global input tensor size if [10, 6]
    
    conv1 = distdl.nn.DistributedConv2d(P_net,
                                        in_channels=1,
                                        out_channels=6,
                                        stride = 2,
                                        kernel_size=(5, 5),
                                        padding=(1, 1))
    
    conv2 = distdl.nn.DistributedConv2d(P_net,
                                        in_channels=6,
                                        out_channels=16,
                                        kernel_size=(4, 4),
                                        padding=(1, 1))

    rank = MPI.COMM_WORLD.Get_rank()

    for i in range(4):
        if rank == i:
            # after go through the first layer, print the halo shape for the input data
            print("halo shape for conv1 at rank %d:" %rank)
            a = conv1._compute_exchange_info([5, 1, 10, 6],
                                conv1.kernel_size,
                                conv1.stride,
                                conv1.padding,
                                conv1.dilation,
                                P_net.active,
                                P_net.shape,
                                P_net.index)
            print(a[0], "\n")
            print("partition shape:\n", compute_subshape(P_net.shape, rank, [5, 1, 10, 6]))            
        MPI.COMM_WORLD.Barrier()

    for i in range(4):
        if rank == i:
            # after go through the first layer, print the halo shape for the input data
            print("halo shape for conv2 at rank %d:" %rank)
            a = conv2._compute_exchange_info([5, 6, 4, 2],
                                conv1.kernel_size,
                                conv1.stride,
                                conv1.padding,
                                conv1.dilation,
                                P_net.active,
                                P_net.shape,
                                P_net.index)
            print(a[0], "\n")     
            print("partition shape:\n", compute_subshape(P_net.shape, rank, [5, 6, 4, 2]))         
        MPI.COMM_WORLD.Barrier()

    print("final partition shape:\n", compute_subshape(P_net.shape, rank, [5, 16, 3, 1]))

if __name__ == '__main__':
    compute_needed_halo()