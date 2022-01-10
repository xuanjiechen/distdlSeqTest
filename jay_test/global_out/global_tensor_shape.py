import numpy as np
import torch
from mpi4py import MPI

import distdl
from distdl.backends.mpi.partition import MPIPartition
from distdl.nn.conv_feature import DistributedFeatureConv2d
from distdl.utilities.slicing import compute_subshape
from distdl.utilities.torch import zero_volume_tensor

# test case to see the halo shape
def test_print_halo_shape():
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
    
    conv1 = distdl.nn.DistributedConv2d(P_net,
                                        in_channels=1,
                                        out_channels=6,
                                        kernel_size=(5, 5),
                                        padding=(2, 2))
    
    conv2 = distdl.nn.DistributedConv2d(P_net,
                                        in_channels=6,
                                        out_channels=16,
                                        kernel_size=(5, 5),
                                        padding=(0, 0))

    pool1 = distdl.nn.DistributedMaxPool2d(P_net,
                                           kernel_size=(2, 2),
                                           stride=(2, 2))

    # get current rank
    rank = MPI.COMM_WORLD.Get_rank()

    print("global tensor shape for input:")
    print(conv1._distdl_backend.assemble_global_tensor_structure(in_data, P_net).shape, "\n")
    MPI.COMM_WORLD.Barrier()

    # print the halo shape for each partition
    # conv1
    output = conv1(in_data)
    
    #if rank == 0:
    print("global tensor shape for output data after conv1:")
    print(conv1._distdl_backend.assemble_global_tensor_structure(output, P_net).shape, "\n")
    MPI.COMM_WORLD.Barrier()

    # conv2
    output = conv2(output)

    #if rank == 0:
    print("global tensor shape for output data after conv2:")
    print(conv2._distdl_backend.assemble_global_tensor_structure(output, P_net).shape)


if __name__ == '__main__':
    test_print_halo_shape()