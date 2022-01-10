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
def test_global_out():
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
    in_data = torch.rand(5, 1, 100, 60)
    # this means the global input tensor size if [10, 6]
    
    conv1 = distdl.nn.DistributedConv2d(P_net,
                                        in_channels=1,
                                        out_channels=6,
                                        stride = 2,
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
    
    conv3 = distdl.nn.DistributedConv2d(P_net,
                                        in_channels=16,
                                        out_channels=10,
                                        kernel_size=(3, 3),
                                        padding=(0, 0))

    pool2 = distdl.nn.DistributedMaxPool2d(P_net,
                                           kernel_size=(1, 1),
                                           stride=(2, 2))

    m = distSeq.DistributedSequential(conv1, conv2, pool1, conv3, pool2)
    
    #print(type(distdl.nn.DistributedConv2d))
    print(m.compute_out_shape([5, 1, 10, 6]))
    print("success!")

if __name__ == '__main__':
    test_global_out()