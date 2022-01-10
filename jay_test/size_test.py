import numpy as np
import torch
from mpi4py import MPI

import distdl
from distdl.backends.mpi.partition import MPIPartition
from distdl.nn.conv_feature import DistributedFeatureConv2d
from distdl.utilities.slicing import compute_subshape
from distdl.utilities.torch import zero_volume_tensor

# test case to see the output
def test_input():
    P_world = distdl.backend.backend.Partition(MPI.COMM_WORLD)
    P_world._comm.Barrier()

    # base partition parameter indicates how many processors we want
    P_base = P_world.create_partition_inclusive(np.arange(4))

    # check to see if the base partition is active or not
    if not P_base.active:
        return
    
    # layer partition parameter:
    # the last two values' product equals to number of processors we request
    # the second value is the number of channels
    P_net = P_base.create_cartesian_topology_partition([1,1,2,2])

    torch.manual_seed(1)
    # input data size
    # the second value equals to the number of channels we request from P_net
    in_data = torch.rand(2, 1, 5, 3)
    print(in_data)


if __name__ == '__main__':
    test_input()