import numpy as np
import torch
import torch.nn.functional as F

from .module import Module
from distdl.utilities.torch import TensorStructure
from distdl.nn.halo_exchange import HaloExchange
from distdl.utilities.slicing import assemble_slices

# PyTorch Sequential:
# https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/container.py#L29

class DistributedSequential(Module):

    _legal_convolutions = ["DistributedFeatureConv1d", "DistributedFeatureConv2d", "DistributedFeatureConv3d"]
    _legal_pooling = ["DistributedMaxPool1d", "DistributedMaxPool2d", "DistributedMaxPool3d"]
    _legal_activation = ["ReLU"] # we are going to skip the activation layers
    _legal_layers = _legal_convolutions + _legal_pooling + _legal_activation

    def __init__(self, *args, buffer_manager=None):

        self.modules = args

        ### set up buffer_manager
        # Back-end specific buffer manager for economic buffer allocation
        if buffer_manager is None:
            buffer_manager = self._distdl_backend.BufferManager()
        elif type(buffer_manager) is not self._distdl_backend.BufferManager:
            raise ValueError("Buffer manager type does not match backend.")
        self.buffer_manager = buffer_manager # buffer_manager is used in halo exchange

        # check args to see if there is an invaild layer
        for idx, layer in enumerate(self.modules):
            if layer not in self._legal_layers:
                raise Exception("The layer number", idx, " is an invaild layer.")

        # Check args to be sure all layers have the same partition
        # first layer's partition
        self.P = self.modules[0].P_x
 
        for idx, layer in enumerate(self.modules):
            # conv, pool
            if (type(layer) in self._legal_convolutions) or \
                (type(layer) in self._legal_pooling):
                # the partions are exactly the same
                if layer.P_x != self.P:
                    raise Exception("The layer number ", idx, " has the different partation.")
 
    
    ##############for Global Tensor##############
    """
    local helper function convert number to tuple
    from https://gist.github.com/JamesDBartlett3/a4398d7cf7ba984f031433c769ddba5c#file-conv_output-py-L5
    """
    def num2tuple(num):
        return num if isinstance(num, tuple) else (num, num)

    """
    Calculating the shape for conv layer
    layer: convlutional layer to use
    input: shape of LOCAL input tensor (2d)
    """
    def conv_out_shape(self, layer, input):

        padding, dilation, kernel_size, stride = self.num2tuple(layer.padding), self.num2tuple(layer.dilation), \
                                                self.num2tuple(layer.kernel_size), self.num2tuple(layer.stride)

        height = np.floor((input[0] + padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) \
                            / stride[0] + 1)

        width = np.floor((input[1] + padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) \
                            / stride[1] + 1)

        return [height, width]

    """
    calculating the shape for pooling layer
    layer: pooling layer to use
    input: shape of LOCAL input tensor (2d)
    """
    def pool_out_shape(self, layer, input):

        padding, dilation, kernel_size, stride = self.num2tuple(layer.padding), self.num2tuple(layer.dilation), \
                                                self.num2tuple(layer.kernel_size), self.num2tuple(layer.stride)

        height = np.floor((input[0] + padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) \
                            / stride[0] + 1)

        width = np.floor((input[1] + padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) \
                            / stride[1] + 1)

        return [height, width]

    """
    calculating the shape for activation layer
    layer: activation layer to use
    input: shape of LOCAL input tensor (2d)
    """
    def act_out_shape(self, layer, input):
        return [input[0], input[1]]

    """
    go through all the layers and compute the out shape size
    """
    def compute_out_shape(self, inshape):
        outshape = inshape
        #self.modules[0]._distdl_backend.assemble_global_tensor_structure(input[0], self.P)

        # go through all the layers
        for layer in self.modules:
            # if it is conv layer
            if type(layer) in self._legal_convolutions:
                outshape = self.conv_out_shape(layer, outshape)
            # if it is pool layer
            elif type(layer) in self._legal_pooling:
                outshape = self.pool_out_shape(layer, outshape)
            # if it is act layer
            else:
                outshape = self.act_out_shape(layer, outshape)
            
        return outshape

    ##############Compute Halo Shape START##############
    def compute_overall_halo(self):
        # layershape?
        outshape = 0
        for layer in self.modules:
            if type(layer) in self._legal_convolutions or \
                type(layer) in self._legal_pooling:
                exchange_info = layer._compute_exchange_info(x_global_shape_after_pad,
                                                    self.kernel_size,
                                                    layer.stride,
                                                    self._expand_parameter(), # padding
                                                    self.dilation,
                                                    self.P_x.active,
                                                    self.P_x.shape,
                                                    self.P_x.index,
                                                    subtensor_shapes=subtensor_shapes)
                halo_shape = exchange_info[0]
            else:
                x = layer(x)
        return
    ###############Compute Halo Shape END###############

    def _expand_parameter(self, param):
        # If the given input is not of size num_dimensions, expand it so.
        # If not possible, raise an exception.
        param = np.atleast_1d(param)
        if len(param) == 1:
            param = np.ones(self.num_dimensions, dtype=int) * param[0]
        elif len(param) == self.num_dimensions:
            pass
        else:
            raise ValueError('Invalid parameter: ' + str(param))
        return tuple(param)


    ### this function is only called when the input changes
    def _distdl_module_setup(self, input):

        # called every time forward is called
        self._distdl_is_setup = True
        self._input_tensor_structure = TensorStructure(input[0])

        if not self.P_x.active:
            return
        ### check conv serial differently? ###

        # Figure out the actual padding needed and setup the halo
        # Compute global and local shapes with padding
        x_global_structure = \
            self._distdl_backend.assemble_global_tensor_structure(input[0], self.P_x)
        x_local_structure = TensorStructure(input[0])
        x_global_shape = x_global_structure.shape
        x_local_shape = x_local_structure.shape
        x_global_shape_after_pad = x_global_shape + 2*self.global_padding
        x_local_shape_after_pad = x_local_shape + np.sum(self.local_padding, axis=1, keepdims=False)
        x_local_structure_after_pad = TensorStructure(input[0])
        x_local_structure_after_pad.shape = x_local_shape_after_pad

        # We need to compute the halos with respect to the explicit padding.
        # So, we assume the padding is already added, then compute the halo regions.
        compute_subtensor_shapes_unbalanced = \
            self._distdl_backend.tensor_decomposition.compute_subtensor_shapes_unbalanced
        subtensor_shapes = \
            compute_subtensor_shapes_unbalanced(x_local_structure_after_pad, self.P_x)

        # This is where you will figure out the actual big halo for the entire sequence

        # Using that information, we can get there rest of the halo information
        exchange_info = self._compute_exchange_info(x_global_shape_after_pad,
                                                    self.kernel_size,
                                                    self.stride,
                                                    self._expand_parameter(0), # padding
                                                    self.dilation,
                                                    self.P_x.active,
                                                    self.P_x.shape,
                                                    self.P_x.index,
                                                    subtensor_shapes=subtensor_shapes)
        halo_shape = exchange_info[0]
        recv_buffer_shape = exchange_info[1]
        send_buffer_shape = exchange_info[2]
        needed_ranges = exchange_info[3]

        self.halo_shape = halo_shape
        
        # Here you will define the halo layer
        
        
    def forward(self, input):

        total_padding = self.local_padding + self.halo_shape
        torch_padding = self._to_torch_padding(total_padding)

        if total_padding.sum() == 0:
            input_padded = input
        else:
            input_padded = F.pad(input, pad=torch_padding, mode='constant', value=self.default_pad_value)

        input_exchanged = self.halo_layer(input_padded)
        input_needed = input_exchanged[self.needed_slices]

        x = input_needed
        for layer in self.modules:
            if type(layer) in self._legal_convolutions:
                x = layer.conv_layer(x)
            elif type(layer) in self._legal_pooling:
                x = layer.pool_layer(x)
            else:
                x = layer(x)

        return x


# Assume there is no padding