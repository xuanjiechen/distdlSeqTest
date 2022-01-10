import numpy as np
import torch
import torch.nn.functional as F

from .module import Module
from distdl.utilities.torch import TensorStructure
from distdl.nn.halo_exchange import HaloExchange
from distdl.utilities.slicing import assemble_slices

# PyTorch Sequential:
# https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/container.py#L29

"""
version 0.1.0: Based on the assumption that we are having balanced partition
            and dealing with 2D data
"""

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
            if type(layer).__name__ not in self._legal_layers:
                raise Exception("The layer number", idx, " is an invaild layer.")

        # Check args to be sure all layers have the same partition
        # first layer's partition
        self.P = self.modules[0].P_x
 
        for idx, layer in enumerate(self.modules):
            # conv, pool
            if (type(layer).__name__ in self._legal_convolutions) or \
                (type(layer).__name__ in self._legal_pooling):
                # the partions are exactly the same
                if layer.P_x != self.P:
                    raise Exception("The layer number ", idx, " has the different partation.")

    
    ############## for Global Tensor Out Shape ##############
    """
    reminder: the following shape functions are for 2d data
              need to change fo other size
    """
    
    """
    For each layer
    convert the info into tuple if it is num
    """
    def _convert_to_list(self, layer):
        
        """
        local helper function convert number to list
        modify from https://gist.github.com/JamesDBartlett3/a4398d7cf7ba984f031433c769ddba5c#file-conv_output-py-L5
        """
        def num2list(num):
            return list(num) if isinstance(num, tuple) else [num, num]

        padding, dilation, kernel_size, stride = num2list(layer.padding), num2list(layer.dilation), \
                                        num2list(layer.kernel_size), num2list(layer.stride)

        return padding, dilation, kernel_size, stride

    """
    Calculating the shape for conv layer
    layer: convlutional layer to use
    inshape: shape of global input tensor (2d)
    """
    def conv_out_shape(self, layer, inshape):

        padding, dilation, kernel_size, stride = self._convert_to_list(layer)

        height = np.floor((inshape[2] + 2*padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) \
                            / stride[0] + 1)

        width = np.floor((inshape[3] + 2*padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) \
                            / stride[1] + 1)

        outshape = [inshape[0], layer.out_channels, height, width]

        return outshape
    
    """
    Calculating the shape for pooling layer
    layer: pooling layer to use
    inshape: shape of global input tensor (2d)
    """
    def pool_out_shape(self, layer, inshape):

        padding, dilation, kernel_size, stride = self._convert_to_list(layer)

        height = np.floor((inshape[2] - dilation[0] * (kernel_size[0] - 1) - 1) \
                            / stride[0] + 1)

        width = np.floor((inshape[3] - dilation[1] * (kernel_size[1] - 1) - 1) \
                            / stride[1] + 1)

        outshape = [inshape[0], inshape[1], height, width]

        return outshape

    """
    go through all the layers and compute the out shape size
    inshape: list format, [batch_size, channel_size, height_size, width_size]
    """
    def compute_out_shape(self, inshape):

        # inshape should be the list format of the torch.Size
        if type(inshape).__name__ != "list":
            raise Exception("wrong shape format to calculate overall shape")

        outshape = inshape
        #self.modules[0]._distdl_backend.assemble_global_tensor_structure(input[0], self.P)

        # go through all the layers
        for layer in self.modules:
            # if it is conv layer
            if type(layer).__name__ in self._legal_convolutions:
                outshape = self.conv_out_shape(layer, outshape)
            # if it is pool layer
            elif type(layer).__name__ in self._legal_pooling:
                outshape = self.pool_out_shape(layer, outshape)
            # if it is act layer
            # size not change
            else:
                outshape = outshape
            
        return outshape
    ############## Global Tensor Out Shape end ##############
    

    ############## for Partition Tensor Out Shape ##############
    """
    Calculating the out shape for conv or pooling layer
    layer: convlutional layer to use
    inshape: shape of current partition subtensor (2d)
    note: based on the assumption that the subtensor has already been padded
          and halo regions are added
    """
    def conv_pool_out_shape(self, layer, inshape):

        padding, dilation, kernel_size, stride = self._convert_to_list(layer)

        height = np.floor((inshape[2] - dilation[0] * (kernel_size[0] - 1) - 1) \
                            / stride[0] + 1)

        width = np.floor((inshape[3] - dilation[1] * (kernel_size[1] - 1) - 1) \
                            / stride[1] + 1)

        outshape = [inshape[0], layer.out_channels, height, width]

        return outshape
    ############## Partition Tensor Out Shape end ##############
    

    ############## for Compute Halo Shape ##############
    def compute_overall_halo(self):
        # layershape?
        outshape = 0
        for layer in self.modules:
            if type(layer).__name__ in self._legal_convolutions or \
                type(layer).__name__ in self._legal_pooling:
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
    ############## Compute Halo Shape end ##############







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



    def _compute_halo_shape(self,
                            shape,
                            index,
                            x_global_shape,
                            kernel_size,
                            stride,
                            padding,
                            dilation,
                            require_nonnegative=True,
                            subtensor_shapes=None):

        x_global_shape = np.asarray(x_global_shape)

        # If subtensor_shapes is not None, then we cannot assume the input is balanced.
        if subtensor_shapes is not None:
            x_local_shape = subtensor_shapes[tuple(index)]
            x_local_start_index = self._compute_local_start_index(index,
                                                                  subtensor_shapes,
                                                                  x_local_shape)
        else:
            x_local_shape = compute_subshape(shape, index, x_global_shape)
            x_local_start_index = compute_start_index(shape, index, x_global_shape)

        # formula from pytorch docs for maxpool
        y_global_shape = self._compute_out_shape(x_global_shape, kernel_size,
                                                 stride, padding, dilation)

        y_local_shape = compute_subshape(shape, index, y_global_shape)
        y_local_start_index = compute_start_index(shape, index, y_global_shape)

        y_local_left_global_index = y_local_start_index
        x_local_left_global_index_needed = self._compute_min_input_range(y_local_left_global_index,
                                                                         kernel_size,
                                                                         stride,
                                                                         padding,
                                                                         dilation)
        # Clamp to the boundary
        x_local_left_global_index_needed = np.maximum(np.zeros_like(x_global_shape),
                                                      x_local_left_global_index_needed)

        y_local_right_global_index = y_local_start_index + y_local_shape - 1
        x_local_right_global_index_needed = self._compute_max_input_range(y_local_right_global_index,
                                                                          kernel_size,
                                                                          stride,
                                                                          padding,
                                                                          dilation)
        # Clamp to the boundary
        x_local_right_global_index_needed = np.minimum(x_global_shape - 1,
                                                       x_local_right_global_index_needed)

        # Compute the actual ghost values
        x_local_left_halo_shape = x_local_start_index - x_local_left_global_index_needed
        x_local_stop_index = x_local_start_index + x_local_shape - 1
        x_local_right_halo_shape = x_local_right_global_index_needed - x_local_stop_index

        # Make sure the halos are always positive, so we get valid buffer shape
        if require_nonnegative:
            x_local_left_halo_shape = np.maximum(x_local_left_halo_shape, 0)
            x_local_right_halo_shape = np.maximum(x_local_right_halo_shape, 0)

        return np.hstack([x_local_left_halo_shape, x_local_right_halo_shape]).reshape(2, -1).T