from .operator import *
from .session import *
from .placeholder import *
import numpy as np
# -----------------
# Convolution
# -----------------
class Convolution2d(Operation):
    def __init__(self, x, filter, strides, padding, name):
        super(self.__class__,self).__init__(x, filter, name=name)
        self.strides = strides
        self.padding = padding


    def compute_output(self):

        x, y = self.input_nodes
        N, H, W, C = x.output_value.shape
        #print(self.filter.output_value)
        Hf, Wf, Cf_in, Cf_out = y.output_value.shape
        n, h, v, c = self.strides

        if self.padding == 'same':
            pad_H = Hf//2
            pad_W = Wf//2
        else:
            pad_H = pad_W = 0


        n_H = (H - Hf + 2*pad_H)//h + 1
        n_W = (W - Wf + 2*pad_W)//v + 1

        x_pad = zero_pad(x.output_value, n_H, n_W)
        y_pad = zero_pad(y.output_value, n_H, n_W)

        self.output_value = np.zeros((N, n_H, n_W, Cf_out))

        for batch in range(N):
            slice = x.output_value[batch]
            #slice = x_pad[batch]
            for vindex in range(n_H):
                for hindex in range(n_W):
                    for cindex in range(Cf_out):
                        vstart = vindex * h
                        vend = vstart + Hf
                        hstart = hindex * v
                        hend = hstart + Wf
                        self.output_value[batch, hindex, vindex, cindex] \
                            = slice_conv(slice[vstart:vend, hstart:hend], y.output_value[...,cindex])#y_pad[...,cindex])#y.output_value[...,cindex])
        #assert(self.output_value.shape == (N, n_H, n_W, Cf_out))
        #print('output_value:', self.output_value.shape)


    def compute_gradient(self, grad=None):
        x,y = self.input_nodes

        N, H, W, C = x.output_value.shape
        Hf, Wf, C, Cf_out = y.output_value.shape
        #print('filter_shape:', self.filter.shape)
        n, h, v, c = self.strides

        if self.padding == 'same':
            pad_H = Hf//2
            pad_W = Wf//2
        else:
            pad_H = pad_W = 0

        n_H = (H - Hf + 2*pad_H)//h + 1
        n_W = (W - Wf + 2*pad_W)//v + 1

        if grad is None:
            grad = np.ones_like((N, n_H, n_W, Cf_out))

        #print(grad)
        #print(grad.shape)
        dfdy = np.zeros((Hf, Wf, C, Cf_out))
        dfdx = np.zeros((N, H, W, C))
        for batch in range(N):
            for hindex in range(n_H):
                for vindex in range(n_W):
                    vstart = hindex
                    vend = hindex + Hf
                    hstart = vindex
                    hend = hstart + Wf
                    for slice in range(Cf_out):
                        dZ = grad[batch, hindex, vindex, slice]
                        dfdx[batch,hstart:hend,vstart:vend] += y.output_value[:,:,:,slice]*dZ
                        dfdy[:,:,:,slice] += x.output_value[batch,hstart:hend,vstart:vend]*dZ

            return [dfdx, dfdy]

def conv2d(x, filter, strides, padding, name=None):
    return Convolution2d(x, filter, strides, padding, name=None)


def slice_conv(slice, filter):
    #print(np.sum(np.multiply(slice, filter)))
    return np.sum(np.multiply(slice, filter))

def zero_pad(x, pad_H, pad_W):
    x_pad = np.pad(x, ((0,0),(pad_H,pad_H),(pad_W,pad_W),(0,0)), 'constant', constant_values=0)
    return x_pad


# -----------------
# Pooling
# -----------------
class MaxPool(Operation):
    def __init__(self):
        super(self.__class__,self).__init__()

    def compute_output(self):
        pass

    def comptute_gradient(self):
        pass

def maxpooling():
    return MaxPool()


# ----------------
# Merge
# ----------------
class Merge(Operation):
    def __init__(self, x):
        super(self.__class__,self).__init__(x)

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = x.output_value.reshape((1,-1))

    def compute_gradient(self, grad=None):
        x, = self.input_nodes
        if grad is None:
            grad = np.ones_like(x.output_value)
        return grad.reshape(x.output_value.shape)

def merge(x):
    return Merge(x)
