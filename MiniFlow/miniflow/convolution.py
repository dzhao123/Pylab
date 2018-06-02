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
        self.N, self.H, self.W, self.C = 0, 0, 0, 0
        self.Hf, self.Wf, self.C, self.Cf_out = 0, 0, 0, 0
        self.n, self.h, self.v, self.c = 0, 0, 0, 0
        self.pad_H, self.pad_W = 0, 0
        self.n_H, self.n_W = 0, 0
        self.x = None
        self.y = None
        self.x_pad = None


    def compute_output(self):

        self.x, self.y = self.input_nodes
        self.N, self.H, self.W, self.C = self.x.output_value.shape
        self.Hf, self.Wf, self.C, self.Cf_out = self.y.output_value.shape
        self.n, self.h, self.v, self.c = self.strides

        if self.padding == 'same':
            self.pad_H = self.Hf//2
            self.pad_W = self.Wf//2
        else:
            self.pad_H = self.pad_W = 0


        self.n_H = (self.H - self.Hf + 2*self.pad_H)//self.h + 1
        self.n_W = (self.W - self.Wf + 2*self.pad_W)//self.v + 1
        self.x_pad = zero_pad(self.x.output_value, self.pad_H, self.pad_W)
        self.output_value = np.zeros((self.N, self.n_H, self.n_W, self.Cf_out))

        for batch in range(self.N):
            for vindex in range(self.n_H):
                for hindex in range(self.n_W):
                    for cindex in range(self.Cf_out):
                        vstart = vindex * self.h
                        vend = vstart + self.Hf
                        hstart = hindex * self.v
                        hend = hstart + self.Wf
                        self.output_value[batch, hindex, vindex, cindex] \
                            = slice_conv(self.x_pad[batch, vstart:vend, hstart:hend], self.y.output_value[...,cindex])


    def compute_gradient(self, grad=None):

        if grad is None:
            grad = np.ones((self.N, self.n_H, self.n_W, self.Cf_out))

        dfdy = np.zeros((self.Hf, self.Wf, self.C, self.Cf_out))
        dfdx = np.zeros_like(self.x_pad)#((N, H, W, C))
        for batch in range(self.N):
            for hindex in range(self.n_H):
                for vindex in range(self.n_W):
                    vstart = hindex * self.v
                    vend = vstart + self.Hf#hindex + Hf
                    hstart = vindex * self.h
                    hend = hstart + self.Wf
                    for slice in range(self.Cf_out):
                        dZ = grad[batch, hindex, vindex, slice]
                        dfdx[batch,hstart:hend,vstart:vend] += dZ * self.y.output_value[:,:,:,slice]
                        dfdy[:,:,:,slice] += dZ * self.x_pad[batch,hstart:hend,vstart:vend]


        if self.padding == 'same':
            dfdx = dfdx[:,self.pad_H:-self.pad_H,self.pad_W:-self.pad_W,:]

        #print('dfdx:', dfdx)
        #print('dfdy:', dfdy)

        return [dfdx, dfdy]



def conv2d(x, filter, strides, padding, name=None):
    return Convolution2d(x, filter, strides, padding, name=None)

def slice_conv(slice, filter):
    return np.sum(np.multiply(slice, filter))

def zero_pad(x, pad_H, pad_W):
    x_pad = np.pad(x, ((0,0),(pad_H,pad_H),(pad_W,pad_W),(0,0)), 'constant', constant_values=0)
    return x_pad


# -----------------
# Pooling
# -----------------
class MaxPool(Operation):
    def __init__(self, x, filter, strides, padding, name):
        super(self.__class__,self).__init__(x, name=name)
        self.strides = strides
        self.padding = padding
        self.N, self.H, self.W, self.C = 0, 0, 0, 0
        self.Hf, self.Wf, self.C, self.Cf_out = 0, 0, 0, 0
        self.n, self.h, self.v, self.c = 0, 0, 0, 0
        self.pad_H, self.pad_W = 0, 0
        self.n_H, self.n_W = 0, 0
        self.x = None
        self.filter = filter
        self.x_pad = None


    def get_shape(self):
        pass

    def compute_output(self):

        self.x, = self.input_nodes
        self.N, self.H, self.W, self.C = self.x.output_value.shape
        _, self.Hf, self.Wf, _ = self.filter
        self.n, self.h, self.v, self.c = self.strides

        if self.padding == 'same':
            self.pad_H = self.Hf//2
            self.pad_W = self.Wf//2
        else:
            self.pad_H = self.pad_W = 0


        self.n_H = (self.H - self.Hf + 2*self.pad_H)//self.h + 1
        self.n_W = (self.W - self.Wf + 2*self.pad_W)//self.v + 1
        self.x_pad = zero_pad(self.x.output_value, self.pad_H, self.pad_W)
        self.output_value = np.zeros((self.N, self.n_H, self.n_W, self.C))

        for batch in range(self.N):
            for vindex in range(self.n_H):
                for hindex in range(self.n_W):
                    for cindex in range(self.C):
                        vstart = vindex * self.h
                        vend = vstart + self.Hf
                        hstart = hindex * self.v
                        hend = hstart + self.Wf
                        #print('output_value:', self.output_value[batch, hindex, vindex, cindex])
                        #print('max_value:', max(self.x_pad[batch, vstart:vend, hstart:hend, cindex]))
                        #print('value:', self.x_pad[batch, vstart:vend, hstart:hend, cindex])
                        self.output_value[batch, hindex, vindex, cindex] \
                            = np.max(self.x_pad[batch, vstart:vend, hstart:hend, cindex])#, self.y.output_value[...,cindex])


    def compute_gradient(self, grad=None):
        if grad is None:
            grad = np.ones((self.N, self.n_H, self.n_W, self.C))

        #dfdy = np.zeros((self.Hf, self.Wf, self.C, self.Cf_out))
        dfdx = np.zeros_like(self.x_pad)#((N, H, W, C))
        dfdf = np.ones(self.filter)

        for batch in range(self.N):
            for hindex in range(self.n_H):
                for vindex in range(self.n_W):
                    vstart = hindex * self.v
                    vend = vstart + self.Hf#hindex + Hf
                    hstart = vindex * self.h
                    hend = hstart + self.Wf
                    for slice in range(self.C):
                        dZ = grad[batch, hindex, vindex]
                        dfdx[batch,hstart:hend,vstart:vend] += dZ * dfdf[batch]#self.y.output_value[:,:,:,slice]
                        #dfdy[:,:,:,slice] += dZ * self.x_pad[batch,hstart:hend,vstart:vend]


        if self.padding == 'same':
            dfdx = dfdx[:,self.pad_H:-self.pad_H,self.pad_W:-self.pad_W,:]

        return dfdx

def maxpooling(x, filter, strides, padding, name=None):
    return MaxPool(x, filter, strides, padding, name)


# ----------------
# Merge
# ----------------
class Merge(Operation):
    def __init__(self, x):
        super(self.__class__,self).__init__(x)
        #self.output_value = []
        self.shape = None

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = x.output_value.reshape((1,-1))
        self.shape = self.output_value.shape

    def compute_gradient(self, grad=None):
        x, = self.input_nodes
        if grad is None:
            grad = np.ones_like(x.output_value)
        return grad.reshape(x.output_value.shape)

def merge(x):
    return Merge(x)
