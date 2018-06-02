from .graph import *
import numpy as np

# -------------
# Constant
# -------------
class Constant(object):
    def __init__(self, value, name=None):
        self.value =value
        self.output_value = None
        self.output_nodes = []
        self.name = name
        DEFAULT_GRAPH.constants.append(self)

    def compute_output(self):
        if self.output_value is None:
            self.output_value = self.value
        return self.output_value

    def compute_gradient(self):
        return np.zeros_like(self.output_value)

    def __add__(self, other):
        return Add(self, other)

    def __neg__(self, other):
        return Negative(self, other)

    def __sub__(self, other):
        return Add(self, Negative(other))

    def __mul__(self, other):
        return Multiply(self, other)

def constant(value, name=None):
    return Constant(value, name=name)


#-------------
# RandomNormal
#-------------
class RandomNormal(Constant):
    def __init__(self, *value, name):
        super(self.__class__,self).__init__(value, name=name)
        #self.output_value = np.random.normal(self.value[1], self.value[2], self.value[0])
        #if self.output_nodes is None:
        #    self.output_nodes.input_

    def compute_output(self):
        #if self.output_value is None:
        self.output_value = np.random.normal(self.value[1], self.value[2], self.value[0])
        return self.output_value

    #def compute_gradient(self):
    #    return np.ones_like(self.output_value)

def random_normal(input_shape, mu, sigma, name=None):
    return RandomNormal(input_shape, mu, sigma, name=name).compute_output()


#---------------
# Normalization
#---------------
class OneHotEncoding(object):
    def __init__(self, name):
        self.output_value = None
        self.name = name

    def normalization(self, y, n_classes):
        normalized = [[0 for _ in range(n_classes)] for _ in range(len(y))]
        for index in range(len(y)):
            normalized[index][y[index]] = 1
        self.output_value = normalized
        return self.output_value

def onehot_encoding(y, n_classes, name=None):
    return OneHotEncoding(name).normalization(y, n_classes)


#---------------
# TruncatedNormal
#---------------
class TruncatedNormal(Constant):
    def __init__(self, *value, name):
       super(self.__class__,self).__init__(value, name=name)

    def compute_output(self):
        if self.output_value is None:
            self.output_value = \
                (np.random.randn(self.value[0][0],self.value[0][1],self.value[0][2],self.value[0][3])+self.value[1])*self.value[2]
        return self.output_value

    #def compute_gradient(self):
    #    return np.ones_like(self.output_value)


def truncated_normal(input_shape, mu=0.0, sigma=0.1, name=None):
    return TruncatedNormal(input_shape, mu, sigma, name=name).compute_output()
