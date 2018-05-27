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
class RandomNormal(object):
    def __init__(self, name):
        self.output_value = None
        self.name = name

    def generate_output(self, input_shape, mu, sigma):
        if self.output_value is None:
            self.output_value = np.random.normal(mu, sigma, input_shape)
        return self.output_value

def random_normal(input_shape, mu, sigma, name=None):
    return RandomNormal(name).generate_output(input_shape, mu, sigma)


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
