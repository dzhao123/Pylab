from .operator import *
from .graph import *
import numpy as np

# -------------
# Variable
# -------------
class Variable(object):
    def __init__(self, initial_value=None, name=None, trainable=True):

        self.initial_value = initial_value
        self.output_value = None
        self.output_nodes = []
        self.name = name
        self.graph = DEFAULT_GRAPH
        self.graph.variables.append(self)
        self.input_nodes = []

        if trainable:
            self.graph.trainable_variables.append(self)

        #self.input_nodes.append(initial_value)

        #self.initial_value.output_nodes.append(self)


    def compute_output(self):
        if self.output_value is None:
            self.output_value = self.initial_value#.output_value
        #print('initial_value:', self.output_value)
        #print('nod_name:', self.name)
        return self.output_value


def variable(initial_value=None, name=None, trainable=True):
    return Variable(initial_value=initial_value, name=name, trainable=trainable)


#-------------
# RandomNormal
#-------------
#class RandomNormal(Variable):
#    def __init__(self, input_shape, mu, sigma, name, trainable):
#        super(self.__class__,self).__init__(name=name, trainable=trainable)
#        self.input_shape = input_shape
#        self.mu = mu
#        self.sigma = sigma

#    def compute_output(self):
#        if self.output_value is None:
#            self.output_value = np.random.normal(self.mu, self.sigma, self.input_shape)
#        return self.output_value


#def random_normal(input_shape, mu, sigma, name=None, trainable=True):
#    return RandomNormal(input_shape, mu, sigma, name, trainable)


#---------------
# TruncatedNormal
#---------------
#class TruncatedNormal(Variable):
#    def __init__(self, name):
#       super(self.__class__,self).__init__(pred, label, name=name)

#    def compute_output(self, input_shape):
#        if self.output_value is None:
#            self.output_value = np.random.randn(input_shape[0],input_shape[1],input_shape[2],input_shape[3])/10
        #print(np.random.randn(input_shape[0],input_shape[1],input_shape[2],input_shape[3]))
#        return self.output_value

#def truncated_normal(*input_shape, name=None):
#    return TruncatedNormal(name).generate_output(input_shape)


#----------------
# global_variables_initializer()
#----------------
