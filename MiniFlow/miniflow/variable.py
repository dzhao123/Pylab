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

        if trainable:
            self.graph.trainable_variables.append(self)


    def compute_output(self):
        if self.output_value is None:
            self.output_value = self.initial_value
        return self.output_value


def variable(initial_value=None, name=None, trainable=True):
    return Variable(initial_value=initial_value, name=name, trainable=trainable)
