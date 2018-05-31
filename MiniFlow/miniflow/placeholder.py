from .graph import *
from .operator import *
# ------------------------------------------------------------------------------
# Placeholder node
# ------------------------------------------------------------------------------

class Placeholder(object):
    def __init__(self, input_shape, name=None):
        self.output_value = None
        self.output_nodes = []
        self.name = name
        self.graph = DEFAULT_GRAPH
        self.graph.placeholders.append(self)
        self.output_shape = input_shape

    def __add__(self, other):
        return Add(self, other)

    def __neg__(self):
        return Negative(self)

    def __sub__(self, other):
        return Add(self, Negative(other))

    def __mul__(self, other):
        return Multiply(self, other)

def placeholder(input_shape=None, name=None):
    return Placeholder(input_shape=input_shape, name=name)
