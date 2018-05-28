from queue import Queue
import numpy as np


# -------------
# Operation
# -------------
class Operation(object):
    def __init__(self, *input_nodes, name=None):
        self.input_nodes = input_nodes
        self.output_nodes = []
        self.output_value = None
        self.name = name
        self.graph = DEFAULT_GRAPH

        for node in input_nodes:
            node.output_nodes.append(self)
        self.graph.operations.append(self)

    def compute_output(self):
        raise NotImplementedError

    def compute_gradient(self):
        raise NotImplementedError

    def __add__(self, other):
        return Add(self, other)

    def __neg__(self, other):
        return Negative(self, other)

    def __sub__(self, other):
        return Add(self, Negative(other))

    def __mul__(self, other):
        return Multiply(self, other)

# -------------
# Add
# -------------
class Add(Operation):
    def __init__(self, x, y, name=None):
        super(self.__class__, self).__init__(x, y, name=name)

    def compute_output(self):
        x, y = self.input_nodes
        self.output_value = np.add(x.output_value, y.output_value)
        return self.output_value

    def compute_gradient(self, grad=None):
        x, y = [node.output_value for node in self.input_nodes]

        if grad is None:
            grad = np.ones_like(self.output_value)

        grad_wrt_x = grad
        while np.ndim(grad_wrt_x) > len(np.shape(x)):
            grad_wrt_x = np.sum(grad_wrt_x, axis=0)
        for axis, size in enumerate(np.shape(x)):
            if size == 1:
                grad_wrt_x = np.sum(grad_wrt_x, axis=axis, keepdims=True)

        grad_wrt_y = grad
        while np.ndim(grad_wrt_y) > len(np.shape(y)):
            grad_wrt_y = np.sum(grad_wrt_y, axis=0)
        for axis, size in enumerate(np.shape(y)):
            if size == 1:
                grad_wrt_y = np.sum(grad_wrt_y, axis=axis, keepdims=True)

        return [grad_wrt_x, grad_wrt_y]

def add(x, y, name=None):
    return Add(x, y, name)


# -------------
# Multiply
# -------------
class Multiply(Operation):
    def __init__(self, x, y, name=None):
        super(self.__class__, self).__init__(x, y, name=name)

    def compute_output(self):
        x, y = self.input_nodes
        self.output_value = np.multiply(x.output_value, y.output_value)
        return self.output_value

    def compute_gradient(self, grad=None):
        x, y = [node.output_value for node in self.input_nodes]

        if grad is None:
            grad = np.ones_like(self.output_value)

        grad_wrt_x = grad*y
        while np.ndim(grad_wrt_x) > len(np.shape(x)):
            grad_wrt_x = np.sum(grad_wrt_x, axis=0)
        for axis, size in enumerate(np.shape(x)):
            if size == 1:
                grad_wrt_x = np.sum(grad_wrt_x, axis=axis, keepdims=True)

        grad_wrt_y = grad*x
        while np.ndim(grad_wrt_y) > len(np.shape(y)):
            grad_wrt_y = np.sum(grad_wrt_y, axis=0)
        for axis, size in enumerate(np.shape(y)):
            if size == 1:
                grad_wrt_y = np.sum(grad_wrt_y, axis=axis, keepdims=True)

        return [grad_wrt_x, grad_wrt_y]

def multiply(x, y, name=None):
    return Nultiply(x, y, name)


# -------------
# MatMul
# -------------
class MatMul(Operation):
    def __init__(self, x, y, name=None):
        super(self.__class__, self).__init__(x, y, name=name)

    def compute_output(self):
        x, y = self.input_nodes
        self.output_value = np.dot(x.output_value, y.output_value)
        return self.output_value

    def compute_gradient(self, grad=None):
        x, y = [node.output_value for node in self.input_nodes]

        if grad is None:
            grad = np.ones_like(self.output_value)

        dfdx = np.dot(grad, np.transpose(y))
        dfdy = np.dot(np.transpose(x), grad)

        return [dfdx, dfdy]

def matmul(x, y, name=None):
    return MatMul(x, y, name)


# -------------
# Activation
# -------------
class Sigmoid(Operation):
    def __init__(self, x, name=None):
        super(self.__class__, self).__init__(x, name=name)

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = 1/(1 + np.exp(-x.output_value))

    def compute_gradient(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.output_value)
        return grad*self.output_value*(1-self.output_value)

def sigmoid(x, name=None):
    return Sigmoid(x, name=name)


class Relu(Operation):
    def __init__(self, x, name=None):
        super(self.__class__, self).__init__(x, name=name)

    def compute_output(self):
        x, = self.input_nodes
        #print(x.output_value)

        self.output_value = np.maximum(x.output_value,0)

    def compute_gradient(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.output_value)

        return grad*np.maximum(np.sign(self.output_value),0)

def relu(x, name=None):
    return Relu(x, name=name)


class BatchAverage(Operation):
    def __init__(self, x, name=None):
        super(self.__class__, self).__init__(x, name=name)
        #self.output_value = np.zeros_like(x.output_value)
        self.total = np.zeros_like(x.output_value)
        self.counter = 0

    def compute_output(self):
        x, = self.input_nodes
        self.counter += 1
        self.total = np.add(self.total, x.output_value)
        self.output_value = self.total/self.counter
        #print(self.total)
        #print(self.output_value)

    def compute_gradient(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.input_nodes[0].output_value)
        return grad

def batch_average(x, name=None):
    return BatchAverage(x, name=name)



# -------------
# Logrithm
# -------------
class Log(Operation):
    def __init__(self, x, name=None):
        super(self.__class__, self).__init__(x, name=name)

    def compute_output(self):
        x, = self.input_ndoes
        self.outptu_value = np.log(x.output_value)
        return self.output_value

    def compute_gradient(self):
        x = self.input_nodes[0].output_value
        if grad is None:
            grad = np.ones_like(self.output_value)
        return grad*1/x

def log(x, name=None):
    return Log(x, name=name)


# -------------
# Negative
# -------------
class Negative(Operation):
    def __init__(self, x, name=None):
        super(self.__class__, self).__init__(x, name=name)

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = -np.array(x.output_value)
        return self.output_value

    def compute_gradient(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.output_value)
        return -np.array(grad)

def negative(x, name=None):
    return Negative(x=x, name=name)


# -------------
# ReduceSum
# -------------
class ReduceSum(Operation):
    def __init__(self, x, axis=None):
        super(self.__class__, self).__init__(x)
        self.axis= axis

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = np.sum(x.output_value, self.axis)
        return self.output_value

    def compute_gradient(self, grad=None):
        x, = self.input_nodes
        input_value = x.output_value

        if grad is None:
            grad = np.ones_like(self.output_value)

        output_shape = np.array(np.shape(input_value))
        output_shape[self.axis] = 1.0
        tile_scaling = np.shape(input_value) // output_shape
        grad = np.reshape(grad, output_shape)
        return np.tile(grad, tile_scaling)

def reduce_sum(x, axis=None):
    return ReduceSum(x, axis=axis)


# -------------
# Square
# -------------
class Square(Operation):
    def __init__(self, x, name=None):
        super(self.__class__, self).__init__(x, name=name)

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = np.square(x.output_value)

    def compute_gradient(self, grad=None):
        input_value = self.input_nodes[0].output_value

        if grad is None:
            grad = np.ones_like(self.output_value)

        return grad*np.multiply(2.0, input_value)

def square(x, name=None):
    return Square(x, name=name)


# -------------------
# Argmax
# -------------------
class Argmax(Operation):
    def __init__(self, input_nodes, axis=None, name=None):
        super(self.__class__,self).__init__(input_nodes, name=name)
        self.axis = axis

    def compute_output(self):
        self.output_value = np.argmax(self.input_nodes[0].output_value, axis=self.axis)
        return self.output_value

    def compute_gradient(self, grad=None):
        #if grad is None:
        grad = np.zeros_like(self.input_nodes[0].output_value)
        return grad

def argmax(input_nodes, axis, name=None):
    return Argmax(input_nodes, axis, name)


# ------------------------
# Equal
# ------------------------
class Equal(Operation):
    def __init__(self, pred, label, name=None):
        super(self.__class__,self).__init__(pred, label, name=name)

    def compute_output(self):
        pred, label = self.input_nodes
        if pred.output_value == label.output_value:
            self.output_value = 1
        else:
            self.output_value = 0
        return self.output_value

    def compute_gradient(self, grad=None):
        #if grad is None:
        grad = np.zeros_like(self.input_nodes.output_value)
        return grad

def equal(pred, label, name=None):
    return Equal(pred, label, name=name)
