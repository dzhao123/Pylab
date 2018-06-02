from .graph import *
from .operator import Operation
from .placeholder import *
from .variable import *
# -------------
# compute_gradints
# -------------
def compute_gradients(target_op, type=''):
    grad_table = {}
    grad_table[target_op] = np.ones_like(target_op.output_value)

    queue = Queue()
    queue.put(target_op)

    visited = set()
    visited.add(target_op)


    while not queue.empty():
        node = queue.get()

        if node != target_op:
            grads_wrt_node_output = []

            for output_node in node.output_nodes:
                if output_node not in grad_table:
                    grad_table[output_node] = output_node.compute_gradient()
                grad_wrt_output_node_output = grad_table[output_node]
                grad_wrt_node_output = output_node.compute_gradient(grad_wrt_output_node_output)
                if len(output_node.input_nodes) > 1:
                    input_node_index = output_node.input_nodes.index(node)
                    grads_wrt_node_output.append(grad_wrt_node_output[input_node_index])

                else:
                    grads_wrt_node_output.append(grad_wrt_node_output)
            tot_grad_wrt_node_output = sum(grads_wrt_node_output)
            grad_table[node] = tot_grad_wrt_node_output

        if hasattr(node, 'input_nodes'):
            for input_node in node.input_nodes:
                if input_node not in visited:
                    visited.add(input_node)
                    queue.put(input_node)

    return grad_table

# ------------------------
# GradientDescentOptimizer
# ------------------------
class GradientDescentOptimizer(object):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def minimize(self, loss):
        learning_rate = self.learning_rate

        class MinimizationOperation(Operation):
            def compute_output(self):
                grad_table = compute_gradients(loss)

                for var in DEFAULT_GRAPH.trainable_variables:
                    if var in grad_table:
                        grad = grad_table[var]

                    #print('var_name:', var.name)
                    #print('var_output_value', var.output_value)
                    #print('grad:', grad)
                    var.output_value = var.output_value - learning_rate*grad
        return MinimizationOperation()


# ------------------------
# GradientDescentOptimizer
# ------------------------
class ExponentialDecay(object):
    def __init__(self, learning_rate, decay_rate):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate

    def minimize(self, loss):
        learning_rate = self.learning_rate
        decay_rate = self.decay_rate
        DEFAULT_GRAPH.parameter.append(learning_rate)
        DEFAULT_GRAPH.parameter.append(decay_rate)

        class MinimizationOperation(Operation):
            def compute_output(self):
                grad_table = compute_gradients(loss)

                for var in DEFAULT_GRAPH.trainable_variables:
                    if var in grad_table:
                        grad = grad_table[var]

                    var.output_value = var.output_value - DEFAULT_GRAPH.parameter[0]*grad
        DEFAULT_GRAPH.parameter[0] = DEFAULT_GRAPH.parameter[0]*np.exp(DEFAULT_GRAPH.parameter[1])
        return MinimizationOperation()


# -------------------------
# MBGradientDescentOptimizer
# -------------------------
class MomentumGDOptimizer(object):
    def __init__(self, learning_rate, decay_rate, beta=0.9):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.beta = beta

    def minimize(self, loss):
        DEFAULT_GRAPH.parameter.append(self.learning_rate)
        DEFAULT_GRAPH.parameter.append(self.decay_rate)
        DEFAULT_GRAPH.parameter.append(self.beta)

        #learning_rate = DEFAULT_GRAPH.parameter[0]
        #decay_rate = DEFAULT_GRAPH.parameter[1]
        #beta = DEFAULT_GRAPH.parameter[2]
        #DEFAULT_GRAPH.parameter.append(learning_rate)
        #DEFAULT_GRAPH.parameter.append(decay_rate)
        #DEFAULT_GRAPH.paramter.append(beta)

        class MinimizationOperation(Operation):
            def compute_output(self):
                grad_table = compute_gradients(loss)

                for var in DEFAULT_GRAPH.trainable_variables:
                    if var in grad_table:
                        grad = grad_table[var]
                        if var in DEFAULT_GRAPH.grad_table:
                            DEFAULT_GRAPH.grad_table[var] = DEFAULT_GRAPH.parameter[2] * DEFAULT_GRAPH.grad_table[var] + (1-DEFAULT_GRAPH.parameter[2]) * grad
                        else:
                            DEFAULT_GRAPH.grad_table[var] = grad
                if DEFAULT_GRAPH.counter == 0:
                    var.output_value = var.output_value - DEFAULT_GRAPH.parameter[0] * np.exp(DEFAULT_GRAPH.parameter[1]) * DEFAULT_GRAPH.grad_table[var]#DEFAULT_GRAPH.parameter[0]*grad

        return MinimizationOperation()



# ------------------------
# AdamOptimizer
# ------------------------
class AdamOptimizer(object):
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=0.0001):
        self.learning_rate = learning_rate
        #self.decay_rate = decay_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def minimize(self, loss):
        DEFAULT_GRAPH.parameter.append(self.learning_rate)
        #DEFAULT_GRAPH.parameter.append(self.decay_rate)
        DEFAULT_GRAPH.parameter.append(self.beta1)
        DEFAULT_GRAPH.parameter.append(self.beta2)
        DEFAULT_GRAPH.parameter.append(self.epsilon)

        class MinimizationOperation(Operation):
            def compute_output(self):
                grad_table = compute_gradients(loss)
                learning_rate = DEFAULT_GRAPH.parameter[0]
                #decay_rate = DEFAULT_GRAPH.parameter[1]
                beta1 = DEFAULT_GRAPH.parameter[1]
                beta2 = DEFAULT_GRAPH.parameter[2]
                epsilon = DEFAULT_GRAPH.parameter[3]


                for var in DEFAULT_GRAPH.trainable_variables:
                    if var in grad_table:
                        grad = grad_table[var]
                        if var in DEFAULT_GRAPH.grad_table:
                            grad = grad_table[var]
                            v = (beta1 * DEFAULT_GRAPH.grad_table[var] + (1 - beta1) * grad) / (1 - np.power(beta1, epsilon))
                            s = (beta2 * DEFAULT_GRAPH.grad_table[var] + (1 - beta2) * grad) / (1 - np.power(beta2, epsilon))
                            #var.output_value = var.output_value - learning_rate * v/np.sqrt(s+epsilon)
                            #DEFAULT_GRAPH.grad_table[var] = DEFAULT_GRAPH.grad_table[var] + (1-DEFAULT_GRAPH.parameter[2]) * grad

                            #DEFAULT_GRAPH.grad_table[var] = DEFAULT_GRAPH.parameter[2] * DEFAULT_GRAPH.grad_table[var] + (1-DEFAULT_GRAPH.parameter[2]) * grad
                        else:
                            DEFAULT_GRAPH.grad_table[var] = grad#/ (1 - np.power(beta1, epsilon))
                if DEFAULT_GRAPH.counter == 0:
                    var.output_value = var.output_value - learning_rate * v/np.sqrt(s+epsilon)
        return MinimizationOperation()
