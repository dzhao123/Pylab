from .graph import *
from .operator import Operation
from .placeholder import *
from .variable import *
# -------------
# compute_gradints
# -------------
def compute_gradients(target_op):
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

                    var.output_value = var.output_value - learning_rate*grad
        return MinimizationOperation()
