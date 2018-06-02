from .graph import *
from .operator import *
from .placeholder import *
from .variable import *

# -------------
# Session
# -------------
class Session(object):
    def __init__(self):
        self.graph = DEFAULT_GRAPH

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.close()

    def close(self):
        all_nodes = (self.graph.constants + self.graph.variables +
                     self.graph.placeholders + self.graph.operations +
                     self.graph.trainable_variables)

        for node in all_nodes:
            node.output_value = None

    def run(self, operation, feed_dict=None):
        #postorder_nodes = _get_prerequisite(operation)

        if type(operation) is list:
            loss_op, optmiz_op = operation[0], operation[1]
            batches = 0
            for key, val in feed_dict.items():
                batches = len(val)
            DEFAULT_GRAPH.counter = batches

            output_value = 0
            postorder_nodes_loss = _get_prerequisite(loss_op)
            postorder_nodes_optmiz = _get_prerequisite(optmiz_op)
            for sample in range(batches):
                DEFAULT_GRAPH.counter -= 1
                for node in postorder_nodes_loss:
                    if type(node) is Placeholder:
                        #node.output_value = feed_dict[node]
                        node.output_value = np.array([feed_dict[node][sample]])
                    else:
                        node.compute_output()

                output_value += loss_op.output_value

                for node in postorder_nodes_optmiz:
                    if type(node) is Placeholder:
                        #node.output_value = feed_dict[node]
                        node.output_value = np.array([feed_dict[node][sample]])
                    else:
                        node.compute_output()
            loss_op.output_value = output_value
            return loss_op.output_value, optmiz_op

        else:
            postorder_nodes = _get_prerequisite(operation)

            batches = 0
            output_value = 0
            DEFAULT_GRAPH.counter = 0
            for key, val in feed_dict.items():
                batches = len(val)

            for sample in range(batches):
                #DEFAULT_GRAPH.counter -= 1
                for node in postorder_nodes:
                    if type(node) is Placeholder:
                        #node.output_value = feed_dict[node]
                        node.output_value = np.array([feed_dict[node][sample]])
                    else:
                        node.compute_output()

                if operation.output_value is None:
                    output_value = None
                    continue
                else:
                    output_value += operation.output_value
                    #print('sample_output_value:', output_value)
            operation.output_value = output_value
            #print('operation_output_value:', operation.output_value)
            return operation.output_value


def _get_prerequisite(operation):
    postorder_nodes = []

    def postorder_traverse(operation):
        if isinstance(operation, Operation):# or isinstance(operation, Variable):
            for input_node in operation.input_nodes:
                #print('input_node:', input_node)
                #print('operation:', operation)
                postorder_traverse(input_node)
        postorder_nodes.append(operation)

    postorder_traverse(operation)
    return postorder_nodes
