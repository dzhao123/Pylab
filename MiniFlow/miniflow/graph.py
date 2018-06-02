class Graph(object):
    def __init__(self):
        self.operations = []
        self.variables = []
        self.placeholders = []
        self.trainable_variables = []
        self.constants = []
        self.parameter = []
        self.counter = 0
        self.grad_table = {}

    def __enter__(self):
        global DEFAULT_GRAPH
        DEFAULT_GRAPH = self
        #global PARAMETER
        #PARAMETER = self.parameter
        #global total
        #total = self.total

        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        global DEFAULT_GRAPH
        global PARAMETER

    def as_default(self):
        return self
