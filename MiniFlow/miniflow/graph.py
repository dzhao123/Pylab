class Graph(object):
    def __init__(self):
        self.operations = []
        self.variables = []
        self.placeholders = []
        self.trainable_variables = []
        self.constants = []
        self.counter = 1
        self.total = 1

    def __enter__(self):
        global DEFAULT_GRAPH
        DEFAULT_GRAPH = self
        global counter
        counter = self.counter
        global total
        total = self.total

        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        global DEFAULT_GRAPH

    def as_default(self):
        return self
