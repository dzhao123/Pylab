class SegmentTree:
    def __init__(self, start, end):
        self.start = start
        self.emd = end
        self.sum = 0
        self.left = None
        self.right = None

