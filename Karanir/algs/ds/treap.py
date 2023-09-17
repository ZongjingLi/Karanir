import numpy as np

class Treap:
    def __init__(self, data, lesser=True):
        self.data = data
        # [Compare Node]
        self.comparator = None
    
    def swim(self):
        return 
    
    def add(self, data):
        return -1

    def top(self, k = None):
        if k is not None:
            outputs = []
            return outputs
        else:
            return self.data[0]