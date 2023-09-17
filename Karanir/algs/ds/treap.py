import numpy as np

def swap(a, b):return b,a

class Treap:
    def __init__(self, data, lesser=True):
        self.data = data
        # [Compare Node]
        self.comparator = None
        self.lesser = lesser
    
    def add(self, node):
        self.data.append(node)
        if self.comparator is not None:
            flag = True
            curr_node = self.data
            parent_node = self.data[-1] 
            while (flag):
                if self.lesser:
                    flag = False
                else:
                    flag = False
        return -1

    def top(self, k = None):
        if k is not None:return self.data[:k]
        else:return self.data[0]