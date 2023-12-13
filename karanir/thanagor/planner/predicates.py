# -*- coding: utf-8 -*-
# @Author: Meleko
# @Date:   2023-10-13 13:07:56
# @Last Modified by:   Melkor
# @Last Modified time: 2023-10-19 04:42:09
import torch
import torch.nn as nn

import numpy as np

import matplotlib.pyplot as plt

from karanir.dklearn.nn import FCBlock


class State:
    def __init__(self, data):
        """ construct a symbolic or hybrid state
        Args:
            data: a diction that maps the data of each state to the actual value
        """
        self.data = data
    
    def get(self, predicate_name): return self.data[predicate_name]

class Precondition:
    def __init__(self, bool_expression):
        self.bool_expression = bool_expression

class Effect:
    def __init__(self, bool_expression):
        self.bool_expression = bool_expression
    
    def __call__(self) -> bool:
        return 0

