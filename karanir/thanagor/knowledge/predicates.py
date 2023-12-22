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
    
    def __call__(self, bool_expression) -> bool:
        return 0

class Effect:
    def __init__(self, bool_expression):
        self.bool_expression = bool_expression
    
    def __call__(self) -> bool:
        return 0
    
    def split_effect(self):
        """split the effect into two parts. effect+, effect-
        Returns:
            effect+: the added predicates to the known state
            effect-: the removed predicates to the known state
        """
        return 0

class Action:
    def __init__(self, action_name, parameters, precondition, effect):
        """ construct an symboic action with preconditinos and effects
        Args:
            action_name: the name of the action
            precondition: a boolean expression that is callable for an input predicate state
            effect: a set of assignment expressions to known predicates
        """
        super().__init__()
        self.action_name = action_name
        self.parameters = parameters
        if isinstance(precondition, Precondition):
            self.precondition = precondition
        else: self.precondition = Precondition(precondition)

        if not isinstance(effect, Effect):
            self.effect = Effect(effect)
        else: self.effect = effect
    
    def apply(self, state):
        if self.precondition(state):
            return