
import os
from lark import Lark, Tree, Transformer, v_args
from typing import Set, Tuple, Dict, List, Sequence, Union, Any
from .knowledge import State, Precondition, Effect, Action
from .types import baseType

class Domain:
    grammar_file = os.path.join(os.path.dirname(__file__), 'icc.grammar')
    def __init__(self):
        with open(self.grammar_file) as file:
            self.lark = Lark(file)
        self.domain_name = None # domain name indication

        self.types = {} # types as a diction, map from the type name to the actual type object

        self.predicates = {} # all the function call to the predicates stored, map from the name to actual type

        self.actions = {} # all the actions in the domain

        """
        the actual implementations of the predicates and slots
        during the actual evaluation, if a predicate does not have an implementation, it will use the default method
        as `state.get(predicate_name)`
        """
        self.implementations = {} 
    
    def check_implementation(self) -> bool:
        """
        check if all the slots in the function
        """
        return True
    
    def define_type(self, type_name, parent_name = None):
        if parent_name is None: parent_name = "object"
        self.types[type_name] = parent_name
    
    def define_predicate(self, predicate_name, parameters, output_type):
        self.predicates[predicate_name] = {"name":predicate_name,"parameters":parameters, "type":output_type}
    
    def define_action(self, action_name, parameters, precondition, effect):
        """ define symbolic action using the action name, parameters, preconditon and effect, the actual implementation is empty.
        Args:
            action_name: the name of the action
            parameters: the parameters of the action in the form of [?x ?y ... ]
            precondtion: the precondition evaluation function as a binary expression
            effect: the effect expression, notice the predicate could be more than binary
        """
        self.actions[action_name] = Action(action_name, parameters, precondition, effect)

    def load_domain_string(self, domain_string):
        pass
    
    def print_summary(self):
        print(f"domain:\n  {self.domain_name}")
        print("types:")
        for key in self.types:
            print(f"  {key} - {self.types[key]}")
        print("predicates:")
        for key in self.predicates:
            predicate_name = self.predicates[key]["name"]
            parameters = self.predicates[key]["parameters"]
            output_type = self.predicates[key]["type"]
            print(f"  {predicate_name}:{parameters} -> {output_type}");
        print("actions:")


_icc_parser = Domain()

def load_domain_string(domain_string):
    tree = _icc_parser.lark.parse(domain_string)
    icc_transformer = ICCTransformer()
    icc_transformer.transform(tree)
    return icc_transformer.domain

class ICCTransformer(Transformer):
    def __init__(self):
        super().__init__()
        self.domain = Domain()

    def domain_definition(self, domain_name):
        domain_name = domain_name[0]
        self.domain.domain_name = str(domain_name)
    
    """
    predicate definition handle part
    """
    def predicate_definition(self, args):
        predicate_name = args[0]
        parameters = args[1]
        output_type = "boolean" if len(args) == 2 else args[-1]
        self.domain.define_predicate(predicate_name, parameters, output_type)
    
    def predicate_name(self, args):return str(args[0])

    def parameters(self, args):return [str(arg) for arg in args]
    
    def parameter(self, args):return str(args[0])
    
    def object_type_name(self, args):return str(args[0])
    
    """
    type definition handler part, use the args in the format of ?x ?y...
    """
    def type_definition(self, args):
        type_name = args[0]
        if len(args) == 1:self.domain.define_type(type_name)
        if len(args) == 2:self.domain.define_type(type_name, args[1])
    
    def value_type_name(self, args):
        return args[0]
    
    def type_name(self, args):
        return str(args[0])
  
    def parent_type_name(self, args):
        return args[0]
    
    def typed_variable(self, args):
        return str(args[0]) + "-" + args[1]
    
    def vector_type_name(self, args):
        vector_choice = args[0]
        vector_size = [str(num) for num in args[1:]]
        return f"vector[{vector_choice},{vector_size}]"
    
    def vector_choice(self, args):return args[0]
    
    def vector_size(self, args):return args[0]
    
    def number(self,args):return str(args[0])

    """
    action definition handler part, use the format of
    (
        action: name
        parameters: ?x ?y ...
        precondition: (and)
        effect: (and)
    )
    """
    def action_definition(self, args):
        print(args[0])
        print(args[1])
        print(args[2])
        print(args[3])
        self.domain.define_action(*args)
    
    def action_name(self, args): return str(args[0])

    def precondition(self, args): return args[0]

    def effect(self, args): return args[0]

    """
    function and expression calls
    """
    def function_call(self, args):
        return args
    
    def function_name(self,args):
        return str(args[0])
    
    def variable(self, args):
        return str(args[0])
    
    def VARNAME(self, args):
        return str(args)

    def RULE(self, args):
        return str(args[0])

    def CONSTNAME(self, args):
        return str(args)
    
    def constant(self, args):
        return [str(arg) for arg in args]

    """Handle the part where functions are slots"""
    def slot(self,args):
        return str(args[0] + args[1][0])
    
    def SLOT(self, args):return str(args)
    
    def slot_name(self,args):
        return args
