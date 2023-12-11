
import os
from lark import Lark, Tree, Transformer, v_args
from typing import Set, Tuple, Dict, List, Sequence, Union, Any

class DomainBase(object):
    name : str
    """The name of the domain"""

    grammar_file = os.path.join(os.path.dirname(__file__), "icc.grammar")

    name: str

    types: Dict

    functions: Dict[str, Any]

    constants: Dict[str, Any]

    operators: Dict[str, Any]
    
    def __init__(self, domain_str = None):
        self.operators = {}
        self.operator_templates = {}
        with open(type(self).grammar_file) as f:
            self.lark = Lark(f)
        
        self.types = {}
        
    def load(self, file) -> Tree:
        with open(file) as f:
            return self.lark.parse(f.read())
    
    def loads(self, string) -> Tree:
        """load a domain or problem string and return the corresponding tree."""
        return self.lark.parse(string)

    def make_domain(self, parse_tree: Tree):
        """Construct a Domain fro a Tree"""
        assert parse_tree.children[0].data == "definition"
        transformer = ICCTransformer(self)
        transformer.transform(parse_tree)
        domain = transformer.domain

        return domain
    
    def print_summary(self):
        """Print out the sumary of the Domain"""
        print(f'Domain: {self.name}')

        print("types:", self.types)
    
    def define_type(self,name, parent_name):
        name = str(name)
        parent_name = str(parent_name)
        if parent_name not in self.types:
            self.types[parent_name] = [name]
        else: self.types[parent_name].append(name)


_icc_parser_ = DomainBase()

def load_domain_file(filename: str):
    tree = _icc_parser_.load(filename)
    domain = _icc_parser_.make_domain(tree)
    return domain

def load_domain_string(domain_string: str):
    tree = _icc_parser_.loads(domain_string)
    domain = _icc_parser_.make_domain(tree)
    return domain

def parse_domain_string(domain_string):
    return domain_string

class ICCTransformer(Transformer):
    """The tree-to-object transformer for PDSketch domain and problem files. Users should not use this class directly"""
    def __init__(self, init_domain, Domain = None):
        super().__init__()
        self.domain = init_domain
        self.problem = None
        self.allow_object_constants = True
        self.ignore_unknown_predicates = True
        self.ingored_predicates: Set[str] = set()

    def definition_decl(self,definition_type):
        definition_type, definition_name = definition_type
        if definition_type.value == "domain":
            self.domain.name = definition_name.value
        
    def type_definition(self, args):
        """Parse a type definition
        """
        if isinstance(args[-1], Tree) and args[-1].data == "parent_type_name":
            parent_line , parent_name = -1, args[-1].children[0].children[0].children[0]
            #parent_line, parent_name = args[-1].children[0]
            args = args[:-1]
        else:
            parent_line, parent_name = -1, 'object'

        for arg in args:
            arg_line, arg_name = arg.children[0],arg.children[0].children[0]
            if arg_line == parent_line:
                self.domain.define_type(arg_name, parent_name)
            else:
                self.domain.define_type(arg_name, parent_name)
            
    def predicate_definition(self, name, *args):
        print(name, args)
        print("")

    def constants_definition(self, *args):
        for arg in args:
            self.domain.constants[arg.name] = arg
