
import os
from lark import Lark, Tree, Transformer, v_args

class DomainBase(object):
    name : str
    """The name of the domain"""

    grammar_file = os.path.join(os.path.dirname(__file__), "icc.grammar")
    
    def __init__(self, domain_str = None):
        self.attributes = 0
        self.relations = 0
        with open(type(self).grammar_file) as f:
            self.lark = Lark(f)
        
    def load(self, file) -> Tree:
        with open(file) as f:
            return self.lark.parse(f.read())
    
    def loads(self, string) -> Tree:
        """load a domain or problem string and return the corresponding tree."""
        return self.lark.parse(string)

_icc_parser_ = DomainBase()

def load_domain_file(filename: str):
    return _icc_parser_.load(filename)

def load_domain_string(domain_string: str):
    return _icc_parser_.loads(domain_string)

def parse_domain_string(domain_string):
    return domain_string