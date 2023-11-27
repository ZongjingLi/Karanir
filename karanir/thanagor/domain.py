
class DomainBase(object):
    name : str
    """The name of the domain"""

    def __init__(self, domain_str):
        self.attributes = 0
        self.relations = 0