from karanir.thanagor.types import *
from karanir.thanagor.program import Program
from torch.utils.data import Dataset, DataLoader
from karanir.algs.search import run_heuristic_search
from karanir.algs.search.configuration_space import ProblemSpace
import karanir.thanagor.domain as dom
from karanir.thanagor.dsl.vqa_primitives import *


domain_string = text = """
(domain blockworld)
(:type
    source - int
    block
    pos - vector[float, 3, 7] ;; the pos vector representation
    category - vector[int, 4] ;; the category vector representation
)
(:predicate
    clear ?x ?y
    is-red ?x
    is-blue ?x
    is-green ?x
)
(:action-definitions
    (
        action: pickup
        parameters: ?x ?y
        precondition:and
        effect:and
    )
    (
        action: placeon
        parameters: ?x ?y
        precondition: holdingxandyisclear
        effect: placesomethingonit
    )
)
"""


domain = dom.load_domain_string(domain_string)
domain.print_summary()

from karanir.thanagor.knowledge import CentralExecutor
from karanir.thanagor.model import KaranirThanagor, config
KFT = KaranirThanagor(domain, config)

KFT.print_summary()


context = {
    "end": logit(torch.ones(7)),
    "features": torch.randn([7,100]),
    "executor": KFT.central_executor
}

red = Primitive("red", Concept, "red")
p = Program.parse("(Count (Filter $0 red))")

print(p.evaluate({0:context})["end"])