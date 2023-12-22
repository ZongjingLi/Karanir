import torch
import torch.nn as nn

import numpy as np

from karanir.thanagor.physics import PropNet
from karanir.thanagor.model import KaranirThanagor, config
from karanir.thanagor.domain import load_domain_string

propnet = PropNet(config)


domain_str = f"""
    (domain BlocksAngrathar)
    (:type
        block
    )
    (:predicate
        ;; boolean predicates listed here
        is-craft ?x-block -> bool
        on ?x-block ?y-block ->bool
        clear ?x-block -> bool
        left ?x - block ?y - block -> bool
        right ?x - block ?y - block -> bool

        ;; continuous predicates start here
        pos ?x-block -> vector[float,2]
        rot ?x-block -> float
    )
    (:action-definitions
        ;; some symbolic actions for place and pick
        (
            action: pick
            parameters: ?x ?y
            precondition: (and (on ?x) (and (on ?y) (clear ?x)) )
            effect: (and (??f ?y) Pi )
        )
    )
"""

from karanir.envs.blockworld import *
game = BlockWorldEnv(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)

domain = load_domain_string(domain_str)

model = KaranirThanagor(domain, config)

arcade.run()

#model.print_summary()