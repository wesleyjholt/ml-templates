# REPLACE THIS CODE WITH YOR OWN OPTIMIZATION ALGORITHM!

import equinox as eqx
import optax
from typing import Any

class AbstractOptimizer(eqx.Module):
    state: Any
    optim: Any

class OptaxOptimizer(AbstractOptimizer):
    state: Any
    optim: Any
    
    def __init__(self, optim_gen, model, **kwargs):
        self.optim = optim_gen(**kwargs)
        self.state = self.optim.init(eqx.filter(model, eqx.is_inexact_array))

class AdamOptimizer(OptaxOptimizer):
    def __init__(self, model, **kwargs):
        super().__init__(optax.adam, model, **kwargs)