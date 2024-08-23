# REPLACE THIS CODE WITH YOR OWN OPTIMIZATION ALGORITHM!

import equinox as eqx
import optax
from typing import Any

class AdamOptimizer(eqx.Module):
    state: Any
    optim: Any
    def __init__(self, model, lr, **kwargs):
        self.optim = optax.adam(lr)
        self.state = self.optim.init(eqx.filter(model, eqx.is_inexact_array))