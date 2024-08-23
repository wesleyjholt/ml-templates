# REPLACE THIS CODE WITH YOUR OWN LOSS FUNCTION!

from jax import vmap
import jax.numpy as jnp
import equinox as eqx
from typing import Callable

class AbstractLoss(eqx.Module):
    recent_accumulated_loss: float
    num_recent: int
    def __init__(self, **kwargs):
        self.num_recent = jnp.array(0)
        self.recent_accumulated_loss = jnp.array(0.0)

class BinaryCrossEntropyLoss(AbstractLoss):
    compute_loss: Callable
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.compute_loss = compute_binary_cross_entropy_loss

class MSELoss(AbstractLoss):
    compute_loss: Callable
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.compute_loss = compute_mse_loss

@eqx.filter_value_and_grad
def compute_binary_cross_entropy_loss(model, x, y):
    pred_y = vmap(model)(x)
    return -jnp.mean(y * jnp.log(pred_y) + (1 - y) * jnp.log(1 - pred_y))

@eqx.filter_value_and_grad
def compute_mse_loss(model, x, y):
    pred_y = vmap(model)(x)
    return jnp.mean((y - pred_y)**2)