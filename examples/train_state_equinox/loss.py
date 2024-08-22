from jax import vmap
import jax.numpy as jnp
import equinox as eqx
from typing import Callable

@eqx.filter_value_and_grad
def compute_loss(model, x, y):
    pred_y = vmap(model)(x)
    # Trains with respect to binary cross-entropy
    return -jnp.mean(y * jnp.log(pred_y) + (1 - y) * jnp.log(1 - pred_y))

class Loss(eqx.Module):
    compute_loss: Callable
    recent_accumulated_loss: float
    num_recent: int
    def __init__(self, **kwargs):
        self.num_recent = jnp.array(0)
        self.recent_accumulated_loss = jnp.array(0.0)
        self.compute_loss = compute_loss