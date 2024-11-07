import jax.numpy as jnp
import equinox as eqx
from typing import NamedTuple, Optional, Any

from .state_factory import State

class IterData(NamedTuple):
    epoch: Optional[Any] = None
    batch: Optional[Any] = None

def increment_epoch(state: State) -> State:
    if state.dataloader.i_batch == 0:
        return eqx.tree_at(lambda x: x.dataloader.i_epoch, state, state.dataloader.i_epoch + 1)
    else:
        return state

@eqx.filter_jit
def train_step(state: State, data: IterData) -> State:
    # Extract data
    x, y = data.batch[1]
    optim = state.optimizer.optim
    opt_state = state.optimizer.state
    model = state.model
    compute_loss = state.loss.compute_loss

    # Do train step
    loss, grads = compute_loss(model, x, y)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)

    # Update state
    state = eqx.tree_at(lambda x: x.model, state, model)
    state = eqx.tree_at(lambda x: x.optimizer.state, state, opt_state)
    state = eqx.tree_at(lambda x: x.dataloader.i_batch, state, state.dataloader.i_batch + 1)
    state = eqx.tree_at(lambda x: x.loss.recent_accumulated_loss, state, state.loss.recent_accumulated_loss + loss)
    state = eqx.tree_at(lambda x: x.loss.num_recent, state, state.loss.num_recent + 1)

    return state

def reset_accumulated_loss(state: State) -> State:
    state = eqx.tree_at(lambda x: x.loss.recent_accumulated_loss, state, jnp.zeros_like(state.loss.recent_accumulated_loss))
    state = eqx.tree_at(lambda x: x.loss.num_recent, state, jnp.zeros_like(state.loss.num_recent))
    return state

def reset_batch_counter(state: State) -> State:
    return eqx.tree_at(lambda x: x.dataloader.i_batch, state, 0)