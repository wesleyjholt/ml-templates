from typing import NamedTuple, Optional, Any
State = Any

class IterData(NamedTuple):
    """Stores the data associated with each layer of the nested loop."""
    # IMPLEMENT ME!
    outer: Optional[Any] = None  # DELETE ME!
    inner: Optional[Any] = None  # DELETE ME!

# Here, we define some pure functions which modify the state. 
# You must define your own, but here are some examples:

# DELETE ME!
def increment_a(state: State) -> State:
    """Increments the value of a by 1."""
    state.a = state.a + 1
    return state

# DELETE ME!
def multiply_b_by_factor(state: State, data: IterData, hyperparams: dict, num_times: int) -> State:
    """Multiplies the value of b by a factor."""
    if hyperparams['train']['alternate_sign']:
        for _ in range(num_times):
            state.b = -state.b * data.inner
    return state