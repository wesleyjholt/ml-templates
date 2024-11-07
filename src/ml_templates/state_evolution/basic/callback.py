# These are functions that do not modify the state, but probe it in some way
# and potentially do some non-pure thing (i.e., function with side-effects).
# You must define your own, but here are some examples:

from typing import Any
State = Any

# DELETE ME!
def print_a(state: State):
    """Prints the value of a."""
    print(f'The value of `a` is {state.a}')

# DELETE ME!
def print_b(state: State, hyperparams: dict, num_decimals: int):
    """Prints the value of b to a certain number of decimal places."""
    print(f'The value of `b` started at {hyperparams['state']['b']:.{num_decimals}f} and is now at {state.b:.{num_decimals}f}')