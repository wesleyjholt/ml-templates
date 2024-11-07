import numpy as np

from state import State
from update import IterData, increment_a, multiply_b_by_factor
from callback import print_a, print_b

def evolve_state(hyperparams: dict):
    """Let the state evolve within nested loops."""
    state = State(a=hyperparams['state']['a'], b=hyperparams['state']['b'])

    for i in range(10):
        
        state = increment_a(state)
        print_a(state)

        for k in np.linspace(0.98, 1.025, 10):

            data = IterData(outer=i, inner=k)
            state = multiply_b_by_factor(state, data, hyperparams, num_times=3)
            print_b(state, hyperparams, num_decimals=4)

if __name__=='__main__':
    from utils import read_yaml
    hyperparams = read_yaml('hyperparams.yml')
    evolve_state(hyperparams)