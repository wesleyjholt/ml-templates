# REPLACE THIS CODE WITH YOUR OWN MODEL!

import jax
from jax import lax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

class RNN(eqx.Module):
    # This is a simple Recurrent Neural Network (RNN) model.
    # From equinox docs: https://docs.kidger.site/equinox/examples/train_rnn/
    cell: eqx.Module
    linear: eqx.nn.Linear
    bias: jax.Array
    hidden_size: int = eqx.field(static=True)  # Note: Things in optax serialisation get messed up it this is not a static field.

    def __init__(self, in_size, out_size, hidden_size, *, key, **kwargs):
        ckey, lkey = jr.split(key)
        self.hidden_size = hidden_size
        self.cell = eqx.nn.GRUCell(in_size, hidden_size, key=ckey)
        self.linear = eqx.nn.Linear(hidden_size, out_size, use_bias=False, key=lkey)
        self.bias = jnp.zeros(out_size)

    def __call__(self, input):
        hidden = jnp.zeros((self.hidden_size,))

        def f(carry, inp):
            return self.cell(inp, carry), None

        out, _ = lax.scan(f, hidden, input)
        # sigmoid because we're performing binary classification
        return jax.nn.sigmoid(self.linear(out) + self.bias)