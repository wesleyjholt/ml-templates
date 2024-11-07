# REPLACE THIS CODE WITH YOUR OWN MODEL!

from abc import abstractmethod
import jax
from jax import lax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
from jaxtyping import Float, Array

class AbstractModel(eqx.Module):
    @abstractmethod
    def __call__(self, x):
        raise NotImplementedError

class RNN(AbstractModel):
    # This is a simple Recurrent Neural Network (RNN) model.
    # From equinox docs: https://docs.kidger.site/equinox/examples/train_rnn/
    cell: eqx.Module
    linear: eqx.nn.Linear
    bias: jax.Array
    hidden_size: int = eqx.field(static=True)  # Note: Things in optax serialisation get messed up it this is not a static field.

    def __init__(self, in_size, out_size, hidden_size, *, seed, **kwargs):
        ckey, lkey = jrandom.split(jrandom.PRNGKey(seed))
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

class CNN(eqx.Module):
    """Convolutional neural network for classifying handwritten digits from the MNIST dataset."""
    layers: list

    def __init__(self, seed):
        key1, key2, key3, key4 = jrandom.split(jrandom.PRNGKey(seed), 4)
        self.layers = [
            eqx.nn.Conv2d(1, 32, kernel_size=3, key=key1),
            jax.nn.relu,
            eqx.nn.MaxPool2d(kernel_size=2),
            eqx.nn.Conv2d(32, 64, kernel_size=3, key=key2),
            jax.nn.relu,
            eqx.nn.MaxPool2d(kernel_size=2),
            jnp.ravel,
            eqx.nn.Linear(30976, 128, key=key3),
            jax.nn.relu,
            eqx.nn.Linear(128, 10, key=key4),
            jax.nn.log_softmax,
        ]

    def __call__(self, x: Float[Array, "1 28 28"]) -> Float[Array, "10"]:  # Side note: These are shaped-array type hints, made possible by the package jaxtyping.
        for layer in self.layers:
            x = layer(x)
        return x