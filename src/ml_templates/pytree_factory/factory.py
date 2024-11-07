"""
Factory module for generating objects from a generator name and hyperparameters.
Useful for saving and loading models.
"""

from abc import ABC, abstractmethod
import json
import jax
import jax.random as jrandom
import jax.numpy as jnp
import equinox as eqx
from ._example_pytrees import StandardScaler, UnscaledModel

class PyTreeFactory(ABC):
    """Base class for a PyTree factory.

    User must implement the __init__ method and initialize the generator index within it.
    
    Properties
    ----------
    generator_index : dict
        A dictionary to store generator functions indexed by their names.
    
    Methods
    -------
    __init__()
        Initializes the generator index. User must implement this method.
    generate(generator_name, hyperparams)
        Generates a PyTree using the specified generator and hyperparameters.
    save_pytree(filename, pytree, generator_name, hyperparams=None)
        Saves a PyTree to a file along with its generator name and hyperparameters.
    load_pytree(filename)
        Loads a PyTree from a file using the stored generator name and hyperparameters.
    """
    def __init__(self, parent_factories=None):
        self.generators = {}
        self.parent_factories = parent_factories if not None else {}
    
    def register_parent_factory(self, name, factory):
        self.parent_factories[name] = factory

    def register_generator(self, name, generator):
        self.generators[name] = generator

    def generate(self, generator_name, hyperparams):
        return self.generators[generator_name](**hyperparams)
    
    def make_skeleton(self, generator_name, hyperparams):
        return eqx.filter_eval_shape(self.generators[generator_name], **hyperparams)

    def save_pytree(
        self, 
        pytree,
        filename, 
        generator_name, 
        hyperparams=None
    ):
        with open(filename, "wb") as f:
            if hyperparams is None:
                hyperparams = {}
            hyperparams_with_gen_name = hyperparams | {'_generator_name': generator_name}
            hyperparam_str = json.dumps(hyperparams_with_gen_name)
            f.write((hyperparam_str + "\n").encode())
            eqx.tree_serialise_leaves(f, pytree)

    def load_pytree(
        self, 
        filename
    ):
        with open(filename, "rb") as f:
            hyperparam_str = f.readline().decode().strip()
            hyperparams = json.loads(hyperparam_str)
            name = hyperparams.pop('_generator_name')
            like = self.make_skeleton(name, hyperparams)
            pytree = eqx.tree_deserialise_leaves(f, like)
            return pytree

# class ExampleFactory(AbstractFactory):
#     """Example factory.
    
#     Can be used for generating MLPs with built-in input/output scaling.
#     """
#     generator_index: dict

#     def __init__(self):
#         self.generator_index = {
#             'make_mlp': self.make_mlp,
#             'make_standard_scaler': self.make_standard_scaler,
#             'make_unscaled_model': self.make_unscaled_model
#         }

#     def make_mlp(self, **kwargs):
#         if 'seed' not in kwargs:
#             kwargs['seed'] = 0
#         if 'activation' not in kwargs:
#             kwargs['activation'] = jax.nn.gelu
#         kwargs['key'] = jrandom.PRNGKey(kwargs.pop('seed'))
#         return eqx.nn.MLP(**kwargs)

#     def make_standard_scaler(self, shape=()):
#         return StandardScaler(mean=jnp.zeros(shape), std=jnp.ones(shape))

#     def make_unscaled_model(
#         self,
#         scaled_model_gen_and_hyp,
#         input_scaler_gen_and_hyp,
#         output_scaler_gen_and_hyp
#     ):
#         # Helper function
#         def _generate(gen_and_hyp):
#             generator_name, hyperparams = gen_and_hyp
#             return self.generate(generator_name, hyperparams)
        
#         # Use factory to generate the contents of the injection model
#         scaled_model = _generate(scaled_model_gen_and_hyp)
#         input_scaler = _generate(input_scaler_gen_and_hyp)
#         output_scaler = _generate(output_scaler_gen_and_hyp)
        
#         return UnscaledModel(scaled_model=scaled_model, input_scaler=input_scaler, output_scaler=output_scaler)