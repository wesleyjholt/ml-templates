import jax
import jax.random as jrandom
import equinox as eqx
from time import time
from typing import Optional

from ml_templates.pytree_factory import PyTreeFactory
from .model import RNN, CNN
from .optimizer import AdamOptimizer
from .dataloader import MNISTDataLoader, SpiralDataLoader
from .loss import BinaryCrossEntropyLoss, MSELoss, CrossEntropyLoss

# Model
model_factory = PyTreeFactory()
model_factory.register_generator("rnn", RNN)
model_factory.register_generator("cnn", CNN)

# Dataloader
dataloader_factory = PyTreeFactory()
dataloader_factory.register_generator("mnist", MNISTDataLoader)
dataloader_factory.register_generator("spiral", SpiralDataLoader)

# Loss
loss_factory = PyTreeFactory()
loss_factory.register_generator("mse", MSELoss)
loss_factory.register_generator("bin_cross_entropy", BinaryCrossEntropyLoss)
loss_factory.register_generator("cross_entropy", CrossEntropyLoss)

# Optimizer
# Note that this will have nonserializable hyperparameters (i.e., the `model` hyperparameter).
# This is okay, because we never plan to actually save the optimizer hyperparameters as a file.
# We only plan to save the state hyperparameters.
optimizer_factory = PyTreeFactory()
optimizer_factory.register_generator("adam", AdamOptimizer)

# Timestamps
class TimeStamps(eqx.Module):
    first_start_time: float
    last_restart_time: float

# State
class State(eqx.Module):
    model: eqx.Module
    dataloader: eqx.Module
    loss: eqx.Module
    optimizer: eqx.Module
    timestamps: eqx.Module

class StateFactory(PyTreeFactory):
    def __init__(self):
        self.parent_factories = {
            "model": model_factory, 
            "dataloader": dataloader_factory,
            "loss": loss_factory,
            "optimizer": optimizer_factory
        }
        self.generators = {"state": self.make_state}
    
    def make_state(
        self,
        model_name,
        model_hyperparams,
        dataloader_name,
        dataloader_hyperparams,
        loss_name,
        loss_hyperparams,
        optimizer_name,
        optimizer_hyperparams
    ):
        model = self.parent_factories["model"].generate(model_name, model_hyperparams)
        dataloader = self.parent_factories["dataloader"].generate(dataloader_name, dataloader_hyperparams)
        loss = self.parent_factories["loss"].generate(loss_name, loss_hyperparams)
        optimizer = self.parent_factories["optimizer"].generate(optimizer_name, dict(model=model, **optimizer_hyperparams))
        current_time = time()
        timestamps = TimeStamps(current_time, current_time)
        return State(model=model, dataloader=dataloader, loss=loss, optimizer=optimizer, timestamps=timestamps)

state_factory = StateFactory()