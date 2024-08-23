# REPLACE THIS CODE WITH YOUR OWN TRAINING STATE!

import jax.random as jr
import equinox as eqx
from time import time

from model import RNN as Model
from optimizer import AdamOptimizer as Optimizer
from dataloader import TorchDataLoader as DataLoader
from loss import BinaryCrossEntropyLoss as Loss

class TimeStamps(eqx.Module):
    first_start_time: float
    last_restart_time: float

class TrainState(eqx.Module):
    model: Model
    optimizer: Optimizer
    dataloader: DataLoader
    loss: Loss
    timestamps: TimeStamps
    def __init__(
        self,
        seed,
        in_size, 
        out_size, 
        hidden_size,
        learning_rate,
        dataset_size,
        batch_size,
        make_skeleton=False,
        **kwargs
    ):
        model_key, dataloader_key = jr.split(jr.PRNGKey(seed))
        current_time = time()
        if not make_skeleton:
            self.model = Model(in_size, out_size, hidden_size, key=model_key)
            self.optimizer = Optimizer(self.model, learning_rate)
            self.dataloader = DataLoader(dataset_size, batch_size, key=dataloader_key)
            self.loss = Loss()
            self.timestamps = TimeStamps(current_time, current_time)
        else:
            self.model = eqx.filter_eval_shape(Model, in_size, out_size, hidden_size, key=model_key)
            self.optimizer = eqx.filter_eval_shape(Optimizer, self.model, learning_rate)
            self.dataloader = eqx.filter_eval_shape(DataLoader, dataset_size, batch_size, key=dataloader_key)
            self.loss = eqx.filter_eval_shape(Loss)
            self.timestamps = eqx.filter_eval_shape(TimeStamps, current_time, current_time)