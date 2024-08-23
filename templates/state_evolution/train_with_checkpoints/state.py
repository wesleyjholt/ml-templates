# REPLACE THIS CODE WITH YOUR OWN TRAINING STATE!

import jax.random as jr
import equinox as eqx

from model import RNN
from optimizer import AdamOptimizer
from dataloader import TorchDataLoader
from loss import Loss

class TrainState(eqx.Module):
    model: RNN
    optimizer: AdamOptimizer
    dataloader: TorchDataLoader
    loss: Loss
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
        if not make_skeleton:
            self.model = RNN(in_size, out_size, hidden_size, key=model_key)
            self.optimizer = AdamOptimizer(self.model, learning_rate)
            self.dataloader = TorchDataLoader(dataset_size, batch_size, key=dataloader_key)
            self.loss = Loss()
        else:
            self.model = eqx.filter_eval_shape(RNN, in_size, out_size, hidden_size, key=model_key)
            self.optimizer = eqx.filter_eval_shape(AdamOptimizer, self.model, learning_rate)
            self.dataloader = eqx.filter_eval_shape(TorchDataLoader, dataset_size, batch_size, key=dataloader_key)
            self.loss = eqx.filter_eval_shape(Loss)