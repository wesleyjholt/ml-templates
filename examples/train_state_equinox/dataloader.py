import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import numpy as np
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader
from typing import Iterable


class TorchDataLoader(eqx.Module):
    i_batch: int
    i_epoch: int
    state_dict: dict
    iterable: Iterable
    def __init__(self, dataset_size, batch_size, *, key, **kwargs):
        dataset = SpiralDataset(dataset_size, key=key)
        self.iterable = StatefulDataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.state_dict = self.iterable.state_dict()
        self.i_batch = 0
        self.i_epoch = 0

def get_data(dataset_size, *, key):
    t = jnp.linspace(0, 2 * np.pi, 16)
    offset = jr.uniform(key, (dataset_size, 1), minval=0, maxval=2 * np.pi)
    x1 = jnp.sin(t + offset) / (1 + t)
    x2 = jnp.cos(t + offset) / (1 + t)
    y = jnp.ones((dataset_size, 1))

    half_dataset_size = dataset_size // 2
    x1 = x1.at[:half_dataset_size].multiply(-1)
    y = y.at[:half_dataset_size].set(0)
    x = jnp.stack([x1, x2], axis=-1)

    return np.array(x), np.array(y)

class SpiralDataset(Dataset):
    def __init__(self, dataset_size, *, key):
        self.data = get_data(dataset_size, key=key)
    
    def __getitem__(self, idx):
        return self.data[0][idx], self.data[1][idx]
    
    def __len__(self):
        return len(self.data[0])

