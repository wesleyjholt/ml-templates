# REPLACE THIS CODE WITH YOUR OWN DATA LOADER!

import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
import numpy as np
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader
import torchvision
from typing import Iterable
import os

class AbstractTrainTestDataLoader(eqx.Module):
    train_state_dict: dict
    train_iterable: Iterable

class TorchEpochBatchTrainTestStatefulDataLoader(AbstractTrainTestDataLoader):
    i_batch: int
    i_epoch: int
    train_state_dict: dict
    train_iterable: Iterable
    test_iterable: Iterable | None
    def __init__(self, train_dataset, test_dataset=None, batch_size=None, **kwargs):
        self.train_iterable = StatefulDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        if test_dataset is not None:
            self.test_iterable = StatefulDataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        else:
            self.test_iterable = None
        self.train_state_dict = self.train_iterable.state_dict()
        self.i_batch = 0
        self.i_epoch = 0

class MNISTDataLoader(TorchEpochBatchTrainTestStatefulDataLoader):
    raw_data_dir: str
    def __init__(self, raw_data_dir, batch_size, **kwargs):
        self.raw_data_dir = raw_data_dir
        train_dataset, test_dataset = self.get_mnist_data()
        super().__init__(train_dataset=train_dataset, test_dataset=test_dataset, batch_size=batch_size, **kwargs)

    def get_mnist_data(self):
        if not os.path.exists(self.raw_data_dir):
            download = True
        else:
            download = False
        normalise_data = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        train_dataset = torchvision.datasets.MNIST(
            "MNIST",
            train=True,
            download=download,
            transform=normalise_data,
        )
        test_dataset = torchvision.datasets.MNIST(
            "MNIST",
            train=False,
            download=download,
            transform=normalise_data,
        )
        return train_dataset, test_dataset

class SpiralDataLoader(TorchEpochBatchTrainTestStatefulDataLoader):
    def __init__(self, dataset_size, batch_size, *, seed, **kwargs):
        dataset = SpiralDataset(dataset_size, key=jrandom.PRNGKey(seed))
        super().__init__(train_dataset=dataset, batch_size=batch_size, shuffle=True, **kwargs)

class SpiralDataset(Dataset):
    def __init__(self, dataset_size, *, key):
        self.data = self.get_spiral_data(dataset_size, key=key)
    
    def __getitem__(self, idx):
        return self.data[0][idx], self.data[1][idx]
    
    def __len__(self):
        return len(self.data[0])

    def get_spiral_data(self, dataset_size, *, key):
        t = jnp.linspace(0, 2 * np.pi, 16)
        offset = jrandom.uniform(key, (dataset_size, 1), minval=0, maxval=2 * np.pi)
        x1 = jnp.sin(t + offset) / (1 + t)
        x2 = jnp.cos(t + offset) / (1 + t)
        y = jnp.ones((dataset_size, 1))

        half_dataset_size = dataset_size // 2
        x1 = x1.at[:half_dataset_size].multiply(-1)
        y = y.at[:half_dataset_size].set(0)
        x = jnp.stack([x1, x2], axis=-1)

        return np.array(x), np.array(y)
