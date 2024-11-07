import jax.numpy as jnp
import equinox as eqx

class StandardScaler(eqx.Module):
    """Standardize input data by removing the mean and scaling to unit variance."""
    mean: float = eqx.field(default_factory=lambda: jnp.array(0.0))
    std: float = eqx.field(default_factory=lambda: jnp.array(1.0))

    @classmethod
    def fit(cls, data, axis=None):
        mean = data.mean(axis=axis)
        std = data.std(axis=axis)
        return cls(mean, std)

    def forward(self, data):
        return (data - self.mean) / self.std

    def inverse(self, data):
        return data * self.std + self.mean

class UnscaledModel(eqx.Module):
    """A model that scales its input before passing it to another model."""
    scaled_model: eqx.Module
    input_scaler: StandardScaler
    output_scaler: StandardScaler

    def __call__(self, x):
        x = self.input_scaler.forward(x)
        x = self.scaled_model(x)
        return self.output_scaler.inverse(x)