# NN training (with checkpoints) as a state evolution

This tempate shows how to use the state evolution organizational pattern to organize your neural network training code. Additionally, it shows how to "checkpoint" your training, which allows you to pick up training from where you left off in case of a crash or other interruption. 

## Usage

To start a new training session:
```bash
python train.py --reset
```
If training gets interrupted, you can resume from the last checkpoint simply by running the script again (without the `--reset` flag):
```bash
python train.py
```

## Details

This template is designed for JAX as the backend. In particular, it uses:
- `equinox` for custom PyTree creation and model building
- `optax` for optimization
- `PyTorch` for data loading
- `orbax` for checkpointing

Here is a closer look at the modules inside:

- `model`: Defines callable pytree, whose parameters will be optimized.
- `loss`: 