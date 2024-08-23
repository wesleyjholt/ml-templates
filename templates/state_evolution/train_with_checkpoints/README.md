# NN training (with checkpoints) as a state evolution

This tempate shows how to use the state evolution organizational pattern to organize your neural network training code. Additionally, it shows how to "checkpoint" your training, which allows you to pick up training from where you left off in case of a crash or other interruption. 

### Usage

To start a new training session:
```bash
python train.py --reset
```
If training gets interrupted, you can resume from the last checkpoint simply by running the script again (without the `--reset` flag).:
```bash
python train.py
```