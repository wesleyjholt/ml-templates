# This is where the training happens.

# TODO: Add time tracking (save to file, automatically add new computing time).

import os
import shutil
import orbax.checkpoint as ocp
import jax
import jax.numpy as jnp
import equinox as eqx
from time import time

# from .state import TrainState
from .state_factory import state_factory, model_factory, State
from .update import IterData, train_step, increment_epoch, reset_accumulated_loss, reset_batch_counter
from .callback import save_checkpoint, print_loss, write_loss
from .checkpoint import StateSave, StateRestore

# user defines and registers the checkpoint save/restore functions here

# REPLACE WITH YOUR OWN DIRECTORY SETUP FUNCTION!
def setup_io(hyperparams: dict, checkpoint_manager: ocp.CheckpointManager):
    """Creates directories and files necessary for logging and checkpointing."""
    # Clear checkpoint history
    for s in checkpoint_manager.all_steps():
        checkpoint_manager.delete(s)
    ocp.test_utils.erase_and_create_empty('checkpoints')

    # Create loss history file
    loss_history_path = hyperparams['train']['loss_history_path']
    if os.path.exists(loss_history_path):
        os.remove(loss_history_path)
    with open(loss_history_path, 'w') as f:
        f.write('epoch,batch,loss\n')

# REPLACE WITH YOUR OWN STATE INITIALIZATION FUNCTION!
# def init_state(checkpoint_manager: ocp.CheckpointManager, hyperparams: dict = None):
#     """Either creates a new state or restores a previous state."""
#     latest_step = checkpoint_manager.latest_step()
#     if latest_step is None:
#         # Create a new state
#         state = state_factory.generate("state", hyperparams["state"])
#         checkpoint_manager.save(0, args=StateSave(state=state, hyperparams=hyperparams["state"]))
#         checkpoint_manager.wait_until_finished()
#     else:
#         # Load a previous state
#         state = checkpoint_manager.restore(latest_step, args=StateRestore())
#         state = eqx.tree_at(lambda x: x.timestamps.last_restart_time, state, time())
#     state.dataloader.train_iterable.load_state_dict(state.dataloader.train_state_dict)
#     return state

# def init_state(checkpoint_manager: ocp.CheckpointManager, hyperparams: dict = None):
#     """Either creates a new state or restores a previous state."""
#     if checkpoint_manager.latest_step() is None:
#         # Create a new state
#         state = state_factory.generate("state", hyperparams["state"])
#         checkpoint_manager.save(0, args=StateSave(state=state, hyperparams=hyperparams["state"]))
#         checkpoint_manager.wait_until_finished()
#     return load_state(checkpoint_manager)

def save_first_checkpoint(hyperparams: dict, checkpoint_manager: ocp.CheckpointManager):
    """Create and save a new state as the first checkpoint"""
    state = state_factory.generate("state", hyperparams["state"])
    checkpoint_manager.save(0, args=StateSave(state=state, hyperparams=hyperparams["state"]))
    checkpoint_manager.wait_until_finished()

def load_state(checkpoint_manager: ocp.CheckpointManager, step: int = None):
    """Load the state from a checkpoint step. If `step` is None, load the latest state."""
    if step is None:
        step = checkpoint_manager.latest_step()
    state = checkpoint_manager.restore(step, args=StateRestore())
    state = eqx.tree_at(lambda x: x.timestamps.last_restart_time, state, time())
    state.dataloader.train_iterable.load_state_dict(state.dataloader.train_state_dict)
    return state

def load_model(checkpoint_manager: ocp.CheckpointManager, step: int = None):
    """Load the model from a checkpoint step. If `step` is None, load the latest model."""
    state = load_state(checkpoint_manager, step=step)
    return state.model

def save_final_model(state: State, hyperparams: dict):
    """Saves the final state."""
    # eqx.tree_serialise_leaves(hyperparams['train']['final_model_path'], state.model)
    model_factory.save_pytree(state.model, hyperparams['train']['final_model_path'], hyperparams['state']['model_name'], hyperparams['state']['model_hyperparams'])

def load_final_model(hyperparams: dict):
    """Loads the final state."""
    # from model import RNN as Model
    # skeleton = eqx.filter_eval_shape(Model, **hyperparams['state'])
    # return eqx.tree_deserialise_leaves(hyperparams['train']['final_model_path'], skeleton)
    return model_factory.load_pytree(hyperparams['train']['final_model_path'])


# REPLACE WITH YOUR OWN TRAINING FUNCTION!
def run_training(*, hyperparams: dict, reset: bool, checkpoint_manager: ocp.CheckpointManager):
    """Runs the training loop, checkpointing along the way."""

    # Create directories for logging/checkpointing
    if reset:
        setup_io(hyperparams, checkpoint_manager)

    # Create checkpoint manager
    # ckpt_dir = hyperparams['train']['checkpoint_directory']
    # ckpt_options = ocp.CheckpointManagerOptions(max_to_keep=5, enable_async_checkpointing=True)
    # with ocp.CheckpointManager(directory=ckpt_dir, options=ckpt_options) as mngr:

    # Initialize state
    # state = init_state(checkpoint_manager, hyperparams)
    # Check if checkpoints is an empty directory

    if checkpoint_manager.latest_step() is None:
        save_first_checkpoint(hyperparams, checkpoint_manager)
    
    state = load_state(checkpoint_manager)

    # Loop through epochs
    num_epochs = hyperparams['train']['num_epochs']
    save_every = hyperparams['train']['save_every']
    while not (state.dataloader.i_epoch == num_epochs and state.dataloader.i_batch == 0):
        state = increment_epoch(state)

        # Loop through batches
        for i, batch_data in enumerate(state.dataloader.train_iterable):
            batch_data = jax.tree.map(lambda x: jnp.array(x), batch_data)
            dataloader_state_dict = state.dataloader.train_state_dict
            state_no_dataloader_state_dict = eqx.tree_at(lambda x: x.dataloader.train_state_dict, state, 0)  # Remove dataloader from state by replacing it with a dummy "zero" value
            iterdata = IterData(epoch=None, batch=(i, batch_data))
            state_no_dataloader_state_dict = train_step(state_no_dataloader_state_dict, iterdata)
            # state_no_dataloader = train_step(state_no_dataloader)
            state = eqx.tree_at(lambda x: x.dataloader.train_state_dict, state_no_dataloader_state_dict, dataloader_state_dict)
            # train_step(iterdata)
            if state.loss.num_recent == save_every:
                print_loss(state, hyperparams)
                write_loss(state, hyperparams)
                state = reset_accumulated_loss(state)
                save_checkpoint(state, hyperparams, checkpoint_manager)

        state = reset_batch_counter(state)

    save_final_model(state, hyperparams)

if __name__=='__main__':

    import argparse
    from utils import read_yaml

    parser = argparse.ArgumentParser()
    parser.add_argument('--hyperparams', type=str)
    parser.add_argument('--reset', action='store_true')
    args = parser.parse_args()

    hyperparams = read_yaml(args.hyperparams)
    run_training(hyperparams, args.reset)
