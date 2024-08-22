import os
import orbax.checkpoint as ocp
import equinox as eqx

from state import TrainState
from update import IterData, train_step, increment_epoch, reset_accumulated_loss, reset_batch_counter
from callback import save_checkpoint, print_loss, write_loss
from checkpointing import EquinoxSave, EquinoxRestore

def setup_io(hyperparams: dict):
    """Creates directories and files necessary for logging and checkpointing."""
    loss_history_path = hyperparams['train']['loss_history_path']
    ocp.test_utils.erase_and_create_empty('checkpoints')
    if os.path.exists(loss_history_path):
        os.remove(loss_history_path)
    with open(loss_history_path, 'w') as f:
        f.write('epoch,batch,loss\n')

def init_state(hyperparams: dict, checkpoint_manager: ocp.CheckpointManager):
    """Either creates a new state or restores a previous state."""
    last_step = checkpoint_manager.latest_step()
    if last_step is None:
        # Create a new state
        state = TrainState(**hyperparams['state'])
        checkpoint_manager.save(0, args=EquinoxSave(state=state, hyperparams=hyperparams))
        checkpoint_manager.wait_until_finished()
    else:
        # Load a previous state
        skeleton = TrainState(**hyperparams['state'], make_skeleton=False)
        state = checkpoint_manager.restore(last_step, args=EquinoxRestore(skeleton=skeleton))
    state.dataloader.iterable.load_state_dict(state.dataloader.state_dict)
    return state

def save_final_model(state: TrainState, hyperparams: dict):
    """Saves the final state."""
    eqx.tree_serialise_leaves(hyperparams['train']['final_model_path'], state.model)

def load_final_model(hyperparams: dict):
    """Loads the final state."""
    from model import RNN
    skeleton = eqx.filter_eval_shape(RNN, **hyperparams['state'])
    return eqx.tree_deserialise_leaves(hyperparams['train']['final_model_path'], skeleton)

def train(hyperparams: dict, reset: bool):
    """Runs the training loop, checkpointing along the way."""

    # Create directories for logging/checkpointing
    if reset:
        setup_io(hyperparams)

    # Create checkpoint manager
    ckpt_dir = hyperparams['train']['checkpoint_directory']
    ckpt_options = ocp.CheckpointManagerOptions(max_to_keep=5, enable_async_checkpointing=True)
    with ocp.CheckpointManager(directory=ckpt_dir, options=ckpt_options) as mngr:

        # Initialize state
        state = init_state(hyperparams, mngr)

        # Loop through epochs
        num_epochs = hyperparams['train']['num_epochs']
        save_every = hyperparams['train']['save_every']
        while not (state.dataloader.i_epoch == num_epochs and state.dataloader.i_batch == 0):
            state = increment_epoch(state)

            # Loop through batches
            for batch_data in enumerate(state.dataloader.iterable):
                iterdata = IterData(epoch=None, batch=batch_data)
                state = train_step(state, iterdata)
                if state.loss.num_recent == save_every:
                    print_loss(state, hyperparams)
                    write_loss(state, hyperparams)
                    state = reset_accumulated_loss(state)
                    save_checkpoint(state, mngr, hyperparams)

            state = reset_batch_counter(state)

    save_final_model(state, hyperparams)

if __name__=='__main__':

    import argparse
    from utils import read_yaml

    parser = argparse.ArgumentParser()
    parser.add_argument('--hyperparams', type=str, default='hyperparams.yml')
    parser.add_argument('--reset', action='store_true')
    args = parser.parse_args()

    hyperparams = read_yaml(args.hyperparams)
    train(hyperparams, args.reset)
