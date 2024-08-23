# REPLACE THESE WITH YOUR OWN CALLBACK FUNCTIONS!
# (These are functions which do not modify the state.)

import equinox as eqx
import orbax.checkpoint as ocp

from state import TrainState
from checkpointing import EquinoxSave
from utils import compute_elapsed_time

def save_checkpoint(state: TrainState, hyperparams: dict, mngr: ocp.CheckpointManager):
    state = eqx.tree_at(lambda x: x.dataloader.state_dict, state, state.dataloader.iterable.state_dict())
    cumulative_iter = (state.dataloader.i_epoch - 1) * len(state.dataloader.iterable) + state.dataloader.i_batch
    mngr.save(cumulative_iter, args=EquinoxSave(state=state, hyperparams=hyperparams))
    mngr.wait_until_finished()

def print_loss(state: TrainState, hyperparams: dict):
    num_epochs = hyperparams['train']['num_epochs']
    total_time = compute_elapsed_time(state.timestamps.first_start_time)
    time_since_last_restart = compute_elapsed_time(state.timestamps.last_restart_time)
    print(f'Epoch {state.dataloader.i_epoch}/{num_epochs}', end='\t')
    print(f'Batch {state.dataloader.i_batch}/{len(state.dataloader.iterable)}', end='\t')
    print(f'Loss: {state.loss.recent_accumulated_loss / state.loss.num_recent :.5f}', end='\t')
    print(f'Total run time: {total_time}', end='\t')
    print(f'Time since last restart: {time_since_last_restart}')

def write_loss(state: TrainState, hyperparams: dict):
    path = hyperparams['train']['loss_history_path']
    epoch = state.dataloader.i_epoch
    batch = state.dataloader.i_batch
    loss = state.loss.recent_accumulated_loss / state.loss.num_recent
    with open(path, 'a') as f:
        f.write(f'{epoch},{batch},{loss}\n')