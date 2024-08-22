import equinox as eqx
import orbax.checkpoint as ocp

from state import TrainState
from checkpointing import EquinoxSave

def save_checkpoint(state: TrainState, mngr: ocp.CheckpointManager, hyperparams: dict):
    state = eqx.tree_at(lambda x: x.dataloader.state_dict, state, state.dataloader.iterable.state_dict())
    cumulative_iter = (state.dataloader.i_epoch - 1) * len(state.dataloader.iterable) + state.dataloader.i_batch
    mngr.save(cumulative_iter, args=EquinoxSave(state=state, hyperparams=hyperparams))
    mngr.wait_until_finished()

def print_loss(state: TrainState, hyperparams: dict):
    num_epochs = hyperparams['train']['num_epochs']
    print(f'Epoch {state.dataloader.i_epoch}/{num_epochs}', end='\t')
    print(f'Batch {state.dataloader.i_batch}/{len(state.dataloader.iterable)}', end='\t')
    print(f'Loss: {state.loss.recent_accumulated_loss / state.loss.num_recent :.4f}')

def write_loss(state: TrainState, hyperparams: dict):
    path = hyperparams['train']['loss_history_path']
    epoch = state.dataloader.i_epoch
    batch = state.dataloader.i_batch
    loss = state.loss.recent_accumulated_loss / state.loss.num_recent
    with open(path, 'a') as f:
        f.write(f'{epoch},{batch},{loss}\n')