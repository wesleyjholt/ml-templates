import os
from dataclasses import dataclass
import equinox as eqx
import orbax.checkpoint as ocp
from typing import Any

from .utils import read_json, write_json
from ...pytree_factory import factory
from .state_factory import state_factory

class StateCheckpointHandler(ocp.CheckpointHandler):
    def save(self, directory: str, args: 'StateSave'):
        state_factory.save_pytree(args.state, os.path.join(directory, 'state.eqx'), "state", args.hyperparams)
    
    def restore(self, directory: str, args: 'StateRestore') -> Any:
        return state_factory.load_pytree(os.path.join(directory, 'state.eqx'))
    
    def metadata(self, directory: str) -> Any:
        hyperparams = read_json(os.path.join(directory, 'hyperparams.json'))
        return dict(hyperparams=hyperparams)

@ocp.args.register_with_handler(StateCheckpointHandler, for_save=True)
@dataclass
class StateSave(ocp.args.CheckpointArgs):
    state: Any
    hyperparams: dict

@ocp.args.register_with_handler(StateCheckpointHandler, for_restore=True)
@dataclass
class StateRestore(ocp.args.CheckpointArgs):
    pass


# if __name__=='__main__':

#     # Test
#     from state import TrainState

#     hyperparams = dict(
#         seed=0,
#         in_size=2,
#         out_size=1,
#         hidden_size=16,
#         learning_rate=3e-3,
#         dataset_size=10000,
#         batch_size=32
#     )

#     state = TrainState(**hyperparams)
#     original_state = state

#     checkpoints_dir = 'checkpoints'
#     ocp.test_utils.erase_and_create_empty(checkpoints_dir)
#     options = ocp.CheckpointManagerOptions(enable_async_checkpointing=False)

#     with ocp.CheckpointManager(
#         directory=checkpoints_dir,
#         options=options
#     ) as mngr:
#         last_step = mngr.latest_step()
#         if last_step is None:
#             # Create a new state
#             state = TrainState(**hyperparams)
#             mngr.save(0, args=EquinoxSave(state=state, hyperparams=hyperparams))
#             mngr.wait_until_finished()
#         else:
#             # Load a previous state
#             skeleton = TrainState(**hyperparams, make_skeleton=True)
#             state = mngr.restore(last_step, args=EquinoxRestore(skeleton=skeleton))

#     assert eqx.filter(original_state, eqx.is_array_like) == eqx.filter(state, eqx.is_array_like), 'Restored state does not match original state'