import pickle
from os.path import join
import jax

class ParameterManager():
    def __init__(self, log_dir) -> None:
        self._param_path = join(log_dir, "runner_state.pkl")

    def save(self, runner_state):
        flat_params, _ = jax.tree_util.tree_flatten(runner_state)
        with open(self._param_path, 'wb') as f:
            pickle.dump((flat_params), f)
    
    def load(self, init_runner_state):
        # Get the treedef from the init_runner_state
        treedef = jax.tree_util.tree_structure(init_runner_state)

        with open(self._param_path, 'rb') as f:
            flat_params = pickle.load(f)
        runner_state = jax.tree_util.tree_unflatten(treedef, flat_params)
        return runner_state

