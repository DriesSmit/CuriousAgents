# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp
import matplotlib.animation
from numpy.typing import NDArray

from jumanji import specs
from jumanji.env import Environment
from curious_agents.environments.minecraft2d.constants import MOVES, AIR, STEVE, DIAMOND_ORE
from curious_agents.environments.minecraft2d.generator import Generator, RandomGenerator
from curious_agents.environments.minecraft2d.types import Observation, Position, State
from curious_agents.environments.minecraft2d.viewer import Minecraft2DEnvViewer
from jumanji.types import TimeStep, restart, termination, transition
from jumanji.viewer import Viewer


class Minecraft2D(Environment[State]):
    """A 2D Minecraft environment.
    """

    FIGURE_NAME = "Minecraft2D"
    FIGURE_SIZE = (6.0, 6.0)

    def __init__(
        self,
        generator: Optional[Generator] = None,
        time_limit: Optional[int] = None,
        viewer: Optional[Viewer[State]] = None,
    ) -> None:
       
        self.generator = generator or RandomGenerator(num_rows=10, num_cols=10)
        self.num_rows = self.generator.num_rows
        self.num_cols = self.generator.num_cols
        self.shape = (self.num_rows, self.num_cols)
        self.time_limit = time_limit or 4 * (self.num_rows + self.num_cols)

        # Create viewer used for rendering
        self._viewer = viewer or Minecraft2DEnvViewer("Minecraft2D", render_mode="human")

    def __repr__(self) -> str:
        return "\n".join(
            [
                "Minecraft2D environment:",
                f" - num_rows: {self.num_rows}",
                f" - num_cols: {self.num_cols}",
                f" - time_limit: {self.time_limit}",
                f" - generator: {self.generator}",
            ]
        )

    def observation_spec(self) -> specs.Spec[Observation]:
       
        map = specs.BoundedArray(
            shape=(self.num_rows, self.num_cols),
            dtype=bool,
            minimum=0,
            maximum=DIAMOND_ORE,
            name="map",
        )
        step_count = specs.Array((), jnp.int32, "step_count")
        action_mask = specs.BoundedArray(
            shape=(4,), dtype=bool, minimum=False, maximum=True, name="action_mask"
        )
        return specs.Spec(
            Observation,
            "ObservationSpec",
            map=map,
            step_count=step_count,
            action_mask=action_mask,
        )

    def action_spec(self) -> specs.DiscreteArray:
     
        return specs.DiscreteArray(4, name="action")

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        
         # Fix the rng_key
        key = jax.random.PRNGKey(0)
        state = self.generator(key)

        # Create the action mask and update the state
        state.action_mask = self._compute_action_mask(state.map, state.agent_position, state.agent_level)

        # Generate the observation from the environment state.
        observation = self._observation_from_state(state)

        # Return a restart timestep whose step type is FIRST.
        timestep = restart(observation)

        return state, timestep

    def step(
        self, state: State, action: chex.Array
    ) -> Tuple[State, TimeStep[Observation]]:
      
        # If the chosen action is invalid, i.e. blocked by a wall, overwrite it to no-op.
        action = jax.lax.select(state.action_mask[action], action, 4)

        # Take the action in the environment:  up, right, down, or left
        # Remember the map coordinates: (0,0) is top left.
        agent_position = jax.lax.switch(
            action,
            [
                lambda position: Position(position.row - 1, position.col),  # Up
                lambda position: Position(position.row, position.col + 1),  # Right
                lambda position: Position(position.row + 1, position.col),  # Down
                lambda position: Position(position.row, position.col - 1),  # Left
                lambda position: position,  # No-op
            ],
            state.agent_position,
        )

        # Check if the agent has moved to a new level.
        level_up = jnp.array(state.map[agent_position.row, agent_position.col] > STEVE, int)
        agent_level = state.agent_level + level_up

        # Compute the reward.
        reward = level_up

        # Generate action mask to keep in the state for the next step and
        # to provide to the agent in the observation.
        action_mask = self._compute_action_mask(state.map, agent_position, agent_level)

        # Update the map.
        map = state.map.at[state.agent_position.row, state.agent_position.col].set(AIR)
        map = map.at[agent_position.row, agent_position.col].set(STEVE)
        
        # Build the state.
        state = State(
            agent_position=agent_position,
            agent_level=agent_level,
            map=map,
            action_mask=action_mask,
            key=state.key,
            step_count=state.step_count + 1,
        )
        # Generate the observation from the environment state.
        observation = self._observation_from_state(state)

        # Check if the episode terminates (i.e. done is True).
        at_max_level = state.agent_level == DIAMOND_ORE
        time_limit_exceeded = state.step_count >= self.time_limit
        no_actions_available = ~jnp.any(state.action_mask)

        done = at_max_level | time_limit_exceeded | no_actions_available

        # Return either a MID or a LAST timestep depending on done.
        timestep = jax.lax.cond(
            done,
            termination,
            transition,
            reward,
            observation,
        )
        return state, timestep

    def _compute_action_mask(
        self, map: chex.Array, agent_position: Position, agent_level: chex.Array,
    ) -> chex.Array:

        def is_move_valid(agent_position: Position, agent_level: chex.Array, move: chex.Array) -> chex.Array:
            x, y = jnp.array([agent_position.row, agent_position.col]) + move
            mask = (
                (x >= 0)
                & (x < self.num_cols)
                & (y >= 0)
                & (y < self.num_rows)
                & (map[x, y] <= agent_level+1)
            )
            return mask

        # vmap over the moves.
        action_mask = jax.vmap(is_move_valid, in_axes=(None, None, 0))(agent_position, agent_level, MOVES)
        return action_mask

    def _observation_from_state(self, state: State) -> Observation:
        """Create an observation from the state of the environment."""
        return Observation(
            map=state.map,
            step_count=state.step_count,
            action_mask=state.action_mask,
        )

    def render(self, state: State) -> Optional[NDArray]:
        """Render the given state of the environment.

        Args:
            state: `State` object containing the current environment state.
        """
        return self._viewer.render(state)

    def animate(
        self,
        states: Sequence[State],
        interval: int = 200,
        save_path: Optional[str] = None,
    ) -> matplotlib.animation.FuncAnimation:
        """Creates an animated gif of the `Minecraft2D` environment based on the sequence of states.

        Args:
            states: sequence of environment states corresponding to consecutive timesteps.
            interval: delay between frames in milliseconds, default to 200.
            save_path: the path where the animation file should be saved. If it is None, the plot
                will not be saved.

        Returns:
            animation.FuncAnimation: the animation object that was created.
        """
        return self._viewer.animate(states, interval, save_path)

    def close(self) -> None:
        """Perform any necessary cleanup.

        Environments will automatically :meth:`close()` themselves when
        garbage collected or when the program exits.
        """
        self._viewer.close()
