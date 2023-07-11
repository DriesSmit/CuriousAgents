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
from curious_agents.environments.minecraft2d.constants import MOVES, ACTION_TO_LEVEL, BLOCK_TO_LEVEL, AIR, STEVE, DIAMOND_ORE, DIAMOND_PICKAXE_LEVEL
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
        time_limit_per_task: Optional[int] = None,
        viewer: Optional[Viewer[State]] = None,
    ) -> None:
       
        self.generator = generator or RandomGenerator(num_rows=5, num_cols=5)
        self.num_rows = self.generator.num_rows
        self.num_cols = self.generator.num_cols
        self.shape = (self.num_rows, self.num_cols)

        # TODO: Fix this.
        self.time_limit_per_task = time_limit_per_task or self.num_rows + self.num_cols
        self.max_level = DIAMOND_PICKAXE_LEVEL

        # Create viewer used for rendering
        self._viewer = viewer or Minecraft2DEnvViewer("Minecraft2D", render_mode="human")

    def __repr__(self) -> str:
        return "\n".join(
            [
                "Minecraft2D environment:",
                f" - num_rows: {self.num_rows}",
                f" - num_cols: {self.num_cols}",
                f" - time_limit_per_task: {self.time_limit_per_task}",
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
        level_step_count = specs.Array((), jnp.int32, "level_step_count")
        action_mask = specs.BoundedArray(
            shape=(4+7,), dtype=bool, minimum=False, maximum=True, name="action_mask"
        )
        return specs.Spec(
            Observation,
            "ObservationSpec",
            map=map,
            level_step_count=level_step_count,
            action_mask=action_mask,
        )

    def action_spec(self) -> specs.DiscreteArray:
     
        return specs.DiscreteArray(4+7, name="action")

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        
        # Generate the environment state
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
      
        # If the chosen action is invalid, overwrite it to no-op.
        move_action = jax.lax.select(state.action_mask[action] & (action < 4), action, 4)

        # Take the action in the environment:  up, right, down, or left
        # Remember the map coordinates: (0,0) is top left.
        agent_position = jax.lax.switch(
            move_action,
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
        move_level_up = jnp.array(state.map[agent_position.row, agent_position.col] > STEVE, int)

        # Update the map.
        map = state.map.at[state.agent_position.row, state.agent_position.col].set(AIR)
        map = map.at[agent_position.row, agent_position.col].set(STEVE)

        # Builded something
        build_level_up = state.action_mask[action] & (action > 3)
        
        level_up = move_level_up + build_level_up
        agent_level = state.agent_level + level_up

        # Compute the reward.
        reward = level_up

        

        # Update the task step count.
        # Reset the task step count if the agent has moved to a new level.
        # Otherwise, increment the task step count.
        level_step_count = (state.level_step_count + 1)*(1-level_up)

        # Generate action mask to keep in the state for the next step and
        # to provide to the agent in the observation.
        action_mask = self._compute_action_mask(state.map, agent_position, agent_level)
        
        # Build the state.
        state = State(
            agent_position=agent_position,
            agent_level=agent_level,
            map=map,
            action_mask=action_mask,
            key=state.key,
            level_step_count=level_step_count,
        )
        # Generate the observation from the environment state.
        observation = self._observation_from_state(state)

        # Check if the episode terminates (i.e. done is True).
        at_max_level = state.agent_level == DIAMOND_PICKAXE_LEVEL
        time_limit_exceeded = state.level_step_count >= self.time_limit_per_task
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

        def is_action_valid(agent_position: Position, agent_level: chex.Array, action: chex.Array) -> chex.Array:
            x, y = jnp.array([agent_position.row, agent_position.col]) + MOVES[action]
            mask = (
                (
                (action < 4)
                & (x >= 0)
                & (x < self.num_cols)
                & (y >= 0)
                & (y < self.num_rows)
                & (BLOCK_TO_LEVEL[map[x, y]] <= agent_level+1)
                )| 
                (
                (action > 3) 
                & (ACTION_TO_LEVEL[action] == agent_level+1)
                 )
            )
            return mask

        # vmap over the actions.
        action_mask = jax.vmap(is_action_valid, in_axes=(None, None, 0))(agent_position, agent_level, jnp.arange(4+7))
        return action_mask

    def _observation_from_state(self, state: State) -> Observation:
        """Create an observation from the state of the environment."""
        return Observation(
            map=state.map,
            level_step_count=state.level_step_count,
            agent_level=state.agent_level,
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
