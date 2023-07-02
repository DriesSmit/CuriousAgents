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
import abc

import chex
import jax
import jax.numpy as jnp

from curious_agents.environments.minecraft2d.types import Position, State
from curious_agents.environments.minecraft2d.constants import AIR, STEVE, WOODEN_LOG, COBBLESTONE, IRON_ORE, DIAMOND_ORE


class Generator(abc.ABC):
    def __init__(self, num_rows: int, num_cols: int):
        """Interface for Minecraft world generator.

        Args:
            num_rows: the width of the Minecraft world to create.
            num_cols: the length of the Minecraft world to create.
        """
        self.num_rows = num_rows
        self.num_cols = num_cols

    @abc.abstractmethod
    def __call__(self, key: chex.PRNGKey) -> chex.Array:
        """Generate a problem instance.

        Args:
            key: random key.

        Returns:
            state: the generated state.
        """

class RandomGenerator(Generator):
    def __init__(self, num_rows: int, num_cols: int) -> None:
        super(RandomGenerator, self).__init__(num_rows=num_rows, num_cols=num_cols)

    def __call__(self, key: chex.PRNGKey) -> State:
        """Generate a random Minecraft world.
        """

        # Generate the map. Place one WOODEN_LOG, COBBLESTONE, IRON_ORE and DIAMOND_ORE on the map.
        map = jnp.ones((self.num_rows, self.num_cols), dtype=jnp.int32) * AIR
        # TODO: Speed this up by selectin 4 values at once and setting them at the same time.
        for i in [WOODEN_LOG, COBBLESTONE, IRON_ORE, DIAMOND_ORE]:
            key, map_key = jax.random.split(key)
            rand_index = jax.random.choice(
                            map_key,
                            jnp.arange(self.num_rows * self.num_cols),
                            replace=False,
                            p=map.flatten()==AIR,
                        )
            (rand_x, rand_y) = jnp.divmod(
                                    rand_index, self.num_cols
                                )

            map = map.at[rand_x, rand_y].set(i)

        # Randomise agent start and target positions.
        key, agent_key = jax.random.split(key)
        start_index = jax.random.choice(
            agent_key,
            jnp.arange(self.num_rows * self.num_cols),
            replace=False,
            p=map.flatten()==AIR,
        )
        (agent_row, agent_col) = jnp.divmod(
            start_index, self.num_cols
        )
        agent_position = Position(row=agent_row, col=agent_col)
        map = map.at[agent_row, agent_col].set(STEVE)

        # Build the state.
        return State(
            agent_position=agent_position,
            agent_level=jnp.array(1, jnp.int32),
            map=map,
            action_mask=None,
            key=key,
            step_count=jnp.array(0, jnp.int32),
        )
