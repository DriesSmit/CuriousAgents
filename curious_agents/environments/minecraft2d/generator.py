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
        things = jnp.array([STEVE, WOODEN_LOG, COBBLESTONE, IRON_ORE, DIAMOND_ORE])

        indices = jax.random.choice(
                            key,
                            jnp.arange(self.num_rows * self.num_cols),
                            shape=(len(things),),
                            replace=False,
                        )
        
        (rand_x, rand_y) = jnp.divmod(
                                indices, self.num_cols
                            )
        
        map = map.at[rand_x, rand_y].set(things)
        agent_position = Position(row=rand_x[0], col=rand_y[0])

        # Build the state.
        return State(
            agent_position=agent_position,
            agent_level=jnp.array(0, jnp.int32),
            map=map,
            action_mask=None,
            key=key,
            level_step_count=jnp.array(0, jnp.int32),
        )
