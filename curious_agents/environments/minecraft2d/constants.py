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

import jax.numpy as jnp

# Actions
MOVES = jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1]] + [[0, 0]]*7)  # Up, Right, Down, Left

# Minecraft map blocks
AIR = 0
STEVE = 1
WOODEN_LOG = 2
COBBLESTONE = 3
IRON_ORE = 4
DIAMOND_ORE = 5

# Minecraft map blocks
STEVE_LEVEL = 0
WOODEN_LOG_LEVEL = 1
WOODEN_PLANK_LEVEL = 2
WOODEN_STICK_LEVEL = 3
WOODEN_PICKAXE_LEVEL = 4
COBBLESTONE_LEVEL = 5
STONE_PICKAXE_LEVEL = 6
IRON_ORE_LEVEL = 7
IRON_IGNOT_LEVEL = 8
IRON_PICKAXE_LEVEL = 9
DIAMOND_ORE_LEVEL = 10
DIAMOND_PICKAXE_LEVEL = 11

# Minecraft block to level
BLOCK_TO_LEVEL = jnp.array([0, 0, WOODEN_LOG_LEVEL, COBBLESTONE_LEVEL, IRON_ORE_LEVEL, DIAMOND_ORE_LEVEL])

# Action to level
ACTION_TO_LEVEL = jnp.array([0, 0, 0, 0, WOODEN_PLANK_LEVEL, WOODEN_STICK_LEVEL, WOODEN_PICKAXE_LEVEL,
                             STONE_PICKAXE_LEVEL, IRON_IGNOT_LEVEL, IRON_PICKAXE_LEVEL, DIAMOND_PICKAXE_LEVEL])





