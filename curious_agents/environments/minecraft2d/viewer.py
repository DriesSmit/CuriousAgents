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

from typing import Optional, Sequence, Callable, Tuple

import chex
import matplotlib
import matplotlib.pyplot as plt
from numpy.typing import NDArray

from curious_agents.environments.minecraft2d.types import State
from curious_agents.environments.minecraft2d.constants import AIR, STEVE, WOODEN_LOG, COBBLESTONE, IRON_ORE, DIAMOND_ORE
from jumanji.viewer import Viewer
import jumanji
import numpy as np
from matplotlib.axes import Axes
from matplotlib import image
from scipy.ndimage import zoom

class Minecraft2DEnvViewer(Viewer):
    FONT_STYLE = "monospace"
    FIGURE_SIZE = (10.0, 10.0)

    def __init__(self, name: str, render_mode: str = "human", resolution=16) -> None:
        """Viewer for a Minecraft environment.

        Args:
            name: the window name to be used when initialising the window.
            render_mode: the mode used to render the environment. Must be one of:
                - "human": render the environment on screen.
                - "rgb_array": return a numpy array frame representing the environment.
        """
        self._name = name
        # The animation must be stored in a variable that lives as long as the
        # animation should run. Otherwise, the animation will get garbage-collected.
        self._animation: Optional[matplotlib.animation.Animation] = None

        self._display: Callable[[plt.Figure], Optional[NDArray]]
        if render_mode == "rgb_array":
            self._display = self._display_rgb_array
        elif render_mode == "human":
            self._display = self._display_human
        else:
            raise ValueError(f"Invalid render mode: {render_mode}")
        
        self._resolution = resolution

        # Equipent labels
        self._equipment_labels = [None, None, 'wooden_plank', 'stick', 'wooden_pickaxe', 'wooden_pickaxe', 
                                    'stone_pickaxe', 'stone_pickaxe', 'iron_ingot', 'iron_pickaxe', 'diamond',
                                    'diamond_pickaxe']

        # Load the images with the correct resolution
        img_path = "curious_agents/environments/minecraft2d/images"
        self._images = self._load_and_resize_images(img_path, resolution)

    def _load_and_resize_image(self, img_path, label, resolution):
        # Load image with matplotlib's image.imread()
        img_np = image.imread(f"{img_path}/{label}.png")

        # Calculate the resize factor
        resize_factor = np.array((resolution / img_np.shape[0], resolution / img_np.shape[1]))

        # Resize the image
        resized_img_np = zoom(img_np, (resize_factor[0], resize_factor[1], 1))

        # Clip values to be within the valid range
        return np.clip(resized_img_np, 0, 1)

    def _load_and_resize_images(self, img_path, resolution):
        img_indices = [STEVE, WOODEN_LOG, COBBLESTONE, IRON_ORE, DIAMOND_ORE]
        img_labels = ['steve', 'wooden_log', 'cobblestone', 'iron_ore', 'diamond_ore']

        # Load and resize images
        images = {}
        for index, label in zip(img_indices, img_labels):
            # Add the resized image to the dictionary
            images[index] = self._load_and_resize_image(img_path, label, resolution)

        # Load and resize images
        for label in set(self._equipment_labels):
            if label is not None:
                # Add the resized image to the dictionary
                images[label] = self._load_and_resize_image(img_path, label, resolution)   

        return images


    def render(self, state: chex.Array) -> Optional[NDArray]:
        """
        Render Minecraft state.

        Args:
            state: the state to render.

        Returns:
            RGB array if the render_mode is RenderMode.RGB_ARRAY.
        """
        self._clear_display()
        fig, ax = self._get_fig_ax()
        ax.clear()

        # TODO: Make this generic
        agent_position = [(state.agent_position.col - 0.1) / 5, 1.0 - (state.agent_position.row+1.1) / 5]

        self._add_grid_image(state.map, state.agent_level, agent_position, ax)
        return self._display(fig)

    def animate(
        self,
        states: Sequence[chex.Array],
        interval: 200,
        save_path: Optional[str] = None,
    ) -> matplotlib.animation.FuncAnimation:
        """Create an animation from a sequence of states.

        Args:
            states: sequence of `Minecraft states` corresponding to consecutive timesteps.
            interval: delay between frames in milliseconds, default to 200.
            save_path: the path where the animation file should be saved. If it is None, the plot
                will not be saved.

        Returns:
            Animation that can be saved as a GIF, MP4, or rendered with HTML.
        """
        fig, ax = plt.subplots(num=f"{self._name}Animation", figsize=self.FIGURE_SIZE)
        plt.close(fig)

        def make_frame(state_index: int) -> None:
            ax.clear()
            state = states[state_index]
            map = state.map
            agent_level = state.agent_level

            # TODO: Make this generic
            agent_position = [(state.agent_position.col - 0.1) / 5, 1.0 - (state.agent_position.row+1.1) / 5]

            self._add_grid_image(map, agent_level, agent_position, ax)

        # Create the animation object.
        self._animation = matplotlib.animation.FuncAnimation(
            fig,
            make_frame,
            frames=len(states),
            interval=interval,
        )

        # Save the animation as a gif.
        if save_path:
            self._animation.save(save_path)

        return self._animation

    def close(self) -> None:
        plt.close(self._name)

    def _get_fig_ax(self) -> Tuple[plt.Figure, plt.Axes]:
        recreate = not plt.fignum_exists(self._name)
        fig = plt.figure(self._name, figsize=self.FIGURE_SIZE)
        if recreate:
            if not plt.isinteractive():
                fig.show()
            ax = fig.add_subplot()
        else:
            ax = fig.get_axes()[0]
        return fig, ax

    def _add_grid_image(self, map: chex.Array, agent_level: chex.Array, agent_pos: chex.Array, ax: Axes) -> image.AxesImage:
        img = self._create_grid_image(map)

        # Write the agent level on the image
        ax.text(0, -1, f"Episode return: {agent_level}", fontsize=20, color='black', fontname=self.FONT_STYLE)
        ax.set_axis_off()

        # Draw a image on the top left corner of the image
        level_label = self._equipment_labels[agent_level]
        if level_label is not None:


            # Create new Axes at specific location
            inset_ax = ax.inset_axes([agent_pos[0], agent_pos[1], 0.1, 0.1])

            # Draw the image on the new Axes
            inset_ax.imshow(self._images[level_label])
            inset_ax.axis('off')  # Hide the axes for this inset plot

        return ax.imshow(img)


    def _create_grid_image(self, map: chex.Array) -> NDArray:
        length = len(map)
        width = len(map[0])
        res = self._resolution
        img = np.ones((length*res, width*res, 4))
        
        # TODO: Draw blocks in the map
        for i in range(length):
            for j in range(width):
                if map[i][j] != AIR:
                    img[i*res:(i+1)*res, j*res:(j+1)*res] = self._images[int(map[i][j])]
        return img

    def _display_human(self, fig: plt.Figure) -> None:
        if plt.isinteractive():
            # Required to update render when using Jupyter Notebook.
            fig.canvas.draw()
            if jumanji.environments.is_colab():
                plt.show(self._name)
        else:
            # Required to update render when not using Jupyter Notebook.
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

    def _display_rgb_array(self, fig: plt.Figure) -> NDArray:
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())

