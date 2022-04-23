"""
@author: Sahand Rezaei-Shoshtari, Olivier Sigaud

Extension of OpenAI Gym continuous mountain car to 3D.

A merge between two sources:

* Adaptation of the MountainCar Environment from the "FAReinforcement" library
of Jose Antonio Martin H. (version 1.0), adapted by  'Tom Schaul, tom@idsia.ch'
and then modified by Arnaud de Broissia

* the gym MountainCar environment
itself from
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""

import math
from typing import Optional

import numpy as np
import pygame
from pygame import gfxdraw

import gym
from gym import spaces
from gym.utils import seeding


class Continuous_MountainCar3DEnv(gym.Env):
    """
    ### Description

    The Mountain Car MDP is a deterministic MDP that consists of a car placed stochastically
    at the bottom of a sinusoidal valley, with the only possible actions being the accelerations
    that can be applied to the car in either direction. The goal of the MDP is to strategically
    accelerate the car to reach the goal state on top of the right hill. There are two versions
    of the mountain car domain in gym: one with discrete actions and one with continuous.
    This version is the one with continuous actions.

    This MDP first appeared in [Andrew Moore's PhD Thesis (1990)](https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-209.pdf)

    ```
    @TECHREPORT{Moore90efficientmemory-based,
        author = {Andrew William Moore},
        title = {Efficient Memory-based Learning for Robot Control},
        institution = {University of Cambridge},
        year = {1990}
    }
    ```

    ### Observation Space

    The observation is a `ndarray` with shape `(2,)` where the elements correspond to the following:

    | Num | Observation                                                 | Min                | Max    | Unit |
    |-----|-------------------------------------------------------------|--------------------|--------|------|
    | 0   | position of the car along the x-axis                        | -Inf               | Inf    | position (m) |
    | 1   | velocity of the car                                         | -Inf               | Inf  | position (m) |

    ### Action Space

    The action is a `ndarray` with shape `(1,)`, representing the directional force applied on the car. The action is clipped in the range `[-1,1]` and multiplied by a power of 0.0015.

    ### Transition Dynamics:

    Given an action, the mountain car follows the following transition dynamics:

    *velocity<sub>t+1</sub> = velocity<sub>t+1</sub> + force * self.power - 0.0025 * cos(3 * position<sub>t</sub>)*

    *position<sub>t+1</sub> = position<sub>t</sub> + velocity<sub>t+1</sub>*

    where force is the action clipped to the range `[-1,1]` and power is a constant 0.0015. The collisions at either end are inelastic with the velocity set to 0 upon collision with the wall. The position is clipped to the range [-1.2, 0.6] and velocity is clipped to the range [-0.07, 0.07].

    ### Reward

    A negative reward of *-0.1 * action<sup>2</sup>* is received at each timestep to penalise for taking actions of large magnitude. If the mountain car reaches the goal then a positive reward of +100 is added to the negative reward for that timestep.

    ### Starting State

    The position of the car is assigned a uniform random value in `[-0.6 , -0.4]`. The starting velocity of the car is always assigned to 0.

    ### Episode Termination

    The episode terminates if either of the following happens:
    1. The position of the car is greater than or equal to 0.45 (the goal position on top of the right hill)
    2. The length of the episode is 999.

    ### Arguments

    ```
    gym.make('MountainCarContinuous-v0')
    ```

    ### Version History

    * v0: Initial versions release (1.0.0)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, goal_velocity=0, curve_along_y=False):
        self.curve_along_y = curve_along_y
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position_y = -1.2
        self.max_position_y = 1.2
        if curve_along_y:
            self.min_position_x = -1.2
            self.max_position_x = 1.2
            self.goal_position = (
                1
            )
        else:
            self.min_position_x = -1.2
            self.max_position_x = 0.6
            self.goal_position = (
                0.45  # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
            )
        self.max_speed = 0.07
        # Goal is based on the height

        self.goal_height = (
            self._height(self.goal_position, 0)
        )
        self.goal_velocity = goal_velocity
        self.power = 0.0015

        self.low_state = np.array(
            [self.min_position_x, self.min_position_y, -self.max_speed, -self.max_speed],
            dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_position_x, self.max_position_y, self.max_speed, self.max_speed],
            dtype=np.float32
        )

        self.screen = None
        self.clock = None
        self.isopen = True

        self.action_space = spaces.Box(
            low=self.min_action, high=self.max_action, shape=(2,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.low_state, high=self.high_state, dtype=np.float32
        )

    def step(self, action):

        position_x = self.state[0]
        position_y = self.state[1]
        velocity_x = self.state[2]
        velocity_y = self.state[3]
        force_x = min(max(action[0], self.min_action), self.max_action)
        force_y = min(max(action[1], self.min_action), self.max_action)

        # Update x component
        velocity_x += force_x * self.power - 0.0025 * math.cos(3 * position_x)
        if velocity_x > self.max_speed:
            velocity_x = self.max_speed
        if velocity_x < -self.max_speed:
            velocity_x = -self.max_speed
        position_x += velocity_x
        if position_x > self.max_position_x:
            position_x = self.max_position_x
        if position_x < self.min_position_x:
            position_x = self.min_position_x
        if position_x == self.min_position_x and velocity_x < 0:
            velocity_x = 0

        # Update y component (no curve along Y-axis)
        if self.curve_along_y:
            velocity_y += force_y * self.power - 0.0025 * math.cos(3 * position_y)
        else:
            velocity_y += force_y * self.power
        if velocity_y > self.max_speed:
            velocity_y = self.max_speed
        if velocity_y < -self.max_speed:
            velocity_y = -self.max_speed
        position_y += velocity_y
        if position_y > self.max_position_y:
            position_y = self.max_position_y
        if position_y < self.min_position_y:
            position_y = self.min_position_y
        if position_y == self.min_position_y and velocity_y < 0:
            velocity_y = 0
        if position_y == self.max_position_y and velocity_y > 0:
            velocity_y = 0

        # Convert a possible numpy bool to a Python bool.
        # Goal is defined based on the height
        done = bool(self._height(position_x, position_y) >= self.goal_height and
                    velocity_x >= self.goal_velocity and
                    velocity_y >= self.goal_velocity)

        reward = 0
        if done:
            reward = 100.0
        reward -= math.pow(action[0], 2) * 0.1

        self.state = np.array([position_x, position_y, velocity_x, velocity_y], dtype=np.float32)
        return self.state, reward, done, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None
    ):
        super().reset(seed=seed)
        self.state = np.array([
            self.np_random.uniform(low=-0.6, high=-0.4),
            self.np_random.uniform(low=-0.6, high=-0.4),
            0,
            0,
        ])
        if not return_info:
            return np.array(self.state, dtype=np.float32)
        else:
            return np.array(self.state, dtype=np.float32), {}

    def _height(self, xs, ys):
        if self.curve_along_y:
            return np.sin(3 * np.sqrt(xs**2 + ys**2) - math.pi / 2) * 0.45 + 0.55
        else:
            return np.sin(3 * xs) * 0.45 + 0.55

    def render(self, mode="human"):
        """
        The rendering is the original 2D render of OpenAI Gym.
        """
        screen_width = 600
        screen_height = 400

        world_width = self.max_position_x - self.min_position_x
        scale = screen_width / world_width
        carwidth = 40
        carheight = 20
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((screen_width, screen_height))
        self.surf.fill((255, 255, 255))

        pos = self.state[0]

        xs = np.linspace(self.min_position_x, self.max_position_x, 100)
        ys = self._height(xs)
        xys = list(zip((xs - self.min_position_x) * scale, ys * scale))

        pygame.draw.aalines(self.surf, points=xys, closed=False, color=(0, 0, 0))

        clearance = 10

        l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
        coords = []
        for c in [(l, b), (l, t), (r, t), (r, b)]:
            c = pygame.math.Vector2(c).rotate_rad(math.cos(3 * pos))
            coords.append(
                (
                    c[0] + (pos - self.min_position_x) * scale,
                    c[1] + clearance + self._height(pos) * scale,
                )
            )

        gfxdraw.aapolygon(self.surf, coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, coords, (0, 0, 0))

        for c in [(carwidth / 4, 0), (-carwidth / 4, 0)]:
            c = pygame.math.Vector2(c).rotate_rad(math.cos(3 * pos))
            wheel = (
                int(c[0] + (pos - self.min_position_x) * scale),
                int(c[1] + clearance + self._height(pos) * scale),
            )

            gfxdraw.aacircle(
                self.surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128)
            )
            gfxdraw.filled_circle(
                self.surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128)
            )

        flagx = int((self.goal_position - self.min_position_x) * scale)
        flagy1 = int(self.goal_height * scale)
        flagy2 = flagy1 + 50
        gfxdraw.vline(self.surf, flagx, flagy1, flagy2, (0, 0, 0))

        gfxdraw.aapolygon(
            self.surf,
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
            (204, 204, 0),
        )
        gfxdraw.filled_polygon(
            self.surf,
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
            (204, 204, 0),
        )

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        if mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        else:
            return self.isopen

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.isopen = False