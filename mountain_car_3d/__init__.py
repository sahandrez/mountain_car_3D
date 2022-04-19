"""
Copyright 2022 Sahand Rezaei-Shoshtari. All Rights Reserved.
"""
from gym.envs.registration import register

register(
    id='MountainCarContinuous3D-v0',
    entry_point='mountain_car_3d.envs:Continuous_MountainCar3DEnv',
    max_episode_steps=1000,
    reward_threshold=90.0,
)
