"""
Copyright 2022 Sahand Rezaei-Shoshtari. All Rights Reserved.
"""
import gym
import mountain_car_3d

env = gym.make('MountainCarContinuous3D-v0', curve_along_y=True)
# env = gym.make('MountainCarContinuous-v0')
env.reset()

for _ in range(1000):
    a = env.action_space.sample()
    s, r, d, _, _ = env.step(a)
    print(f"Action: {a}, State: {s}")
env.close()
