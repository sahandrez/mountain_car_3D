"""
Copyright 2022 Sahand Rezaei-Shoshtari. All Rights Reserved.
"""
import gym
import mountain_car_3d

env = gym.make('MountainCarContinuous3D-v0')
# env = gym.make('MountainCar3D-v0')
env.reset()

for _ in range(999):
    a = env.action_space.sample()
    s, r, d, _ = env.step(a)
    print(s)
env.close()
