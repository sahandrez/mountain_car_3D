"""
Copyright 2022 Sahand Rezaei-Shoshtari. All Rights Reserved.
"""
from setuptools import setup
import sys

assert sys.version_info.major == 3 and sys.version_info.minor >= 6, \
    "The moutain-car-3D package is designed to work with Python 3.6 " \
    "and greater Please install it before proceeding."

setup(
    name='mountain_car_3d',
    py_modules=['mountain_car_3d'],
    install_requires=[
        'gym==0.23.1',
        'numpy',
        'opencv-python',
    ],
    description="Mountain Car 3D Gym environment.",
    author="Sahand Rezaei-Shoshtari",
)
