"""Installation script for the 'isaacgymenvs' python package."""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from setuptools import setup, find_packages

import os


# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "torch==2.0.1",
    "stable-baselines3==2.0.0a8",
    "gymnasium==0.28.1",
    "numpy==1.23.5"
]

# Installation operation
setup(
    name="rlDocking",
    author="Vincent Beau Chen",
    version="2023.7.1.0",
    description="RL environment for space docking with ppo",
    keywords=["robotics", "rl"],
    include_package_data=True,
    install_requires=INSTALL_REQUIRES,
    packages=find_packages("."),
    classifiers=["Natural Language :: English", "Programming Language :: Python :: 3.8, 3.9, 3.10"],
    zip_safe=False,
)
