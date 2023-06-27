from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from setuptools import setup, find_packages

import os


# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "torch>=2.0.1",
    "torchvision>=0.14.1",
    "stable-baselines3>=2.0.0a8",
    "tqdm>=4.64.1",
    "tensorboard>=2.11.2",
    "matplotlib>=3.6.2",
    "scipy>=1.9.3",
    "gymnasium>=0.28.1",
    "numpy>=1.23.5",
    "mpld3>=0.5.8",
]

CLASSIFIERS=[
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

# Installation operation
setup(
    name="stellar",
    author="Vincent-C",
    version="v1.0.0-pre",
    description="Single-Stage-Trajectory-Execution-Looking-to-Learning-in-Autonomous-Rendezvous",
    keywords=["ppo", "rl", "RPOD", "docking"],
    include_package_data=True,
    install_requires=INSTALL_REQUIRES,
    packages=find_packages(exclude=['misc']),
    classifiers=CLASSIFIERS,
    url="https://github.com/Vince-C156/STELLAR",
    python_requires=">=3.9",
    zip_safe=False,
)
