<<<<<<< HEAD

=======
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
>>>>>>> 93258bf2472ca54e0cf85c7a5f855f26cc688051

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
<<<<<<< HEAD
    "python-dateutil>=2.8.2",
    "Pillow",
=======
>>>>>>> 93258bf2472ca54e0cf85c7a5f855f26cc688051
]

CLASSIFIERS=[
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
<<<<<<< HEAD
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.9",
=======
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
>>>>>>> 93258bf2472ca54e0cf85c7a5f855f26cc688051
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
<<<<<<< HEAD
    packages=find_packages(include=['stellar', 'stellar.*'], exclude=['misc']),
=======
    packages=find_packages(exclude=['misc']),
>>>>>>> 93258bf2472ca54e0cf85c7a5f855f26cc688051
    classifiers=CLASSIFIERS,
    url="https://github.com/Vince-C156/STELLAR",
    python_requires=">=3.9",
    zip_safe=False,
)
