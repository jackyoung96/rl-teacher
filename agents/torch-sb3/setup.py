import sys

from setuptools import setup

if sys.version_info.major != 3:
    print("This module is only compatible with Python 3, but you are running "
          "Python {}. The installation will likely fail.".format(sys.version_info.major))

setup(name='torchSB3',
    version='0.0.1',
    install_requires=[
        'mujoco-py ~=2.1.2.14',
        'gym[mujoco] ~=0.26.2',
        'multiprocess ~=0.70.5',
        'torch ~=1.13.1',
        'stable-baselines3 ~=1.6.2',
    ],
)
