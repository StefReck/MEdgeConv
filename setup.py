#!/usr/bin/env python
from setuptools import setup

with open('requirements.txt') as fobj:
    requirements = [l.strip() for l in fobj.readlines()]

setup(
    name='medgeconv',
    author='Stefan Reck',
    author_email='stefan.reck@fau.de',
    version='0.2',
    install_requires=requirements,
    packages=["medgeconv"],
)

