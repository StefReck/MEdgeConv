#!/usr/bin/env python
from setuptools import setup

with open('requirements.txt') as fobj:
    requirements = [l.strip() for l in fobj.readlines()]

with open("README.rst") as fh:
    long_description = fh.read()

setup(
    name='medgeconv',
    author='Stefan Reck',
    author_email='stefan.reck@fau.de',
    long_description=long_description,
    setup_requires=['setuptools_scm'],
    use_scm_version=True,
    install_requires=requirements,
    packages=["medgeconv"],
)
