import sys
from setuptools import setup, Extension, find_packages
from setuptools.command.install import install

setup(
    packages=find_packages(include=["asmk", "asmk.*"]),
    ext_modules=[Extension("asmk.hamming", ["cython/hamming.c"])],
)