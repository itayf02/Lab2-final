# setup.py
from setuptools import setup
from Cython.Build import cythonize
import chess

setup(
    ext_modules=cythonize("alpha_beta_pruning.pyx", language_level="3"),
    install_requires=["chess"],
)