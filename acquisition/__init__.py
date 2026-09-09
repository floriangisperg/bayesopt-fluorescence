"""
Acquisition function module for Bayesian optimization.

Contains acquisition function implementations and utilities for experimental planning.
"""

from .qnehvi import create_qnehvi_acquisition, optimize_qnehvi
from .utils import generate_initial_design, update_experimental_database

__all__ = [
    'create_qnehvi_acquisition',
    'optimize_qnehvi',
    'update_experimental_database',
    'generate_initial_design'
]
