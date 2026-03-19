"""
Physical constraint handling module for protein refolding optimization.

Contains constraint implementations for ensuring physically feasible
experimental conditions.
"""

from .urea_dilution import (
    check_urea_constraint,
    iterative_urea_adjustment,
    correct_constraints_iterative,
    calculate_urea_refolding_concentration
)

__all__ = [
    'check_urea_constraint',
    'iterative_urea_adjustment',
    'correct_constraints_iterative',
    'calculate_urea_refolding_concentration'
]