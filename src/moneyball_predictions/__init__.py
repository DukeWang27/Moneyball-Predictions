"""Moneyball Predictions package."""

__version__ = "0.13.0"

from .model import log5_probability, pythagorean_expectation
from .odds import devig_two_way_decimal, expected_value

__all__ = [
    "devig_two_way_decimal",
    "expected_value",
    "log5_probability",
    "pythagorean_expectation",
]
