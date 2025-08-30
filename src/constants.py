"""This module provides backward compatibility for constants.

It imports all constants from the `constants` package, so that existing code
that imports from `src.constants` will continue to work.
"""
# Backward compatibility - import everything from the constants package
# This file maintains backward compatibility with existing code that imports from src.constants
from constants import *