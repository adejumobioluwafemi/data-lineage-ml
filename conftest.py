"""
Shared pytest fixtures and configuration.
"""

import sys
import os

# Make the src package importable from tests without installing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
