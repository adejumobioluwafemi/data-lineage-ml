"""Shared pytest configuration — makes src/ importable without installing."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
