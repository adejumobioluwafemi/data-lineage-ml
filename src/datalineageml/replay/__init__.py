"""
datalineageml.replay

Counterfactual pipeline replay — Layer 3.3.

  CounterfactualReplayer — re-run a pipeline with a replacement function
                           at the attributed step and measure the delta.
"""

from .replayer import CounterfactualReplayer

__all__ = ["CounterfactualReplayer"]