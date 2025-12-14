"""
Kuhn Poker Agents
=================

Provides the CFRAgent that plays using a trained CFR strategy.
"""

from .base import Agent
from .cfr_agent import CFRAgent

__all__ = [
    "Agent",
    "CFRAgent",
]
