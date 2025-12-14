"""
Kuhn Poker Game Engine
======================

A complete implementation of Kuhn Poker for studying Counterfactual Regret Minimization (CFR).

This package provides:
- A fully functional game engine
- Extensible agent interfaces
- Game tree representation for CFR algorithms
- Multiple CFR solver implementations (Vanilla, CFR+, DCFR, Linear)
"""

from .game import KuhnPoker, GameState, Action, Card
from .agents import Agent, CFRAgent
from .solvers import VanillaCFR, CFRPlus, DiscountedCFR, LinearCFR

__version__ = "1.0.0"
__all__ = [
    # Game
    "KuhnPoker",
    "GameState", 
    "Action",
    "Card",
    # Agent
    "Agent",
    "CFRAgent",
    # Solvers
    "VanillaCFR",
    "CFRPlus",
    "DiscountedCFR",
    "LinearCFR",
]

