"""
CFR Solvers for Kuhn Poker
==========================

This module implements various Counterfactual Regret Minimization (CFR) algorithms
for solving Kuhn Poker.

Available Solvers:
- VanillaCFR: Original Zinkevich algorithm (baseline)
- CFRPlus: CFR+ with alternating updates and regret flooring
- DiscountedCFR: DCFR with discount factors on regrets and strategies
- LinearCFR: CFR with linear weighting for strategy averaging
- QuadraticCFR: CFR with quadratic weighting for strategy averaging
- ExponentialCFR: Custom solver with exponential strategy weighting
- SoftmaxCFR: Custom solver using Softmax/Boltzmann exploration
- RandomSolver: Baseline random player
- PrunedCFR: Custom solver with dynamic tree pruning
"""

from .base_cfr import BaseCFR, InfoSetData
from .vanilla_cfr import VanillaCFR
from .cfr_plus import CFRPlus
from .discounted_cfr import DiscountedCFR
from .linear_cfr import LinearCFR, QuadraticCFR
from .custom_solvers import ExponentialCFR, SoftmaxCFR, RandomSolver, PrunedCFR

__all__ = [
    "BaseCFR",
    "InfoSetData",
    "VanillaCFR",
    "CFRPlus",
    "DiscountedCFR",
    "LinearCFR",
    "QuadraticCFR",
    "ExponentialCFR",
    "SoftmaxCFR",
    "RandomSolver",
    "PrunedCFR",
]
