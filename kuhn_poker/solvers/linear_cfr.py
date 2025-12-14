"""
Linear CFR
==========

Implementation of Linear CFR, which uses linear iteration weighting
for strategy averaging.

Linear CFR weights iteration t's strategy contribution proportionally
to t, giving later (presumably better) strategies more influence on
the final average.

Reference:
    Brown, N., & Sandholm, T. (2019).
    Solving imperfect-information games via discounted regret minimization.
    (Linear CFR is discussed as a special case of DCFR)

Key Properties:
- Standard regret accumulation (like Vanilla CFR)
- Linear weighting for strategy averaging (like CFR+)
- Weight at iteration t is proportional to t
- Simpler than DCFR but often competitive
"""

import numpy as np
from typing import Optional

from .base_cfr import BaseCFR, InfoSetData


class LinearCFR(BaseCFR):
    """
    Linear CFR - CFR with linear iteration weighting.
    
    This variant uses standard regret accumulation but applies
    linear weighting to the strategy average, giving more weight
    to strategies from later iterations.
    
    Strategy contribution at iteration t is weighted by t,
    meaning the average strategy is:
        σ_avg = Σ(t * σ_t) / Σ(t)
    
    This is equivalent to DCFR with α=1, β=1, γ=1,
    but implemented more directly for efficiency.
    
    Example:
        solver = LinearCFR()
        strategy = solver.train(10000)
        solver.print_strategy()
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize Linear CFR solver.
        
        Args:
            seed: Random seed for reproducibility
        """
        super().__init__(seed)
    
    @property
    def name(self) -> str:
        return "Linear CFR"
    
    def _iterate(self) -> None:
        """
        Perform one iteration of Linear CFR.
        
        Standard CFR tree traversal with linear weighting
        applied to strategy contributions.
        """
        for cards in self.all_deals:
            state = self.game.new_game(cards)
            reach_probs = np.array([1.0, 1.0])
            self._cfr_recursive(state, reach_probs, self.iterations)
    
    def _update_regrets(
        self,
        info_set_data: InfoSetData,
        regrets: np.ndarray,
        counterfactual_reach: float,
        iteration: int
    ) -> None:
        """
        Update cumulative regrets using standard CFR.
        
        Linear CFR uses standard regret accumulation
        (no discounting or flooring).
        
        Args:
            info_set_data: The information set data to update
            regrets: Instantaneous regrets for each action
            counterfactual_reach: Opponent's reach probability
            iteration: Current iteration number (unused)
        """
        # Standard regret accumulation (same as Vanilla CFR)
        info_set_data.regret_sum += counterfactual_reach * regrets
    
    def _update_strategy_sum(
        self,
        info_set_data: InfoSetData,
        strategy: np.ndarray,
        player_reach: float,
        iteration: int
    ) -> None:
        """
        Update cumulative strategy with linear weighting.
        
        Each iteration's strategy is weighted by the iteration number,
        giving more importance to later (presumably better) strategies.
        
        Args:
            info_set_data: The information set data to update
            strategy: Current strategy at this info set
            player_reach: Current player's reach probability
            iteration: Current iteration number (used for weighting)
        """
        # Linear weighting: iteration t weighted by t
        weight = iteration
        info_set_data.strategy_sum += weight * player_reach * strategy
        info_set_data.reach_prob_sum += weight * player_reach


class QuadraticCFR(BaseCFR):
    """
    Quadratic CFR - CFR with quadratic iteration weighting.
    
    Similar to Linear CFR but uses quadratic (t^2) weighting,
    giving even more emphasis to later iterations.
    
    This can be useful when early iterations are particularly
    poor and you want to minimize their influence.
    
    Example:
        solver = QuadraticCFR()
        strategy = solver.train(10000)
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize Quadratic CFR solver.
        
        Args:
            seed: Random seed for reproducibility
        """
        super().__init__(seed)
    
    @property
    def name(self) -> str:
        return "Quadratic CFR"
    
    def _iterate(self) -> None:
        """Perform one iteration of Quadratic CFR."""
        for cards in self.all_deals:
            state = self.game.new_game(cards)
            reach_probs = np.array([1.0, 1.0])
            self._cfr_recursive(state, reach_probs, self.iterations)
    
    def _update_regrets(
        self,
        info_set_data: InfoSetData,
        regrets: np.ndarray,
        counterfactual_reach: float,
        iteration: int
    ) -> None:
        """Update regrets using standard accumulation."""
        info_set_data.regret_sum += counterfactual_reach * regrets
    
    def _update_strategy_sum(
        self,
        info_set_data: InfoSetData,
        strategy: np.ndarray,
        player_reach: float,
        iteration: int
    ) -> None:
        """Update strategy with quadratic weighting (t^2)."""
        weight = iteration ** 2
        info_set_data.strategy_sum += weight * player_reach * strategy
        info_set_data.reach_prob_sum += weight * player_reach

