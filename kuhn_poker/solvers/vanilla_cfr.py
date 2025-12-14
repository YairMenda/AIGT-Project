"""
Vanilla CFR (Counterfactual Regret Minimization)
=================================================

Implementation of the original CFR algorithm by Zinkevich et al. (2007).

This serves as the baseline algorithm against which other CFR variants
are compared.

Reference:
    Zinkevich, M., Johanson, M., Bowling, M., & Piccione, C. (2007).
    Regret minimization in games with incomplete information.
    Advances in Neural Information Processing Systems, 20.

Key Properties:
- Uniform weighting of all iterations
- Standard regret matching for strategy computation
- Converges to Nash equilibrium at O(1/sqrt(T)) rate
"""

import numpy as np
from typing import Optional

from .base_cfr import BaseCFR, InfoSetData


class VanillaCFR(BaseCFR):
    """
    Vanilla CFR - The original Counterfactual Regret Minimization algorithm.
    
    This implementation follows the original Zinkevich algorithm exactly:
    - All iterations are weighted equally
    - Regrets accumulate without modification
    - Average strategy is computed with uniform weights
    
    Example:
        solver = VanillaCFR()
        strategy = solver.train(10000)
        solver.print_strategy()
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize Vanilla CFR solver.
        
        Args:
            seed: Random seed for reproducibility
        """
        super().__init__(seed)
    
    @property
    def name(self) -> str:
        return "Vanilla CFR"
    
    def _iterate(self) -> None:
        """
        Perform one iteration of Vanilla CFR.
        
        Traverses the game tree for all possible card deals,
        updating regrets and strategy sums.
        """
        for cards in self.all_deals:
            state = self.game.new_game(cards)
            # Equal probability of each deal
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
        
        In Vanilla CFR, regrets are simply accumulated with
        counterfactual reach probability weighting.
        
        Args:
            info_set_data: The information set data to update
            regrets: Instantaneous regrets for each action
            counterfactual_reach: Opponent's reach probability
            iteration: Current iteration (unused in Vanilla CFR)
        """
        # Standard regret accumulation
        info_set_data.regret_sum += counterfactual_reach * regrets
    
    def _update_strategy_sum(
        self,
        info_set_data: InfoSetData,
        strategy: np.ndarray,
        player_reach: float,
        iteration: int
    ) -> None:
        """
        Update cumulative strategy for averaging.
        
        In Vanilla CFR, strategies are accumulated with
        uniform weighting (player reach probability only).
        
        Args:
            info_set_data: The information set data to update
            strategy: Current strategy at this info set
            player_reach: Current player's reach probability
            iteration: Current iteration (unused in Vanilla CFR)
        """
        # Standard strategy accumulation with reach probability weighting
        info_set_data.strategy_sum += player_reach * strategy
        info_set_data.reach_prob_sum += player_reach

