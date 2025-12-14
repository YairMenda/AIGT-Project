"""
Discounted CFR (DCFR)
=====================

Implementation of Discounted CFR by Brown & Sandholm (2019).

DCFR applies discount factors to both regrets and average strategies,
giving less weight to earlier iterations. This addresses the slow
forgetting of early mistakes in standard CFR.

Reference:
    Brown, N., & Sandholm, T. (2019).
    Solving imperfect-information games via discounted regret minimization.
    Proceedings of the AAAI Conference on Artificial Intelligence, 33, 1829-1836.

Key Properties:
- Discounts positive regrets by α^t
- Discounts negative regrets by β^t  
- Discounts strategy contributions by γ^t
- Faster convergence than Vanilla CFR
- Multiple preset configurations available
"""

import numpy as np
from typing import Optional

from .base_cfr import BaseCFR, InfoSetData


class DiscountedCFR(BaseCFR):
    """
    Discounted CFR (DCFR) - CFR with temporal discounting.
    
    DCFR applies different discount factors to:
    - Positive regrets (alpha parameter)
    - Negative regrets (beta parameter)
    - Strategy contributions (gamma parameter)
    
    This allows the algorithm to "forget" early mistakes faster
    and converge more quickly to equilibrium.
    
    Common configurations:
    - DCFR: alpha=1.5, beta=0, gamma=2 (original paper)
    - CFR+: alpha=inf, beta=inf, gamma=1 (equivalent to CFR+)
    - Linear CFR: alpha=1, beta=1, gamma=1
    
    Example:
        # Use default DCFR parameters
        solver = DiscountedCFR()
        strategy = solver.train(10000)
        
        # Custom parameters
        solver = DiscountedCFR(alpha=2.0, beta=0.5, gamma=3.0)
    """
    
    def __init__(
        self,
        alpha: float = 1.5,
        beta: float = 0.0,
        gamma: float = 2.0,
        seed: Optional[int] = None
    ):
        """
        Initialize DCFR solver.
        
        The discount at iteration t for parameter p is: t^p / (t^p + 1)
        
        Args:
            alpha: Discount exponent for positive regrets (default 1.5)
            beta: Discount exponent for negative regrets (default 0)
            gamma: Discount exponent for strategy averaging (default 2)
            seed: Random seed for reproducibility
        """
        super().__init__(seed)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    @property
    def name(self) -> str:
        return f"DCFR (α={self.alpha}, β={self.beta}, γ={self.gamma})"
    
    def _compute_discount(self, t: int, exponent: float) -> float:
        """
        Compute the discount factor for iteration t.
        
        The discount formula is: t^exponent / (t^exponent + 1)
        
        This gives:
        - Early iterations (small t): discount close to 0
        - Later iterations (large t): discount close to 1
        
        Args:
            t: Iteration number
            exponent: Discount exponent parameter
        
        Returns:
            Discount factor in [0, 1)
        """
        if exponent == float('inf'):
            return 1.0  # No discounting
        
        t_exp = t ** exponent
        return t_exp / (t_exp + 1)
    
    def _iterate(self) -> None:
        """
        Perform one iteration of DCFR.
        
        First applies discounting to existing regrets and strategies,
        then performs standard CFR tree traversal.
        """
        t = self.iterations
        
        # Compute discount factors for this iteration
        alpha_discount = self._compute_discount(t, self.alpha)
        beta_discount = self._compute_discount(t, self.beta)
        gamma_discount = self._compute_discount(t, self.gamma)
        
        # Apply discounts to existing regrets and strategies
        for info_set_data in self.info_sets.values():
            # Discount positive and negative regrets separately
            positive_mask = info_set_data.regret_sum > 0
            negative_mask = info_set_data.regret_sum < 0
            
            info_set_data.regret_sum[positive_mask] *= alpha_discount
            info_set_data.regret_sum[negative_mask] *= beta_discount
            
            # Discount strategy sum
            info_set_data.strategy_sum *= gamma_discount
            info_set_data.reach_prob_sum *= gamma_discount
        
        # Standard CFR tree traversal
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
        Update cumulative regrets in DCFR.
        
        Note: Discounting is applied at the start of each iteration,
        so here we just add the new regrets normally.
        
        Args:
            info_set_data: The information set data to update
            regrets: Instantaneous regrets for each action
            counterfactual_reach: Opponent's reach probability
            iteration: Current iteration number
        """
        # Simply accumulate regrets (discounting was already applied)
        info_set_data.regret_sum += counterfactual_reach * regrets
    
    def _update_strategy_sum(
        self,
        info_set_data: InfoSetData,
        strategy: np.ndarray,
        player_reach: float,
        iteration: int
    ) -> None:
        """
        Update cumulative strategy in DCFR.
        
        Note: Discounting is applied at the start of each iteration,
        so here we just add the new strategy contribution normally.
        
        Args:
            info_set_data: The information set data to update
            strategy: Current strategy at this info set
            player_reach: Current player's reach probability
            iteration: Current iteration number
        """
        # Simply accumulate strategy (discounting was already applied)
        info_set_data.strategy_sum += player_reach * strategy
        info_set_data.reach_prob_sum += player_reach
    
    @classmethod
    def with_preset(cls, preset: str, seed: Optional[int] = None) -> "DiscountedCFR":
        """
        Create a DCFR solver with a preset configuration.
        
        Available presets:
        - "dcfr": Original DCFR (α=1.5, β=0, γ=2)
        - "dcfr_aggressive": More aggressive discounting (α=2, β=0, γ=3)
        - "dcfr_conservative": Less aggressive (α=1, β=0.5, γ=1.5)
        
        Args:
            preset: Name of the preset configuration
            seed: Random seed
        
        Returns:
            Configured DCFR solver
        """
        presets = {
            "dcfr": {"alpha": 1.5, "beta": 0.0, "gamma": 2.0},
            "dcfr_aggressive": {"alpha": 2.0, "beta": 0.0, "gamma": 3.0},
            "dcfr_conservative": {"alpha": 1.0, "beta": 0.5, "gamma": 1.5},
        }
        
        if preset not in presets:
            available = ", ".join(presets.keys())
            raise ValueError(f"Unknown preset: {preset}. Available: {available}")
        
        params = presets[preset]
        return cls(**params, seed=seed)

