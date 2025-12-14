"""
CFR+ (CFR Plus)
===============

Implementation of CFR+ by Tammelin (2014) and Bowling et al. (2015).

CFR+ improves upon Vanilla CFR with two key modifications:
1. Regret flooring: Negative regrets are set to zero
2. Alternating updates: Players are updated on alternating iterations

These modifications lead to faster practical convergence while maintaining
theoretical guarantees.

Reference:
    Tammelin, O. (2014). Solving large imperfect information games using CFR+.
    arXiv preprint arXiv:1407.5042.
    
    Bowling, M., Burch, N., Johanson, M., & Tammelin, O. (2015).
    Heads-up limit hold'em poker is solved.
    Science, 347(6218), 145-149.

Key Properties:
- Regret flooring (regrets are non-negative)
- Optional alternating updates
- Linear weighting for strategy averaging (iteration-weighted)
- Faster practical convergence than Vanilla CFR
"""

import numpy as np
from typing import Optional

from .base_cfr import BaseCFR, InfoSetData


class CFRPlus(BaseCFR):
    """
    CFR+ - An improved CFR variant with regret flooring.
    
    Key improvements over Vanilla CFR:
    1. Regrets are floored at zero (never go negative)
    2. Strategy averaging uses linear iteration weights
    3. Optional alternating updates between players
    
    These changes lead to significantly faster convergence in practice.
    
    Example:
        solver = CFRPlus(alternating=True)
        strategy = solver.train(10000)
        solver.print_strategy()
    """
    
    def __init__(self, alternating: bool = True, seed: Optional[int] = None):
        """
        Initialize CFR+ solver.
        
        Args:
            alternating: If True, update players on alternating iterations
            seed: Random seed for reproducibility
        """
        super().__init__(seed)
        self.alternating = alternating
    
    @property
    def name(self) -> str:
        return "CFR+" + (" (Alternating)" if self.alternating else "")
    
    def _iterate(self) -> None:
        """
        Perform one iteration of CFR+.
        
        If alternating updates are enabled, only one player's
        regrets are updated per iteration.
        """
        if self.alternating:
            # Alternating updates: update one player per iteration
            updating_player = (self.iterations - 1) % 2
            
            for cards in self.all_deals:
                state = self.game.new_game(cards)
                reach_probs = np.array([1.0, 1.0])
                self._cfr_plus_recursive(
                    state, reach_probs, self.iterations, updating_player
                )
        else:
            # Simultaneous updates (like Vanilla CFR but with regret flooring)
            for cards in self.all_deals:
                state = self.game.new_game(cards)
                reach_probs = np.array([1.0, 1.0])
                self._cfr_recursive(state, reach_probs, self.iterations)
    
    def _cfr_plus_recursive(
        self,
        state,
        reach_probs: np.ndarray,
        iteration: int,
        updating_player: int
    ) -> np.ndarray:
        """
        CFR+ recursive traversal with alternating updates.
        
        Args:
            state: Current game state
            reach_probs: Reach probabilities for each player
            iteration: Current iteration number
            updating_player: Which player is being updated this iteration
        
        Returns:
            Expected utility for each player
        """
        from ..game import GameState
        
        # Terminal state - return payoffs
        if state.is_terminal:
            return np.array(state.payoffs, dtype=np.float64)
        
        current_player = state.current_player
        opponent = 1 - current_player
        
        # Get information set and legal actions
        info_set = state.get_info_set(current_player)
        legal_actions = state.get_legal_actions()
        info_set_data = self.get_info_set_data(info_set, legal_actions)
        
        # Get current strategy from regret matching
        strategy = info_set_data.get_strategy()
        
        # Initialize utilities
        action_utilities = np.zeros(len(legal_actions))
        node_utility = np.zeros(2)
        
        # Recurse for each action
        for i, action in enumerate(legal_actions):
            next_state = self.game.apply_action(state, action)
            
            # Update reach probabilities
            new_reach_probs = reach_probs.copy()
            new_reach_probs[current_player] *= strategy[i]
            
            # Recurse
            action_utility = self._cfr_plus_recursive(
                next_state, new_reach_probs, iteration, updating_player
            )
            
            action_utilities[i] = action_utility[current_player]
            node_utility += strategy[i] * action_utility
        
        # Only update regrets for the updating player
        if current_player == updating_player:
            counterfactual_reach = reach_probs[opponent]
            regrets = action_utilities - node_utility[current_player]
            self._update_regrets(info_set_data, regrets, counterfactual_reach, iteration)
        
        # Always update strategy sum (using linear weighting)
        player_reach = reach_probs[current_player]
        self._update_strategy_sum(info_set_data, strategy, player_reach, iteration)
        
        return node_utility
    
    def _update_regrets(
        self,
        info_set_data: InfoSetData,
        regrets: np.ndarray,
        counterfactual_reach: float,
        iteration: int
    ) -> None:
        """
        Update cumulative regrets with regret flooring.
        
        In CFR+, after adding the new regrets, negative values
        are floored to zero. This prevents strategies from being
        "locked out" due to early negative regrets.
        
        Args:
            info_set_data: The information set data to update
            regrets: Instantaneous regrets for each action
            counterfactual_reach: Opponent's reach probability
            iteration: Current iteration number
        """
        # Add new regrets
        info_set_data.regret_sum += counterfactual_reach * regrets
        
        # Floor negative regrets to zero (key CFR+ modification)
        info_set_data.regret_sum = np.maximum(info_set_data.regret_sum, 0)
    
    def _update_strategy_sum(
        self,
        info_set_data: InfoSetData,
        strategy: np.ndarray,
        player_reach: float,
        iteration: int
    ) -> None:
        """
        Update cumulative strategy with linear iteration weighting.
        
        In CFR+, later iterations are weighted more heavily.
        This gives more importance to strategies from later iterations
        when they are presumably closer to equilibrium.
        
        Args:
            info_set_data: The information set data to update
            strategy: Current strategy at this info set
            player_reach: Current player's reach probability
            iteration: Current iteration number (used for linear weighting)
        """
        # Linear weighting: later iterations weighted more heavily
        weight = iteration
        info_set_data.strategy_sum += weight * player_reach * strategy
        info_set_data.reach_prob_sum += weight * player_reach

