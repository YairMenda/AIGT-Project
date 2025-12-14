"""
Custom CFR Solvers
==================

Experimental CFR variants with unique properties.

1. ExponentialCFR: Aggressive strategy forgetting using exponential weighting.
2. SoftmaxCFR: Uses Boltzmann exploration instead of standard Regret Matching.
3. RandomSolver: Baseline random player.
4. PrunedCFR: Custom solver with dynamic tree pruning.
"""

import numpy as np
from typing import List, Optional, Callable, Dict
from dataclasses import dataclass

from .base_cfr import BaseCFR, InfoSetData, Action


class ExponentialCFR(BaseCFR):
    """
    Exponential CFR
    ===============
    
    A variant that weights strategy updates exponentially: w_t = (1 + rate)^t.
    
    Unlike Linear CFR (weight ~ t), this solver aggressively prioritizes
    recent iterations. This allows it to converge extremely fast in simple
    games like Kuhn Poker by rapidly discarding early, poor strategies.
    
    Attributes:
        rate (float): The growth rate for weighting (default 0.001)
    """
    
    def __init__(self, rate: float = 0.001, seed: Optional[int] = None):
        super().__init__(seed)
        self.rate = rate
    
    @property
    def name(self) -> str:
        return f"Exponential CFR (rate={self.rate})"
    
    def _iterate(self) -> None:
        """Standard CFR iteration."""
        for cards in self.all_deals:
            state = self.game.new_game(cards)
            reach_probs = np.array([1.0, 1.0])
            self._cfr_recursive(state, reach_probs, self.iterations)
    
    def _update_regrets(self, info_set_data, regrets, counterfactual_reach, iteration):
        """Standard regret accumulation."""
        info_set_data.regret_sum += counterfactual_reach * regrets
    
    def _update_strategy_sum(self, info_set_data, strategy, player_reach, iteration):
        """
        Update strategy with exponential weighting.
        weight = (1 + rate) ^ iteration
        """
        weight = (1 + self.rate) ** iteration
        info_set_data.strategy_sum += weight * player_reach * strategy
        info_set_data.reach_prob_sum += weight * player_reach


@dataclass
class SoftmaxInfoSetData(InfoSetData):
    """InfoSetData that uses Softmax for strategy generation."""
    temperature: float = 0.1
    
    def get_strategy(self) -> np.ndarray:
        """
        Compute strategy using Softmax (Boltzmann exploration).
        
        P(a) = exp(R(a) / T) / sum(exp(R(a') / T))
        
        This prevents actions with negative regrets from being strictly zeroed out,
        maintaining exploration.
        """
        # Numerical stability: subtract max regret
        regrets = self.regret_sum
        max_regret = np.max(regrets)
        
        # Calculate exp values
        exp_regrets = np.exp((regrets - max_regret) / self.temperature)
        sum_exp = np.sum(exp_regrets)
        
        if sum_exp > 0:
            return exp_regrets / sum_exp
        else:
            return np.ones(len(self.actions)) / len(self.actions)


class SoftmaxCFR(BaseCFR):
    """
    Softmax CFR
    ===========
    
    Replaces the standard Regret Matching (RM) with Softmax (Hedge).
    
    Instead of zeroing out negative regrets, Softmax assigns probabilities
    proportionally to exp(regret), ensuring that even "bad" actions maintain
    a small non-zero probability. This promotes better exploration and
    prevents getting stuck in local optima in more complex games.
    
    Attributes:
        temperature (float): Controls exploration (lower = more greedy).
    """
    
    def __init__(self, temperature: float = 0.1, seed: Optional[int] = None):
        super().__init__(seed)
        self.temperature = temperature
    
    @property
    def name(self) -> str:
        return f"Softmax CFR (temp={self.temperature})"
    
    def get_info_set_data(self, info_set: str, legal_actions: List[Action]) -> InfoSetData:
        """Override to return SoftmaxInfoSetData."""
        if info_set not in self.info_sets:
            # Create custom InfoSetData with specific temperature
            data = SoftmaxInfoSetData(
                actions=legal_actions,
                regret_sum=np.zeros(len(legal_actions)),
                strategy_sum=np.zeros(len(legal_actions))
            )
            data.temperature = self.temperature
            self.info_sets[info_set] = data
        return self.info_sets[info_set]
    
    def _iterate(self) -> None:
        """Standard CFR iteration."""
        for cards in self.all_deals:
            state = self.game.new_game(cards)
            reach_probs = np.array([1.0, 1.0])
            self._cfr_recursive(state, reach_probs, self.iterations)
    
    def _update_regrets(self, info_set_data, regrets, counterfactual_reach, iteration):
        """Standard regret accumulation."""
        info_set_data.regret_sum += counterfactual_reach * regrets
    
    def _update_strategy_sum(self, info_set_data, strategy, player_reach, iteration):
        """Standard strategy accumulation."""
        info_set_data.strategy_sum += player_reach * strategy
        info_set_data.reach_prob_sum += player_reach


class RandomSolver(BaseCFR):
    """
    Random Solver
    =============
    
    A baseline agent that plays uniformly random actions.
    
    It does not learn or update regrets. Its strategy is fixed at uniform random
    for all information sets.
    """
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
    
    @property
    def name(self) -> str:
        return "Random Solver"
        
    def train(self, iterations: int, callback: Optional[Callable[[int, float], None]] = None,
              compute_exploitability_every: int = 100) -> Dict[str, np.ndarray]:
        """
        Mock training loop for Random Solver.
        It doesn't actually train, but we simulate iterations to match the interface.
        """
        # Initialize info sets if not already done
        self._init_info_sets()
        
        for i in range(iterations):
            self.iterations += 1
            
            # No learning happens
            
            if compute_exploitability_every > 0 and (i + 1) % compute_exploitability_every == 0:
                expl = self.compute_exploitability()
                self.exploitability_history.append(expl)
                
                if callback:
                    callback(self.iterations, expl)
                    
        return self.get_average_strategy()

    def _init_info_sets(self):
        """Initialize all info sets with uniform strategy."""
        from ..game import get_all_info_sets
        all_info_sets = get_all_info_sets(self.game)
        
        for info_set, actions in all_info_sets.items():
            if info_set not in self.info_sets:
                num_actions = len(actions)
                uniform_strategy = np.ones(num_actions) / num_actions
                
                self.info_sets[info_set] = InfoSetData(
                    actions=actions,
                    regret_sum=np.zeros(num_actions),
                    strategy_sum=uniform_strategy # Initialize with uniform
                )

    def _iterate(self) -> None:
        pass

    def _update_regrets(self, info_set_data, regrets, counterfactual_reach, iteration):
        pass

    def _update_strategy_sum(self, info_set_data, strategy, player_reach, iteration):
        pass
        
    def get_average_strategy(self) -> Dict[str, np.ndarray]:
        """Return uniform strategy."""
        strategies = {}
        # Ensure info sets are initialized
        if not self.info_sets:
            self._init_info_sets()
            
        for info_set, data in self.info_sets.items():
            strategies[info_set] = data.strategy_sum # Already uniform
        return strategies
    
    def compute_exploitability(self) -> float:
        # Ensure info sets are initialized before computing exploitability
        if not self.info_sets:
            self._init_info_sets()
        return super().compute_exploitability()


class PrunedCFR(BaseCFR):
    """
    Pruned CFR (Novel Idea)
    =======================
    
    A variant that dynamically prunes branches of the game tree with very low 
    probability of being reached. This focuses computational effort on more 
    relevant parts of the game tree.
    
    If the probability of reaching a node is below a threshold, the recursion 
    terminates early.
    
    Attributes:
        pruning_threshold (float): Minimum probability to continue traversal (default 1e-4)
    """
    
    def __init__(self, pruning_threshold: float = 1e-4, seed: Optional[int] = None):
        super().__init__(seed)
        self.pruning_threshold = pruning_threshold
        
    @property
    def name(self) -> str:
        return f"Pruned CFR (thresh={self.pruning_threshold})"
        
    def _iterate(self) -> None:
        """Standard CFR iteration."""
        for cards in self.all_deals:
            state = self.game.new_game(cards)
            reach_probs = np.array([1.0, 1.0])
            self._cfr_recursive(state, reach_probs, self.iterations)
            
    def _cfr_recursive(self, state, reach_probs, iteration) -> np.ndarray:
        # Pruning check: if reach prob for current player is too low, stop
        current_player = state.current_player
        if not state.is_terminal and reach_probs[current_player] < self.pruning_threshold:
             return np.zeros(2) # Approximate 0 return for pruned branch
             
        return super()._cfr_recursive(state, reach_probs, iteration)

    def _update_regrets(self, info_set_data, regrets, counterfactual_reach, iteration):
        """Standard regret accumulation."""
        info_set_data.regret_sum += counterfactual_reach * regrets
    
    def _update_strategy_sum(self, info_set_data, strategy, player_reach, iteration):
        """Standard strategy accumulation."""
        info_set_data.strategy_sum += player_reach * strategy
        info_set_data.reach_prob_sum += player_reach
