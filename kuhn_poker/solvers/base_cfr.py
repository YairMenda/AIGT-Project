"""
Base CFR Solver
===============

Provides the foundational infrastructure for all CFR algorithm variants.
This includes information set tracking, regret accumulation, and strategy computation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
from copy import deepcopy

from ..game import KuhnPoker, GameState, Action, Card


@dataclass
class InfoSetData:
    """
    Data stored for each information set during CFR training.
    
    Attributes:
        actions: List of legal actions at this information set
        regret_sum: Cumulative regrets for each action
        strategy_sum: Cumulative strategy probabilities (for averaging)
        reach_prob_sum: Sum of reach probabilities (for weighted averaging)
    """
    actions: List[Action] = field(default_factory=list)
    regret_sum: np.ndarray = field(default_factory=lambda: np.array([]))
    strategy_sum: np.ndarray = field(default_factory=lambda: np.array([]))
    reach_prob_sum: float = 0.0
    
    def __post_init__(self):
        if len(self.actions) > 0 and len(self.regret_sum) == 0:
            self.regret_sum = np.zeros(len(self.actions))
            self.strategy_sum = np.zeros(len(self.actions))
    
    def get_strategy(self) -> np.ndarray:
        """
        Compute current strategy using regret matching.
        
        Returns:
            Normalized strategy based on positive regrets
        """
        # Use positive regrets only
        positive_regrets = np.maximum(self.regret_sum, 0)
        regret_sum = np.sum(positive_regrets)
        
        if regret_sum > 0:
            return positive_regrets / regret_sum
        else:
            # Uniform strategy when no positive regrets
            return np.ones(len(self.actions)) / len(self.actions)
    
    def get_average_strategy(self) -> np.ndarray:
        """
        Compute the average strategy over all iterations.
        
        Returns:
            Time-averaged strategy
        """
        strategy_sum_total = np.sum(self.strategy_sum)
        
        if strategy_sum_total > 0:
            return self.strategy_sum / strategy_sum_total
        else:
            # Uniform strategy if no strategy accumulated
            return np.ones(len(self.actions)) / len(self.actions)


class BaseCFR(ABC):
    """
    Abstract base class for CFR algorithm implementations.
    
    This class provides the common infrastructure for CFR variants:
    - Game tree traversal
    - Information set management
    - Strategy computation and storage
    
    Subclasses must implement the specific CFR variant logic.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the CFR solver.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.game = KuhnPoker(seed=seed)
        self.info_sets: Dict[str, InfoSetData] = {}
        self.iterations = 0
        self.exploitability_history: List[float] = []
        self.rng = np.random.default_rng(seed)
        
        # Pre-compute all possible deals for iteration
        self.all_deals = self.game.get_all_possible_deals()
    
    @property
    def name(self) -> str:
        """Return the name of this CFR variant."""
        return self.__class__.__name__
    
    def get_info_set_data(self, info_set: str, legal_actions: List[Action]) -> InfoSetData:
        """
        Get or create InfoSetData for an information set.
        
        Args:
            info_set: Information set string
            legal_actions: Legal actions at this info set
        
        Returns:
            InfoSetData for this information set
        """
        if info_set not in self.info_sets:
            self.info_sets[info_set] = InfoSetData(
                actions=legal_actions,
                regret_sum=np.zeros(len(legal_actions)),
                strategy_sum=np.zeros(len(legal_actions))
            )
        return self.info_sets[info_set]
    
    def train(self, iterations: int, callback: Optional[Callable[[int, float], None]] = None,
              compute_exploitability_every: int = 100) -> Dict[str, np.ndarray]:
        """
        Run CFR training for a specified number of iterations.
        
        Args:
            iterations: Number of CFR iterations to run
            callback: Optional callback(iteration, exploitability) called periodically
            compute_exploitability_every: How often to compute exploitability
        
        Returns:
            Dictionary mapping info_set strings to average strategy arrays
        """
        for i in range(iterations):
            self.iterations += 1
            
            # Run one iteration of CFR
            self._iterate()
            
            # Compute exploitability periodically
            if compute_exploitability_every > 0 and (i + 1) % compute_exploitability_every == 0:
                expl = self.compute_exploitability()
                self.exploitability_history.append(expl)
                
                if callback:
                    callback(self.iterations, expl)
        
        return self.get_average_strategy()
    
    @abstractmethod
    def _iterate(self) -> None:
        """
        Perform one iteration of the CFR algorithm.
        
        This must be implemented by each CFR variant.
        """
        pass
    
    def _cfr_recursive(
        self,
        state: GameState,
        reach_probs: np.ndarray,
        iteration: int
    ) -> np.ndarray:
        """
        Recursive CFR tree traversal.
        
        Args:
            state: Current game state
            reach_probs: Reach probabilities for each player [p0, p1]
            iteration: Current iteration number (for weighting schemes)
        
        Returns:
            Expected utility for each player
        """
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
            action_utility = self._cfr_recursive(next_state, new_reach_probs, iteration)
            
            action_utilities[i] = action_utility[current_player]
            node_utility += strategy[i] * action_utility
        
        # Update regrets and strategy (handled by subclass hooks)
        counterfactual_reach = reach_probs[opponent]
        player_reach = reach_probs[current_player]
        
        # Compute regrets
        regrets = action_utilities - node_utility[current_player]
        
        # Let subclass update regrets (different CFR variants do this differently)
        self._update_regrets(info_set_data, regrets, counterfactual_reach, iteration)
        
        # Let subclass update strategy sum (different weighting schemes)
        self._update_strategy_sum(info_set_data, strategy, player_reach, iteration)
        
        return node_utility
    
    @abstractmethod
    def _update_regrets(
        self,
        info_set_data: InfoSetData,
        regrets: np.ndarray,
        counterfactual_reach: float,
        iteration: int
    ) -> None:
        """
        Update cumulative regrets for an information set.
        
        Args:
            info_set_data: The information set data to update
            regrets: Instantaneous regrets for each action
            counterfactual_reach: Opponent's reach probability
            iteration: Current iteration number
        """
        pass
    
    @abstractmethod
    def _update_strategy_sum(
        self,
        info_set_data: InfoSetData,
        strategy: np.ndarray,
        player_reach: float,
        iteration: int
    ) -> None:
        """
        Update cumulative strategy for averaging.
        
        Args:
            info_set_data: The information set data to update
            strategy: Current strategy at this info set
            player_reach: Current player's reach probability
            iteration: Current iteration number
        """
        pass
    
    def get_average_strategy(self) -> Dict[str, np.ndarray]:
        """
        Get the average strategy for all information sets.
        
        Returns:
            Dictionary mapping info_set string to probability distribution
        """
        strategies = {}
        for info_set, data in self.info_sets.items():
            strategies[info_set] = data.get_average_strategy()
        return strategies
    
    def get_current_strategy(self) -> Dict[str, np.ndarray]:
        """
        Get the current (regret-matched) strategy for all information sets.
        
        Returns:
            Dictionary mapping info_set string to probability distribution
        """
        strategies = {}
        for info_set, data in self.info_sets.items():
            strategies[info_set] = data.get_strategy()
        return strategies
    
    def get_strategy_for_info_set(self, info_set: str) -> Tuple[List[Action], np.ndarray]:
        """
        Get the average strategy for a specific information set.
        
        Args:
            info_set: Information set string
        
        Returns:
            Tuple of (actions, probabilities)
        """
        if info_set in self.info_sets:
            data = self.info_sets[info_set]
            return data.actions, data.get_average_strategy()
        else:
            raise KeyError(f"Unknown information set: {info_set}")
    
    def compute_exploitability(self) -> float:
        """
        Compute the exploitability of the current average strategy.
        
        Exploitability measures how much value a best-response opponent
        can extract from the current strategy.
        
        Returns:
            Exploitability in betting units
        """
        strategy = self.get_average_strategy()
        
        # Compute best response values for each player
        br_value_0 = self._compute_best_response_value(0, strategy)
        br_value_1 = self._compute_best_response_value(1, strategy)
        
        # Exploitability is the average of both best response advantages
        # Divide by 2 since it's a two-player zero-sum game
        exploitability = (br_value_0 + br_value_1) / 2
        
        return exploitability
    
    def _compute_best_response_value(
        self,
        br_player: int,
        strategy: Dict[str, np.ndarray]
    ) -> float:
        """
        Compute the value of the best response for one player.
        
        Args:
            br_player: Player computing best response
            strategy: Fixed strategy for the opponent
        
        Returns:
            Expected value against best-responding player
        """
        total_value = 0.0
        num_deals = len(self.all_deals)
        
        for cards in self.all_deals:
            state = self.game.new_game(cards)
            value = self._best_response_recursive(state, br_player, strategy, 1.0)
            total_value += value / num_deals
        
        return total_value
    
    def _best_response_recursive(
        self,
        state: GameState,
        br_player: int,
        strategy: Dict[str, np.ndarray],
        prob: float
    ) -> float:
        """
        Recursively compute best response value.
        
        Args:
            state: Current game state
            br_player: Player computing best response
            strategy: Fixed strategy for opponent
            prob: Probability of reaching this state
        
        Returns:
            Best response value for br_player
        """
        if state.is_terminal:
            return state.payoffs[br_player]
        
        current_player = state.current_player
        info_set = state.get_info_set(current_player)
        legal_actions = state.get_legal_actions()
        
        if current_player == br_player:
            # Best response: take maximum value action
            max_value = float('-inf')
            for action in legal_actions:
                next_state = self.game.apply_action(state, action)
                value = self._best_response_recursive(next_state, br_player, strategy, prob)
                max_value = max(max_value, value)
            return max_value
        else:
            # Opponent plays according to fixed strategy
            if info_set in strategy:
                action_probs = strategy[info_set]
            else:
                # Uniform if not in strategy
                action_probs = np.ones(len(legal_actions)) / len(legal_actions)
            
            expected_value = 0.0
            for i, action in enumerate(legal_actions):
                next_state = self.game.apply_action(state, action)
                action_prob = action_probs[i]
                value = self._best_response_recursive(
                    next_state, br_player, strategy, prob * action_prob
                )
                expected_value += action_prob * value
            
            return expected_value
    
    def get_statistics(self) -> Dict:
        """
        Get training statistics.
        
        Returns:
            Dictionary with training statistics
        """
        return {
            "algorithm": self.name,
            "iterations": self.iterations,
            "num_info_sets": len(self.info_sets),
            "exploitability": self.exploitability_history[-1] if self.exploitability_history else None,
            "exploitability_history": self.exploitability_history.copy(),
        }
    
    def print_strategy(self, average: bool = True) -> None:
        """
        Print the current strategy in a readable format.
        
        Args:
            average: If True, print average strategy; else print current strategy
        """
        strategy = self.get_average_strategy() if average else self.get_current_strategy()
        
        print(f"\n{'='*60}")
        print(f"  {self.name} Strategy ({'Average' if average else 'Current'})")
        print(f"  Iterations: {self.iterations}")
        if self.exploitability_history:
            print(f"  Exploitability: {self.exploitability_history[-1]:.6f}")
        print(f"{'='*60}\n")
        
        # Group by card
        for card in ['J', 'Q', 'K']:
            print(f"  {card} (Jack/Queen/King):")
            for info_set in sorted(strategy.keys()):
                if info_set.startswith(card):
                    probs = strategy[info_set]
                    data = self.info_sets[info_set]
                    
                    prob_str = ", ".join(
                        f"{a}: {p:.3f}" for a, p in zip(data.actions, probs)
                    )
                    print(f"    '{info_set}': [{prob_str}]")
            print()
        
        print(f"{'='*60}\n")
    
    def save_strategy(self, filepath: str) -> None:
        """
        Save the trained strategy to a file.
        
        Args:
            filepath: Path to save the strategy
        """
        import json
        
        strategy_data = {
            "algorithm": self.name,
            "iterations": self.iterations,
            "exploitability": self.exploitability_history[-1] if self.exploitability_history else None,
            "info_sets": {}
        }
        
        for info_set, data in self.info_sets.items():
            avg_strategy = data.get_average_strategy()
            strategy_data["info_sets"][info_set] = {
                "actions": [str(a) for a in data.actions],
                "strategy": avg_strategy.tolist()
            }
        
        with open(filepath, 'w') as f:
            json.dump(strategy_data, f, indent=2)
    
    @classmethod
    def load_strategy(cls, filepath: str) -> Dict[str, Tuple[List[str], np.ndarray]]:
        """
        Load a strategy from a file.
        
        Args:
            filepath: Path to load the strategy from
        
        Returns:
            Dictionary mapping info_set to (actions, probabilities)
        """
        import json
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        strategy = {}
        for info_set, info_data in data["info_sets"].items():
            strategy[info_set] = (
                info_data["actions"],
                np.array(info_data["strategy"])
            )
        
        return strategy

