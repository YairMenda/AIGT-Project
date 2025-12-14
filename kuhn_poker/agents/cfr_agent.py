"""
CFR-Trained Agent
=================

An agent that plays according to a strategy computed by CFR algorithms.
"""

from typing import Dict, List, Optional, TYPE_CHECKING
import numpy as np

from .base import Agent

if TYPE_CHECKING:
    from ..game import GameState, Action
    from ..solvers.base_cfr import BaseCFR


class CFRAgent(Agent):
    """
    An agent that plays using a strategy trained by CFR.
    
    Usage:
        # Train a solver
        solver = VanillaCFR()
        solver.train(10000)
        
        # Create agent from solver
        agent = CFRAgent(solver)
        
        # Play against another agent
        game = KuhnPoker()
        game.play_match(agent, RandomAgent(), num_games=1000)
    """
    
    def __init__(
        self,
        solver: "BaseCFR",
        name: Optional[str] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize the CFR agent with a trained solver.
        
        Args:
            solver: A trained CFR solver
            name: Name for the agent (defaults to solver name)
            seed: Random seed for action sampling
        """
        agent_name = name or f"{solver.name} Agent"
        super().__init__(agent_name)
        
        self.solver = solver
        self.strategy = solver.get_average_strategy()
        self.rng = np.random.default_rng(seed)
        
        # Store action mappings from solver
        self._action_maps: Dict[str, List] = {}
        for info_set, data in solver.info_sets.items():
            self._action_maps[info_set] = data.actions
    
    def get_action(self, state: "GameState") -> "Action":
        """
        Choose an action based on the CFR strategy.
        
        Samples an action according to the trained probability distribution.
        
        Args:
            state: Current game state
        
        Returns:
            The chosen action
        """
        info_set = self.get_info_set(state)
        legal_actions = state.get_legal_actions()
        
        if info_set in self.strategy:
            probs = self.strategy[info_set]
            actions = self._action_maps[info_set]
            
            # Sample action according to probability distribution
            action_idx = self.rng.choice(len(actions), p=probs)
            return actions[action_idx]
        else:
            # Fallback to uniform random if info set not found
            return legal_actions[self.rng.choice(len(legal_actions))]
    
    def get_strategy_at(self, state: "GameState") -> Dict["Action", float]:
        """
        Get the full strategy distribution at a game state.
        
        Args:
            state: Current game state
        
        Returns:
            Dictionary mapping actions to probabilities
        """
        info_set = self.get_info_set(state)
        legal_actions = state.get_legal_actions()
        
        if info_set in self.strategy:
            probs = self.strategy[info_set]
            actions = self._action_maps[info_set]
            return {a: p for a, p in zip(actions, probs)}
        else:
            uniform_prob = 1.0 / len(legal_actions)
            return {a: uniform_prob for a in legal_actions}
    
    def print_strategy(self) -> None:
        """Print the agent's strategy in a readable format."""
        print(f"\n{'='*50}")
        print(f"  {self.name} Strategy")
        print(f"  Exploitability: {self.solver.compute_exploitability():.6f}")
        print(f"{'='*50}\n")
        
        for card in ['J', 'Q', 'K']:
            print(f"  {card} (Jack/Queen/King):")
            for info_set in sorted(self.strategy.keys()):
                if info_set.startswith(card):
                    probs = self.strategy[info_set]
                    actions = self._action_maps[info_set]
                    prob_str = ", ".join(
                        f"{a}: {p:.3f}" for a, p in zip(actions, probs)
                    )
                    print(f"    '{info_set}': [{prob_str}]")
            print()
