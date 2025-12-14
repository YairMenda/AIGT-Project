"""
Base Agent Class
================

Defines the interface for all Kuhn Poker agents.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..game import GameState, Action


class Agent(ABC):
    """
    Abstract base class for Kuhn Poker agents.
    
    All agents must implement the get_action method.
    
    Example:
        class MyAgent(Agent):
            def get_action(self, state):
                # Your strategy here
                return state.get_legal_actions()[0]
    """
    
    def __init__(self, name: str = "Agent"):
        """
        Initialize the agent.
        
        Args:
            name: Human-readable name for the agent
        """
        self.name = name
    
    @abstractmethod
    def get_action(self, state: "GameState") -> "Action":
        """
        Choose an action given the current game state.
        
        The agent can only see information available to the current player:
        - Their own card (state.cards[state.current_player])
        - The action history (state.history)
        - Legal actions (state.get_legal_actions())
        
        Args:
            state: Current game state
        
        Returns:
            The chosen action
        """
        pass
    
    def get_info_set(self, state: "GameState") -> str:
        """
        Get the information set string for the current state.
        
        This is useful for strategy lookup in trained agents.
        
        Args:
            state: Current game state
        
        Returns:
            Information set string (e.g., "Jcb" for Jack after check-bet)
        """
        return state.get_info_set(state.current_player)
    
    def observe_result(self, state: "GameState", player_idx: int) -> None:
        """
        Called at the end of each game with the final state.
        
        Override this method if your agent needs to learn from results.
        
        Args:
            state: Final game state
            player_idx: Which player this agent was (0 or 1)
        """
        pass
    
    def reset(self) -> None:
        """
        Reset the agent's internal state for a new match.
        
        Override this method if your agent maintains state across games.
        """
        pass
    
    def __str__(self) -> str:
        return self.name
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"

