"""
Kuhn Poker Game Engine
======================

Implements the complete game logic for Kuhn Poker, a simplified poker variant
used for studying game theory and CFR algorithms.

Game Rules:
-----------
- 2 players, 3-card deck (Jack, Queen, King)
- Each player antes 1 unit
- Each player receives one private card
- Player 1 acts first: CHECK or BET (1 unit)
- If P1 checks:
    - P2 can CHECK (showdown) or BET (1 unit)
    - If P2 bets, P1 can FOLD or CALL
- If P1 bets:
    - P2 can FOLD or CALL
- Showdown: Higher card wins the pot
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import random
from copy import deepcopy


class Card(Enum):
    """Kuhn Poker cards, ordered by rank."""
    JACK = 0
    QUEEN = 1
    KING = 2
    
    def __lt__(self, other: "Card") -> bool:
        return self.value < other.value
    
    def __gt__(self, other: "Card") -> bool:
        return self.value > other.value
    
    def __str__(self) -> str:
        return self.name[0]  # J, Q, K
    
    def __repr__(self) -> str:
        return f"Card.{self.name}"


class Action(Enum):
    """Possible actions in Kuhn Poker."""
    CHECK = auto()
    BET = auto()
    CALL = auto()
    FOLD = auto()
    
    def __str__(self) -> str:
        return self.name.lower()
    
    def __repr__(self) -> str:
        return f"Action.{self.name}"


@dataclass
class GameState:
    """
    Represents the complete state of a Kuhn Poker game.
    
    Attributes:
        cards: Tuple of cards dealt to (Player 0, Player 1)
        history: List of actions taken so far
        pot: Current pot size
        bets: Tuple of bets made by each player (including ante)
        current_player: Index of the player to act (0 or 1)
        is_terminal: Whether the game has ended
        winner: Index of the winning player (None if not terminal)
        payoffs: Tuple of payoffs for each player (None if not terminal)
    """
    cards: Tuple[Card, Card] = field(default_factory=lambda: (None, None))
    history: List[Action] = field(default_factory=list)
    pot: int = 2  # Both players ante 1
    bets: Tuple[int, int] = (1, 1)  # Initial antes
    current_player: int = 0
    is_terminal: bool = False
    winner: Optional[int] = None
    payoffs: Optional[Tuple[int, int]] = None
    
    def get_info_set(self, player: int) -> str:
        """
        Returns the information set string for a player.
        
        In Kuhn Poker, a player knows:
        - Their own card
        - The sequence of actions taken
        
        Format: "<card><history>" e.g., "Jcb" = Jack, opponent checked, then bet
        """
        card_str = str(self.cards[player])
        history_str = "".join(str(a)[0] for a in self.history)
        return f"{card_str}{history_str}"
    
    def get_legal_actions(self) -> List[Action]:
        """Returns the list of legal actions for the current player."""
        if self.is_terminal:
            return []
        
        history_len = len(self.history)
        
        if history_len == 0:
            # Player 0's first action
            return [Action.CHECK, Action.BET]
        
        last_action = self.history[-1]
        
        if history_len == 1:
            # Player 1's first action
            if last_action == Action.CHECK:
                return [Action.CHECK, Action.BET]
            else:  # Player 0 bet
                return [Action.FOLD, Action.CALL]
        
        if history_len == 2:
            # Player 0's second action (only happens if P0 checked, P1 bet)
            if self.history[0] == Action.CHECK and self.history[1] == Action.BET:
                return [Action.FOLD, Action.CALL]
        
        return []
    
    def copy(self) -> "GameState":
        """Returns a deep copy of the game state."""
        return deepcopy(self)
    
    def __str__(self) -> str:
        cards_str = f"P0: {self.cards[0]}, P1: {self.cards[1]}" if self.cards[0] else "No cards"
        history_str = " -> ".join(str(a) for a in self.history) if self.history else "No actions"
        status = f"Terminal (Winner: P{self.winner}, Payoffs: {self.payoffs})" if self.is_terminal else f"P{self.current_player} to act"
        return f"GameState({cards_str} | {history_str} | Pot: {self.pot} | {status})"


class KuhnPoker:
    """
    Kuhn Poker game engine.
    
    This class manages the game logic, including:
    - Dealing cards
    - Processing actions
    - Determining winners
    - Calculating payoffs
    
    Example usage:
        game = KuhnPoker()
        state = game.new_game()
        
        while not state.is_terminal:
            actions = state.get_legal_actions()
            action = agent.get_action(state)
            state = game.apply_action(state, action)
        
        print(f"Payoffs: {state.payoffs}")
    """
    
    DECK = [Card.JACK, Card.QUEEN, Card.KING]
    ANTE = 1
    BET_SIZE = 1
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the game engine.
        
        Args:
            seed: Optional random seed for reproducibility
        """
        self.rng = random.Random(seed)
    
    def new_game(self, cards: Optional[Tuple[Card, Card]] = None) -> GameState:
        """
        Start a new game.
        
        Args:
            cards: Optional tuple of cards to deal. If None, cards are dealt randomly.
        
        Returns:
            Initial GameState
        """
        if cards is None:
            # Deal random cards
            deck = self.DECK.copy()
            self.rng.shuffle(deck)
            cards = (deck[0], deck[1])
        
        return GameState(
            cards=cards,
            history=[],
            pot=2 * self.ANTE,
            bets=(self.ANTE, self.ANTE),
            current_player=0,
            is_terminal=False,
            winner=None,
            payoffs=None
        )
    
    def apply_action(self, state: GameState, action: Action) -> GameState:
        """
        Apply an action to the game state.
        
        Args:
            state: Current game state
            action: Action to apply
        
        Returns:
            New GameState after the action
        
        Raises:
            ValueError: If the action is not legal
        """
        if state.is_terminal:
            raise ValueError("Cannot apply action to terminal state")
        
        legal_actions = state.get_legal_actions()
        if action not in legal_actions:
            raise ValueError(f"Illegal action {action}. Legal actions: {legal_actions}")
        
        new_state = state.copy()
        new_state.history.append(action)
        
        current_player = state.current_player
        opponent = 1 - current_player
        
        if action == Action.FOLD:
            # Current player folds, opponent wins
            new_state.is_terminal = True
            new_state.winner = opponent
            # Folding player loses their bet, opponent wins it
            new_state.payoffs = self._calculate_fold_payoffs(new_state, current_player)
            
        elif action == Action.CALL:
            # Current player calls, go to showdown
            new_bets = list(new_state.bets)
            new_bets[current_player] = new_bets[opponent]  # Match opponent's bet
            new_state.bets = tuple(new_bets)
            new_state.pot = sum(new_state.bets)
            new_state.is_terminal = True
            new_state.winner, new_state.payoffs = self._showdown(new_state)
            
        elif action == Action.BET:
            # Current player bets
            new_bets = list(new_state.bets)
            new_bets[current_player] += self.BET_SIZE
            new_state.bets = tuple(new_bets)
            new_state.pot = sum(new_state.bets)
            new_state.current_player = opponent
            
        elif action == Action.CHECK:
            # Check - depends on game state
            if len(new_state.history) == 2:
                # Both players checked - showdown
                new_state.is_terminal = True
                new_state.winner, new_state.payoffs = self._showdown(new_state)
            else:
                # First check - switch player
                new_state.current_player = opponent
        
        return new_state
    
    def _calculate_fold_payoffs(self, state: GameState, folder: int) -> Tuple[int, int]:
        """Calculate payoffs when a player folds."""
        winner = 1 - folder
        # Winner wins the pot, minus their contribution
        # Folder loses their bet
        # Net payoffs relative to initial state
        if folder == 0:
            return (-state.bets[0], state.bets[0])
        else:
            return (state.bets[1], -state.bets[1])
    
    def _showdown(self, state: GameState) -> Tuple[int, Tuple[int, int]]:
        """Determine winner and payoffs at showdown."""
        card0, card1 = state.cards
        
        if card0 > card1:
            winner = 0
            # P0 wins P1's bet
            payoffs = (state.bets[1], -state.bets[1])
        else:
            winner = 1
            # P1 wins P0's bet
            payoffs = (-state.bets[0], state.bets[0])
        
        return winner, payoffs
    
    def get_all_possible_deals(self) -> List[Tuple[Card, Card]]:
        """
        Returns all possible card deals.
        
        Useful for CFR algorithms that need to iterate over all possible games.
        
        Returns:
            List of 6 possible (P0 card, P1 card) tuples
        """
        deals = []
        for c1 in self.DECK:
            for c2 in self.DECK:
                if c1 != c2:
                    deals.append((c1, c2))
        return deals
    
    def play_game(self, agent0, agent1, cards: Optional[Tuple[Card, Card]] = None, 
                  verbose: bool = False) -> Tuple[GameState, List[Tuple[GameState, Action]]]:
        """
        Play a complete game between two agents.
        
        Args:
            agent0: Agent playing as Player 0
            agent1: Agent playing as Player 1
            cards: Optional fixed cards to deal
            verbose: Whether to print game progress
        
        Returns:
            Tuple of (final_state, game_history)
            where game_history is a list of (state, action) pairs
        """
        agents = [agent0, agent1]
        state = self.new_game(cards)
        game_history = []
        
        if verbose:
            print(f"\n{'='*50}")
            print("New Kuhn Poker Game")
            print(f"P0 dealt: {state.cards[0]}, P1 dealt: {state.cards[1]}")
            print(f"{'='*50}")
        
        while not state.is_terminal:
            current_agent = agents[state.current_player]
            action = current_agent.get_action(state)
            
            if verbose:
                print(f"P{state.current_player} ({state.cards[state.current_player]}): {action}")
            
            game_history.append((state.copy(), action))
            state = self.apply_action(state, action)
        
        if verbose:
            print(f"\n{'='*50}")
            print(f"Game Over!")
            print(f"Winner: Player {state.winner} ({state.cards[state.winner]})")
            print(f"Payoffs: P0={state.payoffs[0]:+d}, P1={state.payoffs[1]:+d}")
            print(f"{'='*50}\n")
        
        return state, game_history
    
    def play_match(self, agent0, agent1, num_games: int, 
                   alternate_positions: bool = True, verbose: bool = False) -> dict:
        """
        Play a match of multiple games between two agents.
        
        Args:
            agent0: First agent
            agent1: Second agent
            num_games: Number of games to play
            alternate_positions: Whether to alternate starting positions
            verbose: Whether to print individual game results
        
        Returns:
            Dictionary with match statistics
        """
        total_payoffs = [0, 0]
        wins = [0, 0]
        games_played = 0
        
        for i in range(num_games):
            if alternate_positions and i % 2 == 1:
                # Swap positions
                state, _ = self.play_game(agent1, agent0, verbose=verbose)
                # Swap payoffs back
                payoffs = (state.payoffs[1], state.payoffs[0])
                winner = 1 - state.winner
            else:
                state, _ = self.play_game(agent0, agent1, verbose=verbose)
                payoffs = state.payoffs
                winner = state.winner
            
            total_payoffs[0] += payoffs[0]
            total_payoffs[1] += payoffs[1]
            wins[winner] += 1
            games_played += 1
        
        return {
            "games_played": games_played,
            "wins": {"agent0": wins[0], "agent1": wins[1]},
            "total_payoffs": {"agent0": total_payoffs[0], "agent1": total_payoffs[1]},
            "avg_payoffs": {
                "agent0": total_payoffs[0] / games_played,
                "agent1": total_payoffs[1] / games_played
            }
        }


# Game tree utilities for CFR algorithms

class GameTreeNode:
    """
    Represents a node in the Kuhn Poker game tree.
    
    This is useful for CFR algorithms that need to traverse the complete game tree.
    """
    
    def __init__(self, state: GameState, parent: Optional["GameTreeNode"] = None):
        self.state = state
        self.parent = parent
        self.children: dict[Action, "GameTreeNode"] = {}
    
    @property
    def is_terminal(self) -> bool:
        return self.state.is_terminal
    
    @property
    def is_chance(self) -> bool:
        # In our representation, chance nodes are handled separately
        return False
    
    @property
    def player(self) -> int:
        return self.state.current_player
    
    @property
    def info_set(self) -> str:
        return self.state.get_info_set(self.player)
    
    def __str__(self) -> str:
        return f"TreeNode({self.state})"


def build_game_tree(game: KuhnPoker, cards: Tuple[Card, Card]) -> GameTreeNode:
    """
    Build the complete game tree for a given card deal.
    
    Args:
        game: KuhnPoker game engine
        cards: Tuple of cards dealt to each player
    
    Returns:
        Root node of the game tree
    """
    root_state = game.new_game(cards)
    root = GameTreeNode(root_state)
    
    def build_subtree(node: GameTreeNode):
        if node.is_terminal:
            return
        
        for action in node.state.get_legal_actions():
            new_state = game.apply_action(node.state, action)
            child = GameTreeNode(new_state, parent=node)
            node.children[action] = child
            build_subtree(child)
    
    build_subtree(root)
    return root


def get_all_info_sets(game: KuhnPoker) -> dict:
    """
    Get all possible information sets in Kuhn Poker.
    
    Returns:
        Dictionary mapping info_set strings to lists of (legal_actions)
    """
    info_sets = {}
    
    for cards in game.get_all_possible_deals():
        root = build_game_tree(game, cards)
        
        def traverse(node: GameTreeNode):
            if node.is_terminal:
                return
            
            info_set = node.info_set
            if info_set not in info_sets:
                info_sets[info_set] = node.state.get_legal_actions()
            
            for child in node.children.values():
                traverse(child)
        
        traverse(root)
    
    return info_sets

