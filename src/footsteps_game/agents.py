"""Agent class and related functionality"""

import logging
from dataclasses import dataclass
from typing import Optional, Set, List

from .strategy import Strategy
from .config import CONFIG


class Agent:
    """Represents an AI player in the game"""

    _id_counter = 1

    def __init__(
        self,
        strategy: Optional[Strategy] = None,
        genealogy: Optional[Set[str]] = None,
        population_id: int = 1,
        parents: Optional[List[str]] = None,
    ):
        self.id = f"A-{Agent._id_counter:06d}"
        Agent._id_counter += 1
        self.population_id = population_id
        self.strategy = strategy if strategy else Strategy()
        self.genealogy = genealogy if genealogy else {self.id}
        self.fitness = 500
        self.game_counter = 0
        self.games_played_this_generation = 0
        self.games_played = 0
        self.initial_fitness = self.fitness
        self.sum_fitness_change = 0
        self.sum_fitness_change_squared = 0
        self.parents = parents if parents else []
        self.offspring_count = 0

    def get_bid(self, point_budget: int, opponent_point_budget: int) -> int:
        """Determines bid amount based on current game state"""
        game_state = {
            "my_points": point_budget,
            "opponent_points": opponent_point_budget,
        }
        bid_fraction = self.strategy.get_bid_fraction(game_state)
        bid_constant = self.strategy.bid_params["bid_constant"]
        bid = max(1, int(point_budget * bid_fraction) + bid_constant)
        return bid

    def get_move(self, my_position, opponent_position, board_size, goal, opponent_goal):
        """Determines next move based on current game state"""
        game_state = {
            "my_position": my_position,
            "opponent_position": opponent_position,
            "goal": goal,
            "opponent_goal": opponent_goal,
            "board_size": board_size,
        }
        direction = self.strategy.evaluate_tree(self.strategy.decision_tree, game_state)

        x, y = my_position
        potential_moves = {
            "u": (x, y - 1),
            "d": (x, y + 1),
            "l": (x - 1, y),
            "r": (x + 1, y),
            "ul": (x - 1, y - 1),
            "ur": (x + 1, y - 1),
            "dl": (x - 1, y + 1),
            "dr": (x + 1, y + 1),
        }

        # Get valid moves
        valid_moves = [
            move_pos
            for move_pos in potential_moves.values()
            if self.is_valid_move(move_pos, opponent_position, board_size)
        ]

        # Use strategy's move if valid
        move = potential_moves.get(direction)
        if move in valid_moves:
            return move
        elif valid_moves:
            return random.choice(valid_moves)
        else:
            return my_position

    def is_valid_move(self, position, opponent_position, board_size):
        """Checks if a move is valid"""
        x, y = position
        return (
            0 <= x < board_size
            and 0 <= y < board_size
            and position != opponent_position
        )

    def increment_game_counter(self):
        """Increments the game counter"""
        self.game_counter += 1
        logging.info(
            f"Agent {self.id} from Population {self.population_id} "
            f"game_counter incremented to {self.game_counter}"
        )

    def reset_generation_counter(self):
        """Resets the per-generation game counters and fitness tracking"""
        self.games_played_this_generation = 0
        self.sum_fitness_change = 0
        self.sum_fitness_change_squared = 0
        logging.debug(
            f"Agent {self.id} from Population {self.population_id} "
            "generation counters reset."
        )
