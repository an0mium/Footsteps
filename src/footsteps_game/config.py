"""Configuration settings for the Footsteps game."""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Tuple

CONFIG = {
    "BOARD_SIZE": 11,
    "STARTING_POINTS": 5040,  # 12² * 5 * 7
    "ELITE_PERCENTAGE": 0.05,
    "MIN_MUTATION_RATE": 0.005,
    "MAX_MUTATION_RATE": 0.5,
    "BASE_MUTATION_RATE": 0.1,
    "VISUALIZATION_SAMPLE_RATE": 0.0001,  # 0.05%
    "MAX_TURNS": 180,
    "MAX_GENERATIONS": 120,
    "NUM_POPULATIONS": 12,
    "POPULATION_SIZE": 2520,
}

COLORS = {
    "WHITE": "#FFFFFF",
    "BLACK": "#000000",
    "BLUE": "#3498db",
    "RED": "#e74c3c",
    "GREEN": "#2ecc71",
    "YELLOW": "#f1c40f",
    "GRAY": "#95a5a6",
}


class Color(Enum):
    """Enumeration for player colors"""

    WHITE = "white"
    BLACK = "black"


class GamePhase(Enum):
    """Enumeration for game phases"""

    BIDDING = auto()
    ACTION = auto()
    GAME_OVER = auto()


@dataclass
class Position:
    """Data class for board positions"""

    top_value: int = 0
    top_color: Optional[Color] = None
    bottom_value: int = 0
    bottom_color: Optional[Color] = None

    def is_empty(self) -> bool:
        return self.top_value == 0 and self.bottom_value == 0


@dataclass
class Bid:
    """Data class for player bids"""

    white: Optional[int] = None
    black: Optional[int] = None

    def is_complete(self) -> bool:
        return self.white is not None and self.black is not None

    def get_winner(self) -> Optional[Color]:
        if not self.is_complete():
            return None
        if self.white > self.black:
            return Color.WHITE
        if self.black > self.white:
            return Color.BLACK
        return Color.WHITE  # White wins ties
