#!/usr/bin/env python3
"""
Footsteps Strategy Game with Agents Competing
Evolving Strategies
Elite Matching System
Author: scarmani
Date: Nov 3rd 2024
"""

import cProfile
import io
import logging
import math
import platform
import pstats
import random
import select
import sys
import threading
import time
import unittest
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, auto
from logging.handlers import RotatingFileHandler
from textwrap import wrap as wrap_text
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Set matplotlib backend
matplotlib.use("TkAgg")

# Platform-specific imports
if platform.system() == "Windows":
    import msvcrt
else:
    import termios
    import tty


# Configure logging
def configure_logging():
    """Configure logging to write to file only"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Add file handler
    file_handler = RotatingFileHandler(
        "game_simulation.log",
        maxBytes=500 * 1024 * 1024,  # 500 MB
        backupCount=500,
        delay=True,
    )

    # Create and set formatter
    formatter = logging.Formatter(
        '{"time": "%(asctime)s", '
        '"level": "%(levelname)s", '
        '"message": "%(message)s"}'
    )

    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


# Global Configuration
CONFIG = {
    "BOARD_SIZE": 11,
    "STARTING_POINTS": 144,  # 12²
    "ELITE_PERCENTAGE": 0.05,
    "MIN_MUTATION_RATE": 0.005,
    "MAX_MUTATION_RATE": 0.2,
    "BASE_MUTATION_RATE": 0.05,
    "VISUALIZATION_SAMPLE_RATE": 0.0001,  # 0.05%
    "MAX_TURNS": 180,
    "MAX_GENERATIONS": 120,
    "NUM_POPULATIONS": 12,
    "POPULATION_SIZE": 840,
}

# Color scheme for visualization
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


def format_bid_range(bid_range):
    """Formats a bid range list into a percentage string"""
    return f"{bid_range[0]*100:.2f}% - {bid_range[1]*100:.2f}%"


class KeyListener(threading.Thread):
    """Thread class for handling keyboard input"""

    def __init__(self, report_event, continue_event):
        super().__init__()
        self.daemon = True
        self.report_event = report_event
        self.continue_event = continue_event
        self.stop_flag = False
        self.lock = threading.Lock()
        self.platform = platform.system()

        if self.platform != "Windows":
            self.fd = sys.stdin.fileno()
            self.old_settings = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)

    def run(self):
        try:
            while not self.stop_flag:
                if self.is_data():
                    ch = self.read_char()
                    if ch:
                        if not self.report_event.is_set():
                            self.report_event.set()
                            print("\nReport Requested. Generating report...")
                            logging.info("Report Requested. Generating report...")
                        else:
                            self.continue_event.set()
                            print("\nContinuing execution...")
                            logging.info("Continuing execution...")
                time.sleep(0.1)
        finally:
            if self.platform != "Windows":
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    def is_data(self):
        if self.platform == "Windows":
            return msvcrt.kbhit()
        else:
            return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

    def read_char(self):
        if self.platform == "Windows":
            return msvcrt.getch().decode("utf-8")
        else:
            return sys.stdin.read(1)


class Strategy:
    """Handles decision-making logic and mutation for agents"""

    def __init__(self, decision_tree=None, bid_params=None):
        if decision_tree is None:
            self.decision_tree = self.generate_random_tree()
        else:
            self.decision_tree = decision_tree
        if bid_params is None:
            self.bid_params = self.generate_random_bid_params()
        else:
            self.bid_params = bid_params

    def generate_random_bid_params(self):
        """Generates random bidding parameters"""
        thresholds = sorted(random.sample(range(-50000, 50000), 2))
        low_bid_range = sorted([random.uniform(0.001, 0.05), random.uniform(0.01, 0.1)])
        medium_bid_range = sorted(
            [random.uniform(0.01, 0.15), random.uniform(0.02, 0.2)]
        )
        high_bid_range = sorted([random.uniform(0.02, 0.2), random.uniform(0.05, 0.3)])
        bid_constant = random.randint(-100, 100)
        return {
            "thresholds": thresholds,
            "low_bid_range": low_bid_range,
            "medium_bid_range": medium_bid_range,
            "high_bid_range": high_bid_range,
            "bid_constant": bid_constant,
        }

    def mutate(self, mutation_rate=0.05):
        """Mutates the strategy with given mutation rate"""
        mutation_rate = max(0, min(mutation_rate, 1))

        def mutate_node(node):
            if isinstance(node, dict):
                if random.random() < mutation_rate:
                    node["condition"] = random.choice(
                        [
                            "opponent_closer",
                            "edge_near",
                            "random",
                            "opponent_left",
                            "opponent_right",
                            "opponent_above",
                            "opponent_below",
                            "agent_on_opponent_goal",
                            "goal_below_agent",
                            "goal_above_agent",
                            "goal_left_agent",
                            "goal_right_agent",
                            "goal_reachable_in_one_move",
                        ]
                    )
                node["true"] = mutate_node(node["true"])
                node["false"] = mutate_node(node["false"])
            else:
                if random.random() < mutation_rate:
                    node = random.choice(["u", "d", "l", "r", "ul", "ur", "dl", "dr"])
            return node

        self.decision_tree = mutate_node(self.decision_tree)

        # Mutate bid parameters
        if random.random() < mutation_rate * 5:
            index = random.choice([0, 1])
            change = round(random.randint(-50, 50) * (1 + 20 * mutation_rate))
            self.bid_params["thresholds"][index] += change
            self.bid_params["thresholds"] = sorted(self.bid_params["thresholds"])

        for param in ["low_bid_range", "medium_bid_range", "high_bid_range"]:
            if random.random() < mutation_rate * 5:
                index = random.choice([0, 1])
                change = random.uniform(-0.05, 0.05) * (1 + 20 * mutation_rate)
                self.bid_params[param][index] += change
                self.bid_params[param][index] = min(
                    max(self.bid_params[param][index], 0.001), 0.5
                )
                self.bid_params[param] = sorted(self.bid_params[param])

        if random.random() < mutation_rate * 5:
            change = round(random.randint(-10, 10) * (1 + 20 * mutation_rate))
            self.bid_params["bid_constant"] += change
            self.bid_params["bid_constant"] = min(
                max(self.bid_params["bid_constant"], -500), 500
            )

    def get_bid_fraction(self, game_state):
        """Determines bid fraction based on game state"""
        thresholds = self.bid_params["thresholds"]
        low_bid_range = self.bid_params["low_bid_range"]
        medium_bid_range = self.bid_params["medium_bid_range"]
        high_bid_range = self.bid_params["high_bid_range"]
        my_points = game_state["my_points"]
        opp_points = game_state["opponent_points"]
        point_difference = my_points - opp_points

        if point_difference > thresholds[1]:
            bid_fraction = random.uniform(*low_bid_range)
        elif point_difference < thresholds[0]:
            bid_fraction = random.uniform(*high_bid_range)
        else:
            bid_fraction = random.uniform(*medium_bid_range)

        total_points = my_points + opp_points + 1
        normalized_diff = point_difference / total_points
        bid_fraction += max(0.001, 0.005 - 0.005 * normalized_diff)
        return min(max(bid_fraction, 0.01), 0.5)

    def generate_random_tree(self, depth=6, min_depth=4):
        """Generates a random decision tree"""
        if depth <= 0:
            return random.choice(["u", "d", "l", "r", "ul", "ur", "dl", "dr"])

        if depth > min_depth:
            condition = random.choice(
                [
                    "opponent_closer",
                    "edge_near",
                    "random",
                    "opponent_left",
                    "opponent_right",
                    "opponent_above",
                    "opponent_below",
                    "agent_on_opponent_goal",
                    "goal_below_agent",
                    "goal_above_agent",
                    "goal_left_agent",
                    "goal_right_agent",
                    "goal_reachable_in_one_move",
                ]
            )
        else:
            condition = random.choice(
                [
                    "opponent_closer",
                    "edge_near",
                    "opponent_left",
                    "opponent_right",
                    "opponent_above",
                    "opponent_below",
                    "agent_on_opponent_goal",
                    "random",
                ]
            )

        true_branch = self.generate_random_tree(depth - 1, min_depth)
        false_branch = self.generate_random_tree(depth - 1, min_depth)
        return {"condition": condition, "true": true_branch, "false": false_branch}

    def evaluate_condition(self, condition, game_state):
        """Evaluates a condition based on game state"""
        if condition == "opponent_closer":
            my_dist = self.manhattan_distance(
                game_state["my_position"], game_state["goal"]
            )
            opp_dist = self.manhattan_distance(
                game_state["opponent_position"], game_state["goal"]
            )
            return opp_dist < my_dist
        elif condition == "edge_near":
            x, y = game_state["my_position"]
            board_size = game_state["board_size"]
            return x == 0 or y == 0 or x == board_size - 1 or y == board_size - 1
        elif condition == "random":
            return random.choice([True, False])
        elif condition == "opponent_left":
            return game_state["opponent_position"][0] < game_state["my_position"][0]
        elif condition == "opponent_right":
            return game_state["opponent_position"][0] > game_state["my_position"][0]
        elif condition == "opponent_above":
            return game_state["opponent_position"][1] > game_state["my_position"][1]
        elif condition == "opponent_below":
            return game_state["opponent_position"][1] < game_state["my_position"][1]
        elif condition == "agent_on_opponent_goal":
            return game_state["my_position"] == game_state["opponent_goal"]
        elif condition == "goal_below_agent":
            return game_state["goal"][1] < game_state["my_position"][1]
        elif condition == "goal_above_agent":
            return game_state["goal"][1] > game_state["my_position"][1]
        elif condition == "goal_left_agent":
            return game_state["goal"][0] < game_state["my_position"][0]
        elif condition == "goal_right_agent":
            return game_state["goal"][0] > game_state["my_position"][0]
        elif condition == "goal_reachable_in_one_move":
            return self.is_adjacent(game_state["my_position"], game_state["goal"])
        return False

    def evaluate_tree(self, node, game_state, depth=0):
        """Evaluates the decision tree to make a move"""
        if depth > 10:
            logging.warning("Maximum recursion depth reached in evaluate_tree.")
            return "u"
        if isinstance(node, dict):
            condition = node["condition"]
            result = self.evaluate_condition(condition, game_state)
            if result:
                return self.evaluate_tree(node["true"], game_state, depth + 1)
            else:
                return self.evaluate_tree(node["false"], game_state, depth + 1)
        else:
            return node

    def manhattan_distance(self, pos1, pos2):
        """Calculates Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def is_adjacent(self, pos1, pos2):
        """Determines if two positions are adjacent"""
        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])
        return max(dx, dy) == 1

    def crossover(self, other_strategy, min_depth=4):
        """Performs crossover with another strategy"""
        tree1 = deepcopy(self.decision_tree)
        tree2 = deepcopy(other_strategy.decision_tree)

        # Select random crossover points
        subtree1, parent1, key1 = self.get_random_subtree(tree1)
        subtree2, parent2, key2 = self.get_random_subtree(tree2)

        # Swap subtrees
        if parent1 and key1:
            parent1[key1] = subtree2
        else:
            tree1 = subtree2

        if parent2 and key2:
            parent2[key2] = subtree1
        else:
            tree2 = subtree1

        # Crossover bid parameters
        new_bid_params = {}
        for key in self.bid_params:
            if key == "bid_constant":
                new_bid_params[key] = int(
                    (self.bid_params[key] + other_strategy.bid_params[key]) / 2
                )
            else:
                new_bid_params[key] = [
                    random.choice(
                        [
                            self.bid_params[key][0],
                            other_strategy.bid_params[key][0],
                        ]
                    ),
                    random.choice(
                        [
                            self.bid_params[key][1],
                            other_strategy.bid_params[key][1],
                        ]
                    ),
                ]
                new_bid_params[key] = sorted(new_bid_params[key])

        # Create child strategy
        child_tree = random.choice([tree1, tree2])
        child_strategy = Strategy(child_tree, new_bid_params)
        child_strategy.enforce_minimum_depth(min_depth)

        return child_strategy

    def get_random_subtree(self, node, parent=None, key=None):
        """Gets a random subtree from the decision tree"""
        if isinstance(node, dict):
            if random.random() < 0.5:
                return node, parent, key
            else:
                branch = random.choice(["true", "false"])
                return self.get_random_subtree(node[branch], node, branch)
        else:
            return node, parent, key

    def enforce_minimum_depth(self, min_depth=4):
        """Ensures the decision tree meets minimum depth requirements"""
        current_depth = self.calculate_tree_depth(self.decision_tree)
        if current_depth < min_depth:
            self.decision_tree = self.generate_random_tree(
                depth=min_depth, min_depth=min_depth
            )

    def calculate_tree_depth(self, node):
        """Calculates the depth of the decision tree"""
        if isinstance(node, dict):
            return 1 + max(
                self.calculate_tree_depth(node["true"]),
                self.calculate_tree_depth(node["false"]),
            )
        return 1


class Agent:
    """Represents an AI player in the game"""

    _id_counter = 1

    def __init__(self, strategy=None, genealogy=None, population_id=1, parents=None):
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

    def get_bid(self, point_budget, opponent_point_budget):
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

    def asexual_offspring(self):
        """Creates an offspring through asexual reproduction"""
        offspring_strategy = deepcopy(self.strategy)
        offspring_strategy.mutate(mutation_rate=0.05)
        offspring = Agent(
            strategy=offspring_strategy,
            genealogy=self.genealogy,
            population_id=self.population_id,
            parents=[self.id],
        )
        self.offspring_count += 1
        self.fitness = self.fitness * 99 // 100

        logging.info(
            f"Asexual Offspring Created: {offspring.id} inherits genealogy "
            f"{offspring.genealogy}"
        )
        return offspring

    def sexual_offspring(self, other_agent):
        """Creates an offspring through sexual reproduction"""
        child_strategy = self.strategy.crossover(other_agent.strategy)
        child_strategy.mutate(mutation_rate=0.05)
        child_genealogy = self.genealogy.union(other_agent.genealogy)
        child = Agent(
            strategy=child_strategy,
            genealogy=child_genealogy,
            population_id=self.population_id,
            parents=[self.id, other_agent.id],
        )
        self.offspring_count += 1
        other_agent.offspring_count += 1

        self.fitness = self.fitness * 99 // 100
        other_agent.fitness = other_agent.fitness * 99 // 100

        logging.info(
            f"Sexual Offspring Created: {child.id} inherits genealogy "
            f"{child.genealogy}"
        )
        return child

    def increment_game_counter(self):
        """Increments the Agent game counter"""
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

    def estimate_fitness_change(self):
        """Estimates fitness change parameters using Bayesian methods"""
        if self.games_played == 0:
            mean_est = 0
            std_dev = 1
        else:
            mean_est = self.sum_fitness_change / (self.games_played + 12)
            variance_est = (
                (
                    self.sum_fitness_change_squared
                    - (self.sum_fitness_change**2) / (self.games_played + 12)
                )
                / (self.games_played + 12 - 1)
                if self.games_played + 12 - 1 > 0
                else 12
            )
            std_dev = math.sqrt(variance_est) if variance_est > 0 else 12

        min_est = mean_est - std_dev
        max_est = mean_est + std_dev
        return (min_est, mean_est, max_est)


"""Handles fixed-position terminal statistics display"""


class StatsDisplay:
    """Handles fixed-position terminal statistics display"""

    def __init__(self):
        self.display_buffer = {}
        self.original_stdout = sys.stdout
        self.log_buffer = []
        self.is_updating = False
        self.last_update_time = 0
        self.min_update_interval = 0.001  # Update every 1ms
        self._init_display(clear_screen=True)

        # Initialize game counter display
        self.total_games_played = 0
        self._create_game_counter_section()

    def _create_game_counter_section(self):
        """Creates a dedicated section for the game counter"""
        self._create_section("Game Counter", 0, 0, 50, 3)
        self._update_game_counter()

    def _update_game_counter(self):
        """Updates the game counter display with proper logging"""
        counter_text = [f"Total Games Played: {self.total_games_played}"]
        self._update_section("Game Counter", counter_text)
        logging.info(
            f"Game Counter Updated: {counter_text[0]}", extra={"section": "Counter"}
        )

    def increment_game_counter(self):
        """Increments and updates the Total game counter with proper logging"""
        self.total_games_played += 1
        self._update_game_counter()
        logging.info(
            f"Total Games Counter Incremented to: {self.total_games_played}",
            extra={"section": "Counter"},
        )

    def _init_display(self, clear_screen=False):
        """Initialize the display sections with optimized layout"""
        if clear_screen:
            print("\033[2J\033[H")  # Clear screen only once at start
        print("\033[?25l")  # Hide cursor

        # Create sections with adjusted sizes
        self._create_section("Population Stats", 0, 3, 180, 87)
        self._create_section("Metapopulation Stats", 181, 3, 182, 42)
        self._create_section("Current Game Stats", 181, 46, 182, 44)
        self._create_section("Log Output", 0, 71, 364, 10)

        # Force immediate display refresh
        sys.stdout.flush()

    def update_all(self):
        """Forces update of all sections"""
        for section in self.display_buffer:
            content = self.display_buffer[section].get("content", [])
            self._update_section(section, content)
        sys.stdout.flush()

    def _safe_update_log(self):
        """Update the log section with improved buffer management"""
        if len(self.log_buffer) > 28:
            self.log_buffer = self.log_buffer[-28:]

        if not self.is_updating:
            try:
                self.is_updating = True
                colored_logs = []
                for line in self.log_buffer:
                    if "Error" in line or "error" in line:
                        colored_logs.append(f"\033[91m{line}\033[0m")  # Red
                    elif "Warning" in line or "warning" in line:
                        colored_logs.append(f"\033[93m{line}\033[0m")  # Yellow
                    else:
                        colored_logs.append(line)
                self._update_section("Log Output", colored_logs)
                self._log_display_update("Log Output", colored_logs)
            finally:
                self.is_updating = False
                self.original_stdout.flush()

    def _log_display_update(self, section_name, content):
        """Logs display content updates"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"Display Update [{timestamp}] - Section: {section_name}\n"
        log_message += "\n".join(content)
        logging.info(log_message, extra={"section": section_name})

    def _write_at(self, x, y, text):
        """Write text at specified position"""
        self.original_stdout.write(f"\033[{y+1};{x+1}H{text}")

    def cleanup(self):
        """Restore terminal state"""
        self._restore_output()
        print("\033[?25h")  # Show cursor
        print("\033[H")  # Move cursor to home position

    def _update_section(self, section, lines):
        """Update content of a section with improved logging"""
        if section not in self.display_buffer:
            return

        try:
            section_info = self.display_buffer[section]
            x, y = section_info["x"], section_info["y"]
            width, height = section_info["width"], section_info["height"]

            # Store content and log complete update
            self.display_buffer[section]["content"] = lines

            # Log the full content being displayed
            log_message = f"Display Section [{section}] Content:\n"
            log_message += "\n".join(str(line) for line in lines)
            logging.info(log_message)

            # Update display line by line
            for i, line in enumerate(lines, 1):
                if i >= height - 1:
                    break
                padded_line = str(line)[: width - 2].ljust(width - 2)
                self._write_at(x + 1, y + i, padded_line)

            # Force immediate refresh
            self.original_stdout.flush()

        except Exception as e:
            logging.error(f"Error updating section {section}: {e}")

    def _update_log_section(self):
        """Update the log section with recent output"""
        if len(self.log_buffer) > 8:  # Keep only last 8 lines
            self.log_buffer = self.log_buffer[-8:]

        # Use original stdout for the actual update
        old_stdout = sys.stdout
        sys.stdout = self.original_stdout
        try:
            self._update_section("Log Output", self.log_buffer)
        finally:
            sys.stdout = old_stdout

    def _create_section(self, title, x, y, width, height):
        """Create a section in the display"""
        if not self.is_updating:
            self.is_updating = True
            try:
                # Draw top border
                self._write_at(x, y, "+" + "-" * (width - 2) + "+")

                # Draw title
                title_pos = (width - len(title) - 2) // 2
                self._write_at(x + title_pos, y, f" {title} ")

                # Draw sides and empty content
                for i in range(1, height - 1):
                    self._write_at(x, y + i, "|" + " " * (width - 2) + "|")

                # Draw bottom border
                self._write_at(x, y + height - 1, "+" + "-" * (width - 2) + "+")

                # Store section info
                self.display_buffer[title] = {
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "content": [],
                }
            finally:
                self.is_updating = False

    def _redirect_output(self):
        """Redirect stdout to prevent scrolling"""

        class StatsDisplayWriter:
            def __init__(self, stats_display):
                self.stats_display = stats_display

            def write(self, text):
                if text.strip():  # Only handle non-empty text
                    self.stats_display.log_buffer.append(text)
                    # Update the log section if it exists
                    if "Log Output" in self.stats_display.display_buffer:
                        self.stats_display._update_log_section()

            def flush(self):
                pass

        sys.stdout = StatsDisplayWriter(self)

    def _restore_output(self):
        """Restore original stdout"""
        sys.stdout = self.original_stdout

    def update_population_stats(self, population):
        """Update population statistics with expanded information"""
        stats = []

        # Basic population info header
        stats.extend(
            [
                f"Population {population.population_id} Status",
                "=" * 118,  # Separator line
                f"Size: {len(population.agents)} | "
                f"Average Fitness: {population.calculate_average_fitness():.2f} | "
                f"Mutation Rate: {population.mutation_rate:.4f} | "
                f"Elite %: {population.elite_percentage*100:.1f}%",
                "",
            ]
        )

        # Calculate and display average bidding parameters
        avg_bid_params = self._calculate_average_bid_params(population)
        stats.extend(
            [
                "Average Bidding Strategy:",
                f"  Thresholds: {avg_bid_params['thresholds']}",
                f"  Low Bid Range: {format_bid_range(avg_bid_params['low_bid_range'])}",
                f"  Medium Bid Range: {format_bid_range(avg_bid_params['medium_bid_range'])}",
                f"  High Bid Range: {format_bid_range(avg_bid_params['high_bid_range'])}",
                f"  Bid Constant: {avg_bid_params['bid_constant']:.2f}",
                "",
            ]
        )

        # Population-wide statistics
        avg_experience = sum(agent.game_counter for agent in population.agents) / len(
            population.agents
        )
        avg_offspring = sum(agent.offspring_count for agent in population.agents) / len(
            population.agents
        )
        avg_genealogy = sum(
            population.genealogy_counts.get(agent.id, 0) for agent in population.agents
        ) / len(population.agents)

        stats.extend(
            [
                "Population Metrics:",
                f"  Average Experience: {avg_experience:.2f} games",
                f"  Average Offspring: {avg_offspring:.2f}",
                f"  Average Genealogy Count: {avg_genealogy:.2f}",
                f"  Strategy Diversity: {population.calculate_diversity():.4f}",
                "",
            ]
        )

        # Most successful agents section
        most_fit = population.get_most_fit_agent()
        stats.extend(
            [
                "Most Fit Agent:",
                f"  ID: {most_fit.id}",
                f"  Fitness: {most_fit.fitness}",
                f"  Games: {most_fit.game_counter}",
                f"  Offspring: {most_fit.offspring_count}",
                f"  Genealogy Impact: {population.genealogy_counts.get(most_fit.id, 0)}",
                self._format_agent_strategy(most_fit),
                "",
            ]
        )

        # Most experienced agent
        most_exp = max(population.agents, key=lambda a: a.game_counter)
        stats.extend(
            [
                "Most Experienced Agent:",
                f"  ID: {most_exp.id}",
                f"  Fitness: {most_exp.fitness}",
                f"  Games: {most_exp.game_counter}",
                f"  Offspring: {most_exp.offspring_count}",
                f"  Genealogy Impact: {population.genealogy_counts.get(most_exp.id, 0)}",
                self._format_agent_strategy(most_exp),
                "",
            ]
        )

        # Top seminal agents
        stats.append("Top Seminal Agents:")
        seminal_agents = sorted(
            population.genealogy_counts.items(), key=lambda x: x[1], reverse=True
        )[:8]
        for agent_id, count in seminal_agents:
            agent = population.get_agent_by_id(agent_id)
            if agent:
                stats.append(
                    f"  {agent_id}: {count} descendants | "
                    f"Fitness: {agent.fitness} | "
                    f"Games: {agent.game_counter}"
                )
        stats.append("")

        # Oldest agent
        oldest_agent = min(population.agents, key=lambda a: int(a.id.split("-")[1]))
        stats.extend(
            [
                "Oldest Agent:",
                f"  ID: {oldest_agent.id}",
                f"  Fitness: {oldest_agent.fitness}",
                f"  Games: {oldest_agent.game_counter}",
                f"  Offspring: {oldest_agent.offspring_count}",
                f"  Genealogy Count: {population.genealogy_counts.get(oldest_agent.id, 0)}",
                "",
            ]
        )

        # Most experienced agent
        most_exp = max(population.agents, key=lambda a: a.game_counter)
        stats.extend(
            [
                "Most Experienced Agent:",
                f"  ID: {most_exp.id}",
                f"  Fitness: {most_exp.fitness}",
                f"  Games: {most_exp.game_counter}",
                f"  Offspring: {most_exp.offspring_count}",
                f"  Genealogy Count: {population.genealogy_counts.get(most_exp.id, 0)}",
                "",
            ]
        )

        # Most prolific agent
        prolific = max(population.agents, key=lambda a: a.offspring_count)
        stats.extend(
            [
                "Most Prolific Agent:",
                f"  ID: {prolific.id}",
                f"  Fitness: {prolific.fitness}",
                f"  Games: {prolific.game_counter}",
                f"  Offspring: {prolific.offspring_count}",
                f"  Genealogy Count: {population.genealogy_counts.get(prolific.id, 0)}",
                "",
            ]
        )

        # Most/Least fit agents
        most_fit = population.get_most_fit_agent()
        least_fit = population.get_least_fit_agent()
        stats.extend(
            [
                "Most Fit Agent:",
                f"  ID: {most_fit.id}",
                f"  Fitness: {most_fit.fitness}",
                f"  Games: {most_fit.game_counter}",
                "",
                "Least Fit Agent:",
                f"  ID: {least_fit.id}",
                f"  Fitness: {least_fit.fitness}",
                f"  Games: {least_fit.game_counter}",
            ]
        )

        section_name = f"Population {population.population_id} Stats"
        if section_name in self.display_buffer:
            self._update_section(section_name, stats)
            self._log_display_update(section_name, stats)

    def update_current_game_stats(self, game):
        """Update current game statistics with comprehensive information"""
        stats = []

        # Game header and basic info
        stats.extend(
            [
                f"Game #{game.game_number} - {game.game_type}",
                "=" * 180,
                f"Turn: {len(game.history)} | "
                f"Status: {game.outcome_code if game.outcome_code else 'In Progress'} | "
                f"Reason: {game.winning_reason if game.winning_reason else 'N/A'}",
                "",
            ]
        )

        # Player details
        for player_key in ["player1", "player2"]:
            player = game.players[player_key]
            stats.extend(
                [
                    f"{player_key.title()}:",
                    f"  ID: {player.id} (Population {player.population_id})",
                    f"  Position: {game.positions[player_key]} → Goal: {game.goals[player_key]}",
                    f"  Points: {game.point_budgets[player_key]} | Fitness: {player.fitness}",
                    f"  Games: {player.game_counter} | Offspring: {player.offspring_count}",
                    "  Strategy: " + self._format_agent_strategy(player),
                    "",
                ]
            )

        self._update_section("Current Game Stats", stats)

    def update_game_stats(self, game):
        """Update current game statistics display"""
        stats = [
            f"Game #{game.game_number}",
            f"Type: {game.game_type}",
            f"Turn: {len(game.history)}",
            f"Outcome: {game.outcome_code if game.outcome_code else 'In Progress'}",
            f"Reason: {game.winning_reason if game.winning_reason else 'N/A'}",
            "",
            "Points:",
            f" P1: {game.point_budgets['player1']}",
            f" P2: {game.point_budgets['player2']}",
            "",
            "Position:",
            f" P1: {game.positions['player1']}",
            f" P2: {game.positions['player2']}",
            "",
            "Goals:",
            f" P1: {game.goals['player1']}",
            f" P2: {game.goals['player2']}",
        ]

        self._update_section("Current Game", stats)

    def update_agent_stats(self, agent, population, is_player_one=True):
        """Update agent statistics display with comprehensive info"""
        prefix = "Player 1" if is_player_one else "Player 2"
        min_est, mean_est, max_est = agent.estimate_fitness_change()

        if population:
            genealogy_count = population.genealogy_counts.get(agent.id, 0)
        else:
            genealogy_count = 0

        stats = [
            f"{prefix} Details:",
            f"ID: {agent.id}",
            f"Population: {agent.population_id}",
            f"Fitness: {agent.fitness}",
            f"Games Played: {agent.game_counter}",
            f"Offspring: {agent.offspring_count}",
            f"Genealogy Count: {genealogy_count}",
            "",
            "Est. Fitness Change/Game:",
            f" Min: {min_est:.2f}",
            f" Mean: {mean_est:.2f}",
            f" Max: {max_est:.2f}",
            "",
            "Strategy Summary:",
            "Movement: " + self._get_strategy_summary(agent.strategy),
            "Bid Range: " + self._get_bid_summary(agent.strategy),
        ]

        self._update_section("Agent Stats", stats)

    def update_metapopulation_stats(self, meta_population):
        """Update metapopulation statistics with comprehensive metrics"""
        stats = []

        # Header with overall statistics
        total_agents = sum(len(p.agents) for p in meta_population.populations)
        total_fitness = sum(
            sum(a.fitness for a in p.agents) for p in meta_population.populations
        )
        avg_fitness = total_fitness / total_agents if total_agents > 0 else 0

        stats.extend(
            [
                "Metapopulation Status",
                "=" * 118,
                f"Generation: {meta_population.current_generation + 1}",
                f"Total Populations: {len(meta_population.populations)} | "
                f"Total Agents: {total_agents} | "
                f"Average Fitness: {avg_fitness:.2f}",
                "",
            ]
        )

        # Global elite agents
        all_agents = [a for p in meta_population.populations for a in p.agents]
        top_agents = sorted(all_agents, key=lambda a: a.fitness, reverse=True)[:5]

        stats.extend(["Global Elite Agents:"])
        for agent in top_agents:
            stats.extend(
                [
                    f"  {agent.id} (Pop {agent.population_id}):",
                    f"    Fitness: {agent.fitness} | Games: {agent.game_counter} | "
                    f"Offspring: {agent.offspring_count}",
                    self._format_agent_strategy(agent),
                    "",
                ]
            )

        # Population summaries
        stats.append("Population Performance Summary:")
        for pop in meta_population.populations:
            avg_pop_fitness = pop.calculate_average_fitness()
            diversity = pop.calculate_diversity()
            most_fit = pop.get_most_fit_agent()
            stats.extend(
                [
                    f"  Population {pop.population_id}:",
                    f"    Avg Fitness: {avg_pop_fitness:.2f} | "
                    f"Diversity: {diversity:.4f} | "
                    f"Top Agent: {most_fit.id} ({most_fit.fitness})",
                    "",
                ]
            )

        # Add bid parameter evolution
        stats.extend(
            [
                "",
                "Bid Parameter Evolution:",
                "Low Range   : "
                + self._format_bid_evolution(meta_population, "low_bid_range"),
                "Medium Range: "
                + self._format_bid_evolution(meta_population, "medium_bid_range"),
                "High Range  : "
                + self._format_bid_evolution(meta_population, "high_bid_range"),
            ]
        )

        self._update_section("Metapopulation Stats", stats)

    def _calculate_average_bid_params(self, population):
        """Calculate average bidding parameters across population"""
        avg_params = {
            "thresholds": [0, 0],
            "low_bid_range": [0, 0],
            "medium_bid_range": [0, 0],
            "high_bid_range": [0, 0],
            "bid_constant": 0,
        }

        n = len(population.agents)
        for agent in population.agents:
            bid_params = agent.strategy.bid_params
            for key in [
                "thresholds",
                "low_bid_range",
                "medium_bid_range",
                "high_bid_range",
            ]:
                avg_params[key][0] += bid_params[key][0] / n
                avg_params[key][1] += bid_params[key][1] / n
            avg_params["bid_constant"] += bid_params["bid_constant"] / n

        return avg_params

    def _format_agent_strategy(self, agent):
        """Format agent's strategy information"""
        bid_params = agent.strategy.bid_params
        return (
            f"  Strategy: {self._get_strategy_summary(agent.strategy)}\n"
            f"  Bid Range: {bid_params['low_bid_range'][0]*100:.1f}% - "
            f"{bid_params['high_bid_range'][1]*100:.1f}%"
        )

    def _format_bid_evolution(self, meta_population, range_type):
        """Format bid range evolution across populations"""
        ranges = []
        for pop in meta_population.populations:
            avg_params = self._calculate_average_bid_params(pop)
            ranges.append(
                f"{avg_params[range_type][0]*100:.1f}-{avg_params[range_type][1]*100:.1f}%"
            )
        return " | ".join(ranges)

    def _get_strategy_summary(self, strategy):
        """Get a brief summary of the strategy"""
        return (
            "Complex tree"
            if isinstance(strategy.decision_tree, dict)
            else str(strategy.decision_tree)
        )

    def _get_bid_summary(self, strategy):
        """Get a brief summary of bidding parameters"""
        return (
            f"{strategy.bid_params['low_bid_range'][0]*100:.1f}%"
            f"- {strategy.bid_params['high_bid_range'][1]*100:.1f}%"
        )


"""Manages game visualization and statistics display"""


class GameVisualizer:
    def __init__(self):
        self.fig = None
        self.board_ax = None
        self.stats_ax = None
        self._setup_plot()

    def _setup_plot(self):
        """Initialize matplotlib display"""
        self.fig, (self.board_ax, self.stats_ax) = plt.subplots(1, 2, figsize=(15, 8))
        self.board_ax.set_title("Game Board")
        self.stats_ax.set_title("Game Statistics")
        plt.ion()

    def _draw_detailed_agent_info(self, game):
        """Draw detailed agent information in stats axis"""
        self.stats_ax.clear()
        self.stats_ax.axis("off")

        def format_agent_info(player, is_player_one=True):
            prefix = "Player 1" if is_player_one else "Player 2"
            strategy = game.get_strategy_string(player.strategy)
            bid_info = game.format_bid_params(player.strategy.bid_params)

            return (
                f"{prefix}: {player.id}\n"
                f"Population: {player.population_id}\n"
                f"Fitness: {player.fitness}\n"
                f"Games Played: {player.game_counter}\n"
                f"Offspring Count: {player.offspring_count}\n"
                f"Points Remaining: {game.point_budgets['player1' if is_player_one else 'player2']}\n"
                f"\nBidding Strategy:\n{bid_info}\n"
                f"\nMovement Strategy:\n{strategy}"
            )

        # Add agent information
        agent1_info = format_agent_info(game.players["player1"], True)
        agent2_info = format_agent_info(game.players["player2"], False)

        self.stats_ax.text(
            0.02,
            0.98,
            agent1_info,
            va="top",
            ha="left",
            fontsize=8,
            fontfamily="monospace",
            transform=self.stats_ax.transAxes,
        )

        self.stats_ax.text(
            0.52,
            0.98,
            agent2_info,
            va="top",
            ha="left",
            fontsize=8,
            fontfamily="monospace",
            transform=self.stats_ax.transAxes,
        )

    def update_display(self, game, population1=None, population2=None):
        """Update visualization"""
        self._update_plot(game)

    def _update_plot(self, game):
        """Update matplotlib game visualization"""
        self.board_ax.clear()
        self.stats_ax.clear()

        # Draw board grid
        for i in range(game.board_size):
            for j in range(game.board_size):
                self.board_ax.add_patch(
                    plt.Rectangle((i, j), 1, 1, fill=False, color="gray")
                )

        # Draw player positions and trails - increased trail length to 48 moves
        if hasattr(game, "history") and game.history:
            # Draw trails
            p1_positions = [
                (state["player1"][0], state["player1"][1])
                for state in game.history[-48:]
            ]  # Last 48 moves
            p2_positions = [
                (state["player2"][0], state["player2"][1])
                for state in game.history[-48:]
            ]

            if len(p1_positions) > 1:
                x1, y1 = zip(*p1_positions)
                self.board_ax.plot(x1, y1, "c-", alpha=0.3, linewidth=3)

            if len(p2_positions) > 1:
                x2, y2 = zip(*p2_positions)
                self.board_ax.plot(x2, y2, "r-", alpha=0.3, linewidth=3)

        # Draw current positions - doubled marker sizes
        p1_pos = game.positions["player1"]
        p2_pos = game.positions["player2"]
        self.board_ax.scatter(p1_pos[0], p1_pos[1], c="blue", s=400, label="Player 1")
        self.board_ax.scatter(p2_pos[0], p2_pos[1], c="red", s=400, label="Player 2")

        # Draw goals - increased size
        g1_pos = game.goals["player1"]
        g2_pos = game.goals["player2"]
        self.board_ax.scatter(
            g1_pos[0], g1_pos[1], c="cyan", s=200, alpha=0.5, marker="*"
        )
        self.board_ax.scatter(
            g2_pos[0], g2_pos[1], c="pink", s=200, alpha=0.5, marker="*"
        )

        # Add game info
        title_text = (
            f"Game #{game.game_number} - Turn {len(game.history)}\n"
            f"Type: {game.game_type}\n"
            f"Outcome: {game.outcome_code if game.outcome_code else 'In Progress'}\n"
            f"Reason: {game.winning_reason if game.winning_reason else 'N/A'}"
        )
        self.board_ax.set_title(title_text, pad=20)

        # Add detailed agent info to stats axis
        self._draw_detailed_agent_info(game)

        # Set display properties
        self.board_ax.set_xlim(-0.5, game.board_size - 0.5)
        self.board_ax.set_ylim(-0.5, game.board_size - 0.5)
        self.board_ax.legend()

        plt.draw()
        plt.pause(0.1)

    def close(self):
        """Closes the visualization windows"""
        if self.fig:
            plt.close(self.fig)
            self.fig = None  # Set to None after closing


class EliteMatchVisualizer:
    """Handles visualization of elite matches"""

    def __init__(self):
        self.fig = plt.figure(figsize=(15, 10))
        self.gs = self.fig.add_gridspec(2, 2)
        self.board_ax = self.fig.add_subplot(self.gs[:, 0])
        self.stats_ax = self.fig.add_subplot(self.gs[0, 1])
        self.history_ax = self.fig.add_subplot(self.gs[1, 1])

        plt.style.use("seaborn")
        self.colors = {
            "intra_elite": COLORS["GREEN"],
            "inter_elite": COLORS["RED"],
            "player1": COLORS["BLUE"],
            "player2": COLORS["YELLOW"],
        }

    def visualize_elite_match(self, game, match_history=None):
        """Main visualization method for elite matches"""
        self.clear_axes()
        self._draw_board(game)
        self._draw_match_stats(game)
        if match_history:
            self._draw_match_history(match_history)
        plt.tight_layout()
        plt.draw()
        plt.pause(0.5)

    def _draw_board(self, game):
        """Draws the game board"""
        board_size = game.board_size
        self.board_ax.clear()
        self.board_ax.set_xlim(-1, board_size)
        self.board_ax.set_ylim(-1, board_size)

        # Draw grid
        for i in range(board_size):
            for j in range(board_size):
                self.board_ax.add_patch(
                    plt.Rectangle((i, j), 1, 1, fill=False, color=COLORS["GRAY"])
                )

        # Draw current positions and trails
        if hasattr(game, "history"):
            self._draw_movement_trails(game.history)

        p1_pos = game.positions["player1"]
        p2_pos = game.positions["player2"]
        self._draw_elite_position(p1_pos, "player1", game.players["player1"])
        self._draw_elite_position(p2_pos, "player2", game.players["player2"])

        match_type = (
            "Intrapopulation"
            if game.game_type == "intrapopulation_elite"
            else "Interpopulation"
        )
        self.board_ax.set_title(f"Elite Match ({match_type})\nGame {game.game_number}")

    def _draw_elite_position(self, pos, player_key, player):
        """Draws position indicators for elite agents"""
        base_size = 200
        elite_multiplier = 1.5

        # Base marker
        self.board_ax.scatter(
            pos[0],
            pos[1],
            c=self.colors[player_key],
            s=base_size,
            alpha=0.7,
            label=f"{player.id} (Pop {player.population_id})",
        )

        # Elite indicator ring
        if (
            player.fitness
            >= sorted([a.fitness for a in player.population.agents], reverse=True)[
                int(len(player.population.agents) * CONFIG["ELITE_PERCENTAGE"])
            ]
        ):
            self.board_ax.scatter(
                pos[0],
                pos[1],
                c="none",
                s=base_size * elite_multiplier,
                alpha=0.3,
                edgecolors=self.colors[player_key],
                linewidth=2,
            )

    def _draw_match_stats(self, game):
        """Draws match statistics"""
        self.stats_ax.clear()
        self.stats_ax.axis("off")
        stats_text = self._generate_stats_text(game)
        self.stats_ax.text(
            0.05,
            0.95,
            stats_text,
            transform=self.stats_ax.transAxes,
            verticalalignment="top",
            fontsize=8,
            fontfamily="monospace",
        )

    def _draw_match_history(self, history):
        """Draws match history trends"""
        self.history_ax.clear()
        generations = [h["generation"] for h in history]
        p1_fitness = [h["fitness_changes"]["player1"] for h in history]
        p2_fitness = [h["fitness_changes"]["player2"] for h in history]

        self.history_ax.plot(
            generations,
            p1_fitness,
            color=self.colors["player1"],
            label="Player 1 Fitness",
        )
        self.history_ax.plot(
            generations,
            p2_fitness,
            color=self.colors["player2"],
            label="Player 2 Fitness",
        )

        self.history_ax.set_title("Elite Match History")
        self.history_ax.set_xlabel("Generation")
        self.history_ax.set_ylabel("Fitness Change")
        self.history_ax.legend()

    def _generate_stats_text(self, game):
        """Generates statistics text display"""
        p1 = game.players["player1"]
        p2 = game.players["player2"]

        return (
            f"Elite Match Statistics\n"
            f"----------------------\n"
            f"Game Type: {game.game_type}\n"
            f"Game Number: {game.game_number}\n\n"
            f"Player 1 (Pop {p1.population_id}):\n"
            f"  ID: {p1.id}\n"
            f"  Fitness: {p1.fitness}\n"
            f"  Games Played: {p1.game_counter}\n"
            f"  Elite Status: {'Yes' if game.population.verify_elite_status(p1) else 'No'}\n\n"
            f"Player 2 (Pop {p2.population_id}):\n"
            f"  ID: {p2.id}\n"
            f"  Fitness: {p2.fitness}\n"
            f"  Games Played: {p2.game_counter}\n"
            f"  Elite Status: {'Yes' if game.population.verify_elite_status(p2) else 'No'}\n"
        )

    def clear_axes(self):
        """Clears all axes for new visualization"""
        self.board_ax.clear()
        self.stats_ax.clear()
        self.history_ax.clear()


class DisplayCoordinator:
    """Coordinates game visualization and statistics display"""

    def __init__(self):
        self.stats_display = StatsDisplay()
        self.game_visualizer = GameVisualizer()
        self.last_refresh = time.time()
        self.min_refresh_interval = 0.001  # 1ms refresh rate
        self.current_game = None
        self.active_population = None
        self.active_meta_population = None
        self._init_logging()

    def _init_logging(self):
        """Initialize enhanced logging"""
        formatter = logging.Formatter(
            '{"time": "%(asctime)s", "level": "%(levelname)s", '
            '"section": "%(section)s", "content": "%(message)s"}'
        )
        for handler in logging.getLogger().handlers:
            handler.setFormatter(formatter)

    def update_display(self, game, meta_population=None):
        """Updates all displays with improved coordination and timing"""
        current_time = time.time()

        # Check if enough time has passed since last refresh
        if current_time - self.last_refresh >= self.min_refresh_interval:
            try:
                # Update game counter first
                if game:
                    self.stats_display.total_games_played = game.game_number
                    self.stats_display.increment_game_counter()

                # Update all display sections
                self._update_all_sections(game, meta_population)

                # Force immediate refresh
                sys.stdout.flush()
                self.last_refresh = current_time

            except Exception as e:
                logging.error(f"Display update error: {e}")
                self.stats_display._safe_update_log()

    def _update_all_sections(self, game, meta_population):
        """Updates all display sections with comprehensive logging"""
        if game:
            # Update game section
            game_stats = self._format_game_stats(game)
            self.stats_display._update_section("Current Game Stats", game_stats)

            # Log complete game state
            logging.info(
                f"Current Game Display Content:\n{self._format_log_content(game_stats)}",
                extra={"section": "Game"},
            )

        if meta_population:
            # Update metapopulation stats
            meta_stats = self._format_meta_stats(meta_population)
            self.stats_display._update_section("Metapopulation Stats", meta_stats)
            logging.info(
                f"Metapopulation Display Content:\n{self._format_log_content(meta_stats)}",
                extra={"section": "MetaPop"},
            )

            # Update population stats
            for population in meta_population.populations:
                pop_stats = self._format_population_stats(population)
                self.stats_display._update_section(
                    f"Population {population.population_id} Stats", pop_stats
                )
                logging.info(
                    f"Population {population.population_id} Display Content:\n{self._format_log_content(pop_stats)}",
                    extra={"section": f"Pop{population.population_id}"},
                )

    def _format_game_stats(self, game):
        """Formats comprehensive game statistics"""
        return [
            f"Game #{game.game_number} - {game.game_type}",
            "=" * 180,
            f"Turn: {len(game.history)}",
            f"Status: {game.outcome_code if game.outcome_code else 'In Progress'}",
            f"Reason: {game.winning_reason if game.winning_reason else 'N/A'}",
            "",
            "Player 1:",
            f"  ID: {game.players['player1'].id} (Pop {game.players['player1'].population_id})",
            f"  Position: {game.positions['player1']} → Goal: {game.goals['player1']}",
            f"  Points: {game.point_budgets['player1']}",
            f"  Fitness: {game.players['player1'].fitness}",
            f"  Games: {game.players['player1'].game_counter}",
            "",
            "Player 2:",
            f"  ID: {game.players['player2'].id} (Pop {game.players['player2'].population_id})",
            f"  Position: {game.positions['player2']} → Goal: {game.goals['player2']}",
            f"  Points: {game.point_budgets['player2']}",
            f"  Fitness: {game.players['player2'].fitness}",
            f"  Games: {game.players['player2'].game_counter}",
        ]

    def _format_log_content(self, content_list):
        """Formats content for logging"""
        return "\n".join(str(line) for line in content_list)

    def handle_game_completion(self, game, meta_population=None):
        """Special handling for game completion"""
        if game.winner:
            try:
                # Update all displays with final state
                self.stats_display.update_current_game_stats(game)
                if meta_population:
                    self.stats_display.update_metapopulation_stats(meta_population)

                    # Get and update winner's population stats
                    winner_pop = meta_population.get_population_of_agent(game.winner)
                    if winner_pop:
                        self.stats_display.update_population_stats(winner_pop)

                # Log game outcome
                outcome_str = (
                    f"Game #{game.game_number} complete: "
                    f"{game.winner.id} won ({game.outcome_code})"
                )
                self.stats_display.log_buffer.append(outcome_str)
                self.stats_display._safe_update_log()

            except Exception as e:
                logging.error(f"Game completion display error: {e}")

    def cleanup(self):
        """Cleans up all displays"""
        if self.stats_display:
            self.stats_display.cleanup()
        if self.game_visualizer:
            self.game_visualizer.close()


class Game:
    """Main game class handling game logic and state"""

    total_games_played = 0

    @classmethod
    def get_next_game_number(cls):
        cls.total_games_played += 1
        return cls.total_games_played

    def __init__(
        self,
        player1,
        player2,
        board_size=CONFIG["BOARD_SIZE"],
        visualize=False,
        game_type="regular",
        meta_population=None,
    ):
        self.game_number = Game.get_next_game_number()
        self.board_size = board_size
        self.players = {"player1": player1, "player2": player2}
        self.positions = {
            "player1": (0, 0),
            "player2": (board_size - 1, board_size - 1),
        }
        self.goals = {
            "player1": (board_size - 1, board_size - 1),
            "player2": (0, 0),
        }
        self.point_budgets = {
            "player1": CONFIG["STARTING_POINTS"],
            "player2": CONFIG["STARTING_POINTS"],
        }
        self.consecutive_turns_on_goal = {"player1": 0, "player2": 0}
        self.winner = None
        self.loser = None
        self.outcome_code = None
        self.winning_reason = None
        self.visualize = visualize
        self.display_coordinator = DisplayCoordinator() if visualize else None
        self.history = []
        self.game_type = game_type
        self.meta_population = meta_population
        self.match_stats = {
            "generation": getattr(player1, "meta_population", None)
            and player1.meta_population.current_generation,
            "game_type": game_type,
            "initial_fitness": {"player1": player1.fitness, "player2": player2.fitness},
        }

    def play(self):
        """Main game loop with enhanced display updates"""
        try:
            if self.display_coordinator:
                # Update game counter FIRST
                self.display_coordinator.stats_display.total_games_played = (
                    self.game_number
                )
                self.display_coordinator.stats_display._update_game_counter()
                self.display_coordinator.update_display(self, self.meta_population)

            self.players["player1"].increment_game_counter()
            self.players["player2"].increment_game_counter()

            turn = 0
            while not self.winner and turn <= CONFIG["MAX_TURNS"]:
                if not self._process_turn(turn):
                    break

                if self.display_coordinator:
                    # Update display after each turn
                    self.display_coordinator.update_display(self, self.meta_population)
                    self.display_coordinator.handle_game_completion(
                        self, self.meta_population
                    )

                self._update_history(turn)
                turn += 1

            if turn > CONFIG["MAX_TURNS"]:
                self._handle_timeout()

            if self.display_coordinator:
                # Final display update
                self.display_coordinator.update_display(self, self.meta_population)
                self.display_coordinator.handle_game_completion(
                    self, self.meta_population
                )

            return self.winner

        except Exception as e:
            logging.error(f"Error in game play: {e}", exc_info=True)
            if self.display_coordinator:
                self.display_coordinator.cleanup()
            return None

    def _update_history(self, turn):
        """Records current game state"""
        state = {
            "player1": self.positions["player1"],
            "player2": self.positions["player2"],
            "points_p1": self.point_budgets["player1"],
            "points_p2": self.point_budgets["player2"],
            "turn": turn + 1,
            "meta_stats": {
                "player1_fitness": self.players["player1"].fitness,
                "player2_fitness": self.players["player2"].fitness,
                "player1_games": self.players["player1"].game_counter,
                "player2_games": self.players["player2"].game_counter,
            },
        }
        self.history.append(state)

    def get_population_of_player(self, player_key):
        """Helper to get population of a player"""
        player = self.players[player_key]
        if hasattr(player, "population_id") and hasattr(player, "meta_population"):
            return player.meta_population.populations[player.population_id - 1]
        return None

    def _process_turn(self, turn):
        """Processes a single turn with enhanced logging and display updates"""
        if self.display_coordinator:
            # Update game counter and log game start
            self.display_coordinator.stats_display.total_games_played = self.game_number
            self.display_coordinator.stats_display.increment_game_counter()

            # Log comprehensive game state
            self._log_game_state()

        # Process bidding
        if not self._process_bidding():
            return False

        # Process movement
        if not self._process_movement():
            return False

        # Force display update
        if self.display_coordinator:
            self.display_coordinator.update_display(self, self.meta_population)

        return True

    def _log_game_state(self):
        """Logs comprehensive game state information"""
        game_state = {
            "game_number": self.game_number,
            "game_type": self.game_type,
            "turn": len(self.history),
            "status": self.outcome_code if self.outcome_code else "In Progress",
            "reason": self.winning_reason if self.winning_reason else "N/A",
            "players": {
                "player1": {
                    "id": self.players["player1"].id,
                    "population": self.players["player1"].population_id,
                    "position": self.positions["player1"],
                    "goal": self.goals["player1"],
                    "points": self.point_budgets["player1"],
                    "fitness": self.players["player1"].fitness,
                    "games_played": self.players["player1"].game_counter,
                },
                "player2": {
                    "id": self.players["player2"].id,
                    "population": self.players["player2"].population_id,
                    "position": self.positions["player2"],
                    "goal": self.goals["player2"],
                    "points": self.point_budgets["player2"],
                    "fitness": self.players["player2"].fitness,
                    "games_played": self.players["player2"].game_counter,
                },
            },
        }

        logging.info(
            f"Game State Update: Game #{self.game_number}\n"
            f"Type: {game_state['game_type']}\n"
            f"Turn: {game_state['turn']}\n"
            f"Status: {game_state['status']}\n"
            f"Reason: {game_state['reason']}\n\n"
            f"Player 1 ({game_state['players']['player1']['id']}):\n"
            f"  Population: {game_state['players']['player1']['population']}\n"
            f"  Position: {game_state['players']['player1']['position']}\n"
            f"  Goal: {game_state['players']['player1']['goal']}\n"
            f"  Points: {game_state['players']['player1']['points']}\n"
            f"  Fitness: {game_state['players']['player1']['fitness']}\n"
            f"  Games: {game_state['players']['player1']['games_played']}\n\n"
            f"Player 2 ({game_state['players']['player2']['id']}):\n"
            f"  Population: {game_state['players']['player2']['population']}\n"
            f"  Position: {game_state['players']['player2']['position']}\n"
            f"  Goal: {game_state['players']['player2']['goal']}\n"
            f"  Points: {game_state['players']['player2']['points']}\n"
            f"  Fitness: {game_state['players']['player2']['fitness']}\n"
            f"  Games: {game_state['players']['player2']['games_played']}"
        )

    def _process_bidding(self):
        """Handles the bidding phase"""
        bids = {}
        for player_id in ["player1", "player2"]:
            opponent_id = "player2" if player_id == "player1" else "player1"
            bid = self.players[player_id].get_bid(
                self.point_budgets[player_id], self.point_budgets[opponent_id]
            )
            bids[player_id] = bid
            self.point_budgets[player_id] -= bid

            if self.point_budgets[player_id] < 0:
                self.point_budgets[player_id] = 0
                self._handle_out_of_points(player_id)
                return False

        return True

    def _process_movement(self):
        """Handles the movement phase"""
        mover_id = self._determine_mover()
        non_mover_id = "player2" if mover_id == "player1" else "player1"

        move = self.players[mover_id].get_move(
            self.positions[mover_id],
            self.positions[non_mover_id],
            self.board_size,
            self.goals[mover_id],
            self.goals[non_mover_id],
        )

        if not self._validate_and_execute_move(mover_id, non_mover_id, move):
            return False

        return True

    def _determine_mover(self):
        """Determines which player moves based on bids"""
        if self.point_budgets["player1"] > self.point_budgets["player2"]:
            return "player1"
        elif self.point_budgets["player1"] < self.point_budgets["player2"]:
            return "player2"
        else:
            return random.choice(["player1", "player2"])

    def _validate_and_execute_move(self, mover_id, non_mover_id, move):
        """Validates and executes a move"""
        if self.is_valid_move(move, self.positions[non_mover_id]):
            self.positions[mover_id] = move

            if move != self.goals[non_mover_id]:
                self.consecutive_turns_on_goal[mover_id] = 0

            if move == self.goals[mover_id]:
                self._handle_goal_reached(mover_id, non_mover_id)
                return False
        else:
            self._handle_invalid_move(mover_id, non_mover_id)
            return False

        # Check consecutive turns on opponent's goal
        if not self._check_consecutive_goal_turns():
            return False

        return True

    def _handle_goal_reached(self, winner_id, loser_id):
        """Handles when a player reaches their goal"""
        self.winner = self.players[winner_id]
        self.loser = self.players[loser_id]
        self.winning_reason = f"{winner_id} reached their goal."
        self.outcome_code = "reached_goal"
        self.update_fitness()

    def _handle_invalid_move(self, mover_id, non_mover_id):
        """Handles invalid move attempts"""
        self.winner = self.players[non_mover_id]
        self.loser = self.players[mover_id]
        self.winning_reason = f"{mover_id} made an invalid move."
        self.outcome_code = "invalid_move"
        self.update_fitness()

    def _handle_out_of_points(self, loser_id):
        """Handles when a player runs out of points"""
        winner_id = "player2" if loser_id == "player1" else "player1"
        self.winner = self.players[winner_id]
        self.loser = self.players[loser_id]
        self.winning_reason = f"{loser_id} ran out of points."
        self.outcome_code = "ran_out_of_points"
        self.update_fitness()

    def _handle_timeout(self):
        """Handles game timeout"""
        self._determine_timeout_winner()
        self.update_fitness()

    def _determine_timeout_winner(self):
        """Determines winner in case of timeout"""
        distances = {
            "player1": self.manhattan_distance(
                self.positions["player1"], self.goals["player1"]
            ),
            "player2": self.manhattan_distance(
                self.positions["player2"], self.goals["player2"]
            ),
        }

        if distances["player1"] != distances["player2"]:
            winner_id = min(distances, key=distances.get)
            loser_id = "player2" if winner_id == "player1" else "player1"
            self.winner = self.players[winner_id]
            self.loser = self.players[loser_id]
            self.winning_reason = (
                f"{winner_id} was closer to their goal based on proximity."
            )
            self.outcome_code = "proximity_tiebreak"
        else:
            self._resolve_point_tiebreak()

    def _resolve_point_tiebreak(self):
        """Resolves timeout via point tiebreak"""
        if self.point_budgets["player1"] != self.point_budgets["player2"]:
            winner_id = max(self.point_budgets, key=self.point_budgets.get)
            loser_id = "player2" if winner_id == "player1" else "player1"
            self.winner = self.players[winner_id]
            self.loser = self.players[loser_id]
            self.winning_reason = (
                f"{winner_id} had more remaining points as a tiebreaker."
            )
            self.outcome_code = "points_tiebreak"
        else:
            # Random winner for complete tie
            winner_id = random.choice(["player1", "player2"])
            loser_id = "player2" if winner_id == "player1" else "player1"
            self.winner = self.players[winner_id]
            self.loser = self.players[loser_id]
            self.winning_reason = (
                "The game ended in a tie based on proximity and points. "
                f"{self.winner.id} was randomly selected as the winner."
            )
            self.outcome_code = "random_tiebreak"

    def get_fitness_change(self, player):
        """Calculates fitness change for a player"""
        fitness_change = 0
        if player == self.winner:
            if self.outcome_code == "reached_goal":
                fitness_change = 18
            elif self.outcome_code == "proximity_tiebreak":
                fitness_change = 3
            elif self.outcome_code in ["points_tiebreak", "random_tiebreak"]:
                fitness_change = 2
            elif self.outcome_code != "error":
                fitness_change = 1
        elif player == self.loser:
            if self.outcome_code == "ran_out_of_points":
                fitness_change = -6 - ((2 * player.fitness) // 100)
            elif self.outcome_code == "stayed_on_opponent_goal":
                fitness_change = -28 - ((2 * player.fitness) // 100)
            elif self.outcome_code == "invalid_move":
                fitness_change = -28 - ((2 * player.fitness) // 100)
            elif self.outcome_code != "error":
                fitness_change = -3 - ((2 * player.fitness) // 100)
        return fitness_change

    def update_fitness(self):
        """Updates fitness for both players based on game outcome"""
        if self.winner and self.loser:
            fitness_change_winner = self.get_fitness_change(self.winner)
            fitness_change_loser = self.get_fitness_change(self.loser)

            self.winner.fitness += fitness_change_winner
            self.loser.fitness += fitness_change_loser

            logging.info(
                f"Fitness Update - {self.winner.id}: +{fitness_change_winner}, "
                f"{self.loser.id}: {fitness_change_loser}"
            )

            # Update fitness tracking attributes
            self.winner.sum_fitness_change += fitness_change_winner
            self.winner.sum_fitness_change_squared += fitness_change_winner**2
            self.winner.games_played += 1

            self.loser.sum_fitness_change += fitness_change_loser
            self.loser.sum_fitness_change_squared += fitness_change_loser**2
            self.loser.games_played += 1

            self.record_match_stats()

    def record_match_stats(self):
        """Records match statistics"""
        if self.winner:
            self.match_stats.update(
                {
                    "winner_id": self.winner.id,
                    "winner_population": self.winner.population_id,
                    "fitness_changes": {
                        "player1": self.players["player1"].fitness
                        - self.match_stats["initial_fitness"]["player1"],
                        "player2": self.players["player2"].fitness
                        - self.match_stats["initial_fitness"]["player2"],
                    },
                }
            )

    def record_final_state(self, turn):
        """Records the final state of the game"""
        if self.visualize:
            self.history.append(
                {
                    "player1": self.positions["player1"],
                    "player2": self.positions["player2"],
                    "points_p1": self.point_budgets["player1"],
                    "points_p2": self.point_budgets["player2"],
                    "turn": turn + 1,
                }
            )

    def is_valid_move(self, position, opponent_position):
        """Checks if a move is valid"""
        x, y = position
        return (
            0 <= x < self.board_size
            and 0 <= y < self.board_size
            and position != opponent_position
        )

    def manhattan_distance(self, pos1, pos2):
        """Calculates Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _check_consecutive_goal_turns(self):
        """Checks if a player has stayed on opponent's goal too long"""
        max_turns_on_goal = 6
        for pid in ["player1", "player2"]:
            opponent_id = "player2" if pid == "player1" else "player1"
            if self.positions[pid] == self.goals[opponent_id]:
                self.consecutive_turns_on_goal[pid] += 1
                if self.consecutive_turns_on_goal[pid] > max_turns_on_goal:
                    self.winner = self.players[opponent_id]
                    self.loser = self.players[pid]
                    self.winning_reason = f"{pid} stayed on opponent's goal too long"
                    self.outcome_code = "stayed_on_opponent_goal"
                    self.update_fitness()
                    return False
            else:
                self.consecutive_turns_on_goal[pid] = 0
        return True

    def visualize_game(self, player1, player2, initial_fitness_p1, initial_fitness_p2):
        """Visualizes the game state and progress"""
        # Prepare movement trails
        trail_length = 24  # Number of previous positions to display
        p1_trail = []
        p2_trail = []

        # Create a new figure with adjusted size
        fig, ax = plt.subplots(figsize=(22, 10))

        for idx, state in enumerate(self.history):
            board = np.zeros((self.board_size, self.board_size))
            p1_pos = state["player1"]
            p2_pos = state["player2"]

            points_p1 = state["points_p1"]
            points_p2 = state["points_p2"]

            # Update trails
            p1_trail.append(p1_pos)
            p2_trail.append(p2_pos)
            if len(p1_trail) > trail_length:
                p1_trail.pop(0)
            if len(p2_trail) > trail_length:
                p2_trail.pop(0)

            # Clear the axes
            ax.clear()

            # Plot the board
            ax.imshow(board, cmap="gray", vmin=0, vmax=2, origin="lower")

            # Adjust axes limits to include text positions
            ax.set_xlim(-11, self.board_size + 10)
            ax.set_ylim(-4, self.board_size + 3)

            # Plot trails
            if len(p1_trail) > 1:
                p1_trail_array = np.array(p1_trail)
                ax.plot(
                    p1_trail_array[:, 0],
                    p1_trail_array[:, 1],
                    color="cyan",
                    linewidth=2,
                    alpha=0.3,
                )
            if len(p2_trail) > 1:
                p2_trail_array = np.array(p2_trail)
                ax.plot(
                    p2_trail_array[:, 0],
                    p2_trail_array[:, 1],
                    color="red",
                    linewidth=2,
                    alpha=0.3,
                )

            # Plot current positions
            ax.scatter(p1_pos[0], p1_pos[1], c="blue", s=280)
            ax.scatter(p2_pos[0], p2_pos[1], c="yellow", s=240)
            ax.scatter(
                p1_pos[0],
                p1_pos[1],
                c="cyan",
                s=200,
                label=f"Player 1, {player1.__class__.__name__} {player1.id}",
            )
            ax.scatter(
                p2_pos[0],
                p2_pos[1],
                c="red",
                s=200,
                label=f"Player 2, {player2.__class__.__name__} {player2.id}",
            )

            # Generate wrapped genealogy strings
            genealogy_p1 = ", ".join(map(str, sorted(player1.genealogy)))
            genealogy_p2 = ", ".join(map(str, sorted(player2.genealogy)))
            wrapped_genealogy_p1 = "\n".join(wrap_text(genealogy_p1, width=100))
            wrapped_genealogy_p2 = "\n".join(wrap_text(genealogy_p2, width=100))

            # Generate strategy strings
            strategy_p1 = self.get_strategy_string(player1.strategy)
            strategy_p2 = self.get_strategy_string(player2.strategy)

            # Use initial fitness values
            fitness_p1 = initial_fitness_p1
            fitness_p2 = initial_fitness_p2

            # Bid Parameters Information
            bid_info_p1 = self.format_bid_params(player1.strategy.bid_params)
            bid_info_p2 = self.format_bid_params(player2.strategy.bid_params)

            # Print Agent Information
            self._draw_agent_info(
                ax,
                player1,
                p1_pos,
                points_p1,
                fitness_p1,
                wrapped_genealogy_p1,
                strategy_p1,
                bid_info_p1,
                is_player_one=True,
            )
            self._draw_agent_info(
                ax,
                player2,
                p2_pos,
                points_p2,
                fitness_p2,
                wrapped_genealogy_p2,
                strategy_p2,
                bid_info_p2,
                is_player_one=False,
            )

            ax.legend(loc="upper left")
            ax.set_title(f"Game {self.game_number} - Turn {idx + 1}")

            plt.draw()
            plt.pause(0.2)

        plt.pause(2)
        plt.close()

    def _draw_agent_info(
        self,
        ax,
        agent,
        pos,
        points,
        fitness,
        genealogy,
        strategy,
        bid_info,
        is_player_one=True,
    ):
        """Draws agent information on the plot"""
        x_pos = -10 if is_player_one else self.board_size + 9
        y_base = 9.0 if is_player_one else self.board_size + 1.0
        color = "blue" if is_player_one else "red"
        ha = "left" if is_player_one else "right"

        # Basic info
        ax.text(
            x_pos,
            y_base,
            f"Player {'1' if is_player_one else '2'}, {agent.id}\n"
            f"Points: {points}\nGames Played: {agent.game_counter}",
            fontsize=12,
            color=color,
            ha=ha,
            va="top",
            clip_on=False,
        )

        # Fitness
        ax.text(
            x_pos,
            y_base - 2.0,
            f"Fitness: {fitness}",
            fontsize=11,
            color=color,
            ha=ha,
            va="top",
            clip_on=False,
        )

        # Detailed info
        ax.text(
            x_pos,
            y_base - 2.5,
            f"Genealogy:\n{genealogy}\n\n"
            f"Strategy: {strategy}\n\n"
            f"Bid Params:\n{bid_info}",
            fontsize=5,
            color=color,
            ha=ha,
            va="top",
            clip_on=False,
        )

    def get_strategy_string(self, strategy):
        """Converts a strategy to a string representation"""

        def traverse(node):
            if isinstance(node, dict):
                condition = node["condition"]
                true_branch = traverse(node["true"])
                false_branch = traverse(node["false"])
                return f"({condition}? {true_branch} : {false_branch})"
            else:
                return node

        strategy_str = traverse(strategy.decision_tree)
        return "\n".join(wrap_text(strategy_str, width=100))

    def format_bid_params(self, bid_params):
        """Formats bid parameters into a readable string"""
        return (
            f"Thresholds: {bid_params['thresholds']}\n"
            f"Low Bid Range: {format_bid_range(bid_params['low_bid_range'])}\n"
            f"Medium Bid Range: {format_bid_range(bid_params['medium_bid_range'])}\n"
            f"High Bid Range: {format_bid_range(bid_params['high_bid_range'])}\n"
            f"Bid Constant: {bid_params['bid_constant']}"
        )


class GameScheduler:
    """Manages game scheduling and execution"""

    def __init__(self):
        self.scheduled_games = []  # Each game is a tuple: (agent1, agent2, game_type)

    def schedule_game(self, agent1, agent2, game_type="regular"):
        """Schedules a game between two agents"""
        # Ensure both agents are active before scheduling
        if agent1.fitness >= 0 and agent2.fitness >= 0:
            self.scheduled_games.append((agent1, agent2, game_type))
            logging.debug(
                f"Scheduled {game_type} game between {agent1.id} and " f"{agent2.id}."
            )
        else:
            logging.warning(
                f"Cannot schedule {game_type} game between {agent1.id} and "
                f"{agent2.id} as one or both agents are inactive."
            )

    def remove_agent_games(self, agent):
        """Removes all scheduled games involving the specified agent"""
        original_count = len(self.scheduled_games)
        self.scheduled_games = [
            (a1, a2, gt)
            for (a1, a2, gt) in self.scheduled_games
            if a1 != agent and a2 != agent
        ]
        removed_count = original_count - len(self.scheduled_games)
        if removed_count > 0:
            logging.info(
                f"Removed {removed_count} scheduled games involving Agent "
                f"{agent.id}."
            )

    def get_next_game(self):
        """Gets the next scheduled game"""
        if self.scheduled_games:
            return self.scheduled_games.pop(0)
        return None


class Population:
    """Represents a population of agents"""

    def __init__(
        self, size=CONFIG["POPULATION_SIZE"], population_id=1, meta_population=None
    ):
        self.size = size
        self.population_id = population_id
        self.meta_population = meta_population
        self.agents = []
        self.all_agents = []
        self.elite_percentage = CONFIG["ELITE_PERCENTAGE"]
        self.game_scheduler = GameScheduler()
        self.genealogy_counts = {}
        self.agent_lookup = {}

        # Initialize population with agents
        for _ in range(size):
            agent = Agent(population_id=self.population_id)
            self.agents.append(agent)
            self.all_agents.append(agent)
            self.agent_lookup[agent.id] = agent
            self._update_genealogy_counts_on_add(agent)

        # Initialize tracking
        self.previous_most_fit_agent = self.get_unique_most_fit_agent()
        self.previous_least_fit_agent = self.get_unique_least_fit_agent()
        self.base_mutation_rate = CONFIG["BASE_MUTATION_RATE"]
        self.mutation_rate = self.base_mutation_rate
        self.fitness_history = []

    def evaluate_fitness(
        self, meta_population=None, report_event=None, continue_event=None
    ):
        """Evaluates fitness for current generation with alternating elite matches"""
        print("\n--- Evaluating Fitness for Current Generation ---")

        if self.meta_population:
            self.meta_population._update_display()

        # Reset per-generation game counters
        for agent in self.agents:
            agent.reset_generation_counter()

        # Schedule regular games first
        self.schedule_regular_games()

        if self.meta_population:
            self.meta_population._update_display()

        # Execute regular games
        self.execute_regular_games(meta_population, report_event, continue_event)

        if self.meta_population:
            self.meta_population._update_display()

        # Now alternate between intrapopulation and interpopulation elite matches
        self.execute_alternating_elite_matches(
            meta_population, report_event, continue_event
        )

        if self.meta_population:
            self.meta_population._update_display()

        self.report_population_status()
        print("\n--- Fitness Evaluation Complete ---\n")

    def schedule_regular_games(self):
        """Schedules the initial set of regular games"""
        required_games_per_agent = 24
        total_games_needed = (self.size * required_games_per_agent) // 2

        original_agents = list(self.agents)
        random.shuffle(original_agents)

        while len(self.game_scheduler.scheduled_games) < total_games_needed:
            agent1, agent2 = random.sample(original_agents, 2)
            self.game_scheduler.schedule_game(agent1, agent2, game_type="regular")

    def execute_regular_games(self, meta_population, report_event, continue_event):
        """Executes regular scheduled games"""
        while self.game_scheduler.scheduled_games:
            if report_event and report_event.is_set():
                self.handle_report_request(
                    meta_population, report_event, continue_event
                )

            game_data = self.game_scheduler.get_next_game()
            if not game_data:
                break

            agent1, agent2, game_type = game_data

            # Skip if agents no longer in population
            if agent1 not in self.agents or agent2 not in self.agents:
                continue

            if self.meta_population:
                self.meta_population._update_display()

            # Create and play game
            game = Game(
                agent1, agent2, game_type=game_type, meta_population=meta_population
            )
            winner = game.play()
            winner = winner

            if self.meta_population:
                self.meta_population._update_display()

            # Handle game results
            self.handle_game_results(
                game, meta_population, report_event, continue_event
            )

    def execute_alternating_elite_matches(
        self, meta_population, report_event, continue_event
    ):
        """Executes elite matches, alternating between intra and interpopulation"""
        elite_percentage = CONFIG["ELITE_PERCENTAGE"]
        num_elite = max(2, int(self.size * elite_percentage))
        elite_agents = sorted(self.agents, key=lambda a: a.fitness, reverse=True)[
            :num_elite
        ]
        non_elite_agents = [a for a in self.agents if a not in elite_agents]

        for elite_agent in elite_agents:
            # Intrapopulation matches
            for opponent in non_elite_agents:
                if elite_agent == opponent:
                    continue

                if self.meta_population:
                    self.meta_population._update_display()

                # Schedule and execute intrapopulation match
                game = Game(elite_agent, opponent, game_type="intrapopulation_elite")
                winner = game.play()
                winner = winner

                if self.meta_population:
                    self.meta_population._update_display()

                # Handle results
                self.handle_game_results(
                    game, meta_population, report_event, continue_event
                )

                # Immediately follow with interpopulation match if appropriate
                if winner and self.verify_elite_status(winner):
                    other_populations = [
                        p for p in meta_population.populations if p != self
                    ]
                    for other_pop in other_populations:
                        other_elite = other_pop.get_most_fit_agent()
                        if other_elite:
                            inter_game = Game(
                                winner, other_elite, game_type="interpopulation_elite"
                            )

                            if self.meta_population:
                                self.meta_population._update_display()

                            inter_winner = inter_game.play()
                            inter_winner = inter_winner

                            if self.meta_population:
                                self.meta_population._update_display()

                            # Handle interpopulation match results
                            self.handle_game_results(
                                inter_game,
                                meta_population,
                                report_event,
                                continue_event,
                            )

    def handle_game_results(self, game, meta_population, report_event, continue_event):
        """Handles post-game processing including visualization and agent replacement"""

        if self.meta_population:
            self.meta_population._update_display()

        # Check for report request
        if report_event and report_event.is_set():
            self.handle_report_request(meta_population, report_event, continue_event)

        # Handle visualization
        if random.random() < CONFIG["VISUALIZATION_SAMPLE_RATE"]:
            try:
                # Create visualization game and set up display coordinator
                visualizer_game = Game(
                    game.players["player1"],
                    game.players["player2"],
                    board_size=CONFIG["BOARD_SIZE"],
                    visualize=True,
                    game_type=game.game_type,
                    meta_population=meta_population,
                )

                # Copy game state
                visualizer_game.positions = game.positions.copy()
                visualizer_game.point_budgets = game.point_budgets.copy()
                visualizer_game.history = game.history.copy()
                visualizer_game.winner = game.winner
                visualizer_game.loser = game.loser
                visualizer_game.outcome_code = game.outcome_code
                visualizer_game.winning_reason = game.winning_reason

                # Display final state
                if visualizer_game.display_coordinator:
                    visualizer_game.display_coordinator.update_display(
                        visualizer_game, meta_population
                    )
                    time.sleep(2)
                    visualizer_game.display_coordinator.cleanup()
            except Exception as e:
                logging.error(f"Visualization error: {e}")

        # Handle agent replacement if necessary
        least_fit = self.get_least_fit_agent()
        if least_fit and least_fit.fitness < 0:
            self.remove_and_replace_agent(
                agent_to_remove=least_fit,
                winner=game.winner,
                meta_population=meta_population,
            )

    def handle_report_request(self, meta_population, report_event, continue_event):
        """Handles report generation requests"""
        print("\n--- Generating Metapopulation Report ---")
        meta_population.report_metapopulation_status()
        report_event.clear()

        if continue_event:
            continue_event.wait()
            continue_event.clear()

    def verify_elite_status(self, agent):
        """Verifies if an agent qualifies for elite status"""
        sorted_fitness = sorted([a.fitness for a in self.agents], reverse=True)
        elite_threshold = sorted_fitness[int(len(self.agents) * self.elite_percentage)]
        return agent.fitness >= elite_threshold

    def get_agent_by_id(self, agent_id):
        """Retrieves an agent by ID"""
        return self.agent_lookup.get(agent_id, None)

    def _update_genealogy_counts_on_add(self, agent):
        """Updates genealogy counts when adding an agent"""
        for ancestor_id in agent.genealogy:
            if ancestor_id != agent.id:
                self.genealogy_counts[ancestor_id] = (
                    self.genealogy_counts.get(ancestor_id, 0) + 1
                )

    def _update_genealogy_counts_on_remove(self, agent):
        """Updates genealogy counts when removing an agent"""
        for ancestor_id in agent.genealogy:
            if ancestor_id != agent.id:
                if self.genealogy_counts.get(ancestor_id, 0) > 0:
                    self.genealogy_counts[ancestor_id] -= 1

    def add_agent(self, agent):
        """Adds an agent to the population"""
        assert (
            agent not in self.agent_lookup
        ), f"Agent {agent.id} already exists in Population {self.population_id}"

        self.agents.append(agent)
        self.all_agents.append(agent)
        self.agent_lookup[agent.id] = agent
        self._update_genealogy_counts_on_add(agent)

        # Check and maintain population size
        if len(self.agents) > self.size:
            least_fit_agent = self.get_least_fit_agent()
            if least_fit_agent:
                self.remove_agent(least_fit_agent)

        self.report_population_status()

    def remove_agent(self, agent):
        """Removes an agent from the population"""
        if agent in self.agents:
            self.agents.remove(agent)
            del self.agent_lookup[agent.id]
            self._update_genealogy_counts_on_remove(agent)
            self.game_scheduler.remove_agent_games(agent)

            # Update parent offspring counts
            for parent_id in agent.parents:
                parent_agent = self.agent_lookup.get(parent_id)
                if parent_agent:
                    parent_agent.offspring_count = max(
                        parent_agent.offspring_count - 1, 0
                    )

    def calculate_average_fitness(self):
        """Calculates average fitness of the population"""
        if not self.agents:
            return 0
        return sum(agent.fitness for agent in self.agents) / len(self.agents)

    def calculate_diversity(self):
        """
        Calculates diversity based on unique decision trees and bid parameters
        Returns a value between 0 and 1
        """
        if not self.agents:
            return 0

        # Calculate strategy diversity
        unique_strategies = set()
        for agent in self.agents:
            # Create a signature combining decision tree and bid parameters
            strategy_signature = str(agent.strategy.decision_tree) + str(
                sorted(agent.strategy.bid_params.items())
            )
            unique_strategies.add(strategy_signature)

        # Calculate bid parameter diversity
        bid_parameter_sets = set()
        for agent in self.agents:
            bid_params = agent.strategy.bid_params
            # Create a tuple of bid parameters
            bid_signature = (
                tuple(bid_params["thresholds"]),
                tuple(bid_params["low_bid_range"]),
                tuple(bid_params["medium_bid_range"]),
                tuple(bid_params["high_bid_range"]),
                bid_params["bid_constant"],
            )
            bid_parameter_sets.add(bid_signature)

        # Combine both measures of diversity
        strategy_diversity = len(unique_strategies) / len(self.agents)
        bid_diversity = len(bid_parameter_sets) / len(self.agents)

        # Weight the diversity measures (can be adjusted)
        combined_diversity = (0.7 * strategy_diversity) + (0.3 * bid_diversity)

        return combined_diversity

    def get_most_fit_agent(self):
        """Gets the agent with highest fitness"""
        if not self.agents:
            return None
        return max(self.agents, key=lambda a: a.fitness)

    def get_least_fit_agent(self):
        """Gets the agent with lowest fitness"""
        if not self.agents:
            return None
        return min(self.agents, key=lambda a: a.fitness)

    def get_unique_most_fit_agent(self):
        """Gets the unique most fit agent if one exists"""
        if not self.agents:
            return None

        max_fitness = max(agent.fitness for agent in self.agents)
        top_agents = [agent for agent in self.agents if agent.fitness == max_fitness]

        return top_agents[0] if len(top_agents) == 1 else None

    def get_unique_least_fit_agent(self):
        """Gets the unique least fit agent if one exists"""
        if not self.agents:
            return None

        min_fitness = min(agent.fitness for agent in self.agents)
        bottom_agents = [agent for agent in self.agents if agent.fitness == min_fitness]

        return bottom_agents[0] if len(bottom_agents) == 1 else None

    def adjust_mutation_rate(self):
        """Adjusts mutation rate based on fitness trends"""
        if len(self.fitness_history) < 2:
            return

        previous_avg = self.fitness_history[-2]
        current_avg = self.fitness_history[-1]

        if current_avg <= previous_avg:
            self.mutation_rate = min(
                self.mutation_rate * 1.05, CONFIG["MAX_MUTATION_RATE"]
            )
        else:
            self.mutation_rate = max(
                self.mutation_rate * 0.95, CONFIG["MIN_MUTATION_RATE"]
            )

    def adjust_elite_percentage(self):
        """Adjusts elite percentage based on fitness trends"""
        if len(self.fitness_history) < 2:
            return

        previous_avg = self.fitness_history[-2]
        current_avg = self.fitness_history[-1]

        if current_avg > previous_avg * 1.05:
            self.elite_percentage = max(self.elite_percentage * 0.95, 0.002)
        elif current_avg < previous_avg * 0.95:
            self.elite_percentage = min(self.elite_percentage * 1.05, 0.5)

    def generate_next_generation(self):
        """Generates the next generation of agents"""
        sorted_agents = sorted(
            self.agents, key=lambda agent: agent.fitness, reverse=True
        )

        num_elite = max(1, int(self.size * self.elite_percentage))
        survivors = sorted_agents[:num_elite]

        # Remove non-survivors
        for agent in [a for a in self.agents if a not in survivors]:
            self.remove_agent(agent)

        # Generate offspring
        while len(self.agents) < self.size:
            parent1 = random.choice(survivors)
            parent2 = random.choice(survivors)

            if random.random() < 0.3:  # 30% chance of asexual reproduction
                offspring = parent1.asexual_offspring()
            else:
                offspring = parent1.sexual_offspring(parent2)

            self.add_agent(offspring)

    def remove_and_replace_agent(
        self, agent_to_remove, winner=None, meta_population=None, other_population=None
    ):
        """Removes an agent and replaces it with a new one"""
        if meta_population is None:
            meta_population = self.meta_population

        if self.meta_population:
            self.meta_population._update_display()

        # Remove the agent
        self.remove_agent(agent_to_remove)
        logging.info(
            f"Removed Agent {agent_to_remove.id} "
            f"from Population {self.population_id}."
        )

        # Remove scheduled games
        self.game_scheduler.remove_agent_games(agent_to_remove)
        if other_population:
            other_population.game_scheduler.remove_agent_games(agent_to_remove)
        logging.info(f"Removed scheduled games involving Agent {agent_to_remove.id}.")

        # Check for unique most fit agent
        unique_most_fit_agent = self.get_unique_most_fit_agent()

        if winner:
            # Replacement derived from winner's population
            winner_population = meta_population.populations[winner.population_id - 1]
            logging.info(
                "Replacement will be derived from "
                f"winner's population: {winner_population.population_id}"
            )

        if unique_most_fit_agent:
            # Asexual reproduction
            new_offspring = unique_most_fit_agent.asexual_offspring()
            new_offspring.population_id = self.population_id
            self.add_agent(new_offspring)
            logging.info(
                "Asexual Reproduction: Added Offspring "
                f"Agent {new_offspring.id} "
                f"to Population {self.population_id} "
                f"derived from {unique_most_fit_agent.id}."
            )
            return new_offspring
        else:
            # Sexual reproduction
            if other_population:
                # Cross-population reproduction
                fittest_current = self.get_most_fit_agent()
                fittest_other = other_population.get_most_fit_agent()
                if fittest_current and fittest_other:
                    child = fittest_current.sexual_offspring(fittest_other)
                    self.add_agent(child)
                    logging.info(
                        "Interpopulation Reproduction: "
                        f"Added Offspring Agent {child.id} "
                        f"to Population {self.population_id} "
                        f"via Sexual Reproduction between "
                        f"{fittest_current.id} and {fittest_other.id}."
                    )
                    return child
                else:
                    # Fallback to asexual reproduction
                    reproduction_agent = winner_population.get_most_fit_agent()
                    if reproduction_agent:
                        new_offspring = reproduction_agent.asexual_offspring()
                        self.add_agent(new_offspring)
                        logging.info(
                            "Fallback Asexual Reproduction: Added Offspring "
                            f"Agent {new_offspring.id} of "
                            f"{reproduction_agent.id} "
                            f"with fitness {reproduction_agent.fitness} "
                            f"to Population {self.population_id}."
                        )
                        return new_offspring
            else:
                # Intrapopulation reproduction
                fittest_agents = sorted(
                    self.agents, key=lambda a: a.fitness, reverse=True
                )[:2]
                if len(fittest_agents) < 2:
                    # Fallback to asexual reproduction
                    reproduction_agent = self.get_most_fit_agent()
                    if reproduction_agent:
                        new_offspring = reproduction_agent.asexual_offspring()
                        self.add_agent(new_offspring)
                        logging.info(
                            "Fallback Asexual Reproduction: Added Offspring "
                            f"Agent {new_offspring.id} "
                            f"of {reproduction_agent.id} "
                            f"with fitness {reproduction_agent.fitness} "
                            f"to Population {self.population_id}."
                        )
                        return new_offspring
                else:
                    # Sexual reproduction within population
                    parent1, parent2 = fittest_agents
                    child = parent1.sexual_offspring(parent2)
                    self.add_agent(child)
                    logging.info(
                        f"Added Offspring Agent {child.id} to Population "
                        f"{self.population_id} via Sexual Reproduction "
                        f"between {parent1.id} and {parent2.id}."
                    )
                    return child

        # Maintain population size
        self.enforce_population_size()
        return None

    def report_population_status(self):
        """Updates population status with comprehensive logging"""
        if not self.agents:
            logging.info("No agents remaining in the population.")
            return

        stats = [
            f"Population {self.population_id} Status",
            "=" * 118,
            f"Size: {len(self.agents)} | "
            f"Average Fitness: {self.calculate_average_fitness():.2f} | "
            f"Mutation Rate: {self.mutation_rate:.4f}",
            "",
            "Most Fit Agent:",
            f"  ID: {self.get_most_fit_agent().id}",
            f"  Fitness: {self.get_most_fit_agent().fitness}",
            f"  Games: {self.get_most_fit_agent().game_counter}",
            "",
            "Least Fit Agent:",
            f"  ID: {self.get_least_fit_agent().id}",
            f"  Fitness: {self.get_least_fit_agent().fitness}",
            f"  Games: {self.get_least_fit_agent().game_counter}",
        ]

        if self.meta_population and self.meta_population.display_coordinator:
            self.meta_population.display_coordinator.stats_display._update_section(
                f"Population {self.population_id} Stats", stats
            )

        # Log the complete stats
        log_message = "Population Status:\n" + "\n".join(stats)
        logging.info(log_message)

    def _log_population_status(self):
        """Logs population status information without printing to stdout"""
        oldest_agent = min(self.agents, key=lambda a: int(a.id.split("-")[1]))
        most_experienced = max(self.agents, key=lambda a: a.game_counter)
        most_fit = self.get_most_fit_agent()
        least_fit = self.get_least_fit_agent()

        logging.info(
            f"Population {self.population_id} Status:\n"
            f"Size: {len(self.agents)}\n"
            f"Average Fitness: {self.calculate_average_fitness():.2f}\n"
            f"Mutation Rate: {self.mutation_rate:.4f}\n"
            f"Elite Percentage: {self.elite_percentage:.4f}\n"
            f"Most Fit: {most_fit.id} (Fitness: {most_fit.fitness})\n"
            f"Least Fit: {least_fit.id} (Fitness: {least_fit.fitness})\n"
            f"Oldest: {oldest_agent.id} (Games: {oldest_agent.game_counter})\n"
            f"Most Experienced: {most_experienced.id} (Games: {most_experienced.game_counter})"
        )


class MetaPopulation:
    """Manages multiple populations and their interactions"""

    def __init__(
        self,
        num_populations=CONFIG["NUM_POPULATIONS"],
        population_size=CONFIG["POPULATION_SIZE"],
    ):
        # Initialize display refresh tracking
        self.last_refresh = time.time()
        self.min_refresh_interval = 0.001  # 1ms minimum between refreshes
        self.force_refresh_interval = 1.0  # Force full refresh every second

        # Initialize populations
        self.populations = [
            Population(size=population_size, population_id=i + 1, meta_population=self)
            for i in range(num_populations)
        ]
        self.num_populations = num_populations
        self.population_size = population_size
        self.total_population_size = num_populations * population_size
        self.current_generation = 0

        # Initialize display coordinator
        self.display_coordinator = DisplayCoordinator()

        # Initialize tracking
        self.previous_most_fit_agent = self.get_unique_most_fit_agent()
        self.previous_least_fit_agent = self.get_unique_least_fit_agent()
        self.interpopulation_game_scheduler = GameScheduler()

        self._log_initial_state()

    def _should_refresh_display(self):
        """Checks if enough time has passed to refresh the display"""
        current_time = time.time()
        if current_time - self.last_refresh > self.min_refresh_interval:
            self.last_refresh = current_time
            return True
        return False

    def _update_display(self):
        """Updates the display if enough time has passed"""
        current_time = time.time()
        if current_time - self.last_refresh > 0.1:  # Force update every 0.1 seconds
            if self.display_coordinator and self.display_coordinator.stats_display:
                # Force full refresh of all sections
                self.display_coordinator.stats_display.update_all()
                # Log complete contents of each section
                self._log_display_state()
            self.last_refresh = current_time

    def _log_display_state(self):
        """Logs complete state of all display sections"""
        if self.display_coordinator and self.display_coordinator.stats_display:
            display = self.display_coordinator.stats_display
            for section_name, section_info in display.display_buffer.items():
                content = section_info.get("content", [])
                if content:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    log_message = (
                        f"Display Section [{timestamp}] - {section_name}\n"
                        + "\n".join(str(line) for line in content)
                    )
                    logging.info(log_message)

    def _log_initial_state(self):
        """Logs the initial state of the metapopulation"""
        if self.previous_most_fit_agent:
            logging.info(
                "MetaPopulation - Initial Most Fit Agent: "
                f"{self.previous_most_fit_agent.id} from Population "
                f"{self.previous_most_fit_agent.population_id} with "
                f"fitness {self.previous_most_fit_agent.fitness}"
            )
        else:
            logging.info("MetaPopulation - No unique most fit agent at initialization.")

        if self.previous_least_fit_agent:
            logging.info(
                "MetaPopulation - Initial Least Fit Agent: "
                f"{self.previous_least_fit_agent.id} from Population "
                f"{self.previous_least_fit_agent.population_id} with "
                f"fitness {self.previous_least_fit_agent.fitness}"
            )
        else:
            logging.info(
                "MetaPopulation - No unique least fit agent at initialization."
            )

    def evolve(
        self,
        generations=CONFIG["MAX_GENERATIONS"],
        report_event=None,
        continue_event=None,
    ):
        """Main evolution loop with continuous display updates"""
        try:
            for generation in range(generations):
                self._update_display()
                self.current_generation = generation
                logging.info(f"Starting Meta Generation {generation + 1}")

                # Evolution phases with display updates
                self._evolve_populations(report_event, continue_event)
                self._update_display()

                self.cross_population_reproduction()
                self._update_display()

                self.enforce_population_size()
                self._update_display()

                self.conduct_elite_matches()
                self._update_display()

                self.execute_interpopulation_elite_games()
                self._update_display()

                # Update metapopulation stats display
                if self.display_coordinator:
                    self.display_coordinator.stats_display.update_metapopulation_stats(
                        self
                    )

                # Handle report requests
                if report_event and report_event.is_set():
                    self._handle_report_request(report_event, continue_event)

        except Exception as e:
            logging.error(f"Error in evolution: {e}", exc_info=True)
        finally:
            # Ensure display is cleaned up
            if self.display_coordinator:
                self.display_coordinator.cleanup()

    def _handle_report_request(self, report_event, continue_event):
        """Handles report generation requests"""
        logging.info("Generating Metapopulation Report")
        # Update all displays
        self.display_coordinator.stats_display.update_metapopulation_stats(self)
        for population in self.populations:
            self.display_coordinator.stats_display.update_population_stats(population)

        report_event.clear()
        if continue_event:
            continue_event.wait()
            continue_event.clear()

    def report_metapopulation_status(self):
        """Updates metapopulation status in fixed position display"""
        # Update the fixed position display
        if self.display_coordinator and self.display_coordinator.stats_display:
            self.display_coordinator.stats_display.update_metapopulation_stats(self)

        # Log status without printing to stdout
        self._log_metapopulation_status()

    def _log_metapopulation_status(self):
        """Logs metapopulation status without printing to stdout"""
        total_agents = sum(len(p.agents) for p in self.populations)
        total_fitness = sum(sum(a.fitness for a in p.agents) for p in self.populations)
        avg_fitness = total_fitness / total_agents if total_agents > 0 else 0

        logging.info(
            f"Metapopulation Status:\n"
            f"Generation: {self.current_generation + 1}\n"
            f"Total Populations: {len(self.populations)}\n"
            f"Total Agents: {total_agents}\n"
            f"Average Fitness: {avg_fitness:.2f}"
        )

    def _evolve_populations(self, report_event, continue_event):
        """Evolves individual populations with display updates"""
        for population in self.populations:

            self._update_display()

            population.evaluate_fitness(
                meta_population=self,
                report_event=report_event,
                continue_event=continue_event,
            )

            # Update displays after population evaluation
            if self.display_coordinator:
                self.display_coordinator.stats_display.update_population_stats(
                    population
                )
                self.display_coordinator.stats_display.update_metapopulation_stats(self)

            population.adjust_elite_percentage()

            self._update_display()

            population.generate_next_generation()

            # Update displays after generation
            if self.display_coordinator:
                self.display_coordinator.stats_display.update_population_stats(
                    population
                )
                self.display_coordinator.stats_display.update_metapopulation_stats(self)

    def conduct_elite_matches(self):
        """Conducts elite matches between populations"""
        print("\n--- Conducting Elite Matches Between Populations ---")
        elite_percentage = CONFIG["ELITE_PERCENTAGE"]
        total_agents = self.num_populations * self.population_size
        num_elite = max(2, int(total_agents * elite_percentage))

        # Gather all agents
        all_agents = []
        for population in self.populations:
            all_agents.extend(population.agents)

        # Select elite agents
        elite_agents = sorted(all_agents, key=lambda a: a.fitness, reverse=True)[
            :num_elite
        ]

        non_elite_agents = [a for a in all_agents if a not in elite_agents]

        # Schedule elite matches
        for elite_agent in elite_agents:
            for opponent in non_elite_agents:
                if elite_agent.population_id != opponent.population_id:
                    self._update_display()
                    self.interpopulation_game_scheduler.schedule_game(
                        elite_agent, opponent, game_type="interpopulation_elite"
                    )

    def execute_interpopulation_elite_games(self):
        """Executes scheduled interpopulation elite games"""
        print("\n--- Executing Interpopulation Elite Games ---")

        while self.interpopulation_game_scheduler.scheduled_games:
            game_data = self.interpopulation_game_scheduler.get_next_game()
            if not game_data:
                break

            agent1, agent2, game_type = game_data
            population1 = self.get_population_of_agent(agent1)
            population2 = self.get_population_of_agent(agent2)

            if not population1 or not population2:
                continue

            # Play game
            game = Game(agent1, agent2, game_type=game_type, meta_population=self)
            winner = game.play()
            winner = winner
            self._update_display()

            # Handle results
            if winner:
                self._handle_interpopulation_game_results(game)

    def _handle_interpopulation_game_results(self, game):
        """Handles results of interpopulation games"""
        winner = game.winner
        loser = game.loser

        if winner and loser:
            winner_pop = self.get_population_of_agent(winner)
            loser_pop = self.get_population_of_agent(loser)

            if winner_pop and loser_pop:
                if loser.fitness < 0:
                    loser_pop.remove_and_replace_agent(
                        agent_to_remove=loser,
                        winner=winner,
                        meta_population=self,
                        other_population=winner_pop,
                    )

    def cross_population_reproduction(self):
        """Facilitates reproduction between populations"""
        print("\n--- Conducting Cross-Population Reproduction ---")
        elite_percentage = CONFIG["ELITE_PERCENTAGE"]

        # Collect elite agents from each population
        elite_agents = []
        for population in self.populations:
            num_elite = max(2, int(population.size * elite_percentage))
            population_elite = sorted(
                population.agents, key=lambda a: a.fitness, reverse=True
            )[:num_elite]
            elite_agents.extend(population_elite)

        # Perform cross-population reproduction
        random.shuffle(elite_agents)
        for i in range(0, len(elite_agents), 2):
            if i + 1 >= len(elite_agents):
                break

            self._update_display()

            parent1 = elite_agents[i]
            parent2 = elite_agents[i + 1]

            if parent1.population_id != parent2.population_id:
                child = parent1.sexual_offspring(parent2)
                parent_pop = self.populations[parent1.population_id - 1]
                parent_pop.add_agent(child)

    def get_population_of_agent(self, agent):
        """Finds the population containing the given agent"""
        for population in self.populations:
            if agent in population.agents:
                return population
        return None

    def get_most_fit_agent(self):
        """Gets the most fit agent across all populations"""
        most_fit = None
        max_fitness = float("-inf")
        self._update_display()
        for population in self.populations:
            agent = population.get_most_fit_agent()
            if agent and agent.fitness > max_fitness:
                max_fitness = agent.fitness
                most_fit = agent
        return most_fit

    def get_least_fit_agent(self):
        """Gets the least fit agent across all populations"""
        least_fit = None
        min_fitness = float("inf")
        self._update_display()
        for population in self.populations:
            agent = population.get_least_fit_agent()
            if agent and agent.fitness < min_fitness:
                min_fitness = agent.fitness
                least_fit = agent
        return least_fit

    def get_unique_most_fit_agent(self):
        """Gets the unique most fit agent if one exists"""
        most_fit_agents = []
        max_fitness = float("-inf")

        self._update_display()  # Use the managed display update

        for population in self.populations:
            agent = population.get_most_fit_agent()
            if agent:
                if agent.fitness > max_fitness:
                    most_fit_agents = [agent]
                    max_fitness = agent.fitness
                elif agent.fitness == max_fitness:
                    most_fit_agents.append(agent)
        return most_fit_agents[0] if len(most_fit_agents) == 1 else None

    def get_unique_least_fit_agent(self):
        """Gets the unique least fit agent if one exists"""
        least_fit_agents = []
        min_fitness = float("inf")
        self._update_display()
        for population in self.populations:
            agent = population.get_least_fit_agent()
            if agent:
                if agent.fitness < min_fitness:
                    least_fit_agents = [agent]
                    min_fitness = agent.fitness
                elif agent.fitness == min_fitness:
                    least_fit_agents.append(agent)
        return least_fit_agents[0] if len(least_fit_agents) == 1 else None

    def enforce_population_size(self):
        """Ensures all populations maintain their size"""
        for population in self.populations:
            population.enforce_population_size()

    def _report_overall_statistics(
        self, total_agents, overall_avg_fitness, overall_diversity, all_agents
    ):

        self._update_display()

        """Reports overall metapopulation statistics"""
        print("\n--- Overall Metapopulation Statistics ---")
        print(f"Total Agents: {total_agents}")
        print(f"Average Fitness: {overall_avg_fitness:.2f}")
        print(f"Overall Diversity: {overall_diversity:.4f}")

        # Report most fit agent
        most_fit = max(all_agents, key=lambda a: a.fitness)
        print(
            f"Most Fit Agent: {most_fit.id} from Population "
            f"{most_fit.population_id} (Fitness: {most_fit.fitness})"
        )

        # Report least fit agent
        least_fit = min(all_agents, key=lambda a: a.fitness)
        print(
            f"Least Fit Agent: {least_fit.id} from Population "
            f"{least_fit.population_id} (Fitness: {least_fit.fitness})"
        )


# Utility Functions
def print_strategy(tree, indent=0):
    """Pretty prints a strategy decision tree"""
    if isinstance(tree, dict):
        print("  " * indent + f"Condition: {tree['condition']}")
        print("  " * indent + "True:")
        print_strategy(tree["true"], indent + 1)
        print("  " * indent + "False:")
        print_strategy(tree["false"], indent + 1)
    else:
        print("  " * indent + f"Action: {tree}")


# Main Execution Logic


def profile_meta_population():
    """Profiles the metapopulation performance"""
    pr = cProfile.Profile()
    meta_population = MetaPopulation(
        num_populations=CONFIG["NUM_POPULATIONS"],
        population_size=CONFIG["POPULATION_SIZE"],
    )

    pr.enable()
    meta_population.evolve(generations=1)
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(10)
    print(s.getvalue())


def main():
    """Main execution function"""
    configure_logging()

    try:
        print(f"Using matplotlib backend: {matplotlib.get_backend()}")
        Game.total_games_played = 0

        # Initialize events for reporting
        report_event = threading.Event()
        continue_event = threading.Event()

        # Initialize and start key listener
        key_listener = KeyListener(report_event, continue_event)
        key_listener.start()

        # Create and evolve meta-population
        meta_population = MetaPopulation(
            num_populations=CONFIG["NUM_POPULATIONS"],
            population_size=CONFIG["POPULATION_SIZE"],
        )

        # Profile first generation
        if CONFIG.get("PROFILE_FIRST_GEN", False):
            profile_meta_population()

        # Main evolution loop
        for generation in range(CONFIG["MAX_GENERATIONS"]):
            print(f"\n--- Meta Generation {generation + 1} ---")
            logging.info(f"MetaPopulation - Starting Meta Generation {generation + 1}.")
            meta_population._update_display()
            meta_population.evolve(
                generations=1, report_event=report_event, continue_event=continue_event
            )

        # Clean up
        key_listener.stop_flag = True
        key_listener.join()

        # Final report
        print("\nFinal Strategy of Most Fit Agent:")
        most_fit = meta_population.get_most_fit_agent()
        if most_fit:
            print_strategy(most_fit.strategy.decision_tree)

    except Exception as e:
        logging.error(f"Error in main execution: {e}", exc_info=True)
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        unittest.main(argv=["first-arg-is-ignored"])
    else:
        try:
            profiler = cProfile.Profile()
            profiler.enable()
            main()
            profiler.disable()
            stats = pstats.Stats(profiler).sort_stats("cumtime")
            stats.print_stats(10)
        except KeyboardInterrupt:
            print("\nExecution interrupted by user")
            sys.exit(0)
