import cProfile
import logging
import math
import pstats
import random
import textwrap
import unittest
from copy import deepcopy
from logging.handlers import RotatingFileHandler

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("TkAgg")  # Use 'TkAgg' backend

# Create a logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Set the desired logging level

# Create a RotatingFileHandler
handler = RotatingFileHandler(
    "game_simulation.log",
    maxBytes=500 * 1024 * 1024,  # 500 MB per log file
    backupCount=50,  # Keep up to 50 backup files
    delay=True,  # Delays file creation until the first log message
)

# Create a formatter and set it for the handler
formatter = logging.Formatter(
    '{"time": "%(asctime)s", '
    '"level": "%(levelname)s", '
    '"message": "%(message)s"}'
)

handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(handler)

# (Optional) Also log to console with a StreamHandler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)  # Set console log level
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class TestPopulationLogging(unittest.TestCase):
    def test_unique_least_fit_agent_logging(self):
        population = Population(size=3, population_id=1, meta_population=None)
        # Set agent fitnesses to create a unique least fit agent
        population.agents[0].fitness = 10
        population.agents[1].fitness = 5
        population.agents[2].fitness = 15
        unique_least_fit = population.get_unique_least_fit_agent()
        self.assertIsNotNone(unique_least_fit)
        self.assertEqual(unique_least_fit.id, population.agents[1].id)

    def test_non_unique_least_fit_agent_logging(self):
        population = Population(size=3, population_id=1)
        # Set agent fitnesses to create non-unique least fit agents
        population.agents[0].fitness = 5
        population.agents[1].fitness = 5
        population.agents[2].fitness = 10
        unique_least_fit = population.get_unique_least_fit_agent()
        self.assertIsNone(unique_least_fit)

    def test_unique_most_fit_agent_logging(self):
        population = Population(size=3, population_id=1)
        # Set agent fitnesses to create a unique most fit agent
        population.agents[0].fitness = 10
        population.agents[1].fitness = 15
        population.agents[2].fitness = 5
        unique_most_fit = population.get_unique_most_fit_agent()
        self.assertIsNotNone(unique_most_fit)
        self.assertEqual(unique_most_fit.id, population.agents[1].id)

    def test_non_unique_most_fit_agent_logging(self):
        population = Population(size=3, population_id=1)
        # Set agent fitnesses to create non-unique most fit agents
        population.agents[0].fitness = 15
        population.agents[1].fitness = 15
        population.agents[2].fitness = 10
        unique_most_fit = population.get_unique_most_fit_agent()
        self.assertIsNone(unique_most_fit)


class TestCrossPopulationReproduction(unittest.TestCase):
    def setUp(self):
        self.meta_population = MetaPopulation(
            num_populations=2, population_size=10
        )

    def test_mutation_rate_access(self):
        population1 = self.meta_population.populations[0]
        population2 = self.meta_population.populations[1]

        # Ensure populations have agents
        self.assertTrue(len(population1.agents) > 0)
        self.assertTrue(len(population2.agents) > 0)

        # Assign known mutation rates
        population1.mutation_rate = 0.1
        population2.mutation_rate = 0.2

        # Collect elite agents
        self.meta_population.cross_population_reproduction()

        # Check that offspring strategies are mutated at correct mutation rate
        # This would require inspecting the offspring's strategy
        # For brevity, this is left as a conceptual test


class TestGameCounter(unittest.TestCase):
    def test_game_counter_increment(self):
        agent1 = Agent(population_id=1)
        agent2 = Agent(population_id=1)
        initial_counter1 = agent1.game_counter
        initial_counter2 = agent2.game_counter

        game = Game(agent1, agent2)
        game.play()

        self.assertEqual(agent1.game_counter, initial_counter1 + 1)
        self.assertEqual(agent2.game_counter, initial_counter2 + 1)


class TestPopulation(unittest.TestCase):
    def setUp(self):
        self.meta_population = MetaPopulation(
            num_populations=2, population_size=10
        )
        self.population1 = self.meta_population.populations[0]
        self.population2 = self.meta_population.populations[1]
        self.agent1 = self.population1.agents[0]
        self.agent2 = self.population2.agents[0]

    def test_remove_agent_correct_population(self):
        # Remove agent1 from population1
        self.population1.remove_and_replace_agent(agent_to_remove=self.agent1)
        self.assertNotIn(self.agent1, self.population1.agents)
        self.assertIn(
            self.agent1, self.population1.all_agents
        )  # Assuming all_agents retains history

    def test_remove_agent_wrong_population(self):
        # Attempt to remove agent1 from population2
        self.population2.remove_and_replace_agent(agent_to_remove=self.agent1)
        self.assertIn(
            self.agent1, self.population1.agents
        )  # Should still be in population1
        self.assertNotIn(self.agent1, self.population2.agents)

    def test_schedule_game_and_remove_agent_games(self):
        # Schedule a game involving agent1 and agent2
        self.population1.game_scheduler.schedule_game(self.agent1, self.agent2)
        self.assertIn(
            (self.agent1, self.agent2),
            self.population1.game_scheduler.scheduled_games,
        )

        # Remove agent1 and ensure the game is removed
        self.population1.remove_and_replace_agent(agent_to_remove=self.agent1)
        self.population1.game_scheduler.remove_agent_games(self.agent1)
        self.assertNotIn(
            (self.agent1, self.agent2),
            self.population1.game_scheduler.scheduled_games,
        )


def wrap_text(text, width=100):
    """
    Wraps the input text to the specified width.

    Parameters:
        text (str): The text to wrap.
        width (int): The maximum number of characters per line.

    Returns:
        str: The wrapped text with newline characters.
    """
    return "\n".join(textwrap.wrap(text, width=width))


def format_bid_range(bid_range):
    """
    Formats a bid range list into a percentage string with two decimal places.

    Args:
        bid_range (list): A list of two floats representing the bid range.

    Returns:
        str: A formatted string like '1.00% - 5.00%'.
    """
    return f"{bid_range[0]*100:.2f}% - {bid_range[1]*100:.2f}%"


plt.ion()  # Turn on interactive mode


# 1. Define the Strategy Class
#
class Strategy:
    def __init__(self, decision_tree=None, bid_params=None):
        if decision_tree is None:
            self.decision_tree = self.generate_random_tree()
        else:
            self.decision_tree = decision_tree
        if bid_params is None:
            self.bid_params = self.generate_random_bid_params()
        else:
            self.bid_params = bid_params

    def mutate(self, mutation_rate=0.05):
        # Use the passed mutation rate instead of a fixed value

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
                    node = random.choice(
                        ["u", "d", "l", "r", "ul", "ur", "dl", "dr"]
                    )
            return node

        self.decision_tree = mutate_node(self.decision_tree)

        # Mutate bid_params
        # Mutate thresholds
        if random.random() < mutation_rate * 5:
            index = random.choice([0, 1])
            change = round(random.randint(-50, 50) * (1 + 20 * mutation_rate))
            self.bid_params["thresholds"][index] += change
            # Ensure thresholds remain sorted
            self.bid_params["thresholds"] = sorted(
                self.bid_params["thresholds"]
            )

        # Mutate bid ranges
        for param in ["low_bid_range", "medium_bid_range", "high_bid_range"]:
            if random.random() < mutation_rate * 5:
                index = random.choice([0, 1])
                change = random.uniform(-0.05, 0.05) * (1 + 20 * mutation_rate)
                self.bid_params[param][index] += change
                # Clamp the values between 0.01 and 0.5
                self.bid_params[param][index] = min(
                    max(self.bid_params[param][index], 0.01), 0.5
                )
                # Sort the bid ranges
                self.bid_params[param] = sorted(self.bid_params[param])

        # Mutate bid_constant
        if random.random() < mutation_rate * 5:
            change = round(
                random.randint(-10, 10) * (1 + 20 * mutation_rate)
            )  # Adjust the mutation step as needed
            self.bid_params["bid_constant"] += change
            # Clamp bid_constant between -500 and 500 to prevent extreme values
            self.bid_params["bid_constant"] = min(
                max(self.bid_params["bid_constant"], -500), 500
            )

    def generate_random_bid_params(self):
        thresholds = sorted(random.sample(range(-5000, 5000), 2))
        low_bid_range = sorted(
            [random.uniform(0.001, 0.05), random.uniform(0.01, 0.1)]
        )
        medium_bid_range = sorted(
            [random.uniform(0.01, 0.15), random.uniform(0.02, 0.2)]
        )
        high_bid_range = sorted(
            [random.uniform(0.02, 0.2), random.uniform(0.05, 0.3)]
        )
        bid_constant = random.randint(-100, 100)  # Initialize bid_constant
        return {
            "thresholds": thresholds,
            "low_bid_range": low_bid_range,
            "medium_bid_range": medium_bid_range,
            "high_bid_range": high_bid_range,
            "bid_constant": bid_constant,
        }

    def is_adjacent(self, pos1, pos2):
        """
        Determines if pos2 is adjacent to pos1 (including diagonally).
        """
        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])
        return max(dx, dy) == 1

    def get_bid_fraction(self, game_state):
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

        # Normalize point difference
        total_points = my_points + opp_points + 1  # Avoid division by zero
        normalized_diff = point_difference / total_points

        # Use a probabilistic approach based on normalized difference
        # The more ahead the agent is, the less it bids, and vice versa
        bid_fraction += max(
            0.001, 0.005 - 0.005 * normalized_diff
        )  # Ensure minimum bid fraction

        # Clamp bid_fraction to stay within 1% to 50%
        bid_fraction = min(max(bid_fraction, 0.01), 0.5)

        return bid_fraction

    def make_decision(self, game_state):
        return self.evaluate_tree(self.decision_tree, game_state)

    def generate_random_tree(self, depth=6, min_depth=4):
        if depth <= 0:
            action = random.choice(
                ["u", "d", "l", "r", "ul", "ur", "dl", "dr"]
            )
            return action
        else:
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
                # Force conditionals to reach minimum depth
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
            return {
                "condition": condition,
                "true": true_branch,
                "false": false_branch,
            }

    def evaluate_tree(self, node, game_state, depth=0):
        if depth > 10:
            print("Maximum recursion depth reached in evaluate_tree.")
            return "u"  # Default action
        if isinstance(node, dict):
            condition = node["condition"]
            result = self.evaluate_condition(condition, game_state)
            if result:
                return self.evaluate_tree(node["true"], game_state, depth + 1)
            else:
                return self.evaluate_tree(node["false"], game_state, depth + 1)
        else:
            return node

    def evaluate_condition(self, condition, game_state):
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
            return (
                x == 0 or y == 0 or x == board_size - 1 or y == board_size - 1
            )
        elif condition == "random":
            return random.choice([True, False])
        elif condition == "opponent_left":
            return (
                game_state["opponent_position"][0]
                < game_state["my_position"][0]
            )
        elif condition == "opponent_right":
            return (
                game_state["opponent_position"][0]
                > game_state["my_position"][0]
            )
        elif condition == "opponent_above":
            return (
                game_state["opponent_position"][1]
                > game_state["my_position"][1]
            )
        elif condition == "opponent_below":
            return (
                game_state["opponent_position"][1]
                < game_state["my_position"][1]
            )
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
            my_pos = game_state["my_position"]
            goal_pos = game_state["goal"]
            return self.is_adjacent(my_pos, goal_pos)
        else:
            return False

    def manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def get_random_subtree(self, node, parent=None, key=None):
        """
        Recursively selects a random subtree from the decision tree.

        Args:
            node (dict or str): The current node in the decision tree.
            parent (dict, optional): The parent node containing the current
            node.key (str, optional): The key in the parent dict
            ('true' or 'false')
            where the current node is stored.

        Returns:
            tuple: (subtree, parent, key)
                - subtree: The selected subtree.
                - parent: The parent node containing the subtree.
                - key: The key in the parent dict ('true' or 'false') where
                       the subtree is stored.
        """
        if isinstance(node, dict):
            if random.random() < 0.5:
                return node, parent, key  # Selecting the current node
            else:
                branch = random.choice(["true", "false"])
                return self.get_random_subtree(node[branch], node, branch)
        else:
            return node, parent, key  # Leaf node

    def crossover(self, other_strategy, min_depth=4):
        # Enhanced tree-based crossover
        # Deep copy the decision trees to avoid modifying parents
        tree1 = deepcopy(self.decision_tree)
        tree2 = deepcopy(other_strategy.decision_tree)

        # Select random crossover points in both trees
        subtree1, parent1, key1 = self.get_random_subtree(tree1)
        subtree2, parent2, key2 = self.get_random_subtree(tree2)

        # Swap the subtrees between the two trees
        if parent1 and key1:
            parent1[key1] = subtree2
        else:
            tree1 = subtree2  # If subtree1 is the root

        if parent2 and key2:
            parent2[key2] = subtree1
        else:
            tree2 = subtree1  # If subtree2 is the root

        # Crossover bid_params more uniformly
        new_bid_params = {}
        for key in self.bid_params:
            if key == "bid_constant":
                # For bid_constant, take the average of both parents
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

        # Decide which tree to use for the child (randomly pick one)
        child_tree = random.choice([tree1, tree2])

        # Create the child strategy
        child_strategy = Strategy(child_tree, bid_params=new_bid_params)

        # Ensure the child strategy tree has at least min_depth
        child_strategy.enforce_minimum_depth(min_depth)

        return child_strategy

    def calculate_tree_depth(self, node):
        if isinstance(node, dict):
            return 1 + max(
                self.calculate_tree_depth(node["true"]),
                self.calculate_tree_depth(node["false"]),
            )
        else:
            return 1

    def enforce_minimum_depth(self, min_depth=4):
        current_depth = self.calculate_tree_depth(self.decision_tree)
        if current_depth >= min_depth:
            return  # Nothing to do
        else:
            # Increase depth by replacing shallow subtrees
            # with deeper random trees
            def replace_shallow(node, current_depth=1):
                if isinstance(node, dict):
                    if current_depth < min_depth:
                        node["true"] = replace_shallow(
                            node["true"], current_depth + 1
                        )
                        node["false"] = replace_shallow(
                            node["false"], current_depth + 1
                        )
                return node

            self.decision_tree = replace_shallow(
                self.decision_tree, current_depth=1
            )
            # After replacement, if depth is still less than min_depth,
            # regenerate the entire tree
            if self.calculate_tree_depth(self.decision_tree) < min_depth:
                self.decision_tree = self.generate_random_tree(
                    depth=min_depth, min_depth=min_depth
                )


class Player:
    _id_counter = 1  # Class variable for unique ID generation

    def __init__(self, player_type="Agent"):
        self.id = f"{player_type}-{Player._id_counter:06d}"
        Player._id_counter += 1
        self.fitness = 500  # Default fitness value

    def get_bid(self, point_budget, opponent_point_budget):
        raise NotImplementedError(
            "This method should be implemented by subclasses."
        )

    def get_move(
        self, my_position, opponent_position, board_size, goal, opponent_goal
    ):
        raise NotImplementedError(
            "This method should be implemented by subclasses."
        )

    def increment_game_counter(self):
        """
        Default implementation does nothing.
        Subclasses can override this method if they track game counts.
        """


# 2. Define the Agent Class
#
class Agent(Player):
    _id_counter = 1  # Class variable for unique ID generation

    def __init__(
        self, strategy=None, genealogy=None, population_id=1, parents=None
    ):
        super().__init__(player_type="A")
        self.population_id = population_id
        if strategy is None:
            self.strategy = Strategy()
        else:
            self.strategy = strategy
        if genealogy is None:
            self.genealogy = set([self.id])  # Starts with itself
        else:
            self.genealogy = set(genealogy).union({self.id})
        self.game_counter = 0  # Total games played across generations
        self.games_played_this_generation = (
            0  # Games played in the current generation
        )
        self.games_played = (
            0  # Total games played **New Attribute: Total Games Played**
        )

        # **Fitness Tracking Attributes**
        self.initial_fitness = (
            self.fitness
        )  # Store initial fitness (default 500)
        self.sum_fitness_change = 0  # Sum of fitness changes
        self.sum_fitness_change_squared = 0  # Sum of squared fitness changes

        # **New Attributes for Prolific Agent Tracking**
        self.parents = parents if parents else []  # List of parent agent IDs
        self.offspring_count = 0  # Number of living immediate offspring

    def asexual_offspring(self):
        """
        Produces an asexual offspring (clone) of this agent
        with potential mutations.
        """
        # Deep copy the strategy to ensure a separate instance
        offspring_strategy = deepcopy(self.strategy)
        # Apply mutation to the offspring's strategy
        offspring_strategy.mutate(mutation_rate=0.05)
        # Create a new Agent with the mutated strategy and inherit genealogy
        offspring = Agent(
            strategy=offspring_strategy,
            genealogy=self.genealogy,
            population_id=self.population_id,
            parents=[self.id],  # Set parent to self for asexual reproduction
        )
        self.offspring_count += 1  # Increment offspring count
        return offspring

    def sexual_offspring(self, other_agent):
        """
        Produces a sexual offspring by crossing over with another agent.
        """
        # Perform crossover to create a new strategy
        child_strategy = self.strategy.crossover(other_agent.strategy)
        child_strategy.mutate(mutation_rate=0.05)
        # Combine genealogies
        child_genealogy = self.genealogy.union(other_agent.genealogy)
        # Create a new Agent with the combined genealogy and parent IDs
        child = Agent(
            strategy=child_strategy,
            genealogy=child_genealogy,
            population_id=self.population_id,
            parents=[
                self.id,
                other_agent.id,
            ],  # Set parents for sexual reproduction
        )
        # Increment offspring counts for both parents
        self.offspring_count += 1
        other_agent.offspring_count += 1
        return child

    def increment_game_counter(self):
        """
        Overrides the base method to increment the game counter.
        """
        self.game_counter += 1
        print(
            f"Agent {self.id} from Population {self.population_id} "
            f"game_counter incremented to {self.game_counter}"
        )
        logging.info(
            f"Agent {self.id} from Population {self.population_id} "
            f"game_counter incremented to {self.game_counter}"
        )

    def get_bid(self, point_budget, opponent_point_budget):
        game_state = {
            "my_points": point_budget,
            "opponent_points": opponent_point_budget,
        }
        bid_fraction = self.strategy.get_bid_fraction(game_state)
        # Incorporate bid_constant into the bid calculation
        bid_constant = self.strategy.bid_params["bid_constant"]
        bid = max(1, int(point_budget * bid_fraction) + bid_constant)
        return bid

    def get_move(
        self, my_position, opponent_position, board_size, goal, opponent_goal
    ):
        game_state = {
            "my_position": my_position,
            "opponent_position": opponent_position,
            "goal": goal,
            "opponent_goal": opponent_goal,
            "board_size": board_size,
        }
        direction = self.strategy.make_decision(game_state)
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

        # Generate all possible valid moves
        valid_moves = []
        for move_pos in potential_moves.values():
            if self.is_valid_move(move_pos, opponent_position, board_size):
                valid_moves.append(move_pos)

        # If the strategy's move is valid, use it
        move = potential_moves.get(direction)
        if move in valid_moves:
            return move
        else:
            # Strategy suggested an invalid move; pick a random valid move
            if valid_moves:
                return random.choice(valid_moves)
            else:
                # No valid moves available; stay in place (this should be rare)
                return my_position

    def is_valid_move(self, position, opponent_position, board_size):
        x, y = position
        return (
            0 <= x < board_size
            and 0 <= y < board_size
            and position != opponent_position
        )

    # Estimate Fitness Change Parameters Using Bayesian Methods**
    def estimate_fitness_change(self):
        """
        Estimates the fitness change per game as (min, mean, max)
        using Bayesian methods.
        - Initial mean estimated fitness change per game: 0
        - After some games: (total fitness change) / (games played + 12)
        - Confidence interval: 3 standard deviations

        Returns:
            tuple: (min_est, mean_est, max_est)
        """
        if self.games_played == 0:
            mean_est = 0
            std_dev = 1  # Initial standard deviation
        else:
            # Apply smoothing by adding 12 to the number of games
            mean_est = self.sum_fitness_change / (self.games_played + 12)
            # Calculate variance with Bessel's correction
            variance_est = (
                (
                    self.sum_fitness_change_squared
                    - (self.sum_fitness_change**2) / (self.games_played + 12)
                )
                / (self.games_played + 12 - 1)
                if self.games_played + 12 - 1 > 0
                else 1
            )
            std_dev = math.sqrt(variance_est) if variance_est > 0 else 12

        min_est = mean_est - 3 * std_dev
        max_est = mean_est + 3 * std_dev
        return (min_est, mean_est, max_est)


# 3. Define the HumanPlayer Class
#
class HumanPlayer(Player):
    def __init__(self):
        super().__init__(player_type="Human")
        print(f"Initialized HumanPlayer with id: {self.id}")
        # You can initialize additional attributes if needed

    def get_bid(self, point_budget, opponent_point_budget):
        """
        Allows a human player to input their bid.
        The opponent_point_budget is ignored in this implementation.
        """
        print(f"Your current point budget: {point_budget}")
        while True:
            bid_input = input(
                "Enter your bid (positive integer up to your"
                "current point budget): "
            ).strip()
            if bid_input.isdigit():
                bid = int(bid_input)
                if 0 < bid <= point_budget:
                    return bid
                else:
                    print("Invalid bid amount. Try again.")
            else:
                print("Invalid input. Please enter a positive integer.")

    def get_move(
        self, my_position, opponent_position, board_size, goal, opponent_goal
    ):
        """
        Allows a human player to input their move.
        """
        print(f"Your position: {my_position}")
        print(f"Opponent position: {opponent_position}")
        print(f"Your goal: {goal}")
        print(f"Opponent's goal: {opponent_goal}")
        print(
            "Enter your move: u (up), d (down), l (left), r (right), "
            "ul, ur, dl, dr:"
        )
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
        while True:
            move = input().strip()
            if move in potential_moves:
                move_pos = potential_moves[move]
                if self.is_valid_move(move_pos, opponent_position, board_size):
                    return move_pos
                else:
                    print(
                        "Invalid move (out of bounds or occupied). Try again."
                    )
            else:
                print("Invalid input. Try again.")

    def is_valid_move(self, position, opponent_position, board_size):
        x, y = position
        return (
            0 <= x < board_size
            and 0 <= y < board_size
            and position != opponent_position
        )


# 4a. Define the Game Class (Now Before Population Class)
#
class Game:
    total_games_played = 0  # Class variable to track total games

    @classmethod
    def get_next_game_number(cls):
        cls.total_games_played += 1
        return cls.total_games_played

    def __init__(self, player1, player2, board_size=8, visualize=False):
        self.game_number = Game.get_next_game_number()
        self.board_size = board_size
        self.players = {"player1": player1, "player2": player2}
        self.positions = {
            "player1": (0, 0),
            "player2": (board_size - 1, board_size - 1),
        }
        # Set fixed goals for each player
        self.goals = {
            "player1": (board_size - 1, board_size - 1),
            "player2": (0, 0),
        }
        self.point_budgets = {"player1": 100000, "player2": 100000}
        self.consecutive_turns_on_goal = {"player1": 0, "player2": 0}
        self.winner = None
        self.loser = None  # Initialize loser
        self.outcome_code = None  # Initialize outcome_code
        self.visualize = visualize
        self.history = []

    def is_valid_move(self, position, opponent_position):
        x, y = position
        return (
            0 <= x < self.board_size
            and 0 <= y < self.board_size
            and position != opponent_position
        )

    def manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def get_player_id(self, player):
        for pid, p in self.players.items():
            if p == player:
                return pid
        return None

    def get_fitness_change(self, player):
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
            # Add other positive outcomes if any
        elif player == self.loser:
            if self.outcome_code == "ran_out_of_points":
                fitness_change = -6 - player.fitness // 100
            elif self.outcome_code == "stayed_on_opponent_goal":
                fitness_change = -18 - player.fitness // 100
            elif self.outcome_code == "invalid_move":
                fitness_change = -28 - player.fitness // 100
            elif self.outcome_code != "error":
                fitness_change = -3 - player.fitness // 100
            # Add other negative outcomes if any
        fitness_change += 1
        return fitness_change

    def play(self):
        print(f"\nStarting Game {self.game_number}")
        logging.info(
            f"Starting Game {self.game_number} "
            f"between {self.players['player1'].id} "
            f"(Pop {self.players['player1'].population_id}) "
            f"and {self.players['player2'].id} "
            f"(Pop {self.players['player2'].population_id})"
        )

        try:
            # **Store Initial Fitness Before the Game**
            self.players["player1"].fitness
            self.players["player2"].fitness

            # Increment game counters for both agents
            self.players["player1"].increment_game_counter()
            self.players["player2"].increment_game_counter()
            turn = 0
            while not self.winner:
                # Both players submit bids
                bids = {}
                player1_points = self.point_budgets["player1"]
                player2_points = self.point_budgets["player2"]

                bids["player1"] = self.players["player1"].get_bid(
                    player1_points, player2_points
                )
                bids["player2"] = self.players["player2"].get_bid(
                    player2_points, player1_points
                )

                # Deduct bids from point budgets
                for player_id in bids:
                    self.point_budgets[player_id] -= bids[player_id]
                    if self.point_budgets[player_id] < 0:
                        # Ensure point budget doesn't go negative
                        self.point_budgets[player_id] = 0

                # Determine who moves based on bids
                if bids["player1"] > bids["player2"]:
                    mover_id = "player1"
                    non_mover_id = "player2"
                elif bids["player1"] < bids["player2"]:
                    mover_id = "player2"
                    non_mover_id = "player1"
                else:
                    # If bids are equal, randomly select who moves
                    mover_id = random.choice(["player1", "player2"])
                    non_mover_id = (
                        "player2" if mover_id == "player1" else "player1"
                    )

                mover = self.players[mover_id]
                self.players[non_mover_id]

                # Mover makes a move
                move = mover.get_move(
                    self.positions[mover_id],
                    self.positions[non_mover_id],
                    self.board_size,
                    self.goals[mover_id],
                    self.goals[non_mover_id],  # Pass opponent's goal
                )

                if self.is_valid_move(move, self.positions[non_mover_id]):
                    self.positions[mover_id] = move
                    # Reset consecutive turns on goal if moved away
                    if move != self.goals[non_mover_id]:
                        self.consecutive_turns_on_goal[mover_id] = 0
                    # Check if mover has reached their own goal
                    if move == self.goals[mover_id]:
                        # Mover wins by reaching goal
                        self.winner = self.players[mover_id]
                        self.loser = self.players[non_mover_id]
                        self.winning_reason = f"{mover_id} reached their goal."
                        self.outcome_code = "reached_goal"
                        self.update_fitness()
                        self.record_final_state(turn)
                        return self.winner
                else:
                    # Invalid move; opponent wins
                    self.winner = self.players[non_mover_id]
                    self.loser = self.players[mover_id]
                    self.winning_reason = f"{mover_id} made an invalid move."
                    self.outcome_code = "invalid_move"
                    self.update_fitness()
                    self.record_final_state(turn)
                    return self.winner

                # Non-mover stays in place
                # Check for consecutive turns on opponent's goal
                for pid in ["player1", "player2"]:
                    opponent_id = "player2" if pid == "player1" else "player1"
                    if self.positions[pid] == self.goals[opponent_id]:
                        self.consecutive_turns_on_goal[pid] += 1
                        # When a player stays on opponent's goal too long,
                        # player loses
                        if self.consecutive_turns_on_goal[pid] > 2:
                            self.winner = self.players[opponent_id]
                            self.loser = self.players[pid]
                            self.winning_reason = (
                                f"{pid} stayed on opponent's "
                            )
                            "goal for 3 consecutive turns."
                            self.outcome_code = "stayed_on_opponent_goal"
                            self.update_fitness()
                            self.record_final_state(turn)
                            return self.winner
                    else:
                        self.consecutive_turns_on_goal[pid] = 0

                # Check if any player cannot bid
                for pid in ["player1", "player2"]:
                    if self.point_budgets[pid] <= 0:
                        opponent_id = (
                            "player2" if pid == "player1" else "player1"
                        )
                        self.winner = self.players[opponent_id]
                        self.loser = self.players[pid]
                        self.winning_reason = f"{pid} ran out of points."
                        self.outcome_code = "ran_out_of_points"
                        print(
                            f"Player {pid} cannot bid. Player {opponent_id} "
                            "wins!"
                        )
                        self.update_fitness()
                        self.record_final_state(turn)
                        return self.winner

                # Record history if visualizing
                if self.visualize:
                    # Store positions and point budgets
                    self.history.append(
                        {
                            "player1": self.positions["player1"],
                            "player2": self.positions["player2"],
                            "points_p1": self.point_budgets["player1"],
                            "points_p2": self.point_budgets["player2"],
                            "turn": turn + 1,
                        }
                    )

                turn += 1
                if turn > 120:  # Adjust as needed
                    # Determine winner based on proximity to goal or points
                    distances = {
                        "player1": self.manhattan_distance(
                            self.positions["player1"], self.goals["player1"]
                        ),
                        "player2": self.manhattan_distance(
                            self.positions["player2"], self.goals["player2"]
                        ),
                    }
                    if distances["player1"] < distances["player2"]:
                        self.winner = self.players["player1"]
                        self.loser = self.players["player2"]
                        self.winning_reason = f"{self.players['player1'].id} "
                        "was closer to their goal based on proximity."
                        self.outcome_code = "proximity_tiebreak"
                    elif distances["player1"] > distances["player2"]:
                        self.winner = self.players["player2"]
                        self.loser = self.players["player1"]
                        self.winning_reason = f"{self.players['player2'].id} "
                        "was closer to their goal based on proximity."
                        self.outcome_code = "proximity_tiebreak"
                    else:
                        # Tie-breaker based on remaining points
                        if (
                            self.point_budgets["player1"]
                            > self.point_budgets["player2"]
                        ):
                            self.winner = self.players["player1"]
                            self.loser = self.players["player2"]
                            self.winning_reason = "Agent "
                            f"{self.players['player1'].id} had more remaining "
                            "points as a tiebreaker."
                            self.outcome_code = "points_tiebreak"
                        elif (
                            self.point_budgets["player1"]
                            < self.point_budgets["player2"]
                        ):
                            self.winner = self.players["player2"]
                            self.loser = self.players["player1"]
                            self.winning_reason = "Agent "
                            f"{self.players['player2'].id} had more remaining "
                            "points as a tiebreaker."
                            self.outcome_code = "points_tiebreak"
                        else:
                            # Random winner
                            self.winner = random.choice(
                                [
                                    self.players["player1"],
                                    self.players["player2"],
                                ]
                            )
                            self.loser = (
                                self.players["player2"]
                                if self.winner == self.players["player1"]
                                else self.players["player1"]
                            )
                            self.winning_reason = "The game ended in a tie "
                            f"based on proximity and points. {self.winner.id} "
                            "was randomly selected as the winner."
                            self.outcome_code = "random_tiebreak"
                    self.update_fitness()
                    self.record_final_state(turn)
                    return self.winner

        except Exception as e:
            print(f"An exception occurred during the game play: {e}")
            import traceback

            traceback.print_exc()
            self.winner = None
            self.loser = None
            self.winning_reason = "Error occurred during game play."
            self.outcome_code = "error"
            self.update_fitness()
            self.record_final_state(turn)
            return self.winner

        # Visualization and logging after the loop are removed
        # to prevent redundant execution

    def update_fitness(self):
        """
        Updates the fitness of both players based on the game's outcome.
        Ensures that fitness changes are applied consistently
        and only once per game.
        """
        if self.winner and self.loser:
            # Calculate fitness changes
            fitness_change_winner = self.get_fitness_change(self.winner)
            fitness_change_loser = self.get_fitness_change(self.loser)

            # Update fitness
            self.winner.fitness += fitness_change_winner
            self.loser.fitness += fitness_change_loser

            logging.info(
                f"Fitness Update - {self.winner.id}: +"
                f"{fitness_change_winner}, "
                f"{self.loser.id}: "
                f"{fitness_change_loser}"
            )

            # **Update Fitness Tracking Attributes**
            self.winner.sum_fitness_change += fitness_change_winner
            self.winner.sum_fitness_change_squared += fitness_change_winner**2
            self.winner.games_played += 1

            self.loser.sum_fitness_change += fitness_change_loser
            self.loser.sum_fitness_change_squared += fitness_change_loser**2
            self.loser.games_played += 1

            logging.info(
                f"After Game {self.game_number}: {self.winner.id}"
                f" (Pop {self.winner.population_id}) won against "
                f"{self.loser.id} (Pop {self.loser.population_id})"
            )
            logging.info(
                f"{self.winner.id} new fitness: {self.winner.fitness}"
            )
            logging.info(f"{self.loser.id} new fitness: {self.loser.fitness}")

            # Estimate fitness changes for the winner
            min_est_winner, mean_est_winner, max_est_winner = (
                self.winner.estimate_fitness_change()
            )
            logging.info(
                f"Estimated Fitness Change for {self.winner.id} "
                f"(Pop {self.winner.population_id}): "
                f"Min: {min_est_winner:.2f}, "
                f"Mean: {mean_est_winner:.2f}"
            )

            # Estimate fitness changes for the loser
            # min_est_loser, mean_est_loser, max_est_loser = (
            # self.loser.estimate_fitness_change()
            # )
            # logging.info(
            #    f"Estimated Fitness Change for {self.loser.id} "
            #    f"(Pop {self.loser.population_id}): "
            #    f"Min: {min_est_loser:.2f}, "
            #    f"Mean: {mean_est_loser:.2f}, "
            #    f"Max: {max_est_loser:.2f}"
            # )

    def record_final_state(self, turn):
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

    def get_strategy_string(self, strategy):
        # Simplify strategy representation for display
        def traverse(node):
            if isinstance(node, dict):
                condition = node["condition"]
                true_branch = traverse(node["true"])
                false_branch = traverse(node["false"])
                return f"({condition}? {true_branch} : {false_branch})"
            else:
                return node

        strategy_str = traverse(strategy.decision_tree)
        # Wrap the strategy string to 100 characters per line
        wrapped_strategy = wrap_text(strategy_str, width=100)
        return wrapped_strategy

    def format_bid_params(self, bid_params):
        """
        Formats the bid parameters into a readable string,
        including bid_constant.

        Args:
            bid_params (dict): The bid parameters of an agent.

        Returns:
            str: A formatted string of bid parameters.
        """
        return (
            f"Thresholds: {bid_params['thresholds']}\n"
            "Low Bid Range: "
            f"{format_bid_range(bid_params['low_bid_range'])}\n"
            "Medium Bid Range: "
            f"{format_bid_range(bid_params['medium_bid_range'])}\n"
            "High Bid Range: "
            f"{format_bid_range(bid_params['high_bid_range'])}\n"
            "Bid Constant: "
            f"{bid_params['bid_constant']}"
        )

    # VISUALIZE GAME
    def visualize_game(
        self, player1, player2, initial_fitness_p1, initial_fitness_p2
    ):
        # Prepare movement trails
        trail_length = 24  # Number of previous positions to display
        p1_trail = []
        p2_trail = []

        # Create a new figure with adjusted size
        fig, ax = plt.subplots(
            figsize=(22, 10)
        )  # Increased size to accommodate text

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
            wrapped_genealogy_p1 = wrap_text(genealogy_p1, width=100)
            wrapped_genealogy_p2 = wrap_text(genealogy_p2, width=100)

            # Generate strategy strings
            strategy_p1 = self.get_strategy_string(player1.strategy)
            strategy_p2 = self.get_strategy_string(player2.strategy)

            # Use initial fitness values
            fitness_p1 = initial_fitness_p1
            fitness_p2 = initial_fitness_p2

            # Agent 1 Bid Params Information
            bid_params_p1 = player1.strategy.bid_params
            bid_info_p1 = self.format_bid_params(bid_params_p1)  # Updated call

            # Agent 2 Bid Params Information
            bid_params_p2 = player2.strategy.bid_params
            bid_info_p2 = self.format_bid_params(bid_params_p2)  # Updated call

            # Print Agent 1 Information Text
            agent1_text = f"Player 1, {player1.id}\nPoints: {points_p1}\n"
            "Games Played: {player1.game_counter}"
            ax.text(
                -10,
                9.0,
                agent1_text,
                fontsize=12,
                color="blue",
                ha="left",
                va="top",
                clip_on=False,  # Prevent clipping
            )
            ax.text(
                -10,
                7.0,
                f"Fitness: {fitness_p1}",
                fontsize=11,
                color="blue",
                ha="left",
                va="top",
                clip_on=False,
            )
            ax.text(
                -10,
                6.5,
                f"Genealogy:\n{wrapped_genealogy_p1}\n\n"
                f"Strategy: {strategy_p1}\n\n"
                f"Bid Params:\n{bid_info_p1}",
                fontsize=5,
                color="blue",
                ha="left",
                va="top",
                clip_on=False,
            )

            # Print Agent 2 Information Text
            agent2_text = f"Player 2, {player2.id}\nPoints: {points_p2}\n"
            "Games Played: {player2.game_counter}"
            ax.text(
                self.board_size + 9,
                self.board_size + 1.0,
                agent2_text,
                fontsize=12,
                color="red",
                ha="right",
                va="top",
                clip_on=False,
            )
            ax.text(
                self.board_size + 9,
                self.board_size - 1.0,
                f"Fitness: {fitness_p2}",
                fontsize=11,
                color="red",
                ha="right",
                va="top",
                clip_on=False,
            )
            ax.text(
                self.board_size + 9,
                self.board_size - 1.5,
                f"Genealogy:\n{wrapped_genealogy_p2}\n\n"
                f"Strategy: {strategy_p2}\n\n"
                f"Bid Params:\n{bid_info_p2}",
                fontsize=5,
                color="red",
                ha="right",
                va="top",
                clip_on=False,
            )

            ax.legend(loc="upper left")
            ax.set_title(
                f"Agent vs Agent Game {self.game_number} - Turn {idx + 1}"
            )

            plt.draw()
            plt.pause(0.2)
            # plt.close(fig)
            # Remove plt.close(fig) to keep the plot open
            # Optionally, add a condition to close or manage the plot

        # plt.pause(2)

        # Re-plot the last state
        if self.history:
            last_state = self.history[-1]
            p1_pos = last_state["player1"]
            p2_pos = last_state["player2"]
            points_p1 = last_state["points_p1"]
            points_p2 = last_state["points_p2"]

            # Generate wrapped genealogy strings
            genealogy_p1 = ", ".join(map(str, sorted(player1.genealogy)))
            genealogy_p2 = ", ".join(map(str, sorted(player2.genealogy)))
            wrapped_genealogy_p1 = wrap_text(genealogy_p1, width=100)
            wrapped_genealogy_p2 = wrap_text(genealogy_p2, width=100)

            # Generate strategy strings
            strategy_p1 = self.get_strategy_string(player1.strategy)
            strategy_p2 = self.get_strategy_string(player2.strategy)

            # Use updated fitness values
            fitness_p1 = player1.fitness
            fitness_p2 = player2.fitness

            # Redefine agent texts with the final points
            agent1_text = f"Player 1, {player1.id}\nPoints: {points_p1}\n"
            "Games Played: {player1.game_counter}"
            agent2_text = f"Player 2, {player2.id}\nPoints: {points_p2}\n"
            "Games Played: {player2.game_counter}"

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
                label=f"Player 1, {player1.__class__.__name__}: {player1.id}",
            )
            ax.scatter(
                p2_pos[0],
                p2_pos[1],
                c="red",
                s=200,
                label=f"Player 2, {player2.__class__.__name__}: {player2.id}",
            )

            # Display agent information with wrapped genealogy
            # Agent 1 Information
            ax.text(
                -10,
                9.0,
                agent1_text,
                fontsize=12,
                color="blue",
                ha="left",
                va="top",
                clip_on=False,
            )
            ax.text(
                -10,
                7.0,
                f"Fitness: {initial_fitness_p1} => {fitness_p1}",
                fontsize=12,
                color="blue",
                ha="left",
                va="top",
                clip_on=False,
            )
            ax.text(
                -10,
                6.5,
                f"Genealogy:\n{wrapped_genealogy_p1}\n\n"
                f"Strategy: {strategy_p1}\n\n"
                f"Bid Params:\n{bid_info_p1}",
                fontsize=5,
                color="blue",
                ha="left",
                va="top",
                clip_on=False,
            )

            # Agent 2 Information
            ax.text(
                self.board_size + 9,
                self.board_size + 1.0,
                agent2_text,
                fontsize=12,
                color="red",
                ha="right",
                va="top",
                clip_on=False,
            )
            ax.text(
                self.board_size + 9,
                self.board_size - 1.0,
                f"Fitness:  {initial_fitness_p2} => {fitness_p2}",
                fontsize=12,
                color="red",
                ha="right",
                va="top",
                clip_on=False,
            )
            ax.text(
                self.board_size + 9,
                self.board_size - 1.5,
                f"Genealogy:\n{wrapped_genealogy_p2}"
                f"\n\nStrategy: {strategy_p2}\n\n"
                f"Bid Params:\n{bid_info_p2}",
                fontsize=5,
                color="red",
                ha="right",
                va="top",
                clip_on=False,
            )

            ax.legend(loc="upper left")

            # **Define winner_id Properly**
            winner_id = (
                self.get_player_id(self.winner) if self.winner else "None"
            )

            # Set the title with the winner
            ax.set_title(f"Game {self.game_number} Over - Winner: {winner_id}")

            # Display the winning reason
            if hasattr(self, "winning_reason"):
                ax.text(
                    (self.board_size / 2) - 1,
                    self.board_size + 1,
                    f"Reason: {self.winning_reason}",
                    fontsize=11,
                    color="black",
                    ha="center",
                    va="bottom",
                    clip_on=False,
                )

            plt.draw()
            plt.pause(2)
            plt.close(fig)
            # Remove plt.close(fig) to keep the plot open
            # Optionally, add a condition to close or manage the plot


# 4b. Define the GameScheduler Class
#
class GameScheduler:
    def __init__(self):
        self.scheduled_games = (
            []
        )  # Each game is a tuple: (agent1, agent2, game_type)

    def schedule_game(self, agent1, agent2, game_type="regular"):
        # Ensure both agents are active before scheduling
        if agent1.fitness >= 0 and agent2.fitness >= 0:
            self.scheduled_games.append((agent1, agent2, game_type))
            logging.debug(
                f"Scheduled {game_type} game between {agent1.id} and "
                f"{agent2.id}."
            )
        else:
            logging.warning(
                f"Cannot schedule {game_type} game between {agent1.id} and "
                f"{agent2.id} as one or both agents are inactive."
            )

    def remove_agent_games(self, agent):
        original_count = len(self.scheduled_games)
        self.scheduled_games = [
            (a1, a2, gt)
            for (a1, a2, gt) in self.scheduled_games
            if a1 != agent and a2 != agent
        ]
        removed_count = original_count - len(self.scheduled_games)
        if removed_count > 0:
            print(
                f"Removed {removed_count} scheduled games involving Agent "
                f"{agent.id}."
            )
            logging.info(
                f"Removed {removed_count} scheduled games involving Agent "
                f"{agent.id}."
            )

    def get_next_game(self):
        if self.scheduled_games:
            return self.scheduled_games.pop(0)
        return None


# 5a. Define the Population Class
#
class Population:
    def __init__(self, size=2880, population_id=1, meta_population=None):
        self.size = size
        self.population_id = population_id
        self.meta_population = meta_population  # Reference to MetaPopulation
        self.agents = []
        self.all_agents = []
        self.elite_percentage = (
            0.002  # 0.2% elite agents for intrapopulation matches
        )

        for _ in range(size):
            agent = Agent(population_id=self.population_id)
            self.agents.append(agent)
            self.all_agents.append(agent)

        self.game_scheduler = (
            GameScheduler()
        )  # Initialize GameScheduler for the population

        # Initialize current and previous most fit agent
        self.current_most_fit_agent = self.get_unique_most_fit_agent()
        self.previous_most_fit_agent = None  # Historical value initialization

        # Initialize current and previous least fit agent
        self.current_least_fit_agent = self.get_unique_least_fit_agent()
        self.previous_least_fit_agent = None  # Historical value initialization

        if self.current_most_fit_agent:
            logging.info(
                f"Initial most fit agent: {self.current_most_fit_agent.id} "
                f"from Population {self.current_most_fit_agent.population_id} "
                f"with fitness {self.current_most_fit_agent.fitness}"
            )
        else:
            logging.info("No unique most fit agent at initialization.")

        # Mutation rate parameters
        self.base_mutation_rate = 0.05
        self.mutation_rate = self.base_mutation_rate
        self.fitness_history = []  # To track fitness over generations

    def enforce_population_size(self):
        """
        Ensures that the population size does not exceed the predefined limit.
        Removes the least fit agents if necessary.
        """
        while len(self.agents) > self.size:
            least_fit_agent = self.get_least_fit_agent()
            if least_fit_agent:
                self.remove_agent(least_fit_agent)
                logging.info(
                    f"Enforced Population Size: Removed Least Fit Agent "
                    f"{least_fit_agent.id} with "
                    f"fitness {least_fit_agent.fitness} "
                    f"from Population {self.population_id}."
                )
                print(
                    f"Enforced Population Size: Removed Least Fit Agent "
                    f"{least_fit_agent.id} with "
                    f"fitness {least_fit_agent.fitness} "
                    f"from Population {self.population_id}."
                )
                self.game_scheduler.remove_agent_games(least_fit_agent)
            else:
                # No agents to remove, break to prevent infinite loop
                break

    def add_agent(self, agent):
        """
        Adds an agent to the population and ensures the population size
        remains stable.
        If the population exceeds its size, removes the least fit agent.

        Args:
            agent (Agent): The agent to be added.
        """
        assert (
            agent not in self.agents
        ), f"Agent {agent.id} already exists in Population"
        f"{self.population_id}."

        self.agents.append(agent)
        self.all_agents.append(agent)
        logging.info(
            f"Added Agent {agent.id} to Population {self.population_id}."
        )
        print(
            f"Added Agent {agent.id} with fitness {agent.fitness} to "
            f"Population {self.population_id}."
        )

        # Check and maintain population size
        if len(self.agents) > self.size:
            # Remove the least fit agent
            least_fit_agent = self.get_least_fit_agent()
            if least_fit_agent:
                self.remove_agent(least_fit_agent)
                logging.info(
                    f"Maintained Population Size: Removed Least Fit Agent "
                    f"{least_fit_agent.id} with fitness "
                    f"{least_fit_agent.fitness} from Population "
                    f"{self.population_id}."
                )
                print(
                    f"Maintained Population Size: Removed Least Fit Agent "
                    f"{least_fit_agent.id} with fitness "
                    f"{least_fit_agent.fitness} from Population "
                    f"{self.population_id}."
                )
                self.game_scheduler.remove_agent_games(least_fit_agent)

    def remove_agent(self, agent):
        if agent in self.agents:
            self.agents.remove(agent)
            print(
                f"Removed Agent {agent.id} from Population "
                f"{self.population_id} with fitness {agent.fitness}"
            )
            logging.info(
                f"Removed Agent {agent.id} from Population "
                f"{self.population_id} with fitness {agent.fitness}"
            )
            self.game_scheduler.remove_agent_games(agent)

            # Decrement offspring_count for parents
            for parent_id in agent.parents:
                parent_agent = next(
                    (a for a in self.agents if a.id == parent_id), None
                )
                if parent_agent:
                    parent_agent.offspring_count = max(
                        parent_agent.offspring_count - 1, 0
                    )
                    logging.info(
                        "Decremented offspring_count for "
                        f"Parent Agent {parent_agent.id} "
                        f"to {parent_agent.offspring_count}"
                    )
                    print(
                        "Decremented offspring_count for "
                        f"Parent Agent {parent_agent.id} "
                        f"to {parent_agent.offspring_count}"
                    )
        else:
            print(
                f"Attempted to remove Agent {agent.id} from Population "
                f"{self.population_id}, but agent was not found."
            )
            logging.warning(
                f"Attempted to remove Agent {agent.id} from Population "
                f"{self.population_id}, but agent was not found."
            )

    def calculate_average_fitness(self):
        if not self.agents:
            return 0
        total_fitness = sum(agent.fitness for agent in self.agents)
        return total_fitness / len(self.agents)

    def adjust_elite_percentage(self):
        """
        Adjusts the elite percentage based on recent fitness trends.
        """
        if len(self.fitness_history) < 2:
            return  # Not enough data to adjust

        previous_avg = self.fitness_history[-2]
        current_avg = self.fitness_history[-1]

        if current_avg > previous_avg * 1.05:
            # Rapid improvement, reduce elite percentage to maintain diversity
            self.elite_percentage = max(self.elite_percentage * 0.95, 0.2)
            logging.info(
                f"Population {self.population_id}: Decreasing elite_percentage"
                f" to {self.elite_percentage:.2f}"
            )
        elif current_avg < previous_avg * 0.95:
            # Decline or stagnation, increase elite percentage
            # to focus on top performers
            self.elite_percentage = min(self.elite_percentage * 1.05, 0.5)
            logging.info(
                f"Population {self.population_id}: Increasing elite_percentage"
                f" to {self.elite_percentage:.2f}"
            )

    def adjust_mutation_rate(self):
        """
        Adjusts the mutation rate based on the change in average fitness.
        """
        if len(self.fitness_history) < 2:
            return  # Not enough data to adjust

        previous_avg = self.fitness_history[-2]
        current_avg = self.fitness_history[-1]

        if current_avg <= previous_avg:
            # Fitness stagnating or decreasing, increase mutation rate
            self.mutation_rate = min(
                self.mutation_rate * 1.05, 0.2
            )  # Cap mutation rate
            logging.info(
                f"Population {self.population_id}: Increasing mutation rate to"
                f" {self.mutation_rate:.4f}"
            )
        else:
            # Fitness improving, decrease mutation rate
            self.mutation_rate = max(
                self.mutation_rate * 0.95, 0.005
            )  # Minimum mutation rate
            logging.info(
                f"Population {self.population_id}: Decreasing mutation rate to"
                f"{self.mutation_rate:.4f}"
            )

    def get_most_fit_agent(self):
        if not self.agents:
            logging.warning(f"Population {self.population_id} has no agents.")
            return None
        max_fitness = max(agent.fitness for agent in self.agents)
        top_agents = [
            agent for agent in self.agents if agent.fitness == max_fitness
        ]
        if top_agents:
            ", ".join([agent.id for agent in top_agents])
            # logging.info(f"Population {self.population_id} -
            # Max Fitness: {max_fitness} held by Agents: {agent_ids}")
            return random.choice(top_agents)
        else:
            logging.warning(
                f"Population {self.population_id} - No top agents found."
            )
            return None

    def get_unique_most_fit_agent(self):
        if not self.agents:
            return None  # Handle empty population gracefully

        # Determine the maximum fitness value
        max_fitness = max(agent.fitness for agent in self.agents)

        # Find all agents with the maximum fitness
        top_agents = [
            agent for agent in self.agents if agent.fitness == max_fitness
        ]

        if len(top_agents) == 1:
            # logging.info(f"Population {self.population_id} -
            # Max Fitness: {max_fitness} held by Agent: {top_agents}[1]")
            return top_agents[0]
        else:
            return None  # No unique most fit agent

    def get_least_fit_agent(self):
        """
        Retrieves the agent with the lowest fitness in the population.

        Returns:
            Agent: The least fit agent, or None if the population is empty.
        """
        if not self.agents:
            logging.warning(f"Population {self.population_id} has no agents.")
            return None
        min_fitness = min(agent.fitness for agent in self.agents)
        bottom_agents = [
            agent for agent in self.agents if agent.fitness == min_fitness
        ]
        if bottom_agents:
            return random.choice(bottom_agents)
        else:
            logging.warning(
                f"Population {self.population_id} - No bottom agents found."
            )
            return None

    def get_unique_least_fit_agent(self):
        """
        Retrieves the unique agent with the lowest fitness in the population.

        Returns:
            Agent: The unique least fit agent,
            or None if no unique agent exists.
        """
        if not self.agents:
            return None  # Population is empty

        min_fitness = min(agent.fitness for agent in self.agents)
        bottom_agents = [
            agent for agent in self.agents if agent.fitness == min_fitness
        ]

        if len(bottom_agents) == 1:
            return bottom_agents[0]
        else:
            return None  # No unique least fit agent

    def remove_and_replace_agent(
        self,
        agent_to_remove,
        winner=None,
        meta_population=None,
        other_population=None,
    ):
        if meta_population is None:
            meta_population = self.meta_population

        # Remove the agent using the centralized method
        self.remove_agent(agent_to_remove)

        # Remove any scheduled games involving this agent
        self.game_scheduler.remove_agent_games(agent_to_remove)
        if other_population:
            other_population.game_scheduler.remove_agent_games(agent_to_remove)
        logging.info(
            f"Removed scheduled games involving Agent {agent_to_remove.id}"
        )

        # Check for unique most fit agent and reproduce accordingly
        unique_most_fit_agent = self.get_unique_most_fit_agent()

        if winner:
            # Replacement derived from winner's population
            winner_population = meta_population.populations[
                winner.population_id - 1
            ]

        if unique_most_fit_agent:
            # Asexual reproduction
            new_offspring = unique_most_fit_agent.asexual_offspring()
            new_offspring.population_id = self.population_id
            # Ensure correct population_id
            self.add_agent(new_offspring)
            logging.info(
                "Asexual Reproduction: Added Offspring"
                f"Agent {new_offspring.id} "
                f"to Population {self.population_id}"
                f"derived from {unique_most_fit_agent.id}"
            )
            # Optionally adjust fitness if needed
            unique_most_fit_agent.fitness = (
                unique_most_fit_agent.fitness * 99 // 100
            )
            return new_offspring
        else:
            # Proceed with existing reproduction logic
            if other_population:
                # Sexual reproduction between populations
                fittest_current = self.get_most_fit_agent()
                fittest_other = other_population.get_most_fit_agent()
                if fittest_current and fittest_other:
                    child = fittest_current.sexual_offspring(fittest_other)
                    self.add_agent(child)
                    logging.info(
                        "Interpopulation Reproduction: "
                        "Added Offspring Agent "
                        f"{child.id} to Population "
                        f"{self.population_id} "
                        "via Sexual Reproduction "
                        f"between {fittest_current.id} "
                        f"and {fittest_other.id}."
                    )
                    print(
                        "Interpopulation Reproduction: "
                        "Added Offspring Agent "
                        f"{child.id} to Population "
                        f"{self.population_id} "
                        "via Sexual Reproduction "
                        f"between {fittest_current.id} "
                        f"and {fittest_other.id}."
                    )
                else:
                    # Fallback to asexual reproduction
                    reproduction_agent = winner_population.get_most_fit_agent()
                    if reproduction_agent:
                        new_offspring = reproduction_agent.asexual_offspring()
                        self.add_agent(new_offspring)
                        logging.info(
                            "Fallback Asexual Reproduction: "
                            "Added Offspring "
                            f"Agent {new_offspring.id} of "
                            f"{reproduction_agent.id} "
                            f"with fitness {reproduction_agent.fitness} "
                            f"to Population {self.population_id}."
                        )
                        print(
                            f"Added Offspring Agent {new_offspring.id} of "
                            f"{reproduction_agent.id} with fitness "
                            f"{reproduction_agent.fitness} to Population "
                            f"{self.population_id} via Asexual Reproduction."
                        )
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
                            f"Fallback Asexual Reproduction: Added Offspring "
                            f"Agent {new_offspring.id} of "
                            f"{reproduction_agent.id} "
                            f"with fitness {reproduction_agent.fitness} "
                            f"to Population {self.population_id}."
                        )
                        print(
                            f"Added Offspring Agent {new_offspring.id} of "
                            f"{reproduction_agent.id} with fitness "
                            f"{reproduction_agent.fitness} to Population "
                            f"{self.population_id} via Asexual Reproduction."
                        )
                else:
                    # Sexual reproduction within population
                    parent1, parent2 = fittest_agents
                    child = parent1.sexual_offspring(parent2)
                    self.add_agent(child)
                    logging.info(
                        f"Added Offspring Agent {child.id} to Population "
                        f"{self.population_id} via Sexual Reproduction "
                        "between "
                        f"{parent1.id} and {parent2.id}."
                    )
                    print(
                        f"Added Offspring Agent {child.id} to Population "
                        f"{self.population_id} via Sexual Reproduction "
                        "between "
                        f"{parent1.id} and {parent2.id}."
                    )

                    # Adjust fitness of reproduction agents
                    parent1.fitness = parent1.fitness * 99 // 100
                    parent2.fitness = parent2.fitness * 99 // 100

        # Maintain population integrity
        assert (
            len(self.agents) == self.size
        ), f"Population {self.population_id} size mismatch: "
        f"Expected {self.size}, Found {len(self.agents)}"
        unique_ids = set(agent.id for agent in self.agents)
        assert len(unique_ids) == len(
            self.agents
        ), f"Duplicate agents detected in Population {self.population_id}!"

        # **Report Population Status**
        self.report_population_status()

        return child

    def conduct_intrapopulation_elite_matches(self):
        """
        Conducts intrapopulation elite matches
        where the top 0.2% agents by fitness
        play against all other agents in the same population.
        """
        elite_percentage = 0.002  # 0.2%
        num_elite = max(
            2, int(self.size * elite_percentage)
        )  # Ensure at least 2 elite agents
        elite_agents = sorted(
            self.agents, key=lambda a: a.fitness, reverse=True
        )[:num_elite]
        non_elite_agents = [
            agent for agent in self.agents if agent not in elite_agents
        ]

        print(
            f"\n--- Intrapopulation Elite Matches for "
            f"Population {self.population_id} ---"
        )
        logging.info(
            f"Population {self.population_id} - Conducting Intrapopulation "
            f"Elite Matches with {num_elite} elite agents."
        )

        # Schedule elite matches via GameScheduler
        for elite_agent in elite_agents:
            for opponent in non_elite_agents:
                if elite_agent == opponent:
                    continue  # Skip self-match
                self.game_scheduler.schedule_game(
                    elite_agent, opponent, game_type="intrapopulation_elite"
                )

    def evaluate_fitness(self, meta_population=None):
        print("\n--- Evaluating Fitness for Current Generation ---")

        # Reset per-generation game counters
        for agent in self.agents:
            agent.reset_generation_counter()

        # Determine required number of games
        required_games_per_agent = 24  # Adjust as needed
        total_games_needed = (
            self.size * required_games_per_agent
        ) // 2  # Each game involves two agents

        # Capture the original agents at the start of the generation
        original_agents = list(self.agents)

        # Shuffle original agents to ensure random pairing
        random.shuffle(original_agents)

        # Schedule regular games using GameScheduler
        # until the number of total games needed has been reached
        while len(self.game_scheduler.scheduled_games) < total_games_needed:
            agent1, agent2 = random.sample(original_agents, 2)
            self.game_scheduler.schedule_game(
                agent1, agent2, game_type="regular"
            )
            logging.debug(
                f"Scheduled Game between {agent1.id} "
                f"(Pop {agent1.population_id}) and "
                f"{agent2.id} (Pop {agent2.population_id})"
            )

        print(
            "Total games scheduled for this generation: "
            f"{len(self.game_scheduler.scheduled_games)}"
        )

        logging.info(
            "Total games scheduled for this generation: "
            f"{len(self.game_scheduler.scheduled_games)}"
        )

        # Execute the scheduled games
        game_number = 0  # Initialize game counter

        while self.game_scheduler.scheduled_games:
            game_data = self.game_scheduler.get_next_game()
            if not game_data:
                break
            agent1, agent2, game_type = game_data
            game_number += 1

            # Check if both agents are still in the population
            if agent1 not in self.agents or agent2 not in self.agents:
                logging.warning(
                    f"Skipping Game {game_number}: "
                    f"One or both agents ({agent1.id}, "
                    f"{agent2.id}) have been removed."
                )
                continue  # Skip this game

            # Determine if this game should be visualized (optional)
            visualize_game = False
            sample_probability = 0.00005  # 0.005%
            if random.random() < sample_probability:
                visualize_game = True
                logging.info(
                    f"Visualizing sampled Game {game_number} "
                    f"between {agent1.id} and {agent2.id}"
                )

            # Store initial fitness before the game
            initial_fitness_agent1 = agent1.fitness
            initial_fitness_agent2 = agent2.fitness

            # Play the game
            game = Game(agent1, agent2, visualize=visualize_game)
            winner = game.play()

            # Increment game counters
            agent1.games_played_this_generation += 1
            agent2.games_played_this_generation += 1

            # Handle visualization
            if visualize_game:
                try:
                    game.visualize_game(
                        agent1,
                        agent2,
                        initial_fitness_agent1,
                        initial_fitness_agent2,
                    )
                    logging.info(
                        f"Visualized Game {game_number} between "
                        f"{agent1.id} and {agent2.id}"
                    )
                except Exception as e:
                    logging.error(
                        "An exception occurred during visualization"
                        f"of Game {game_number} between {agent1.id} "
                        f"and {agent2.id}: {e}",
                        exc_info=True,
                    )

            # Identify least fit agent
            least_fit_agent = self.get_least_fit_agent()
            logging.info(
                f"Least Fit Agent after Game {game_number}: "
                f"{least_fit_agent.id}, from Population "
                f"{least_fit_agent.population_id}, "
                f"Fitness: {least_fit_agent.fitness}"
            )

            # Identify most fit agent
            most_fit_agent = self.get_most_fit_agent()
            logging.info(
                f"Most Fit Agent after Game {game_number}: "
                f"{most_fit_agent.id}, from Population "
                f"{most_fit_agent.population_id}, "
                f"Fitness: {most_fit_agent.fitness}"
            )

            # Handle Agent Removal and Replacement
            if least_fit_agent.fitness < 0:
                agent_population = meta_population.get_population_of_agent(
                    least_fit_agent
                )
                if agent_population:
                    # Determine if the game was interpopulation
                    is_inter_population = (
                        agent1.population_id != agent2.population_id
                    )
                    if is_inter_population and winner:
                        agent_population.remove_and_replace_agent(
                            agent_to_remove=least_fit_agent,
                            winner=winner,
                            meta_population=meta_population,
                            other_population=(
                                meta_population.get_population_of_agent(winner)
                            ),
                        )
                    else:
                        # Intrapopulation replacement
                        agent_population.remove_and_replace_agent(
                            agent_to_remove=least_fit_agent
                        )
                    logging.info(
                        f"Agent {least_fit_agent.id} from "
                        f"Population {least_fit_agent.population_id}"
                        "removed and replaced."
                    )
                else:
                    logging.warning(
                        f"Agent {least_fit_agent.id} "
                        "not found in any population."
                    )

        # Handle Most Fit Agent Change
        most_fit_agent_unique = self.get_unique_most_fit_agent()
        if most_fit_agent_unique:
            if most_fit_agent_unique != self.current_most_fit_agent:
                previous_agent = self.current_most_fit_agent
                if previous_agent:
                    logging.info(
                        "Most fit agent in Population "
                        f"{most_fit_agent_unique.population_id} changed"
                        f" from {previous_agent.id} to "
                        f"{most_fit_agent_unique.id}, with "
                        f"fitness {most_fit_agent_unique.fitness}."
                    )
                else:
                    logging.info(
                        "Most fit agent in Population "
                        f"{most_fit_agent_unique.population_id} is now "
                        f"{most_fit_agent_unique.id}, with "
                        f"fitness {most_fit_agent_unique.fitness}."
                    )

                # Update the current most fit agent reference
                self.current_most_fit_agent = most_fit_agent_unique
        else:
            reproduction_agent = self.get_most_fit_agent()
            logging.info(
                "No unique most fit agent found. One most fit Agent is"
                f"{reproduction_agent.id} in Population "
                f"{reproduction_agent.population_id}, with "
                f"fitness {reproduction_agent.fitness}."
            )

        self.report_population_status()

        # Schedule intrapopulation elite matches
        self.conduct_intrapopulation_elite_matches()

        print("\n--- Fitness Evaluation Complete ---\n")

    def apply_fitness_sharing(self):
        """
        Applies fitness sharing to reduce the fitness of similar agents.
        """
        sigma_share = 10  # Sharing distance threshold
        alpha = 1  # Sharing exponent

        # Convert decision trees to strings for similarity comparison
        [str(agent.strategy.decision_tree) for agent in self.agents]

        # Initialize fitness sharing reductions
        fitness_sharing = [0 for _ in self.agents]

        for i, agent_i in enumerate(self.agents):
            for j, agent_j in enumerate(self.agents):
                if i == j:
                    print(".")
                    continue
                distance = self.calculate_strategy_distance(
                    agent_i.strategy, agent_j.strategy
                )
                if distance < sigma_share:
                    fitness_sharing[i] += 1 - 1 * (
                        (distance / sigma_share) ** alpha
                    )
                    print(
                        f"Low distance {distance} detected between "
                        f"{agent_i.id} and {agent_j} in Population "
                        f"{self.population_id}."
                    )

        # Apply the sharing reductions to fitness
        for i, agent in enumerate(self.agents):
            agent.fitness -= (
                fitness_sharing[i] * 2
            )  # Scaling factor for sharing impact
            # Ensure fitness doesn't drop below a minimum threshold
            agent.fitness = max(agent.fitness, -1000)

    def calculate_strategy_distance(self, strategy1, strategy2):
        """
        Calculates a simple distance metric between two strategies
        based on their decision trees.
        A more sophisticated method (like tree edit distance)
        can be implemented for better accuracy.
        """
        tree1 = strategy1.decision_tree
        tree2 = strategy2.decision_tree

        # Flatten the trees into lists of conditions and actions
        def flatten_tree(node):
            if isinstance(node, dict):
                return (
                    [node["condition"]]
                    + flatten_tree(node["true"])
                    + flatten_tree(node["false"])
                )
            else:
                return [node]

        flat1 = flatten_tree(tree1)
        flat2 = flatten_tree(tree2)

        # Compute the number of differing elements
        differences = sum(1 for a, b in zip(flat1, flat2) if a != b)
        distance = differences
        return distance

    def calculate_diversity(self):
        """
        Calculates diversity based on unique decision trees.
        """
        unique_strategies = set()
        for agent in self.agents:
            strategy_repr = str(agent.strategy.decision_tree)
            unique_strategies.add(strategy_repr)
        diversity = (
            len(unique_strategies) / len(self.agents) if self.agents else 0
        )
        return diversity

    def get_top_two_agents(self):
        """
        Retrieves the top two agents based on fitness.
        Returns:
            tuple: (Agent1, Agent2) if at least two agents exist, else None.
        """
        if len(self.agents) < 2:
            return None
        sorted_agents = sorted(
            self.agents, key=lambda a: a.fitness, reverse=True
        )
        return (sorted_agents[0], sorted_agents[1])

    def get_seminal_agents(self):
        """
        Identifies seminal agents based on the following criteria:
        1. At least 5% of agents currently in the population have
        the seminal agent in their genealogy set.

        Returns:
            List of tuples: [(agent_id, offspring_count), ...]
        """
        seminal_agents = []
        threshold = int(self.size * 0.05)  # 5% of the population

        for agent in self.all_agents:
            agent_id = agent.id
            # Count how many current agents have this agent in their genealogy
            count = sum(
                1
                for a in self.agents
                if agent_id in a.genealogy and a.id != agent_id
            )
            if count >= threshold:
                seminal_agents.append((agent_id, count))

        return seminal_agents

    def report_population_status(self, game_number=None):
        if not self.agents:
            print("No agents remaining in the population.")
            logging.info("No agents remaining in the population.")
            print("---=-=---")
            logging.info("---=-=---")
            return

        # 1. The Oldest Remaining Agent
        oldest_agent = min(self.agents, key=lambda a: int(a.id.split("-")[1]))
        oldest_agent_info = (
            f"Oldest Remaining Agent in Population {self.population_id}: "
            f"{oldest_agent.id} (Fitness: {oldest_agent.fitness})"
        )
        print(oldest_agent_info)
        logging.info(oldest_agent_info)

        # 2. The Most Experienced Agent
        most_experienced_agent = max(self.agents, key=lambda a: a.game_counter)
        most_experienced_info = (
            f"Most Experienced Agent in Population {self.population_id}: "
            f"{most_experienced_agent.id} "
            f"(Games Played: {most_experienced_agent.game_counter}) "
            f"(Fitness: {most_experienced_agent.fitness})"
        )
        print(most_experienced_info)
        logging.info(most_experienced_info)

        # 3. The Most Prolific Agent
        # Identify the living agent with the highest number
        # of living immediate offspring
        most_prolific_agent = max(
            self.agents, key=lambda a: a.offspring_count, default=None
        )
        if most_prolific_agent and most_prolific_agent.offspring_count > 0:
            most_prolific_info = (
                f"Most Prolific Agent in Population {self.population_id}: "
                f"{most_prolific_agent.id} "
                f"(Living Offspring: {most_prolific_agent.offspring_count}) "
                f"(Fitness: {most_prolific_agent.fitness})"
            )
            print(most_prolific_info)
            logging.info(most_prolific_info)
        else:
            print("No Prolific Agents found.")
            logging.info("No Prolific Agents found.")

        # 4. Seminal Agents
        seminal_agents = self.get_seminal_agents()
        if seminal_agents:
            print("Seminal Agents:")
            logging.info(f"Seminal Agents in Population {self.population_id}:")
            for agent_id, count in seminal_agents:
                agent_info = f" - {agent_id} (Living Offspring: {count})"
                print(agent_info)
                logging.info(agent_info)
        else:
            print("No Seminal Agents Found.")
            logging.info("No Seminal Agents Found.")

        # 5. Average Population Fitness
        avg_fitness = self.calculate_average_fitness()
        avg_fitness_info = (
            f"Average Fitness in Population {self.population_id}:"
            f" {avg_fitness:.2f}"
        )
        print(avg_fitness_info)
        logging.info(avg_fitness_info)

        # 6. Diversity
        diversity = self.calculate_diversity()
        diversity_info = (
            f"Diversity in Population {self.population_id}: {diversity:.4f}"
        )
        print(diversity_info)
        logging.info(diversity_info)

        # 7. Most Fit Agent Change Logging
        unique_most_fit_agent = self.get_unique_most_fit_agent()
        if unique_most_fit_agent:
            if unique_most_fit_agent != self.previous_most_fit_agent:
                change_info = (
                    f"Most fit agent in Population {self.population_id}"
                    "changed from "
                    f"{self.previous_most_fit_agent.id} "
                    f"to {unique_most_fit_agent.id}, "
                    f"with fitness {unique_most_fit_agent.fitness}."
                )
                print(change_info)
                logging.info(change_info)
                self.previous_most_fit_agent = unique_most_fit_agent
        else:
            reproduction_agent = self.get_most_fit_agent()
            if reproduction_agent:
                change_info = (
                    f"Most fit agent in Population {self.population_id} "
                    "is now "
                    f"{reproduction_agent.id}, with fitness "
                    f"{reproduction_agent.fitness}."
                )
                print(change_info)
                logging.info(change_info)
                self.previous_most_fit_agent = reproduction_agent
            else:
                print("No unique most fit agent currently.")
                logging.info("No unique most fit agent currently.")
                self.previous_most_fit_agent = None

        # 8. Least Fit Agent Change Logging
        unique_least_fit_agent = self.get_unique_least_fit_agent()
        if unique_least_fit_agent:
            if unique_least_fit_agent != self.previous_least_fit_agent:
                change_info = (
                    f"Least fit agent in Population {self.population_id} "
                    "changed from "
                    f"{self.previous_least_fit_agent.id} "
                    f"to {unique_least_fit_agent.id}, with fitness "
                    f"{unique_least_fit_agent.fitness}."
                )
                print(change_info)
                logging.info(change_info)
                self.previous_least_fit_agent = unique_least_fit_agent
        else:
            if self.previous_least_fit_agent is not None:
                change_info = "No unique least fit agent currently."
                print(change_info)
                logging.info(change_info)
                self.previous_least_fit_agent = None

        print("\n--- Population Status Report Complete ---")
        logging.info("\n--- Population Status Report Complete ---\n")

    def visualize_game_change(self, previous_agent, new_agent):
        """
        Visualize the transition of the most fit agent.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title("Most Fit Agent Change")

        # Plot previous most fit agent if it exists
        if previous_agent:
            ax.scatter(
                0.3,
                0.5,
                c="green",
                s=200,
                label=f"Previous Most Fit:\n{previous_agent.id}\nFitness: "
                f"{previous_agent.fitness}",
            )

        # Plot new most fit agent
        ax.scatter(
            0.7,
            0.5,
            c="gold",
            s=200,
            label=f"New Most Fit:\n{new_agent.id}\nFitness: "
            f"{new_agent.fitness}",
        )

        ax.legend(loc="upper left")
        ax.axis("off")  # Hide axes for clarity

        plt.show()
        plt.pause(0.6)  # Display for 0.6 seconds
        plt.close(fig)

    def select_parents(self):
        total_fitness = sum(agent.fitness for agent in self.agents)
        if total_fitness == 0:
            return random.choices(self.agents, k=len(self.agents))
        probabilities = [
            agent.fitness / total_fitness for agent in self.agents
        ]
        return random.choices(
            self.agents, weights=probabilities, k=len(self.agents)
        )

    def generate_next_generation(self):
        # Sort agents by fitness in descending order
        sorted_agents = sorted(
            self.agents, key=lambda agent: agent.fitness, reverse=True
        )

        # Determine the number of top agents to keep
        # based on dynamic elite_percentage
        num_elite = int(self.size * self.elite_percentage)
        if num_elite < 1:
            num_elite = 1  # Ensure at least one agent survives

        # Keep the top agents
        survivors = sorted_agents[:num_elite]

        # These agents retain their IDs
        next_generation = survivors.copy()

        # Calculate how many offspring need to be generated
        # to maintain population size
        num_offspring_needed = self.size - len(survivors)

        # Generate offspring
        offspring = []
        # Fitness values for survivors
        survivor_fitness = [agent.fitness for agent in survivors]
        total_fitness = sum(survivor_fitness)

        # If total_fitness is zero (edge case), assign equal probability
        if total_fitness == 0:
            survivor_probabilities = [1 / len(survivors)] * len(survivors)
        else:
            survivor_probabilities = [
                fitness / total_fitness for fitness in survivor_fitness
            ]

        while len(offspring) < num_offspring_needed:
            # Select parents based on fitness proportionate selection
            parent1 = random.choices(
                survivors, weights=survivor_probabilities, k=1
            )[0]
            parent2 = random.choices(
                survivors, weights=survivor_probabilities, k=1
            )[0]

            # Check if there is a unique most fit agent
            unique_most_fit_agent = self.get_unique_most_fit_agent()
            if unique_most_fit_agent:
                # Perform asexual reproduction
                new_offspring = unique_most_fit_agent.asexual_offspring()
                offspring.append(new_offspring)
            else:
                # Sexual reproduction
                child_strategy = parent1.strategy.crossover(parent2.strategy)
                child_strategy.mutate(mutation_rate=self.mutation_rate)
                # Combine genealogies
                child_genealogy = parent1.genealogy.union(parent2.genealogy)
                child = Agent(
                    strategy=child_strategy,
                    genealogy=child_genealogy,
                    population_id=self.population_id,
                )
                offspring.append(child)

                # Asexual reproduction from parent1
                if len(offspring) < num_offspring_needed:
                    child_strategy = deepcopy(parent1.strategy)
                    child_strategy.mutate(mutation_rate=self.mutation_rate)
                    # Inherit genealogy from parent1
                    child_genealogy = set(parent1.genealogy)
                    child = Agent(
                        strategy=child_strategy, genealogy=child_genealogy
                    )
                    offspring.append(child)

        # Combine survivors and offspring to form the next generation
        self.agents = next_generation + offspring


# 5b. Define the MetaPopulation Class
#
class MetaPopulation:
    def __init__(self, num_populations=12, population_size=5040):
        self.populations = [
            Population(
                size=population_size, population_id=i + 1, meta_population=self
            )
            for i in range(num_populations)
        ]
        self.num_populations = num_populations
        self.population_size = population_size
        self.total_population_size = num_populations * population_size

        # Initialize previous fit agents
        self.previous_most_fit_agent = self.get_unique_most_fit_agent()
        self.previous_least_fit_agent = self.get_unique_least_fit_agent()

        # Initialize a GameScheduler for interpopulation elite matches
        self.interpopulation_game_scheduler = GameScheduler()

        # Log initial state
        if self.previous_most_fit_agent:
            logging.info(
                "MetaPopulation - Initial Most Fit Agent: "
                f"{self.previous_most_fit_agent.id} from Population "
                f"{self.previous_most_fit_agent.population_id} with "
                f"fitness {self.previous_most_fit_agent.fitness}"
            )
        else:
            logging.info(
                "MetaPopulation - No unique most fit agent at initialization."
            )

        if self.previous_least_fit_agent:
            logging.info(
                f"MetaPopulation - Initial Least Fit Agent: "
                f"{self.previous_least_fit_agent.id} from Population "
                f"{self.previous_least_fit_agent.population_id} with "
                f"fitness {self.previous_least_fit_agent.fitness}"
            )
        else:
            logging.info(
                "MetaPopulation - No unique least fit agent at initialization."
            )

    def evolve(self, generations=100):
        for generation in range(generations):
            print(f"\n--- Meta Generation {generation + 1} ---")
            logging.info(
                f"MetaPopulation - Starting Meta Generation {generation + 1}."
            )

            # Evolve each population separately
            for population in self.populations:
                print(f"\nEvolving Population {population.population_id}")
                population.evaluate_fitness(meta_population=self)
                population.generate_next_generation()
                # Report population status after generation
                self.report_metapopulation_status()

            # Schedule Cross-population sexual reproduction via GameScheduler
            self.cross_population_reproduction()
            self.report_metapopulation_status()

            # Enforce population sizes
            self.enforce_population_size()
            self.report_metapopulation_status()

            # Schedule Elite Matches
            self.conduct_elite_matches()
            self.report_metapopulation_status()

            # Execute Regular and Intrapopulation Elite Games
            self.execute_scheduled_games()
            self.report_metapopulation_status()

            # Execute Interpopulation Elite Games
            self.execute_interpopulation_elite_games()
            self.report_metapopulation_status()

            # **Selective Logging for MetaPopulation
            # Most and Least Fit Agents**
            # Removed from here; now handled
            # within report_metapopulation_status()

        self.report_metapopulation_status()

    def execute_scheduled_games(self):
        """
        Executes all scheduled games in the GameScheduler.
        This includes regular games and intrapopulation elite matches.
        """
        print(
            "\n--- Executing Scheduled Regular "
            "and Intrapopulation Elite Games ---"
        )
        logging.info(
            "\n--- Executing Scheduled Regular "
            "and Intrapopulation Elite Games ---"
        )

        for population in self.populations:
            while True:
                game = population.game_scheduler.get_next_game()
                if not game:
                    break
                agent1, agent2, game_type = game

                # Check if both agents are still in the population
                if (
                    agent1 not in population.agents
                    or agent2 not in population.agents
                ):
                    print(
                        f"Skipping {game_type} Game: "
                        "One or both agents have been removed."
                    )
                    logging.warning(
                        f"Skipping {game_type} Game between "
                        f"{agent1.id} and {agent2.id}: "
                        "One or both agents have been removed."
                    )
                    continue

                # Determine if this game should be visualized
                visualize_game = False
                if game_type in ["intrapopulation_elite"]:
                    visualize_game = False
                    # Typically, elite matches
                    # are not visualized unless sampled

                # Store initial fitness before the game
                initial_fitness_agent1 = agent1.fitness
                initial_fitness_agent2 = agent2.fitness

                # Play the game
                game_instance = Game(agent1, agent2, visualize=visualize_game)
                winner = game_instance.play()

                # Update fitness based on game outcome
                fitness_change_agent1 = game_instance.get_fitness_change(
                    agent1
                )
                agent1.fitness += fitness_change_agent1

                fitness_change_agent2 = game_instance.get_fitness_change(
                    agent2
                )
                agent2.fitness += fitness_change_agent2

                # Increment game counters
                agent1.games_played_this_generation += 1
                agent2.games_played_this_generation += 1

                # Handle visualization if sampled
                if visualize_game:
                    try:
                        game_instance.visualize_game(
                            agent1,
                            agent2,
                            initial_fitness_agent1,
                            initial_fitness_agent2,
                        )
                    except Exception as e:
                        print(
                            f"An exception occurred during visualization: {e}"
                        )
                        logging.error(
                            f"Visualization error for Game between {agent1.id}"
                            f" and {agent2.id}: {e}"
                        )

                # Print game results
                print(
                    f"{game_type.capitalize()} Game: {agent1.id} "
                    f"(Pop {agent1.population_id}) vs {agent2.id} "
                    f"(Pop {agent2.population_id})"
                )
                print(f" - {agent1.id} fitness: {agent1.fitness}")
                print(f" - {agent2.id} fitness: {agent2.fitness}")

                # Handle agent removal and replacement
                for agent in [agent1, agent2]:
                    if agent.fitness < 0:
                        agent_population = self.get_population_of_agent(agent)
                        if agent_population:
                            # Replace the agent using the winner's population
                            # if interpopulation
                            if (
                                game_type == "interpopulation_elite"
                                and winner
                                and winner != agent
                            ):
                                # Assign the result of the function call
                                # to a variable
                                other_population = (
                                    self.get_population_of_agent(winner)
                                )

                                # Use the variable in the function call
                                agent_population.remove_and_replace_agent(
                                    agent_to_remove=agent,
                                    winner=winner,
                                    meta_population=self,
                                    other_population=other_population,
                                )
                            else:
                                # Intrapopulation replacement
                                agent_population.remove_and_replace_agent(
                                    agent_to_remove=agent
                                )
                        else:
                            print(
                                f"Agent {agent.id} not found in any population"
                            )
                            logging.warning(
                                f"Agent {agent.id} not found in any population"
                            )

                # After determining winner, update fitness
                # and handle remove and replacement
                print(
                    f"Regular Game {game_instance.game_number}: {agent1.id} "
                    f"(Pop {agent1.population_id}) vs {agent2.id} "
                    f"(Pop {agent2.population_id})"
                )
                print(f" - {agent1.id} fitness: {agent1.fitness}")
                print(f" - {agent2.id} fitness: {agent2.fitness}")

        print("--- Scheduled Games Execution Complete ---")
        logging.info("--- Scheduled Games Execution Complete ---")

    def execute_interpopulation_elite_games(self):
        """
        Executes all scheduled interpopulation elite games
        in the interpopulation_game_scheduler.
        """
        print("\n--- Executing Scheduled Interpopulation Elite Games ---")
        logging.info(
            "\n--- Executing Scheduled Interpopulation Elite Games ---"
        )

        while self.interpopulation_game_scheduler.scheduled_games:
            game = self.interpopulation_game_scheduler.get_next_game()
            agent1, agent2, game_type = game

            # Retrieve populations
            population1 = self.get_population_of_agent(agent1)
            population2 = self.get_population_of_agent(agent2)

            if not population1 or not population2:
                print(
                    f"Skipping {game_type} Game: One or both agents"
                    "have been removed."
                )
                logging.warning(
                    f"Skipping {game_type} Game between {agent1.id} and "
                    f"{agent2.id}: One or both agents have been removed."
                )
                continue

            # Determine if this game should be visualized
            visualize_game = False
            # Typically, interpopulation elite matches
            # are not visualized unless sampled

            # Store initial fitness before the game
            agent1.fitness
            agent2.fitness

            # Play the game
            game_instance = Game(agent1, agent2, visualize=visualize_game)
            winner = game_instance.play()

            # Update fitness based on game outcome
            fitness_change_agent1 = game_instance.get_fitness_change(agent1)
            agent1.fitness += fitness_change_agent1

            fitness_change_agent2 = game_instance.get_fitness_change(agent2)
            agent2.fitness += fitness_change_agent2

            # Increment game counters
            agent1.games_played_this_generation += 1
            agent2.games_played_this_generation += 1

            # Handle visualization if sampled (optional)
            # You can introduce sampling logic
            # similar to regular games if needed

            # Print game results
            print(
                f"{game_type.capitalize()} Game: {agent1.id} "
                f"(Pop {agent1.population_id}) vs "
                f"{agent2.id} (Pop {agent2.population_id})"
            )
            print(f" - {agent1.id} fitness: {agent1.fitness}")
            print(f" - {agent2.id} fitness: {agent2.fitness}")

            # Handle agent removal and replacement
            for agent in [agent1, agent2]:
                if agent.fitness < 0:
                    agent_population = self.get_population_of_agent(agent)
                    if agent_population:
                        # Replace the agent
                        # using the winner's population if interpopulation
                        if (
                            game_type == "interpopulation_elite"
                            and winner
                            and winner != agent
                        ):
                            agent_population.remove_and_replace_agent(
                                agent_to_remove=agent,
                                winner=winner,
                                meta_population=self,
                                other_population=self.get_population_of_agent(
                                    winner
                                ),
                            )
                        else:
                            # Intrapopulation replacement
                            agent_population.remove_and_replace_agent(
                                agent_to_remove=agent
                            )
                    else:
                        print(f"Agent {agent.id} not found in any population.")
                        logging.warning(
                            f"Agent {agent.id} not found in any population."
                        )

        print("--- Interpopulation Elite Games Execution Complete ---")
        logging.info("--- Interpopulation Elite Games Execution Complete ---")

    def report_metapopulation_status(self):
        """
        Reports the status of the metapopulation by
        aggregating the status of all populations.
        Includes population-specific statistics as
        well as overall metapopulation statistics.
        Also includes selective logging for
        MetaPopulation's most and least fit agents.
        """
        print("\n=== Metapopulation Status Report ===")
        logging.info("=== Metapopulation Status Report ===")

        # Initialize variables to aggregate statistics
        total_agents = 0
        total_fitness = 0
        total_diversity = 0
        all_agents = []

        # Iterate through each population and report their status
        for population in self.populations:
            print(f"\n--- Population {population.population_id} Status ---")
            logging.info(
                f"\n--- Population {population.population_id} Status ---"
            )
            population.report_population_status()

            # Aggregate statistics
            num_agents = len(population.agents)
            avg_fitness = population.calculate_average_fitness()
            diversity = population.calculate_diversity()

            total_agents += num_agents
            total_fitness += (
                avg_fitness * num_agents
            )  # Weighted sum for average
            total_diversity += diversity
            all_agents.extend(population.agents)

        # Compute overall metapopulation statistics
        if total_agents > 0:
            overall_average_fitness = total_fitness / total_agents
        else:
            overall_average_fitness = 0

        overall_diversity = (
            total_diversity / self.num_populations
            if self.num_populations > 0
            else 0
        )

        # Identify overall most fit and least fit agents
        if all_agents:
            overall_most_fit_agent = max(all_agents, key=lambda a: a.fitness)
            overall_least_fit_agent = min(all_agents, key=lambda a: a.fitness)
        else:
            overall_most_fit_agent = None
            overall_least_fit_agent = None

        # Print and log overall statistics
        print("\n--- Overall Metapopulation Statistics ---")
        logging.info("\n--- Overall Metapopulation Statistics ---")
        print(f"Total Number of Agents: {total_agents}")
        logging.info(f"Total Number of Agents: {total_agents}")
        print(f"Overall Average Fitness: {overall_average_fitness:.2f}")
        logging.info(f"Overall Average Fitness: {overall_average_fitness:.2f}")
        print(f"Overall Diversity: {overall_diversity:.4f}")
        logging.info(f"Overall Diversity: {overall_diversity:.4f}")

        if overall_most_fit_agent:
            print(
                f"Overall Most Fit Agent: {overall_most_fit_agent.id} from "
                f"Population {overall_most_fit_agent.population_id} with "
                f"Fitness {overall_most_fit_agent.fitness}"
            )
            logging.info(
                f"Overall Most Fit Agent: {overall_most_fit_agent.id} from "
                f"Population {overall_most_fit_agent.population_id} with "
                f"Fitness {overall_most_fit_agent.fitness}"
            )

            # **Selective Logging for MetaPopulation Most Fit Agent**
            if overall_most_fit_agent != self.previous_most_fit_agent:
                is_unique = (
                    len(
                        [
                            agent
                            for agent in all_agents
                            if agent.fitness == overall_most_fit_agent.fitness
                        ]
                    )
                    == 1
                )
                if is_unique:
                    logging.info(
                        "MetaPopulation - Most Fit Agent Changed: "
                        f"{overall_most_fit_agent.id} from Population "
                        f"{overall_most_fit_agent.population_id} with "
                        f"fitness {overall_most_fit_agent.fitness}"
                    )
                    print(
                        f"MetaPopulation - Most Fit Agent Changed: "
                        f"{overall_most_fit_agent.id} from Population "
                        f"{overall_most_fit_agent.population_id} with "
                        f"fitness {overall_most_fit_agent.fitness}"
                    )
                    print(
                        "\n--- Most Fit Agent Estimated "
                        "Fitness Change Per Game ---"
                    )
                    logging.info(
                        "--- Most Fit Agent Estimated "
                        "Fitness Change Per Game ---"
                    )
                    min_est, mean_est, max_est = (
                        overall_most_fit_agent.estimate_fitness_change()
                    )
                    agent_est_info = (
                        f"Agent {overall_most_fit_agent.id} "
                        f"(Population {overall_most_fit_agent.population_id}):"
                        f"\n - Estimated Min Change: {min_est:.2f}\n"
                        f" - Estimated Mean Change: {mean_est:.2f}\n"
                        f" - Estimated Max Change: {max_est:.2f}"
                    )
                    print(agent_est_info)
                    logging.info(agent_est_info)
                    self.previous_most_fit_agent = overall_most_fit_agent
        else:
            print("No agents found in the MetaPopulation.")
            logging.info("No agents found in the MetaPopulation.")

        # Log overall least fit agent
        if overall_least_fit_agent:
            print(
                f"Overall Least Fit Agent: {overall_least_fit_agent.id} from "
                f"Population {overall_least_fit_agent.population_id} with "
                f"Fitness {overall_least_fit_agent.fitness}"
            )
            logging.info(
                f"Overall Least Fit Agent: {overall_least_fit_agent.id} from "
                f"Population {overall_least_fit_agent.population_id} with "
                f"Fitness {overall_least_fit_agent.fitness}"
            )

            # **Selective Logging for MetaPopulation Least Fit Agent**
            if overall_least_fit_agent != self.previous_least_fit_agent:
                is_unique = (
                    len(
                        [
                            agent
                            for agent in all_agents
                            if agent.fitness == overall_least_fit_agent.fitness
                        ]
                    )
                    == 1
                )
                if is_unique:
                    logging.info(
                        "MetaPopulation - Least Fit Agent Changed: "
                        f"{overall_least_fit_agent.id} from "
                        f"Population {overall_least_fit_agent.population_id} "
                        f"with fitness {overall_least_fit_agent.fitness}"
                    )
                    print(
                        "MetaPopulation - Least Fit Agent Changed: "
                        f"{overall_least_fit_agent.id} from "
                        f"Population {overall_least_fit_agent.population_id} "
                        f"with fitness {overall_least_fit_agent.fitness}"
                    )
                    self.previous_least_fit_agent = overall_least_fit_agent
        else:
            print("No agents found in the MetaPopulation.")
            logging.info("No agents found in the MetaPopulation.")

        # Print and log overall statistics
        print("\n--- Overall Metapopulation Statistics Complete ---")
        logging.info("\n--- Overall Metapopulation Statistics Complete ---\n")

    def cross_population_reproduction(self):
        """
        Facilitates sexual reproduction between elite agents
        from different populations.
        """
        try:
            print("\n--- Conducting Cross-Population Reproduction ---")
            logging.info("\n--- Conducting Cross-Population Reproduction ---")
            elite_percentage = 0.01
            # Define elite percentage for cross-population reproduction

            # Collect elite agents from each population
            elite_agents = []
            for population in self.populations:
                sorted_agents = sorted(
                    population.agents, key=lambda a: a.fitness, reverse=True
                )
                num_elite = max(2, int(population.size * elite_percentage))
                elite_agents.extend(sorted_agents[:num_elite])

            # Shuffle elite agents to randomize pairing
            random.shuffle(elite_agents)

            # Pair elite agents from different populations
            for i in range(0, len(elite_agents), 2):
                if i + 1 >= len(elite_agents):
                    break  # No pair available
                parent1 = elite_agents[i]
                parent2 = elite_agents[i + 1]

                # Ensure parents are from different populations
                if parent1.population_id == parent2.population_id:
                    # Find a parent from a different population
                    for j in range(i + 2, len(elite_agents)):
                        if (
                            elite_agents[j].population_id
                            != parent1.population_id
                        ):
                            parent2 = elite_agents[j]
                            elite_agents[j], elite_agents[i + 1] = (
                                elite_agents[i + 1],
                                elite_agents[j],
                            )
                            break
                    else:
                        # No parent from a different population found;
                        # skip reproduction
                        print(
                            f"Skipping reproduction between {parent1.id} "
                            f"and {parent2.id} due to same population."
                        )
                        logging.warning(
                            f"Skipping reproduction between {parent1.id} "
                            f"and {parent2.id} due to same population."
                        )
                        continue

                # Retrieve the Population objects using population_id
                parent_population = self.populations[
                    parent1.population_id - 1
                ]  # Assuming population_id starts at 1
                # Optionally, verify that parent2
                # is from a different population
                self.populations[parent2.population_id - 1]

                # Perform crossover to produce offspring
                child_strategy = parent1.strategy.crossover(parent2.strategy)
                child_strategy.mutate(
                    mutation_rate=parent_population.mutation_rate
                )  # Use parent's mutation rate

                # Combine genealogies
                child_genealogy = parent1.genealogy.union(parent2.genealogy)
                child = Agent(
                    strategy=child_strategy,
                    genealogy=child_genealogy,
                    population_id=parent1.population_id,
                )

                # Use remove_and_replace_agent to
                # ensure population size remains stable
                parent_population.remove_and_replace_agent(
                    agent_to_remove=parent_population.get_least_fit_agent(),
                    winner=None,
                )  # Remove least fit agent
                parent_population.game_scheduler.remove_agent_games(
                    agent_to_remove=parent_population.get_least_fit_agent()
                )

                # Optionally, remove the least fit agent
                # to maintain population size
                least_fit_agent = parent_population.get_least_fit_agent()
                if least_fit_agent:
                    parent_population.remove_agent(least_fit_agent)
                    logging.info(
                        "Cross-Population Reproduction: "
                        f"Removed Least Fit Agent {least_fit_agent.id} from "
                        f"Population {parent_population.population_id} with "
                        f"fitness {least_fit_agent.fitness}."
                    )
                    print(
                        "Cross-Population Reproduction: "
                        f"Removed Least Fit Agent {least_fit_agent.id} from "
                        f"Population {parent_population.population_id} with "
                        f"fitness {least_fit_agent.fitness}."
                    )
                    parent_population.game_scheduler.remove_agent_games(
                        agent_to_remove=parent_population.get_least_fit_agent()
                    )

                # Add offspring to the first parent's population
                parent_population.add_agent(child)  # Add the new offspring
                logging.info(
                    "Cross-Population Reproduction: Added Offspring Agent "
                    f"{child.id} to Population "
                    f"{parent_population.population_id}"
                    f"via Sexual Reproduction between {parent1.id} with "
                    f"fitness {parent1.fitness} and {parent2.id} with "
                    f"fitness {parent2.fitness}."
                )
                print(
                    "Cross-Population Reproduction: Added Offspring Agent "
                    f"{child.id} to Population "
                    f"{parent_population.population_id}"
                    f"via Sexual Reproduction between {parent1.id} with "
                    f"fitness {parent1.fitness} and {parent2.id} with "
                    f"fitness {parent2.fitness}."
                )

            self.report_metapopulation_status()
            print("--- Cross-Population Reproduction Complete ---")

        except IndexError as ie:
            print(f"IndexError during cross-population reproduction: {ie}")
            logging.error(
                f"IndexError during cross-population reproduction: {ie}"
            )
        except Exception as e:
            print(
                "An unexpected error occurred during "
                f"cross-population reproduction: {e}"
            )
            logging.error(
                f"Unexpected error during cross-population reproduction: {e}"
            )

    def get_least_fit_agent(self):
        """
        Retrieves the agent with the lowest fitness across all populations
        in the meta-population.

        Returns:
        Agent: The least fit agent across all populations,
        or None if no agents exist.
        """
        least_fit_agent = None
        min_fitness = float("inf")
        for population in self.populations:
            agent = population.get_least_fit_agent()
            if agent and agent.fitness < min_fitness:
                min_fitness = agent.fitness
                least_fit_agent = agent
        if least_fit_agent:
            return least_fit_agent
        else:
            return None

    def get_unique_least_fit_agent(self):
        """
        Retrieves the unique least fit agent across all populations.

        Returns:
            Agent: The unique least fit agent,
            or None if no unique agent exists.
        """
        least_fit_agents = []
        min_fitness = float("inf")
        for population in self.populations:
            agent = population.get_least_fit_agent()
            if agent:
                if agent.fitness < min_fitness:
                    least_fit_agents = [agent]
                    min_fitness = agent.fitness
                elif agent.fitness == min_fitness:
                    least_fit_agents.append(agent)
        if len(least_fit_agents) == 1:
            return least_fit_agents[0]
        else:
            return None  # No unique least fit agent

    def get_most_fit_agent(self):
        """
        Retrieves the agent with the highest fitness
        across all populations in the meta-population.

        Returns:
        Agent: The most fit agent across all populations,
        or None if no agents exist.
        """
        most_fit_agent = None
        max_fitness = float("-inf")
        for population in self.populations:
            agent = population.get_most_fit_agent()
            if agent and agent.fitness > max_fitness:
                max_fitness = agent.fitness
                most_fit_agent = agent
        if most_fit_agent:
            return most_fit_agent
        else:
            return None

    def get_unique_most_fit_agent(self):
        """
        Retrieves the unique most fit agent across all populations.

        Returns:
            Agent: The unique most fit agent,
            or None if no unique agent exists.
        """
        most_fit_agents = []
        max_fitness = float("-inf")
        for population in self.populations:
            agent = population.get_most_fit_agent()
            if agent:
                if agent.fitness > max_fitness:
                    most_fit_agents = [agent]
                    max_fitness = agent.fitness
                elif agent.fitness == max_fitness:
                    most_fit_agents.append(agent)
        if len(most_fit_agents) == 1:
            return most_fit_agents[0]
        else:
            return None  # No unique most fit agent

    def enforce_population_size(self):
        """
        Ensures that each population within the MetaPopulation
        maintains its defined size.
        Removes the least fit agents from each population if necessary.
        """
        for population in self.populations:
            population.enforce_population_size()

    def conduct_elite_matches(self):
        """
        Conducts interpopulation elite matches where the top 0.1% agents
        by fitness
        across the metapopulation play against all other agents
        in the metapopulation.
        """
        print("\n--- Conducting Elite Matches Between Populations ---\n")
        logging.info(
            "\n--- Conducting Elite Matches Between Populations ---\n"
        )
        elite_percentage = 0.001  # 0.1%
        total_agents = self.num_populations * self.population_size
        num_elite = max(
            2, int(total_agents * elite_percentage)
        )  # Ensure at least 2 elite agents

        # Gather all agents across the metapopulation
        all_agents = [
            agent
            for population in self.populations
            for agent in population.agents
        ]
        elite_agents = sorted(
            all_agents, key=lambda a: a.fitness, reverse=True
        )[:num_elite]
        non_elite_agents = [
            agent for agent in all_agents if agent not in elite_agents
        ]

        print("\n--- Interpopulation Elite Matches for MetaPopulation ---")
        logging.info(
            "MetaPopulation - Conducting Interpopulation Elite Matches with "
            f"{num_elite} elite agents."
        )

        # Schedule interpopulation elite matches
        # via interpopulation_game_scheduler
        for elite_agent in elite_agents:
            for opponent in non_elite_agents:
                if elite_agent.population_id == opponent.population_id:
                    continue  # Skip intrapopulation matches here
            self.interpopulation_game_scheduler.schedule_game(
                elite_agent, opponent, game_type="interpopulation_elite"
            )

    def get_population_of_agent(self, agent):
        """
        Retrieves the Population instance that the agent belongs to.

        Args:
            agent (Agent): The agent whose population is to be found.

        Returns:
            Population: The population containing the agent,
            or None if not found.
        """
        for population in self.populations:
            if agent in population.agents:
                return population
        return None


# 6. Utility Functions
#
def print_strategy(tree, indent=0):
    if isinstance(tree, dict):
        print("  " * indent + f"Condition: {tree['condition']}")
        print("  " * indent + "True:")
        print_strategy(tree["true"], indent + 1)
        print("  " * indent + "False:")
        print_strategy(tree["false"], indent + 1)
    else:
        print("  " * indent + f"Action: {tree}")


# 7. Main Execution Logic
#
def test_visualization():
    # Create two agents
    agent1 = Agent()
    agent2 = Agent()
    print("Test Visualization: Created Agent1 and Agent2")

    # Store initial fitness before the game
    initial_fitness_agent1 = agent1.fitness
    initial_fitness_agent2 = agent2.fitness

    # Create a game with visualization enabled
    game = Game(agent1, agent2, visualize=True)
    print("Test Visualization: Created Game")
    game.play()
    print("Test Visualization: Played Game")

    # Update fitness based on game outcome
    agent1.fitness += game.get_fitness_change(
        agent1
    )  # Or adjust according to your fitness rules
    agent2.fitness += game.get_fitness_change(
        agent2
    )  # Or adjust according to your fitness rules

    # Visualize the game with initial fitness
    game.visualize_game(
        agent1, agent2, initial_fitness_agent1, initial_fitness_agent2
    )


def main():
    try:
        Game.total_games_played = 0  # Reset game counter
        print(f"Using matplotlib backend: {matplotlib.get_backend()}")

        # Initialize meta-population
        meta_population = MetaPopulation(
            num_populations=12, population_size=5040
        )
        # Evolve meta-population
        meta_population.evolve(generations=120)

        # View evolution of strategies (printing the first agent's strategy
        # from the first population)
        print("\nEvolved Strategy of First Agent from Population 1:")
        print_strategy(
            meta_population.populations[0].agents[0].strategy.decision_tree
        )

        # Optionally, play a game between a human and an agent
        # from any population
        human = HumanPlayer()
        agent = random.choice(
            random.choice(meta_population.populations).agents
        )
        game = Game(human, agent, visualize=True)

        # Store initial fitness (assuming human fitness is 500)
        initial_fitness_human = getattr(human, "fitness", 500)
        initial_fitness_agent = agent.fitness

        winner = game.play()

        # Update fitness using Game's method
        fitness_change_agent = game.get_fitness_change(agent)
        agent.fitness += fitness_change_agent

        if winner == human:
            print("You win!")
        else:
            print("Agent wins!")

        # Visualize the game with initial fitness values
        try:
            game.visualize_game(
                human, agent, initial_fitness_human, initial_fitness_agent
            )
        except Exception as e:
            print(
                "An exception occurred during human vs "
                f"agent game visualization: {e}"
            )
            logging.error(
                f"Human vs Agent Game Visualization error between {human.id} "
                f"and {agent.id}: {e}"
            )

    except AttributeError as e:
        print(f"AttributeError encountered: {e}")
        logging.error(f"AttributeError in main execution: {e}")
        import traceback

        traceback.print_exc()

    except Exception as e:
        print(f"An exception occurred: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    # unittest.main()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats(10)  # Print top 10 time-consuming functions
