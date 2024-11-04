"""Strategy implementation for agents"""

import random
import math
from copy import deepcopy
from typing import Dict, Any, Union, Optional


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

    def generate_random_bid_params(self) -> Dict[str, Any]:
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

    def mutate(self, mutation_rate: float = 0.05) -> None:
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
        self._mutate_bid_params(mutation_rate)

    def _mutate_bid_params(self, mutation_rate: float) -> None:
        """Mutates the bid parameters"""
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

    def crossover(self, other_strategy: "Strategy", min_depth: int = 4) -> "Strategy":
        """Performs crossover with another strategy"""
        tree1 = deepcopy(self.decision_tree)
        tree2 = deepcopy(other_strategy.decision_tree)

        subtree1, parent1, key1 = self.get_random_subtree(tree1)
        subtree2, parent2, key2 = self.get_random_subtree(tree2)

        if parent1 and key1:
            parent1[key1] = subtree2
        else:
            tree1 = subtree2

        if parent2 and key2:
            parent2[key2] = subtree1
        else:
            tree2 = subtree1

        # Crossover bid parameters
        new_bid_params = self._crossover_bid_params(other_strategy.bid_params)

        child_tree = random.choice([tree1, tree2])
        child_strategy = Strategy(child_tree, new_bid_params)
        child_strategy.enforce_minimum_depth(min_depth)

        return child_strategy

    def _crossover_bid_params(self, other_params: Dict) -> Dict:
        """Crosses over bid parameters with another strategy"""
        new_params = {}
        for key in self.bid_params:
            if key == "bid_constant":
                new_params[key] = int((self.bid_params[key] + other_params[key]) / 2)
            else:
                new_params[key] = [
                    random.choice([self.bid_params[key][0], other_params[key][0]]),
                    random.choice([self.bid_params[key][1], other_params[key][1]]),
                ]
                new_params[key] = sorted(new_params[key])
        return new_params
