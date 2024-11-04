"""Tests for the Strategy class"""

import pytest
from src.footsteps_game.strategy import Strategy


def test_strategy_initialization():
    """Test strategy initialization"""
    strategy = Strategy()
    assert strategy.decision_tree is not None
    assert strategy.bid_params is not None
    assert "thresholds" in strategy.bid_params
    assert "low_bid_range" in strategy.bid_params


def test_strategy_mutation():
    """Test strategy mutation"""
    strategy = Strategy()
    original_tree = deepcopy(strategy.decision_tree)
    original_params = deepcopy(strategy.bid_params)

    strategy.mutate(mutation_rate=1.0)  # Force mutation

    # Check that something changed
    assert (
        strategy.decision_tree != original_tree
        or strategy.bid_params != original_params
    )


def test_strategy_crossover():
    """Test strategy crossover"""
    strategy1 = Strategy()
    strategy2 = Strategy()

    child = strategy1.crossover(strategy2)

    assert isinstance(child, Strategy)
    assert child.decision_tree is not None
    assert child.bid_params is not None


def test_bid_params_ranges():
    """Test bid parameters are within valid ranges"""
    strategy = Strategy()

    assert all(0 <= x <= 1 for x in strategy.bid_params["low_bid_range"])
    assert all(0 <= x <= 1 for x in strategy.bid_params["medium_bid_range"])
    assert all(0 <= x <= 1 for x in strategy.bid_params["high_bid_range"])
    assert -500 <= strategy.bid_params["bid_constant"] <= 500
