"""Tests for the Agent class"""

import pytest
from src.footsteps_game.agents import Agent
from src.footsteps_game.strategy import Strategy


def test_agent_initialization():
    """Test basic agent initialization"""
    agent = Agent()
    assert agent.id.startswith("A-")
    assert agent.population_id == 1
    assert isinstance(agent.strategy, Strategy)
    assert agent.fitness == 500
    assert agent.game_counter == 0


def test_agent_bid():
    """Test agent bidding behavior"""
    agent = Agent()
    bid = agent.get_bid(100, 100)
    assert isinstance(bid, int)
    assert bid > 0
    assert bid <= 100


def test_agent_move():
    """Test agent move generation"""
    agent = Agent()
    move = agent.get_move(
        my_position=(0, 0),
        opponent_position=(5, 5),
        board_size=11,
        goal=(10, 10),
        opponent_goal=(0, 0),
    )
    assert isinstance(move, tuple)
    assert len(move) == 2
    assert all(isinstance(x, int) for x in move)
