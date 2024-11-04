"""Tests for configuration settings."""

import pytest
from src.footsteps_game.config import CONFIG, Color, GamePhase, Position, Bid


def test_config_values():
    """Test that config contains required values"""
    assert "BOARD_SIZE" in CONFIG
    assert "STARTING_POINTS" in CONFIG
    assert CONFIG["BOARD_SIZE"] > 0
    assert CONFIG["STARTING_POINTS"] > 0


def test_position():
    """Test Position dataclass"""
    pos = Position()
    assert pos.is_empty()

    pos = Position(top_value=1, top_color=Color.WHITE)
    assert not pos.is_empty()
    assert pos.top_value == 1
    assert pos.top_color == Color.WHITE


def test_bid():
    """Test Bid dataclass"""
    bid = Bid()
    assert not bid.is_complete()

    bid = Bid(white=10, black=5)
    assert bid.is_complete()
    assert bid.get_winner() == Color.WHITE

    bid = Bid(white=5, black=10)
    assert bid.get_winner() == Color.BLACK
