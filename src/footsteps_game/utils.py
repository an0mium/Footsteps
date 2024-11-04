"""Utility functions for the Footsteps game."""

import logging
from logging.handlers import RotatingFileHandler
import platform
import sys
import select
import termios
import tty
import threading


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
        """Check if keyboard input is available"""
        if self.platform == "Windows":
            return msvcrt.kbhit()
        else:
            return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

    def read_char(self):
        """Read a single character from keyboard"""
        if self.platform == "Windows":
            return msvcrt.getch().decode("utf-8")
        else:
            return sys.stdin.read(1)
