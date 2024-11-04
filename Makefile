.PHONY: install format lint test clean run

install:
pip install -r requirements.txt

format:
black src/ tests/

lint:
flake8 src/ tests/
black --check src/ tests/

test:
pytest tests/

clean:
find . -type d -name "__pycache__" -exec rm -r {} +
find . -type d -name ".pytest_cache" -exec rm -r {} +
find . -type d -name ".coverage" -exec rm -r {} +

run:
python -m src.footsteps_game.main
