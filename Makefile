format:
	isort .
	black .

lint:
	pre-commit run --all-files

test:
	pytest

clean:
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf *.egg-info
	rm -rf dist site build
	rm -rf *.ipynb_checkpoints
