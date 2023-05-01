format:
	isort .  --quiet
	black .  --quiet

lint:
	pre-commit run --all-files

test:
	python -m doctest README.md
	python -m doctest docs/index.md
	pytest -x -v


clean:
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf *.egg-info
	rm -rf dist site build
	rm -rf *.ipynb_checkpoints

doc:
	mkdocs build

run-benchmarks:
	python docs/benchmarks/control_variates.py
	python docs/benchmarks/jacobian_squared.py
