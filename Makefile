format:
	isort .  --quiet
	black .  --quiet
	blackdoc docs/*.md *.md  --line-length 82

lint:
	pre-commit run --all-files

test:
	python -m doctest README.md
	pytest -x -v


clean:
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf .mypy_cache
	rm -rf .coverage
	rm -rf *.egg-info
	rm -rf dist site build htmlcov
	rm -rf *.ipynb_checkpoints
	rm matfree/_version.py
	rm -rf docs/API/

doc-preview:
	python scripts/generate_api_docs.py --source matfree --skip backend/* --target docs/API_documentation/
	cp README.md docs/index.md
	mkdocs serve

doc:
	python scripts/generate_api_docs.py --source matfree --skip backend/* --target docs/API_documentation/
	cp README.md docs/index.md
	mkdocs build

run-benchmarks:
	python docs/benchmarks/control_variates.py
	python docs/benchmarks/jacobian_squared.py
