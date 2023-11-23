format:
	isort .  --quiet
	black .  --quiet
	blackdoc *.md  --line-length 82

lint:
	pre-commit run --all-files

test:
	python -m doctest README.md
	pytest -x -v


clean-preview:
	git clean -xdn

clean:
	git clean -xdf


doc-preview:
	python scripts/generate_api_docs.py
	python scripts/readme_to_dev_docs.py
	python scripts/tutorials_to_py_light.py
	mkdocs serve

doc-build:
	python scripts/generate_api_docs.py
	python scripts/readme_to_dev_docs.py
	python scripts/tutorials_to_py_light.py
	mkdocs build

run-benchmarks:
	python docs/benchmarks/control_variates.py
	python docs/benchmarks/jacobian_squared.py
