format-and-lint:
	pre-commit run --all-files
	blackdoc *.md  --line-length 82

test:
	pytest
	python -m doctest README.md
	python -m doctest matfree/*.py
	python tutorials/1_log_determinants.py
	python tutorials/2_pytree_logdeterminants.py
	python tutorials/3_uncertainty_quantification.py
	python tutorials/4_control_variates.py
	python tutorials/5_vector_calculus.py
	python tutorials/6_low_memory_trace_estimation.py

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
