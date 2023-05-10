# Developer documentation

## Tests

Install dependencies (assumes JAX is installed; if not, run `pip install .[cpu]`).
```commandline
pip install .[test]
```


Run tests:

```commandline
make test
```

## Format

Install dependencies
```commandline
pip install .[format]
```

Format code:
```commandline
make format
```

## Lint

Install the pre-commit hook:

```commandline
pip install .[lint]
pre-commit install

```

Run linters:

```commandline
make lint
```

## Docs


Install dependencies
```commandline
pip install .[doc]
```


Local preview of docs

```commandline
mkdocs serve
```

Check doc build:
```commandline
make doc
```
