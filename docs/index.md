# matfree
Randomised and deterministic matrix-free methods for trace estimation, matrix functions, and/or matrix factorisations.
Builds on [JAX](https://jax.readthedocs.io/en/latest/).

## Installation

To install the package, run

```commandline
pip install matfree
```

**Important:** This assumes you already have a working installation of JAX.
To install JAX, [follow these instructions](https://github.com/google/jax#installation).
To combine `matfree` with a CPU version of JAX, run

```commandline
pip install matfree[cpu]
```
which is equivalent to combining `pip install jax[cpu]` with `pip install matfree`.


## Minimal example

Imports:
```python
>>> import jax
>>> import jax.numpy as jnp
>>> from matfree import hutch, montecarlo, slq

>>> a = jnp.reshape(jnp.arange(12.), (6, 2))
>>> key = jax.random.PRNGKey(1)

```

### Traces

Estimate traces as such:
```python
>>> sample_fun = montecarlo.normal(shape=(2,))
>>> matvec = lambda x: a.T @ (a @ x)
>>> trace = hutch.trace(matvec, key=key, sample_fun=sample_fun)
>>> print(jnp.round(trace))
515.0
>>> # for comparison:
>>> print(jnp.round(jnp.trace(a.T @ a)))
506.0

```
The number of keys determines the number of sequential batches.
Many small batches reduces memory.
Few large batches increases memory and runtime.

Determine the number of samples per batch as follows.

```python
>>> trace = hutch.trace(matvec, key=key, sample_fun=sample_fun, num_batches=10)
>>> print(jnp.round(trace))
507.0
>>> # for comparison:
>>> print(jnp.round(jnp.trace(a.T @ a)))
506.0

```

### Traces and diagonals

Jointly estimating traces and diagonals improves performance.
Here is how to use it:

```python
>>> trace, diagonal = hutch.trace_and_diagonal(matvec, key=key, num_levels=10_000, sample_fun=sample_fun)
>>> print(jnp.round(trace))
507.0

>>> print(jnp.round(diagonal))
[221. 287.]

>>> # for comparison:
>>> print(jnp.round(jnp.trace(a.T @ a)))
506.0

>>> print(jnp.round(jnp.diagonal(a.T @ a)))
[220. 286.]


```

Why is the argument called `num_levels`? Because under the hood,
`trace_and_diagonal` implements a multilevel diagonal-estimation scheme:
```python
>>> _, diagonal_1 = hutch.trace_and_diagonal(matvec, key=key, num_levels=10_000, sample_fun=sample_fun)
>>> init = jnp.zeros(shape=(2,), dtype=float)
>>> diagonal_2 = hutch.diagonal_multilevel(matvec, init, key=key, num_levels=10_000, sample_fun=sample_fun)

>>> print(jnp.round(diagonal_1, 4))
[220.54979 286.7476 ]

>>> print(jnp.round(diagonal_2, 4))
[220.54979 286.7476 ]

>>> diagonal = hutch.diagonal_multilevel(matvec, init, key=key, num_levels=10, num_samples_per_batch=1000, num_batches_per_level=10, sample_fun=sample_fun)
>>> print(jnp.round(diagonal))
[219. 285.]

```

Does the multilevel scheme help? That is not always clear; but [here](https://github.com/pnkraemer/matfree/blob/main/docs/benchmarks/control_variates.py) is a benchmark.

### Determinants


Estimate log-determinants as such:
```python
>>> a = jnp.reshape(jnp.arange(36.), (6, 6)) / 36
>>> sample_fun = montecarlo.normal(shape=(6,))
>>> matvec = lambda x: a.T @ (a @ x) + x
>>> order = 3
>>> logdet = slq.logdet(matvec, order, key=key, sample_fun=sample_fun)
>>> print(jnp.round(logdet))
3.0
>>> # for comparison:
>>> print(jnp.round(jnp.linalg.slogdet(a.T @ a + jnp.eye(6))[1]))
3.0

```

## Contributing

Contributions are absolutely welcome!
Most contributions start with an issue.
Please don't hesitate to create issues in which you
ask for features, give feedback on performances, or simply want to reach out.

To make a pull request, proceed as follows:
Fork the repository.
Install all dependencies with `pip install .[full]` or `pip install -e .[full]`.
Make your changes.
From the root of the project, run the tests via `make test`, and check out `make format` and `make lint` as well.
Use the pre-commit hook if you like.



When making a pull request, keep in mind the following (rough) guidelines:

* Most PRs resolve an issue and change ~10 lines of code (or less)
* Most PRs contain a single commit. [Here is how we can write better commit messages](https://www.freecodecamp.org/news/how-to-write-better-git-commit-messages/).
* Almost every enhancement (e.g. a new feature) is covered by a test.
