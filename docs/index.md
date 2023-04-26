# matfree
Randomised and deterministic matrix-free methods for trace estimation, matrix functions, and/or matrix factorisations.


## Minimal example

Imports:
```python
>>> import jax
>>> import jax.numpy as jnp
>>> from matfree import hutch, montecarlo

>>> a = jnp.reshape(jnp.arange(12.), (6, 2))
>>> key = jax.random.PRNGKey(1)

```

### Traces

Estimate traces as such:
```python
>>> sample_fun = montecarlo.normal(shape=(2,), dtype=float)
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
>>> keys = jax.random.split(key, num=10_000)  
>>> trace, diagonal = hutch.trace_and_diagonal(matvec, keys=keys, sample_fun=sample_fun)
>>> print(jnp.round(trace))
509.0

>>> print(jnp.round(diagonal))
[222. 287.]

>>> # for comparison:
>>> print(jnp.round(jnp.trace(a.T @ a)))
506.0

>>> print(jnp.round(jnp.diagonal(a.T @ a)))
[220. 286.]


```

## Contributing

Contributions are absolutely welcome! 
To set up your developer setup, fork the repository, and install all dependencies with `pip install .[full]` or `pip install -e .[full]`.
From the root of the project, run the tests via `make test`, and check out `make format` and `make lint`.
There is a pre-commit hook that can be used, too.

When making a pull request, keep in mind the following (rough) guidelines:

* Every PR usually resolves an issue and changes ~10 lines of code (or less)
* Every PR usually contains a single commit. The commit message is usually very descriptive
* Every enhancement is usually covered by a test.
