# hutch
Matrix-free stochastic trace- and diagonal-estimation and related functions.


## Minimal example

Imports:
```python
>>> import jax
>>> import jax.numpy as jnp
>>> from hutch import hutch, sample

>>> a = jnp.reshape(jnp.arange(12.), (6, 2))
>>> key = jax.random.PRNGKey(1)

```

### Traces

Estimate traces as such:
```python
>>> sample_fun = sample.normal(shape=(2,), dtype=float)
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
