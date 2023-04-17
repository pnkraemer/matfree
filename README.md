# hutch
Matrix-free stochastic trace- and diagonal-estimation and related functions.


## Minimal example

Imports:
```python
>>> import jax
>>> import jax.numpy as jnp
>>> from hutch import hutch

>>> a = jnp.reshape(jnp.arange(12.), (6, 2))
>>> key = jax.random.PRNGKey(1)

```

### Traces

Estimate traces as such:
```python
>>> 
>>> trace = hutch.trace(
...     key=key,
...     matvec_fn=lambda x: a.T @ (a @ x), 
...     tangents_shape=(2,), 
...     tangents_dtype=float,
...     num_batches=2,
... )
>>> print(jnp.round(trace))
508.0
>>> # for comparison:
>>> print(jnp.round(jnp.trace(a.T @ a)))
506.0

```
The number of keys determines the number of sequential batches.
Many small batches reduces memory.
Few large batches increases memory and runtime.

Determine the number of samples per batch as follows.

```python
>>> trace = hutch.trace(
...     key=key,
...     num_batches=10,
...     matvec_fn=lambda x: a.T @ (a @ x), 
...     tangents_shape=(2,), 
...     tangents_dtype=float, 
...     num_samples_per_batch=1000
... )
>>> print(jnp.round(trace))
513.0
>>> # for comparison:
>>> print(jnp.round(jnp.trace(a.T @ a)))
506.0

```

### Traces and diagonals

Jointly estimating traces and diagonals improves performance.
Here is how to use it:

```python
>>> keys = jax.random.split(key, num=10_000)  
>>> trace, diagonal = hutch.trace_and_diagonal(
...     keys=keys,
...     matvec_fn=lambda x: a.T @ (a @ x), 
...     tangents_shape=(2,), 
...     tangents_dtype=float, 
... )
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
