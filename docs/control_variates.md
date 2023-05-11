# Multilevel estimation and control variates

Jointly estimating traces and diagonals improves performance.
Here is how to use it:


Imports:
```python
>>> import jax
>>> import jax.numpy as jnp
>>> from matfree import hutchinson, montecarlo, slq

>>> a = jnp.reshape(jnp.arange(12.0), (6, 2))
>>> key = jax.random.PRNGKey(1)

>>> matvec = lambda x: a.T @ (a @ x)
>>> sample_fun = montecarlo.normal(shape=(2,))

```

```python
>>> trace, diagonal = hutchinson.trace_and_diagonal(
...     matvec, key=key, num_levels=10_000, sample_fun=sample_fun
... )
>>> print(jnp.round(trace))
497.0

>>> print(jnp.round(diagonal))
[216. 281.]

>>> # for comparison:
>>> print(jnp.round(jnp.trace(a.T @ a)))
506.0

>>> print(jnp.round(jnp.diagonal(a.T @ a)))
[220. 286.]


```

Why is the argument called `num_levels`? Because under the hood,
`trace_and_diagonal` implements a multilevel diagonal-estimation scheme:
```python
>>> _, diagonal_1 = hutchinson.trace_and_diagonal(
...     matvec, key=key, num_levels=10_000, sample_fun=sample_fun
... )
>>> init = jnp.zeros(shape=(2,), dtype=float)
>>> diagonal_2 = hutchinson.diagonal_multilevel(
...     matvec, init, key=key, num_levels=10_000, sample_fun=sample_fun
... )

>>> print(jnp.round(diagonal_1, 4))
[215.7592 281.245 ]

>>> print(jnp.round(diagonal_2, 4))
[215.7592 281.245 ]

>>> diagonal = hutchinson.diagonal_multilevel(
...     matvec,
...     init,
...     key=key,
...     num_levels=10,
...     num_samples_per_batch=1000,
...     num_batches_per_level=10,
...     sample_fun=sample_fun,
... )
>>> print(jnp.round(diagonal))
[220. 286.]

```

Does the multilevel scheme help? That is not always clear; but [here](https://github.com/pnkraemer/matfree/blob/main/docs/benchmarks/control_variates.py) is a benchmark.
