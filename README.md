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
>>> keys = jax.random.split(key, num=2)  # 2 sequential batches

```


Estimate traces as such:
```python
>>> 
>>> trace = hutch.trace(
...     keys=keys,
...     matvec_fn=lambda x: a.T @ (a @ x), 
...     tangents_shape=(2,), 
...     tangents_dtype=float, 
... )
>>> print(jnp.round(trace))
504.0

```
The number of keys determines the number of sequential batches.
Many small batches reduces memory.
Few large batches increases memory and runtime.

Determine the number of samples per batch as follows.

```python
>>> # This time, 10 sequential batches
>>> keys = jax.random.split(key, num=10)  
>>> trace = hutch.trace(
...     keys=keys,
...     matvec_fn=lambda x: a.T @ (a @ x), 
...     tangents_shape=(2,), 
...     tangents_dtype=float, 
...     num_samples_per_key=100
... )
>>> print(jnp.round(trace))
510.0

```
