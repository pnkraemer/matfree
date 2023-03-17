# hutch
Stochastic trace estimation and so on



```python
>>> import jax
>>> import jax.numpy as jnp
>>> from hutch import hutch

>>> a = jnp.reshape(jnp.arange(12.), (6, 2))
>>> key = jax.random.PRNGKey(1)
>>> 
>>> trace = hutch.trace(matvec_fn=lambda x: a.T @ (a @ x), tangents_shape=(2,), tangents_dtype=float, batch_size=100, batch_keys=jnp.asarray([key]))
>>> print(trace)
476.0

```
