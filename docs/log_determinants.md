# Determinants



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


Estimate log-determinants as such:
```python
>>> a = jnp.reshape(jnp.arange(36.0), (6, 6)) / 36
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
