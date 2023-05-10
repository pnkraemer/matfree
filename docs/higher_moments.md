# Higher moments and UQ

```python
>>> import jax
>>> import jax.numpy as jnp
>>> from matfree import hutchinson, montecarlo, slq

>>> a = jnp.reshape(jnp.arange(12.0), (6, 2))
>>> key = jax.random.PRNGKey(1)

>>> matvec = lambda x: a.T @ (a @ x)
>>> sample_fun = montecarlo.normal(shape=(2,))

```

## Higher moments

Trace estimation involves estimating expected values of random variables.
Sometimes, second and higher moments of a random variable are interesting.
Compute them as such

```python
>>> a = jnp.reshape(jnp.arange(36.0), (6, 6)) / 36
>>> sample_fun = montecarlo.normal(shape=(6,))
>>> matvec = lambda x: a.T @ (a @ x) + x
>>> first, second = hutchinson.trace_moments(matvec, key=key, sample_fun=sample_fun)
>>> print(jnp.round(first, 1))
17.5
>>> print(jnp.round(second, 1))
631.7

```

For normal-style samples, we know that the variance is twice the squared Frobenius norm:

```python
>>> print(jnp.round(second - first**2, 1))
326.9

>>> A = a.T @ a + jnp.eye(6)
>>> print(jnp.round(2 * jnp.linalg.norm(A, ord="fro") ** 2, 1))
321.80002

```


### Uncertainty quantification

Variance estimation leads to uncertainty quantification:
The variance of the estimator is equal to the variance of the random variable
divided by the number of samples.
Implement this as follows:

```python
>>> a = jnp.reshape(jnp.arange(36.0), (6, 6)) / 36
>>> sample_fun = montecarlo.normal(shape=(6,))
>>> num_samples = 10_000
>>> matvec = lambda x: a.T @ (a @ x) + x
>>> first, second = hutchinson.trace_moments(
...     matvec,
...     key=key,
...     sample_fun=sample_fun,
...     moments=(1, 2),
...     num_batches=1,
...     num_samples_per_batch=num_samples,
... )
>>> variance = (second - first**2) / num_samples
>>> print(jnp.round(variance, 2))
0.03

```
