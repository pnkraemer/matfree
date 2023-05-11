# matfree: Matrix-free linear algebra in JAX

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
>>> from matfree import hutchinson, montecarlo, slq

>>> a = jnp.reshape(jnp.arange(12.0), (6, 2))
>>> key = jax.random.PRNGKey(1)

```

Estimate traces as such:
```python
>>> sample_fun = montecarlo.normal(shape=(2,))
>>> matvec = lambda x: a.T @ (a @ x)
>>> trace = hutchinson.trace(matvec, key=key, sample_fun=sample_fun)
>>> print(jnp.round(trace))
514.0
>>> # for comparison:
>>> print(jnp.round(jnp.trace(a.T @ a)))
506.0

```
The number of keys determines the number of sequential batches.
Many small batches reduces memory.
Few large batches increases memory and runtime.

Determine the number of samples per batch as follows.

```python
>>> trace = hutchinson.trace(matvec, key=key, sample_fun=sample_fun, num_batches=10)
>>> print(jnp.round(trace))
508.0
>>> # for comparison:
>>> print(jnp.round(jnp.trace(a.T @ a)))
506.0

```


## More tutorials

- **Control variates:** Use control variates and multilevel schemes to reduce variances.  [(LINK)](https://pnkraemer.github.io/matfree/control_variates/)
- **Log-determinants:**  Use stochastic Lanczos quadrature to compute matrix functions. [(LINK)](https://pnkraemer.github.io/matfree/log_determinants/)
- **Higher moments and UQ:** Compute means, variances, and other moments simultaneously. [(LINK)](https://pnkraemer.github.io/matfree/higher_moments/)
- **Vector calculus:** Use matrix-free linear algebra to implement vector calculus. [(LINK)](https://pnkraemer.github.io/matfree/vector_calculus/)



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
