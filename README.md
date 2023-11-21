# matfree: Matrix-free linear algebra in JAX

[![Actions status](https://github.com/pnkraemer/matfree/workflows/ci/badge.svg)](https://github.com/pnkraemer/matfree/actions)
[![image](https://img.shields.io/pypi/v/matfree.svg)](https://pypi.python.org/pypi/matfree)
[![image](https://img.shields.io/pypi/l/matfree.svg)](https://pypi.python.org/pypi/matfree)
[![image](https://img.shields.io/pypi/pyversions/matfree.svg)](https://pypi.python.org/pypi/matfree)

Randomised and deterministic matrix-free methods for trace estimation, matrix functions, and/or matrix factorisations.
Builds on [JAX](https://jax.readthedocs.io/en/latest/).



- ⚡ A stand-alone implementation of **stochastic Lanczos quadrature**
- ⚡ Stochastic **trace estimation** including batching, control variates, and uncertainty quantification
- ⚡ Matrix-free matrix decompositions for **large sparse eigenvalue problems**

and many other things.
Everything is natively compatible with JAX' feature set:
JIT compilation, automatic differentiation, vectorisation, and pytrees.
[_Let us know what you think about matfree!_](https://github.com/pnkraemer/matfree/issues)

[**Installation**](#installation) |
[**Minimal example**](#minimal-example) |
[**Tutorials**](#tutorials) |
[**Contributing**](#contributing) |
[**API docs**](https://pnkraemer.github.io/matfree/api/hutchinson/)


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
(But do not only use matfree on CPU!)

## Minimal example

Import matfree and JAX, and set up a test problem.

```python
>>> import jax
>>> import jax.numpy as jnp
>>> from matfree import hutchinson, montecarlo, slq

>>> A = jnp.reshape(jnp.arange(12.0), (6, 2))
>>>
>>> def matvec(x):
...     return A.T @ (A @ x)
...

```

Estimate the trace of the matrix:

```python
>>> key = jax.random.PRNGKey(1)
>>> normal = hutchinson.normal(shape=(2,))
>>> trace = hutchinson.trace(matvec, key=key, sample_fun=normal)
>>>
>>> print(jnp.round(trace))
514.0
>>>
>>> # for comparison:
>>> print(jnp.round(jnp.trace(A.T @ A)))
506.0

```
Adjust the batch-size to improve the performance
- More, smaller batches reduce memory but increase the runtime.
- Fewer, larger batches increase memory but reduce the runtime.


Change the number of batches as follows:

```python
>>> trace = hutchinson.trace(matvec, key=key, sample_fun=normal, num_batches=10)
>>> print(jnp.round(trace))
508.0
>>>
>>> # for comparison:
>>> print(jnp.round(jnp.trace(A.T @ A)))
506.0

```


## Tutorials

Here are some more advanced tutorials:

- **Control variates:** Use control variates and multilevel schemes to reduce variances.  [(LINK)](https://pnkraemer.github.io/matfree/control_variates/)
- **Log-determinants:**  Use stochastic Lanczos quadrature to compute matrix functions. [(LINK)](https://pnkraemer.github.io/matfree/log_determinants/)
- **Higher moments and UQ:** Compute means, variances, and other moments simultaneously. [(LINK)](https://pnkraemer.github.io/matfree/higher_moments/)
- **Vector calculus:** Use matrix-free linear algebra to implement vector calculus. [(LINK)](https://pnkraemer.github.io/matfree/vector_calculus/)
- **Pytree-valued states:** Combining neural-network Jacobians with stochastic Lanczos quadrature. [(LINK)](https://pnkraemer.github.io/matfree/pytree_logdeterminants/)

[_Let us know_](https://github.com/pnkraemer/matfree/issues) what you use matfree for!

## Contributing

Contributions are absolutely welcome!

**Issues:**

Most contributions start with an issue.
Please don't hesitate to [_create issues_](https://github.com/pnkraemer/matfree/issues) in which you
ask for features, give feedback on performances, or simply want to reach out.

**Pull requests:**

To make a pull request, proceed as follows:

- Fork the repository.
- Install all dependencies with `pip install .[full]` or `pip install -e .[full]`.
- Make your changes.
- From the root of the project, run the tests via `make test`, and check out `make format` and `make lint` as well. Use the pre-commit hook if you like.



When making a pull request, keep in mind the following (rough) guidelines:

* Most PRs resolve an issue.
* Most PRs contain a single commit. [Here is how we can write better commit messages](https://www.freecodecamp.org/news/how-to-write-better-git-commit-messages/).
* Most enhancements (e.g. new features) are covered by tests.
