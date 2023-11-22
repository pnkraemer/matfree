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


**Installation**

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

**Minimal example**

Import matfree and JAX, and set up a test problem.

```python
>>> import jax
>>> import jax.numpy as jnp
>>> from matfree import hutchinson
>>>
>>> jnp.set_printoptions(1)

>>> A = jnp.reshape(jnp.arange(12.0), (6, 2))
>>>
>>> def matvec(x):
...     return A.T @ (A @ x)
...

```

Estimate the trace of the matrix:

```python
>>> # Determine the shape of the base-samples
>>> input_like = jnp.zeros((2,), dtype=float)
>>> sampler = hutchinson.sampler_rademacher(input_like, num=10_000)
>>>
>>> # Set Hutchinson's method up to compute the traces
>>> # (instead of, e.g., diagonals)
>>> integrand = hutchinson.integrand_trace(matvec)
>>>
>>> # Compute an estimator
>>> estimate = hutchinson.hutchinson(integrand, sampler)

>>> # Estimate
>>> key = jax.random.PRNGKey(1)
>>> trace = jax.jit(estimate)(key)
>>>
>>> print(trace)
508.9
>>>
>>> # for comparison:
>>> print((jnp.trace(A.T @ A)))
506.0

```


**Tutorials**

Here are some more advanced tutorials:

- **Control variates:** Use control variates and multilevel schemes to reduce variances.  [(LINK)](https://pnkraemer.github.io/matfree/control_variates/)
- **Log-determinants:**  Use stochastic Lanczos quadrature to compute matrix functions. [(LINK)](https://pnkraemer.github.io/matfree/log_determinants/)
- **Higher moments and UQ:** Compute means, variances, and other moments simultaneously. [(LINK)](https://pnkraemer.github.io/matfree/higher_moments/)
- **Vector calculus:** Use matrix-free linear algebra to implement vector calculus. [(LINK)](https://pnkraemer.github.io/matfree/vector_calculus/)
- **Pytree-valued states:** Combining neural-network Jacobians with stochastic Lanczos quadrature. [(LINK)](https://pnkraemer.github.io/matfree/pytree_logdeterminants/)

[_Let us know_](https://github.com/pnkraemer/matfree/issues) what you use matfree for!


## Extended installation guide


**Tests:**

Install dependencies (assumes JAX is installed; if not, run `pip install .[cpu]`).
```commandline
pip install .[test]
```


Run tests:

```commandline
make test
```

**Format:**

Install dependencies
```commandline
pip install .[format]
```

Format code:
```commandline
make format
```

**Lint:**

Install the pre-commit hook:

```commandline
pip install .[lint]
pre-commit install

```

Run linters:

```commandline
make lint
```

**Docs:**


Install dependencies
```commandline
pip install .[doc]
```


Local preview of docs

```commandline
make doc-preview
```

Check doc build:
```commandline
make doc-build
```


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
