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


## Continuous integration


To install all test-related dependencies, (assuming JAX is installed; if not, run `pip install .[cpu]`), execute
```commandline
pip install .[test]
```
Then, run the tests via
```commandline
make test
```

Install all formatting-related dependencies via
```commandline
pip install .[format]
```
and format the code via
```commandline
make format
```

To lint the code, install the pre-commit hook

```commandline
pip install .[lint]
pre-commit install

```
and run the linters via
```commandline
make lint
```

Install the documentation-related dependencies as

```commandline
pip install .[doc]
```
Preview the documentation via

```commandline
make doc-preview
```

and check whether the docs build correctly via

```commandline
make doc-build
```


## Contributing to Matfree

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


## Extending the documentation

**Writing a new tutorial:**

To add a new tutorial, create a Python file in `tutorials/` and fill it with content.
Use docstrings (mirror the style in the existing tutorials).
Make sure to satisfy the formatter and linter.
That's all.

Then, the documentation pipeline will automatically convert those into a format compatible
with Jupytext, which subsequently includes it into the documentation.
If you do not want to make the tutorial part of the documentation, make the filename
have a leading underscore.


**Extending the developer documentation:**

To extend the developer documentation, create a new section in the README.
Use a second-level header (a header that starts with "##") and fill the section
with content.
Then, the documentation pipeline will turn this section into a page in the developer documentation.


**Creating a new module:**

To make a new module appear in the documentation, create the new module in `matfree/`,
and fill it with content.
Unless the module name starts with an underscore or is placed in the backend,
the documentation pipeline will take care of the rest.
