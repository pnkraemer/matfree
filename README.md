# matfree: Matrix-free linear algebra in JAX

[![Actions status](https://github.com/pnkraemer/matfree/workflows/ci/badge.svg)](https://github.com/pnkraemer/matfree/actions)
[![image](https://img.shields.io/pypi/v/matfree.svg)](https://pypi.python.org/pypi/matfree)
[![image](https://img.shields.io/pypi/l/matfree.svg)](https://pypi.python.org/pypi/matfree)
[![image](https://img.shields.io/pypi/pyversions/matfree.svg)](https://pypi.python.org/pypi/matfree)

Randomised and deterministic matrix-free methods for trace estimation, functions of matrices, and/or matrix factorisations.
Builds on [JAX](https://jax.readthedocs.io/en/latest/).


- ⚡ Stochastic **trace estimation** including batching, control variates, and uncertainty quantification
- ⚡ A stand-alone implementation of **stochastic Lanczos quadrature** for traces of functions of matrices
- ⚡ Matrix-decomposition algorithms for **large sparse eigenvalue problems**: tridiagonalisation, bidiagonalisation, Hessenberg factorisation via Lanczos and Arnoldi iterations
- ⚡ Chebyshev, Lanczos, and Arnoldi-based methods for approximating **functions of large matrices**
- ⚡ **Gradients of functions of large matrices** (like in [this paper](https://arxiv.org/abs/2405.17277)) via differentiable Lanczos and Arnoldi iterations
- ⚡ Partial Cholesky **preconditioners** with and without pivoting

and many other things.
Everything is natively compatible with the rest of JAX:
JIT compilation, automatic differentiation, vectorisation, and PyTrees.
[_Let us know what you think about matfree!_](https://github.com/pnkraemer/matfree/issues)


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
>>> from matfree import stochtrace
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
>>> sampler = stochtrace.sampler_rademacher(input_like, num=10_000)
>>>
>>> # Set Hutchinson's method up to compute the traces
>>> # (instead of, e.g., diagonals)
>>> integrand = stochtrace.integrand_trace(matvec)
>>>
>>> # Compute an estimator
>>> estimate = stochtrace.estimator(integrand, sampler)

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

Find many more tutorials in [Matfree's documentation](https://pnkraemer.github.io/matfree/).

These tutorials include, among other things:

- **Log-determinants:**  Use stochastic Lanczos quadrature to compute matrix functions.
- **Pytree-valued states:** Combining neural-network Jacobians with stochastic Lanczos quadrature.
- **Control variates:** Use control variates and multilevel schemes to reduce variances.
- **Higher moments and UQ:** Compute means, variances, and other moments simultaneously.
- **Vector calculus:** Use matrix-free linear algebra to implement vector calculus.
- **Low-memory trace estimation:** Combine Matfree's API with JAX's function transformations for low-memory stochastic trace estimation.


[_Let us know_](https://github.com/pnkraemer/matfree/issues) what you use matfree for!


**Citation**

Thank you for using Matfree!
If you are using Matfree's differentiable Lanczos or Arnoldi iterations, then you
are using the algorithms from [this paper](https://arxiv.org/abs/2405.17277).
We would appreciate if you cited it as follows:

```bibtex
@article{kraemer2024gradients,
    title={Gradients of functions of large matrices},
    author={Kr\"amer, Nicholas and Moreno-Mu\~noz, Pablo and Roy, Hrittik and Hauberg S\o{}ren},
    journal={arXiv preprint arXiv:2405.17277},
    year={2024}
}
```

Some of Matfree's docstrings contain additional bibliographic information.
For example, the functions in `matfree.bounds` link to bibtex entries for the articles associated with each bound.
Go check out the [API documentation](https://pnkraemer.github.io/matfree/).


## Use Matfree's continuous integration


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
pip install .[format-and-lint]
pre-commit install
```
and format the code via
```commandline
make format-and-lint
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


## Contribute to Matfree

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
- From the root of the project, run the tests via `make test`, and check out `make format-and-lint` as well. Use the pre-commit hook if you like.


When making a pull request, keep in mind the following (rough) guidelines:

* Most PRs resolve an issue.
* Most PRs contain a single commit. [Here is how we can write better commit messages](https://www.freecodecamp.org/news/how-to-write-better-git-commit-messages/).
* Most enhancements (e.g. new features) are covered by tests.


## Extend Matfree's documentation

**Write a new tutorial:**

To add a new tutorial, create a Python file in `tutorials/` and fill it with content.
Use docstrings (mirror the style in the existing tutorials).
Make sure to satisfy the formatter and linter.
That's all.

Then, the documentation pipeline will automatically convert those into a format compatible
with Jupytext, which subsequently includes it into the documentation.
If you do not want to make the tutorial part of the documentation, make the filename
have a leading underscore.


**Extend the developer documentation:**

To extend the developer documentation, create a new section in the README.
Use a second-level header (a header that starts with "##") and fill the section
with content.
Then, the documentation pipeline will turn this section into a page in the developer documentation.


**Create a new module:**

To make a new module appear in the documentation, create the new module in `matfree/`,
and fill it with content.
Unless the module name starts with an underscore or is placed in the backend,
the documentation pipeline will take care of the rest.


## Understand Matfree's API policy

With the upcoming `v0.1.0` release, Matfree will not be experimental anymore.
This means it will start following [semantic versioning](https://semver.org/).
(It already does to some extent, but not very strictly.)

However, _Matfree remains a work in progress_, and parts of its API may change frequently and without warning.
To quote from [semantic versioning](https://semver.org/):

> Major version zero (0.y.z) is for initial development. Anything MAY change at any time. The public API SHOULD NOT be considered stable.


We do not implement an official deprecation policy just yet, but handle all API change communication via version increments:

- If a change is backwards-compatible (e.g. a backwards-compatible new feature, or a bugfix), the patch version is incremented: e.g., from `v0.1.5` to `v0.1.6`.
- If a change is not backwards-compatible, the minor version is incremented: e.g., from `v0.1.6` to `v0.2.0`.

To depend on Matfree's API, pin the minor version (e.g. `matfree <= 0.2.0`) to avoid breaking your code, but please feel encouraged to upgrade regularly to enjoy the new stuff!
