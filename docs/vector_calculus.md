# Vector calculus

Implementing vector calculus with conventional algorithmic differentiation can be inefficient.
For example, computing the divergence of a vector field requires computing the trace of a Jacobian.
The divergence of a vector field is important when evaluating Laplacians of scalar functions.

Here is how we can implement divergences and Laplacians without forming full Jacobian matrices:



```python
>>> import jax
>>> import jax.numpy as jnp
>>> from matfree import hutchinson, montecarlo

```

## Divergences and Laplacians

The divergence of a vector field is the trace of its Jacobian.
The conventional implementation would look like this:

```python

>>> def divergence_dense(vf):
...     """Compute the divergence of a vector field."""
...
...     def div_fn(x):
...         J = jax.jacfwd(vf)
...         return jnp.trace(J(x))
...
...     return div_fn
...

```
This implementation computes the divergence of a vector field:

```python
>>> def fun(x):
...     """A scalar valued function."""
...     return jnp.dot(x, x) ** 2
...
>>> x0 = jnp.arange(1.0, 4.0)
>>> gradient = jax.grad(fun)
>>> laplacian = divergence_dense(gradient)
>>> print(jax.hessian(fun)(x0))
[[ 64.  16.  24.]
 [ 16.  88.  48.]
 [ 24.  48. 128.]]

>>> print(laplacian(x0))
280.0

```

But the implementation above requires $O(d^2)$ storage because it evaluates the dense Jacobian.
This is problematic for high-dimensional problems.

## Matrix-free implementation

If we have access to Jacobian-vector products (which we usually do), we can use matrix-free trace estimation
to approximate divergences and Laplacians without forming full Jacobians:


```python

>>> def divergence_matfree(vf, *, key, sample_fun):
...     """Compute the divergence of a vector field only using matrix-vector products."""
...
...     def div_fn(x):
...         def jvp(v):
...             _vf_value, jvp_value = jax.jvp(fun=vf, primals=(x,), tangents=(v,))
...             return jvp_value
...
...         return hutchinson.trace(jvp, key=key, sample_fun=sample_fun)
...
...     return div_fn
...

```
The difference to the above implementation is that `divergence_matfree` does not form dense Jacobians.
It requires $O(d)$ memory and  $O(d N)$ operations (for $N$ Monte-Carlo samples).
For large-scale problems, it may be the only way of computing Laplacians reliably.

```python
>>> sample_fun = montecarlo.normal(shape=(3,))
>>> laplacian_dense = divergence_dense(gradient)
>>> laplacian_matfree = divergence_matfree(
...     gradient, key=jax.random.PRNGKey(1), sample_fun=sample_fun
... )
>>>
>>> print(jnp.round(laplacian_dense(x0), 1))
280.0

>>> print(jnp.round(laplacian_matfree(x0), 1))
278.4

```

In summary, compute matrix-free linear algebra and algorithmic differentiation to implement vector calculus.

## Jacobian diagonals

If we replace trace estimation with diagonal estimation, we can compute the diagonal of Jacobian matrices in $O(d)$ memory and $O(dN)$ operations
