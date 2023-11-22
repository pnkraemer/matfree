# Log-determinants of pytree-valued functions

Can we compute log-determinants if the matrix-vector products are pytree-valued?
Yes, we can. Here is how.

Imports:

```python

import jax
import jax.flatten_util  # this is important!
import jax.numpy as jnp

from matfree import slq, hutchinson

```
Create a test-problem: a function that maps a pytree (dict) to a pytree (tuple).
Its (regularised) Gauss--Newton Hessian shall be the matrix-vector product
whose log-determinant we estimate.

```python
def testfunc(x):
    """Map a dictionary to a tuple with some arbitrary values."""
    return jnp.linalg.norm(x["weights"]), x["bias"]

# Create a test-input
b = jnp.arange(1.0, 40.0)
W = jnp.stack([b + 1.0, b + 2.0])
x0 = {"weights": W, "bias": b}

# Linearise the functions
f0, jvp = jax.linearize(testfunc, x0)
_f0, vjp = jax.vjp(testfunc, x0)

# Look at the Jacobians -- oh no, they are pytree-valued
print(jax.tree_util.tree_map(jnp.shape, f0))
((), (39,))

print(jax.tree_util.tree_map(jnp.shape, jvp(x0)))
((), (39,))

print(jax.tree_util.tree_map(jnp.shape, vjp(f0)))
({'bias': (39,), 'weights': (2, 39)},)

```

To compute log-determinants, we need to transform the functions and states.
The reason is that the linear algebra that underlies stochastic Lanczos quadrature
has no means of handling arbitrary pytrees -- only matrices and matrix-vector products.

The transformation we are looking for is "ravelling" a pytree
(think: flattening of the tree).

```python
x0_flat, unravel_func_x = jax.flatten_util.ravel_pytree(x0)
f0_flat, unravel_func_f = jax.flatten_util.ravel_pytree(f0)

def matvec(f_flat, alpha=1e-1):
    """Matrix-vector product x -> (J J^\top + \alpha I) x."""
    f_unravelled = unravel_func_f(f_flat)
    vjp_eval = vjp(f_unravelled)
    matvec_eval = jvp(*vjp_eval)
    f_eval, _unravel_func = jax.flatten_util.ravel_pytree(matvec_eval)
    return f_eval + alpha * f_flat


```
Now, we can compute the log-determinant with the flattened inputs as usual:

```python
# Compute the log-determinant
key = jax.random.PRNGKey(seed=1)
sample_fun = hutchinson.sampler_normal(shape=f0_flat.shape)
order = 3
logdet = slq.logdet_spd(order, matvec, key=key, sample_fun=sample_fun)

# Look at the results
print(jnp.round(logdet, 2))
3.81

# Materialise the matrix-vector product and compute the true log-determinant.
M = jax.jacfwd(matvec)(f0_flat)
print(jnp.round(jnp.linalg.slogdet(M)[1], 2))
3.81

```
