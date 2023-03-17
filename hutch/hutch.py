# """Hutchinson-style trace and diagonal estimation.
#
# See e.g.
#
# http://www.nowozin.net/sebastian/blog/thoughts-on-trace-estimation-in-deep-learning.html
#
# """
#
# import functools
#
# import jax
# import jax.numpy as jnp
#
#
# @functools.partial(
#     jax.jit,
#     static_argnames=(
#         "matvec_fn",
#         "tangents_shape",
#         "tangents_dtype",
#         "batch_size",
#         "generate_samples_fn",
#     ),
# )
# def trace(matvec_fn, **kwargs):
#     """Estimate the trace of a matrix stochastically."""
#
#     def Q(x):
#         return jnp.dot(x, matvec_fn(x))
#
#     return _stochastic_estimate(Q, **kwargs)
#
#
# def diagonal(matvec_fn, **kwargs):
#     """Estimate the diagonal of a matrix stochastically."""
#
#     def Q(x):
#         return x * matvec_fn(x)
#
#     return _stochastic_estimate(Q, **kwargs)
#
#
# def _stochastic_estimate(*args, batch_keys, **kwargs):
#     """Hutchinson-style stochastic estimation."""
#
#     @jax.jit
#     def f(key):
#         return _stochastic_estimate_batch(*args, key=key, **kwargs)
#
#     # Compute batches sequentially to reduce memory.
#     batch_keys = jnp.atleast_2d(batch_keys)
#     means = jax.lax.map(f, xs=batch_keys)
#
#     # Mean of batches is the mean of the total expectation
#     return jnp.mean(means, axis=0)
#
#
# def _stochastic_estimate_batch(
#     Q,
#     /,
#     *,
#     tangents_shape,
#     tangents_dtype,
#     key,
#     batch_size,
#     generate_samples_fn=jax.random.rademacher,
# ):
#     shape = (batch_size,) + tangents_shape
#     samples = generate_samples_fn(key, shape=shape, dtype=tangents_dtype)
#     return jnp.mean(jax.vmap(Q)(samples), axis=0)
#
#
# @functools.partial(
#     jax.jit,
#     static_argnames=(
#         "matvec_fn",
#         "tangents_shape",
#         "tangents_dtype",
#         "generate_samples_fn",
#     ),
# )
# def trace_and_diagonal(
#     matvec_fn,
#     *,
#     tangents_shape,
#     tangents_dtype,
#     keys,
#     generate_samples_fn,
# ):
#     """Jointly estimate the trace and the diagonal stochastically.
#
#     The advantage of computing both quantities simultaneously is
#     that the diagonal estimate
#     may serve as a control variate for the trace estimate,
#     thus reducing the variance of the estimator
#     (and thereby accelerating convergence.)
#
#     See:
#     Adams et al., Estimating the Spectral Density of Large Implicit Matrices, 2018.
#     """
#     zeros = jnp.zeros(shape=tangents_shape, dtype=tangents_dtype)
#     init = (0.0, zeros, 0)  # trace, diag, n
#
#     def body_fn(carry, key):
#         trace, diag, n = carry
#
#         z = generate_samples_fn(key, shape=tangents_shape, dtype=tangents_dtype)
#         y = z * (matvec_fn(z) - diag * z)
#
#         trace_new = _increment(trace, n, jnp.sum(y) + sum(diag))
#         diag_new = _increment(diag, n, y + diag)
#
#         return (trace_new, diag_new, n + 1), ()
#
#     (trace_final, diag_final, _), _ = jax.lax.scan(body_fn, init=init, xs=keys)
#     return trace_final, diag_final
#
#
# def _increment(old, count, incoming):
#     return (old * count + incoming) / (count + 1)
