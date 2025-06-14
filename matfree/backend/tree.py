"""PyTree utilities."""

import jax
import jax.flatten_util


def tree_all(pytree, /):
    return jax.tree.all(pytree)


def tree_map(func, pytree, *rest):
    return jax.tree.map(func, pytree, *rest)


def ravel_pytree(pytree, /):
    ravelled, unravel = jax.flatten_util.ravel_pytree(pytree)

    # Wrap through a partial_pytree() to make ravel_pytree
    # compatible with jax.eval_shape
    return ravelled, partial_pytree(unravel)


def tree_leaves(pytree, /):
    return jax.tree.leaves(pytree)


def tree_structure(pytree, /):
    return jax.tree.structure(pytree)


def partial_pytree(func, /):
    return jax.tree_util.Partial(func)


def register_dataclass(dcls, /):
    return jax.tree_util.register_dataclass(dcls)
