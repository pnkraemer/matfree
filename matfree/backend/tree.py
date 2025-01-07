"""PyTree utilities."""

import jax
import jax.flatten_util


def tree_all(pytree, /):
    return jax.tree.all(pytree)


def tree_map(func, pytree, *rest):
    return jax.tree.map(func, pytree, *rest)


def ravel_pytree(pytree, /):
    return jax.flatten_util.ravel_pytree(pytree)


def tree_leaves(pytree, /):
    return jax.tree.leaves(pytree)


def tree_structure(pytree, /):
    return jax.tree.structure(pytree)


def partial_pytree(func, /):
    return jax.tree.Partial(func)
