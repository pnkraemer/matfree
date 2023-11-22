"""PyTree utilities."""

import jax
import jax.flatten_util


def tree_all(tree, /):
    return jax.tree_util.tree_all(tree)


def tree_map(func, tree, *rest):
    return jax.tree_util.tree_map(func, tree, *rest)


def ravel_pytree(tree, /):
    return jax.flatten_util.ravel_pytree(tree)


def tree_leaves(tree, /):
    return jax.tree_util.tree_leaves(tree)


def tree_structure(tree, /):
    return jax.tree_util.tree_structure(tree)


def partial_pytree(func, /):
    return jax.tree_util.Partial(func)
