"""Create developer documentation from the README."""

import contextlib
import os
from re import sub


def mkdir_unless_exists(f: str, /) -> None:
    """Create a directory."""
    with contextlib.suppress(FileExistsError):
        os.mkdir(f)


def filename(s, remove):
    """Turn a string into a file-name.

    Removes a list of sub-strings and returns a snake_case version.
    """
    for r in remove:
        s = s.replace(r, "")
    return snake_case(s)


def snake_case(s):
    """Convert a string to snake case.

    Source:
    https://www.w3resource.com/python-exercises/string/python-data-type-string-exercise-97.php
    """
    # Replace hyphens with spaces,
    # then apply regular expression substitutions for title case conversion
    # and add an underscore between words,
    # finally convert the result to lowercase
    return "_".join(
        sub(
            "([A-Z][a-z]+)", r" \1", sub("([A-Z]+)", r" \1", s.replace("-", " "))
        ).split()
    ).lower()


def h2_to_h1(string):
    """Turn a 2-header into a 1-header.

    Remove initial '#'.
    """
    return string[1:]


if __name__ == "__main__":
    # Create dev docs
    source = "README.md"
    directory = "docs/Developer_documentation/"
    mkdir_unless_exists("docs/Developer_documentation/")

    # Read the README
    with open(source) as file:
        full_readme = file.readlines()

    # Find locations of H2's
    is_header = [x[:3] == "## " for x in full_readme]
    indices = [i for i, x in enumerate(is_header) if x]

    # Split the README into "intro" and dev_docs

    # First block is the index
    block = full_readme[: indices[0]]
    title = filename(block[0], remove=["# ", ":"])

    # Save index as file
    with open("docs/index.md", "w") as file:
        file.writelines(block)

    # Create all dev-docs pages

    indices = [*indices, -1]
    for i, (idx_from, idx_to) in enumerate(zip(indices[:-1], indices[1:])):
        # Extract subsection of README
        block = full_readme[idx_from:idx_to]

        # Create a title
        block[0] = h2_to_h1(block[0])
        string = filename(block[0], remove=["# ", ":"])
        title = f"{1+i}_{string}"

        # Save as file
        with open(f"{directory}{title}.md", "w") as file:
            file.writelines(block)
