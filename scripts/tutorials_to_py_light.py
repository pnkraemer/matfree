"""Transform py:light scripts into Jupyter notebooks."""
import contextlib
import os
from re import sub


def py_to_py_light(*, source, target):
    """Transform a Python script into py:light (as expected by JupyText).

    https://jupytext.readthedocs.io/en/latest/formats-scripts.html#the-light-format
    """
    for dir_src, _children, files_src in os.walk(source):
        # Source-directory to target directory
        dir_trg = replace_prefix(dir_src, source=source, target=target)

        # Create directory if necessary
        mkdir_unless_exists(dir_trg)

        # Transform each python file in directory
        for i, filename_with_py in enumerate(sorted(files_src)):
            if filename_with_py[0] != "_":
                # Split docstring (to be manipulated) from rest of source
                header, rest = split_docstring_from_content(
                    f"{dir_src}/{filename_with_py}"
                )

                # If first line ends on '.' (which it should), remove ".".
                # We implement this by removing all "."'s from the string.
                # More than one period is not supposed
                # to be in the first-line docstring.
                header[0] = header[0].replace(".", "")

                # Turn header into markdown
                markdown_title = [f"## {header[0]}"]
                markdown_rest = [f"# {h}" for h in header[1:]]
                markdown = [*markdown_title, *markdown_rest]

                # Save results
                title = filename(header[0], remove=["(", ")", "[", "]"])
                save_lines_as_file(
                    [*markdown, "\n", *rest], target=f"{dir_trg}/{i+1}_{title}.py"
                )


def split_docstring_from_content(file):
    """Split a file into two parts: docstring and content."""
    with open(file) as original:
        line_first, *lines_other = original.readlines()

        if line_first[:2] == 'r"' or line_first[:2] == 'f"':
            line_first = line_first[1:]

        # Remove leading '"""'
        line_first = line_first[3:]

        # Case 1: Single-line docstring
        if line_first[-4:] == '"""\n':
            return [line_first[:-4]], lines_other

        # Case 2: Multi-line docstring
        idx = lines_other.index('"""\n')
        return [line_first, *lines_other[:idx]], lines_other[idx + 1 :]


def save_lines_as_file(data, /, *, target):
    """Save a list of strings as a file."""
    with open(target, "w") as modified:
        x = data[0]
        for remaining in data[1:]:
            x += f"{remaining}"
        modified.write(x)


def replace_prefix(f: str, *, source: str, target: str) -> str:
    """Replace the source-prefix with a target prefix."""
    f_without_source = string_without_prefix(f, prefix=source)
    return f"{target}{f_without_source}"


def string_without_prefix(f: str, /, *, prefix: str) -> str:
    """Remove a prefix from a string."""
    return f[len(prefix) :]


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


if __name__ == "__main__":
    mkdir_unless_exists("docs/Tutorials/")
    py_to_py_light(source="tutorials/", target="docs/Tutorials/")
