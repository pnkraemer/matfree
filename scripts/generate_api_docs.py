"""Generate API docs from the source."""
import pathlib


def path_as_module(p: pathlib.Path) -> str:
    """Turn a file-path to a python-module-like string."""
    p_as_string = str(p)
    return p_as_string.replace(".py", "").replace("/", ".")


def path_as_markdown_file(p: pathlib.Path) -> str:
    """Turn a file-path to a markdown-file-name."""
    return p.relative_to(path_source).stem


if __name__ == "__main__":
    # Read the arguments
    path_source = pathlib.Path("matfree")
    path_skip = "backend/*"
    path_target = pathlib.Path("docs/API_documentation/")

    # Make the target directory unless it exists
    path_target.mkdir(parents=True, exist_ok=True)

    # Loop recursively through the source
    for path in path_source.rglob("*.py"):
        # Skip "private" modules and
        # skip required files. (E.g., backend/*)
        if path.name[0] != "_" and not path.match(path_skip):
            # Construct the contents
            p_as_module = path_as_module(path)
            p_as_markdown = path_as_markdown_file(path)

            # Open the markdown file and write content
            with open(f"{path_target}/{p_as_markdown}.md", "w") as file:
                content = f"# {p_as_module} \n\n:::{p_as_module}"
                file.write(content)
