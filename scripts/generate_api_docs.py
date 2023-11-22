"""Generate API docs from the source."""
import os
import pathlib


def path_as_module(p):
    p_as_string = str(p)
    return p_as_string.replace(".py", "").replace("/", ".")


def path_as_markdown_file(p):
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
        # Skip "private" modules
        if path.name[0] != "_":
            # Skip required files. (E.g., backend/*)
            if not path.match(path_skip):
                # Construct the contents
                p_as_module = path_as_module(path)
                p_as_markdown = path_as_markdown_file(path)

                # Open the markdown file and write content
                with open(f"{path_target}/{p_as_markdown}.md", "w") as file:
                    content = f"# {p_as_module} \n\n:::{p_as_module}"
                    file.write(content)
