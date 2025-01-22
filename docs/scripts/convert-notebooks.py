"""
Converts all .ipynb files from the docs/ folder into .mdx files.

This script uses nbconvert to do the processing.
"""

import multiprocessing
import re
from pathlib import Path

import nbformat
from nbconvert.exporters import MarkdownExporter
from nbconvert.preprocessors import Preprocessor

SOURCE_DIR = Path("docs/")
NOTEBOOK_BASE_EDIT_URL = "https://github.com/mlflow/mlflow/edit/master/docs/"
NOTEBOOK_BASE_DOWNLOAD_URL = "https://raw.githubusercontent.com/mlflow/mlflow/master/docs/"


class EscapeBackticksPreprocessor(Preprocessor):
    def preprocess_cell(self, cell, resources, cell_index):
        if cell.cell_type == "code":
            # escape backticks, as code blocks will be rendered
            # inside a custom react component like:
            # <NotebookCellOutput>`{{ content }}`</NotebookCellOutput>
            # and having the backticks causes issues
            cell.source = cell.source.replace("`", r"\`")

            if "outputs" in cell:
                for i, output in enumerate(cell["outputs"]):
                    if "text" in output:
                        output["text"] = output["text"].replace("`", r"\`")
                    elif "data" in output:
                        for key, value in output["data"].items():
                            if isinstance(value, str):
                                output["data"][key] = value.replace("`", r"\`")
        elif cell.cell_type == "raw":
            cell.source = cell.source.replace("<br>", "<br />")

        return cell, resources


exporter = MarkdownExporter(
    preprocessors=[EscapeBackticksPreprocessor],
    template_name="mdx",
    extra_template_basedirs=["./scripts/nbconvert_templates"],
)


def add_frontmatter(
    body: str,
    nb_path: Path,
) -> str:
    frontmatter = {"custom_edit_url": NOTEBOOK_BASE_EDIT_URL + str(nb_path)}
    formatted_frontmatter = "\n".join(f"{key}: {value}" for key, value in frontmatter.items())

    return f"""---
{formatted_frontmatter}
---

{body}"""


def add_download_button(
    body: str,
    nb_path: Path,
) -> str:
    download_url = NOTEBOOK_BASE_DOWNLOAD_URL + str(nb_path)

    # Insert the notebook underneath the first H1 header (assumed to be the title)
    pattern = r"(^#.*$)"
    parts = re.split(pattern, body, maxsplit=1, flags=re.MULTILINE)

    if len(parts) < 3:
        raise Exception(
            f"The notebook at {nb_path} does not have any H1 headers. Please ensure that the title "
            "of the notebook is a H1 header, as the notebook download button will be inserted after it."
        )

    # should not occur due to maxsplit=1
    if len(parts) > 3:
        raise Exception(
            f"Error while parsing notebook at {nb_path}. Please check the format of the notebook."
        )

    # parts[0] = everything before the first header
    # parts[1] = the first header (match group from the regex)
    # parts[2] = the rest of the text
    return f"""{parts[0]}{parts[1]}

<NotebookDownloadButton href="{download_url}">Download this notebook</NotebookDownloadButton>
{parts[2]}"""


# add the imports for our custom cell output components
def add_custom_component_imports(
    body: str,
) -> str:
    return f"""import {{ NotebookCodeCell }} from "@site/src/components/NotebookCodeCell"
import {{ NotebookCellOutput }} from "@site/src/components/NotebookCellOutput"
import {{ NotebookHTMLOutput }} from "@site/src/components/NotebookHTMLOutput"
import {{ NotebookDownloadButton }} from "@site/src/components/NotebookDownloadButton"

{body}
"""


def convert_path(nb_path: Path):
    mdx_path = nb_path.with_suffix(".mdx")
    with open(nb_path) as f:
        nb = nbformat.read(f, as_version=4)

        body, _ = exporter.from_notebook_node(nb)
        body = add_custom_component_imports(body)
        body = add_frontmatter(body, nb_path)
        body = add_download_button(body, nb_path)

        with open(mdx_path, "w") as f:
            f.write(body)

        return mdx_path


def main():
    nb_paths = list(SOURCE_DIR.rglob("*.ipynb"))

    with multiprocessing.Pool() as pool:
        pool.map(convert_path, nb_paths)


if __name__ == "__main__":
    main()
