"""
Converts all .ipynb files from the docs/ folder into .mdx files.

This script uses nbconvert to do the processing.
"""

import multiprocessing
import re
from pathlib import Path

import nbformat
import yaml
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
    frontmatter = {
        "custom_edit_url": NOTEBOOK_BASE_EDIT_URL + str(nb_path),
        "slug": nb_path.stem,
    }
    formatted_frontmatter = yaml.dump(frontmatter)

    return f"""---
{formatted_frontmatter}
---

{body}"""


def add_download_button(
    body: str,
    nb_path: Path,
) -> str:
    download_url = NOTEBOOK_BASE_DOWNLOAD_URL + str(nb_path)
    download_button = f'<NotebookDownloadButton href="{download_url}">Download this notebook</NotebookDownloadButton>'

    # Insert the notebook underneath the first H1 header (assumed to be the title)
    pattern = r"(^#\s+.+$)"
    return re.sub(pattern, rf"\1\n\n{download_button}", body, count=1, flags=re.M)


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
    mdx_path = nb_path.with_stem(nb_path.stem + "-ipynb").with_suffix(".mdx")
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
