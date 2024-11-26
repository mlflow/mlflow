"""
Converts all .ipynb files from the docs/ folder into .mdx files.

This script uses nbconvert to do the processing.
"""

import multiprocessing
from pathlib import Path

import nbformat
from nbconvert.exporters import MarkdownExporter
from nbconvert.preprocessors import Preprocessor

SOURCE_DIR = Path("docs/")

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

# add the imports for our custom cell output components
def add_custom_component_imports(
  body: str,
) -> str:
  return f"""import {{ NotebookCodeCell }} from "@site/src/components/NotebookCodeCell"
import {{ NotebookCellOutput }} from "@site/src/components/NotebookCellOutput"

{body}
"""

def convert_path(nb_path: Path):
  mdx_path = nb_path.with_suffix(".mdx")
  with open(nb_path) as f:
    nb = nbformat.read(f, as_version=4)

    body, _ = exporter.from_notebook_node(nb)
    body = add_custom_component_imports(body)

    with open(mdx_path, "w") as f:
          f.write(body)
    
    return mdx_path

def main():
  nb_paths = list(SOURCE_DIR.rglob("*.ipynb"))

  with multiprocessing.Pool() as pool:
    pool.map(convert_path, nb_paths)


if __name__ == "__main__":
  main()