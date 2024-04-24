import os
import pathlib
from mimetypes import guess_type


# TODO: Create a module to define constants to avoid circular imports
#  and move MLMODEL_FILE_NAME and MLPROJECT_FILE_NAME in the module.
def get_text_extensions():
    from mlflow.models.model import MLMODEL_FILE_NAME
    from mlflow.projects._project_spec import MLPROJECT_FILE_NAME

    return [
        "txt",
        "log",
        "err",
        "cfg",
        "conf",
        "cnf",
        "cf",
        "ini",
        "properties",
        "prop",
        "hocon",
        "toml",
        "yaml",
        "yml",
        "xml",
        "json",
        "js",
        "py",
        "py3",
        "csv",
        "tsv",
        "md",
        "rst",
        MLMODEL_FILE_NAME,
        MLPROJECT_FILE_NAME,
    ]


def _guess_mime_type(file_path):
    filename = pathlib.Path(file_path).name
    extension = os.path.splitext(filename)[-1].replace(".", "")
    # for MLmodel/mlproject with no extensions
    if extension == "":
        extension = filename
    if extension in get_text_extensions():
        return "text/plain"
    mime_type, _ = guess_type(filename)
    if not mime_type:
        # As a fallback, if mime type is not detected, treat it as a binary file
        return "application/octet-stream"
    return mime_type
