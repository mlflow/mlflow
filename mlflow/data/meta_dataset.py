import hashlib
import json
from typing import Any, Optional

from mlflow.data.dataset import Dataset
from mlflow.data.dataset_source import DatasetSource
from mlflow.types import Schema


class MetaDataset(Dataset):
    """Dataset that only contains metadata.

    This class is used to represent a dataset that only contains metadata, which is useful when
    users only want to log metadata to MLflow without logging the actual data. For example, users
    build a custom dataset from a text file publicly hosted in the Internet, and they want to log
    the text file's URL to MLflow for future tracking instead of the dataset itself.

    Args:
        source: dataset source of type `DatasetSource`, indicates where the data is from.
        name: name of the dataset. If not specified, a name is automatically generated.
        digest: digest (hash, fingerprint) of the dataset. If not specified, a digest is
            automatically computed.
        schame: schema of the dataset.

    .. code-block:: python
        :caption: Create a MetaDataset

        import mlflow

        mlflow.set_experiment("/test-mlflow-meta-dataset")

        source = mlflow.data.http_dataset_source.HTTPDatasetSource(
            url="https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
        )
        ds = mlflow.data.meta_dataset.MetaDataset(source)

        with mlflow.start_run() as run:
            mlflow.log_input(ds)

    .. code-block:: python
        :caption: Create a MetaDataset with schema

        import mlflow

        mlflow.set_experiment("/test-mlflow-meta-dataset")

        source = mlflow.data.http_dataset_source.HTTPDatasetSource(
            url="https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
        )
        schema = Schema(
            [
                ColSpec(type=mlflow.types.DataType.string, name="text"),
                ColSpec(type=mlflow.types.DataType.integer, name="label"),
            ]
        )
        ds = mlflow.data.meta_dataset.MetaDataset(source, schema=schema)

        with mlflow.start_run() as run:
            mlflow.log_input(ds)
    """

    def __init__(
        self,
        source: DatasetSource,
        name: Optional[str] = None,
        digest: Optional[str] = None,
        schema: Optional[Schema] = None,
    ):
        # Set `self._schema` before calling the superclass constructor because
        # `self._compute_digest` depends on `self._schema`.
        self._schema = schema
        super().__init__(source=source, name=name, digest=digest)

    def _compute_digest(self) -> str:
        """Computes a digest for the dataset.

        The digest computation of `MetaDataset` is based on the dataset's name, source, source type,
        and schema instead of the actual data. Basically we compute the sha256 hash of the config
        dict.
        """
        config = {
            "name": self.name,
            "source": self.source.to_json(),
            "source_type": self.source._get_source_type(),
            "schema": self.schema.to_dict() if self.schema else "",
        }
        return hashlib.sha256(json.dumps(config).encode("utf-8")).hexdigest()[:8]

    @property
    def schema(self) -> Optional[Any]:
        """Returns the schema of the dataset."""
        return self._schema

    def to_dict(self) -> dict[str, str]:
        """Create config dictionary for the MetaDataset.

        Returns a string dictionary containing the following fields: name, digest, source, source
        type, schema, and profile.
        """
        config = super().to_dict()
        if self.schema:
            schema = json.dumps({"mlflow_colspec": self.schema.to_dict()}) if self.schema else None
            config["schema"] = schema
        return config
