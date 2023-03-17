import json
from abc import abstractmethod
from typing import Optional, Any, Dict

from mlflow.data.dataset_source import DatasetSource


class Dataset:
    """
    Represents a dataset for use with MLflow Tracking, including the name, digest (hash),
    schema, and profile of the dataset as well as source information (e.g. the S3 bucket or
    managed Delta table from which the dataset was derived).
    """

    def __init__(
        self, source: DatasetSource, name: Optional[str] = None, digest: Optional[str] = None
    ):
        """
        Base constructor for a dataset. All subclasses must call this
        constructor.
        """
        self._name = name
        self._source = source
        # Note: Subclasses should call super() once they've initialized all of
        # the class attributes necessary for digest computation
        self._digest = digest or self._compute_digest()

    @abstractmethod
    def _compute_digest(self) -> str:
        """
        Computes a digest for the dataset. Called if the user doesn't supply
        a digest when constructing the dataset.

        :return: A string digest for the dataset. We recommend a maximum digest length
                 of 10 characters with an ideal length of 8 characters.
        """

    @abstractmethod
    def _to_dict(self, base_dict: Dict[str, str]) -> Dict[str, str]:
        """
        :param base_dict: A string dictionary of base information about the
                          dataset, including: name, digest, source, and source
                          type.
        :return: A string dictionary containing the following fields: name,
                 digest, source, source type, schema (optional), profile
                 (optional).
        """

    def to_json(self) -> str:
        """
        Obtains a JSON string representation of the dataset.

        :return: A JSON string representation of the dataset.
        """
        base_dict = {
            "name": self.name,
            "digest": self.digest,
            "source": self._source.to_json(),
            "source_type": self._source._get_source_type(),
        }
        return json.dumps(self._to_dict(base_dict))

    @property
    def name(self) -> str:
        """
        The name of the dataset.

        :return: The name of the dataset. E.g. "myschema.mycatalog.mytable@2".
        """
        if self._name is not None:
            return self._name
        else:
            # TODO: Compute the name from the digest and source
            return "placeholder_name"

    @property
    def digest(self) -> str:
        """
        The digest (hash, fingerprint) of the dataset.

        :return: The digest (hash, fingerprint) of the dataset,
                 e.g. "498c74967f07246428783efd292cb9cc".
        """
        return self._digest

    @property
    def source(self) -> DatasetSource:
        """
        Dataset source information.

        :return: A DatasetSource instance containing information about the dataset's source,
                 e.g. the S3 bucket or managed Delta table from which the dataset was derived.
        """
        return self._source

    @property
    @abstractmethod
    def profile(self) -> Optional[Any]:
        """
        Optional summary statistics for the dataset.

        :return: Summary statistics for the dataset, e.g. number of rows in a
                 table, mean / median / std of each column, etc. Returns None if no
                 summary statistics are available for the dataset.
        """

    @property
    @abstractmethod
    def schema(self) -> Optional[Any]:
        """
        Optional dataset schema.

        :return: The schema of the dataset, e.g. an instance of `mlflow.types.Schema`. Returns None
                 if no schema is defined for the dataset.
        """
