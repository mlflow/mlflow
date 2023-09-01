import json
from abc import abstractmethod
from typing import Any, Dict, Optional

from mlflow.data.dataset_source import DatasetSource
from mlflow.entities import Dataset as DatasetEntity
from mlflow.utils.annotations import experimental


@experimental
class Dataset:
    """
    Represents a dataset for use with MLflow Tracking, including the name, digest (hash),
    schema, and profile of the dataset as well as source information (e.g. the S3 bucket or
    managed Delta table from which the dataset was derived). Most datasets expose features
    and targets for training and evaluation as well.
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
        Obtains a JSON string representation of the
        :py:class:`Dataset <mlflow.data.dataset.Dataset>`.

        :return: A JSON string representation of the
                 :py:class:`Dataset <mlflow.data.dataset.Dataset>`.
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
        The name of the dataset, e.g. ``"iris_data"``, ``"myschema.mycatalog.mytable@v1"``, etc.
        """
        if self._name is not None:
            return self._name
        else:
            return "dataset"

    @property
    def digest(self) -> str:
        """
        A unique hash or fingerprint of the dataset, e.g. ``"498c7496"``.
        """
        return self._digest

    @property
    def source(self) -> DatasetSource:
        """
        Information about the dataset's source, represented as an instance of
        :py:class:`DatasetSource <mlflow.data.dataset_source.DatasetSource>`. For example, this
        may be the S3 location or the name of the managed Delta Table from which the dataset
        was derived.
        """
        return self._source

    @property
    @abstractmethod
    def profile(self) -> Optional[Any]:
        """
        Optional summary statistics for the dataset, such as the number of rows in a table, the
        mean / median / std of each table column, etc.
        """

    @property
    @abstractmethod
    def schema(self) -> Optional[Any]:
        """
        Optional dataset schema, such as an instance of :py:class:`mlflow.types.Schema` representing
        the features and targets of the dataset.
        """

    def _to_mlflow_entity(self) -> DatasetEntity:
        """
        :return: A DatasetEntity instance representing the dataset.
        """
        dataset_json = json.loads(self.to_json())
        return DatasetEntity(
            name=dataset_json["name"],
            digest=dataset_json["digest"],
            source_type=dataset_json["source_type"],
            source=dataset_json["source"],
            schema=dataset_json.get("schema"),
            profile=dataset_json.get("profile"),
        )
