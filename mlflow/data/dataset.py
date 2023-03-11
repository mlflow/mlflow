import json
from abc import abstractmethod
from typing import Optional, Any, Dict

from mlflow.data.dataset_source import DatasetSource


class Dataset:
    def __init__(
        self, source: DatasetSource, name: Optional[str] = None, digest: Optional[str] = None
    ):
        """
        Base constructor for a dataset. All subclasses must call this
        constructor.
        """
        self._name = name
        self._source = source
        # Note: Users should call super() once they've initialized all of
        # the class attributes necessary for digest computation
        self._digest = digest or self._compute_digest()

    @abstractmethod
    def _compute_digest(self) -> str:
        """
        Computes a digest for the dataset. Called if the user doesn't supply
        a digest when constructing the dataset.
        """

    @abstractmethod
    def _to_dict(self, base_dict: Dict[str, str]) -> Dict[str, str]:
        """
        :param base_dict: A string dictionary of base information about the
                          dataset, including: name, digest, source, and source
                          type.
        :return: A string dictionary containing the following fields: name,
                 digest, source, source type, schema (optional), size
                 (optional).
        """

    def to_json(self) -> str:
        base_dict = {
            "name": self._name,
            "digest": self._digest,
            "source": json.dumps(self._source.to_dict()),
            "source_type": self._source.source_type,
        }
        return json.dumps(self._to_dict(base_dict))

    @property
    def name(self) -> str:
        """
        The name of the dataset. E.g. "myschema.mycatalog.mytable@2"
        """
        if self._name is not None:
            return self._name
        else:
            # TODO: Compute the name from the digest and source
            return "dummy_name"

    @property
    def digest(self) -> str:
        """
        The digest (hash) of the dataset, e.g. "498c74967f07246428783efd292cb9cc"
        """
        return self._digest

    @property
    def source(self) -> DatasetSource:
        """
        Dataset source information
        """
        return self._source

    @property
    @abstractmethod
    def size(self) -> Optional[Any]:
        """
        Dataset size information, e.g. number of rows, size in bytes, etc.
        """

    @property
    @abstractmethod
    def schema(self) -> Optional[Any]:
        """
        Optional. The schema of the dataset, e.g. an instance of `mlflow.types.Schema`
        """
