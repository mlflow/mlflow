from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib
import logging
import os
import pathlib
import posixpath
import sys
from abc import abstractmethod
from typing import Dict, Any, List, TypeVar, Optional, Union
from urllib.parse import urlparse

from mlflow.artifacts import download_artifacts
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, BAD_REQUEST
from mlflow.store.artifact.artifact_repo import (
    _NUM_DEFAULT_CPUS,
    _NUM_MAX_THREADS_PER_CPU,
    _NUM_MAX_THREADS,
)
from mlflow.utils.file_utils import (
    TempDir,
    get_local_path_or_none,
    local_file_uri_to_path,
    write_pandas_df_as_parquet,
    read_parquet_as_pandas_df,
    download_file_using_http_uri,
)
from mlflow.utils._spark_utils import (
    _get_active_spark_session,
    _create_local_spark_session_for_recipes,
)

_logger = logging.getLogger(__name__)

_DatasetType = TypeVar("_Dataset")

_USER_DEFINED_INGEST_STEP_MODULE = "steps.ingest"


class _Dataset:
    """
    Base class representing an ingestable dataset.
    """

    def __init__(self, dataset_format: str):
        """
        :param dataset_format: The format of the dataset (e.g. 'csv', 'parquet', ...).
        """
        self.dataset_format = dataset_format

    @abstractmethod
    def resolve_to_parquet(self, dst_path: str):
        """
        Fetches the dataset, converts it to parquet, and stores it at the specified `dst_path`.

        :param dst_path: The local filesystem path at which to store the resolved parquet dataset
                         (e.g. `<execution_directory_path>/steps/ingest/outputs/dataset.parquet`).
        """
        pass

    @classmethod
    def from_config(cls, dataset_config: Dict[str, Any], recipe_root: str) -> _DatasetType:
        """
        Constructs a dataset instance from the specified dataset configuration
        and recipe root path.

        :param dataset_config: Dictionary representation of the recipe dataset configuration
                               (i.e. the `data` section of recipe.yaml).
        :param recipe_root: The absolute path of the associated recipe root directory on the
                              local filesystem.
        :return: A `_Dataset` instance representing the configured dataset.
        """
        if not cls.handles_format(dataset_config.get("using")):
            raise MlflowException(
                f"Invalid format {dataset_config.get('using')} for dataset {cls}",
                error_code=INVALID_PARAMETER_VALUE,
            )
        return cls._from_config(dataset_config, recipe_root)

    @classmethod
    @abstractmethod
    def _from_config(cls, dataset_config, recipe_root) -> _DatasetType:
        """
        Constructs a dataset instance from the specified dataset configuration
        and recipe root path.

        :param dataset_config: Dictionary representation of the recipe dataset configuration
                               (i.e. the `data` section of recipe.yaml).
        :param recipe_root: The absolute path of the associated recipe root directory on the
                              local filesystem.
        :return: A `_Dataset` instance representing the configured dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def handles_format(dataset_format: str) -> bool:
        """
        Determines whether or not the dataset class is a compatible representation of the
        specified dataset format.

        :param dataset_format: The format of the dataset (e.g. 'csv', 'parquet', ...).
        :return: `True` if the dataset class is a compatible representation of the specified
                 dataset format, `False` otherwise.
        """
        pass

    @classmethod
    def _get_required_config(cls, dataset_config: Dict[str, Any], key: str) -> Any:
        """
        Obtains the value associated with the specified dataset configuration key, first verifying
        that the key is present in the config and throwing if it is not.

        :param dataset_config: Dictionary representation of the recipe dataset configuration
                               (i.e. the `data` section of recipe.yaml).
        :param key: The key within the dataset configuration for which to fetch the associated
                    value.
        :return: The value associated with the specified configuration key.
        """
        try:
            return dataset_config[key]
        except KeyError:
            raise MlflowException(
                f"The `{key}` configuration key must be specified for dataset with"
                f" using '{dataset_config.get('using')}' format"
            ) from None


class _LocationBasedDataset(_Dataset):
    """
    Base class representing an ingestable dataset with a configurable `location` attribute.
    """

    def __init__(
        self,
        location: Union[str, List[str]],
        dataset_format: str,
        recipe_root: str,
    ):
        """
        :param location: The location of the dataset
                         (one dataset as a string or list of multiple datasets)
                         (e.g. '/tmp/myfile.parquet', './mypath', 's3://mybucket/mypath',
                         or YAML list:
                            location:
                                - http://www.myserver.com/dataset/df1.csv
                                - http://www.myserver.com/dataset/df1.csv
                        )

        :param dataset_format: The format of the dataset (e.g. 'csv', 'parquet', ...).
        :param recipe_root: The absolute path of the associated recipe root directory on the
                              local filesystem.
        """
        super().__init__(dataset_format=dataset_format)
        self.location = (
            _LocationBasedDataset._sanitize_local_dataset_multiple_locations_if_necessary(
                dataset_location=location,
                recipe_root=recipe_root,
            )
        )

    @abstractmethod
    def resolve_to_parquet(self, dst_path: str):
        pass

    @classmethod
    def _from_config(cls, dataset_config: Dict[str, Any], recipe_root: str) -> _DatasetType:
        return cls(
            location=cls._get_required_config(dataset_config=dataset_config, key="location"),
            recipe_root=recipe_root,
            dataset_format=cls._get_required_config(dataset_config=dataset_config, key="using"),
        )

    @staticmethod
    def _sanitize_local_dataset_multiple_locations_if_necessary(
        dataset_location: Union[str, List[str]], recipe_root: str
    ) -> List[str]:
        if isinstance(dataset_location, str):
            return [
                _LocationBasedDataset._sanitize_local_dataset_location_if_necessary(
                    dataset_location, recipe_root
                )
            ]
        elif isinstance(dataset_location, list):
            return [
                _LocationBasedDataset._sanitize_local_dataset_location_if_necessary(
                    locaton, recipe_root
                )
                for locaton in dataset_location
            ]
        else:
            raise MlflowException(f"Unsupported location type: {type(dataset_location)}")

    @staticmethod
    def _sanitize_local_dataset_location_if_necessary(
        dataset_location: str, recipe_root: str
    ) -> str:
        """
        Checks whether or not the specified `dataset_location` is a local filesystem location and,
        if it is, converts it to an absolute path if it is not already absolute.

        :param dataset_location: The dataset location from the recipe dataset configuration.
        :param recipe_root: The absolute path of the recipe root directory on the local
                              filesystem.
        :return: The sanitized dataset location.
        """
        local_dataset_path_or_none = get_local_path_or_none(path_or_uri=dataset_location)
        if local_dataset_path_or_none is None:
            return dataset_location

        # If the local dataset path is a file: URI, convert it to a filesystem path
        local_dataset_path = local_file_uri_to_path(uri=local_dataset_path_or_none)
        local_dataset_path = pathlib.Path(local_dataset_path)
        if local_dataset_path.is_absolute():
            return str(local_dataset_path)
        else:
            # Use pathlib to join the local dataset relative path with the recipe root
            # directory to correctly handle the case where the root path is Windows-formatted
            # and the local dataset relative path is POSIX-formatted
            return str(pathlib.Path(recipe_root) / local_dataset_path)

    @staticmethod
    @abstractmethod
    def handles_format(dataset_format: str) -> bool:
        pass


class _DownloadThenConvertDataset(_LocationBasedDataset):
    """
    Base class representing a location-based ingestible dataset that is resolved in two distinct
    phases: 1. Download the dataset files to the local filesystem. 2. Convert the dataset files to
    parquet format, aggregating them together as a single parquet file.
    `_DownloadThenConvertDataset` implements phase (1) and provides an abstract method
    for phase (2).
    """

    _FILE_DOWNLOAD_CHUNK_SIZE_BYTES = 10**7  # 10MB

    def resolve_to_parquet(self, dst_path: str):
        with TempDir(chdr=True) as tmpdir:
            _logger.debug("Resolving input data from '%s'", self.location)
            local_dataset_path = _DownloadThenConvertDataset._download_dataset(
                dataset_location=self.location,
                dst_path=tmpdir.path(),
            )

            if os.path.isdir(local_dataset_path):
                # NB: Sort the file names alphanumerically to ensure a consistent
                # ordering across invocations
                if self.dataset_format == "custom":
                    dataset_file_paths = sorted(pathlib.Path(local_dataset_path).glob("*"))
                else:
                    dataset_file_paths = sorted(
                        pathlib.Path(local_dataset_path).glob(f"*.{self.dataset_format}")
                    )
                if len(dataset_file_paths) == 0:
                    raise MlflowException(
                        message=(
                            "Did not find any data files with the specified format"
                            f" '{self.dataset_format}' in the resolved data directory with path"
                            f" '{local_dataset_path}'. Directory contents:"
                            f" {os.listdir(local_dataset_path)}."
                        ),
                        error_code=INVALID_PARAMETER_VALUE,
                    )
            else:
                if self.dataset_format != "custom" and not local_dataset_path.endswith(
                    f".{self.dataset_format}"
                ):
                    raise MlflowException(
                        message=(
                            f"Resolved data file with path '{local_dataset_path}' does not have the"
                            f" expected format '{self.dataset_format}'."
                        ),
                        error_code=INVALID_PARAMETER_VALUE,
                    )
                dataset_file_paths = [local_dataset_path]

            _logger.debug("Resolved input data to '%s'", local_dataset_path)
            _logger.debug("Converting dataset to parquet format, if necessary")
            return self._convert_to_parquet(
                dataset_file_paths=dataset_file_paths,
                dst_path=dst_path,
            )

    @staticmethod
    def _download_dataset(dataset_location: List[str], dst_path: str):
        dest_locations = _DownloadThenConvertDataset._download_all_datasets_in_parallel(
            dataset_location, dst_path
        )
        if len(dest_locations) == 1:
            return dest_locations[0]
        else:
            res_path = pathlib.Path(dest_locations[0])
            if res_path.is_dir():
                return str(res_path)
            else:
                return str(res_path.parent)

    @staticmethod
    def _download_all_datasets_in_parallel(dataset_location, dst_path):
        num_cpus = os.cpu_count() or _NUM_DEFAULT_CPUS
        with ThreadPoolExecutor(
            max_workers=min(num_cpus * _NUM_MAX_THREADS_PER_CPU, _NUM_MAX_THREADS)
        ) as executor:
            futures = []
            for location in dataset_location:
                future = executor.submit(
                    _DownloadThenConvertDataset._download_one_dataset,
                    dataset_location=location,
                    dst_path=dst_path,
                )
                futures.append(future)

            dest_locations = []
            failed_downloads = []
            for future in as_completed(futures):
                try:
                    dest_locations.append(future.result())
                except Exception as e:
                    failed_downloads.append(repr(e))
            if len(failed_downloads) > 0:
                raise MlflowException(
                    "During downloading of the datasets a number "
                    + f"of errors have occurred: {failed_downloads}"
                )
            return dest_locations

    @staticmethod
    def _download_one_dataset(dataset_location: str, dst_path: str):
        parsed_location_uri = urlparse(dataset_location)
        if parsed_location_uri.scheme in ["http", "https"]:
            dst_file_name = posixpath.basename(parsed_location_uri.path)
            dst_file_path = os.path.join(dst_path, dst_file_name)
            download_file_using_http_uri(
                http_uri=dataset_location,
                download_path=dst_file_path,
                chunk_size=_DownloadThenConvertDataset._FILE_DOWNLOAD_CHUNK_SIZE_BYTES,
            )
            return dst_file_path
        else:
            return download_artifacts(artifact_uri=dataset_location, dst_path=dst_path)

    @abstractmethod
    def _convert_to_parquet(self, dataset_file_paths: List[str], dst_path: str):
        """
        Converts the specified dataset files to parquet format and aggregates them together,
        writing the consolidated parquet file to the specified destination path.

        :param dataset_file_paths: A list of local filesystem of dataset files to convert to
                                   parquet format.
        :param dst_path: The local filesystem path at which to store the resolved parquet dataset
                         (e.g. `<execution_directory_path>/steps/ingest/outputs/dataset.parquet`).
        """
        pass


class _PandasConvertibleDataset(_DownloadThenConvertDataset):
    """
    Base class representing a location-based ingestable dataset that can be parsed and converted to
    parquet using a series of Pandas DataFrame ``read_*`` and ``concat`` operations.
    """

    def _convert_to_parquet(self, dataset_file_paths: List[str], dst_path: str):
        import pandas as pd

        aggregated_dataframe = None
        for data_file_path in dataset_file_paths:
            _path = pathlib.Path(data_file_path)
            data_file_as_dataframe = self._load_file_as_pandas_dataframe(
                local_data_file_path=data_file_path,
            )
            aggregated_dataframe = (
                pd.concat([aggregated_dataframe, data_file_as_dataframe])
                if aggregated_dataframe is not None
                else data_file_as_dataframe
            )

        write_pandas_df_as_parquet(df=aggregated_dataframe, data_parquet_path=dst_path)

    @abstractmethod
    def _load_file_as_pandas_dataframe(self, local_data_file_path: str):
        """
        Loads the specified file as a Pandas DataFrame.

        :param local_data_file_path: The local filesystem path of the file to load.
        :return: A Pandas DataFrame representation of the specified file.
        """
        pass

    @staticmethod
    @abstractmethod
    def handles_format(dataset_format: str) -> bool:
        pass


class ParquetDataset(_PandasConvertibleDataset):
    """
    Representation of a dataset in parquet format with files having the `.parquet` extension.
    """

    def _load_file_as_pandas_dataframe(self, local_data_file_path: str):
        return read_parquet_as_pandas_df(data_parquet_path=local_data_file_path)

    @staticmethod
    def handles_format(dataset_format: str) -> bool:
        return dataset_format == "parquet"


class CustomDataset(_PandasConvertibleDataset):
    """
    Representation of a location-based dataset with files containing a consistent, custom
    extension (e.g. 'csv', 'csv.gz', 'json', ...), as well as a custom function used to load
    and convert the dataset to parquet format.
    """

    def __init__(
        self,
        location: str,
        dataset_format: str,
        loader_method: str,
        recipe_root: str,
    ):
        """
        :param location: The location of the dataset
                         (e.g. '/tmp/myfile.parquet', './mypath', 's3://mybucket/mypath', ...).
        :param dataset_format: The format of the dataset (e.g. 'csv', 'parquet', ...).
        :param loader_method: The custom loader method used to load and convert the dataset
                              to parquet format, e.g.`load_file_as_dataframe`.
        :param recipe_root: The absolute path of the associated recipe root directory on the
                              local filesystem.
        """
        super().__init__(
            location=location,
            dataset_format=dataset_format,
            recipe_root=recipe_root,
        )
        self.recipe_root = recipe_root
        self.loader_method = loader_method

    def _validate_user_code_output(self, func, *args):
        import pandas as pd

        ingested_df = func(*args)
        if not isinstance(ingested_df, pd.DataFrame):
            raise MlflowException(
                message=(
                    "The `ingested_data` is not a DataFrame, please make sure "
                    f"'{_USER_DEFINED_INGEST_STEP_MODULE}.{self.loader_method}' "
                    "returns a Pandas DataFrame object."
                ),
                error_code=INVALID_PARAMETER_VALUE,
            ) from None
        return ingested_df

    def _load_file_as_pandas_dataframe(self, local_data_file_path: str):
        try:
            sys.path.append(self.recipe_root)
            loader_method = getattr(
                importlib.import_module(_USER_DEFINED_INGEST_STEP_MODULE),
                self.loader_method,
            )
        except Exception as e:
            raise MlflowException(
                message=(
                    "Failed to import custom dataset loader function"
                    f" '{_USER_DEFINED_INGEST_STEP_MODULE}.{self.loader_method}' for"
                    f" ingesting dataset with format '{self.dataset_format}'.",
                ),
                error_code=BAD_REQUEST,
            ) from e

        try:
            return self._validate_user_code_output(
                loader_method, local_data_file_path, self.dataset_format
            )
        except MlflowException as e:
            raise e
        except NotImplementedError:
            raise MlflowException(
                message=(
                    f"Unable to load data file at path '{local_data_file_path}' with format"
                    f" '{self.dataset_format}' using custom loader method"
                    f" '{loader_method.__name__}' because it is not"
                    " supported. Please update the custom loader method to support this"
                    " format."
                ),
                error_code=INVALID_PARAMETER_VALUE,
            ) from None
        except Exception as e:
            raise MlflowException(
                message=(
                    f"Unable to load data file at path '{local_data_file_path}' with format"
                    f" '{self.dataset_format}' using custom loader method"
                    f" '{loader_method.__name__}'."
                ),
                error_code=BAD_REQUEST,
            ) from e

    @classmethod
    def _from_config(cls, dataset_config: Dict[str, Any], recipe_root: str) -> _DatasetType:
        return cls(
            location=cls._get_required_config(dataset_config=dataset_config, key="location"),
            dataset_format=cls._get_required_config(dataset_config=dataset_config, key="using"),
            loader_method=cls._get_required_config(
                dataset_config=dataset_config, key="loader_method"
            ),
            recipe_root=recipe_root,
        )

    @staticmethod
    def handles_format(dataset_format: str) -> bool:
        return dataset_format is not None


class _SparkDatasetMixin:
    """
    Mixin class providing Spark-related utilities for Datasets that use Spark for resolution
    and conversion to parquet format.
    """

    def _get_or_create_spark_session(self):
        """
        Obtains the active Spark session, throwing if a session does not exist.

        :return: The active Spark session.
        """
        try:
            spark_session = _get_active_spark_session()
            if spark_session:
                _logger.debug("Found active spark session")
            else:
                spark_session = _create_local_spark_session_for_recipes()
                _logger.debug("Creating new spark session")
            return spark_session
        except Exception as e:
            raise MlflowException(
                message=(
                    f"Encountered an error while searching for an active Spark session to"
                    f" load the dataset with format '{self.dataset_format}'. Please create a"
                    f" Spark session and try again."
                ),
                error_code=BAD_REQUEST,
            ) from e


class DeltaTableDataset(_SparkDatasetMixin, _LocationBasedDataset):
    """
    Representation of a dataset in delta format with files having the `.delta` extension.
    """

    def __init__(
        self,
        location: str,
        dataset_format: str,
        recipe_root: str,
        version: Optional[int] = None,
        timestamp: Optional[str] = None,
    ):
        """
        :param location: The location of the dataset
                         (e.g. '/tmp/myfile.parquet', './mypath', 's3://mybucket/mypath', ...).
        :param dataset_format: The format of the dataset (e.g. 'csv', 'parquet', ...).
        :param recipe_root: The absolute path of the associated recipe root directory on the
                              local filesystem.
        :param version: The version of the Delta table to read.
        :param timestamp: The timestamp at which to read the Delta table.
        """
        super().__init__(location=location, dataset_format=dataset_format, recipe_root=recipe_root)
        self.version = version
        self.timestamp = timestamp

    def resolve_to_parquet(self, dst_path: str):
        spark_session = self._get_or_create_spark_session()
        spark_read_op = spark_session.read.format("delta")
        if self.version is not None:
            spark_read_op = spark_read_op.option("versionAsOf", self.version)
        if self.timestamp is not None:
            spark_read_op = spark_read_op.option("timestampAsOf", self.timestamp)
        spark_df = spark_read_op.load(self.location)
        pandas_df = spark_df.toPandas()
        write_pandas_df_as_parquet(df=pandas_df, data_parquet_path=dst_path)

    @staticmethod
    def handles_format(dataset_format: str) -> bool:
        return dataset_format == "delta"

    @classmethod
    def _from_config(cls, dataset_config: Dict[str, Any], recipe_root: str) -> _DatasetType:
        return cls(
            location=cls._get_required_config(dataset_config=dataset_config, key="location"),
            recipe_root=recipe_root,
            dataset_format=cls._get_required_config(dataset_config=dataset_config, key="using"),
            version=dataset_config.get("version"),
            timestamp=dataset_config.get("timestamp"),
        )


class SparkSqlDataset(_SparkDatasetMixin, _Dataset):
    """
    Representation of a Spark SQL dataset defined by a Spark SQL query string
    (e.g. `SELECT * FROM my_spark_table`).
    """

    def __init__(self, sql: str, location: str, dataset_format: str):
        """
        :param sql: The Spark SQL query string that defines the dataset
                    (e.g. 'SELECT * FROM my_spark_table').
        :param location: The location of the dataset
                    (e.g. 'catalog.schema.table', 'schema.table', 'table').
        :param dataset_format: The format of the dataset (e.g. 'csv', 'parquet', ...).
        """
        super().__init__(dataset_format=dataset_format)
        self.sql = sql
        self.location = location

    def resolve_to_parquet(self, dst_path: str):
        if self.location is None and self.sql is None:
            raise MlflowException(
                "Either location or sql configuration key must be specified for "
                "dataset with format spark_sql"
            ) from None
        spark_session = self._get_or_create_spark_session()
        spark_df = None
        if self.sql is not None:
            spark_df = spark_session.sql(self.sql)
        elif self.location is not None:
            spark_df = spark_session.table(self.location)
        pandas_df = spark_df.toPandas()
        write_pandas_df_as_parquet(df=pandas_df, data_parquet_path=dst_path)

    @classmethod
    def _from_config(cls, dataset_config: Dict[str, Any], recipe_root: str) -> _DatasetType:
        return cls(
            sql=dataset_config.get("sql"),
            location=dataset_config.get("location"),
            dataset_format=cls._get_required_config(dataset_config=dataset_config, key="using"),
        )

    @staticmethod
    def handles_format(dataset_format: str) -> bool:
        return dataset_format == "spark_sql"
