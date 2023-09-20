import logging
import os
import shutil
import sys
import time
from os.path import join

from mlflow.entities.model_registry import (
    ModelVersion,
    ModelVersionTag,
    RegisteredModel,
    RegisteredModelAlias,
    RegisteredModelTag,
)
from mlflow.entities.model_registry.model_version_stages import (
    ALL_STAGES,
    DEFAULT_STAGES_FOR_GET_LATEST_VERSIONS,
    STAGE_ARCHIVED,
    STAGE_DELETED_INTERNAL,
    STAGE_NONE,
    get_canonical_stage,
)
from mlflow.environment_variables import MLFLOW_REGISTRY_DIR
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
)
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.model_registry import (
    DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH,
    SEARCH_MODEL_VERSION_MAX_RESULTS_THRESHOLD,
    SEARCH_REGISTERED_MODEL_MAX_RESULTS_THRESHOLD,
)
from mlflow.store.model_registry.abstract_store import AbstractStore
from mlflow.utils.file_utils import (
    contains_path_separator,
    exists,
    find,
    is_directory,
    list_all,
    list_subdirs,
    local_file_uri_to_path,
    make_containing_dirs,
    mkdir,
    overwrite_yaml,
    read_file,
    read_yaml,
    write_to,
    write_yaml,
)
from mlflow.utils.search_utils import SearchModelUtils, SearchModelVersionUtils, SearchUtils
from mlflow.utils.string_utils import is_string_type
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.validation import (
    _validate_model_alias_name,
    _validate_model_version,
    _validate_model_version_tag,
    _validate_registered_model_tag,
    _validate_tag_name,
)
from mlflow.utils.validation import (
    _validate_model_name as _original_validate_model_name,
)


def _default_root_dir():
    return MLFLOW_REGISTRY_DIR.get() or os.path.abspath(DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH)


def _validate_model_name(name):
    _original_validate_model_name(name)
    if contains_path_separator(name):
        raise MlflowException(
            f"Invalid name: '{name}'. Registered model name cannot contain path separator",
            INVALID_PARAMETER_VALUE,
        )


class FileStore(AbstractStore):
    MODELS_FOLDER_NAME = "models"
    META_DATA_FILE_NAME = "meta.yaml"
    TAGS_FOLDER_NAME = "tags"
    MODEL_VERSION_TAGS_FOLDER_NAME = "tags"
    CREATE_MODEL_VERSION_RETRIES = 3
    REGISTERED_MODELS_ALIASES_FOLDER_NAME = "aliases"

    def __init__(self, root_directory=None):
        """
        Create a new FileStore with the given root directory.
        """

        super().__init__()
        self.root_directory = local_file_uri_to_path(root_directory or _default_root_dir())
        # Create models directory if needed
        if not exists(self.models_directory):
            mkdir(self.models_directory)

    @property
    def models_directory(self):
        return os.path.join(self.root_directory, FileStore.MODELS_FOLDER_NAME)

    def _check_root_dir(self):
        """
        Run checks before running directory operations.
        """
        if not exists(self.root_directory):
            raise Exception(f"'{self.root_directory}' does not exist.")
        if not is_directory(self.root_directory):
            raise Exception(f"'{self.root_directory}' is not a directory.")

    def _validate_registered_model_does_not_exist(self, name):
        model_path = self._get_registered_model_path(name)
        if exists(model_path):
            raise MlflowException(
                f"Registered Model (name={name}) already exists.",
                RESOURCE_ALREADY_EXISTS,
            )

    def _save_registered_model_as_meta_file(self, registered_model, meta_dir=None, overwrite=True):
        registered_model_dict = dict(registered_model)
        # tags are stored under TAGS_FOLDER_NAME so remove them in meta file.
        del registered_model_dict["tags"]
        del registered_model_dict["latest_versions"]
        meta_dir = meta_dir or self._get_registered_model_path(registered_model.name)
        if overwrite:
            overwrite_yaml(
                meta_dir,
                FileStore.META_DATA_FILE_NAME,
                registered_model_dict,
            )
        else:
            write_yaml(
                meta_dir,
                FileStore.META_DATA_FILE_NAME,
                registered_model_dict,
            )

    def _update_registered_model_last_updated_time(self, name, updated_time):
        registered_model = self.get_registered_model(name)
        registered_model.last_updated_timestamp = updated_time
        self._save_registered_model_as_meta_file(registered_model)

    def create_registered_model(self, name, tags=None, description=None):
        """
        Create a new registered model in backend store.

        :param name: Name of the new model. This is expected to be unique in the backend store.
        :param tags: A list of :py:class:`mlflow.entities.model_registry.RegisteredModelTag`
                     instances associated with this registered model.
        :param description: Description of the model.
        :return: A single object of :py:class:`mlflow.entities.model_registry.RegisteredModel`
                 created in the backend.
        """

        self._check_root_dir()
        _validate_model_name(name)
        self._validate_registered_model_does_not_exist(name)
        for tag in tags or []:
            _validate_registered_model_tag(tag.key, tag.value)
        meta_dir = self._get_registered_model_path(name)
        mkdir(meta_dir)
        creation_time = get_current_time_millis()
        latest_versions = []
        registered_model = RegisteredModel(
            name=name,
            creation_timestamp=creation_time,
            last_updated_timestamp=creation_time,
            description=description,
            latest_versions=latest_versions,
            tags=tags,
        )
        self._save_registered_model_as_meta_file(
            registered_model, meta_dir=meta_dir, overwrite=False
        )
        if tags is not None:
            for tag in tags:
                self.set_registered_model_tag(name, tag)
        return registered_model

    def _get_registered_model_path(self, name):
        self._check_root_dir()
        _validate_model_name(name)
        return join(self.root_directory, FileStore.MODELS_FOLDER_NAME, name)

    def _get_registered_model_from_path(self, model_path):
        meta = FileStore._read_yaml(model_path, FileStore.META_DATA_FILE_NAME)
        meta["tags"] = self.get_all_registered_model_tags_from_path(model_path)
        meta["aliases"] = self.get_all_registered_model_aliases_from_path(model_path)
        registered_model = RegisteredModel.from_dictionary(meta)
        registered_model.latest_versions = self.get_latest_versions(os.path.basename(model_path))
        return registered_model

    def update_registered_model(self, name, description):
        """
        Update description of the registered model.

        :param name: Registered model name.
        :param description: New description.
        :return: A single updated :py:class:`mlflow.entities.model_registry.RegisteredModel` object.
        """
        registered_model = self.get_registered_model(name)
        updated_time = get_current_time_millis()
        registered_model.description = description
        registered_model.last_updated_timestamp = updated_time
        self._save_registered_model_as_meta_file(registered_model)
        return registered_model

    def rename_registered_model(self, name, new_name):
        """
        Rename the registered model.

        :param name: Registered model name.
        :param new_name: New proposed name.
        :return: A single updated :py:class:`mlflow.entities.model_registry.RegisteredModel` object.
        """
        model_path = self._get_registered_model_path(name)
        if not exists(model_path):
            raise MlflowException(
                f"Registered Model with name={name} not found",
                RESOURCE_DOES_NOT_EXIST,
            )
        registered_model = self._get_registered_model_from_path(model_path)

        new_meta_dir = self._get_registered_model_path(new_name)
        if not exists(new_meta_dir):
            mkdir(new_meta_dir)
            updated_time = get_current_time_millis()
            registered_model.name = new_name
            registered_model.last_updated_timestamp = updated_time
            self._save_registered_model_as_meta_file(
                registered_model, meta_dir=new_meta_dir, overwrite=False
            )
            model_versions = self._list_model_versions_under_path(model_path)
            for mv in model_versions:
                mv.name = new_name
                mv.last_updated_timestamp = updated_time
                new_model_version_dir = join(new_meta_dir, f"version-{mv.version}")
                mkdir(new_model_version_dir)
                self._save_model_version_as_meta_file(
                    mv, meta_dir=new_model_version_dir, overwrite=False
                )
                if mv.tags is not None:
                    for tag in mv.tags:
                        self.set_model_version_tag(new_name, mv.version, tag)
            shutil.rmtree(model_path)
        else:
            raise MlflowException(
                f"Registered Model (name={new_name}) already exists.",
                RESOURCE_ALREADY_EXISTS,
            )

        return registered_model

    def delete_registered_model(self, name):
        """
        Delete the registered model.
        Backend raises exception if a registered model with given name does not exist.

        :param name: Registered model name.
        :return: None
        """
        meta_dir = self._get_registered_model_path(name)
        if not exists(meta_dir):
            raise MlflowException(
                f"Registered Model with name={name} not found",
                RESOURCE_DOES_NOT_EXIST,
            )
        shutil.rmtree(meta_dir)

    def list_registered_models(self, max_results, page_token):
        """
        List of all registered models.

        :param max_results: Maximum number of registered models desired.
        :param page_token: Token specifying the next page of results. It should be obtained from
                            a ``list_registered_models`` call.
        :return: A PagedList of :py:class:`mlflow.entities.model_registry.RegisteredModel` objects
                that satisfy the search expressions. The pagination token for the next page can be
                obtained via the ``token`` attribute of the object.
        """
        return self.search_registered_models(max_results=max_results, page_token=page_token)

    def _list_all_registered_models(self):
        registered_model_paths = self._get_all_registered_model_paths()
        registered_models = []
        for path in registered_model_paths:
            registered_models.append(self._get_registered_model_from_path(path))
        return registered_models

    def search_registered_models(
        self, filter_string=None, max_results=None, order_by=None, page_token=None
    ):
        """
        Search for registered models in backend that satisfy the filter criteria.

        :param filter_string: Filter query string, defaults to searching all registered models.
        :param max_results: Maximum number of registered models desired.
        :param order_by: List of column names with ASC|DESC annotation, to be used for ordering
                         matching search results.
        :param page_token: Token specifying the next page of results. It should be obtained from
                            a ``search_registered_models`` call.
        :return: A PagedList of :py:class:`mlflow.entities.model_registry.RegisteredModel` objects
                that satisfy the search expressions. The pagination token for the next page can be
                obtained via the ``token`` attribute of the object.
        """
        if not isinstance(max_results, int) or max_results < 1:
            raise MlflowException(
                "Invalid value for max_results. It must be a positive integer,"
                f" but got {max_results}",
                INVALID_PARAMETER_VALUE,
            )

        if max_results > SEARCH_REGISTERED_MODEL_MAX_RESULTS_THRESHOLD:
            raise MlflowException(
                "Invalid value for request parameter max_results. It must be at most "
                f"{SEARCH_REGISTERED_MODEL_MAX_RESULTS_THRESHOLD}, but got value {max_results}",
                INVALID_PARAMETER_VALUE,
            )

        registered_models = self._list_all_registered_models()
        filtered_rms = SearchModelUtils.filter(registered_models, filter_string)
        sorted_rms = SearchModelUtils.sort(filtered_rms, order_by)
        start_offset = SearchUtils.parse_start_offset_from_page_token(page_token)
        final_offset = start_offset + max_results

        paginated_rms = sorted_rms[start_offset:final_offset]
        next_page_token = None
        if final_offset < len(sorted_rms):
            next_page_token = SearchUtils.create_page_token(final_offset)
        return PagedList(paginated_rms, next_page_token)

    def get_registered_model(self, name):
        """
        Get registered model instance by name.

        :param name: Registered model name.
        :return: A single :py:class:`mlflow.entities.model_registry.RegisteredModel` object.
        """
        _validate_model_name(name)
        model_path = self._get_registered_model_path(name)
        if not exists(model_path):
            raise MlflowException(
                f"Registered Model with name={name} not found",
                RESOURCE_DOES_NOT_EXIST,
            )
        return self._get_registered_model_from_path(model_path)

    def get_latest_versions(self, name, stages=None):
        """
        Latest version models for each requested stage. If no ``stages`` argument is provided,
        returns the latest version for each stage.

        :param name: Registered model name.
        :param stages: List of desired stages. If input list is None, return latest versions for
                       each stage.
        :return: List of :py:class:`mlflow.entities.model_registry.ModelVersion` objects.
        """
        registered_model_path = self._get_registered_model_path(name)
        if not exists(registered_model_path):
            raise MlflowException(
                f"Registered Model with name={name} not found",
                RESOURCE_DOES_NOT_EXIST,
            )
        model_versions = self._list_model_versions_under_path(registered_model_path)
        if stages is None or len(stages) == 0:
            expected_stages = {get_canonical_stage(stage) for stage in ALL_STAGES}
        else:
            expected_stages = {get_canonical_stage(stage) for stage in stages}
        latest_versions = {}
        for mv in model_versions:
            if mv.current_stage in expected_stages:
                if (
                    mv.current_stage not in latest_versions
                    or latest_versions[mv.current_stage].version < mv.version
                ):
                    latest_versions[mv.current_stage] = mv

        return [latest_versions[stage] for stage in expected_stages if stage in latest_versions]

    def _get_registered_model_tag_path(self, name, tag_name):
        _validate_model_name(name)
        _validate_tag_name(tag_name)
        registered_model_path = self._get_registered_model_path(name)
        if not exists(registered_model_path):
            raise MlflowException(
                f"Registered Model with name={name} not found",
                RESOURCE_DOES_NOT_EXIST,
            )
        return os.path.join(registered_model_path, FileStore.TAGS_FOLDER_NAME, tag_name)

    def _get_registered_model_tag_from_file(self, parent_path, tag_name):
        _validate_tag_name(tag_name)
        tag_data = read_file(parent_path, tag_name)
        return RegisteredModelTag(tag_name, tag_data)

    def _get_registered_model_alias_from_file(self, parent_path, alias_name):
        alias_data = read_file(parent_path, alias_name)
        return RegisteredModelAlias(alias_name, alias_data)

    def _get_resource_files(self, root_dir, subfolder_name):
        source_dirs = find(root_dir, subfolder_name, full_path=True)
        if len(source_dirs) == 0:
            return root_dir, []
        file_names = []
        for root, _, files in os.walk(source_dirs[0]):
            for name in files:
                abspath = join(root, name)
                file_names.append(os.path.relpath(abspath, source_dirs[0]))
        if sys.platform == "win32":
            # Turn registered models / model versions relative path into metric name.
            # Registered models and model versions can have '/' in the name.
            # On windows, '/' is interpreted as a separator.
            # When the model / model version is read back the path will use '\' for separator.
            # We need to translate the path into posix path.
            from mlflow.utils.file_utils import relative_path_to_artifact_path

            file_names = [relative_path_to_artifact_path(x) for x in file_names]
        return source_dirs[0], file_names

    def get_all_registered_model_tags_from_path(self, model_path):
        parent_path, tag_files = self._get_resource_files(model_path, FileStore.TAGS_FOLDER_NAME)
        tags = []
        for tag_file in tag_files:
            tags.append(self._get_registered_model_tag_from_file(parent_path, tag_file))
        return tags

    def get_all_registered_model_aliases_from_path(self, model_path):
        parent_path, alias_files = self._get_resource_files(
            model_path, FileStore.REGISTERED_MODELS_ALIASES_FOLDER_NAME
        )
        aliases = []
        for alias_file in alias_files:
            aliases.append(self._get_registered_model_alias_from_file(parent_path, alias_file))
        return aliases

    def _writeable_value(self, tag_value):
        if tag_value is None:
            return ""
        elif is_string_type(tag_value):
            return tag_value
        else:
            return f"{tag_value}"

    def set_registered_model_tag(self, name, tag):
        """
        Set a tag for the registered model.

        :param name: Registered model name.
        :param tag: :py:class:`mlflow.entities.model_registry.RegisteredModelTag` instance to log.
        :return: None
        """
        _validate_registered_model_tag(tag.key, tag.value)
        tag_path = self._get_registered_model_tag_path(name, tag.key)
        make_containing_dirs(tag_path)
        write_to(tag_path, self._writeable_value(tag.value))
        updated_time = get_current_time_millis()
        self._update_registered_model_last_updated_time(name, updated_time)

    def delete_registered_model_tag(self, name, key):
        """
        Delete a tag associated with the registered model.

        :param name: Registered model name.
        :param key: Registered model tag key.
        :return: None
        """
        tag_path = self._get_registered_model_tag_path(name, key)
        if exists(tag_path):
            os.remove(tag_path)
            updated_time = get_current_time_millis()
            self._update_registered_model_last_updated_time(name, updated_time)

    # CRUD API for ModelVersion objects

    def _get_registered_model_version_tag_from_file(self, parent_path, tag_name):
        _validate_tag_name(tag_name)
        tag_data = read_file(parent_path, tag_name)
        return ModelVersionTag(tag_name, tag_data)

    def _get_model_version_tags_from_dir(self, directory):
        parent_path, tag_files = self._get_resource_files(directory, FileStore.TAGS_FOLDER_NAME)
        tags = []
        for tag_file in tag_files:
            tags.append(self._get_registered_model_version_tag_from_file(parent_path, tag_file))
        return tags

    def _get_model_version_dir(self, name, version):
        registered_model_path = self._get_registered_model_path(name)
        if not exists(registered_model_path):
            raise MlflowException(
                f"Registered Model with name={name} not found",
                RESOURCE_DOES_NOT_EXIST,
            )
        return join(registered_model_path, f"version-{version}")

    def _get_model_version_aliases(self, directory):
        aliases = self.get_all_registered_model_aliases_from_path(os.path.dirname(directory))
        version = os.path.basename(directory).replace("version-", "")
        return [alias.alias for alias in aliases if alias.version == version]

    def _get_model_version_from_dir(self, directory):
        meta = FileStore._read_yaml(directory, FileStore.META_DATA_FILE_NAME)
        meta["tags"] = self._get_model_version_tags_from_dir(directory)
        meta["aliases"] = self._get_model_version_aliases(directory)
        return ModelVersion.from_dictionary(meta)

    def _save_model_version_as_meta_file(self, model_version, meta_dir=None, overwrite=True):
        model_version_dict = dict(model_version)
        del model_version_dict["tags"]
        meta_dir = meta_dir or self._get_model_version_dir(
            model_version.name, model_version.version
        )
        if overwrite:
            overwrite_yaml(
                meta_dir,
                FileStore.META_DATA_FILE_NAME,
                model_version_dict,
            )
        else:
            write_yaml(
                meta_dir,
                FileStore.META_DATA_FILE_NAME,
                model_version_dict,
            )

    def create_model_version(
        self,
        name,
        source,
        run_id=None,
        tags=None,
        run_link=None,
        description=None,
        local_model_path=None,
    ):
        """
        Create a new model version from given source and run ID.

        :param name: Registered model name.
        :param source: Source path where the MLflow model is stored.
        :param run_id: Run ID from MLflow tracking server that generated the model.
        :param tags: A list of :py:class:`mlflow.entities.model_registry.ModelVersionTag`
                     instances associated with this model version.
        :param run_link: Link to the run from an MLflow tracking server that generated this model.
        :param description: Description of the version.
        :return: A single object of :py:class:`mlflow.entities.model_registry.ModelVersion`
                 created in the backend.
        """

        def next_version(registered_model_name):
            path = self._get_registered_model_path(registered_model_name)
            model_versions = self._list_model_versions_under_path(path)
            if model_versions:
                return max(mv.version for mv in model_versions) + 1
            else:
                return 1

        _validate_model_name(name)
        for tag in tags or []:
            _validate_model_version_tag(tag.key, tag.value)
        for attempt in range(self.CREATE_MODEL_VERSION_RETRIES):
            try:
                creation_time = get_current_time_millis()
                registered_model = self.get_registered_model(name)
                registered_model.last_updated_timestamp = creation_time
                self._save_registered_model_as_meta_file(registered_model)
                version = next_version(name)
                model_version = ModelVersion(
                    name=name,
                    version=version,
                    creation_timestamp=creation_time,
                    last_updated_timestamp=creation_time,
                    description=description,
                    current_stage=STAGE_NONE,
                    source=source,
                    run_id=run_id,
                    run_link=run_link,
                    tags=tags,
                    aliases=[],
                )
                model_version_dir = self._get_model_version_dir(name, version)
                mkdir(model_version_dir)
                self._save_model_version_as_meta_file(
                    model_version, meta_dir=model_version_dir, overwrite=False
                )
                self._save_registered_model_as_meta_file(registered_model)
                if tags is not None:
                    for tag in tags:
                        self.set_model_version_tag(name, version, tag)
                return model_version
            except Exception as e:
                more_retries = self.CREATE_MODEL_VERSION_RETRIES - attempt - 1
                logging.warning(
                    "Model Version creation error (name=%s) Retrying %s more time%s.",
                    name,
                    str(more_retries),
                    "s" if more_retries > 1 else "",
                )
                if more_retries == 0:
                    raise MlflowException(
                        f"Model Version creation error (name={name}). Error: {e}. Giving up after "
                        f"{self.CREATE_MODEL_VERSION_RETRIES} attempts."
                    )

    def update_model_version(self, name, version, description):
        """
        Update metadata associated with a model version in backend.

        :param name: Registered model name.
        :param version: Registered model version.
        :param description: New model description.
        :return: A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.
        """
        updated_time = get_current_time_millis()
        model_version = self.get_model_version(name=name, version=version)
        model_version.description = description
        model_version.last_updated_timestamp = updated_time
        self._save_model_version_as_meta_file(model_version)
        return model_version

    def transition_model_version_stage(self, name, version, stage, archive_existing_versions):
        """
        Update model version stage.

        :param name: Registered model name.
        :param version: Registered model version.
        :param stage: New desired stage for this model version.
        :param archive_existing_versions: If this flag is set to ``True``, all existing model
            versions in the stage will be automatically moved to the "archived" stage. Only valid
            when ``stage`` is ``"staging"`` or ``"production"`` otherwise an error will be raised.

        :return: A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.
        """
        is_active_stage = get_canonical_stage(stage) in DEFAULT_STAGES_FOR_GET_LATEST_VERSIONS
        if archive_existing_versions and not is_active_stage:
            msg_tpl = (
                "Model version transition cannot archive existing model versions "
                "because '{}' is not an Active stage. Valid stages are {}"
            )
            raise MlflowException(msg_tpl.format(stage, DEFAULT_STAGES_FOR_GET_LATEST_VERSIONS))

        last_updated_time = get_current_time_millis()
        model_versions = []
        if archive_existing_versions:
            registered_model_path = self._get_registered_model_path(name)
            model_versions = self._list_model_versions_under_path(registered_model_path)
            for mv in model_versions:
                if mv.version != version and mv.current_stage == get_canonical_stage(stage):
                    mv.current_stage = STAGE_ARCHIVED
                    mv.last_updated_timestamp = last_updated_time
                    self._save_model_version_as_meta_file(mv)

        model_version = self.get_model_version(name, version)
        model_version.current_stage = get_canonical_stage(stage)
        model_version.last_updated_timestamp = last_updated_time
        self._save_model_version_as_meta_file(model_version)
        self._update_registered_model_last_updated_time(name, last_updated_time)
        return model_version

    def delete_model_version(self, name, version):
        """
        Delete model version in backend.

        :param name: Registered model name.
        :param version: Registered model version.
        :return: None
        """
        model_version = self.get_model_version(name=name, version=version)
        model_version.current_stage = STAGE_DELETED_INTERNAL
        updated_time = get_current_time_millis()
        model_version.last_updated_timestamp = updated_time
        self._save_model_version_as_meta_file(model_version)
        self._update_registered_model_last_updated_time(name, updated_time)
        for alias in model_version.aliases:
            self.delete_registered_model_alias(name, alias)

    def _fetch_model_version_if_exists(self, name, version):
        registered_model_version_dir = self._get_model_version_dir(name, version)
        if not exists(registered_model_version_dir):
            raise MlflowException(
                f"Model Version (name={name}, version={version}) not found",
                RESOURCE_DOES_NOT_EXIST,
            )
        model_version = self._get_model_version_from_dir(registered_model_version_dir)
        if model_version.current_stage == STAGE_DELETED_INTERNAL:
            raise MlflowException(
                f"Model Version (name={name}, version={version}) not found",
                RESOURCE_DOES_NOT_EXIST,
            )
        return model_version

    def get_model_version(self, name, version):
        """
        Get the model version instance by name and version.

        :param name: Registered model name.
        :param version: Registered model version.
        :return: A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.
        """
        _validate_model_name(name)
        _validate_model_version(version)
        return self._fetch_model_version_if_exists(name, version)

    def get_model_version_download_uri(self, name, version):
        """
        Get the download location in Model Registry for this model version.
        NOTE: For first version of Model Registry, since the models are not copied over to another
              location, download URI points to input source path.

        :param name: Registered model name.
        :param version: Registered model version.
        :return: A single URI location that allows reads for downloading.
        """
        model_version = self.get_model_version(name, version)
        return model_version.source

    def _get_all_registered_model_paths(self):
        self._check_root_dir()
        return list_subdirs(join(self.root_directory, FileStore.MODELS_FOLDER_NAME), full_path=True)

    def _list_model_versions_under_path(self, path):
        model_versions = []
        model_version_dirs = list_all(
            path,
            filter_func=lambda x: os.path.isdir(x)
            and os.path.basename(os.path.normpath(x)).startswith("version-"),
            full_path=True,
        )
        for directory in model_version_dirs:
            model_versions.append(self._get_model_version_from_dir(directory))
        return model_versions

    def search_model_versions(
        self, filter_string=None, max_results=None, order_by=None, page_token=None
    ):
        """
        Search for model versions in backend that satisfy the filter criteria.

        :param filter_string: A filter string expression. Currently supports a single filter
                              condition either name of model like ``name = 'model_name'`` or
                              ``run_id = '...'``.
        :param max_results: Maximum number of model versions desired.
        :param order_by: List of column names with ASC|DESC annotation, to be used for ordering
                         matching search results.
        :param page_token: Token specifying the next page of results. It should be obtained from
                            a ``search_model_versions`` call.
        :return: A PagedList of :py:class:`mlflow.entities.model_registry.ModelVersion`
                 objects that satisfy the search expressions. The pagination token for the next
                 page can be obtained via the ``token`` attribute of the object.
        """
        if not isinstance(max_results, int) or max_results < 1:
            raise MlflowException(
                "Invalid value for max_results. It must be a positive integer,"
                f" but got {max_results}",
                INVALID_PARAMETER_VALUE,
            )

        if max_results > SEARCH_MODEL_VERSION_MAX_RESULTS_THRESHOLD:
            raise MlflowException(
                "Invalid value for request parameter max_results. It must be at most "
                f"{SEARCH_MODEL_VERSION_MAX_RESULTS_THRESHOLD}, but got value {max_results}",
                INVALID_PARAMETER_VALUE,
            )

        registered_model_paths = self._get_all_registered_model_paths()
        model_versions = []
        for path in registered_model_paths:
            model_versions.extend(self._list_model_versions_under_path(path))
        filtered_mvs = SearchModelVersionUtils.filter(model_versions, filter_string)

        sorted_mvs = SearchModelVersionUtils.sort(
            filtered_mvs,
            order_by or ["last_updated_timestamp DESC", "name ASC", "version_number DESC"],
        )
        start_offset = SearchUtils.parse_start_offset_from_page_token(page_token)
        final_offset = start_offset + max_results

        paginated_mvs = sorted_mvs[start_offset:final_offset]
        next_page_token = None
        if final_offset < len(sorted_mvs):
            next_page_token = SearchUtils.create_page_token(final_offset)
        return PagedList(paginated_mvs, next_page_token)

    def _get_registered_model_version_tag_path(self, name, version, tag_name):
        _validate_model_name(name)
        _validate_model_version(version)
        _validate_tag_name(tag_name)
        self._fetch_model_version_if_exists(name, version)
        registered_model_version_path = self._get_model_version_dir(name, version)
        return os.path.join(registered_model_version_path, FileStore.TAGS_FOLDER_NAME, tag_name)

    def set_model_version_tag(self, name, version, tag):
        """
        Set a tag for the model version.

        :param name: Registered model name.
        :param version: Registered model version.
        :param tag: :py:class:`mlflow.entities.model_registry.ModelVersionTag` instance to log.
        :return: None
        """
        _validate_model_version_tag(tag.key, tag.value)
        tag_path = self._get_registered_model_version_tag_path(name, version, tag.key)
        make_containing_dirs(tag_path)
        write_to(tag_path, self._writeable_value(tag.value))
        updated_time = get_current_time_millis()
        self._update_registered_model_last_updated_time(name, updated_time)

    def delete_model_version_tag(self, name, version, key):
        """
        Delete a tag associated with the model version.

        :param name: Registered model name.
        :param version: Registered model version.
        :param key: Tag key.
        :return: None
        """
        tag_path = self._get_registered_model_version_tag_path(name, version, key)
        if exists(tag_path):
            os.remove(tag_path)
            updated_time = get_current_time_millis()
            self._update_registered_model_last_updated_time(name, updated_time)

    def _get_registered_model_alias_path(self, name, alias):
        _validate_model_name(name)
        _validate_model_alias_name(alias)
        registered_model_path = self._get_registered_model_path(name)
        if not exists(registered_model_path):
            raise MlflowException(
                f"Registered Model with name={name} not found",
                RESOURCE_DOES_NOT_EXIST,
            )
        return os.path.join(
            registered_model_path, FileStore.REGISTERED_MODELS_ALIASES_FOLDER_NAME, alias
        )

    def set_registered_model_alias(self, name, alias, version):
        """
        Set a registered model alias pointing to a model version.

        :param name: Registered model name.
        :param alias: Name of the alias.
        :param version: Registered model version number.
        :return: None
        """
        alias_path = self._get_registered_model_alias_path(name, alias)
        self._fetch_model_version_if_exists(name, version)
        make_containing_dirs(alias_path)
        write_to(alias_path, self._writeable_value(version))
        updated_time = get_current_time_millis()
        self._update_registered_model_last_updated_time(name, updated_time)

    def delete_registered_model_alias(self, name, alias):
        """
        Delete an alias associated with a registered model.

        :param name: Registered model name.
        :param alias: Name of the alias.
        :return: None
        """
        alias_path = self._get_registered_model_alias_path(name, alias)
        if exists(alias_path):
            os.remove(alias_path)
            updated_time = get_current_time_millis()
            self._update_registered_model_last_updated_time(name, updated_time)

    def get_model_version_by_alias(self, name, alias):
        """
        Get the model version instance by name and alias.

        :param name: Registered model name.
        :param alias: Name of the alias.
        :return: A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.
        """
        alias_path = self._get_registered_model_alias_path(name, alias)
        if exists(alias_path):
            version = read_file(os.path.dirname(alias_path), os.path.basename(alias_path))
            return self.get_model_version(name, version)
        else:
            raise MlflowException(
                f"Registered model alias {alias} not found.", INVALID_PARAMETER_VALUE
            )

    @staticmethod
    def _read_yaml(root, file_name, retries=2):
        """
        Read data from yaml file and return as dictionary, retrying up to
        a specified number of times if the file contents are unexpectedly
        empty due to a concurrent write.

        :param root: Directory name.
        :param file_name: File name. Expects to have '.yaml' extension.
        :param retries: The number of times to retry for unexpected empty content.
        :return: Data in yaml file as dictionary
        """

        def _read_helper(root, file_name, attempts_remaining=2):
            result = read_yaml(root, file_name)
            if result is not None or attempts_remaining == 0:
                return result
            else:
                time.sleep(0.1 * (3 - attempts_remaining))
                return _read_helper(root, file_name, attempts_remaining - 1)

        return _read_helper(root, file_name, attempts_remaining=retries)
