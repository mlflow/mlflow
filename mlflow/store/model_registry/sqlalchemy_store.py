import logging
import urllib
from typing import Any, Optional, Union

import sqlalchemy
from sqlalchemy.future import select

import mlflow.store.db.utils
from mlflow.entities.model_registry.model_version_stages import (
    ALL_STAGES,
    DEFAULT_STAGES_FOR_GET_LATEST_VERSIONS,
    STAGE_ARCHIVED,
    STAGE_DELETED_INTERNAL,
    get_canonical_stage,
)
from mlflow.entities.model_registry.prompt_version import IS_PROMPT_TAG_KEY
from mlflow.exceptions import MlflowException
from mlflow.prompt.registry_utils import handle_resource_already_exist_error, has_prompt_tag
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    INVALID_STATE,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
)
from mlflow.store.artifact.utils.models import _parse_model_uri
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.model_registry import (
    SEARCH_MODEL_VERSION_MAX_RESULTS_DEFAULT,
    SEARCH_MODEL_VERSION_MAX_RESULTS_THRESHOLD,
    SEARCH_REGISTERED_MODEL_MAX_RESULTS_DEFAULT,
    SEARCH_REGISTERED_MODEL_MAX_RESULTS_THRESHOLD,
)
from mlflow.store.model_registry.abstract_store import AbstractStore
from mlflow.store.model_registry.dbmodels.models import (
    SqlModelVersion,
    SqlModelVersionTag,
    SqlRegisteredModel,
    SqlRegisteredModelAlias,
    SqlRegisteredModelTag,
)
from mlflow.tracking.client import MlflowClient
from mlflow.utils.search_utils import SearchModelUtils, SearchModelVersionUtils, SearchUtils
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.uri import extract_db_type_from_uri
from mlflow.utils.validation import (
    _validate_model_alias_name,
    _validate_model_name,
    _validate_model_renaming,
    _validate_model_version,
    _validate_model_version_tag,
    _validate_registered_model_tag,
    _validate_tag_name,
)

_logger = logging.getLogger(__name__)

# For each database table, fetch its columns and define an appropriate attribute for each column
# on the table's associated object representation (Mapper). This is necessary to ensure that
# columns defined via backreference are available as Mapper instance attributes (e.g.,
# ``SqlRegisteredModel.model_versions``). For more information, see
# https://docs.sqlalchemy.org/en/latest/orm/mapping_api.html#sqlalchemy.orm.configure_mappers
# and https://docs.sqlalchemy.org/en/latest/orm/mapping_api.html#sqlalchemy.orm.mapper.Mapper
sqlalchemy.orm.configure_mappers()


class SqlAlchemyStore(AbstractStore):
    """
    This entity may change or be removed in a future release without warning.
    SQLAlchemy compliant backend store for tracking meta data for MLflow entities. MLflow
    supports the database dialects ``mysql``, ``mssql``, ``sqlite``, and ``postgresql``.
    As specified in the
    `SQLAlchemy docs <https://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls>`_ ,
    the database URI is expected in the format
    ``<dialect>+<driver>://<username>:<password>@<host>:<port>/<database>``. If you do not
    specify a driver, SQLAlchemy uses a dialect's default driver.

    This store interacts with SQL store using SQLAlchemy abstractions defined for MLflow entities.
    :py:class:`mlflow.store.model_registry.models.RegisteredModel` and
    :py:class:`mlflow.store.model_registry.models.ModelVersion`
    """

    CREATE_MODEL_VERSION_RETRIES = 3

    def __init__(self, db_uri):
        """
        Create a database backed store.

        Args:
            db_uri: The SQLAlchemy database URI string to connect to the database. See
                the `SQLAlchemy docs
                <https://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls>`_
                for format specifications. MLflow supports the dialects ``mysql``,
                ``mssql``, ``sqlite``, and ``postgresql``.
            default_artifact_root: Path/URI to location suitable for large data (such as a blob
                store object, DBFS path, or shared NFS file system).
        """
        super().__init__()
        self.db_uri = db_uri
        self.db_type = extract_db_type_from_uri(db_uri)
        self.engine = mlflow.store.db.utils.create_sqlalchemy_engine_with_retry(db_uri)
        if not mlflow.store.db.utils._all_tables_exist(self.engine):
            mlflow.store.db.utils._initialize_tables(self.engine)
        # Verify that all model registry tables exist.
        SqlAlchemyStore._verify_registry_tables_exist(self.engine)
        SessionMaker = sqlalchemy.orm.sessionmaker(bind=self.engine)
        self.ManagedSessionMaker = mlflow.store.db.utils._get_managed_session_maker(
            SessionMaker, self.db_type
        )
        # TODO: verify schema here once we add logic to initialize the registry tables if they
        # don't exist (schema verification will fail in tests otherwise)
        # mlflow.store.db.utils._verify_schema(self.engine)

    def _get_dialect(self):
        return self.engine.dialect.name

    def _dispose_engine(self):
        self.engine.dispose()

    @staticmethod
    def _verify_registry_tables_exist(engine):
        # Verify that all tables have been created.
        inspected_tables = set(sqlalchemy.inspect(engine).get_table_names())
        expected_tables = [
            SqlRegisteredModel.__tablename__,
            SqlModelVersion.__tablename__,
        ]
        if any(table not in inspected_tables for table in expected_tables):
            # TODO: Replace the MlflowException with the following line once it's possible to run
            # the registry against a different DB than the tracking server:
            # mlflow.store.db.utils._initialize_tables(self.engine)
            raise MlflowException("Database migration in unexpected state. Run manual upgrade.")

    @staticmethod
    def _get_eager_registered_model_query_options():
        """
        A list of SQLAlchemy query options that can be used to eagerly
        load the following registered model attributes
        when fetching a registered model: ``registered_model_tags``.
        """
        # Use a subquery load rather than a joined load in order to minimize the memory overhead
        # of the eager loading procedure. For more information about relationship loading
        # techniques, see https://docs.sqlalchemy.org/en/13/orm/
        # loading_relationships.html#relationship-loading-techniques
        return [sqlalchemy.orm.subqueryload(SqlRegisteredModel.registered_model_tags)]

    @staticmethod
    def _get_eager_model_version_query_options():
        """
        A list of SQLAlchemy query options that can be used to eagerly
        load the following model version attributes
        when fetching a model version: ``model_version_tags``.
        """
        # Use a subquery load rather than a joined load in order to minimize the memory overhead
        # of the eager loading procedure. For more information about relationship loading
        # techniques, see https://docs.sqlalchemy.org/en/13/orm/
        # loading_relationships.html#relationship-loading-techniques
        return [sqlalchemy.orm.subqueryload(SqlModelVersion.model_version_tags)]

    def create_registered_model(self, name, tags=None, description=None, deployment_job_id=None):
        """
        Create a new registered model in backend store.

        Args:
            name: Name of the new model. This is expected to be unique in the backend store.
            tags: A list of :py:class:`mlflow.entities.model_registry.RegisteredModelTag`
                instances associated with this registered model.
            description: Description of the version.
            deployment_job_id: Optional deployment job ID.

        Returns:
            A single object of :py:class:`mlflow.entities.model_registry.RegisteredModel`
            created in the backend.
        """
        _validate_model_name(name)
        for tag in tags or []:
            _validate_registered_model_tag(tag.key, tag.value)
        with self.ManagedSessionMaker() as session:
            try:
                creation_time = get_current_time_millis()
                registered_model = SqlRegisteredModel(
                    name=name,
                    creation_time=creation_time,
                    last_updated_time=creation_time,
                    description=description,
                )
                tags_dict = {}
                for tag in tags or []:
                    tags_dict[tag.key] = tag.value
                registered_model.registered_model_tags = [
                    SqlRegisteredModelTag(key=key, value=value) for key, value in tags_dict.items()
                ]
                session.add(registered_model)
                session.flush()
                return registered_model.to_mlflow_entity()
            except sqlalchemy.exc.IntegrityError:
                existing_model = self.get_registered_model(name)
                handle_resource_already_exist_error(
                    name, has_prompt_tag(existing_model._tags), has_prompt_tag(tags)
                )

    @classmethod
    def _get_registered_model(cls, session, name, eager=False):  # noqa: D417
        """
        Args:
            eager: If ``True``, eagerly loads the registered model's tags. If ``False``, these
                attributes are not eagerly loaded and will be loaded when their corresponding object
                properties are accessed from the resulting ``SqlRegisteredModel`` object.
        """
        _validate_model_name(name)
        query_options = cls._get_eager_registered_model_query_options() if eager else []
        rms = (
            session.query(SqlRegisteredModel)
            .options(*query_options)
            .filter(SqlRegisteredModel.name == name)
            .all()
        )

        if len(rms) == 0:
            raise MlflowException(
                f"Registered Model with name={name} not found", RESOURCE_DOES_NOT_EXIST
            )
        if len(rms) > 1:
            raise MlflowException(
                f"Expected only 1 registered model with name={name}. Found {len(rms)}.",
                INVALID_STATE,
            )
        return rms[0]

    def update_registered_model(self, name, description, deployment_job_id=None):
        """
        Update description of the registered model.

        Args:
            name: Registered model name.
            description: New description.
            deployment_job_id: Optional deployment job ID.

        Returns:
            A single updated :py:class:`mlflow.entities.model_registry.RegisteredModel` object.

        """
        with self.ManagedSessionMaker() as session:
            sql_registered_model = self._get_registered_model(session, name)
            updated_time = get_current_time_millis()
            sql_registered_model.description = description
            sql_registered_model.last_updated_time = updated_time
            session.add(sql_registered_model)
            session.flush()
            return sql_registered_model.to_mlflow_entity()

    def rename_registered_model(self, name, new_name):
        """
        Rename the registered model.

        Args:
            name: Registered model name.
            new_name: New proposed name.

        Returns:
            A single updated :py:class:`mlflow.entities.model_registry.RegisteredModel` object.

        """
        _validate_model_renaming(new_name)
        with self.ManagedSessionMaker() as session:
            sql_registered_model = self._get_registered_model(session, name)
            try:
                updated_time = get_current_time_millis()
                sql_registered_model.name = new_name
                for sql_model_version in sql_registered_model.model_versions:
                    sql_model_version.name = new_name
                    sql_model_version.last_updated_time = updated_time
                sql_registered_model.last_updated_time = updated_time
                session.add_all([sql_registered_model] + sql_registered_model.model_versions)
                session.flush()
                return sql_registered_model.to_mlflow_entity()
            except sqlalchemy.exc.IntegrityError as e:
                raise MlflowException(
                    f"Registered Model (name={new_name}) already exists. Error: {e}",
                    RESOURCE_ALREADY_EXISTS,
                )

    def delete_registered_model(self, name):
        """
        Delete the registered model.
        Backend raises exception if a registered model with given name does not exist.

        Args:
            name: Registered model name.

        Returns:
            None
        """
        with self.ManagedSessionMaker() as session:
            sql_registered_model = self._get_registered_model(session, name)
            session.delete(sql_registered_model)

    def _compute_next_token(self, max_results_for_query, current_size, offset, max_results):
        next_token = None
        if max_results_for_query == current_size:
            final_offset = offset + max_results
            next_token = SearchUtils.create_page_token(final_offset)
        return next_token

    def search_registered_models(
        self,
        filter_string=None,
        max_results=SEARCH_REGISTERED_MODEL_MAX_RESULTS_DEFAULT,
        order_by=None,
        page_token=None,
    ):
        """
        Search for registered models in backend that satisfy the filter criteria.

        Args:
            filter_string: Filter query string, defaults to searching all registered models.
            max_results: Maximum number of registered models desired.
            order_by: List of column names with ASC|DESC annotation, to be used for ordering
                matching search results.
            page_token: Token specifying the next page of results. It should be obtained from
                a ``search_registered_models`` call.

        Returns:
            A PagedList of :py:class:`mlflow.entities.model_registry.RegisteredModel` objects
            that satisfy the search expressions. The pagination token for the next page can be
            obtained via the ``token`` attribute of the object.
        """
        if max_results > SEARCH_REGISTERED_MODEL_MAX_RESULTS_THRESHOLD:
            raise MlflowException(
                "Invalid value for request parameter max_results. It must be at most "
                f"{SEARCH_REGISTERED_MODEL_MAX_RESULTS_THRESHOLD}, but got value {max_results}",
                INVALID_PARAMETER_VALUE,
            )

        parsed_filters = SearchModelUtils.parse_search_filter(filter_string)

        filter_query = self._get_search_registered_model_filter_query(
            parsed_filters, self.engine.dialect.name
        )

        parsed_orderby = self._parse_search_registered_models_order_by(order_by)
        offset = SearchUtils.parse_start_offset_from_page_token(page_token)
        # we query for max_results + 1 items to check whether there is another page to return.
        # this remediates having to make another query which returns no items.
        max_results_for_query = max_results + 1

        with self.ManagedSessionMaker() as session:
            query = (
                filter_query.options(*self._get_eager_registered_model_query_options())
                .order_by(*parsed_orderby)
                .limit(max_results_for_query)
            )
            if page_token:
                query = query.offset(offset)
            sql_registered_models = session.execute(query).scalars(SqlRegisteredModel).all()
            next_page_token = self._compute_next_token(
                max_results_for_query, len(sql_registered_models), offset, max_results
            )
            rm_entities = [rm.to_mlflow_entity() for rm in sql_registered_models][:max_results]
            return PagedList(rm_entities, next_page_token)

    @classmethod
    def _get_search_registered_model_filter_query(cls, parsed_filters, dialect):
        attribute_filters = []
        tag_filters = {}
        for f in parsed_filters:
            type_ = f["type"]
            key = f["key"]
            comparator = f["comparator"]
            value = f["value"]
            if type_ == "attribute":
                if key != "name":
                    raise MlflowException(
                        f"Invalid attribute name: {key}", error_code=INVALID_PARAMETER_VALUE
                    )
                if comparator not in ("=", "!=", "LIKE", "ILIKE"):
                    raise MlflowException(
                        f"Invalid comparator for attribute: {comparator}",
                        error_code=INVALID_PARAMETER_VALUE,
                    )
                attr = getattr(SqlRegisteredModel, key)
                attr_filter = SearchUtils.get_sql_comparison_func(comparator, dialect)(attr, value)
                attribute_filters.append(attr_filter)
            elif type_ == "tag":
                if comparator not in ("=", "!=", "LIKE", "ILIKE"):
                    raise MlflowException.invalid_parameter_value(
                        f"Invalid comparator for tag: {comparator}"
                    )
                if key not in tag_filters:
                    key_filter = SearchUtils.get_sql_comparison_func("=", dialect)(
                        SqlRegisteredModelTag.key, key
                    )
                    tag_filters[key] = [key_filter]

                val_filter = SearchUtils.get_sql_comparison_func(comparator, dialect)(
                    SqlRegisteredModelTag.value, value
                )
                tag_filters[key].append(val_filter)
            else:
                raise MlflowException(
                    f"Invalid token type: {type_}", error_code=INVALID_PARAMETER_VALUE
                )

        rm_query = select(SqlRegisteredModel).filter(*attribute_filters)

        if not cls._is_querying_prompt(parsed_filters):
            rm_query = cls._update_query_to_exclude_prompts(
                rm_query, tag_filters, dialect, SqlRegisteredModel, SqlRegisteredModelTag
            )

        if tag_filters:
            sql_tag_filters = (sqlalchemy.and_(*x) for x in tag_filters.values())
            tag_filter_query = (
                select(SqlRegisteredModelTag.name)
                .filter(sqlalchemy.or_(*sql_tag_filters))
                .group_by(SqlRegisteredModelTag.name)
                .having(sqlalchemy.func.count(sqlalchemy.literal(1)) == len(tag_filters))
                .subquery()
            )

            return rm_query.join(
                tag_filter_query, SqlRegisteredModel.name == tag_filter_query.c.name
            )
        else:
            return rm_query

    @classmethod
    def _get_search_model_versions_filter_clauses(cls, parsed_filters, dialect):
        attribute_filters = []
        tag_filters = {}
        for f in parsed_filters:
            type_ = f["type"]
            key = f["key"]
            comparator = f["comparator"]
            value = f["value"]
            if type_ == "attribute":
                if key not in SearchModelVersionUtils.VALID_SEARCH_ATTRIBUTE_KEYS:
                    raise MlflowException(
                        f"Invalid attribute name: {key}", error_code=INVALID_PARAMETER_VALUE
                    )
                if key in SearchModelVersionUtils.NUMERIC_ATTRIBUTES:
                    if (
                        comparator
                        not in SearchModelVersionUtils.VALID_NUMERIC_ATTRIBUTE_COMPARATORS
                    ):
                        raise MlflowException(
                            f"Invalid comparator for attribute {key}: {comparator}",
                            error_code=INVALID_PARAMETER_VALUE,
                        )
                elif (
                    comparator not in SearchModelVersionUtils.VALID_STRING_ATTRIBUTE_COMPARATORS
                    or (comparator == "IN" and key != "run_id")
                ):
                    raise MlflowException(
                        f"Invalid comparator for attribute: {comparator}",
                        error_code=INVALID_PARAMETER_VALUE,
                    )
                if key == "source_path":
                    key_name = "source"
                elif key == "version_number":
                    key_name = "version"
                else:
                    key_name = key
                attr = getattr(SqlModelVersion, key_name)
                if comparator == "IN":
                    # Note: Here the run_id values in databases contain only lower case letters,
                    # so we already filter out comparison values containing upper case letters
                    # in `SearchModelUtils._get_value`. This addresses MySQL IN clause case
                    # in-sensitive issue.
                    val_filter = attr.in_(value)
                else:
                    val_filter = SearchUtils.get_sql_comparison_func(comparator, dialect)(
                        attr, value
                    )
                attribute_filters.append(val_filter)
            elif type_ == "tag":
                if comparator not in ("=", "!=", "LIKE", "ILIKE"):
                    raise MlflowException.invalid_parameter_value(
                        f"Invalid comparator for tag: {comparator}",
                    )
                if key not in tag_filters:
                    key_filter = SearchUtils.get_sql_comparison_func("=", dialect)(
                        SqlModelVersionTag.key, key
                    )
                    tag_filters[key] = [key_filter]

                val_filter = SearchUtils.get_sql_comparison_func(comparator, dialect)(
                    SqlModelVersionTag.value, value
                )
                tag_filters[key].append(val_filter)
            else:
                raise MlflowException(
                    f"Invalid token type: {type_}", error_code=INVALID_PARAMETER_VALUE
                )

        mv_query = select(SqlModelVersion).filter(*attribute_filters)

        if not cls._is_querying_prompt(parsed_filters):
            mv_query = cls._update_query_to_exclude_prompts(
                mv_query, tag_filters, dialect, SqlModelVersion, SqlModelVersionTag
            )

        if tag_filters:
            sql_tag_filters = (sqlalchemy.and_(*x) for x in tag_filters.values())
            tag_filter_query = (
                select(SqlModelVersionTag.name, SqlModelVersionTag.version)
                .filter(sqlalchemy.or_(*sql_tag_filters))
                .group_by(SqlModelVersionTag.name, SqlModelVersionTag.version)
                .having(sqlalchemy.func.count(sqlalchemy.literal(1)) == len(tag_filters))
                .subquery()
            )
            return mv_query.join(
                tag_filter_query,
                sqlalchemy.and_(
                    SqlModelVersion.name == tag_filter_query.c.name,
                    SqlModelVersion.version == tag_filter_query.c.version,
                ),
            )
        else:
            return mv_query

    @classmethod
    def _update_query_to_exclude_prompts(
        cls,
        query: Any,
        tag_filters: dict[str, list[Any]],
        dialect: str,
        main_db_model: Union[SqlModelVersion, SqlRegisteredModel],
        tag_db_model: Union[SqlModelVersionTag, SqlRegisteredModelTag],
    ):
        """
        Update query to exclude all prompt rows and return only normal model or model versions.

        Prompts and normal models are distinguished by the `mlflow.prompt.is_prompt` tag.
        The search API should only return normal models by default. However, simply filtering
        rows using the tag like this does not work because models do not have the prompt tag.

            tags.`mlflow.prompt.is_prompt` != 'true'
            tags.`mlflow.prompt.is_prompt` = 'false'

        To workaround this, we need to use a subquery to get all prompt rows and then use an
        anti-join for excluding prompts.
        """
        # If the tag filter contains the prompt tag, remove it
        tag_filters.pop(IS_PROMPT_TAG_KEY, [])

        # Filter to get all prompt rows
        equal = SearchUtils.get_sql_comparison_func("=", dialect)
        prompts_subquery = (
            select(tag_db_model.name)
            .filter(
                equal(tag_db_model.key, IS_PROMPT_TAG_KEY),
                equal(tag_db_model.value, "true"),
            )
            .group_by(tag_db_model.name)
            .subquery()
        )
        return query.join(
            prompts_subquery, main_db_model.name == prompts_subquery.c.name, isouter=True
        ).filter(prompts_subquery.c.name.is_(None))

    @classmethod
    def _is_querying_prompt(cls, parsed_filters: list[dict[str, Any]]) -> bool:
        for f in parsed_filters:
            if f["type"] != "tag" or f["key"] != IS_PROMPT_TAG_KEY:
                continue

            return (f["comparator"] == "=" and f["value"].lower() == "true") or (
                f["comparator"] == "!=" and f["value"].lower() == "false"
            )

        # Query should return only normal models by default
        return False

    @classmethod
    def _parse_search_registered_models_order_by(cls, order_by_list):
        """Sorts a set of registered models based on their natural ordering and an overriding set
        of order_bys. Registered models are naturally ordered first by name ascending.
        """
        clauses = []
        observed_order_by_clauses = set()
        if order_by_list:
            for order_by_clause in order_by_list:
                (
                    attribute_token,
                    ascending,
                ) = SearchUtils.parse_order_by_for_search_registered_models(order_by_clause)
                if attribute_token == SqlRegisteredModel.name.key:
                    field = SqlRegisteredModel.name
                elif attribute_token in SearchUtils.VALID_TIMESTAMP_ORDER_BY_KEYS:
                    field = SqlRegisteredModel.last_updated_time
                else:
                    raise MlflowException(
                        f"Invalid order by key '{attribute_token}' specified."
                        + "Valid keys are "
                        + f"'{SearchUtils.RECOMMENDED_ORDER_BY_KEYS_REGISTERED_MODELS}'",
                        error_code=INVALID_PARAMETER_VALUE,
                    )
                if field.key in observed_order_by_clauses:
                    raise MlflowException(f"`order_by` contains duplicate fields: {order_by_list}")
                observed_order_by_clauses.add(field.key)
                if ascending:
                    clauses.append(field.asc())
                else:
                    clauses.append(field.desc())

        if SqlRegisteredModel.name.key not in observed_order_by_clauses:
            clauses.append(SqlRegisteredModel.name.asc())
        return clauses

    def get_registered_model(self, name):
        """
        Get registered model instance by name.

        Args:
            name: Registered model name.

        Returns:
            A single :py:class:`mlflow.entities.model_registry.RegisteredModel` object.
        """
        with self.ManagedSessionMaker() as session:
            return self._get_registered_model(session, name, eager=True).to_mlflow_entity()

    def get_latest_versions(self, name, stages=None):
        """
        Latest version models for each requested stage. If no ``stages`` argument is provided,
        returns the latest version for each stage.

        Args:
            name: Registered model name.
            stages: List of desired stages. If input list is None, return latest versions for
                each stage.

        Returns:
            List of :py:class:`mlflow.entities.model_registry.ModelVersion` objects.

        """
        with self.ManagedSessionMaker() as session:
            sql_registered_model = self._get_registered_model(session, name)
            # Convert to RegisteredModel entity first and then extract latest_versions
            latest_versions = sql_registered_model.to_mlflow_entity().latest_versions
            if stages is None or len(stages) == 0:
                expected_stages = {get_canonical_stage(stage) for stage in ALL_STAGES}
            else:
                expected_stages = {get_canonical_stage(stage) for stage in stages}
            mvs = [mv for mv in latest_versions if mv.current_stage in expected_stages]

            # Populate aliases for each model version
            for mv in mvs:
                model_aliases = sql_registered_model.registered_model_aliases
                mv.aliases = [alias.alias for alias in model_aliases if alias.version == mv.version]

            return mvs

    @classmethod
    def _get_registered_model_tag(cls, session, name, key):
        tags = (
            session.query(SqlRegisteredModelTag)
            .filter(SqlRegisteredModelTag.name == name, SqlRegisteredModelTag.key == key)
            .all()
        )
        if len(tags) == 0:
            return None
        if len(tags) > 1:
            raise MlflowException(
                f"Expected only 1 registered model tag with name={name}, key={key}. "
                f"Found {len(tags)}.",
                INVALID_STATE,
            )
        return tags[0]

    def set_registered_model_tag(self, name, tag):
        """
        Set a tag for the registered model.

        Args:
            name: Registered model name.
            tag: :py:class:`mlflow.entities.model_registry.RegisteredModelTag` instance to log.

        Returns:
            None
        """
        _validate_model_name(name)
        _validate_registered_model_tag(tag.key, tag.value)
        with self.ManagedSessionMaker() as session:
            # check if registered model exists
            self._get_registered_model(session, name)
            session.merge(SqlRegisteredModelTag(name=name, key=tag.key, value=tag.value))

    def delete_registered_model_tag(self, name, key):
        """
        Delete a tag associated with the registered model.

        Args:
            name: Registered model name.
            key: Registered model tag key.

        Returns:
            None
        """
        _validate_model_name(name)
        _validate_tag_name(key)
        with self.ManagedSessionMaker() as session:
            # check if registered model exists
            self._get_registered_model(session, name)
            existing_tag = self._get_registered_model_tag(session, name, key)
            if existing_tag is not None:
                session.delete(existing_tag)

    # CRUD API for ModelVersion objects

    def create_model_version(
        self,
        name,
        source,
        run_id=None,
        tags=None,
        run_link=None,
        description=None,
        local_model_path=None,
        model_id: Optional[str] = None,
    ):
        """
        Create a new model version from given source and run ID.

        Args:
            name: Registered model name.
            source: URI indicating the location of the model artifacts.
            run_id: Run ID from MLflow tracking server that generated the model.
            tags: A list of :py:class:`mlflow.entities.model_registry.ModelVersionTag`
                instances associated with this model version.
            run_link: Link to the run from an MLflow tracking server that generated this model.
            description: Description of the version.
            local_model_path: Unused.
            model_id: The ID of the model (from an Experiment) that is being promoted to a
                registered model version, if applicable.

        Returns:
            A single object of :py:class:`mlflow.entities.model_registry.ModelVersion`
            created in the backend.

        """

        def next_version(sql_registered_model):
            if sql_registered_model.model_versions:
                return max([mv.version for mv in sql_registered_model.model_versions]) + 1
            else:
                return 1

        _validate_model_name(name)
        for tag in tags or []:
            _validate_model_version_tag(tag.key, tag.value)
        storage_location = source
        if urllib.parse.urlparse(source).scheme == "models":
            parsed_model_uri = _parse_model_uri(source)
            try:
                if parsed_model_uri.model_id is not None:
                    # TODO: Propagate tracking URI to file sqlalchemy directly, rather than relying
                    # on global URI (individual MlflowClient instances may have different tracking
                    # URIs)
                    model = MlflowClient().get_logged_model(parsed_model_uri.model_id)
                    storage_location = model.artifact_location
                    run_id = run_id or model.source_run_id
                else:
                    storage_location = self.get_model_version_download_uri(
                        parsed_model_uri.name, parsed_model_uri.version
                    )
            except Exception as e:
                raise MlflowException(
                    f"Unable to fetch model from model URI source artifact location '{source}'."
                    f"Error: {e}"
                ) from e
        with self.ManagedSessionMaker() as session:
            creation_time = get_current_time_millis()
            for attempt in range(self.CREATE_MODEL_VERSION_RETRIES):
                try:
                    sql_registered_model = self._get_registered_model(session, name)
                    sql_registered_model.last_updated_time = creation_time
                    version = next_version(sql_registered_model)
                    model_version = SqlModelVersion(
                        name=name,
                        version=version,
                        creation_time=creation_time,
                        last_updated_time=creation_time,
                        source=source,
                        storage_location=storage_location,
                        run_id=run_id,
                        run_link=run_link,
                        description=description,
                    )
                    tags_dict = {}
                    for tag in tags or []:
                        tags_dict[tag.key] = tag.value
                    model_version.model_version_tags = [
                        SqlModelVersionTag(key=key, value=value) for key, value in tags_dict.items()
                    ]
                    session.add_all([sql_registered_model, model_version])
                    session.flush()
                    return self._populate_model_version_aliases(
                        session, name, model_version.to_mlflow_entity()
                    )
                except sqlalchemy.exc.IntegrityError:
                    more_retries = self.CREATE_MODEL_VERSION_RETRIES - attempt - 1
                    _logger.info(
                        "Model Version creation error (name=%s) Retrying %s more time%s.",
                        name,
                        str(more_retries),
                        "s" if more_retries > 1 else "",
                    )
        raise MlflowException(
            f"Model Version creation error (name={name}). Giving up after "
            f"{self.CREATE_MODEL_VERSION_RETRIES} attempts."
        )

    @classmethod
    def _populate_model_version_aliases(cls, session, name, version):
        model_aliases = cls._get_registered_model(session, name).registered_model_aliases
        version.aliases = [
            alias.alias for alias in model_aliases if alias.version == version.version
        ]
        return version

    @classmethod
    def _get_model_version_from_db(cls, session, name, version, conditions, query_options=None):
        if query_options is None:
            query_options = []
        versions = session.query(SqlModelVersion).options(*query_options).filter(*conditions).all()

        if len(versions) == 0:
            raise MlflowException(
                f"Model Version (name={name}, version={version}) not found",
                RESOURCE_DOES_NOT_EXIST,
            )
        if len(versions) > 1:
            raise MlflowException(
                f"Expected only 1 model version with (name={name}, version={version}). "
                f"Found {len(versions)}.",
                INVALID_STATE,
            )
        return versions[0]

    @classmethod
    def _get_sql_model_version(cls, session, name, version, eager=False):  # noqa: D417
        """
        Args:
            eager: If ``True``, eagerly loads the model version's tags.
                If ``False``, these attributes are not eagerly loaded and
                will be loaded when their corresponding object properties
                are accessed from the resulting ``SqlModelVersion`` object.
        """
        _validate_model_name(name)
        _validate_model_version(version)
        query_options = cls._get_eager_model_version_query_options() if eager else []
        conditions = [
            SqlModelVersion.name == name,
            SqlModelVersion.version == version,
            SqlModelVersion.current_stage != STAGE_DELETED_INTERNAL,
        ]
        return cls._get_model_version_from_db(session, name, version, conditions, query_options)

    def _get_sql_model_version_including_deleted(self, name, version):
        """
        Private method to retrieve model versions including those that are internally deleted.
        Used in tests to verify redaction behavior on deletion.

        Args:
            name: Registered model name.
            version: Registered model version.

        Returns:
            A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.
        """
        with self.ManagedSessionMaker() as session:
            conditions = [
                SqlModelVersion.name == name,
                SqlModelVersion.version == version,
            ]
            sql_model_version = self._get_model_version_from_db(session, name, version, conditions)
            return self._populate_model_version_aliases(
                session, name, sql_model_version.to_mlflow_entity()
            )

    def update_model_version(self, name, version, description=None):
        """
        Update metadata associated with a model version in backend.

        Args:
            name: Registered model name.
            version: Registered model version.
            description: New model description.

        Returns:
            A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.

        """
        with self.ManagedSessionMaker() as session:
            updated_time = get_current_time_millis()
            sql_model_version = self._get_sql_model_version(session, name=name, version=version)
            sql_model_version.description = description
            sql_model_version.last_updated_time = updated_time
            session.add(sql_model_version)
            return self._populate_model_version_aliases(
                session, name, sql_model_version.to_mlflow_entity()
            )

    def transition_model_version_stage(self, name, version, stage, archive_existing_versions):
        """
        Update model version stage.

        Args:
            name: Registered model name.
            version: Registered model version.
            stage: New desired stage for this model version.
            archive_existing_versions: If this flag is set to ``True``, all existing model
                versions in the stage will be automatically moved to the "archived" stage. Only
                valid when ``stage`` is ``"staging"`` or ``"production"`` otherwise an error will
                be raised.

        Returns:
            A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.

        """
        is_active_stage = get_canonical_stage(stage) in DEFAULT_STAGES_FOR_GET_LATEST_VERSIONS
        if archive_existing_versions and not is_active_stage:
            msg_tpl = (
                "Model version transition cannot archive existing model versions "
                "because '{}' is not an Active stage. Valid stages are {}"
            )
            raise MlflowException(msg_tpl.format(stage, DEFAULT_STAGES_FOR_GET_LATEST_VERSIONS))

        with self.ManagedSessionMaker() as session:
            last_updated_time = get_current_time_millis()

            model_versions = []
            if archive_existing_versions:
                conditions = [
                    SqlModelVersion.name == name,
                    SqlModelVersion.version != version,
                    SqlModelVersion.current_stage == get_canonical_stage(stage),
                ]
                model_versions = session.query(SqlModelVersion).filter(*conditions).all()
                for mv in model_versions:
                    mv.current_stage = STAGE_ARCHIVED
                    mv.last_updated_time = last_updated_time

            sql_model_version = self._get_sql_model_version(
                session=session, name=name, version=version
            )
            sql_model_version.current_stage = get_canonical_stage(stage)
            sql_model_version.last_updated_time = last_updated_time
            sql_registered_model = sql_model_version.registered_model
            sql_registered_model.last_updated_time = last_updated_time
            session.add_all([*model_versions, sql_model_version, sql_registered_model])
            return self._populate_model_version_aliases(
                session, name, sql_model_version.to_mlflow_entity()
            )

    def delete_model_version(self, name, version):
        """
        Delete model version in backend.

        Args:
            name: Registered model name.
            version: Registered model version.

        Returns:
            None
        """
        # currently delete model version still keeps the tags associated with the version
        with self.ManagedSessionMaker() as session:
            updated_time = get_current_time_millis()
            sql_model_version = self._get_sql_model_version(session, name, version)
            sql_registered_model = sql_model_version.registered_model
            sql_registered_model.last_updated_time = updated_time
            aliases = sql_registered_model.registered_model_aliases
            for alias in aliases:
                if alias.version == version:
                    session.delete(alias)
            sql_model_version.current_stage = STAGE_DELETED_INTERNAL
            sql_model_version.last_updated_time = updated_time
            sql_model_version.description = None
            sql_model_version.user_id = None
            sql_model_version.source = "REDACTED-SOURCE-PATH"
            sql_model_version.run_id = "REDACTED-RUN-ID"
            sql_model_version.run_link = "REDACTED-RUN-LINK"
            sql_model_version.status_message = None
            session.add_all([sql_registered_model, sql_model_version])

    def get_model_version(self, name, version):
        """
        Get the model version instance by name and version.

        Args:
            name: Registered model name.
            version: Registered model version.

        Returns:
            A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.
        """
        with self.ManagedSessionMaker() as session:
            sql_model_version = self._get_sql_model_version(session, name, version, eager=True)
            return self._populate_model_version_aliases(
                session, name, sql_model_version.to_mlflow_entity()
            )

    def get_model_version_download_uri(self, name, version):
        """
        Get the download location in Model Registry for this model version.
        NOTE: For first version of Model Registry, since the models are not copied over to another
              location, download URI points to input source path.

        Args:
            name: Registered model name.
            version: Registered model version.

        Returns:
            A single URI location that allows reads for downloading.
        """
        with self.ManagedSessionMaker() as session:
            sql_model_version = self._get_sql_model_version(session, name, version)
            return sql_model_version.storage_location or sql_model_version.source

    def search_model_versions(
        self,
        filter_string=None,
        max_results=SEARCH_MODEL_VERSION_MAX_RESULTS_DEFAULT,
        order_by=None,
        page_token=None,
    ):
        """
        Search for model versions in backend that satisfy the filter criteria.

        Args:
            filter_string: A filter string expression. Currently supports a single filter
                condition either name of model like ``name = 'model_name'`` or
                ``run_id = '...'``.
            max_results: Maximum number of model versions desired.
            order_by: List of column names with ASC|DESC annotation, to be used for ordering
                matching search results.
            page_token: Token specifying the next page of results. It should be obtained from
                a ``search_model_versions`` call.

        Returns:
            A PagedList of :py:class:`mlflow.entities.model_registry.ModelVersion`
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

        parsed_filters = SearchModelVersionUtils.parse_search_filter(filter_string)

        filter_query = self._get_search_model_versions_filter_clauses(
            parsed_filters, self.engine.dialect.name
        )

        parsed_orderby = self._parse_search_model_versions_order_by(
            order_by or ["last_updated_timestamp DESC", "name ASC", "version_number DESC"]
        )
        offset = SearchUtils.parse_start_offset_from_page_token(page_token)
        # we query for max_results + 1 items to check whether there is another page to return.
        # this remediates having to make another query which returns no items.
        max_results_for_query = max_results + 1

        with self.ManagedSessionMaker() as session:
            query = (
                filter_query.options(*self._get_eager_model_version_query_options())
                .filter(SqlModelVersion.current_stage != STAGE_DELETED_INTERNAL)
                .order_by(*parsed_orderby)
                .limit(max_results_for_query)
            )
            if page_token:
                query = query.offset(offset)
            sql_model_versions = session.execute(query).scalars(SqlModelVersion).all()
            next_page_token = self._compute_next_token(
                max_results_for_query, len(sql_model_versions), offset, max_results
            )
            model_versions = [mv.to_mlflow_entity() for mv in sql_model_versions][:max_results]
            return PagedList(model_versions, next_page_token)

    @classmethod
    def _parse_search_model_versions_order_by(cls, order_by_list):
        """Sorts a set of model versions based on their natural ordering and an overriding set
        of order_bys. Model versions are naturally ordered first by name ascending, then by
        version ascending.
        """
        clauses = []
        observed_order_by_clauses = set()
        if order_by_list:
            for order_by_clause in order_by_list:
                (
                    _,
                    key,
                    ascending,
                ) = SearchModelVersionUtils.parse_order_by_for_search_model_versions(
                    order_by_clause
                )
                if key not in SearchModelVersionUtils.VALID_ORDER_BY_ATTRIBUTE_KEYS:
                    raise MlflowException(
                        f"Invalid order by key '{key}' specified. "
                        "Valid keys are "
                        f"{SearchModelVersionUtils.VALID_ORDER_BY_ATTRIBUTE_KEYS}",
                        error_code=INVALID_PARAMETER_VALUE,
                    )
                else:
                    if key == "version_number":
                        field = SqlModelVersion.version
                    elif key == "creation_timestamp":
                        field = SqlModelVersion.creation_time
                    elif key == "last_updated_timestamp":
                        field = SqlModelVersion.last_updated_time
                    else:
                        field = getattr(SqlModelVersion, key)
                if field.key in observed_order_by_clauses:
                    raise MlflowException(f"`order_by` contains duplicate fields: {order_by_list}")
                observed_order_by_clauses.add(field.key)
                if ascending:
                    clauses.append(field.asc())
                else:
                    clauses.append(field.desc())

        if SqlModelVersion.name.key not in observed_order_by_clauses:
            clauses.append(SqlModelVersion.name.asc())
        if SqlModelVersion.version.key not in observed_order_by_clauses:
            clauses.append(SqlModelVersion.version.desc())
        return clauses

    @classmethod
    def _get_model_version_tag(cls, session, name, version, key):
        tags = (
            session.query(SqlModelVersionTag)
            .filter(
                SqlModelVersionTag.name == name,
                SqlModelVersionTag.version == version,
                SqlModelVersionTag.key == key,
            )
            .all()
        )
        if len(tags) == 0:
            return None
        if len(tags) > 1:
            raise MlflowException(
                f"Expected only 1 model version tag with name={name}, version={version}, "
                f"key={key}. Found {len(tags)}.",
                INVALID_STATE,
            )
        return tags[0]

    def set_model_version_tag(self, name, version, tag):
        """
        Set a tag for the model version.

        Args:
            name: Registered model name.
            version: Registered model version.
            tag: :py:class:`mlflow.entities.model_registry.ModelVersionTag` instance to log.

        Returns:
            None
        """
        _validate_model_name(name)
        _validate_model_version(version)
        _validate_model_version_tag(tag.key, tag.value)
        with self.ManagedSessionMaker() as session:
            # check if model version exists
            self._get_sql_model_version(session, name, version)
            session.merge(
                SqlModelVersionTag(name=name, version=version, key=tag.key, value=tag.value)
            )

    def delete_model_version_tag(self, name, version, key):
        """
        Delete a tag associated with the model version.

        Args:
            name: Registered model name.
            version: Registered model version.
            key: Tag key.

        Returns:
            None
        """
        _validate_model_name(name)
        _validate_model_version(version)
        _validate_tag_name(key)
        with self.ManagedSessionMaker() as session:
            # check if model version exists
            self._get_sql_model_version(session, name, version)
            existing_tag = self._get_model_version_tag(session, name, version, key)
            if existing_tag is not None:
                session.delete(existing_tag)

    @classmethod
    def _get_registered_model_alias(cls, session, name, alias):
        return (
            session.query(SqlRegisteredModelAlias)
            .filter(
                SqlRegisteredModelAlias.name == name,
                SqlRegisteredModelAlias.alias == alias,
            )
            .first()
        )

    def set_registered_model_alias(self, name, alias, version):
        """
        Set a registered model alias pointing to a model version.

        Args:
            name: Registered model name.
            alias: Name of the alias.
            version: Registered model version number.

        Returns:
            None
        """
        _validate_model_name(name)
        _validate_model_alias_name(alias)
        _validate_model_version(version)
        with self.ManagedSessionMaker() as session:
            # check if model version exists
            self._get_sql_model_version(session, name, version)
            session.merge(SqlRegisteredModelAlias(name=name, alias=alias, version=version))

    def delete_registered_model_alias(self, name, alias):
        """
        Delete an alias associated with a registered model.

        Args:
            name: Registered model name.
            alias: Name of the alias.

        Returns:
            None
        """
        _validate_model_name(name)
        _validate_model_alias_name(alias)
        with self.ManagedSessionMaker() as session:
            # check if registered model exists
            self._get_registered_model(session, name)
            existing_alias = self._get_registered_model_alias(session, name, alias)
            if existing_alias is not None:
                session.delete(existing_alias)

    def get_model_version_by_alias(self, name, alias):
        """
        Get the model version instance by name and alias.

        Args:
            name: Registered model name.
            alias: Name of the alias.

        Returns:
            A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.
        """
        _validate_model_name(name)
        _validate_model_alias_name(alias)
        with self.ManagedSessionMaker() as session:
            existing_alias = self._get_registered_model_alias(session, name, alias)
            if existing_alias is not None:
                sql_model_version = self._get_sql_model_version(
                    session, existing_alias.name, existing_alias.version
                )
                return self._populate_model_version_aliases(
                    session, name, sql_model_version.to_mlflow_entity()
                )
            else:
                raise MlflowException(
                    f"Registered model alias {alias} not found.", INVALID_PARAMETER_VALUE
                )

    def _await_model_version_creation(self, mv, await_creation_for):
        """
        Does not wait for the model version to become READY as a successful creation will
        immediately place the model version in a READY state.
        """
