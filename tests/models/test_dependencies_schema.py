from unittest import mock

from mlflow.models.dependencies_schemas import (
    DependenciesSchemas,
    DependenciesSchemasType,
    RetrieverSchema,
    _get_dependencies_schemas,
    _get_retriever_schema,
    set_retriever_schema,
)


def test_retriever_creation():
    vsi = RetrieverSchema(
        name="index-name",
        primary_key="primary-key",
        text_column="text-column",
        doc_uri="doc-uri",
        other_columns=["column1", "column2"],
    )
    assert vsi.name == "index-name"
    assert vsi.primary_key == "primary-key"
    assert vsi.text_column == "text-column"
    assert vsi.doc_uri == "doc-uri"
    assert vsi.other_columns == ["column1", "column2"]


def test_retriever_to_dict():
    vsi = RetrieverSchema(
        name="index-name",
        primary_key="primary-key",
        text_column="text-column",
        doc_uri="doc-uri",
        other_columns=["column1", "column2"],
    )
    expected_dict = {
        DependenciesSchemasType.RETRIEVERS.value: [
            {
                "name": "index-name",
                "primary_key": "primary-key",
                "text_column": "text-column",
                "doc_uri": "doc-uri",
                "other_columns": ["column1", "column2"],
            }
        ]
    }
    assert vsi.to_dict() == expected_dict


def test_retriever_from_dict():
    data = {
        "name": "index-name",
        "primary_key": "primary-key",
        "text_column": "text-column",
        "doc_uri": "doc-uri",
        "other_columns": ["column1", "column2"],
    }
    vsi = RetrieverSchema.from_dict(data)
    assert vsi.name == "index-name"
    assert vsi.primary_key == "primary-key"
    assert vsi.text_column == "text-column"
    assert vsi.doc_uri == "doc-uri"
    assert vsi.other_columns == ["column1", "column2"]


def test_dependencies_schemas_to_dict():
    vsi = RetrieverSchema(
        name="index-name",
        primary_key="primary-key",
        text_column="text-column",
        doc_uri="doc-uri",
        other_columns=["column1", "column2"],
    )
    schema = DependenciesSchemas(retriever_schemas=[vsi])
    expected_dict = {
        "dependencies_schemas": {
            DependenciesSchemasType.RETRIEVERS.value: [
                {
                    "name": "index-name",
                    "primary_key": "primary-key",
                    "text_column": "text-column",
                    "doc_uri": "doc-uri",
                    "other_columns": ["column1", "column2"],
                }
            ]
        }
    }
    assert schema.to_dict() == expected_dict


def test_set_retriever_schema_creation():
    set_retriever_schema(
        primary_key="primary-key",
        text_column="text-column",
        doc_uri="doc-uri",
        other_columns=["column1", "column2"],
    )
    with _get_dependencies_schemas() as schema:
        assert schema.to_dict()["dependencies_schemas"] == {
            DependenciesSchemasType.RETRIEVERS.value: [
                {
                    "doc_uri": "doc-uri",
                    "name": "retriever",
                    "other_columns": ["column1", "column2"],
                    "primary_key": "primary-key",
                    "text_column": "text-column",
                }
            ]
        }

    # Schema is automatically reset
    with _get_dependencies_schemas() as schema:
        assert schema.to_dict() is None
    assert _get_retriever_schema() == []


def test_set_retriever_schema_creation_with_name():
    set_retriever_schema(
        name="my_ret_2",
        primary_key="primary-key",
        text_column="text-column",
        doc_uri="doc-uri",
        other_columns=["column1", "column2"],
    )
    with _get_dependencies_schemas() as schema:
        assert schema.to_dict()["dependencies_schemas"] == {
            DependenciesSchemasType.RETRIEVERS.value: [
                {
                    "doc_uri": "doc-uri",
                    "name": "my_ret_2",
                    "other_columns": ["column1", "column2"],
                    "primary_key": "primary-key",
                    "text_column": "text-column",
                }
            ]
        }

    # Schema is automatically reset
    with _get_dependencies_schemas() as schema:
        assert schema.to_dict() is None
    assert _get_retriever_schema() == []


def test_set_retriever_schema_empty_creation():
    with _get_dependencies_schemas() as schema:
        assert schema.to_dict() is None


def test_multiple_set_retriever_schema_creation_with_name():
    set_retriever_schema(
        name="my_ret_1",
        primary_key="primary-key-2",
        text_column="text-column-1",
        doc_uri="doc-uri-3",
        other_columns=["column1", "column2"],
    )

    set_retriever_schema(
        name="my_ret_2",
        primary_key="primary-key",
        text_column="text-column",
        doc_uri="doc-uri",
        other_columns=["column1", "column2"],
    )
    with _get_dependencies_schemas() as schema:
        assert schema.to_dict()["dependencies_schemas"] == {
            DependenciesSchemasType.RETRIEVERS.value: [
                {
                    "doc_uri": "doc-uri-3",
                    "name": "my_ret_1",
                    "other_columns": ["column1", "column2"],
                    "primary_key": "primary-key-2",
                    "text_column": "text-column-1",
                },
                {
                    "doc_uri": "doc-uri",
                    "name": "my_ret_2",
                    "other_columns": ["column1", "column2"],
                    "primary_key": "primary-key",
                    "text_column": "text-column",
                },
            ]
        }

    # Schema is automatically reset
    with _get_dependencies_schemas() as schema:
        assert schema.to_dict() is None
    assert _get_retriever_schema() == []


def test_multiple_set_retriever_schema_with_same_name_with_different_schemas():
    set_retriever_schema(
        name="my_ret_1",
        primary_key="primary-key-2",
        text_column="text-column-1",
        doc_uri="doc-uri-3",
        other_columns=["column1", "column2"],
    )
    set_retriever_schema(
        name="my_ret_2",
        primary_key="primary-key",
        text_column="text-column",
        doc_uri="doc-uri",
        other_columns=["column1", "column2"],
    )

    with mock.patch("mlflow.models.dependencies_schemas._logger") as mock_logger:
        set_retriever_schema(
            name="my_ret_1",
            primary_key="primary-key",
            text_column="text-column",
            doc_uri="doc-uri",
            other_columns=["column1", "column2"],
        )
        mock_logger.warning.assert_called_once_with(
            "A retriever schema with the name 'my_ret_1' already exists. "
            "Overriding the existing schema."
        )

    with _get_dependencies_schemas() as schema:
        assert schema.to_dict()["dependencies_schemas"] == {
            DependenciesSchemasType.RETRIEVERS.value: [
                {
                    "doc_uri": "doc-uri",
                    "name": "my_ret_1",
                    "other_columns": ["column1", "column2"],
                    "primary_key": "primary-key",
                    "text_column": "text-column",
                },
                {
                    "doc_uri": "doc-uri",
                    "name": "my_ret_2",
                    "other_columns": ["column1", "column2"],
                    "primary_key": "primary-key",
                    "text_column": "text-column",
                },
            ]
        }


def test_multiple_set_retriever_schema_with_same_name_with_same_schema():
    set_retriever_schema(
        name="my_ret_1",
        primary_key="primary-key",
        text_column="text-column",
        doc_uri="doc-uri",
        other_columns=["column1", "column2"],
    )
    set_retriever_schema(
        name="my_ret_2",
        primary_key="primary-key",
        text_column="text-column",
        doc_uri="doc-uri",
        other_columns=["column1", "column2"],
    )

    with mock.patch("mlflow.models.dependencies_schemas._logger") as mock_logger:
        set_retriever_schema(
            name="my_ret_1",
            primary_key="primary-key",
            text_column="text-column",
            doc_uri="doc-uri",
            other_columns=["column1", "column2"],
        )
        mock_logger.warning.assert_not_called()

    with _get_dependencies_schemas() as schema:
        assert schema.to_dict()["dependencies_schemas"] == {
            DependenciesSchemasType.RETRIEVERS.value: [
                {
                    "doc_uri": "doc-uri",
                    "name": "my_ret_1",
                    "other_columns": ["column1", "column2"],
                    "primary_key": "primary-key",
                    "text_column": "text-column",
                },
                {
                    "doc_uri": "doc-uri",
                    "name": "my_ret_2",
                    "other_columns": ["column1", "column2"],
                    "primary_key": "primary-key",
                    "text_column": "text-column",
                },
            ]
        }
