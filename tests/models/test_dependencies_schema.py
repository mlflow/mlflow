from dataclasses import asdict
from unittest import mock

import pytest

from mlflow.models.dependencies_schemas import (
    DependenciesSchemasType,
    RetrieverSchema,
    clear_dependencies_schemas,
    get_dependencies_schemas,
    set_retriever_schema,
)


@pytest.fixture(autouse=True)
def clean_up():
    yield

    clear_dependencies_schemas()
    assert get_dependencies_schemas() == {}


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


def test_retriever_as_dict():
    vsi = RetrieverSchema(
        name="index-name",
        primary_key="primary-key",
        text_column="text-column",
        doc_uri="doc-uri",
        other_columns=["column1", "column2"],
    )
    assert asdict(vsi) == {
        "name": "index-name",
        "primary_key": "primary-key",
        "text_column": "text-column",
        "doc_uri": "doc-uri",
        "other_columns": ["column1", "column2"],
    }


def test_set_retriever_schema_creation():
    set_retriever_schema(
        primary_key="primary-key",
        text_column="text-column",
        doc_uri="doc-uri",
        other_columns=["column1", "column2"],
    )
    assert get_dependencies_schemas() == {
        DependenciesSchemasType.RETRIEVERS.value: [
            RetrieverSchema(
                name="retriever",
                primary_key="primary-key",
                text_column="text-column",
                doc_uri="doc-uri",
                other_columns=["column1", "column2"],
            )
        ]
    }


def test_set_retriever_schema_creation_with_name():
    set_retriever_schema(
        name="my_ret_2",
        primary_key="primary-key",
        text_column="text-column",
        doc_uri="doc-uri",
        other_columns=["column1", "column2"],
    )
    assert get_dependencies_schemas() == {
        DependenciesSchemasType.RETRIEVERS.value: [
            RetrieverSchema(
                name="my_ret_2",
                primary_key="primary-key",
                text_column="text-column",
                doc_uri="doc-uri",
                other_columns=["column1", "column2"],
            )
        ]
    }


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

    assert get_dependencies_schemas() == {
        DependenciesSchemasType.RETRIEVERS.value: [
            RetrieverSchema(
                name="my_ret_1",
                primary_key="primary-key-2",
                text_column="text-column-1",
                doc_uri="doc-uri-3",
                other_columns=["column1", "column2"],
            ),
            RetrieverSchema(
                name="my_ret_2",
                primary_key="primary-key",
                text_column="text-column",
                doc_uri="doc-uri",
                other_columns=["column1", "column2"],
            ),
        ]
    }


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

    assert get_dependencies_schemas() == {
        DependenciesSchemasType.RETRIEVERS.value: [
            RetrieverSchema(
                name="my_ret_1",
                primary_key="primary-key",
                text_column="text-column",
                doc_uri="doc-uri",
                other_columns=["column1", "column2"],
            ),
            RetrieverSchema(
                name="my_ret_2",
                primary_key="primary-key",
                text_column="text-column",
                doc_uri="doc-uri",
                other_columns=["column1", "column2"],
            ),
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

    assert get_dependencies_schemas() == {
        DependenciesSchemasType.RETRIEVERS.value: [
            RetrieverSchema(
                name="my_ret_1",
                primary_key="primary-key",
                text_column="text-column",
                doc_uri="doc-uri",
                other_columns=["column1", "column2"],
            ),
            RetrieverSchema(
                name="my_ret_2",
                primary_key="primary-key",
                text_column="text-column",
                doc_uri="doc-uri",
                other_columns=["column1", "column2"],
            ),
        ]
    }


# TODO: Add tests for set_dependencies_schema_from_model
