import dataclasses
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from mlflow.genai import label_schemas
from mlflow.genai.label_schemas import (
    EXPECTED_FACTS,
    EXPECTED_RESPONSE,
    GUIDELINES,
    InputCategorical,
    InputCategoricalList,
    InputNumeric,
    InputText,
    InputTextList,
    LabelSchema,
    LabelSchemaType,
    create_label_schema,
    delete_label_schema,
    get_label_schema,
)
from mlflow.genai.label_schemas.label_schemas import (
    InputType,
)
from mlflow.genai.scorers.validation import IS_DBX_AGENTS_INSTALLED

if not IS_DBX_AGENTS_INSTALLED:
    pytest.skip("Skipping Databricks only test.", allow_module_level=True)


@pytest.fixture
def mock_databricks_labeling_store():
    """
    Fixture providing a fully mocked Databricks labeling store environment.

    Returns:
        A context manager that provides mocked store, review app, and databricks modules.
    """

    @contextmanager
    def _mock_context():
        with patch("mlflow.genai.labeling.stores._get_labeling_store") as mock_get_store:
            from mlflow.genai.labeling.stores import DatabricksLabelingStore

            mock_store = DatabricksLabelingStore()
            mock_get_store.return_value = mock_store

            # Mock the databricks modules and review app
            with patch("databricks.agents.review_app.get_review_app") as mock_get_app:
                mock_app = MagicMock()
                mock_get_app.return_value = mock_app

                yield {
                    "store": mock_store,
                    "get_store": mock_get_store,
                    "app": mock_app,
                    "get_app": mock_get_app,
                }

    return _mock_context


@pytest.fixture
def mock_review_app():
    """
    Fixture providing just the review app mock for simpler test cases.

    Returns:
        A context manager that provides a mocked review app.
    """

    @contextmanager
    def _mock_context():
        with patch("databricks.agents.review_app.get_review_app") as mock_get_app:
            mock_app = MagicMock()
            mock_get_app.return_value = mock_app

            yield {
                "app": mock_app,
                "get_app": mock_get_app,
            }

    return _mock_context


# InputCategorical tests
def test_input_categorical_init():
    """Test InputCategorical initialization."""
    options = ["yes", "no", "maybe"]
    input_cat = InputCategorical(options=options)
    assert input_cat.options == options


def test_input_categorical_to_databricks_input():
    """Test conversion to Databricks input type."""
    options = ["good", "bad", "neutral"]
    input_cat = InputCategorical(options=options)

    mock_databricks_input = MagicMock()
    with patch(
        "databricks.agents.review_app.label_schemas.InputCategorical",
        return_value=mock_databricks_input,
    ) as mock_class:
        result = input_cat._to_databricks_input()

        mock_class.assert_called_once_with(options=options)
        assert result == mock_databricks_input


def test_input_categorical_from_databricks_input():
    """Test creation from Databricks input type."""
    options = ["excellent", "good", "poor"]
    mock_databricks_input = MagicMock()
    mock_databricks_input.options = options

    result = InputCategorical._from_databricks_input(mock_databricks_input)

    assert isinstance(result, InputCategorical)
    assert result.options == options


def test_input_categorical_empty_options():
    """Test InputCategorical with empty options."""
    input_cat = InputCategorical(options=[])
    assert input_cat.options == []


def test_input_categorical_single_option():
    """Test InputCategorical with single option."""
    input_cat = InputCategorical(options=["only_option"])
    assert input_cat.options == ["only_option"]


# InputCategoricalList tests
def test_input_categorical_list_init():
    """Test InputCategoricalList initialization."""
    options = ["red", "green", "blue"]
    input_cat_list = InputCategoricalList(options=options)
    assert input_cat_list.options == options


def test_input_categorical_list_to_databricks_input():
    """Test conversion to Databricks input type."""
    options = ["python", "java", "javascript"]
    input_cat_list = InputCategoricalList(options=options)

    mock_databricks_input = MagicMock()
    with patch(
        "databricks.agents.review_app.label_schemas.InputCategoricalList",
        return_value=mock_databricks_input,
    ) as mock_class:
        result = input_cat_list._to_databricks_input()

        mock_class.assert_called_once_with(options=options)
        assert result == mock_databricks_input


def test_input_categorical_list_from_databricks_input():
    """Test creation from Databricks input type."""
    options = ["feature1", "feature2", "feature3"]
    mock_databricks_input = MagicMock()
    mock_databricks_input.options = options

    result = InputCategoricalList._from_databricks_input(mock_databricks_input)

    assert isinstance(result, InputCategoricalList)
    assert result.options == options


# InputText tests
def test_input_text_init_with_max_length():
    """Test InputText initialization with max_length."""
    input_text = InputText(max_length=100)
    assert input_text.max_length == 100


def test_input_text_init_without_max_length():
    """Test InputText initialization without max_length."""
    input_text = InputText()
    assert input_text.max_length is None


def test_input_text_to_databricks_input():
    """Test conversion to Databricks input type."""
    max_length = 500
    input_text = InputText(max_length=max_length)

    mock_databricks_input = MagicMock()
    with patch(
        "databricks.agents.review_app.label_schemas.InputText",
        return_value=mock_databricks_input,
    ) as mock_class:
        result = input_text._to_databricks_input()

        mock_class.assert_called_once_with(max_length=max_length)
        assert result == mock_databricks_input


def test_input_text_from_databricks_input():
    """Test creation from Databricks input type."""
    max_length = 250
    mock_databricks_input = MagicMock()
    mock_databricks_input.max_length = max_length

    result = InputText._from_databricks_input(mock_databricks_input)

    assert isinstance(result, InputText)
    assert result.max_length == max_length


def test_input_text_from_databricks_input_none_max_length():
    """Test creation from Databricks input type with None max_length."""
    mock_databricks_input = MagicMock()
    mock_databricks_input.max_length = None

    result = InputText._from_databricks_input(mock_databricks_input)

    assert isinstance(result, InputText)
    assert result.max_length is None


# InputTextList tests
def test_input_text_list_init_with_all_params():
    """Test InputTextList initialization with all parameters."""
    input_text_list = InputTextList(max_length_each=50, max_count=5)
    assert input_text_list.max_length_each == 50
    assert input_text_list.max_count == 5


def test_input_text_list_init_with_partial_params():
    """Test InputTextList initialization with partial parameters."""
    input_text_list = InputTextList(max_count=3)
    assert input_text_list.max_length_each is None
    assert input_text_list.max_count == 3


def test_input_text_list_init_with_no_params():
    """Test InputTextList initialization with no parameters."""
    input_text_list = InputTextList()
    assert input_text_list.max_length_each is None
    assert input_text_list.max_count is None


def test_input_text_list_to_databricks_input():
    """Test conversion to Databricks input type."""
    max_length_each = 100
    max_count = 10
    input_text_list = InputTextList(max_length_each=max_length_each, max_count=max_count)

    mock_databricks_input = MagicMock()
    with patch(
        "databricks.agents.review_app.label_schemas.InputTextList",
        return_value=mock_databricks_input,
    ) as mock_class:
        result = input_text_list._to_databricks_input()

        mock_class.assert_called_once_with(max_length_each=max_length_each, max_count=max_count)
        assert result == mock_databricks_input


def test_input_text_list_from_databricks_input():
    """Test creation from Databricks input type."""
    max_length_each = 75
    max_count = 8
    mock_databricks_input = MagicMock()
    mock_databricks_input.max_length_each = max_length_each
    mock_databricks_input.max_count = max_count

    result = InputTextList._from_databricks_input(mock_databricks_input)

    assert isinstance(result, InputTextList)
    assert result.max_length_each == max_length_each
    assert result.max_count == max_count


# InputNumeric tests
def test_input_numeric_init_with_all_params():
    """Test InputNumeric initialization with all parameters."""
    input_numeric = InputNumeric(min_value=0.0, max_value=10.0)
    assert input_numeric.min_value == 0.0
    assert input_numeric.max_value == 10.0


def test_input_numeric_init_with_partial_params():
    """Test InputNumeric initialization with partial parameters."""
    input_numeric = InputNumeric(min_value=5.0)
    assert input_numeric.min_value == 5.0
    assert input_numeric.max_value is None


def test_input_numeric_init_with_no_params():
    """Test InputNumeric initialization with no parameters."""
    input_numeric = InputNumeric()
    assert input_numeric.min_value is None
    assert input_numeric.max_value is None


def test_input_numeric_to_databricks_input():
    """Test conversion to Databricks input type."""
    min_value = 1.5
    max_value = 9.5
    input_numeric = InputNumeric(min_value=min_value, max_value=max_value)

    mock_databricks_input = MagicMock()
    with patch(
        "databricks.agents.review_app.label_schemas.InputNumeric",
        return_value=mock_databricks_input,
    ) as mock_class:
        result = input_numeric._to_databricks_input()

        mock_class.assert_called_once_with(min_value=min_value, max_value=max_value)
        assert result == mock_databricks_input


def test_input_numeric_from_databricks_input():
    """Test creation from Databricks input type."""
    min_value = -5.0
    max_value = 15.0
    mock_databricks_input = MagicMock()
    mock_databricks_input.min_value = min_value
    mock_databricks_input.max_value = max_value

    result = InputNumeric._from_databricks_input(mock_databricks_input)

    assert isinstance(result, InputNumeric)
    assert result.min_value == min_value
    assert result.max_value == max_value


def test_input_numeric_negative_values():
    """Test InputNumeric with negative values."""
    input_numeric = InputNumeric(min_value=-100.0, max_value=-10.0)
    assert input_numeric.min_value == -100.0
    assert input_numeric.max_value == -10.0


def test_input_numeric_integer_values():
    """Test InputNumeric with integer values."""
    input_numeric = InputNumeric(min_value=1, max_value=100)
    assert input_numeric.min_value == 1
    assert input_numeric.max_value == 100


# InputType tests
def test_input_type_abstract_methods():
    """Test that InputType is abstract and requires implementation."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class InputType"):
        InputType()


@pytest.mark.parametrize(
    "input_class",
    [
        InputCategorical,
        InputCategoricalList,
        InputText,
        InputTextList,
        InputNumeric,
    ],
)
def test_input_type_all_inputs_inherit_from_input_type(input_class):
    """Test that all input classes inherit from InputType."""
    assert issubclass(input_class, InputType)


@pytest.mark.parametrize(
    "input_obj",
    [
        InputCategorical(options=["test"]),
        InputCategoricalList(options=["test"]),
        InputText(),
        InputTextList(),
        InputNumeric(),
    ],
)
def test_input_type_all_inputs_implement_required_methods(input_obj):
    """Test that all input classes implement required abstract methods."""
    assert hasattr(input_obj, "_to_databricks_input")
    assert callable(getattr(input_obj, "_to_databricks_input"))
    assert hasattr(input_obj.__class__, "_from_databricks_input")
    assert callable(getattr(input_obj.__class__, "_from_databricks_input"))


# LabelSchemaType tests
@pytest.mark.parametrize(
    ("enum_member", "expected_value"),
    [
        (LabelSchemaType.FEEDBACK, "feedback"),
        (LabelSchemaType.EXPECTATION, "expectation"),
    ],
)
def test_label_schema_type_enum_values(enum_member, expected_value):
    """Test LabelSchemaType enum values."""
    assert enum_member == expected_value


@pytest.mark.parametrize(
    ("value", "should_be_member"),
    [
        ("feedback", True),
        ("expectation", True),
        ("invalid", False),
        ("", False),
        ("FEEDBACK", False),
    ],
)
def test_label_schema_type_enum_membership(value, should_be_member):
    """Test enum membership."""
    if should_be_member:
        assert value in LabelSchemaType
    else:
        assert value not in LabelSchemaType


# LabelSchema tests
def test_label_schema_init_with_categorical_input():
    """Test LabelSchema initialization with categorical input."""
    input_cat = InputCategorical(options=["good", "bad"])
    schema = LabelSchema(
        name="quality",
        type=LabelSchemaType.FEEDBACK,
        title="Rate the quality",
        input=input_cat,
    )

    assert schema.name == "quality"
    assert schema.type == LabelSchemaType.FEEDBACK
    assert schema.title == "Rate the quality"
    assert schema.input == input_cat
    assert schema.instruction is None
    assert schema.enable_comment is False


def test_label_schema_init_with_all_params():
    """Test LabelSchema initialization with all parameters."""
    input_text = InputText(max_length=200)
    schema = LabelSchema(
        name="feedback_schema",
        type=LabelSchemaType.EXPECTATION,
        title="Provide feedback",
        input=input_text,
        instruction="Please be detailed",
        enable_comment=True,
    )

    assert schema.name == "feedback_schema"
    assert schema.type == LabelSchemaType.EXPECTATION
    assert schema.title == "Provide feedback"
    assert schema.input == input_text
    assert schema.instruction == "Please be detailed"
    assert schema.enable_comment is True


def test_label_schema_init_with_numeric_input():
    """Test LabelSchema initialization with numeric input."""
    input_numeric = InputNumeric(min_value=1.0, max_value=5.0)
    schema = LabelSchema(
        name="rating",
        type=LabelSchemaType.FEEDBACK,
        title="Rate from 1 to 5",
        input=input_numeric,
    )

    assert schema.input == input_numeric


def test_label_schema_init_with_text_list_input():
    """Test LabelSchema initialization with text list input."""
    input_text_list = InputTextList(max_length_each=50, max_count=3)
    schema = LabelSchema(
        name="suggestions",
        type=LabelSchemaType.EXPECTATION,
        title="Provide suggestions",
        input=input_text_list,
    )

    assert schema.input == input_text_list


def test_label_schema_init_with_categorical_list_input():
    """Test LabelSchema initialization with categorical list input."""
    input_cat_list = InputCategoricalList(options=["tag1", "tag2", "tag3"])
    schema = LabelSchema(
        name="tags",
        type=LabelSchemaType.FEEDBACK,
        title="Select relevant tags",
        input=input_cat_list,
    )

    assert schema.input == input_cat_list


def test_label_schema_frozen_dataclass():
    """Test that LabelSchema is frozen (immutable)."""
    input_cat = InputCategorical(options=["test"])
    schema = LabelSchema(
        name="test",
        type=LabelSchemaType.FEEDBACK,
        title="Test",
        input=input_cat,
    )

    with pytest.raises(dataclasses.FrozenInstanceError, match="cannot assign to field"):
        schema.name = "new_name"


def test_label_schema_from_databricks_label_schema():
    """Test creation from Databricks label schema."""
    # Create a mock databricks input object
    mock_databricks_input = MagicMock()

    # Mock Databricks schema
    mock_databricks_schema = MagicMock()
    mock_databricks_schema.name = "test_schema"
    mock_databricks_schema.type = LabelSchemaType.FEEDBACK
    mock_databricks_schema.title = "Test Schema"
    mock_databricks_schema.instruction = "Test instruction"
    mock_databricks_schema.enable_comment = True
    mock_databricks_schema.input = mock_databricks_input

    expected_input = InputText(max_length=100)

    with patch("databricks.agents.review_app.label_schemas") as mock_label_schemas:
        mock_label_schemas.InputText = type(mock_databricks_input)

        # Mock the _from_databricks_input method
        with patch.object(
            InputText, "_from_databricks_input", return_value=expected_input
        ) as mock_from_db:
            result = LabelSchema._from_databricks_label_schema(mock_databricks_schema)

            assert isinstance(result, LabelSchema)
            assert result.name == "test_schema"
            assert result.type == LabelSchemaType.FEEDBACK
            assert result.title == "Test Schema"
            assert result.instruction == "Test instruction"
            assert result.enable_comment is True
            assert result.input == expected_input
            mock_from_db.assert_called_once_with(mock_databricks_input)


def test_convert_databricks_input():
    """Test _convert_databricks_input converts Databricks input types to MLflow types."""

    # Create a simple mock that can be used as dict key
    class MockInputTextList:
        pass

    mock_input = MockInputTextList()
    expected = InputTextList(max_count=5)

    # Patch the import and the method
    with patch("databricks.agents.review_app.label_schemas") as mock_schemas:
        mock_schemas.InputTextList = MockInputTextList

        with patch.object(
            InputTextList, "_from_databricks_input", return_value=expected
        ) as mock_from_db:
            result = LabelSchema._convert_databricks_input(mock_input)
            assert result == expected
            mock_from_db.assert_called_once_with(mock_input)


def test_convert_databricks_input_unknown_type():
    """Test _convert_databricks_input raises ValueError for unknown types."""
    with patch("databricks.agents.review_app.label_schemas"):
        unknown_input = MagicMock()
        unknown_input.__class__ = MagicMock()  # Unknown type

        with pytest.raises(ValueError, match="Unknown input type"):
            LabelSchema._convert_databricks_input(unknown_input)


def test_from_databricks_label_schema_uses_convert_input():
    """Test _from_databricks_label_schema properly converts input via _convert_databricks_input."""
    mock_schema = MagicMock()
    mock_schema.name = "test"
    mock_schema.type = LabelSchemaType.FEEDBACK
    mock_schema.title = "Test"

    expected_input = InputTextList(max_count=3)
    with patch.object(
        LabelSchema, "_convert_databricks_input", return_value=expected_input
    ) as mock_convert:
        result = LabelSchema._from_databricks_label_schema(mock_schema)

        assert result.input == expected_input
        mock_convert.assert_called_once_with(mock_schema.input)


# Integration tests
def test_integration_complete_workflow_categorical():
    """Test complete workflow with categorical input."""
    # Create InputCategorical
    options = ["excellent", "good", "fair", "poor"]
    input_cat = InputCategorical(options=options)

    # Convert to Databricks input and back
    with patch("databricks.agents.review_app.label_schemas.InputCategorical") as mock_class:
        mock_databricks_input = MagicMock()
        mock_databricks_input.options = options
        mock_class.return_value = mock_databricks_input

        # To Databricks
        databricks_input = input_cat._to_databricks_input()

        # From Databricks
        result = InputCategorical._from_databricks_input(databricks_input)

        assert isinstance(result, InputCategorical)
        assert result.options == options


def test_integration_complete_workflow_numeric():
    """Test complete workflow with numeric input."""
    # Create InputNumeric
    min_val = 0.0
    max_val = 10.0
    input_numeric = InputNumeric(min_value=min_val, max_value=max_val)

    # Convert to Databricks input and back
    with patch("databricks.agents.review_app.label_schemas.InputNumeric") as mock_class:
        mock_databricks_input = MagicMock()
        mock_databricks_input.min_value = min_val
        mock_databricks_input.max_value = max_val
        mock_class.return_value = mock_databricks_input

        # To Databricks
        databricks_input = input_numeric._to_databricks_input()

        # From Databricks
        result = InputNumeric._from_databricks_input(databricks_input)

        assert isinstance(result, InputNumeric)
        assert result.min_value == min_val
        assert result.max_value == max_val


@pytest.mark.parametrize(
    ("input_type", "schema_name"),
    [
        (InputCategorical(options=["yes", "no"]), "categorical_schema"),
        (InputCategoricalList(options=["a", "b", "c"]), "categorical_list_schema"),
        (InputText(max_length=100), "text_schema"),
        (InputTextList(max_count=5), "text_list_schema"),
        (InputNumeric(min_value=1, max_value=10), "numeric_schema"),
    ],
)
def test_integration_label_schema_with_different_input_types(input_type, schema_name):
    """Test LabelSchema works with all input types."""
    schema = LabelSchema(
        name=schema_name,
        type=LabelSchemaType.FEEDBACK,
        title=f"Schema for {schema_name}",
        input=input_type,
    )

    assert schema.input == input_type
    assert isinstance(schema.input, InputType)


# Edge case tests
def test_edge_cases_empty_string_values():
    """Test handling of empty string values."""
    schema = LabelSchema(
        name="",
        type=LabelSchemaType.FEEDBACK,
        title="",
        input=InputCategorical(options=[]),
        instruction="",
    )

    assert schema.name == ""
    assert schema.title == ""
    assert schema.instruction == ""


@pytest.mark.parametrize(
    ("min_value", "max_value", "description"),
    [
        (1e10, 1e20, "very_large_values"),
        (-1000.5, -0.1, "negative_range"),
        (0.0, 0.0, "zero_range"),
        (-float("inf"), float("inf"), "infinite_range"),
        (1.123456789, 2.987654321, "high_precision_decimals"),
    ],
)
def test_edge_cases_numeric_value_ranges(min_value, max_value, description):
    """Test handling of various numeric value ranges."""
    input_numeric = InputNumeric(min_value=min_value, max_value=max_value)
    assert input_numeric.min_value == min_value
    assert input_numeric.max_value == max_value


def test_edge_cases_zero_max_length_text():
    """Test handling of zero max_length for text."""
    input_text = InputText(max_length=0)
    assert input_text.max_length == 0


def test_edge_cases_zero_max_count_text_list():
    """Test handling of zero max_count for text list."""
    input_text_list = InputTextList(max_count=0)
    assert input_text_list.max_count == 0


@pytest.mark.parametrize(
    "options",
    [
        [
            "option with spaces",
            "option-with-dashes",
            "option_with_underscores",
            "option@with$pecial!chars",
        ],
        ["🙂", "😢", "🤔", "αβγ", "中文"],
        ["", "empty_and_normal", ""],
        ["UPPERCASE", "lowercase", "MiXeD_CaSe"],
    ],
)
def test_edge_cases_special_and_unicode_characters_in_options(options):
    """Test handling of special characters and unicode in categorical options."""
    input_cat = InputCategorical(options=options)
    assert input_cat.options == options


# API integration tests
def test_create_label_schema_calls_to_databricks_input(mock_databricks_labeling_store):
    """Test that create_label_schema calls _to_databricks_input on the input."""
    input_cat = InputCategorical(options=["good", "bad"])

    with mock_databricks_labeling_store() as mocks:
        # Configure the mock app for this test
        mocks["app"].create_label_schema.return_value = MagicMock()

        # Mock the _to_databricks_input method
        with patch.object(input_cat, "_to_databricks_input") as mock_to_databricks:
            mock_databricks_input = MagicMock()
            mock_to_databricks.return_value = mock_databricks_input

            # Import here to avoid early import errors
            from mlflow.genai.label_schemas import create_label_schema

            create_label_schema(
                name="test_schema",
                type="feedback",
                title="Test Schema",
                input=input_cat,
            )

            # Verify _to_databricks_input was called
            mock_to_databricks.assert_called_once()
            # Verify the result was passed to create_label_schema
            mocks["app"].create_label_schema.assert_called_once_with(
                name="test_schema",
                type="feedback",
                title="Test Schema",
                input=mock_databricks_input,
                instruction=None,
                enable_comment=False,
                overwrite=False,
            )


def test_get_label_schema_calls_from_databricks_label_schema(mock_databricks_labeling_store):
    """Test that get_label_schema calls _from_databricks_label_schema."""
    # Mock databricks label schema
    mock_databricks_schema = MagicMock()
    mock_databricks_schema.name = "test_schema"

    with mock_databricks_labeling_store() as mocks:
        # Configure the mock app for this test
        mocks["app"].label_schemas = [mock_databricks_schema]

        # Mock the _from_databricks_label_schema method
        with patch.object(LabelSchema, "_from_databricks_label_schema") as mock_from_databricks:
            mock_label_schema = MagicMock()
            mock_from_databricks.return_value = mock_label_schema

            # Import here to avoid early import errors
            from mlflow.genai.label_schemas import get_label_schema

            result = get_label_schema("test_schema")

            # Verify _from_databricks_label_schema was called
            mock_from_databricks.assert_called_once_with(mock_databricks_schema)
            # Verify the result was returned
            assert result == mock_label_schema


@pytest.mark.parametrize(
    ("input_type", "schema_name"),
    [
        (InputCategorical(options=["yes", "no"]), "categorical_api_test"),
        (InputCategoricalList(options=["a", "b", "c"]), "categorical_list_api_test"),
        (InputText(max_length=100), "text_api_test"),
        (InputTextList(max_count=5), "text_list_api_test"),
        (InputNumeric(min_value=1, max_value=10), "numeric_api_test"),
    ],
)
def test_api_integration_with_all_input_types(
    input_type, schema_name, mock_databricks_labeling_store
):
    """Test that API integration works with all input types."""
    with mock_databricks_labeling_store() as mocks:
        # Configure the mock app for this test
        mocks["app"].create_label_schema.return_value = MagicMock()

        # Mock the _to_databricks_input method
        with patch.object(input_type, "_to_databricks_input") as mock_to_databricks:
            mock_databricks_input = MagicMock()
            mock_to_databricks.return_value = mock_databricks_input

            # Import here to avoid early import errors
            from mlflow.genai.label_schemas import create_label_schema

            create_label_schema(
                name=schema_name,
                type="feedback",
                title=f"Test Schema for {schema_name}",
                input=input_type,
            )

            # Verify _to_databricks_input was called for each type
            mock_to_databricks.assert_called_once()


# Import tests
def test_databricks_label_schemas_is_importable():
    """Test that all constants, classes, and functions are importable."""
    # Test constants
    assert label_schemas.EXPECTED_FACTS == EXPECTED_FACTS
    assert label_schemas.GUIDELINES == GUIDELINES
    assert label_schemas.EXPECTED_RESPONSE == EXPECTED_RESPONSE

    # Test classes
    assert label_schemas.LabelSchemaType == LabelSchemaType
    assert label_schemas.LabelSchema == LabelSchema
    assert label_schemas.InputCategorical == InputCategorical
    assert label_schemas.InputCategoricalList == InputCategoricalList
    assert label_schemas.InputNumeric == InputNumeric
    assert label_schemas.InputText == InputText
    assert label_schemas.InputTextList == InputTextList

    # Test functions
    assert label_schemas.create_label_schema == create_label_schema
    assert label_schemas.get_label_schema == get_label_schema
    assert label_schemas.delete_label_schema == delete_label_schema
