import pandas as pd
import numpy as np
import pytest

from mlflow.models import infer_signature
from mlflow.models.utils import _enforce_schema
from mlflow.types.schema import DataType, ColSpec, Schema
from mlflow.exceptions import MlflowException


@pytest.mark.categorical
def test_datatype_from_type_with_categorical():
    """
    Tests that DataType.from_type correctly maps pandas CategoricalDtype to DataType.category.
    """
    assert DataType.from_type(pd.CategoricalDtype()) == DataType.category
    assert DataType.category.to_pandas() == pd.CategoricalDtype
    assert DataType.category.to_numpy() == np.dtype("object")
    assert DataType.category.to_python() == str


@pytest.mark.categorical
def test_infer_signature_with_categorical_column():
    """
    Tests that infer_signature correctly infers DataType.category for pandas categorical columns.
    """
    df = pd.DataFrame({
        "col_str": ["A", "B", "C"],
        "col_cat": pd.Series(["X", "Y", "X"], dtype="category"),
        "col_int": [1, 2, 3]
    })

    signature = infer_signature(df)
    assert signature is not None
    assert len(signature.inputs.inputs) == 3

    # Check col_str
    assert signature.inputs.inputs[0].name == "col_str"
    assert signature.inputs.inputs[0].type == DataType.string

    # Check col_cat
    assert signature.inputs.inputs[1].name == "col_cat"
    assert signature.inputs.inputs[1].type == DataType.category

    # Check col_int
    assert signature.inputs.inputs[2].name == "col_int"
    assert signature.inputs.inputs[2].type == DataType.long


@pytest.mark.categorical
def test_enforce_schema_with_categorical_column_conversion():
    """
    Tests that _enforce_schema correctly converts a string column to a categorical column
    when the schema expects DataType.category.
    """
    # Schema expecting a categorical column
    schema = Schema([
        ColSpec(type=DataType.string, name="col_str"),
        ColSpec(type=DataType.category, name="col_cat"),
        ColSpec(type=DataType.long, name="col_int")
    ])

    # Input DataFrame with a string column that should be converted to categorical
    input_df = pd.DataFrame({
        "col_str": ["hello", "world"],
        "col_cat": ["apple", "banana"],  # This will be converted to category
        "col_int": [10, 20]
    })

    enforced_df = _enforce_schema(input_df, schema)

    assert isinstance(enforced_df, pd.DataFrame)
    assert enforced_df["col_str"].dtype == object # string type in pandas is object
    assert isinstance(enforced_df["col_cat"].dtype, pd.CategoricalDtype)
    assert enforced_df["col_int"].dtype == np.int64


@pytest.mark.categorical
def test_enforce_schema_with_already_categorical_column():
    """
    Tests that _enforce_schema handles an already categorical column correctly.
    """
    schema = Schema([
        ColSpec(type=DataType.category, name="col_cat")
    ])

    input_df = pd.DataFrame({
        "col_cat": pd.Series(["red", "green", "blue"], dtype="category")
    })

    enforced_df = _enforce_schema(input_df, schema)

    assert isinstance(enforced_df, pd.DataFrame)
    assert isinstance(enforced_df["col_cat"].dtype, pd.CategoricalDtype)
    pd.testing.assert_series_equal(input_df["col_cat"], enforced_df["col_cat"])


@pytest.mark.categorical
def test_enforce_schema_categorical_with_unseen_categories():
    """
    Tests _enforce_schema behavior when input has unseen categories.
    Since DataType.category doesn't store categories, pandas' default `astype('category')`
    behavior (which infers new categories from the data) is expected.
    """
    schema = Schema([
        ColSpec(type=DataType.category, name="col_cat")
    ])

    # Create an initial categorical series to simulate original data
    original_categories = pd.CategoricalDtype(categories=["A", "B"])
    input_df = pd.DataFrame({
        "col_cat": pd.Series(["A", "C"], dtype=object) # 'C' is unseen
    })

    # When converting to 'category' without specifying categories, pandas will infer them
    # from the input data.
    enforced_df = _enforce_schema(input_df, schema)

    assert isinstance(enforced_df, pd.DataFrame)
    assert isinstance(enforced_df["col_cat"].dtype, pd.CategoricalDtype)
    # The categories in the enforced DataFrame should now include 'C'
    assert "A" in enforced_df["col_cat"].cat.categories
    assert "C" in enforced_df["col_cat"].cat.categories
    assert "B" not in enforced_df["col_cat"].cat.categories # 'B' was not in input_df


@pytest.mark.categorical
def test_enforce_schema_categorical_with_nan_values():
    """
    Tests _enforce_schema with categorical columns containing NaN values.
    """
    schema = Schema([
        ColSpec(type=DataType.category, name="col_cat")
    ])

    input_df = pd.DataFrame({
        "col_cat": pd.Series(["X", np.nan, "Y"], dtype="category")
    })

    enforced_df = _enforce_schema(input_df, schema)

    assert isinstance(enforced_df, pd.DataFrame)
    assert isinstance(enforced_df["col_cat"].dtype, pd.CategoricalDtype)
    assert enforced_df["col_cat"].isnull().any()
    assert enforced_df["col_cat"].cat.categories.tolist() == ["X", "Y"]


@pytest.mark.categorical
def test_enforce_schema_categorical_type_mismatch_error():
    """
    Tests that _enforce_schema raises an error for incompatible type conversion to categorical.
    """
    schema = Schema([
        ColSpec(type=DataType.category, name="col_cat")
    ])

    # Attempt to convert a numeric column to categorical, which is not a safe conversion
    input_df = pd.DataFrame({
        "col_cat": [1, 2, 3]
    })

    with pytest.raises(MlflowException, match="Incompatible input types for column col_cat"):
        _enforce_schema(input_df, schema)

