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


def test_databricks_label_schemas_is_importable():
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
