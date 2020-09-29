import pytest

from mlflow.exceptions import MlflowException
from mlflow.utils.search_models_utils import SearchModelsUtils


def _get_regex_error_message(error_message):
    return r".*{}.*".format(error_message)


@pytest.mark.parametrize(
    "filter_string, parsed_filter",
    [
        ("", []),
        (
            "model.name = 'mod'",
            [{"type": "attribute", "comparator": "=", "key": "name", "value": "mod"}],
        ),
        (
            "name ILIKE '%final new%'",
            [{"type": "attribute", "comparator": "ILIKE", "key": "name", "value": "%final new%"}],
        ),
        (
            "registered_model.name LIKE '%new%'",
            [{"type": "attribute", "comparator": "LIKE", "key": "name", "value": "%new%"}],
        ),
        (
            "tag.`training algorithm` != 'xgboost'",
            [{"type": "tag", "comparator": "!=", "key": "training algorithm", "value": "xgboost"}],
        ),
        (
            '`model_tag`.owner LIKE "%cc"',
            [{"type": "tag", "comparator": "LIKE", "key": "owner", "value": "%cc"}],
        ),
    ],
)
def test_filter_for_registered_model(filter_string, parsed_filter):
    assert SearchModelsUtils.parse_filter_for_registered_models(filter_string) == parsed_filter


@pytest.mark.parametrize(
    "filter_string, parsed_filter",
    [
        ("model_tag.m = 'LR'", [{"type": "tag", "comparator": "=", "key": "m", "value": "LR"}]),
        ('model_tag.m = "LR"', [{"type": "tag", "comparator": "=", "key": "m", "value": "LR"}]),
        ('model_tag.m = "LR"', [{"type": "tag", "comparator": "=", "key": "m", "value": "LR"}]),
        ('tag.m = "L\'Hosp"', [{"type": "tag", "comparator": "=", "key": "m", "value": "L'Hosp"}]),
    ],
)
def test_correct_quote_trimming_for_registered_model(filter_string, parsed_filter):
    assert SearchModelsUtils.parse_filter_for_registered_models(filter_string) == parsed_filter


@pytest.mark.parametrize(
    "filter_string, error_message",
    [
        (
            "tag.owner = 'zero qu'; name = 'some mod'",
            "Search filter contained multiple expression",
        ),
        ("metric.acc >= 0.94", "Invalid entity type"),
        ("model.registeredModelId = 0000", "Invalid attribute key"),
        ("param.model = 'LR'", "Invalid entity type"),
        ("attri.x != 1", "Invalid entity type"),
        ("tag.owner = 'zero qu' OR name = 'some mod'", "Invalid clause\\(s\\) in filter string"),
        ("tag.owner = 'zero qu' NAND name = 'some mod'", "Invalid clause\\(s\\) in filter string"),
        ("tag.owner = 'zero qu' AND (name = 'some mod')", "Invalid clause\\(s\\) in filter string"),
        ("`tag.A = 'B'", "Invalid clause\\(s\\) in filter string"),
        ("tag`.A = 'B'", "Invalid clause\\(s\\) in filter string"),
        ("attribute.ID != '1'", "Invalid attribute key"),
        ("attribute.stage != '1'", "Invalid attribute key"),
        ("attribute.end_time != '1'", "Invalid attribute key"),
        ("attribute.run_id != '1'", "Invalid attribute key"),
        ("attribute.run_uuid != '1'", "Invalid attribute key"),
        ("attribute._status != 'RUNNING'", "Invalid attribute key"),
        ("attribute.run_id != '1'", "Invalid attribute key"),
        ("attribute.run_uuid != '1'", "Invalid attribute key"),
        ("run_id != '1'", "Invalid attribute key"),
        ("stage != '1'", "Invalid attribute key"),
        ("attribute.status = true", "Invalid clause\\(s\\) in filter string"),
    ],
)
def test_error_filter_for_registered_model(filter_string, error_message):
    with pytest.raises(MlflowException, match=_get_regex_error_message(error_message)):
        SearchModelsUtils.parse_filter_for_registered_models(filter_string)


@pytest.mark.parametrize(
    "filter_string, error_message",
    [
        ("tag.acc = 5", "Expected a quoted string value for tag"),
        ("tags.acc = 5", "Expected a quoted string value for tag"),
        (
            "tag.model != tag.model",
            "Parameter value is either not quoted or unidentified quote types",
        ),
        ("1.0 > tag.acc", "Expected 'Identifier' found"),
        ("attribute.name = 1", "Expected a quoted string value for attributes"),
        ("name = 1", "Expected a quoted string value for attributes"),
    ],
)
def test_error_comparison_clauses_for_registered_model(filter_string, error_message):
    with pytest.raises(MlflowException, match=_get_regex_error_message(error_message)):
        SearchModelsUtils.parse_filter_for_registered_models(filter_string)


@pytest.mark.parametrize(
    "filter_string, error_message",
    [
        ("tags.acc = LR", "value is either not quoted or unidentified quote types"),
        ("tags.acc = `LR`", "value is either not quoted or unidentified quote types"),
        ("tags.'acc = LR", "Invalid clause\\(s\\) in filter string"),
        ("tags.acc = 'LR", "Invalid clause\\(s\\) in filter string"),
        ("tags.acc = LR'", "Invalid clause\\(s\\) in filter string"),
        ("tags.acc = \"LR'", "Invalid clause\\(s\\) in filter string"),
        ("tags.acc = = 'LR'", "Invalid clause\\(s\\) in filter string"),
        ("attribute.name IS 'some mod'", "Invalid clause\\(s\\) in filter string"),
        ("name IS 'some mod'", "Invalid clause\\(s\\) in filter string"),
    ],
)
def test_bad_quotes_for_registered_model(filter_string, error_message):
    with pytest.raises(MlflowException, match=_get_regex_error_message(error_message)):
        SearchModelsUtils.parse_filter_for_registered_models(filter_string)


@pytest.mark.parametrize(
    "filter_string, error_message",
    [
        ("tags.acc LR !=", "Invalid clause\\(s\\) in filter string"),
        ("tags.acc LR", "Invalid clause\\(s\\) in filter string"),
        ("tags.acc !=", "Invalid clause\\(s\\) in filter string"),
        ("acc != 1.0", "Invalid attribute key"),
        ("foo is null", "Invalid clause\\(s\\) in filter string"),
        ("1=1", "Expected 'Identifier' found"),
        ("1==2", "Expected 'Identifier' found"),
    ],
)
def test_invalid_clauses_for_registered_model(filter_string, error_message):
    with pytest.raises(MlflowException, match=_get_regex_error_message(error_message)):
        SearchModelsUtils.parse_filter_for_registered_models(filter_string)


@pytest.mark.parametrize(
    "order_by, error_message",
    [
        ("m.acc", "Invalid entity type"),
        ("acc", "Invalid attribute key"),
        ("attri.x", "Invalid entity type"),
        ("`tags.A", "Invalid order_by clause"),
        ("`tags.A`", "Invalid entity type"),
        ("attribute.start", "Invalid attribute key"),
        ("attribute.stage", "Invalid attribute key"),
        ("start", "Invalid order_by clause"),
        ("tags.A != 1", "Invalid order_by clause"),
        ("attribute.name ACS", "Invalid ordering key"),
        ("attribute.name decs", "Invalid ordering key"),
        ("name ACS", "Invalid ordering key"),
        ("name decs", "Invalid ordering key"),
        ("creation_timestamp DESC", "Invalid attribute key"),
        ("last_updated_timestamp DESC blah", "Invalid order_by clause"),
        ("", "Invalid order_by clause"),
        ("timestamp somerandomstuff ASC", "Invalid order_by clause"),
        ("timestamp somerandomstuff", "Invalid order_by clause"),
        ("timestamp decs", "Invalid order_by clause"),
        ("timestamp ACS", "Invalid order_by clause"),
        ("name aCs", "Invalid ordering key"),
    ],
)
def test_invalid_order_by_search_registered_models(order_by, error_message):
    with pytest.raises(MlflowException, match=_get_regex_error_message(error_message)):
        SearchModelsUtils.parse_order_by_for_search_registered_models(order_by)


@pytest.mark.parametrize(
    "order_by, ascending_expected",
    [
        ("tags.`Mean Square Error`", True),
        ("tags.`Mean Square Error` ASC", True),
        ("tags.`Mean Square Error` DESC", False),
    ],
)
def test_space_order_by_search_registered_model(order_by, ascending_expected):
    (
        identifier_type,
        identifier_name,
        ascending,
    ) = SearchModelsUtils.parse_order_by_for_search_registered_models(order_by)
    assert identifier_type == "tag"
    assert identifier_name == "Mean Square Error"
    assert ascending == ascending_expected
