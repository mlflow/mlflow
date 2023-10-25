# Location: mlflow/mlflow/tracking/fluent.py:1289
import pytest


@pytest.mark.parametrize('_', [' mlflow/mlflow/tracking/fluent.py:1289 '])
def test(_):
    import mlflow


    def assert_experiment_names_equal(experiments, expected_names):
        actual_names = [e.name for e in experiments if e.name != "Default"]
        assert actual_names == expected_names, (actual_names, expected_names)


    mlflow.set_tracking_uri("sqlite:///:memory:")

    # Create experiments
    for name, tags in [
        ("a", None),
        ("b", None),
        ("ab", {"k": "v"}),
        ("bb", {"k": "V"}),
    ]:
        mlflow.create_experiment(name, tags=tags)

    # Search for experiments with name "a"
    experiments = mlflow.search_experiments(filter_string="name = 'a'")
    assert_experiment_names_equal(experiments, ["a"])

    # Search for experiments with name starting with "a"
    experiments = mlflow.search_experiments(filter_string="name LIKE 'a%'")
    assert_experiment_names_equal(experiments, ["ab", "a"])

    # Search for experiments with tag key "k" and value ending with "v" or "V"
    experiments = mlflow.search_experiments(filter_string="tags.k ILIKE '%v'")
    assert_experiment_names_equal(experiments, ["bb", "ab"])

    # Search for experiments with name ending with "b" and tag {"k": "v"}
    experiments = mlflow.search_experiments(filter_string="name LIKE '%b' AND tags.k = 'v'")
    assert_experiment_names_equal(experiments, ["ab"])

    # Sort experiments by name in ascending order
    experiments = mlflow.search_experiments(order_by=["name"])
    assert_experiment_names_equal(experiments, ["a", "ab", "b", "bb"])

    # Sort experiments by ID in descending order
    experiments = mlflow.search_experiments(order_by=["experiment_id DESC"])
    assert_experiment_names_equal(experiments, ["bb", "ab", "b", "a"])


if __name__ == "__main__":
    test()
