from mlflow.utils.name_utils import _generate_random_name, _generate_unique_integer_id


def test_random_name_generation():
    # Validate exhausted loop truncation
    name = _generate_random_name(max_length=8)
    assert len(name) == 8

    # Validate default behavior while calling 1000 times that names end in integer
    names = [_generate_random_name() for _ in range(1000)]
    assert all(len(name) <= 20 for name in names)
    assert all(name[-1].isnumeric() for name in names)


def test_experiment_id_generation():
    generate_count = 1000000
    generated_values = {_generate_unique_integer_id() for _ in range(generate_count)}

    # validate that in 1 million experiments written to a set, no collisions occur
    assert len(generated_values) == generate_count
