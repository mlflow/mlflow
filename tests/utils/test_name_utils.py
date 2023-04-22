import hashlib

from mlflow.utils import name_utils
from mlflow.utils.name_utils import (
    _generate_random_name,
    _generate_unique_integer_id,
    _generate_dataset_name,
)


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


def test_dataset_name_generation():
    """
    This test verifies that the logic used to generate a deterministic name for an MLflow Tracking
    dataset based on the dataset's hash does not change. Be **very** careful if editing this test,
    and do not change the deterministic mapping behavior of _generate_dataset_name; otherwise,
    user workflows may break.
    """
    assert len(name_utils._GENERATOR_DATASET_PREDICATES) == 250
    assert name_utils._GENERATOR_DATASET_PREDICATES == sorted(
        name_utils._GENERATOR_DATASET_PREDICATES
    )
    predicates_hash = hashlib.md5()
    for predicate in name_utils._GENERATOR_DATASET_PREDICATES:
        predicates_hash.update(predicate.encode("utf-8"))
    assert predicates_hash.hexdigest() == "2e538c423ae4073efed38ff592f7c53a"
    assert name_utils._GENERATOR_DATASET_DIGEST_MODULO_DIVISOR == 24999983
    assert name_utils._GENERATOR_DATASET_INTEGER_UPPER_BOUND == 10**5

    assert _generate_dataset_name("bazbar") == "useful-data-5234"
    assert _generate_dataset_name("") == "uplifted-data-35722"
    assert _generate_dataset_name("121232131233124234") == "flawless-data-9118"
    assert _generate_dataset_name("121232131233124235") == "amiable-data-47546"
