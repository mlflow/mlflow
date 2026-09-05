import numpy as np
import pytest

from mlflow.data.evaluation_dataset import convert_data_to_mlflow_dataset
from mlflow.exceptions import MlflowException


@pytest.mark.parametrize("data", [[[1], [2], [3]], [1, 2, 3]])
def test_convert_list_data_with_numpy_targets(data):
    targets = np.array([1, 2, 3])

    dataset = convert_data_to_mlflow_dataset(
        data=data,
        targets=targets,
    )

    assert np.array_equal(dataset.targets, targets)


def test_convert_list_data_without_targets():
    dataset = convert_data_to_mlflow_dataset(data=[[1], [2], [3]], targets=None)

    assert dataset.targets is None


def test_convert_list_data_with_empty_targets_uses_standard_validation():
    dataset = convert_data_to_mlflow_dataset(data=[[1], [2], [3]], targets=[])

    with pytest.raises(MlflowException, match="same length"):
        dataset.to_evaluation_dataset()


def test_convert_empty_list_uses_standard_evaluation_dataset_validation():
    dataset = convert_data_to_mlflow_dataset(data=[], targets=[1, 2, 3])

    with pytest.raises(MlflowException, match="2-dimensional"):
        dataset.to_evaluation_dataset()
