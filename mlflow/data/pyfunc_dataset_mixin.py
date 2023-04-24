from mlflow.models.utils import PyFuncInput, PyFuncOutput
from typing import List
from abc import abstractmethod
from dataclasses import dataclass
from mlflow.models.evaluation.base import EvaluationDataset


@dataclass
class PyFuncInputsOutputs:
    inputs: List[PyFuncInput]
    outputs: List[PyFuncOutput] = None


class PyFuncConvertibleDatasetMixin:
    @abstractmethod
    def to_pyfunc(self) -> PyFuncInputsOutputs:
        """
        Converts the dataset to a collection of pyfunc inputs and outputs for model
        evaluation. Required for use with mlflow.evaluate().
        May not be implemented by all datasets.
        """

    @abstractmethod
    def to_evaluation_dataset(self, path=None, feature_names=None) -> EvaluationDataset:
        """
        Converts the dataset to an EvaluationDataset for model evaluation.
        May not be implemented by all datasets.
        """
