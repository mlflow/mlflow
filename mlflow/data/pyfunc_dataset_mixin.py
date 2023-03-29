from mlflow.models.utils import PyFuncInput, PyFuncOutput
from typing import List
from abc import abstractmethod
from dataclasses import dataclass


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
