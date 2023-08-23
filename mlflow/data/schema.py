from typing import Any, Dict

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.types.schema import Schema


class TensorDatasetSchema:
    """
    Represents the schema of a dataset with tensor features and targets.
    """

    def __init__(self, features: Schema, targets: Schema = None):
        if not isinstance(features, Schema):
            raise MlflowException(
                f"features must be mlflow.types.Schema, got '{type(features)}'",
                INVALID_PARAMETER_VALUE,
            )
        if targets is not None and not isinstance(targets, Schema):
            raise MlflowException(
                f"targets must be either None or mlflow.types.Schema, got '{type(features)}'",
                INVALID_PARAMETER_VALUE,
            )
        self.features = features
        self.targets = targets

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize into a 'jsonable' dictionary.

        :return: dictionary representation of the schema's features and targets (if defined).
        """

        return {
            "mlflow_tensorspec": {
                "features": self.features.to_json(),
                "targets": self.targets.to_json() if self.targets is not None else None,
            },
        }

    @classmethod
    def from_dict(cls, schema_dict: Dict[str, Any]):
        """
        Deserialize from dictionary representation.

        :param schema_dict: Dictionary representation of model signature.
                            Expected dictionary format:
                            `{'features': <json string>, 'targets': <json string>" }`

        :return: TensorDatasetSchema populated with the data from the dictionary.
        """
        if "mlflow_tensorspec" not in schema_dict:
            raise MlflowException(
                "TensorDatasetSchema dictionary is missing expected key 'mlflow_tensorspec'",
                INVALID_PARAMETER_VALUE,
            )

        schema_dict = schema_dict["mlflow_tensorspec"]
        features = Schema.from_json(schema_dict["features"])
        if "targets" in schema_dict and schema_dict["targets"] is not None:
            targets = Schema.from_json(schema_dict["targets"])
            return cls(features, targets)
        else:
            return cls(features)

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, TensorDatasetSchema)
            and self.features == other.features
            and self.targets == other.targets
        )

    def __repr__(self) -> str:
        return f"features:\n  {self.features!r}\ntargets:\n  {self.targets!r}\n"
