from datetime import datetime
import json
import logging

import yaml
import os

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.models.signature import ModelSignature
from mlflow.models.utils import save_example, ModelInputExample
from mlflow.utils.file_utils import TempDir

_logger = logging.getLogger(__name__)


class Model(object):
    """
    An MLflow Model that can support multiple model flavors. Provides APIs for implementing
    new Model flavors.
    """

    def __init__(self, artifact_path=None, run_id=None, utc_time_created=None, flavors=None,
                 signature: ModelSignature = None, input_example: ModelInputExample = None,
                 **kwargs):
        # store model id instead of run_id and path to avoid confusion when model gets exported
        if run_id:
            self.run_id = run_id
            self.artifact_path = artifact_path
        self.utc_time_created = str(utc_time_created or datetime.utcnow())
        self.flavors = flavors if flavors is not None else {}
        if signature is not None:
            self.signature = signature
        if input_example is not None:
            self.input_example = input_example
        self.__dict__.update(kwargs)

    def __eq__(self, other):
        if not isinstance(other, Model):
            return False
        return self.__dict__ == other.__dict__

    def add_flavor(self, name, **params):
        """Add an entry for how to serve the model in a given format."""
        self.flavors[name] = params
        return self

    def to_dict(self):
        res = self.__dict__.copy()
        if res.get("signature") is not None:
            res["signature"] = res["signature"].to_dict()
        return res

    def to_yaml(self, stream=None):
        return yaml.safe_dump(self.to_dict(), stream=stream, default_flow_style=False)

    def __str__(self):
        return self.to_yaml()

    def to_json(self):
        return json.dumps(self.to_dict())

    def save(self, path):
        """Write the model as a local YAML file."""
        with open(path, 'w') as out:
            self.to_yaml(out)

    @classmethod
    def load(cls, path):
        """Load a model from its YAML representation."""
        if os.path.isdir(path):
            path = os.path.join(path, "MLmodel")
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f.read()))

    @classmethod
    def from_dict(cls, model_dict):
        """Load a model from its YAML representation."""
        if "signature" in model_dict and isinstance(model_dict["signature"], dict):
            model_dict = model_dict.copy()
            model_dict["signature"] = ModelSignature.from_dict(model_dict["signature"])
        return cls(**model_dict)

    @classmethod
    def log(cls, artifact_path, flavor, registered_model_name=None,
            signature: ModelSignature = None, input_example: ModelInputExample = None,
            **kwargs):
        """
        Log model using supplied flavor module. If no run is active, this method will create a new
        active run.

        :param artifact_path: Run relative path identifying the model.
        :param flavor: Flavor module to save the model with. The module must have
                       the ``save_model`` function that will persist the model as a valid
                       MLflow model.
        :param registered_model_name: Note:: Experimental: This argument may change or be removed
                                      in a future release without warning. If given, create a model
                                      version under ``registered_model_name``, also creating a
                                      registered model if one with the given name does not exist.
        :param signature: Note:: Experimental: This argument may change or be removed in a
                                future release without warning. Model signature describes model
                                input and output schema. Model signature can be inferred from a
                                dataset by calling
                                :py:func:`mlflow.models.signature.infer_signature`
                                or constructed by hand, see
                                :py:class:``mlflow.models.signature.ModelSignature`
        :param input_example: Note:: Experimental: This argument may change or be removed in a
                              future release without warning. Input example provides one or several
                              examples of valid model input. The example can be used as a hint of
                              what data to feed the model. The example is saved using
                              :py:func:`mlflow.model.signatures.save_example`. Exception is raised
                              if save_example call fails.
        :param kwargs: Extra args passed to the model flavor.
        """
        with TempDir() as tmp:
            local_path = tmp.path("model")
            run_id = mlflow.tracking.fluent._get_or_start_run().info.run_id
            mlflow_model = cls(artifact_path=artifact_path, run_id=run_id)
            if signature is not None:
                mlflow_model.signature = signature
            flavor.save_model(path=local_path, mlflow_model=mlflow_model, **kwargs)
            if input_example is not None:
                mlflow_model.input_example = save_example(local_path, input_example)
                mlflow_model.save(os.path.join(local_path, "MLmodel"))
            mlflow.tracking.fluent.log_artifacts(local_path, artifact_path)
            try:
                mlflow.tracking.fluent._record_logged_model(mlflow_model)
            except MlflowException:
                # We need to swallow all mlflow exceptions to maintain backwards compatibility with
                # older tracking servers. Only print out a warning for now.
                _logger.warning(
                    "Logging model metadata to the tracking server has failed, possibly due older "
                    "server version. The model artifacts have been logged successfully under %s. "
                    "In addition to exporting model artifacts, MLflow clients 1.7.0 and above "
                    "attempt to record model metadata to the  tracking store. If logging to a "
                    "mlflow server via REST, consider  upgrading the server version to MLflow "
                    "1.7.0 or above.", mlflow.get_artifact_uri())
            if registered_model_name is not None:
                run_id = mlflow.tracking.fluent.active_run().info.run_id
                mlflow.register_model("runs:/%s/%s" % (run_id, artifact_path),
                                      registered_model_name)
