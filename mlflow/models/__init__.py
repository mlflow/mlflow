"""
The ``mlflow.models`` module provides an API for saving machine learning models in
"flavors" that can be understood by different downstream tools.

The built-in flavors are:

- :py:mod:`mlflow.pyfunc`
- :py:mod:`mlflow.h2o`
- :py:mod:`mlflow.keras`
- :py:mod:`mlflow.pytorch`
- :py:mod:`mlflow.sklearn`
- :py:mod:`mlflow.spark`
- :py:mod:`mlflow.tensorflow`

For details, see `MLflow Models <../models.html>`_.
"""
import os
from abc import abstractmethod, ABCMeta
from datetime import datetime
import json
import logging

import yaml

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.models.signature import ModelSignature, ModelInputExample, \
    save_example
from mlflow.utils.file_utils import TempDir


_logger = logging.getLogger(__name__)


class Model(object):
    """
    An MLflow Model that can support multiple model flavors. Provides APIs for implementing
    new Model flavors.
    """

    def __init__(self, artifact_path=None, run_id=None, utc_time_created=None, flavors=None,
                 signature: ModelSignature=None, input_example: ModelInputExample = None,
                 **kwargs):
        # store model id instead of run_id and path to avoid confusion when model gets exported
        if run_id:
            self.run_id = run_id
            self.artifact_path = artifact_path
        self.utc_time_created = str(utc_time_created or datetime.utcnow())
        self.flavors = flavors if flavors is not None else {}
        self.signature = signature
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
        import os
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
            model_signature: ModelSignature = None, input_example: ModelInputExample=None,
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
        :param model_signature: Note:: Experimental: This argument may change or be removed in a
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
            if model_signature is not None:
                mlflow_model.signature = model_signature
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


class FlavorBackend(object):
    """
        Abstract class for Flavor Backend.
        This class defines the API interface for local model deployment of MLflow model flavors.
    """

    __metaclass__ = ABCMeta

    def __init__(self, config, **kwargs):  # pylint: disable=unused-argument
        self._config = config

    @abstractmethod
    def predict(self, model_uri, input_path, output_path, content_type, json_format):
        """
        Generate predictions using a saved MLflow model referenced by the given URI.
        Input and output are read from and written to a file or stdin / stdout.

        :param model_uri: URI pointing to the MLflow model to be used for scoring.
        :param input_path: Path to the file with input data. If not specified, data is read from
                           stdin.
        :param output_path: Path to the file with output predictions. If not specified, data is
                            written to stdout.
        :param content_type: Specifies the input format. Can be one of {``json``, ``csv``}
        :param json_format: Only applies if ``content_type == json``. Specifies how is the input
                            data encoded in json. Can be one of {``split``, ``records``} mirroring
                            the behavior of Pandas orient attribute. The default is ``split`` which
                            expects dict like data: ``{'index' -> [index], 'columns' -> [columns],
                            'data' -> [values]}``, where index is optional.
                            For more information see
                            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_json.html
        """
        pass

    @abstractmethod
    def serve(self, model_uri, port, host):
        """
        Serve the specified MLflow model locally.

        :param model_uri: URI pointing to the MLflow model to be used for scoring.
        :param port: Port to use for the model deployment.
        :param host: Host to use for the model deployment. Defaults to ``localhost``.
        """
        pass

    def prepare_env(self, model_uri):
        """
        Performs any preparation necessary to predict or serve the model, for example
        downloading dependencies or initializing a conda environment. After preparation,
        calling predict or serve should be fast.
        """
        pass

    @abstractmethod
    def can_score_model(self):
        """
        Check whether this flavor backend can be deployed in the current environment.

        :return: True if this flavor backend can be applied in the current environment.
        """
        pass

    def can_build_image(self):
        """
        :return: True if this flavor has a `build_image` method defined for building a docker
                 container capable of serving the model, False otherwise.
        """
        return callable(getattr(self.__class__, 'build_image', None))
