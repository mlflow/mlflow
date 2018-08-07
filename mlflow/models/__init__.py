from datetime import datetime

import yaml


import mlflow
from mlflow.utils.file_utils import TempDir


class Model(object):
    """A MLflow model that can support multiple model flavors."""

    def __init__(self, artifact_path=None, run_id=None, utc_time_created=datetime.utcnow(),
                 flavors=None):
        # store model id instead of run_id and path to avoid confusion when model gets exported
        if run_id:
            self.run_id = run_id
            self.artifact_path = artifact_path
        self.utc_time_created = str(utc_time_created)
        self.flavors = flavors if flavors is not None else {}

    def add_flavor(self, name, **params):
        """Add an entry for how to serve the model in a given format."""
        self.flavors[name] = params
        return self

    def to_yaml(self, stream=None):
        return yaml.safe_dump(self.__dict__, stream=stream, default_flow_style=False)

    def save(self, path):
        """Write this model as a YAML file to a local file."""
        with open(path, 'w') as out:
            self.to_yaml(out)

    @classmethod
    def load(cls, path):
        """Load a model from its YAML representation."""
        with open(path) as f:
            return cls(**yaml.safe_load(f.read()))

    @classmethod
    def log(cls, artifact_path, flavor, **kwargs):
        """
        Log model using supplied flavor module.

        :param artifact_path: Run relative path identifying this model.
        :param flavor: Flavor module / object to save the model with. The module / object must have
            the ``save_model`` function that will persist the model as a valid MLflow model.
        :param kwargs: Extra args passed to the model flavor.
        """
        with TempDir() as tmp:
            local_path = tmp.path("model")
            run_id = mlflow.tracking._get_or_start_run().run_info.run_uuid
            mlflow_model = cls(artifact_path=artifact_path, run_id=run_id)
            flavor.save_model(path=local_path, mlflow_model=mlflow_model, **kwargs)
            mlflow.tracking.log_artifacts(local_path, artifact_path)
