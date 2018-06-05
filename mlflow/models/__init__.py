from datetime import datetime

import yaml


class Model(object):
    """A servable MLflow model, which can support multiple model flavors."""

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
        """Write this Servable as a YAML file to a local file."""
        with open(path, 'w') as out:
            self.to_yaml(out)

    @classmethod
    def load(cls, path):
        """Load a Servable from its YAML representation."""
        with open(path) as f:
            return cls(**yaml.safe_load(f.read()))
