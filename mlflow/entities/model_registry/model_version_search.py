from mlflow.entities.model_registry import ModelVersion


class ModelVersionSearch(ModelVersion):
    def __init__(self, *args, **kwargs):
        kwargs["tags"] = []
        kwargs["aliases"] = []
        super().__init__(*args, **kwargs)

    def tags(self):
        raise Exception(
            "UC Model Versions gathered through search_model_versions do not have tags. "
            "Please use get_model_version to obtain an individual version's tags."
        )

    def aliases(self):
        raise Exception(
            "UC Model Versions gathered through search_model_versions do not have aliases. "
            "Please use get_model_version to obtain an individual version's aliases."
        )

    def __eq__(self, other):
        if type(other) in {type(self), ModelVersion}:
            return self.__dict__ == other.__dict__
        return False
