class _DatasetSummary:
    """
    DatasetSummary object.

    This is used to return a list of dataset summaries across one or more experiments in the UI.
    """

    def __init__(self, experiment_id, name, digest, context):
        self._experiment_id = experiment_id
        self._name = name
        self._digest = digest
        self._context = context

    def __eq__(self, other) -> bool:
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

    @property
    def experiment_id(self):
        return self._experiment_id

    @property
    def name(self):
        return self._name

    @property
    def digest(self):
        return self._digest

    @property
    def context(self):
        return self._context

    def to_dict(self):
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "digest": self.digest,
            "context": self.context,
        }
