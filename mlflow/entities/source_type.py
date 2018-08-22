class SourceType(object):
    """Enum describing the originating source of a :py:class:`mlflow.entities.run.Run`."""
    NOTEBOOK, JOB, PROJECT, LOCAL, UNKNOWN = range(1, 6)
