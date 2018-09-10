class SourceType(object):
    """Enum for originating source of a :py:class:`mlflow.entities.Run`."""
    NOTEBOOK, JOB, PROJECT, LOCAL, UNKNOWN = range(1, 6)
