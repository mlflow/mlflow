import threading


class RunArtifacts:
    def __init__(
        self,
        local_file: str,
        artifact_path: str,
        cleanup: bool,
        completion_event: threading.Event,
    ) -> None:
        """Initializes an instance of `RunArtifacts`.

        Args:
            local_file: Path to artifact to log.
            artifact_path: Directory within the run's artifact directory in which to log the
                artifact.
            cleanup: Indicator of whether to cleanup local file after upload.
            completion_event: A threading.Event object.
        """
        self.local_file = local_file
        self.artifact_path = artifact_path
        self.cleanup = cleanup
        self.completion_event = completion_event
        self._exception = None

    @property
    def exception(self):
        """Exception raised during logging the batch."""
        return self._exception

    @exception.setter
    def exception(self, exception):
        self._exception = exception
