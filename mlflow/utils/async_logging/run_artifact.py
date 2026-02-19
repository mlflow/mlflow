import threading
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    import PIL


class RunArtifact:
    def __init__(
        self,
        filename: str,
        artifact_path: str,
        artifact: Union["PIL.Image.Image"],
        completion_event: threading.Event,
    ) -> None:
        """Initializes an instance of `RunArtifacts`.

        Args:
            filename: Filename of the artifact to be logged
            artifact_path: Directory within the run's artifact directory in which to log the
                artifact.
            artifact: The artifact to be logged.
            completion_event: A threading.Event object.
        """
        self.filename = filename
        self.artifact_path = artifact_path
        self.artifact = artifact
        self.completion_event = completion_event
        self._exception = None

    @property
    def exception(self):
        """Exception raised during logging the batch."""
        return self._exception

    @exception.setter
    def exception(self, exception):
        self._exception = exception
