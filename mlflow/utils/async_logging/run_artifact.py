import posixpath
import tempfile
import threading
from typing import Optional


class RunArtifact:
    def __init__(
        self,
        filename: str,
        artifact_path: str,
        artifact: Optional["PIL.Image.Image"] = None,
        completion_event: Optional[threading.Event] = None,
        local_dir: Optional[str] = None,
    ) -> None:
        """Initializes an instance of `RunArtifacts`.

        Args:
            filename: Real filename of the artifact to be logged
            artifact_path: Directory within the run's artifact directory in which to log the
                artifact.
            artifact (Deprecated): A PIL image to be logged.
            completion_event: A threading.Event object.
            local_dir: Local directory in which the artifact is or will be stored. If not provided,
                a temporary directory will be created and used.
        """
        self._tmpdir = None
        self.artifact_path = artifact_path
        self.filename = filename
        self.image = artifact
        if local_dir is None:
            self._tmpdir = tempfile.TemporaryDirectory()
            local_dir = self._tmpdir.name

        self.local_filepath = posixpath.join(local_dir, filename)
        self._exception = None
        self.completion_event = threading.Event() if completion_event is None else completion_event
        self._handle_image()

    def _handle_image(self):
        if self.image is not None:
            from PIL import Image
            if isinstance(self.image, Image.Image):
                self.image.save(self.local_filepath)

    def close(self):
        """Explicitly cleanup temp resources."""
        if self._tmpdir:
            self._tmpdir.cleanup()

    def __del__(self):
        """
        Fallback cleanup if `close()` wasn't called.
        """
        self.close()

    @property
    def exception(self):
        """Exception raised during logging the batch."""
        return self._exception

    @exception.setter
    def exception(self, exception):
        self._exception = exception
