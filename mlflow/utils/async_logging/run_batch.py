import threading

from mlflow.entities.metric import Metric
from mlflow.entities.param import Param
from mlflow.entities.run_tag import RunTag


class RunBatch:
    def __init__(
        self,
        run_id: str,
        params: list["Param"] | None = None,
        tags: list["RunTag"] | None = None,
        metrics: list["Metric"] | None = None,
        completion_event: threading.Event | None = None,
    ):
        """Initializes an instance of `RunBatch`.

        Args:
            run_id: The ID of the run.
            params: A list of parameters. Default is None.
            tags: A list of tags. Default is None.
            metrics: A list of metrics. Default is None.
            completion_event: A threading.Event object. Default is None.
        """
        self.run_id = run_id
        self.params = params or []
        self.tags = tags or []
        self.metrics = metrics or []
        self.completion_event = completion_event
        self._exception = None
        self.child_batches = []

    @property
    def exception(self):
        """Exception raised during logging the batch."""
        return self._exception

    @exception.setter
    def exception(self, exception):
        self._exception = exception

    def add_child_batch(self, child_batch):
        """Add a child batch to the current batch.

        This is useful when merging child batches into a parent batch. Child batches are kept so
        that we can properly notify the system when child batches have been processed.
        """
        self.child_batches.append(child_batch)

    def complete(self):
        """Mark the batch as completed."""
        if self.completion_event:
            self.completion_event.set()

        for child_batch in self.child_batches:
            child_batch.complete()
