import threading

from mlflow.entities.metric import Metric
from mlflow.entities.param import Param
from mlflow.entities.run_tag import RunTag


class RunBatch:
    def __init__(
        self,
        run_id: str,
        params: [Param],
        tags: [RunTag],
        metrics: [Metric],
        completion_event: threading.Event,
    ) -> None:
        """
        Initializes an instance of RunBatch.

        Args:
            run_id: The ID of the run.
            params: A list of parameters.
            tags: A list of tags.
            metrics: A list of metrics.
            completion_event: A threading.Event object.
        """
        self.run_id = run_id
        self.params = params or []
        self.tags = tags or []
        self.metrics = metrics or []
        self.completion_event = completion_event
        self.exception = None

    def is_empty(self):
        """
        Returns True if the batch is empty (i.e., contains no parameters, tags, or metrics),
          False otherwise.
        """
        return len(self.params) == 0 and len(self.tags) == 0 and len(self.metrics) == 0
