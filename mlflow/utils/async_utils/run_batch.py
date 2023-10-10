class RunBatch:
    def __init__(self, id, run_id, params, tags, metrics, event) -> None:
        """
        Initializes an instance of MyClass.

        Args:
            id (int): The ID of the instance.
            run_id (int): The ID of the run.
            params (dict): A dictionary of parameters.
            tags (dict): A dictionary of tags.
            metrics (dict): A dictionary of metrics.
            event (Event): An event object.
        """
        self.id = id
        self.run_id = run_id
        self.params = params
        self.tags = tags
        self.metrics = metrics
        self.event = event
        self.exception = None

    def is_empty(self):
        """
        Returns True if the batch is empty (i.e., contains no parameters, tags, or metrics),
          False otherwise.
        """
        return len(self.params) == 0 and len(self.tags) == 0 and len(self.metrics) == 0
