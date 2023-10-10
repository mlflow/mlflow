class RunBatch:
    def __init__(self, id, run_id, params, tags, metrics, event) -> None:
        self.id = id
        self.run_id = run_id
        self.params = params
        self.tags = tags
        self.metrics = metrics
        self.event = event
        self.exception = None

    def is_empty(self):
        return len(self.params) == 0 and len(self.tags) == 0 and len(self.metrics) == 0
