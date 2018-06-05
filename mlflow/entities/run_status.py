class RunStatus(object):
    RUNNING, SCHEDULED, FINISHED, FAILED = range(1, 5)
    _STATUSES = {"RUNNING": RUNNING, "SCHEDULED": SCHEDULED, "FINISHED": FINISHED, "FAILED": FAILED}

    @staticmethod
    def from_string(status_str):
        if status_str not in RunStatus._STATUSES:
            raise Exception("Could not get run status corresponding to string %s. Valid "
                            "run statuses: %s" % (status_str, list(RunStatus._STATUSES.keys())))
        return RunStatus._STATUSES[status_str]
