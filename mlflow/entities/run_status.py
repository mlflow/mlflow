class RunStatus(object):
    RUNNING, SCHEDULED, FINISHED, FAILED = range(1, 5)
    _STRING_TO_STATUS = {"RUNNING": RUNNING, "SCHEDULED": SCHEDULED, "FINISHED": FINISHED, "FAILED": FAILED}
    _STATUS_TO_STRING = {value: key for key, value in _STRING_TO_STATUS.items()}

    @staticmethod
    def from_string(status_str):
        if status_str not in RunStatus._STRING_TO_STATUS:
            raise Exception(
                "Could not get run status corresponding to string %s. Valid run "
                "statuses: %s" % (status_str, list(RunStatus._STRING_TO_STATUS.keys())))
        return RunStatus._STRING_TO_STATUS[status_str]

    @staticmethod
    def to_string(status):
        if status not in RunStatus._STATUS_TO_STRING:
            raise Exception("Could not get string corresponding to run status %s." % status)
        return RunStatus._STATUS_TO_STRING[status]