# NB: These keys are placeholders and subject to change
class TraceMetadataKey:
    NAME = "name"
    INPUTS = "inputs"
    OUTPUTS = "outputs"
    SOURCE = "source"


class TraceStatusCode:
    """
    Status codes for a span and trace.
    """

    UNSPECIFIED = "TRACE_STATUS_UNSPECIFIED"
    OK = "OK"
    ERROR = "ERROR"


MAX_CHARS_IN_TRACE_INFO_ATTRIBUTE = 300  # TBD
TRUNCATION_SUFFIX = "..."
