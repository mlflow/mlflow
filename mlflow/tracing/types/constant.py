# NB: These keys are placeholders and subject to change
class TraceMetadataKey:
    INPUTS = "inputs"
    OUTPUTS = "outputs"
    SOURCE = "source"

class TraceTagKey:
    TRACE_NAME = "mlflow.traceName"

# All storage backends are guaranteed to support key values up to 250 characters
MAX_CHARS_IN_TRACE_INFO_METADATA_AND_TAGS = 250
TRUNCATION_SUFFIX = "..."
