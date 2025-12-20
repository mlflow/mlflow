"""Custom exceptions for agent evaluation workflow.

These exceptions provide more specific error types than generic Exception,
making it easier to handle different error scenarios and provide better
error messages to users.
"""


class AgentEvaluationError(Exception):
    """Base exception for agent evaluation errors.

    Attributes:
        message: Error description
        suggestion: Actionable suggestion for fixing the error
        diagnostic_cmd: Command to run for diagnostics
    """

    def __init__(
        self, message: str, suggestion: str | None = None, diagnostic_cmd: str | None = None
    ):
        self.message = message
        self.suggestion = suggestion
        self.diagnostic_cmd = diagnostic_cmd
        super().__init__(message)

    def __str__(self):
        """Format error message with suggestion and diagnostic command."""
        lines = [f"✗ {self.message}"]

        if self.suggestion:
            lines.append(f"  Suggestion: {self.suggestion}")

        if self.diagnostic_cmd:
            lines.append(f"  Diagnostic: {self.diagnostic_cmd}")

        return "\n".join(lines)


class EnvironmentError(AgentEvaluationError):
    """Environment configuration errors (missing vars, wrong versions)."""

    pass


class AuthenticationError(AgentEvaluationError):
    """Authentication and credential errors."""

    pass


class TracingError(AgentEvaluationError):
    """Tracing integration errors (missing decorators, autolog issues)."""

    pass


class DatasetError(AgentEvaluationError):
    """Dataset creation and loading errors."""

    pass


class DatasetLoadError(DatasetError):
    """Failed to load or parse dataset."""

    pass


class DatasetSchemaError(DatasetError):
    """Invalid dataset schema or format."""

    pass


class ScorerError(AgentEvaluationError):
    """Scorer definition and execution errors."""

    pass


class EvaluationError(AgentEvaluationError):
    """Evaluation execution errors."""

    pass


class LLMProviderError(AgentEvaluationError):
    """LLM provider configuration or API errors."""

    pass
