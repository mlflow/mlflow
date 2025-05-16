import functools
import warnings

# warning messages
_DATABRICKS_SDK_RETRY_AFTER_SECS_DEPRECATION_WARNING = (
    "The 'retry_after_secs' parameter of DatabricksError is deprecated"
)

# ANSI escape code
ANSI_BASE = "\033["
COLORS = {
    "default_bold": f"{ANSI_BASE}1m",
    "red": f"{ANSI_BASE}31m",
    "red_bold": f"{ANSI_BASE}1;31m",
    "yellow": f"{ANSI_BASE}33m",
    "yellow_bold": f"{ANSI_BASE}1;33m",
    "blue": f"{ANSI_BASE}34m",
    "blue_bold": f"{ANSI_BASE}1;34m",
}
RESET = "\033[0m"


def color_warning(message: str, stacklevel: int, color: str, category: type[Warning] = UserWarning):
    if color in COLORS:
        message = f"{COLORS[color]}{message}{RESET}"

    warnings.warn(
        message=message,
        category=category,
        stacklevel=stacklevel + 1,
    )


def suppress_warnings_containing(pattern):
    """
    Creates a decorator that suppresses warnings containing the specified pattern in their message.

    Args:
        pattern: String to match in warning messages

    Returns:
        A decorator function that will suppress matching warnings
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Use filterwarnings to ignore warnings with the pattern in the message
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=f".*{pattern}.*")
                return func(*args, **kwargs)

        return wrapper

    return decorator
