import warnings

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
