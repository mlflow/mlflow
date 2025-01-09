import warnings

# ANSI escape code
COLORS = {
    "black": 30,
    "red": 31,
    "green": 32,
    "yellow": 33,
    "blue": 34,
    "purple": 35,
    "cyan": 36,
    "white": 37,
    "light_black": 90,
    "light_red": 91,
    "light_green": 92,
    "light_yellow": 93,
    "light_blue": 94,
    "light_purple": 95,
    "light_cyan": 96,
    "light_white": 97,
}


def color_warning(message: str, category: type[Warning], stacklevel: int, color: str):
    RESET = "\033[0m"
    if color in COLORS:
        message = f"\033[{COLORS[color]}m{message}{RESET}"

    warnings.warn(
        message=message,
        category=category,
        stacklevel=stacklevel + 1,
    )
