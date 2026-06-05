from __future__ import annotations

import sys

import click

if sys.platform != "win32":
    import termios
    import tty


def _read_key() -> str:
    """Read a single keystroke (or escape sequence) from stdin in raw mode."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            ch += sys.stdin.read(2)
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def arrow_select(prompt_text: str, options: list[str]) -> int:
    """Render `options` and let the user pick one via arrow keys; return its index.

    Falls back to a numeric prompt on Windows or when stdio isn't a TTY.
    """
    if sys.platform == "win32" or not (sys.stdin.isatty() and sys.stderr.isatty()):
        click.secho(prompt_text, bold=True, err=True)
        for i, opt in enumerate(options, 1):
            click.echo(f"  {click.style(str(i), fg='cyan')}. {opt}", err=True)
        choice = click.prompt(
            click.style("Select", fg="cyan", bold=True),
            type=click.IntRange(1, len(options)),
            default=1,
            err=True,
        )
        return choice - 1

    click.secho(
        f"{prompt_text} (↑/↓ to navigate, Enter to select)",
        fg="cyan",
        bold=True,
        err=True,
    )
    idx = 0
    n = len(options)

    def render() -> None:
        for i, opt in enumerate(options):
            if i == idx:
                click.secho(f"❯ {opt}", fg="cyan", err=True)
            else:
                click.echo(f"  {opt}", err=True)

    def rewind() -> None:
        sys.stderr.write(f"\x1b[{n}A\x1b[J")
        sys.stderr.flush()

    sys.stderr.write("\x1b[?25l")  # hide cursor
    sys.stderr.flush()
    try:
        render()
        while True:
            key = _read_key()
            match key:
                case "\r" | "\n":
                    rewind()
                    click.secho(f"❯ {options[idx]}", fg="green", err=True)
                    return idx
                case "\x03":
                    rewind()
                    raise click.Abort()
                case "\x1b[A" | "k":
                    idx = (idx - 1) % n
                case "\x1b[B" | "j":
                    idx = (idx + 1) % n
                case _:
                    continue
            rewind()
            render()
    finally:
        sys.stderr.write("\x1b[?25h")  # show cursor
        sys.stderr.flush()
