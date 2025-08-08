import sys
from pathlib import Path

# Mapping of characters to normalize. Start with quotes; extend as needed.
CHAR_MAP = {
    "\u2018": "'",  # left single quotation mark
    "\u2019": "'",  # right single quotation mark
    "\u201c": '"',  # left double quotation mark
    "\u201d": '"',  # right double quotation mark
}


def fix_file(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Non-UTF8 (likely binary) — skip
        return False

    new_text = text
    for bad, good in CHAR_MAP.items():
        new_text = new_text.replace(bad, good)

    if new_text != text:
        path.write_text(new_text, encoding="utf-8")
        return True
    return False


def main(argv: list[str]) -> int:
    changed = 0
    for arg in argv:
        p = Path(arg)
        if p.is_file():
            if fix_file(p):
                changed += 1
    if changed:
        print(f"Normalized characters in {changed} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
