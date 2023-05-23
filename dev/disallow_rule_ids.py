import argparse
import re
import sys
import tokenize


def extract_comments(f):
    for tok_type, tok, start, _, _ in tokenize.generate_tokens(f.readline):
        if tok_type == tokenize.COMMENT:
            yield tok, start


RULE_ID_REGEX = re.compile(r"[A-Z][0-9_]*")


def is_rule_id(s):
    return bool(RULE_ID_REGEX.fullmatch(s))


DISABLE_COMMENT_REGEX = re.compile(r"pylint:\s+disable=([\w\-,]+)")
DELIMITER_REGEX = re.compile(r"\s*,\s*")


def extract_codes(comment):
    if m := DISABLE_COMMENT_REGEX.search(comment):
        return DELIMITER_REGEX.split(m.group(1))
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*")
    args = parser.parse_args()

    exit_code = 0
    msg = "Use rule name (e.g. unused-variable) instead of rule ID (e.g. W0612)"
    for path in args.files:
        with open(path) as f:
            for comment, (row, col) in extract_comments(f):
                if codes := extract_codes(comment):
                    if any(is_rule_id(c) for c in codes):
                        print(f"{path}:{row}:{col}:", msg)
                        exit_code = 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
