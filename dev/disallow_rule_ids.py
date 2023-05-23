import argparse
import re
import sys
import tokenize


def iter_comments(f):
    for tok_type, tok, start, _, _ in tokenize.generate_tokens(f.readline):
        if tok_type == tokenize.COMMENT:
            yield tok, start


RULE_ID_REGEX = re.compile(r"[A-Z][0-9]*")


def is_rule_id(s):
    return bool(RULE_ID_REGEX.fullmatch(s))


DISABLE_COMMENT_REGEX = re.compile(r"pylint:\s+disable=([\w\s\-,]+)")
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
            for comment, (row, col) in iter_comments(f):
                if codes := extract_codes(comment):
                    if code := next(filter(is_rule_id, codes), None):
                        print(f"{path}:{row}:{col}: {code}: {msg}")
                        exit_code = 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
