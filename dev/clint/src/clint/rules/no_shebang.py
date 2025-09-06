from clint.rules.base import Rule


class NoShebang(Rule):
    def _message(self) -> str:
        return "Python scripts should not contain shebang lines"

    @staticmethod
    def check(file_content: str) -> bool:
        """
        Returns True if the file contains a shebang line at the beginning.

        A shebang line is a line that starts with '#!' (typically #!/usr/bin/env python).
        """
        if not file_content.strip():
            return False

        first_line = file_content.split("\n")[0]
        return first_line.lstrip().startswith("#!")
