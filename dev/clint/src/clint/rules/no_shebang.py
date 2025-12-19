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
        return file_content.startswith("#!")
