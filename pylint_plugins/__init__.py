from pylint_plugins.import_checker import ImportChecker
from pylint_plugins.no_rst import NoRstChecker


def register(linter):
    linter.register_checker(ImportChecker(linter))
    linter.register_checker(NoRstChecker(linter))
