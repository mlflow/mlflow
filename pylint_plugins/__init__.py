from pylint_plugins.assign_checker import AssignChecker
from pylint_plugins.direct_import_blocker import DirectImportBlocker
from pylint_plugins.import_checker import ImportChecker


def register(linter):
    linter.register_checker(ImportChecker(linter))
    linter.register_checker(AssignChecker(linter))
    linter.register_checker(DirectImportBlocker(linter))
