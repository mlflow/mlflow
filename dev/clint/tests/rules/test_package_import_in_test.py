from pathlib import Path

from clint.config import Config
from clint.linter import Location, lint_file
from clint.rules.package_import_in_test import PackageImportInTest


def test_package_import_in_test_function(index_path: Path) -> None:
    """Test that package imports within test functions are flagged."""
    code = """
import sys  # OK - builtin at top level

# Bad - non-builtin package imported in test
def test_func():
    import pandas as pd
    import numpy

# Good - imports at top level
import mlflow
"""
    config = Config(select={PackageImportInTest.name})
    violations = lint_file(Path("test_file.py"), code, config, index_path)
    assert len(violations) == 2
    assert all(isinstance(v.rule, PackageImportInTest) for v in violations)
    # Check line numbers
    assert violations[0].loc == Location(5, 4)
    assert violations[1].loc == Location(6, 4)


def test_package_import_from_in_test_function(index_path: Path) -> None:
    """Test that 'from package import' within test functions are flagged."""
    code = """
import os  # OK - builtin at top level

# Bad - non-builtin package imported in test
def test_func():
    from sklearn import metrics
    from pandas import DataFrame

# Good - imports at top level
from mlflow import log_metric
"""
    config = Config(select={PackageImportInTest.name})
    violations = lint_file(Path("test_file.py"), code, config, index_path)
    assert len(violations) == 2
    assert all(isinstance(v.rule, PackageImportInTest) for v in violations)
    assert violations[0].loc == Location(5, 4)
    assert violations[1].loc == Location(6, 4)


def test_builtin_imports_allowed_in_test(index_path: Path) -> None:
    """Test that builtin module imports are allowed within test functions."""
    code = """
def test_func():
    import os
    import sys
    import json
    from pathlib import Path
    from collections import defaultdict
"""
    config = Config(select={PackageImportInTest.name})
    violations = lint_file(Path("test_file.py"), code, config, index_path)
    assert len(violations) == 0


def test_imports_in_non_test_functions(index_path: Path) -> None:
    """Test that imports in non-test functions are allowed."""
    code = """
def helper_function():
    import pandas as pd
    from sklearn import metrics

def setup_something():
    import numpy as np
"""
    config = Config(select={PackageImportInTest.name})
    violations = lint_file(Path("test_file.py"), code, config, index_path)
    assert len(violations) == 0


def test_no_violation_in_non_test_files(index_path: Path) -> None:
    """Test that package imports in non-test files are not flagged."""
    code = """
def some_function():
    import pandas as pd
    from sklearn import metrics
"""
    config = Config(select={PackageImportInTest.name})
    violations = lint_file(Path("mymodule.py"), code, config, index_path)
    assert len(violations) == 0


def test_imports_at_module_level_in_test_file(index_path: Path) -> None:
    """Test that imports at module level in test files are allowed."""
    code = """
import pandas as pd
import numpy as np
from sklearn import metrics
from mlflow import log_metric

def test_func():
    # Use the imports
    pass
"""
    config = Config(select={PackageImportInTest.name})
    violations = lint_file(Path("test_file.py"), code, config, index_path)
    assert len(violations) == 0


def test_relative_imports_in_test(index_path: Path) -> None:
    """Test that relative imports within test functions are allowed."""
    code = """
def test_func():
    from . import helper
    from .. import utils
    from ...package import module
"""
    config = Config(select={PackageImportInTest.name})
    violations = lint_file(Path("test_file.py"), code, config, index_path)
    assert len(violations) == 0


def test_mixed_imports_in_test(index_path: Path) -> None:
    """Test a mix of builtin and package imports in test functions."""
    code = """
def test_func():
    import os  # OK - builtin
    import sys  # OK - builtin
    import pandas  # Bad - package
    from pathlib import Path  # OK - builtin
    from numpy import array  # Bad - package
"""
    config = Config(select={PackageImportInTest.name})
    violations = lint_file(Path("test_file.py"), code, config, index_path)
    assert len(violations) == 2
    assert all(isinstance(v.rule, PackageImportInTest) for v in violations)
    assert violations[0].loc == Location(4, 4)
    assert violations[1].loc == Location(6, 4)


def test_nested_function_not_considered_test(index_path: Path) -> None:
    """Test that nested functions are not considered test functions."""
    code = """
def test_outer():
    def inner_function():
        import pandas  # Should not be flagged - inner function is not a test
    inner_function()
"""
    config = Config(select={PackageImportInTest.name})
    violations = lint_file(Path("test_file.py"), code, config, index_path)
    assert len(violations) == 0


def test_import_with_alias(index_path: Path) -> None:
    """Test that imports with aliases are properly detected."""
    code = """
def test_func():
    import pandas as pd  # Bad
    import numpy as np  # Bad
    import sys as system  # OK - builtin
"""
    config = Config(select={PackageImportInTest.name})
    violations = lint_file(Path("test_file.py"), code, config, index_path)
    assert len(violations) == 2
    assert violations[0].loc == Location(2, 4)
    assert violations[1].loc == Location(3, 4)


def test_submodule_imports(index_path: Path) -> None:
    """Test that submodule imports are properly detected."""
    code = """
def test_func():
    import sklearn.metrics  # Bad - sklearn is a package
    import os.path  # OK - os is builtin
    from sklearn.metrics import accuracy_score  # Bad
    from os.path import join  # OK - os is builtin
"""
    config = Config(select={PackageImportInTest.name})
    violations = lint_file(Path("test_file.py"), code, config, index_path)
    assert len(violations) == 2
    assert violations[0].loc == Location(2, 4)
    assert violations[1].loc == Location(4, 4)


def test_multiple_test_functions(index_path: Path) -> None:
    """Test that violations are detected across multiple test functions."""
    code = """
def test_first():
    import pandas  # Bad

def test_second():
    from numpy import array  # Bad

def test_third():
    import sys  # OK - builtin
"""
    config = Config(select={PackageImportInTest.name})
    violations = lint_file(Path("test_file.py"), code, config, index_path)
    assert len(violations) == 2
    assert violations[0].loc == Location(2, 4)
    assert violations[1].loc == Location(5, 4)
