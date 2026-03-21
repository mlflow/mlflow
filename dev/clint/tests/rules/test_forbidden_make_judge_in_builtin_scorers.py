from pathlib import Path

from clint.config import Config
from clint.linter import lint_file
from clint.rules.forbidden_make_judge_in_builtin_scorers import (
    ForbiddenMakeJudgeInBuiltinScorers,
)


def test_forbidden_make_judge_in_builtin_scorers(index_path: Path) -> None:
    code = """
from mlflow.genai.judges.make_judge import make_judge
from mlflow.genai.judges import InstructionsJudge

# BAD - direct call after import
judge1 = make_judge(name="test", instructions="test")

# BAD - module qualified call
from mlflow.genai import judges
judge2 = judges.make_judge(name="test", instructions="test")

# GOOD - using InstructionsJudge instead
judge3 = InstructionsJudge(name="test", instructions="test")
"""
    config = Config(select={ForbiddenMakeJudgeInBuiltinScorers.name})
    violations = lint_file(Path("builtin_scorers.py"), code, config, index_path)

    # Should detect: 1 import + 2 calls = 3 violations
    assert len(violations) == 3
    assert all(isinstance(v.rule, ForbiddenMakeJudgeInBuiltinScorers) for v in violations)


def test_make_judge_allowed_in_other_files(index_path: Path) -> None:
    code = """
from mlflow.genai.judges.make_judge import make_judge

# GOOD - allowed in other files
judge = make_judge(name="test", instructions="test")
"""
    config = Config(select={ForbiddenMakeJudgeInBuiltinScorers.name})
    violations = lint_file(Path("some_other_file.py"), code, config, index_path)

    # Should NOT trigger in other files
    assert len(violations) == 0


def test_instructions_judge_not_flagged(index_path: Path) -> None:
    code = """
from mlflow.genai.judges import InstructionsJudge

# GOOD - InstructionsJudge is the correct approach
judge = InstructionsJudge(name="test", instructions="test")
"""
    config = Config(select={ForbiddenMakeJudgeInBuiltinScorers.name})
    violations = lint_file(Path("builtin_scorers.py"), code, config, index_path)

    assert len(violations) == 0


def test_nested_make_judge_call(index_path: Path) -> None:
    code = """
from mlflow.genai.judges.make_judge import make_judge

# BAD - nested call
result = some_function(make_judge(name="test", instructions="test"))
"""
    config = Config(select={ForbiddenMakeJudgeInBuiltinScorers.name})
    violations = lint_file(Path("builtin_scorers.py"), code, config, index_path)

    # Should detect: 1 import + 1 call = 2 violations
    assert len(violations) == 2
    assert all(isinstance(v.rule, ForbiddenMakeJudgeInBuiltinScorers) for v in violations)


def test_make_judge_in_comment_not_flagged(index_path: Path) -> None:
    code = """
from mlflow.genai.judges import InstructionsJudge

# This comment mentions make_judge but should not trigger
judge = InstructionsJudge(name="test", instructions="test")
"""
    config = Config(select={ForbiddenMakeJudgeInBuiltinScorers.name})
    violations = lint_file(Path("builtin_scorers.py"), code, config, index_path)

    assert len(violations) == 0
