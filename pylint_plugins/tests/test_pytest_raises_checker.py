import pytest
import astroid
from pylint.testutils import MessageTest
from pylint.testutils import CheckerTestCase

from pylint_plugins import PytestRaisesChecker


@pytest.fixture(scope="module")
def test_case():
    class TestPytestRaisesWithoutMatch(CheckerTestCase):
        CHECKER_CLASS = PytestRaisesChecker

    test_case = TestPytestRaisesWithoutMatch()
    test_case.setup_method()
    return test_case


def without_raises_bad_cases():
    # Single context manager
    root_node = astroid.extract_node(
        """
with pytest.raises(Exception):
    raise Exception("failed")
"""
    )
    yield root_node, root_node.items[0][0], (2, 5)

    # Multiple context managers
    root_node = astroid.extract_node(
        """
with context_manager, pytest.raises(Exception):
    raise Exception("failed")
"""
    )
    yield root_node, root_node.items[1][0], (2, 22)

    # Without `with`
    root_node = astroid.extract_node(
        """
pytest.raises(Exception)
"""
    )
    yield root_node, root_node, (2, 0)


def without_raises_iter_good_cases():
    # Single context manager
    yield astroid.extract_node(
        """
with pytest.raises(Exception, match="failed"):
    raise Exception("failed")
"""
    )

    # Multiple context managers
    yield astroid.extract_node(
        """
with context_manager, pytest.raises(Exception, match="failed"):
    raise Exception("failed")
"""
    )

    # Without `with`
    yield astroid.extract_node(
        """
pytest.raises(Exception, match="failed")
"""
    )


def test_without_raises(test_case):
    for root_node, error_node, (line, col_offset) in without_raises_bad_cases():
        with test_case.assertAddsMessages(
            MessageTest(
                "pytest-raises-without-match",
                node=error_node,
                line=line,
                col_offset=col_offset,
            )
        ):
            test_case.walk(root_node)

    for root_node in without_raises_iter_good_cases():
        with test_case.assertNoMessages():
            test_case.walk(root_node)


def multiple_statements_bad_cases():
    root_node = astroid.extract_node(
        """
with pytest.raises(Exception, match="failed"):
    a = 1
    b = 2
    assert a == b
"""
    )
    yield root_node, root_node, (2, 0)

    root_node = astroid.extract_node(
        """
with pytest.raises(Exception, match="failed"), mock_patch("module.function"):
    a = 1
    b = 2
    assert a == b
"""
    )
    yield root_node, root_node, (2, 0)


def multiple_statements_good_cases():
    yield astroid.extract_node(
        """
with pytest.raises(Exception, match="failed"):
    function_that_throws()
"""
    )

    yield astroid.extract_node(
        """
with pytest.raises(Exception, match="failed"):
    if condition:
        do_a()
    else:
        do_b()
"""
    )


def test_multiple_statements(test_case):
    for root_node, error_node, (line, col_offset) in multiple_statements_bad_cases():
        with test_case.assertAddsMessages(
            MessageTest(
                "pytest-raises-multiple-statements",
                node=error_node,
                line=line,
                col_offset=col_offset,
            )
        ):
            test_case.walk(root_node)

    for root_node in multiple_statements_good_cases():
        with test_case.assertNoMessages():
            test_case.walk(root_node)
