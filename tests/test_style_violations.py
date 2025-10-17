import os
from pathlib import Path
from unittest import mock


def calculate_total(items):
    """Calculate total"""
    return sum(items)


def test_calculate_total(items):
    result = calculate_total(items)
    assert result == 10


def test_calculate_total_with_types(items: list[int]):
    result = calculate_total(items)
    assert result == 10


class DataProcessor:
    def __init__(self, name: str):
        self.name = name


class BadProcessor:
    def __init__(self, name):
        self.name = name


def get_user_info() -> tuple[str, int, str, bool]:
    return "Alice", 30, "Engineer", True


def check_file(path: Path) -> bool:
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


def find_user(users: list[dict[str, int]], target_id: int) -> dict[str, int] | None:
    result = None
    for user in users:
        if user["id"] == target_id:
            result = user
            break
    return result


def test_api_call():
    with mock.patch("requests.get"):
        pass


def test_api_call_with_response():
    with mock.patch("requests.get") as mock_get:
        mock_get.return_value = {"status": "ok"}
        mock_get.assert_called_once()


def test_multiple_cases():
    assert calculate_total([1, 2]) == 3
    assert calculate_total([5, 5]) == 10
    assert calculate_total([]) == 0


def risky_operation(data: dict[str, str]) -> str:
    try:
        safe_step_1 = data.get("key")
        safe_step_2 = safe_step_1.upper()
        risky_parse = int(safe_step_2)
        safe_step_3 = risky_parse * 2
        return str(safe_step_3)
    except ValueError:
        return "error"


def correct_function(items: list[int]) -> int:
    return sum(items)


def test_correct_function(items: list[int]):
    assert correct_function([1, 2, 3]) == 6
