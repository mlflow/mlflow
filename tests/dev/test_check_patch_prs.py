import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add dev directory to path to import the module
dev_dir = Path(__file__).resolve().parents[2] / "dev"
sys.path.insert(0, str(dev_dir))

from check_patch_prs import Commit, get_commits


def test_commit_dataclass_has_date_field():
    commit = Commit(sha="abc123", pr_num=123, date="2024-01-01T00:00:00Z")
    assert commit.sha == "abc123"
    assert commit.pr_num == 123
    assert commit.date == "2024-01-01T00:00:00Z"


def test_get_commits_sorts_by_date_oldest_first():
    with (
        patch("check_patch_prs.get_commit_count") as mock_get_commit_count,
        patch("check_patch_prs.requests.get") as mock_get,
    ):
        # Setup mock for commit count
        mock_get_commit_count.return_value = 3

        # Create mock response with commits in non-chronological order
        mock_response = Mock()
        mock_response.json.return_value = [
            {
                "sha": "commit3",
                "commit": {
                    "message": "Third commit (#103)",
                    "committer": {"date": "2024-01-03T00:00:00Z"},
                },
            },
            {
                "sha": "commit1",
                "commit": {
                    "message": "First commit (#101)",
                    "committer": {"date": "2024-01-01T00:00:00Z"},
                },
            },
            {
                "sha": "commit2",
                "commit": {
                    "message": "Second commit (#102)",
                    "committer": {"date": "2024-01-02T00:00:00Z"},
                },
            },
        ]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Call the function
        commits = get_commits("test-branch")

        # Verify commits are sorted by date (oldest first)
        assert len(commits) == 3
        assert commits[0].sha == "commit1"
        assert commits[0].pr_num == 101
        assert commits[0].date == "2024-01-01T00:00:00Z"

        assert commits[1].sha == "commit2"
        assert commits[1].pr_num == 102
        assert commits[1].date == "2024-01-02T00:00:00Z"

        assert commits[2].sha == "commit3"
        assert commits[2].pr_num == 103
        assert commits[2].date == "2024-01-03T00:00:00Z"


def test_get_commits_handles_multiple_pages():
    with (
        patch("check_patch_prs.get_commit_count") as mock_get_commit_count,
        patch("check_patch_prs.requests.get") as mock_get,
    ):
        # Setup mock for commit count (200 commits = 2 pages with per_page=100)
        mock_get_commit_count.return_value = 200

        # Create mock responses for two pages with dates out of order
        def create_response(commits_data):
            mock_response = Mock()
            mock_response.json.return_value = commits_data
            mock_response.raise_for_status.return_value = None
            return mock_response

        # Page 1: Latest commits (returned first by API)
        page1_data = [
            {
                "sha": "commit_page1",
                "commit": {
                    "message": "Page 1 commit (#201)",
                    "committer": {"date": "2024-01-05T00:00:00Z"},
                },
            }
        ]

        # Page 2: Older commits (returned second by API)
        page2_data = [
            {
                "sha": "commit_page2",
                "commit": {
                    "message": "Page 2 commit (#202)",
                    "committer": {"date": "2024-01-01T00:00:00Z"},
                },
            }
        ]

        # Mock requests.get to return different responses for each page
        mock_get.side_effect = [create_response(page1_data), create_response(page2_data)]

        # Call the function
        commits = get_commits("test-branch")

        # Verify commits are sorted by date (oldest first), regardless of page order
        assert len(commits) == 2
        assert commits[0].sha == "commit_page2"  # Older commit from page 2 comes first
        assert commits[0].date == "2024-01-01T00:00:00Z"
        assert commits[1].sha == "commit_page1"  # Newer commit from page 1 comes second
        assert commits[1].date == "2024-01-05T00:00:00Z"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
