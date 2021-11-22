import json
from unittest import mock

from dev import list_failed_cross_version_tests


class MockResponse:
    def __init__(self, data):
        self.data = data

    def json(self):
        return self.data

    def raise_for_status(self):
        pass


def test_some_jobs_fail(capsys):
    side_effect = map(
        MockResponse,
        [
            {"workflow_runs": [{"id": 0}]},
            {
                "jobs": [
                    {"id": 1, "conclusion": "success"},
                    {"id": 2, "conclusion": "failure"},
                ]
            },
        ],
    )
    with mock.patch("requests.Session.get", side_effect=side_effect):
        list_failed_cross_version_tests.main()
        actual = json.loads(capsys.readouterr().out)
        assert actual == [{"id": 2, "conclusion": "failure"}]


def test_no_jobs_fail(capsys):
    side_effect = map(
        MockResponse,
        [
            {"workflow_runs": [{"id": 0}]},
            {
                "jobs": [
                    {"id": 1, "conclusion": "success"},
                    {"id": 2, "conclusion": "success"},
                ]
            },
        ],
    )
    with mock.patch("requests.Session.get", side_effect=side_effect):
        list_failed_cross_version_tests.main()
        actual = json.loads(capsys.readouterr().out)
        assert actual == []


def test_pagination(capsys):
    jobs = [{"id": i, "conclusion": ("failure" if i % 2 else "success")} for i in range(150)]
    side_effect = map(
        MockResponse,
        [
            {"workflow_runs": [{"id": 0}]},
            {"jobs": jobs[:100]},  # page 1
            {"jobs": jobs[100:]},  # page 2
        ],
    )
    with mock.patch("requests.Session.get", side_effect=side_effect):
        list_failed_cross_version_tests.main()
        actual = json.loads(capsys.readouterr().out)
        expected = [j for j in jobs if j["conclusion"] == "failure"]
        assert actual == expected
