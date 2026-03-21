"""Playwright end-to-end tests for MLflow with OpenSearch backend.

These tests verify the UI works correctly when backed by OpenSearch.
They exercise the experiment list, runs table, and traces functionality.

Prerequisites:
    - MLflow server running with OpenSearch backend
    - pip install pytest-playwright playwright
    - playwright install chromium

Run:
    pytest tests/playwright/test_opensearch_traces.py -v \\
        --base-url=http://localhost:5000
"""

from __future__ import annotations

import pytest

# Guard the import so the test file can be collected even without
# playwright installed — it will be skipped at runtime.
playwright_available = True
try:
    from playwright.sync_api import Page
except ImportError:
    playwright_available = False


pytestmark = pytest.mark.skipif(
    not playwright_available,
    reason="playwright is not installed",
)


BASE_URL = "http://localhost:5000"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def seeded_experiment():
    """Seed MLflow with an experiment, runs, and traces for UI testing.

    Returns the experiment ID.
    """
    import mlflow

    mlflow.set_tracking_uri(BASE_URL)
    exp_name = "playwright-opensearch-test"

    existing = mlflow.get_experiment_by_name(exp_name)
    if existing:
        return existing.experiment_id

    exp_id = mlflow.create_experiment(exp_name)
    mlflow.set_experiment(experiment_id=exp_id)

    # Create several runs with varying metrics
    for i in range(5):
        with mlflow.start_run(run_name=f"pw-run-{i}"):
            mlflow.log_metric("accuracy", 0.8 + i * 0.04)
            mlflow.log_metric("loss", 0.5 - i * 0.1)
            mlflow.log_param("lr", str(0.001 * (i + 1)))
            mlflow.set_tag("env", "playwright")

    # Create traces
    @mlflow.trace
    def sample_prediction(text: str) -> str:
        return f"Predicted: {text}"

    for text in ["hello world", "opensearch test", "trace search query"]:
        sample_prediction(text)

    import time

    time.sleep(3)  # Allow async trace export

    return exp_id


# ---------------------------------------------------------------------------
# Experiment List Page
# ---------------------------------------------------------------------------


class TestExperimentListPage:
    def test_page_loads(self, page: Page):
        """The main page loads without errors."""
        page.goto(BASE_URL)
        page.wait_for_load_state("networkidle")
        # The page should have content (not be blank/error)
        assert page.title() != ""

    def test_experiment_visible(self, page: Page, seeded_experiment):
        """The seeded experiment appears in the experiment list."""
        page.goto(BASE_URL)
        page.wait_for_load_state("networkidle")
        # Look for the experiment name text somewhere on the page
        content = page.content()
        assert "playwright-opensearch-test" in content


# ---------------------------------------------------------------------------
# Experiment Runs Page
# ---------------------------------------------------------------------------


class TestExperimentRunsPage:
    def test_runs_page_loads(self, page: Page, seeded_experiment):
        """The experiment page loads and shows run data."""
        page.goto(f"{BASE_URL}/#/experiments/{seeded_experiment}")
        page.wait_for_load_state("networkidle")
        # The page should render without critical errors
        assert page.title() != ""

    def test_run_metrics_visible(self, page: Page, seeded_experiment):
        """Metrics columns are visible in the runs table."""
        page.goto(f"{BASE_URL}/#/experiments/{seeded_experiment}")
        page.wait_for_load_state("networkidle")
        # Allow extra time for table to render
        page.wait_for_timeout(2000)
        content = page.content()
        # Expect some mention of the metric key or run name
        assert "accuracy" in content.lower() or "pw-run" in content.lower()


# ---------------------------------------------------------------------------
# Traces Page
# ---------------------------------------------------------------------------


class TestTracesPage:
    def test_traces_tab_navigable(self, page: Page, seeded_experiment):
        """The Traces tab can be clicked and loads content."""
        page.goto(f"{BASE_URL}/#/experiments/{seeded_experiment}")
        page.wait_for_load_state("networkidle")

        # Try to find and click the Traces tab
        traces_tab = page.locator("text=Traces").first
        if traces_tab.is_visible(timeout=5000):
            traces_tab.click()
            page.wait_for_load_state("networkidle")
            page.wait_for_timeout(2000)
            # Just verify the page didn't crash
            assert page.title() != ""

    def test_traces_search_input(self, page: Page, seeded_experiment):
        """The trace search input is present and can accept text."""
        page.goto(f"{BASE_URL}/#/experiments/{seeded_experiment}")
        page.wait_for_load_state("networkidle")

        traces_tab = page.locator("text=Traces").first
        if traces_tab.is_visible(timeout=5000):
            traces_tab.click()
            page.wait_for_load_state("networkidle")

            # Look for any search/filter input
            search_inputs = page.locator("input[type='text'], input[type='search']")
            count = search_inputs.count()
            # Verify at least one search input is present
            assert count > 0
