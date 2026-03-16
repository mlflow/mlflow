"""
Tests that the autologging warnings controller correctly preserves
user-installed ``warnings.showwarning`` handlers.

Regression tests for https://github.com/mlflow/mlflow/issues/21689
"""

import warnings

import pytest


def _make_custom_showwarning():
    """Return a distinct callable to use as a custom showwarning handler."""
    def custom_showwarning(message, category, filename, lineno, file=None, line=None):
        pass  # intentional no-op for testing identity

    return custom_showwarning


@pytest.fixture
def warnings_controller():
    """Create a fresh _WarningsController for each test."""
    from mlflow.utils.autologging_utils.logging_and_warnings import _WarningsController

    return _WarningsController()


class TestShowwarningRestoreAfterPatchCycle:
    """Verify that patch/unpatch restores the handler that was active at patch time."""

    def test_custom_handler_restored_after_global_disable_cycle(self, warnings_controller):
        """
        A custom ``warnings.showwarning`` set *after* import must be restored
        when the warnings controller finishes its patch/unpatch cycle.
        """
        custom = _make_custom_showwarning()
        warnings.showwarning = custom

        try:
            # Patch: controller should snapshot `custom`
            warnings_controller.set_mlflow_warnings_disablement_state_globally(disabled=True)
            assert warnings.showwarning != custom, (
                "showwarning should be patched while controller is active"
            )

            # Unpatch: controller should restore `custom`
            warnings_controller.set_mlflow_warnings_disablement_state_globally(disabled=False)
            assert warnings.showwarning is custom, (
                "showwarning should be restored to the user's custom handler"
            )
        finally:
            # Restore default so we don't leak into other tests
            warnings.showwarning = warnings._showwarning_orig

    def test_custom_handler_restored_after_reroute_cycle(self, warnings_controller):
        """Same as above but using the rerouting code path."""
        custom = _make_custom_showwarning()
        warnings.showwarning = custom

        try:
            warnings_controller.set_mlflow_warnings_rerouting_state_globally(rerouted=True)
            assert warnings.showwarning != custom

            warnings_controller.set_mlflow_warnings_rerouting_state_globally(rerouted=False)
            assert warnings.showwarning is custom
        finally:
            warnings.showwarning = warnings._showwarning_orig

    def test_custom_handler_restored_after_thread_disable_cycle(self, warnings_controller):
        """Same as above but using per-thread warning disablement."""
        custom = _make_custom_showwarning()
        warnings.showwarning = custom

        try:
            warnings_controller.set_non_mlflow_warnings_disablement_state_for_current_thread(
                disabled=True
            )
            assert warnings.showwarning != custom

            warnings_controller.set_non_mlflow_warnings_disablement_state_for_current_thread(
                disabled=False
            )
            assert warnings.showwarning is custom
        finally:
            warnings.showwarning = warnings._showwarning_orig

    def test_custom_handler_restored_after_thread_reroute_cycle(self, warnings_controller):
        """Same as above but using per-thread warning rerouting."""
        custom = _make_custom_showwarning()
        warnings.showwarning = custom

        try:
            warnings_controller.set_non_mlflow_warnings_rerouting_state_for_current_thread(
                rerouted=True
            )
            assert warnings.showwarning != custom

            warnings_controller.set_non_mlflow_warnings_rerouting_state_for_current_thread(
                rerouted=False
            )
            assert warnings.showwarning is custom
        finally:
            warnings.showwarning = warnings._showwarning_orig


class TestShowwarningSnapshotTiming:
    """Verify that the snapshot is taken at patch time, not at import time."""

    def test_handler_set_between_import_and_patch_is_preserved(self, warnings_controller):
        """
        The core regression: if a user sets ``warnings.showwarning`` after
        ``import mlflow`` but before autologging activates, that handler must
        survive the patch/unpatch cycle.
        """
        # Simulate the user setting a handler after import
        handler_a = _make_custom_showwarning()
        warnings.showwarning = handler_a

        try:
            # First patch cycle
            warnings_controller.set_mlflow_warnings_disablement_state_globally(disabled=True)
            warnings_controller.set_mlflow_warnings_disablement_state_globally(disabled=False)
            assert warnings.showwarning is handler_a

            # User changes handler again
            handler_b = _make_custom_showwarning()
            warnings.showwarning = handler_b

            # Second patch cycle should snapshot handler_b, not handler_a
            warnings_controller.set_mlflow_warnings_disablement_state_globally(disabled=True)
            warnings_controller.set_mlflow_warnings_disablement_state_globally(disabled=False)
            assert warnings.showwarning is handler_b
        finally:
            warnings.showwarning = warnings._showwarning_orig

    def test_fallback_in_patched_showwarning_uses_snapshot(self, warnings_controller):
        """
        When the patched showwarning falls through to the original handler
        (non-MLflow, non-disabled, non-rerouted warning), it should call the
        handler that was active at patch time.
        """
        call_log = []

        def tracking_showwarning(message, category, filename, lineno, file=None, line=None):
            call_log.append(str(message))

        warnings.showwarning = tracking_showwarning

        try:
            # Enable rerouting for MLflow warnings only (non-MLflow warnings
            # fall through to the original handler)
            warnings_controller.set_mlflow_warnings_rerouted_to_event_logs = True
            warnings_controller._modify_patch_state_if_necessary()

            # Emit a non-MLflow warning -- should go through tracking_showwarning
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn_explicit(
                    "test message",
                    UserWarning,
                    "/some/non/mlflow/path.py",
                    42,
                )

            assert len(call_log) == 1
            assert "test message" in call_log[0]
        finally:
            # Clean up
            warnings_controller.set_mlflow_warnings_rerouted_to_event_logs = False
            warnings_controller._modify_patch_state_if_necessary()
            warnings.showwarning = warnings._showwarning_orig
