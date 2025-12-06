import pytest
from unittest import mock
import sys
import warnings

def test_generate_signature_output_deprecation():
    """
    Verifies that generate_signature_output raises a FutureWarning.
    """
    # Mock transformers and mlflow.transformers.signature to avoid dependencies
    with mock.patch.dict(sys.modules, {
        "transformers": mock.MagicMock(),
        "mlflow.transformers.signature": mock.MagicMock(),
    }):
        # We need to import mlflow.transformers inside the patch if it hasn't been imported yet,
        # or reload it. But since we are in a test, it might be already imported.
        # If it's already imported, patching sys.modules might not affect existing imports in it.
        # However, generate_signature_output imports transformers INSIDE the function.
        # So patching sys.modules['transformers'] should work for that import.
        
        import mlflow.transformers
        
        # Mock the pipeline object
        pipeline_mock = mock.MagicMock()
        # We need to ensure isinstance(pipeline, transformers.Pipeline) passes.
        # Since we mocked transformers, transformers.Pipeline is a MagicMock.
        # We can set the spec of pipeline_mock or just rely on MagicMock behavior if configured.
        # But simpler: just mock isinstance? No.
        # If transformers.Pipeline is a Mock, then isinstance(obj, Mock) checks if obj is a child of Mock? No.
        # It checks if obj.__class__ is Mock.
        
        # Let's just assume the warning is raised BEFORE the check?
        # The warning is at the top of the function.
        # So we don't need to pass the check.
        
        with pytest.warns(FutureWarning, match="The `generate_signature_output` function is deprecated"):
            try:
                mlflow.transformers.generate_signature_output(pipeline_mock, "data")
            except Exception:
                # We expect it might fail after the warning due to mocks or checks
                pass
