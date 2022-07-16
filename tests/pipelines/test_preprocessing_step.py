# pylint: disable=unused-import
from tests.pipelines.helper_functions import (
    clear_custom_metrics_module_cache,
    tmp_pipeline_exec_path,
    tmp_pipeline_root_path,
    train_and_log_model,
)  # pylint: enable=unused-import


def test_always_pass(tmp_pipeline_root_path, tmp_pipeline_exec_path):
    assert tmp_pipeline_exec_path != tmp_pipeline_exec_path
    assert True
