import mlflow
import time

def test_log_event_manual(tmp_path):
    with mlflow.start_run():
        mlflow.log_event("custom_event")
        mlflow.log_event("custom_stage", start_time=time.time()-5)
        # Should not raise

def test_log_stage_context(tmp_path):
    with mlflow.start_run():
        with mlflow.log_stage("data_loading"):
            time.sleep(0.1)
        # Should log a stage event
