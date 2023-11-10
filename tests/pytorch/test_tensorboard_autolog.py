import time

import mlflow
import mlflow.pytorch

NUM_EPOCHS = 20
START_STEP = 3


def test_pytorch_autolog_logs_expected_data(tmp_path):
    from torch.utils.tensorboard import SummaryWriter

    mlflow.pytorch.autolog(log_every_n_step=1)
    writer = SummaryWriter(str(tmp_path))

    timestamps = []
    with mlflow.start_run() as run:
        for i in range(NUM_EPOCHS):
            t0 = time.time()
            writer.add_scalar("loss", 42.0 + i + START_STEP, global_step=START_STEP + i)
            t1 = time.time()
            timestamps.append((int(t0 * 1000), int(t1 * 1000)))

        writer.add_hparams({"hparam1": 42, "hparam2": "foo"}, {"final_loss": 8})
        writer.close()

    # Checking if metrics are logged.
    client = mlflow.tracking.MlflowClient()
    metric_history = client.get_metric_history(run.info.run_id, "loss")
    assert len(metric_history) == NUM_EPOCHS
    for i, (m, (t0, t1)) in enumerate(zip(metric_history, timestamps), START_STEP):
        assert m.step == i
        assert m.value == 42.0 + i
        assert t0 <= m.timestamp <= t1

    run = client.get_run(run.info.run_id)
    assert run.data.params == {"hparam1": "42", "hparam2": "foo"}
    assert run.data.metrics == {"loss": 64.0, "final_loss": 8}
