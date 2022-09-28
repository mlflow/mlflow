import mlflow
import mlflow.pytorch

NUM_EPOCHS = 20
START_STEP = 3


def test_pytorch_autolog_logs_expected_data(tmpdir):
    from torch.utils.tensorboard import SummaryWriter

    mlflow.pytorch.autolog()
    writer = SummaryWriter(str(tmpdir))

    with mlflow.start_run():
        for i in range(NUM_EPOCHS):
            writer.add_scalar("loss", 42.0 + i + START_STEP, global_step=START_STEP + i)

        writer.add_hparams(dict(hparam1=42, hparam2="foo"), dict(final_loss=8))
        writer.close()

    # Checking if metrics are logged.
    client = mlflow.tracking.MlflowClient()
    run_id = client.list_run_infos("0")[0].run_id
    client.set_terminated(run_id)
    metric_history = client.get_metric_history(run_id, "loss")
    assert len(metric_history) == NUM_EPOCHS
    for i, m in enumerate(metric_history, START_STEP):
        assert m.step == i
        assert m.value == 42.0 + i

    run = client.get_run(run_id)
    assert run.data.params == dict(hparam1="42", hparam2="foo")
    assert run.data.metrics == dict(loss=64.0, final_loss=8)

