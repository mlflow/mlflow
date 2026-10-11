from mlflow.entities import LoggedModel, LoggedModelStatus


def test_logged_model_proto_round_trip_keeps_status_message():
    model = LoggedModel(
        experiment_id="1",
        model_id="m-123",
        name="model",
        artifact_location="file:///tmp/model",
        creation_timestamp=1,
        last_updated_timestamp=2,
        status=LoggedModelStatus.FAILED,
        status_message="Upload failed",
    )

    proto = model.to_proto()
    assert proto.info.status_message == "Upload failed"
    assert LoggedModel.from_proto(proto).status_message == "Upload failed"
