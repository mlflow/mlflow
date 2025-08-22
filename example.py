import mlflow
from mlflow.tracing.attachments import Attachment

with mlflow.start_span() as span:
    print(span)
    span.set_inputs(
        {
            "my_image": Attachment.from_file("tests/datasets/cat.png"),
            "my_audio": Attachment.from_file("tests/datasets/apollo11_launch.wav"),
            "my_pdf": Attachment.from_file("mlflow-charter.pdf"),
        }
    )


with mlflow.start_run():
    mlflow.log_artifact("tests/datasets/cat.png")
