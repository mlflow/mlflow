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


tr = mlflow.get_trace(span.trace_id)
my_image_ref = tr.data.spans[0].inputs["my_image"]
print(my_image_ref)  # mlflow-attachments://...
att = Attachment.from_ref(my_image_ref)
