import mlflow
from mlflow.tracing.attachments import Attachment

with mlflow.start_span() as span:
    # Attach files to span inputs
    span.set_inputs(
        {
            # From file path - automatically detects content type
            "image": Attachment.from_file("tests/datasets/cat.png"),
            "audio": Attachment.from_file("tests/datasets/apollo11_launch.wav"),
            # From bytes with explicit content type
            "document": Attachment(content_type="application/pdf", content_bytes=b"%PDF-1.4..."),
        }
    )

    # Attach files to span outputs
    span.set_outputs(
        {
            "analysis_report": Attachment(
                content_type="text/markdown", content_bytes=b"# Analysis Report\n..."
            ),
        }
    )


trace = mlflow.get_trace(span.trace_id)
image_ref = trace.data.spans[0].inputs["image"]
print(image_ref)  # mlflow-attachments://<id>?content_type=image/png&trace_id=tr-xxx

# Download attachment
image_attachment = Attachment.from_ref(image_ref)
image_data = image_attachment.content_bytes
image_type = image_attachment.content_type

print(image_data[:20])
