import base64
import mlflow
import transformers
import requests


# Acquire an audio file
audio_file = requests.get("https://www.nasa.gov/62283main_landing.wav").content

task = "automatic-speech-recognition"

audio_transcription_pipeline = transformers.pipeline(task=task, model="openai/whisper-small")

# Note that if the input type is of raw binary audio, the generated signature will match the
# one created here. For other supported types (i.e., numpy array of float32 with the
# correct bitrate extraction), a signature is required to override the default of "binary" input
# type.
signature = mlflow.models.infer_signature(
    audio_file,
    mlflow.transformers.generate_signature_output(audio_transcription_pipeline, audio_file),
)

inference_config = {
    "return_timestamps": "word",
    "chunk_length_s": 20,
    "stride_length_s": [5, 3],
}

# Log the pipeline
with mlflow.start_run():
    model_info = mlflow.transformers.log_model(
        transformers_model=audio_transcription_pipeline,
        artifact_path="whisper_transcriber",
        signature=signature,
        input_example=audio_file,
        inference_config=inference_config,
    )

# Load the pipeline in its native format
loaded_transcriber = mlflow.transformers.load_model(model_uri=model_info.model_uri)

transcription = loaded_transcriber(audio_file, **inference_config)

print(transcription)

# Load the pipeline as a pyfunc with the audio file being encoded as base64
pyfunc_transcriber = mlflow.pyfunc.load_model(model_uri=model_info.model_uri)

pyfunc_transcription = pyfunc_transcriber.predict(base64.b64encode(audio_file).decode("ascii"))

# Note: the pyfunc return type if `return_timestamps` is set is a JSON encoded string.
print(pyfunc_transcription)
