import transformers
from packaging.version import Version
import requests

import mlflow

# Acquire an audio file
audio = requests.get("https://www.nasa.gov/62283main_landing.wav").content

task = "automatic-speech-recognition"
architecture = "openai/whisper-small"

model = transformers.WhisperForConditionalGeneration.from_pretrained(architecture)
tokenizer = transformers.WhisperTokenizer.from_pretrained(architecture)
feature_extractor = transformers.WhisperFeatureExtractor.from_pretrained(architecture)
if Version(transformers.__version__) > Version("4.30.2"):
    model.generation_config.alignment_heads = [
        [5, 3],
        [5, 9],
        [8, 0],
        [8, 4],
        [8, 7],
        [8, 8],
        [9, 0],
        [9, 7],
        [9, 9],
        [10, 5],
    ]
audio_transcription_pipeline = transformers.pipeline(
    task=task, model=model, tokenizer=tokenizer, feature_extractor=feature_extractor
)

# Note that if the input type is of raw binary audio, the generated signature will match the
# one created here. For other supported types (i.e., numpy array of float32 with the
# correct bitrate extraction), a signature is required to override the default of "binary" input
# type.
signature = mlflow.models.infer_signature(
    audio,
    mlflow.transformers.generate_signature_output(audio_transcription_pipeline, audio),
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
        input_example=audio,
        inference_config=inference_config,
    )

# Load the pipeline in its native format
loaded_transcriber = mlflow.transformers.load_model(model_uri=model_info.model_uri)

transcription = loaded_transcriber(audio, **inference_config)

print(f"\nWhisper native output transcription:\n{transcription}")

# Load the pipeline as a pyfunc with the audio file being encoded as base64
pyfunc_transcriber = mlflow.pyfunc.load_model(model_uri=model_info.model_uri)

pyfunc_transcription = pyfunc_transcriber.predict([audio])

# Note: the pyfunc return type if `return_timestamps` is set is a JSON encoded string.
print(f"\nPyfunc output transcription:\n{pyfunc_transcription}")
