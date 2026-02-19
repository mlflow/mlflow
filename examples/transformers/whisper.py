import requests
import transformers

import mlflow

# Acquire an audio file
resp = requests.get(
    "https://github.com/mlflow/mlflow/raw/master/tests/datasets/apollo11_launch.wav"
)
resp.raise_for_status()
audio = resp.content

task = "automatic-speech-recognition"
architecture = "openai/whisper-tiny"

model = transformers.WhisperForConditionalGeneration.from_pretrained(architecture)
# workaround for https://github.com/huggingface/transformers/issues/37172
model.generation_config.input_ids = model.generation_config.forced_decoder_ids
model.generation_config.forced_decoder_ids = None

tokenizer = transformers.WhisperTokenizer.from_pretrained(architecture)
feature_extractor = transformers.WhisperFeatureExtractor.from_pretrained(architecture)
model.generation_config.alignment_heads = [[2, 2], [3, 0], [3, 2], [3, 3], [3, 4], [3, 5]]
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
    "return_timestamps": False,
    "chunk_length_s": 20,
    "stride_length_s": [5, 3],
}

# Log the pipeline
with mlflow.start_run():
    model_info = mlflow.transformers.log_model(
        transformers_model=audio_transcription_pipeline,
        name="whisper_transcriber",
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
