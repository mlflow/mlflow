from sentence_transformers import SentenceTransformer

import mlflow
import mlflow.sentence_transformers

model = SentenceTransformer("all-MiniLM-L6-v2")

example_sentences = ["This is a sentence.", "This is another sentence."]

# Define the signature
signature = mlflow.models.infer_signature(
    model_input=example_sentences,
    model_output=model.encode(example_sentences),
)

# Log the model using mlflow
with mlflow.start_run():
    logged_model = mlflow.sentence_transformers.log_model(
        model=model,
        name="sbert_model",
        signature=signature,
        input_example=example_sentences,
    )

# Load option 1: mlflow.pyfunc.load_model returns a PyFuncModel
loaded_model = mlflow.pyfunc.load_model(logged_model.model_uri)
embeddings1 = loaded_model.predict(["hello world", "i am mlflow"])

# Load option 2: mlflow.sentence_transformers.load_model returns a SentenceTransformer
loaded_model = mlflow.sentence_transformers.load_model(logged_model.model_uri)
embeddings2 = loaded_model.encode(["hello world", "i am mlflow"])

print(embeddings1)

"""
>> [[-3.44772562e-02  3.10232025e-02  6.73496164e-03  2.61089969e-02
  ...
  2.37922110e-02 -2.28897743e-02  3.89375277e-02  3.02067865e-02]
 [ 4.81191138e-03 -9.33756605e-02  6.95968643e-02  8.09735525e-03
  ...
   6.57437667e-02 -2.72239652e-02  4.02687863e-02 -1.05599344e-01]]
"""
