import mlflow
import mlflow.sentence_transformers

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# Log the model using mlflow
with mlflow.start_run():
    logged_model = mlflow.sentence_transformers.log_model(model, "sbert_model")

# Load option 1: mlflow.pyfunc.load_model returns a PyFuncModel
loaded_model = mlflow.pyfunc.load_model(logged_model.model_uri)
embeddings1 = loaded_model.predict(["hello world", "i am mlflow"])

# Load option 2: mlflow.sentence_transformers.load_model returns a SentenceTransformer
loaded_model = mlflow.sentence_transformers.load_model(logged_model.model_uri)
embeddings2 = loaded_model.encode(["hello world", "i am mlflow"])

assert (embeddings1 == embeddings2).all()

print(embeddings1)