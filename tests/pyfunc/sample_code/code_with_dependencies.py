import mlflow
from mlflow.models import set_model, set_retriever_schema
from mlflow.pyfunc import PythonModel


class MyModel(PythonModel):
    def _call_retriver(self, id):
        return f"Retriever called with ID: {id}. Output: 42."

    @mlflow.trace
    def predict(self, context, model_input):
        return f"Input: {model_input}. {self._call_retriver(model_input)}"

    @mlflow.trace
    def predict_stream(self, context, model_input, params=None):
        yield f"Input: {model_input}. {self._call_retriver(model_input)}"


set_model(MyModel())
set_retriever_schema(
    primary_key="primary-key",
    text_column="text-column",
    doc_uri="doc-uri",
    other_columns=["column1", "column2"],
)
