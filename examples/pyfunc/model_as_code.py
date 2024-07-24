# This example demonstrates defining a model directly from code.
# This feature allows for defining model logic within a python script, module, or notebook that is stored
# directly as serialized code, as opposed to object serialization that would otherwise occur when saving
# or logging a model object.
# This script defines the model's logic and specifies which class within the file contains the model code.
# The companion example to this, model_as_code_driver.py, is the driver code that performs the  logging and
# loading of this model definition.
import os

import pandas as pd

import mlflow
from mlflow import pyfunc

assert "OPENAI_API_KEY" in os.environ, "Please set the OPENAI_API_KEY environment variable."


class AIModel(pyfunc.PythonModel):
    @mlflow.trace(name="chain", span_type="CHAIN")
    def predict(self, context, model_input):
        if isinstance(model_input, pd.DataFrame):
            model_input = model_input["input"].tolist()

        responses = []
        for user_input in model_input:
            response = self.get_open_ai_model_response(str(user_input))
            responses.append(response.choices[0].message.content)

        return pd.DataFrame({"response": responses})

    @mlflow.trace(name="open_ai", span_type="LLM")
    def get_open_ai_model_response(self, user_input):
        from openai import OpenAI

        return OpenAI().chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. You are here to provide useful information to the user.",
                },
                {
                    "role": "user",
                    "content": user_input,
                },
            ],
        )


# IMPORTANT: The model code needs to call `mlflow.models.set_model()` to set the model,
# which will be loaded back using `mlflow.pyfunc.load_model` for inference.
mlflow.models.set_model(AIModel())
