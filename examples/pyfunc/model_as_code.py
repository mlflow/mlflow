# This is the example model as code that are logged as code using
# the mlflow.pyfunc.log_model API.
# The model_as_code_driver.py is the driver code that logs the model as code
# and loads the model back using mlflow.pyfunc.load_model.
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
            model="gpt-3.5-turbo",
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


# Using mlflow.models.set_model, set the model that will loaded back
# using mlflow.pyfunc.load_model, eventually used for inference.
mlflow.models.set_model(AIModel())
