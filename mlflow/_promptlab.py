# loader/gateway_loader_module.py
from typing import List, Dict
from string import Template
import pandas as pd
import mlflow.gateway
from mlflow.pyfunc.model import PythonModel

mlflow.gateway.set_gateway_uri("databricks")


class PromptlabModel(PythonModel):
    def __init__(self):
        self.santized_prompt_template = ""
        self.prompt_parameters = {}
        self.python_parameters = {}
        self.model_route = ""

        self.prompt_template = Template(self.santized_prompt_template)

    def predict(self, inputs: pd.DataFrame) -> List[str]:
        results = []
        for idx in inputs.index:
            python_inputs = {
                param.key: inputs["{param.key}"][idx] for param in self.prompt_parameters
            }
            prompt = self.prompt_template.substitute(python_inputs)
            result = mlflow.gateway.query(
                route=self.model_route,
                data={
                    {
                        "prompt": prompt,
                    }.update(self.python_parameters),
                },
            )
            results.append(result["candidates"][0]["text"])
        return results
