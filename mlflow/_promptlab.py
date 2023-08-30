# loader/gateway_loader_module.py
from typing import List, Dict
from string import Template
import pandas as pd
import mlflow.gateway

mlflow.gateway.set_gateway_uri("databricks")


class GatewayModel:
    def __init__(self):
        # read this from json files
        self.santized_prompt_template = ""
        self.prompt_parameters = ""
        self.python_parameters = {}
        self.model_routes = ""
        self.prompt_template = Template(
            """
__sanitized_python_template__
"""
        )

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


def _load_pyfunc(model_path):
    return GatewayModel(model_path)
