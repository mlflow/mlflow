# Amazon Nova Example

This is a simple example of using MLflow with an Amazon Nova model on Bedrock.

```python
import mlflow
from mlflow.genai import Bedrock

# Set up the Bedrock model
client = Bedrock(model_name="amazon-nova-text")

# Example input
prompt = "Write a short poem about open source."

# Get the response from the model
response = client.generate(prompt)
print(response)
