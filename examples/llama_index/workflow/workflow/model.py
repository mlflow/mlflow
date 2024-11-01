from workflow.workflow import HybridRAGWorkflow

import mlflow

# Get model config from ModelConfig singleton (specified via `model_config` parameter when logging the model)
model_config = mlflow.models.ModelConfig()
retrievers = model_config.get("retrievers")

# Create the workflow instance.
workflow = HybridRAGWorkflow(retrievers=retrievers, timeout=300)

# Set the model instance logging. This is mandatory for using model-from-code logging method.
# Refer to https://mlflow.org/docs/latest/models.html#models-from-code for more details.
mlflow.models.set_model(workflow)
