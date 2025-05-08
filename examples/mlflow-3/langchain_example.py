from langchain_community.chat_models import ChatDatabricks
from langchain_core.prompts import ChatPromptTemplate

import mlflow

# Define the chain
chat_model = ChatDatabricks(
    endpoint="databricks-llama-2-70b-chat",
    temperature=0.1,
    max_tokens=2000,
)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a chatbot that can answer questions about Databricks.",
        ),
        ("user", "{question}"),
    ]
)

chain = prompt | chat_model

# Log the chain with MLflow
model = mlflow.langchain.log_model(
    lc_model=chain,
    name="basic_chain",
    params={"temperature": 0.1, "max_tokens": 2000, "prompt_template": str(prompt)},
    # Specify the model type as "agent"
    model_type="agent",
)
model_id = model.model_id
print("\n")
print(model)

# Trace the chain.
# Note: All of this boilerplate except for `mlflow.langchain.autolog()` will go away shortly (prototyping in progress)
with mlflow.start_span(model_id=model_id) as span:
    mlflow.langchain.autolog()
    inputs = {"question": "What is Unity Catalog?"}
    span.set_inputs(inputs)

    outputs = chain.invoke(inputs)
    span.set_outputs(outputs)

# Fetch the traces by model ID
print(mlflow.search_traces(model_id=model_id)[["request", "response"]])

import pandas as pd

# Start a run to represent the evaluation job
with mlflow.start_run() as evaluation_run:
    # Load the evaluation dataset with MLflow. We will link evaluation metrics to this dataset.
    eval_dataset: mlflow.data.pandas_dataset.PandasDataset = mlflow.data.from_pandas(
        df=pd.DataFrame.from_dict(
            {
                "question": ["Question1", "Question2", "..."],
                "ground_truth": ["Answer1", "Answer2", "..."],
            }
        ),
        name="eval_dataset",
    )

    def mock_evaluate(chain, dataset):
        return {
            "correctness_score": 0.7,
            "toxicity_detected_binary": 0,
        }

    # TODO: Substitute mlflow.evaluate() into this example
    metrics = mock_evaluate(chain, eval_dataset)
    mlflow.log_metrics(
        metrics=metrics,
        dataset=eval_dataset,
        # Specify the ID of the agent logged above
        model_id=model_id,
    )

model = mlflow.get_logged_model(model_id)
# Feedback: it would be nice if the model linked to *all* evaluation runs, not just the source!

model.metrics

evaluation_run = mlflow.get_run(evaluation_run.info.run_id)
print(evaluation_run)
print("\n")
# Feedback: The dataset should also be an input here
print(evaluation_run.inputs)

import torch
import torch.nn.functional as F
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch import nn

import mlflow.pytorch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

all_X, all_Y = load_iris(as_frame=True, return_X_y=True)
all_X["targets"] = all_Y
train, test = train_test_split(all_X)


def prepare_data(X_y):
    X = train_dataset.df.drop(["targets"], axis=1)
    y = train_dataset.df[["targets"]]

    return torch.FloatTensor(X.to_numpy()).to(device), torch.LongTensor(y.to_numpy().flatten()).to(
        device
    )


def compute_accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        predict_out = model(X)
        _, predict_y = torch.max(predict_out, 1)

        return float(accuracy_score(y.cpu(), predict_y.cpu()))


class IrisClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.2)
        x = self.fc3(x)
        return x


model = IrisClassifier()
model = model.to(device)
scripted_model = torch.jit.script(model)  # scripting the model

# Start a run to represent the training job
with mlflow.start_run():
    # Load the training dataset with MLflow. We will link training metrics to this dataset.
    train_dataset: mlflow.data.pandas_dataset.PandasDataset = mlflow.data.from_pandas(
        train, name="train_dataset"
    )
    X_train, y_train = prepare_data(train_dataset.df)

    # Log training job parameters
    mlflow.log_param("num_gpus", 1)
    mlflow.log_param("optimizer", "adam")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(scripted_model.parameters(), lr=0.01)

    for epoch in range(100):
        out = scripted_model(X_train)
        loss = criterion(out, y_train).to(device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            # Log a checkpoint with metrics every 10 epochs
            mlflow.log_metric(
                "accuracy",
                compute_accuracy(scripted_model, X_train, y_train),
                step=epoch,
                dataset=train_dataset,
            )
            mlflow.pytorch.log_model(
                pytorch_model=scripted_model,
                name="torch-iris",
                # "hyperparams=?"
                # Feedback: No need for this, just inherit from the run params!
                params={
                    # Log model parameters
                    "n_layers": 3,
                },
                # Specify the epoch at which the model was logged
                step=epoch,
                # Specify the training dataset with which the metric is associated
                dataset=train_dataset,
                # Feedback: Should support checkpoint TTL, automatically purge checkpoints with lower performance
                # Feedback: Checkpointing for stability (checkpoint every Y mins) vs performance (checkpoint per X epochs + evals)
            )

ranked_checkpoints = mlflow.search_logged_models(
    filter_string="params.n_layers = '3' AND metrics.accuracy > 0",
    order_by=["metrics.accuracy DESC"],
    output_format="list",
)
worst_checkpoint = ranked_checkpoints[-1]
print("WORST CHECKPOINT", worst_checkpoint)

print("\n")

best_checkpoint = ranked_checkpoints[0]
print("BEST CHECKPOINT", best_checkpoint)

# Feedback: Consider renaming `Model` to `Checkpoint`
# perhaps some field on the Model indicating whether its a checkpoint so that we can limit the # of checkpoints
# displayed in the UI by default (e.g. only show the best or most recent ones), automatically TTL the checkpoints,
# would be quite nice

# Start a run to represent the test dataset evaluation job
with mlflow.start_run() as evaluation_run:
    # Load the test dataset with MLflow. We will link test metrics to this dataset.
    test_dataset: mlflow.data.pandas_dataset.PandasDataset = mlflow.data.from_pandas(
        test, name="test_dataset"
    )
    X_test, y_test = prepare_data(test_dataset.df)

    # Load the best checkpoint
    model = mlflow.pytorch.load_model(f"models:/{best_checkpoint.model_id}")
    model = model.to(device)
    scripted_model = torch.jit.script(model)

    # Evaluate the model on the test dataset and log metrics to MLflow
    mlflow.log_metric(
        "accuracy",
        compute_accuracy(scripted_model, X_test, y_test),
        # Specify the ID of the checkpoint to which to link the metrics
        model_id=best_checkpoint.model_id,
        # Specify the test dataset with which the metric is associated
        dataset=test_dataset,
    )

mlflow.get_logged_model(best_checkpoint.model_id)

print([m.to_dictionary() for m in mlflow.get_logged_model(best_checkpoint.model_id).metrics])
