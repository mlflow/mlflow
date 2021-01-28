import numpy as np
import mlflow

from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error


# Enable autologging
mlflow.sklearn.autolog()

# Load data
iris_dataset = load_iris()
data, target, target_names = iris_dataset["data"], iris_dataset[
    "target"], iris_dataset["target_names"]

# Instantiate model
model = GradientBoostingClassifier()

# Split training and validation data
np.random.shuffle(data)
np.random.shuffle(target)
train_x, train_y = data[:100], target[:100]
val_x, val_y = data[100:], target[100:]

# Train and evaluate model
with mlflow.start_run() as run:
    model.fit(train_x, train_y)
print("MSE:", mean_squared_error(model.predict(val_x), val_y))
print("Target names: ", target_names)

# Save model
mlflow.sklearn.save_model(model, "iris_model")