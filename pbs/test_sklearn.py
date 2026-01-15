import sys

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

print(f"Python {sys.version}")  # noqa: T201
print(f"Executable: {sys.executable}")  # noqa: T201

X, y = load_iris(return_X_y=True)
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y)
print(f"Accuracy: {model.score(X, y):.2f}")  # noqa: T201
