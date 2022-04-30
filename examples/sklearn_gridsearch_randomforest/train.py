# Import mlflow
import mlflow
import mlflow.sklearn

# Import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Load data
data = load_breast_cancer()
x = data['data']
y = data['target']

# Build model using grid search and cross fold validation
parameters = {'n_estimators': [10, 50, 100], 'criterion': ['gini', 'entropy'], 'max_depth': [3, 5, 10]}
metrics = ['f1', 'recall', 'precision', 'roc_auc', 'neg_log_loss', 'neg_brier_score', 
           'average_precision', 'balanced_accuracy']
rf = RandomForestClassifier(max_depth=2, random_state=0)
clf = GridSearchCV(rf, parameters, scoring=metrics, refit='f1')
clf.fit(x, y)

# Log artifacts to MLflow
mlflow.sklearn.log_model(clf.best_estimator_, "best model")
mlflow.log_metric('best score', clf.best_score_)
for k in clf.best_params_.keys():
    mlflow.log_param(k, clf.best_params_[k])
print("Model saved in run %s" % mlflow.active_run().info.run_uuid)
