import { Button, Empty, Modal, Spacer, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { CodeSnippet } from '@databricks/web-shared/snippet';
import { useState } from 'react';
import { FormattedMessage } from 'react-intl';

const DOCS_LINK =
  'https://docs.google.com/document/d/1bjvwiEckOEMKTupt8ZxAmZLv7Es6Vy0BaAfFbjI1eYc/edit?tab=t.0#heading=h.yq2j4iyu8yqa';

const EXAMPLE_INSTALL_CODE = `pip install git+https://github.com/mlflow/mlflow@mlflow-3`;
const EXAMPLE_CODE = `
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn
from mlflow.entities import Dataset

# Helper function to compute metrics
def compute_metrics(actual, predicted):
    rmse = mean_squared_error(actual, predicted) 
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    return rmse, mae, r2

# Load Iris dataset and prepare the DataFrame
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['quality'] = (iris.target == 2).astype(int)  # Create a binary target for simplicity

# Split into training and testing datasets
train_df, test_df = train_test_split(iris_df, test_size=0.2, random_state=42)

# Start a run to represent the training job
with mlflow.start_run() as training_run:
    # Load the training dataset with MLflow. We will link training metrics to this dataset.
    train_dataset: Dataset = mlflow.data.from_pandas(train_df, name="train")
    train_x = train_dataset.df.drop(["quality"], axis=1)
    train_y = train_dataset.df[["quality"]]

    # Fit a model to the training dataset
    lr = ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42)
    lr.fit(train_x, train_y)

    # Log the model, specifying its ElasticNet parameters (alpha, l1_ratio)
    # As a new feature, the LoggedModel entity is linked to its name and params
    model_info = mlflow.sklearn.log_model(
        sk_model=lr,
        name="elasticnet",
        params={
            "alpha": 0.5,
            "l1_ratio": 0.5,
        },
        input_example = train_x
    )

    # Inspect the LoggedModel and its properties
    logged_model = mlflow.get_logged_model(model_info.model_id)
    print(logged_model.model_id, logged_model.params)

    # Evaluate the model on the training dataset and log metrics
    # These metrics are now linked to the LoggedModel entity
    predictions = lr.predict(train_x)
    (rmse, mae, r2) = compute_metrics(train_y, predictions)
    mlflow.log_metrics(
        metrics={
            "rmse": rmse,
            "r2": r2,
            "mae": mae,
        },
        model_id=logged_model.model_id,
        dataset=train_dataset
    )

    # Inspect the LoggedModel, now with metrics
    logged_model = mlflow.get_logged_model(model_info.model_id)
    print(logged_model.model_id, logged_model.metrics)`.trim();

export const ExperimentLoggedModelListPageTableEmpty = ({
  displayShowExampleButton = true,
}: {
  displayShowExampleButton?: boolean;
}) => {
  const { theme } = useDesignSystemTheme();

  const [isCodeExampleVisible, setIsCodeExampleVisible] = useState(false);

  return (
    <div
      css={{
        inset: 0,
        top: theme.general.heightBase + theme.spacing.lg,
        position: 'absolute',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        minHeight: 160,
      }}
    >
      <Empty
        title={
          <FormattedMessage
            defaultMessage="No models logged"
            description="Placeholder for empty models table on the logged models list page"
          />
        }
        description={
          <FormattedMessage
            defaultMessage="Your models will appear here once you log them using newest version of MLflow. <link>Learn more</link>."
            description="Placeholder for empty models table on the logged models list page"
            values={{
              link: (chunks) => (
                <Typography.Link
                  componentId="mlflow.logged_models.list.no_results_learn_more"
                  openInNewTab
                  href={DOCS_LINK}
                >
                  {chunks}
                </Typography.Link>
              ),
            }}
          />
        }
        button={
          displayShowExampleButton ? (
            <Button
              type="primary"
              componentId="mlflow.logged_models.list.show_example_code"
              onClick={() => setIsCodeExampleVisible(!isCodeExampleVisible)}
            >
              <FormattedMessage
                defaultMessage="Show example code"
                description="Button for showing logged models quickstart example code"
              />
            </Button>
          ) : null
        }
      />
      <Modal
        size="wide"
        visible={isCodeExampleVisible}
        onCancel={() => setIsCodeExampleVisible(false)}
        title={
          <FormattedMessage
            defaultMessage="Example code"
            description="Title of the modal with the logged models quickstart example code"
          />
        }
        componentId="mlflow.logged_models.list.example_code_modal"
        okText={
          <FormattedMessage
            defaultMessage="Close"
            description="Button for closing modal with the logged models quickstart example code"
          />
        }
        onOk={() => setIsCodeExampleVisible(false)}
      >
        <Typography.Text>
          <FormattedMessage
            defaultMessage="Install <code>mlflow</code> from <code>mlflow-3</code> branch:"
            description="Instruction for installing MLflow from mlflow-3 branch in log MLflow 3 models"
            values={{ code: (chunks) => <code>{chunks}</code> }}
          />
        </Typography.Text>
        <CodeSnippet language="text">{EXAMPLE_INSTALL_CODE}</CodeSnippet>
        <Spacer size="sm" />
        <FormattedMessage
          defaultMessage="Run example training code:"
          description="Instruction for running example training code in order to log MLflow 3 models"
        />
        <CodeSnippet language="python">{EXAMPLE_CODE}</CodeSnippet>
      </Modal>
    </div>
  );
};
