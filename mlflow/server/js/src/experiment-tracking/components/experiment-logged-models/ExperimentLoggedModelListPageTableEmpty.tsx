import { Button, DangerIcon, Empty, Modal, Spacer, Typography, useDesignSystemTheme } from '@databricks/design-system';
import versionsEmptyImg from '@mlflow/mlflow/src/common/static/versions-empty.svg';
import { CodeSnippet } from '@databricks/web-shared/snippet';
import { useState } from 'react';
import { FormattedMessage } from 'react-intl';
import { ExperimentKind, getMlflow3DocsLink } from '../../constants';
import { getExperimentKindFromTags } from '../../utils/ExperimentKindUtils';
import { useGetExperimentQuery } from '../../hooks/useExperimentQuery';
import { useParams } from '../../../common/utils/RoutingUtils';
import invariant from 'invariant';

const EXAMPLE_INSTALL_CODE = `pip install -U 'mlflow>=3.1'`;

const getGenAILearnMoreLink = (cloud: 'AWS' | 'GCP' | 'Azure') => {
  return 'https://mlflow.org/docs/latest/genai/prompt-version-mgmt/version-tracking/';
};

const getMLLearnMoreLink = (cloud: 'AWS' | 'GCP' | 'Azure') => {
  return 'https://mlflow.org/docs/latest/ml/mlflow-3/deep-learning/';
};

const getExampleCode = (isGenAIExperiment: boolean, experimentId: string) => {
  if (isGenAIExperiment) {
    return getExampleCodeGenAI(experimentId);
  }
  return getExampleCodeML(experimentId);
};

const getExampleCodeML = (experimentId: string) =>
  `
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn
from mlflow.entities import Dataset

mlflow.set_experiment(experimentid="${experimentId}")

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

const getExampleCodeGenAI = (experimentId: string) =>
  `
import mlflow

mlflow.set_experiment(experiment_id="${experimentId}")

# Define a new GenAI app version, represented as an MLflow LoggedModel
mlflow.set_active_model(name="my-app-v1")

# Log LLM hyperparameters, prompts, and more
mlflow.log_model_params({
    "prompt_template": "My prompt template",
    "llm": "databricks-llama-4-maverick",
    "temperature": 0.2,
})

# Define application code and add MLflow tracing to capture requests and responses.
# (Replace this with your GenAI application or agent code)
@mlflow.trace
def predict(query):
    return f"Response to query: {query}"

# Run your application code. Resulting traces are automatically linked to
# your GenAI app version.
predict("What is MLflow?")
`.trim();

export const ExperimentLoggedModelListPageTableEmpty = ({
  displayShowExampleButton = true,
  isFilteringActive = false,
  badRequestError,
}: {
  displayShowExampleButton?: boolean;
  isFilteringActive?: boolean;
  badRequestError?: Error;
}) => {
  const { theme } = useDesignSystemTheme();
  const { experimentId } = useParams();
  const cloud = 'AWS';

  invariant(experimentId, 'Experiment ID must be defined');

  const [isCodeExampleVisible, setIsCodeExampleVisible] = useState(false);
  const { data: experimentEntity, loading: isExperimentLoading } = useGetExperimentQuery({
    experimentId,
  });
  const experiment = experimentEntity;
  const experimentKind = getExperimentKindFromTags(experiment?.tags);
  const isGenAIExperiment =
    experimentKind === ExperimentKind.GENAI_DEVELOPMENT || experimentKind === ExperimentKind.GENAI_DEVELOPMENT_INFERRED;

  const isEmpty = !badRequestError && !isFilteringActive;

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
      {isEmpty ? (
        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            maxWidth: 'min(100%, 600px)',
            padding: `0px ${theme.spacing.md}px`,
          }}
        >
          <Typography.Title level={3} color="secondary">
            {isGenAIExperiment ? (
              <FormattedMessage
                defaultMessage="Track and compare versions of your GenAI app"
                description="Empty state title displayed when no models are logged in the genai logged models list page"
              />
            ) : (
              <FormattedMessage
                defaultMessage="Track and compare versions of your models"
                description="Empty state title displayed when no models are logged in the machine learning logged models list page"
              />
            )}
          </Typography.Title>
          <Typography.Paragraph color="secondary" css={{ textAlign: 'center' }}>
            {isGenAIExperiment ? (
              <FormattedMessage
                defaultMessage="Track every version of your app's code and prompts to understand how quality changes over time. {learnMoreLink}"
                description="Empty state description displayed when no models are logged in the genai logged models list page"
                values={{
                  learnMoreLink: (
                    <Typography.Link
                      componentId="mlflow.logged_models.list.genai_no_results_learn_more"
                      openInNewTab
                      href={getGenAILearnMoreLink(cloud as 'AWS' | 'GCP' | 'Azure')}
                      css={{ whiteSpace: 'nowrap' }}
                    >
                      <FormattedMessage defaultMessage="Learn more" description="Learn more link text" />
                    </Typography.Link>
                  ),
                }}
              />
            ) : (
              <FormattedMessage
                defaultMessage="Track every version of your model to understand how quality changes over time. {learnMoreLink}"
                description="Empty state description displayed when no models are logged in the machine learning logged models list page"
                values={{
                  learnMoreLink: (
                    <Typography.Link
                      componentId="mlflow.logged_models.list.ml_no_results_learn_more"
                      openInNewTab
                      href={getMLLearnMoreLink(cloud as 'AWS' | 'GCP' | 'Azure')}
                      css={{ whiteSpace: 'nowrap' }}
                    >
                      <FormattedMessage defaultMessage="Learn more" description="Learn more link text" />
                    </Typography.Link>
                  ),
                }}
              />
            )}
          </Typography.Paragraph>
          <img css={{ maxWidth: 'min(100%, 600px)' }} src={versionsEmptyImg} alt="No models found" />
          <div css={{ display: 'flex', gap: theme.spacing.sm, marginTop: theme.spacing.md }}>
            <Button
              componentId="mlflow.logged_models.list.show_example_code"
              onClick={() => setIsCodeExampleVisible(!isCodeExampleVisible)}
            >
              {isGenAIExperiment ? (
                <FormattedMessage
                  defaultMessage="Create version"
                  description="Button for creating a new genai model version"
                />
              ) : (
                <FormattedMessage
                  defaultMessage="Create model version"
                  description="Button for creating a new classic ML model version"
                />
              )}
            </Button>
          </div>
        </div>
      ) : (
        <Empty
          title={
            badRequestError ? (
              <FormattedMessage
                defaultMessage="Request error"
                description="Error state title displayed in the logged models list page"
              />
            ) : (
              <FormattedMessage
                defaultMessage="No models found"
                description="Empty state title displayed when all models are filtered out in the logged models list page"
              />
            )
          }
          description={
            badRequestError ? (
              badRequestError.message
            ) : isFilteringActive ? (
              <FormattedMessage
                defaultMessage="We couldn't find any models matching your search criteria. Try changing your search filters."
                description="Empty state message displayed when all models are filtered out in the logged models list page"
              />
            ) : (
              <FormattedMessage
                defaultMessage="Your models will appear here once you log them using newest version of MLflow. <link>Learn more</link>."
                description="Placeholder for empty models table on the logged models list page"
                values={{
                  link: (chunks) => (
                    <Typography.Link
                      componentId="mlflow.logged_models.list.no_results_learn_more"
                      openInNewTab
                      href={getMlflow3DocsLink()}
                    >
                      {chunks}
                    </Typography.Link>
                  ),
                }}
              />
            )
          }
          image={badRequestError ? <DangerIcon /> : undefined}
          button={
            displayShowExampleButton && !isFilteringActive && !badRequestError ? (
              <Button
                type="primary"
                componentId="mlflow.logged_models.list.show_example_code"
                loading={isExperimentLoading}
                onClick={() => setIsCodeExampleVisible(!isCodeExampleVisible)}
              >
                {isGenAIExperiment ? (
                  <FormattedMessage
                    defaultMessage="Create version"
                    description="Button for creating a new genai model version"
                  />
                ) : (
                  <FormattedMessage
                    defaultMessage="Create model version"
                    description="Button for creating a new classic ML model version"
                  />
                )}
              </Button>
            ) : null
          }
        />
      )}
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
            defaultMessage="Install MLflow 3:"
            description="Instruction for installing MLflow from mlflow-3 branch in log MLflow 3 models"
          />
        </Typography.Text>
        <CodeSnippet language="text">{EXAMPLE_INSTALL_CODE}</CodeSnippet>
        <Spacer size="sm" />
        {isGenAIExperiment ? (
          <FormattedMessage
            defaultMessage="Run example code:"
            description="Instruction for running example GenAI code in order to log MLflow 3 models"
          />
        ) : (
          <FormattedMessage
            defaultMessage="Run example training code:"
            description="Instruction for running example training code in order to log MLflow 3 models"
          />
        )}
        <CodeSnippet language="python">{getExampleCode(isGenAIExperiment, experimentId)}</CodeSnippet>
      </Modal>
    </div>
  );
};
