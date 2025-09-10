import { Button, ChartLineIcon, Modal, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { CodeSnippet } from '@mlflow/mlflow/src/shared/web-shared/snippet';
import { useState } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';

const getCodeSnippet = (experimentId: string) => `import mlflow
from mlflow.genai import datasets, evaluate, scorers

mlflow.set_experiment(experiment_id="${experimentId}")

# Step 1: Define evaluation dataset
eval_dataset = [{
  "inputs": {
    "query": "What is MLflow?",
  }
}]

# Step 2: Define predict_fn
# predict_fn will be called for every row in your evaluation
# dataset. Replace with your app's prediction function.
# NOTE: The **kwargs to predict_fn are the same as the keys of
# the \`inputs\` in your dataset.
def predict(query):
  return query + " an answer"

# Step 3: Run evaluation
evaluate(
  data=eval_dataset,
  predict_fn=predict,
  scorers=scorers.get_all_scorers()
)

# Results will appear back in this UI`;

export const RunEvaluationButton = ({ experimentId }: { experimentId: string }) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const [isOpen, setIsOpen] = useState(false);
  const evalInstructions = (
    <FormattedMessage
      defaultMessage="Run the following code to start an evaluation."
      description="Instructions for running the evaluation code in OSS"
    />
  );
  const evalCodeSnippet = (
    <CodeSnippet theme={theme.isDarkMode ? 'duotoneDark' : 'light'} language="python">
      {getCodeSnippet(experimentId)}
    </CodeSnippet>
  );

  return (
    <>
      <Button componentId="mlflow.eval-runs.start-run-button" icon={<ChartLineIcon />} onClick={() => setIsOpen(true)}>
        <FormattedMessage
          defaultMessage="Run evaluation"
          description="Label for a button that displays instructions for starting a new evaluation run"
        />
      </Button>
      <Modal
        componentId="mlflow.eval-runs.start-run-modal"
        // eslint-disable-next-line formatjs/enforce-description
        title={<FormattedMessage defaultMessage="Run evaluation" />}
        visible={isOpen}
        okText="Discard"
        footer={null}
        onCancel={() => setIsOpen(false)}
      >
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
          <Typography.Text>{evalInstructions}</Typography.Text>
          {evalCodeSnippet}
        </div>
      </Modal>
    </>
  );
};
