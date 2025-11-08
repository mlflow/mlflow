import {
  Button,
  CopyIcon,
  Input,
  Modal,
  LegacyTabPane,
  LegacyTabs,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { CodeSnippet } from '@databricks/web-shared/snippet';
import { CopyButton } from '../../../shared/building_blocks/CopyButton';

type Props = {
  isOpen: boolean;
  closeModal: () => void;
  experimentId: string;
};

const SNIPPET_LINE_HEIGHT = 18;

export const CreateNotebookRunModal = ({ isOpen, closeModal, experimentId }: Props): JSX.Element => {
  const { theme } = useDesignSystemTheme();

  const codeSnippetTheme = theme.isDarkMode ? 'duotoneDark' : 'light';

  const classical_ml_text = `
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

# set the experiment id
mlflow.set_experiment(experiment_id="${experimentId}")

mlflow.autolog()
db = load_diabetes()

X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

# Create and train models.
rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
rf.fit(X_train, y_train)

# Use the model to make predictions on the test dataset.
predictions = rf.predict(X_test)
`.trimStart();

  const llm_text = `
import mlflow
import openai
import os
import pandas as pd

# you must set the OPENAI_API_KEY environment variable
assert (
  "OPENAI_API_KEY" in os.environ
), "Please set the OPENAI_API_KEY environment variable."

# set the experiment id
mlflow.set_experiment(experiment_id="${experimentId}")

system_prompt = (
  "The following is a conversation with an AI assistant."
  + "The assistant is helpful and very friendly."
)

# start a run
mlflow.start_run()
mlflow.log_param("system_prompt", system_prompt)

# Create a question answering model using prompt engineering
# with OpenAI. Log the model to MLflow Tracking
logged_model = mlflow.openai.log_model(
    model="gpt-4o-mini",
    task=openai.chat.completions,
    artifact_path="model",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "{question}"},
    ],
)

# Evaluate the model on some example questions
questions = pd.DataFrame(
    {
        "question": [
            "How do you create a run with MLflow?",
            "How do you log a model with MLflow?",
            "What is the capital of France?",
        ]
    }
)
mlflow.evaluate(
    model=logged_model.model_uri,
    model_type="question-answering",
    data=questions,
)
mlflow.end_run()
`.trimStart();

  const codeSnippetMessage = () => {
    return 'Run this code snippet in a notebook or locally, to create an experiment run';
  };

  // Calculate stable height for the code snippet UI area, based on the line count of the shortest one
  const snippetHeight =
    (Math.min(...[classical_ml_text, llm_text].map((text) => text.split('\n').length)) + 1) * SNIPPET_LINE_HEIGHT;

  return (
    <Modal
      componentId="codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_createnotebookrunmodal.tsx_111"
      visible={isOpen}
      onCancel={closeModal}
      onOk={closeModal}
      footer={
        <div css={{ display: 'flex', gap: theme.spacing.sm, justifyContent: 'flex-end' }}>
          <Button
            componentId="codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_createnotebookrunmodal.tsx_117"
            onClick={closeModal}
            type="primary"
          >
            <FormattedMessage
              defaultMessage="Okay"
              description="Experiment page > new notebook run modal > okay button label"
            />
          </Button>
        </div>
      }
      title={
        <div>
          <Typography.Title level={2} css={{ marginTop: theme.spacing.sm, marginBottom: theme.spacing.xs }}>
            <FormattedMessage
              defaultMessage="New run using notebook"
              description="Experiment page > new notebook run modal > modal title"
            />
          </Typography.Title>
          <Typography.Hint css={{ marginTop: 0, fontWeight: 'normal' }}>{codeSnippetMessage()}</Typography.Hint>
        </div>
      }
    >
      <LegacyTabs>
        <LegacyTabPane
          tab={<FormattedMessage defaultMessage="Classical ML" description="Example text snippet for classical ML" />}
          key="classical-ml"
        >
          <CodeSnippet
            style={{ padding: '5px', height: snippetHeight }}
            language="python"
            theme={codeSnippetTheme}
            actions={
              <div
                style={{
                  marginTop: theme.spacing.sm,
                  marginRight: theme.spacing.md,
                }}
              >
                <CopyButton copyText={classical_ml_text} showLabel={false} icon={<CopyIcon />} />
              </div>
            }
          >
            {classical_ml_text}
          </CodeSnippet>
        </LegacyTabPane>
        <LegacyTabPane
          tab={<FormattedMessage defaultMessage="LLM" description="Example text snippet for LLM" />}
          key="llm"
        >
          <CodeSnippet
            style={{ padding: '5px', height: snippetHeight }}
            language="python"
            theme={codeSnippetTheme}
            actions={
              <div
                style={{
                  marginTop: theme.spacing.sm,
                  marginRight: theme.spacing.md,
                }}
              >
                <CopyButton copyText={llm_text} showLabel={false} icon={<CopyIcon />} />
              </div>
            }
          >
            {llm_text}
          </CodeSnippet>
        </LegacyTabPane>
      </LegacyTabs>
    </Modal>
  );
};

const styles = {
  formItem: { marginBottom: 16 },
};
