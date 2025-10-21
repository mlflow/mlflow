import { CopyIcon, Modal, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { CodeSnippet, type CodeSnippetLanguage } from '@databricks/web-shared/snippet';
import { CopyButton } from '@mlflow/mlflow/src/shared/building_blocks/CopyButton';
import { FormattedMessage } from 'react-intl';

interface Props {
  visible: boolean;
  promptName: string;
  promptVersion: string;
  onCancel: () => void;
}

const CodeSnippetWithCopy = ({ code, language }: { code: string; language: CodeSnippetLanguage }) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div css={{ position: 'relative' }}>
      <CopyButton
        css={{ zIndex: 1, position: 'absolute', top: theme.spacing.xs, right: theme.spacing.xs }}
        showLabel={false}
        copyText={code}
        icon={<CopyIcon />}
      />
      <CodeSnippet
        language={language}
        showLineNumbers={false}
        style={{
          padding: theme.spacing.sm,
          color: theme.colors.textPrimary,
          backgroundColor: theme.colors.backgroundSecondary,
          whiteSpace: 'pre-wrap',
        }}
        wrapLongLines
      >
        {code}
      </CodeSnippet>
    </div>
  );
};

export const OptimizeModal = ({ visible, promptName, promptVersion, onCancel }: Props) => {
  const { theme } = useDesignSystemTheme();

  const bashCode = `pip install -U 'mlflow>=3.5.0' 'dspy>=3.0.0' openai`;

  const pythonCode = `import os
from typing import Any
import openai
import mlflow
from mlflow.genai.scorers import Correctness, Safety
from mlflow.genai.optimize import GepaPromptOptimizer
from mlflow.genai import datasets

EVAL_DATASET_NAME='<YOUR DATASET NAME>' # Replace with your dataset
dataset = datasets.get_dataset(EVAL_DATASET_NAME)

# Define your prediction function
def predict_fn(**kwargs) -> str:
    prompt = mlflow.genai.load_prompt("prompts:/${promptName}/${promptVersion}")
    completion = openai.OpenAI().chat.completions.create(
        model="gpt-5-mini", # Replace with your model
        messages=[{"role": "user", "content": prompt.format(**kwargs)}],
    )
    return completion.choices[0].message.content

# Optimize your prompt
result = mlflow.genai.optimize_prompts(
    predict_fn=predict_fn,
    train_data=dataset,
    prompt_uris=["prompts:/${promptName}/${promptVersion}"],
    optimizer=GepaPromptOptimizer(reflection_model="openai:/gpt-5"),
    scorers=[Correctness(model="openai:/gpt-5")), Safety(model="openai:/gpt-5"))],
)

# Open the prompt registry page to check the new prompt
print(f"The new prompt URI: {result.optimized_prompts[0].uri}")`;

  return (
    <Modal
      componentId="mlflow.experiment.prompt.optimize-modal"
      title={<FormattedMessage defaultMessage="Optimize Prompt" description="Title of the optimize prompt modal" />}
      footer={null}
      visible={visible}
      onCancel={onCancel}
    >
      <div style={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
        <Typography.Paragraph>
          <FormattedMessage
            defaultMessage="Here's how to optimize your prompt with your dataset in your Python code:"
            description="Description of how to optimize a prompt with a dataset in Python"
          />
        </Typography.Paragraph>
        <CodeSnippetWithCopy code={bashCode} language="text" />
        <CodeSnippetWithCopy code={pythonCode} language="python" />
        <Typography.Paragraph>
          <FormattedMessage
            defaultMessage="See {mlflowLink} for more details."
            description="Link to MLflow prompt optimization documentation"
            values={{
              mlflowLink: (
                <Typography.Link
                  componentId="mlflow.experiment.prompt.optimize-modal.mlflow-link"
                  target="_blank"
                  href="https://mlflow.org/docs/latest/genai/prompt-registry/optimize-prompts/"
                >
                  <FormattedMessage defaultMessage="MLflow documentation" description="Link to MLflow documentation" />
                </Typography.Link>
              ),
            }}
          />
        </Typography.Paragraph>
      </div>
    </Modal>
  );
};
