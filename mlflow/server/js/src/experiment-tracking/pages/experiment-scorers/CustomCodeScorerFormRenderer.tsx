import React from 'react';
import { useDesignSystemTheme, Typography, CopyIcon } from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { CodeSnippet } from '@databricks/web-shared/snippet';
import { CopyButton } from '@mlflow/mlflow/src/shared/building_blocks/CopyButton';
import { useWatch, type Control } from 'react-hook-form';
import type { SCORER_TYPE } from './constants';
import { COMPONENT_ID_PREFIX, type ScorerFormMode, SCORER_FORM_MODE } from './constants';
import EvaluateTracesSectionRenderer from './EvaluateTracesSectionRenderer';

const getDocLink = () => {
  return 'https://mlflow.org/docs/latest/genai/eval-monitor/scorers/custom/';
};

/**
 * Code block component with copy button using OSS CodeSnippet
 */
const CodeBlockWithCopy: React.FC<{
  code: string;
  language: 'bash' | 'python';
  theme: any;
}> = ({ code, language, theme }) => {
  // Map bash to text since CodeSnippet doesn't support bash
  const snippetLanguage = language === 'bash' ? 'text' : language;

  return (
    <div css={{ position: 'relative' }}>
      <CopyButton
        css={{ zIndex: 1, position: 'absolute', top: theme.spacing.xs, right: theme.spacing.xs }}
        showLabel={false}
        copyText={code}
        icon={<CopyIcon />}
      />
      <CodeSnippet
        language={snippetLanguage}
        theme={theme.isDarkMode ? 'duotoneDark' : 'light'}
        style={{ padding: theme.spacing.sm }}
      >
        {code}
      </CodeSnippet>
    </div>
  );
};

export interface CustomCodeScorerFormData {
  name: string;
  code: string;
  sampleRate: number;
  filterString?: string;
  scorerType: typeof SCORER_TYPE.CUSTOM_CODE;
}

interface CustomCodeScorerFormRendererProps {
  control: Control<CustomCodeScorerFormData>;
  mode: ScorerFormMode;
}

const CustomCodeScorerFormRenderer: React.FC<CustomCodeScorerFormRendererProps> = ({ control, mode }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const code = useWatch({ control, name: 'code' });

  const sectionStyles = {
    display: 'flex' as const,
    flexDirection: 'column' as const,
    gap: theme.spacing.sm,
    paddingLeft: mode === SCORER_FORM_MODE.DISPLAY ? theme.spacing.lg : 0,
  };
  const stopPropagationClick = (e: React.MouseEvent) => {
    e.stopPropagation();
  };

  if (mode === SCORER_FORM_MODE.CREATE) {
    const step1Code = `pip install --upgrade "mlflow>=3.1.0"`;

    const step2Code = `from mlflow.genai.scorers import scorer, ScorerSamplingConfig
from typing import Optional, Any
from mlflow.entities import Feedback
import mlflow.entities

@scorer
def my_custom_scorer(
  inputs: Optional[dict[str, Any]],  # The agent's raw input, parsed from the Trace or dataset, as a Python dict
  outputs: Optional[Any],  # The agent's raw output, parsed from the Trace or dataset
  expectations: Optional[dict[str, Any]],  # The expectations passed to evaluate(data=...), as a Python dict
  trace: Optional[mlflow.entities.Trace]  # The app's resulting Trace containing spans and other metadata
) -> int | float | bool | str | Feedback | list[Feedback]:
    """
    Custom scorer template - implement your scoring logic here.
    Return a score as int, float, bool, str, or Feedback object(s).
    """
    # TODO: Implement your custom scoring logic
    return 1.0`;

    const step3Code = `import mlflow

eval_dataset = [
    {
        "inputs": {"question": "What is the capital of France?"},
        "outputs": "The capital of France is Paris.",
    },
]

mlflow.genai.evaluate(
    data=eval_dataset,
    scorers=[my_custom_scorer],
)`;

    return (
      <div css={{ display: 'flex', flexDirection: 'column' }}>
        <Typography.Text>
          <FormattedMessage
            defaultMessage="Follow these steps to create a custom judge using your own code. {link}"
            description="Brief instructions for custom judge functions"
            values={{
              link: (
                <Typography.Link
                  componentId={`${COMPONENT_ID_PREFIX}.custom-scorer-form.documentation-link`}
                  href={getDocLink()}
                  openInNewTab
                >
                  <FormattedMessage defaultMessage="Learn more" description="Documentation link text" />
                </Typography.Link>
              ),
            }}
          />
        </Typography.Text>
        {/* Step 1: Install MLflow */}
        <div>
          <Typography.Title level={4} css={{ marginBottom: theme.spacing.sm }}>
            <FormattedMessage
              defaultMessage="Step 1: Install MLflow"
              description="Step 1 title for custom judge creation"
            />
          </Typography.Title>
          <Typography.Text css={{ display: 'block', marginBottom: theme.spacing.md, maxWidth: 800 }}>
            <FormattedMessage
              defaultMessage="Install or upgrade MLflow to ensure you have the latest judge functionality."
              description="Step 1 description for installing MLflow"
            />
          </Typography.Text>
          <CodeBlockWithCopy
            // Dummy comment to ensure copybara won't fail with formatting issues
            code={step1Code}
            language="bash"
            theme={theme}
          />
        </div>
        {/* Step 2: Define your scorer */}
        <div>
          <Typography.Title level={4} css={{ marginBottom: theme.spacing.sm }}>
            <FormattedMessage
              defaultMessage="Step 2: Define your judge function"
              description="Step 2 title for custom judge creation"
            />
          </Typography.Title>
          <Typography.Text css={{ display: 'block', marginBottom: theme.spacing.md, maxWidth: 800 }}>
            <FormattedMessage
              defaultMessage="Create a custom judge function using the {decorator} decorator. Implement your scoring logic in the function body. {link}"
              description="Step 2 description for defining judge function"
              values={{
                decorator: <Typography.Text code>@scorer</Typography.Text>,
                link: (
                  <Typography.Link
                    componentId={`${COMPONENT_ID_PREFIX}.custom-scorer-form.step2-documentation-link`}
                    href={getDocLink()}
                    openInNewTab
                  >
                    <FormattedMessage defaultMessage="Learn more" description="Documentation link text" />
                  </Typography.Link>
                ),
              }}
            />
          </Typography.Text>
          <CodeBlockWithCopy
            // Dummy comment to ensure copybara won't fail with formatting issues
            code={step2Code}
            language="python"
            theme={theme}
          />
        </div>
        {/* Step 3: Run the scorer */}
        <div>
          <Typography.Title level={4} css={{ marginBottom: theme.spacing.sm }}>
            <FormattedMessage
              defaultMessage="Step 3: Run the judge"
              description="Step 3 title for custom judge creation"
            />
          </Typography.Title>
          <Typography.Text css={{ display: 'block', marginBottom: theme.spacing.md, maxWidth: 800 }}>
            <FormattedMessage
              defaultMessage="Pass the function directly to {evaluate}, just like other predefined or LLM-based judges."
              description="Step 3 description for running the judge"
              values={{
                evaluate: <Typography.Text code>mlflow.genai.evaluate</Typography.Text>,
              }}
            />
          </Typography.Text>
          <CodeBlockWithCopy
            // Dummy comment to ensure copybara won't fail with formatting issues
            code={step3Code}
            language="python"
            theme={theme}
          />
        </div>
      </div>
    );
  }

  return (
    <div css={sectionStyles}>
      <div
        onClick={stopPropagationClick}
        css={{ cursor: 'auto', marginBottom: mode === SCORER_FORM_MODE.DISPLAY ? theme.spacing.sm : 0 }}
      >
        <CodeBlockWithCopy
          // Dummy comment to ensure copybara won't fail with formatting issues
          code={code || ''}
          language="python"
          theme={theme}
        />
      </div>

      <EvaluateTracesSectionRenderer control={control} mode={mode} />
    </div>
  );
};

export default CustomCodeScorerFormRenderer;
