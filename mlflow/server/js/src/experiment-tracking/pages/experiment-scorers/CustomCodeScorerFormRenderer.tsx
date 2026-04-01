import React from 'react';
import { useDesignSystemTheme, Typography, CopyIcon } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import { CodeSnippet } from '@databricks/web-shared/snippet';
import { CopyButton } from '@mlflow/mlflow/src/shared/building_blocks/CopyButton';
import { useWatch, type Control } from 'react-hook-form';
import type { SCORER_TYPE } from './constants';
import { type ScorerFormMode, SCORER_FORM_MODE } from './constants';
import EvaluateTracesSection from './EvaluateTracesSection';

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
  isSessionLevelScorer?: boolean;
}

interface CustomCodeScorerFormRendererProps {
  control: Control<CustomCodeScorerFormData>;
  mode: ScorerFormMode;
}

const CustomCodeScorerFormRenderer: React.FC<CustomCodeScorerFormRendererProps> = ({ control, mode }) => {
  const { theme } = useDesignSystemTheme();

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
    const step1Code = `from mlflow.genai.scorers import scorer
from mlflow.entities import Feedback

@scorer
def my_custom_judge(
  inputs,  # The agent's raw input, parsed from the Trace or dataset
  outputs,  # The agent's raw output, parsed from the Trace or dataset
) -> Feedback:
    """Check if the output is non-empty and return a Feedback object."""
    is_non_empty = outputs is not None and len(str(outputs).strip()) > 0
    return Feedback(
        value=is_non_empty,
        rationale="Output is non-empty" if is_non_empty else "Output is empty",
    )`;

    const step2Code = `import mlflow

eval_dataset = [
    {
        "inputs": {"question": "What is the capital of France?"},
        "outputs": "The capital of France is Paris.",
    },
]

mlflow.genai.evaluate(
    data=eval_dataset,
    scorers=[my_custom_judge],
)`;

    return (
      <div css={{ display: 'flex', flexDirection: 'column' }}>
        <Typography.Text>
          <FormattedMessage
            defaultMessage="Follow these steps to create a custom code judge using your own code. {link}"
            description="Brief instructions for custom code judge creation"
            values={{
              link: (
                <Typography.Link
                  componentId="codegen_no_dynamic_mlflow_web_js_src_experiment_tracking_pages_experiment_scorers_customcodescorerformrenderer_152"
                  href={getDocLink()}
                  openInNewTab
                >
                  <FormattedMessage defaultMessage="Learn more" description="Documentation link text" />
                </Typography.Link>
              ),
            }}
          />
        </Typography.Text>
        {/* Step 1: Define your judge */}
        <div>
          <Typography.Title level={4} css={{ marginTop: theme.spacing.sm }}>
            <FormattedMessage
              defaultMessage="Step 1: Define your judge function"
              description="Step 1 title for custom code judge creation"
            />
          </Typography.Title>
          <Typography.Text css={{ display: 'block', marginBottom: theme.spacing.md, maxWidth: 800 }}>
            <FormattedMessage
              defaultMessage="Create a custom judge function using the {decorator} decorator. All parameters ({inputs}, {outputs}, {expectations}, {trace}) are optional — include only the ones your logic needs. {link}"
              description="Step 1 description for defining judge function"
              values={{
                decorator: <Typography.Text code>@scorer</Typography.Text>,
                inputs: <Typography.Text code>inputs</Typography.Text>,
                outputs: <Typography.Text code>outputs</Typography.Text>,
                expectations: <Typography.Text code>expectations</Typography.Text>,
                trace: <Typography.Text code>trace</Typography.Text>,
                link: (
                  <Typography.Link
                    componentId="codegen_no_dynamic_mlflow_web_js_src_experiment_tracking_pages_experiment_scorers_customcodescorerformrenderer_209"
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
            code={step1Code}
            language="python"
            theme={theme}
          />
        </div>
        {/* Step 2: Run the judge */}
        <div>
          <Typography.Title level={4} css={{ marginTop: theme.spacing.sm }}>
            <FormattedMessage
              defaultMessage="Step 2: Run the judge"
              description="Step 2 title for custom code judge creation"
            />
          </Typography.Title>
          <Typography.Text css={{ display: 'block', marginBottom: theme.spacing.md, maxWidth: 800 }}>
            <FormattedMessage
              defaultMessage="Pass the function directly to {evaluate}, just like other predefined or LLM-based judges."
              description="Step 2 description for running the judge"
              values={{
                evaluate: <Typography.Text code>mlflow.genai.evaluate</Typography.Text>,
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

      <EvaluateTracesSection control={control} mode={mode} />
    </div>
  );
};

export default CustomCodeScorerFormRenderer;
