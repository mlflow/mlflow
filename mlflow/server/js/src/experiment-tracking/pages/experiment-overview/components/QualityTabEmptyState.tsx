import { useMemo } from 'react';
import { Empty, SearchIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { CodeSnippet, SnippetCopyAction } from '@databricks/web-shared/snippet';
import { useMonitoringFilters } from '../../../hooks/useMonitoringFilters';
import {
  getNamedDateFilters,
  type NamedDateFilter,
} from '../../../components/experiment-page/components/traces-v3/utils/dateUtils';
import { useOverviewChartContext } from '../OverviewChartContext';

const getExampleCode = (experimentId: string) => `import openai

import mlflow
from mlflow.genai.scorers import Correctness

mlflow.set_tracking_uri(YOUR_TRACKING_URI)
mlflow.set_experiment(experiment_id="${experimentId}")

# Define your model's predict function
def my_model(question: str) -> str:
    response = openai.OpenAI().chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}],
    )
    return response.choices[0].message.content

eval_dataset = [
    {
        "inputs": {"question": "How do I log a model with MLflow?"},
        "expectations": {
            "expected_response": "You can log a model by using the mlflow.<flavor>.log_model function."
        },
    },
]

# Run evaluation with built-in scorer Correctness
mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=my_model,
    scorers=[Correctness()],
)`;

interface QualityTabEmptyStateProps {
  /** Whether there are assessments outside the current time range */
  hasAssessmentsOutsideTimeRange?: boolean;
}

export const QualityTabEmptyState = ({ hasAssessmentsOutsideTimeRange = false }: QualityTabEmptyStateProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [monitoringFilters] = useMonitoringFilters();
  const namedDateFilters = useMemo(() => getNamedDateFilters(intl), [intl]);
  const { experimentIds } = useOverviewChartContext();
  const experimentId = experimentIds[0] || '';
  const exampleCode = useMemo(() => getExampleCode(experimentId), [experimentId]);

  // If there are assessments outside the time range, show a simpler message suggesting a longer time range
  if (hasAssessmentsOutsideTimeRange) {
    const filterLabel =
      namedDateFilters.find((filter: NamedDateFilter) => filter.key === monitoringFilters.startTimeLabel)?.label || '';

    return (
      <Empty
        image={<SearchIcon />}
        title={
          <FormattedMessage
            defaultMessage="No assessments available"
            description="Message shown when there are no assessments to display"
          />
        }
        description={
          <>
            <FormattedMessage
              defaultMessage='Some assessments are hidden by your time range filter: "{filterLabel}".'
              description="Message shown when assessments are hidden by time filter"
              values={{
                filterLabel: <strong>{filterLabel}</strong>,
              }}
            />
            <br />
            <FormattedMessage
              defaultMessage="Try selecting a longer time range."
              description="Suggestion to select a longer time range"
            />
          </>
        }
      />
    );
  }

  // No assessments at all - show full guidance
  return (
    <div css={{ flex: 1, flexDirection: 'column', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <Typography.Text color="secondary" css={{ marginBottom: theme.spacing.lg }}>
        <FormattedMessage
          defaultMessage="No assessments available"
          description="Message shown when there are no assessments to display"
        />
      </Typography.Text>
      <Typography.Title level={3} color="secondary">
        <FormattedMessage
          defaultMessage="Monitor quality metrics from scorers"
          description="Empty state title for the quality tab in overview page"
        />
      </Typography.Title>
      <Typography.Paragraph color="secondary" css={{ maxWidth: 600 }}>
        <FormattedMessage
          defaultMessage="MLflow allows you to evaluate your GenAI applications using scorers. Scorers compute quality metrics like relevance, correctness, and custom assessments. Copy the code snippet below to run an evaluation, or visit the documentation for a more in-depth example."
          description="Empty state description for the quality tab in overview page"
        />
        <Typography.Link
          componentId="mlflow.quality_tab.empty_state.learn_more_link"
          css={{ marginLeft: theme.spacing.xs }}
          href="https://mlflow.org/docs/latest/genai/eval-monitor/"
          openInNewTab
        >
          <FormattedMessage
            defaultMessage="Learn more"
            description="Link to the documentation page for GenAI evaluation"
          />
        </Typography.Link>
      </Typography.Paragraph>
      <div css={{ position: 'relative' }}>
        <SnippetCopyAction
          componentId="mlflow.overview.quality_tab.empty_state.example_code_copy"
          css={{ position: 'absolute', top: theme.spacing.xs, right: theme.spacing.xs }}
          copyText={exampleCode}
        />
        <CodeSnippet language="python" showLineNumbers>
          {exampleCode}
        </CodeSnippet>
      </div>
    </div>
  );
};
