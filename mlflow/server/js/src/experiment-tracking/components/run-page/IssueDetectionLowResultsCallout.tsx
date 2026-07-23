import type { ReactNode } from 'react';
import { Alert, Button, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { useNavigate, useParams } from '../../../common/utils/RoutingUtils';
import Routes from '../../routes';
import { ExperimentPageTabName } from '../../constants';

const HIGH_TRACE_COUNT_THRESHOLD = 50;
const USER_FEEDBACK_DOCS_URL = 'https://mlflow.org/docs/latest/genai/tracing/collect-user-feedback/';

interface IssueDetectionLowResultsCalloutProps {
  issueCount: number;
  tracesAnalyzed?: number;
}

/**
 * Guidance shown when an issue detection run returns 0 or 1 issues (~80% of
 * runs), so the results page offers next steps instead of a dead end.
 */
export const IssueDetectionLowResultsCallout = ({
  issueCount,
  tracesAnalyzed,
}: IssueDetectionLowResultsCalloutProps) => {
  const { theme } = useDesignSystemTheme();
  const navigate = useNavigate();
  const { experimentId } = useParams<{ experimentId: string }>();

  const suggestMoreTraces = tracesAnalyzed === undefined || tracesAnalyzed < HIGH_TRACE_COUNT_THRESHOLD;

  return (
    <Alert
      componentId="mlflow.traces.issue-detection.low-results-callout"
      type="info"
      closable={false}
      message={
        issueCount === 0 ? (
          <FormattedMessage
            defaultMessage="0 issues doesn't always mean all clear"
            description="Title of the guidance callout when an issue detection run found no issues"
          />
        ) : (
          <FormattedMessage
            defaultMessage="Only 1 issue found. There may be more."
            description="Title of the guidance callout when an issue detection run found a single issue"
          />
        )
      }
      description={
        <div>
          <FormattedMessage
            defaultMessage="Small trace samples and weaker models can miss real issues. Ways to improve coverage:"
            description="Introduction of the guidance callout on low-result issue detection runs"
          />
          <ul css={{ margin: `${theme.spacing.xs}px 0`, paddingLeft: theme.spacing.lg }}>
            {suggestMoreTraces && (
              <li>
                {tracesAnalyzed !== undefined ? (
                  <FormattedMessage
                    defaultMessage="Add more traces. Only {count, plural, one {1 trace was} other {# traces were}} analyzed."
                    description="Suggestion to analyze more traces, with the number analyzed in this run"
                    values={{ count: tracesAnalyzed }}
                  />
                ) : (
                  <FormattedMessage
                    defaultMessage="Add more traces for broader coverage"
                    description="Suggestion to analyze more traces"
                  />
                )}
              </li>
            )}
            <li>
              <FormattedMessage
                defaultMessage="Try a stronger model. Model quality directly affects detection accuracy."
                description="Suggestion to use a stronger model for issue detection"
              />
            </li>
            <li>
              <FormattedMessage
                defaultMessage="<link>Add user feedback</link> to traces so real failures stand out"
                description="Suggestion to collect user feedback on traces, with documentation link"
                values={{
                  link: (chunks: ReactNode) => (
                    <Typography.Link
                      componentId="mlflow.traces.issue-detection.low-results-callout.feedback-docs-link"
                      href={USER_FEEDBACK_DOCS_URL}
                      openInNewTab
                    >
                      {chunks}
                    </Typography.Link>
                  ),
                }}
              />
            </li>
            <li>
              <FormattedMessage
                defaultMessage="<link>Annotate traces</link> in the review queue to capture known issues"
                description="Suggestion to annotate traces in the review queue, with link to the review queue"
                values={{
                  link: (chunks: ReactNode) => (
                    <Typography.Link
                      componentId="mlflow.traces.issue-detection.low-results-callout.review-queue-link"
                      onClick={() => {
                        if (experimentId) {
                          navigate(Routes.getExperimentPageTabRoute(experimentId, ExperimentPageTabName.ReviewQueue));
                        }
                      }}
                    >
                      {chunks}
                    </Typography.Link>
                  ),
                }}
              />
            </li>
          </ul>
          <Button
            componentId="mlflow.traces.issue-detection.low-results-callout.run-again"
            data-testid="low-results-run-again"
            size="small"
            onClick={() => {
              if (experimentId) {
                navigate(`${Routes.getExperimentPageTracesTabRoute(experimentId)}?detectIssues=true`);
              }
            }}
          >
            <FormattedMessage
              defaultMessage="Run detection again"
              description="Button to re-run issue detection from the low-results callout"
            />
          </Button>
        </div>
      }
    />
  );
};
