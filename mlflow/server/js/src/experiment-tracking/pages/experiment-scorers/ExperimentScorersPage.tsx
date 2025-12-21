import React, { useMemo } from 'react';
import { useDesignSystemTheme, Empty, Spacer, SparkleIcon, Typography, Alert } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { ErrorBoundary } from 'react-error-boundary';
import ExperimentScorersContentContainer from './ExperimentScorersContentContainer';
import { useParams } from '../../../common/utils/RoutingUtils';
import { enableScorersUI } from '../../../common/utils/FeatureUtils';
import { isExperimentEvalResultsMonitoringUIEnabled } from '../../../common/utils/FeatureUtils';
import { usePrefetchTraces } from './useEvaluateTraces';
import { DEFAULT_TRACE_COUNT } from './constants';

const getProductionMonitoringDocUrl = () => {
  return 'https://mlflow.org/docs/latest/genai/eval-monitor/';
};

interface ExperimentScorersPageProps {
  experimentId?: string;
}
const ErrorFallback = ({ error }: { error?: Error }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        alignItems: 'center',
        justifyContent: 'center',
        padding: theme.spacing.lg,
      }}
    >
      <Empty
        title={
          <FormattedMessage
            defaultMessage="Unable to load experiment judges"
            description="Error message when experiment judges page fails to load"
          />
        }
        description={
          error ? (
            <span>{error.message}</span>
          ) : (
            <FormattedMessage
              defaultMessage="We encountered an issue loading the judges interface. Please refresh the page or contact support if the problem persists."
              description="Error description for experiment judges page loading failure"
            />
          )
        }
      />
    </div>
  );
};

const ExperimentScorersPage: React.FC<ExperimentScorersPageProps> = () => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { experimentId } = useParams();
  const isFeatureEnabled = enableScorersUI();

  const prefetchParams = useMemo(
    () => ({
      traceCount: DEFAULT_TRACE_COUNT,
      locations: experimentId
        ? [{ mlflow_experiment: { experiment_id: experimentId }, type: 'MLFLOW_EXPERIMENT' as const }]
        : [],
    }),
    [experimentId],
  );

  // Prefetch traces when the page loads
  usePrefetchTraces(prefetchParams);
  return (
    <ErrorBoundary FallbackComponent={ErrorFallback}>
      {!isFeatureEnabled ? (
        // Show empty state with documentation link when feature flag is off
        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
            height: '100%',
            alignItems: 'center',
            justifyContent: 'center',
            padding: theme.spacing.lg,
          }}
        >
          <Empty
            image={<SparkleIcon css={{ fontSize: 48, color: theme.colors.textSecondary }} />}
            title={
              <FormattedMessage
                defaultMessage="Create and manage judges"
                description="Title for the empty state of the judges page"
              />
            }
            description={
              <div css={{ maxWidth: 600, textAlign: 'center' }}>
                <Spacer size="sm" />
                <FormattedMessage
                  defaultMessage="Configure predefined judges, create guidelines-based LLM judges, or build custom judge functions to track your unique metrics. {link}"
                  description="Description for the empty state of the judges page"
                  values={{
                    link: (
                      <Typography.Link
                        componentId="mlflow.experiment-scorers.documentation-link"
                        href="https://mlflow.org/docs/latest/genai/eval-monitor/"
                        target="_blank"
                        rel="noreferrer"
                      >
                        <FormattedMessage
                          defaultMessage="Learn more about configuring judges"
                          description="Link text for configuring judges documentation"
                        />
                      </Typography.Link>
                    ),
                  }}
                />
              </div>
            }
          />
        </div>
      ) : // Show the actual scorers UI when feature flag is on
      experimentId ? (
        <ExperimentScorersContentContainer experimentId={experimentId || ''} />
      ) : null}
    </ErrorBoundary>
  );
};

export default ExperimentScorersPage;
