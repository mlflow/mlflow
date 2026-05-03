import { useCallback, useEffect, useMemo, useState } from 'react';
import { CheckCircleIcon, ClockIcon, Tag, Typography, XCircleIcon, useDesignSystemTheme } from '@databricks/design-system';
import type { TagColors } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { Link, useParams } from '../../../common/utils/RoutingUtils';
import Routes from '../../routes';
import { MlflowService } from '../../sdk/MlflowService';
import { EXPERIMENT_PARENT_ID_TAG } from '../experiment-page/utils/experimentPage.common-utils';

const MLFLOW_TEST_OUTCOME_TAG = 'mlflow.test.outcome';
const MLFLOW_TEST_DURATION_TAG = 'mlflow.test.duration';

interface ChildTestRun {
  runUuid: string;
  runName: string;
  outcome: string;
  duration: string;
  metrics: Record<string, number>;
}

const getOutcomeColor = (outcome: string): TagColors | undefined => {
  if (outcome === 'passed') {
    return 'teal';
  }
  if (outcome === 'failed') {
    return 'coral';
  }
  if (outcome === 'skipped') {
    return 'lemon';
  }
  return undefined;
};

const getOutcomeIcon = (outcome: string) => {
  if (outcome === 'passed') {
    return <CheckCircleIcon />;
  }
  if (outcome === 'failed') {
    return <XCircleIcon />;
  }
  if (outcome === 'skipped') {
    return <ClockIcon />;
  }
  return null;
};

const OutcomeBadge = ({ outcome }: { outcome: string }) => {
  const icon = getOutcomeIcon(outcome);
  return (
    <Tag componentId="mlflow.pytest-results.outcome-badge" color={getOutcomeColor(outcome)}>
      {icon && <span css={{ display: 'inline-flex', alignItems: 'center', marginRight: 4 }}>{icon}</span>}
      {outcome}
    </Tag>
  );
};

export const RunViewPytestResultsTab = ({ runUuid }: { runUuid: string }) => {
  const { theme } = useDesignSystemTheme();
  const { experimentId } = useParams<{ experimentId: string }>();
  const [childRuns, setChildRuns] = useState<ChildTestRun[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [hasError, setHasError] = useState(false);

  const loadChildRuns = useCallback(async () => {
    if (!experimentId) {
      return;
    }
    setIsLoading(true);
    try {
      const allRuns: ChildTestRun[] = [];
      let pageToken: string | undefined;

      do {
        const res = await MlflowService.searchRuns({
          experiment_ids: [experimentId],
          filter: `tags.\`${EXPERIMENT_PARENT_ID_TAG}\` = '${runUuid}'`,
          order_by: ['attributes.start_time ASC'],
          max_results: 200,
          ...(pageToken ? { page_token: pageToken } : {}),
        });

        const runs: ChildTestRun[] = (res.runs ?? []).map((run: any) => {
          const tags: Record<string, string> = {};
          for (const tag of run.data?.tags ?? []) {
            tags[tag.key] = tag.value;
          }
          const metrics: Record<string, number> = {};
          for (const metric of run.data?.metrics ?? []) {
            metrics[metric.key] = metric.value;
          }
          return {
            runUuid: run.info.run_uuid ?? run.info.run_id,
            runName: run.info.run_name ?? run.info.run_uuid ?? '',
            outcome: tags[MLFLOW_TEST_OUTCOME_TAG] ?? 'unknown',
            duration: tags[MLFLOW_TEST_DURATION_TAG] ?? '-',
            metrics,
          };
        });

        allRuns.push(...runs);
        pageToken = res.next_page_token;
      } while (pageToken);

      setChildRuns(allRuns);
      setHasError(false);
    } catch {
      setHasError(true);
    } finally {
      setIsLoading(false);
    }
  }, [experimentId, runUuid]);

  useEffect(() => {
    loadChildRuns();
  }, [loadChildRuns]);

  const summary = useMemo(() => {
    const passed = childRuns.filter((r) => r.outcome === 'passed').length;
    const failed = childRuns.filter((r) => r.outcome === 'failed').length;
    const skipped = childRuns.filter((r) => r.outcome === 'skipped').length;
    return { passed, failed, skipped };
  }, [childRuns]);

  if (isLoading) {
    return (
      <div css={{ padding: theme.spacing.md }}>
        <Typography.Text>
          <FormattedMessage
            defaultMessage="Loading test results..."
            description="Run page > pytest results tab > loading state"
          />
        </Typography.Text>
      </div>
    );
  }

  if (hasError) {
    return (
      <div css={{ padding: theme.spacing.md }}>
        <Typography.Text color="error">
          <FormattedMessage
            defaultMessage="Failed to load test results."
            description="Run page > pytest results tab > error state"
          />
        </Typography.Text>
      </div>
    );
  }

  if (childRuns.length === 0) {
    return (
      <div css={{ padding: theme.spacing.md }}>
        <Typography.Text>
          <FormattedMessage
            defaultMessage="No test results found."
            description="Run page > pytest results tab > empty state"
          />
        </Typography.Text>
      </div>
    );
  }

  return (
    <div css={{ padding: theme.spacing.md, width: '100%' }}>
      {/* Summary bar */}
      <div
        data-testid="pytest-results-summary"
        css={{
          display: 'flex',
          gap: theme.spacing.md,
          marginBottom: theme.spacing.md,
          alignItems: 'center',
        }}
      >
        <Typography.Text bold>
          <FormattedMessage
            defaultMessage="{passed} passed"
            description="Run page > pytest results tab > summary passed count"
            values={{ passed: summary.passed }}
          />
        </Typography.Text>
        {summary.failed > 0 && (
          <Typography.Text color="error" bold>
            <FormattedMessage
              defaultMessage="{failed} failed"
              description="Run page > pytest results tab > summary failed count"
              values={{ failed: summary.failed }}
            />
          </Typography.Text>
        )}
        {summary.skipped > 0 && (
          <Typography.Text bold>
            <FormattedMessage
              defaultMessage="{skipped} skipped"
              description="Run page > pytest results tab > summary skipped count"
              values={{ skipped: summary.skipped }}
            />
          </Typography.Text>
        )}
      </div>

      {/* Results table */}
      <table
        data-testid="pytest-results-table"
        css={{
          width: '100%',
          borderCollapse: 'collapse',
          '& th, & td': {
            padding: theme.spacing.sm,
            textAlign: 'left',
            borderBottom: `1px solid ${theme.colors.borderDecorative}`,
          },
          '& th': {
            fontWeight: 600,
            color: theme.colors.textSecondary,
          },
        }}
      >
        <thead>
          <tr>
            <th>
              <FormattedMessage
                defaultMessage="Test Name"
                description="Run page > pytest results tab > table header > test name"
              />
            </th>
            <th>
              <FormattedMessage
                defaultMessage="Outcome"
                description="Run page > pytest results tab > table header > outcome"
              />
            </th>
            <th>
              <FormattedMessage
                defaultMessage="Duration (s)"
                description="Run page > pytest results tab > table header > duration"
              />
            </th>
            <th>
              <FormattedMessage
                defaultMessage="Metrics"
                description="Run page > pytest results tab > table header > metrics"
              />
            </th>
          </tr>
        </thead>
        <tbody>
          {childRuns.map((run) => (
            <tr key={run.runUuid} data-testid={`pytest-result-row-${run.runUuid}`}>
              <td>
                <Link
                  componentId="mlflow.pytest-results.test-link"
                  to={Routes.getRunPageRoute(experimentId ?? '', run.runUuid)}
                >
                  {run.runName}
                </Link>
              </td>
              <td>
                <OutcomeBadge outcome={run.outcome} />
              </td>
              <td>{run.duration}</td>
              <td>
                {Object.entries(run.metrics).length > 0
                  ? Object.entries(run.metrics)
                      .map(([k, v]) => `${k}=${Number(v).toFixed(3)}`)
                      .join(', ')
                  : '-'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};
