/**
 * Per-run page for a regression-test session (a run tagged with
 * ``mlflow.runType=regression_test``, produced by an ``@mlflow.assertions``
 * pytest invocation).
 *
 * Renders the test cases table for the run, with a compare-to-run selector so
 * two regression-test runs can be diffed side by side.
 */
import { BeakerIcon, InfoPopover, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { useMemo } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import { Link, useParams } from '../../../common/utils/RoutingUtils';
import Utils from '../../../common/utils/Utils';
import Routes from '../../routes';
import { ExperimentPageTabName } from '../../constants';
import { useSearchRunsQuery } from '../../components/run-page/hooks/useSearchRunsQuery';
import TestCasesTab from './regression-test-run/TestCasesTab';

const RegressionTestRunPage = () => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { experimentId, runUuid } = useParams<{ experimentId: string; runUuid: string }>();

  const { data: runData } = useSearchRunsQuery({
    experimentIds: experimentId ? [experimentId] : [],
    filter: `attributes.run_id = "${runUuid}"`,
    disabled: !experimentId || !runUuid,
  });

  const runDisplayName = Utils.getRunDisplayName(runData?.info, runUuid);
  const runInfo = runData?.info;

  const infoRows = useMemo(() => {
    if (!runInfo) return null;
    const rows: { label: string; value: string }[] = [];
    if (runInfo.startTime) {
      rows.push({
        label: intl.formatMessage({
          defaultMessage: 'Created at',
          description: 'Regression test run info popover: created-at label',
        }),
        value: Utils.formatTimestamp(runInfo.startTime, intl),
      });
    }
    if (runInfo.userId) {
      rows.push({
        label: intl.formatMessage({
          defaultMessage: 'Created by',
          description: 'Regression test run info popover: created-by label',
        }),
        value: runInfo.userId,
      });
    }
    if (runInfo.experimentId) {
      rows.push({
        label: intl.formatMessage({
          defaultMessage: 'Experiment ID',
          description: 'Regression test run info popover: experiment-id label',
        }),
        value: runInfo.experimentId,
      });
    }
    if (runInfo.status) {
      rows.push({
        label: intl.formatMessage({
          defaultMessage: 'Status',
          description: 'Regression test run info popover: status label',
        }),
        value: runInfo.status.charAt(0) + runInfo.status.slice(1).toLowerCase(),
      });
    }
    if (runInfo.runUuid) {
      rows.push({
        label: intl.formatMessage({
          defaultMessage: 'Run ID',
          description: 'Regression test run info popover: run-id label',
        }),
        value: runInfo.runUuid,
      });
    }
    if (runInfo.startTime && runInfo.endTime) {
      rows.push({
        label: intl.formatMessage({
          defaultMessage: 'Duration',
          description: 'Regression test run info popover: duration label',
        }),
        value: Utils.getDuration(runInfo.startTime, runInfo.endTime) ?? '',
      });
    }
    return rows;
  }, [runInfo, intl]);

  return (
    <div css={{ display: 'flex', flexDirection: 'column', padding: theme.spacing.lg, gap: theme.spacing.md }}>
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
        <Link
          componentId="mlflow.regression-test-run.back-to-eval-runs"
          to={Routes.getExperimentPageTabRoute(experimentId ?? '', ExperimentPageTabName.EvaluationRuns)}
        >
          <FormattedMessage
            defaultMessage="&larr; Back to Evaluation runs"
            description="Back link from the regression test run page to the evaluation runs list"
          />
        </Link>
      </div>
      <span css={{ display: 'inline-flex', alignItems: 'center', gap: theme.spacing.sm }}>
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            backgroundColor: theme.colors.backgroundSecondary,
            padding: 6,
            borderRadius: theme.spacing.lg,
          }}
        >
          <BeakerIcon css={{ display: 'flex', color: theme.colors.textSecondary }} />
        </div>
        <span css={{ fontSize: 24, fontWeight: 600, lineHeight: '32px' }}>{runDisplayName}</span>
        {infoRows && (
          <InfoPopover iconTitle="Info">
            <div
              css={{
                display: 'flex',
                flexDirection: 'column',
                gap: theme.spacing.xs,
                flexWrap: 'nowrap',
              }}
            >
              {infoRows.map(({ label, value }) => (
                <div key={label} css={{ whiteSpace: 'nowrap' }}>
                  {label}: {value}
                </div>
              ))}
            </div>
          </InfoPopover>
        )}
      </span>
      <div css={{ marginTop: theme.spacing.md }}>
        <TestCasesTab experimentId={experimentId} runUuid={runUuid} />
      </div>
    </div>
  );
};

export default RegressionTestRunPage;
