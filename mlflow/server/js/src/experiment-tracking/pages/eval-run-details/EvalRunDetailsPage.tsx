import { useEffect } from 'react';
import { FormattedMessage } from 'react-intl';
import {
  Breadcrumb,
  DangerIcon,
  Empty,
  Icon,
  ParagraphSkeleton,
  Tag,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { Link, useNavigate, useParams } from '../../../common/utils/RoutingUtils';
import { shouldEnableNewEvalRunsUi } from '../../../common/utils/FeatureUtils';
import { MonitoringConfigProvider } from '../../hooks/useMonitoringConfig';
import Routes from '../../routes';
import { ExperimentPageTabName } from '../../constants';
import { RunViewEvaluationsTab } from '../../components/evaluations/RunViewEvaluationsTab';
import { useRunLoggedTraceTableArtifacts } from '../../components/evaluations/hooks/useRunLoggedTraceTableArtifacts';
import { RunIcon } from '../../components/run-page/assets/RunIcon';
import { RunNotFoundView } from '../../components/RunNotFoundView';
import { TracesV4PageWrapper } from '../../components/experiment-page/components/traces-v4/components/TracesV4PageWrapper';
import { useEvalRunProgress } from './useEvalRunProgress';
import { EvalRunTracesSidebar } from './EvalRunTracesSidebar';
import { EvalRunTracesV4Table } from './EvalRunTracesV4Table';
import {
  useExperimentPageRevampContext,
  useSetExperimentBreadcrumbTitle,
} from '../experiment-page-tabs/ExperimentPageRevampContext';

const EvalRunDetailsLoadingSkeleton = () => {
  const { theme } = useDesignSystemTheme();
  return (
    <div css={{ padding: theme.spacing.lg }}>
      {[...Array(8).keys()].map((i) => (
        <ParagraphSkeleton label="Loading..." key={i} seed={`eval-run-details-${i}`} />
      ))}
    </div>
  );
};

const EvalRunDetailsIcon = () => {
  const { theme } = useDesignSystemTheme();
  return (
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
      <Icon component={RunIcon} css={{ display: 'flex', color: theme.colors.textSecondary }} />
    </div>
  );
};

const EvalRunDetailsPageContent = ({ experimentId, runUuid }: { experimentId: string; runUuid: string }) => {
  const { theme } = useDesignSystemTheme();
  const { enabled: revampEnabled } = useExperimentPageRevampContext();

  const { isLoading, isNotFound, loadError, status, runName, experiment, runTags, isFinished } = useEvalRunProgress({
    runUuid,
  });
  const displayName = runName || runUuid;
  const runStatusLabel = !isFinished && status ? status : undefined;
  useSetExperimentBreadcrumbTitle(displayName, runStatusLabel);

  // Runs that logged their results as trace table artifacts keep the artifact-backed results view.
  const traceTablesLoggedInRun = useRunLoggedTraceTableArtifacts(runTags);

  if (isNotFound) {
    return <RunNotFoundView runId={runUuid} />;
  }

  if (loadError !== undefined) {
    return (
      <div css={{ marginTop: theme.spacing.lg }}>
        <Empty
          title={
            <FormattedMessage
              defaultMessage="Can't load run details"
              description="Eval run detail page > error loading the run title"
            />
          }
          description={loadError}
          image={<DangerIcon />}
        />
      </div>
    );
  }

  if (isLoading) {
    return <EvalRunDetailsLoadingSkeleton />;
  }

  const tracesTable =
    traceTablesLoggedInRun.length > 0 ? (
      <RunViewEvaluationsTab
        experimentId={experimentId}
        experiment={experiment}
        runUuid={runUuid}
        runTags={runTags}
        runDisplayName={displayName}
        showCompareSelector={false}
        showRefreshButton
      />
    ) : (
      <TracesV4PageWrapper resetKey={`${experimentId}.${runUuid}`}>
        <MonitoringConfigProvider>
          <EvalRunTracesV4Table experimentId={experimentId} runUuid={runUuid} />
        </MonitoringConfigProvider>
      </TracesV4PageWrapper>
    );

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.md,
        padding: theme.spacing.lg,
        // The page shell already supplies the top gutter below the breadcrumb; skip our own top padding
        // so the content sits close to it.
        paddingTop: revampEnabled ? 0 : theme.spacing.lg,
        // Fill the shell's content area instead of growing past it, so the table scrolls inside its
        // own body and the toolbar and header row never slide under the page shell's breadcrumb.
        height: '100%',
        minHeight: 0,
        boxSizing: 'border-box',
      }}
    >
      {!revampEnabled && (
        <nav aria-label="Breadcrumb">
          <Breadcrumb includeTrailingCaret>
            <Breadcrumb.Item>
              <Link
                componentId="mlflow.eval_run_details.breadcrumb.evaluation_runs"
                to={Routes.getExperimentPageTabRoute(experimentId, ExperimentPageTabName.EvaluationRuns)}
              >
                <FormattedMessage
                  defaultMessage="Evaluation Runs"
                  description="Breadcrumb link back to the evaluation runs list from the eval-run detail page"
                />
              </Link>
            </Breadcrumb.Item>
          </Breadcrumb>
        </nav>
      )}

      {/* The page shell shows the run name and status in the breadcrumb, so no in-page header there. */}
      {!revampEnabled && (
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            gap: theme.spacing.sm,
            flexWrap: 'wrap',
          }}
        >
          <EvalRunDetailsIcon />
          <Typography.Title level={3} withoutMargins ellipsis>
            {displayName}
          </Typography.Title>
          {runStatusLabel && (
            <Tag componentId="mlflow.eval_run_details.status_tag" color="lemon">
              {runStatusLabel}
            </Tag>
          )}
        </div>
      )}

      <div css={{ display: 'flex', flex: 1, minHeight: 0, minWidth: 0 }}>
        <div css={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column' }}>{tracesTable}</div>
        <EvalRunTracesSidebar key={runUuid} experimentId={experimentId} runUuid={runUuid} />
      </div>
    </div>
  );
};

const EvalRunDetailsPage = () => {
  const { experimentId, runUuid } = useParams<'experimentId' | 'runUuid'>();
  const navigate = useNavigate();
  const flagEnabled = shouldEnableNewEvalRunsUi();

  useEffect(() => {
    if (flagEnabled || !experimentId) {
      return;
    }
    navigate(Routes.getExperimentPageRoute(experimentId), { replace: true });
  }, [flagEnabled, experimentId, navigate]);

  if (!flagEnabled || !runUuid || !experimentId) {
    return null;
  }

  return <EvalRunDetailsPageContent experimentId={experimentId} runUuid={runUuid} />;
};

export default EvalRunDetailsPage;
