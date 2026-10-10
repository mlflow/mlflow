import { useEffect, useMemo, useRef, useState } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import {
  Alert,
  BeakerIcon,
  Button,
  ChartLineIcon,
  ForkHorizontalIcon,
  SparkleIcon,
  Spinner,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useLocalStorage } from '@databricks/web-shared/hooks';
import { createTraceLocationForExperiment, useSearchMlflowTraces } from '@databricks/web-shared/genai-traces-table';
import { DEFAULT_TRACES_V4_TIME_LABEL } from '../../components/experiment-page/components/traces-v4/utils/timeRange';
import { shouldEnableIssueDetection } from '../../../common/utils/FeatureUtils';
import { generatePath, Link, useLocation, useNavigate, useParams } from '../../../common/utils/RoutingUtils';
import Routes, { RoutePaths } from '../../routes';
import { ExperimentPageTabName, RunPageTabName } from '../../constants';
import { useAssistant } from '../../../assistant';
import { useGenAIOverviewTraceState } from './hooks/useGenAIOverviewState';
import { useGenAIOverviewAnalyzeState } from './hooks/useGenAIOverviewAnalyzeState';
import { type GenAIOverviewEvalState, useGenAIOverviewEvalState } from './hooks/useGenAIOverviewEvalState';
import { useGenAIOverviewMonitorState } from './hooks/useGenAIOverviewMonitorState';
import { GenAIOverviewCompactStageCard, GenAIOverviewStageCard } from './components/GenAIOverviewStageCard';
import { GenAIOverviewPromptBox } from './components/GenAIOverviewPromptBox';
import { GenAIOverviewIssueSeverityChart } from './components/GenAIOverviewIssueSeverityChart';
import { GenAIOverviewEvalScoresChart } from './components/GenAIOverviewEvalScoresChart';
import {
  GenAIOverviewMonitorScorerSelector,
  GenAIOverviewMonitorTrendChart,
} from './components/GenAIOverviewMonitorTrendChart';
import { GenAIOverviewTraceActivityChart } from './components/GenAIOverviewTraceActivityChart';
import { GenAIOverviewStaleIssueDetectionAlert } from './components/GenAIOverviewStaleIssueDetectionAlert';
import { ManualTracingSetupLink, TracingSetup, WaitingForFirstTrace } from './components/GenAIOverviewTracingSetup';
import type { GenAIOverviewChartContext, GenAIOverviewStageStatus } from './genAIOverview.types';
import { useExperimentContainsTraces } from '../../components/traces/hooks/useExperimentContainsTraces';
import { IssueDetectionModal } from '../../components/experiment-page/components/traces-v3/IssueDetectionModal';
import { useGenAIOverviewIssueDetectionProgress } from './hooks/useGenAIOverviewIssueDetectionProgress';
import { OverviewTab } from './hooks/useOverviewTab';
import { useGenAIOverviewStaleIssueDetectionState } from './hooks/useGenAIOverviewStaleIssueDetectionState';
import { useSqlWarehouseContextSafe } from '../experiment-page-tabs/SqlWarehouseContext';

const GENAI_OVERVIEW_ISSUE_DETECTION_PROMPT =
  'Analyze these traces for systematic issues. Cover response quality (correctness, relevance, adherence to instructions, safety) as well as errors and latency. A trace can complete without errors and still be low quality — inspect the actual inputs and outputs, do not judge on status alone. Report the most important issues grouped by root cause.';

const GENAI_OVERVIEW_EVAL_SETUP_PROMPT =
  'Set up and run an offline evaluation for this MLflow experiment. Inspect its traces and issue-detection results, propose an evaluation dataset and offline evaluation scorers, and ask for confirmation. After confirmation, create the dataset and any scorer definitions needed only for this offline evaluation, then run it using MLflow so the result is logged as an evaluation run in this experiment. Do not register or schedule online scorers, enable monitoring, or change sampling settings.';

const EVAL_REGRESSION_MARGIN = 0.05;

const buildGenAIOverviewChartContextPrompt = (experimentId: string, chartContext: GenAIOverviewChartContext) => {
  const selectionSummary = (() => {
    switch (chartContext.stage) {
      case 'trace':
        return { traceCount: chartContext.traceCount };
      case 'analyze':
        return { issueCounts: chartContext.issueCounts };
      case 'eval':
        return {
          runCount: chartContext.runs.length,
          totalScoreCount: chartContext.totalScoreCount,
          runs: chartContext.runs.map(({ timestampMs, ...run }) => ({
            ...run,
            timestamp: new Date(timestampMs).toISOString(),
          })),
        };
      case 'monitor':
        return {
          scorerName: chartContext.scorerName,
          points: chartContext.points.map(({ timestampMs, value }) => ({
            timestamp: new Date(timestampMs).toISOString(),
            value,
          })),
        };
    }
  })();

  return [
    '## Selected Overview chart context',
    'The user explicitly attached this chart selection as context for their next question.',
    '```json',
    JSON.stringify(
      {
        experimentId,
        stage: chartContext.stage,
        startTime: new Date(chartContext.startTimeMs).toISOString(),
        endTimeExclusive: new Date(chartContext.endTimeMsExclusive).toISOString(),
        selectionSummary,
      },
      null,
      2,
    ),
    '```',
  ].join('\n');
};

const getEvalRegressionCount = (state: GenAIOverviewEvalState) => {
  if (state.status !== 'ready' || state.scorePoints.length < 2) return 0;
  const previousPoint = state.scorePoints[state.scorePoints.length - 2];
  const latestPoint = state.scorePoints[state.scorePoints.length - 1];
  if (!latestPoint.datasetName || latestPoint.datasetName !== previousPoint.datasetName) return 0;

  return state.assessmentScoreNames.filter((scoreName) => {
    const previousValue = previousPoint.scores[scoreName];
    const latestValue = latestPoint.scores[scoreName];
    if (
      typeof previousValue !== 'number' ||
      !Number.isFinite(previousValue) ||
      previousValue < 0 ||
      previousValue > 1 ||
      typeof latestValue !== 'number' ||
      !Number.isFinite(latestValue) ||
      latestValue < 0 ||
      latestValue > 1
    ) {
      return false;
    }
    return previousValue - latestValue >= EVAL_REGRESSION_MARGIN - Number.EPSILON;
  }).length;
};

export const ExperimentGenAIJourneyOverviewPage = () => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { experimentId = '' } = useParams<{ experimentId: string }>();
  const { search } = useLocation();
  const navigate = useNavigate();
  const dashboardRoute = Routes.getExperimentPageTabRoute(experimentId, ExperimentPageTabName.Dashboard);
  const [isDashboardMoveNoticeDismissed, setIsDashboardMoveNoticeDismissed] = useLocalStorage({
    key: 'mlflow.genaiOverview.dashboardMoveNoticeDismissed',
    version: 0,
    initialValue: false,
  });
  const isIssueDetectionEnabled = shouldEnableIssueDetection();
  const useRollingSevenDayRange = true;
  const overviewTimeLabel = DEFAULT_TRACES_V4_TIME_LABEL;
  const sqlWarehouseContext = useSqlWarehouseContextSafe();
  const { canUseAssistant, openPanel, prefillPrompt, sendMessageWhenReady } = useAssistant();
  const { state: traceState, retry: retryTraceQuery } = useGenAIOverviewTraceState(
    experimentId,
    useRollingSevenDayRange,
  );
  const { state: analyzeState, refetch: refetchAnalyzeState } = useGenAIOverviewAnalyzeState(
    experimentId,
    useRollingSevenDayRange,
  );
  const evalState = useGenAIOverviewEvalState(experimentId);
  const monitorState = useGenAIOverviewMonitorState(experimentId);
  const [selectedMonitorScorerName, setSelectedMonitorScorerName] = useState<string>();
  const [promptChartContext, setPromptChartContext] = useState<GenAIOverviewChartContext>();
  const [isIssueDetectionModalOpen, setIsIssueDetectionModalOpen] = useState(false);
  const hasExistingIssues = analyzeState.status === 'completed' && analyzeState.issueCount > 0;
  const {
    phase: issueDetectionPhase,
    canStartDetection: canStartIssueDetection,
    cancelDetection: cancelIssueDetection,
    isCancelling: isCancellingIssueDetection,
    registerJob: registerIssueDetectionJob,
  } = useGenAIOverviewIssueDetectionProgress(experimentId);
  const staleIssueDetectionState = useGenAIOverviewStaleIssueDetectionState({
    experimentId,
    latestRunUuid: analyzeState.status === 'completed' ? analyzeState.runUuid : undefined,
    latestRunCompletedAtMs: analyzeState.status === 'completed' ? analyzeState.latestRunCompletedAtMs : undefined,
    enabled: isIssueDetectionEnabled,
  });
  const previousIssueDetectionPhaseRef = useRef(issueDetectionPhase);
  useEffect(() => {
    const previousPhase = previousIssueDetectionPhaseRef.current;
    previousIssueDetectionPhaseRef.current = issueDetectionPhase;
    const wasInProgress = previousPhase === 'starting' || previousPhase === 'running';
    const isInProgress = issueDetectionPhase === 'starting' || issueDetectionPhase === 'running';
    if (wasInProgress && !isInProgress) {
      void refetchAnalyzeState();
    }
  }, [issueDetectionPhase, refetchAnalyzeState]);
  const isIssueDetectionInProgress = issueDetectionPhase === 'starting' || issueDetectionPhase === 'running';
  const isTracingOnboardingCandidate = traceState.status === 'empty' || traceState.status === 'unavailable';
  const { containsTraces, isLoading: isCheckingTraceHistory } = useExperimentContainsTraces({
    experimentId,
    enabled: isTracingOnboardingCandidate,
  });
  const tracesRoute = useMemo(() => {
    const searchParams = new URLSearchParams(search);
    searchParams.set('startTimeLabel', overviewTimeLabel);
    searchParams.delete('startTime');
    searchParams.delete('endTime');
    searchParams.delete('page');
    return `${Routes.getExperimentPageTabRoute(experimentId, ExperimentPageTabName.Traces)}?${searchParams.toString()}`;
  }, [experimentId, overviewTimeLabel, search]);
  const isResolvingOverviewState =
    traceState.status === 'loading' || (isTracingOnboardingCandidate && isCheckingTraceHistory);
  const showTracingOnboarding = isTracingOnboardingCandidate && !isCheckingTraceHistory && !containsTraces;
  const issueDetectionTraceLocations = useMemo(
    () => sqlWarehouseContext?.traceSearchLocations ?? [createTraceLocationForExperiment(experimentId)],
    [experimentId, sqlWarehouseContext?.traceSearchLocations],
  );
  const { data: issueDetectionTraces } = useSearchMlflowTraces({
    locations: issueDetectionTraceLocations,
    disabled: !isIssueDetectionModalOpen,
    limit: 50,
    pageSize: 50,
    enablePagination: false,
    sqlWarehouseId: sqlWarehouseContext?.warehouseId ?? undefined,
  });
  const scorersRoute = Routes.getExperimentPageTabRoute(experimentId, ExperimentPageTabName.Judges);
  const qualityDashboardRoute = generatePath(RoutePaths.experimentPageTabDashboard, {
    experimentId,
    overviewTab: OverviewTab.Quality,
  });

  const traceCaption = <FormattedMessage defaultMessage="traces" description="Trace step caption" />;
  const issueCaption = <FormattedMessage defaultMessage="issues" description="Analyze step caption" />;

  const defaultTraceDescription = (
    <FormattedMessage
      defaultMessage="Capture inputs, tool calls, latency, and cost to understand how your agent behaves."
      description="Trace step description"
    />
  );
  const tracingOnboardingDescription = (
    <FormattedMessage
      defaultMessage="Start tracing your agent with one command to send traces to MLflow."
      description="Tracing quick setup explanation"
    />
  );
  const tracePresentation = (() => {
    switch (traceState.status) {
      case 'warehouse-required':
        return {
          description: (
            <FormattedMessage
              defaultMessage="A SQL warehouse is required to load trace activity. Select or create one from the sidebar."
              description="Trace step SQL warehouse required state description"
            />
          ),
        };
      case 'permission-denied':
        return {
          description: (
            <FormattedMessage
              defaultMessage="You don't have permission to view trace activity. Ask an experiment owner for access."
              description="Trace step permission denied state description"
            />
          ),
        };
      case 'error':
        return {
          description: (
            <FormattedMessage
              defaultMessage="We couldn't load trace activity. Try again."
              description="Trace step error state description"
            />
          ),
          action: (
            <Button componentId="mlflow.genai-overview.retry-traces" size="small" onClick={retryTraceQuery}>
              <FormattedMessage defaultMessage="Retry" description="Retry loading trace activity action" />
            </Button>
          ),
        };
      case 'loading':
      case 'ready':
      case 'empty':
      case 'unavailable':
        return { description: defaultTraceDescription };
    }
  })();
  const askAssistant = async (prompt: string, chartContext?: GenAIOverviewChartContext) => {
    if (!canUseAssistant) return 'unavailable' as const;
    try {
      openPanel();
      sendMessageWhenReady(
        chartContext ? `${prompt}\n\n${buildGenAIOverviewChartContextPrompt(experimentId, chartContext)}` : prompt,
      );
      return 'submitted' as const;
    } catch {
      return 'failed' as const;
    }
  };
  const openAssistantWithPrompt = (prompt: string) => {
    openPanel();
    prefillPrompt(prompt);
  };
  const primaryStage =
    traceState.status === 'empty'
      ? 'trace'
      : traceState.status === 'ready'
        ? analyzeState.status !== 'completed'
          ? 'analyze'
          : evalState.status !== 'ready'
            ? 'eval'
            : monitorState.status !== 'ready'
              ? 'monitor'
              : undefined
        : undefined;
  const analyzeCardState = useMemo<GenAIOverviewStageStatus>(() => {
    if (analyzeState.status === 'loading') return { status: 'loading' };
    if (analyzeState.status === 'completed') {
      return { status: 'ready', totalCount: analyzeState.issueCount, activity: analyzeState.activity };
    }
    if (analyzeState.status === 'never-run') return { status: 'empty', activity: [] };
    if (analyzeState.status === 'error') return { status: 'error' };
    return { status: 'unavailable' };
  }, [analyzeState]);
  const evalCardState = useMemo<GenAIOverviewStageStatus>(() => {
    if (evalState.status === 'loading') return { status: 'loading' };
    if (evalState.status === 'ready') {
      return {
        status: 'ready',
        totalCount: evalState.runCount,
        activity: evalState.scorePoints.map(({ timestampMs }) => ({ timestampMs, count: 1 })),
      };
    }
    if (evalState.status === 'empty') return { status: 'empty', activity: [] };
    return { status: 'error' };
  }, [evalState]);
  const evalRegressionCount = useMemo(() => getEvalRegressionCount(evalState), [evalState]);
  const monitorCardState = useMemo<GenAIOverviewStageStatus>(() => {
    if (monitorState.status === 'loading') return { status: 'loading' };
    if (monitorState.status === 'ready') {
      return { status: 'ready', totalCount: monitorState.onlineScorerCount, activity: [] };
    }
    if (monitorState.status === 'empty') return { status: 'empty', activity: [] };
    return { status: 'error' };
  }, [monitorState]);
  const activeMonitorScorerName = useMemo(() => {
    if (monitorState.status !== 'ready') return undefined;
    if (selectedMonitorScorerName && monitorState.onlineScorerNames.includes(selectedMonitorScorerName)) {
      return selectedMonitorScorerName;
    }
    return (
      monitorState.onlineScorerNames.find((scorerName) =>
        monitorState.trendByScorerName
          ?.get(scorerName)
          ?.some((point) => typeof point.value === 'number' && Number.isFinite(point.value)),
      ) ?? monitorState.onlineScorerNames[0]
    );
  }, [monitorState, selectedMonitorScorerName]);
  const monitorTrendSummary = useMemo(() => {
    if (monitorState.status !== 'ready' || !activeMonitorScorerName) return undefined;
    const points = (monitorState.trendByScorerName?.get(activeMonitorScorerName) ?? [])
      .flatMap((point) =>
        typeof point.value === 'number' && Number.isFinite(point.value)
          ? [{ timestampMs: point.timestampMs, value: point.value }]
          : [],
      )
      .sort((firstPoint, secondPoint) => firstPoint.timestampMs - secondPoint.timestampMs);
    const latestPoint = points[points.length - 1];
    if (!latestPoint) return undefined;
    const values = points.map(({ value }) => value);
    const previousValue = points[points.length - 2]?.value;
    const isFractional = values.every((value) => value >= 0 && value <= 1);
    const delta = previousValue === undefined ? undefined : latestPoint.value - previousValue;
    const valueLabel = isFractional
      ? intl.formatNumber(latestPoint.value, {
          style: 'percent',
          maximumFractionDigits: 1,
        })
      : intl.formatNumber(latestPoint.value, { maximumFractionDigits: 3 });
    const deltaLabel =
      delta === undefined || delta === 0
        ? undefined
        : intl.formatMessage(
            {
              defaultMessage: '{direction} {delta}{unit}',
              description: 'Change from the previous monitoring score shown in the overview',
            },
            {
              direction: delta > 0 ? '↑' : '↓',
              delta: intl.formatNumber(Math.abs(delta) * (isFractional ? 100 : 1), {
                maximumFractionDigits: isFractional ? 1 : 3,
              }),
              unit: isFractional ? 'pt' : '',
            },
          );
    const latestDateLabel = intl.formatDate(latestPoint.timestampMs, {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      timeZone: 'UTC',
    });
    return { valueLabel, delta, deltaLabel, latestDateLabel };
  }, [activeMonitorScorerName, intl, monitorState]);
  const monitorDashboardAriaLabel =
    activeMonitorScorerName && monitorTrendSummary
      ? intl.formatMessage(
          {
            defaultMessage: 'View dashboard for {scorer}: {value}, last scored {date}',
            description: 'Accessible label for the selected quality monitoring dashboard',
          },
          {
            scorer: activeMonitorScorerName,
            value: monitorTrendSummary.valueLabel,
            date: monitorTrendSummary.latestDateLabel,
          },
        )
      : intl.formatMessage({
          defaultMessage: 'View quality monitoring dashboard',
          description: 'Accessible label for the quality monitoring dashboard',
        });
  const latestIssueDetectionRunRoute =
    analyzeState.status === 'completed'
      ? analyzeState.issueCount > 0
        ? Routes.getIssueDetectionRunDetailsTabRoute(experimentId, analyzeState.runUuid, RunPageTabName.ISSUES)
        : Routes.getIssueDetectionRunDetailsRoute(experimentId, analyzeState.runUuid)
      : undefined;
  const startIssueDetection = () => {
    if (isIssueDetectionEnabled) {
      setIsIssueDetectionModalOpen(true);
    } else if (canUseAssistant) {
      openAssistantWithPrompt(GENAI_OVERVIEW_ISSUE_DETECTION_PROMPT);
    }
  };
  const analyzeAction =
    isIssueDetectionInProgress && !hasExistingIssues ? (
      <Button
        componentId="mlflow.genai-overview.cancel-issue-detection"
        size="small"
        loading={isCancellingIssueDetection}
        onClick={() => {
          void cancelIssueDetection?.();
        }}
      >
        <FormattedMessage defaultMessage="Cancel" description="Cancel issue detection action" />
      </Button>
    ) : !latestIssueDetectionRunRoute &&
      traceState.status === 'ready' &&
      (isIssueDetectionEnabled || canUseAssistant) ? (
      <Button
        componentId="mlflow.genai-overview.detect-issues"
        type={primaryStage === 'analyze' ? 'primary' : undefined}
        size="small"
        onClick={startIssueDetection}
      >
        <FormattedMessage defaultMessage="Detect Issues" description="Button to detect issues in traces" />
      </Button>
    ) : undefined;

  if (isResolvingOverviewState) {
    return (
      <main
        css={{
          display: 'flex',
          flex: 1,
          alignItems: 'center',
          justifyContent: 'center',
          overflowY: 'auto',
          padding: `${theme.spacing.sm}px ${theme.spacing.lg}px ${theme.spacing.lg}px`,
        }}
      >
        <Spinner
          label={<FormattedMessage defaultMessage="Loading overview" description="GenAI overview loading state" />}
        />
      </main>
    );
  }

  return (
    <main
      css={{
        display: 'flex',
        flexDirection: 'column',
        flex: 1,
        gap: theme.spacing.lg,
        overflowY: 'auto',
        padding: `${theme.spacing.sm}px ${theme.spacing.lg}px ${theme.spacing.lg}px`,
      }}
    >
      {!isDashboardMoveNoticeDismissed && (
        <Alert
          componentId="mlflow.genai-overview.dashboard-move-notice"
          type="info"
          closable
          onClose={() => setIsDashboardMoveNoticeDismissed(true)}
          message={
            <FormattedMessage
              defaultMessage="The previous Overview page is now under <dashboardLink>Dashboard</dashboardLink>, with no changes to its functionality."
              description="Notice that the previous experiment Overview charts moved to the Dashboard tab"
              values={{
                dashboardLink: (chunks) => (
                  <Link to={dashboardRoute} componentId="mlflow.genai-overview.dashboard-move-notice.link">
                    {chunks}
                  </Link>
                ),
              }}
            />
          }
        />
      )}
      <div
        css={{
          width: '100%',
          maxWidth: 800,
          alignSelf: 'center',
          display: 'flex',
          flexDirection: 'column',
          gap: theme.spacing.xl,
          flex: 1,
          justifyContent: 'center',
        }}
      >
        <section css={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: theme.spacing.md }}>
          <div css={{ display: 'flex', lineHeight: 0 }}>
            <SparkleIcon aria-hidden="true" color="ai" css={{ fontSize: theme.spacing.md }} />
          </div>
          <Typography.Title level={2} withoutMargins>
            {showTracingOnboarding ? (
              <FormattedMessage
                defaultMessage="Send your first trace to MLflow in just a minute"
                description="First trace onboarding title"
              />
            ) : (
              <FormattedMessage
                defaultMessage="Let's improve your agent with MLflow"
                description="GenAI overview title"
              />
            )}
          </Typography.Title>
          {!showTracingOnboarding && (
            <GenAIOverviewPromptBox
              chartContext={promptChartContext}
              onClearChartContext={() => setPromptChartContext(undefined)}
              onSubmit={askAssistant}
            />
          )}
        </section>

        <section css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
          {!showTracingOnboarding && (
            <div
              css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: theme.spacing.lg }}
            >
              <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
                <Typography.Title level={3} withoutMargins>
                  <FormattedMessage
                    defaultMessage="Build a continuous quality loop in 4 steps"
                    description="Quality loop title"
                  />
                </Typography.Title>
                <Typography.Text color="secondary" size="sm">
                  <FormattedMessage
                    defaultMessage="Trace, analyze, evaluate, and monitor your agent as it evolves."
                    description="Quality loop description"
                  />
                </Typography.Text>
              </div>
              <Typography.Link
                componentId="mlflow.genai-overview.quality-loop-docs"
                href="https://mlflow.org/docs/latest/genai/"
                target="_blank"
                rel="noopener noreferrer"
              >
                <FormattedMessage defaultMessage="Learn how it works ↗" description="Quality loop documentation link" />
              </Typography.Link>
            </div>
          )}
          {showTracingOnboarding ? (
            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
              <div
                css={{
                  overflow: 'hidden',
                  border: `1px solid ${theme.colors.border}`,
                  borderRadius: theme.borders.borderRadiusLg,
                  backgroundColor: theme.colors.backgroundPrimary,
                  boxShadow: theme.shadows.sm,
                }}
              >
                <GenAIOverviewStageCard
                  componentId="mlflow.genai-overview.stage.trace"
                  icon={<ForkHorizontalIcon />}
                  label={<FormattedMessage defaultMessage="1. Trace" description="Trace step" />}
                  caption={traceCaption}
                  description={tracingOnboardingDescription}
                  status={traceState}
                  content={<TracingSetup experimentIds={[experimentId]} />}
                  action={<WaitingForFirstTrace />}
                  expanded
                />
              </div>
              <div css={{ display: 'flex', justifyContent: 'flex-end' }}>
                <ManualTracingSetupLink experimentIds={[experimentId]} />
              </div>
              <div
                css={{
                  display: 'flex',
                  flexDirection: 'column',
                  gap: theme.spacing.md,
                  marginTop: theme.spacing.sm,
                  paddingTop: theme.spacing.md,
                  borderTop: `1px solid ${theme.colors.border}`,
                }}
              >
                <Typography.Text color="secondary" size="sm">
                  <FormattedMessage
                    defaultMessage="Unlocks with your first trace"
                    description="Label for overview steps available after the first trace"
                  />
                </Typography.Text>
                <div
                  css={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(3, minmax(0, 1fr))',
                    gap: theme.spacing.lg,
                    '@media (max-width: 800px)': {
                      gridTemplateColumns: 'minmax(0, 1fr)',
                    },
                  }}
                >
                  <GenAIOverviewCompactStageCard
                    icon={<SparkleIcon />}
                    label={<FormattedMessage defaultMessage="2. Analyze" description="Analyze step" />}
                    description={
                      <FormattedMessage
                        defaultMessage="Review real traces to surface failure modes and quality issues worth fixing."
                        description="Analyze step description"
                      />
                    }
                  />
                  <GenAIOverviewCompactStageCard
                    icon={<BeakerIcon />}
                    label={<FormattedMessage defaultMessage="3. Eval / Test" description="Evaluation step" />}
                    description={
                      <FormattedMessage
                        defaultMessage="Turn issues into a regression test and catch them before you ship."
                        description="Evaluation step description"
                      />
                    }
                  />
                  <GenAIOverviewCompactStageCard
                    icon={<ChartLineIcon />}
                    label={<FormattedMessage defaultMessage="4. Monitor" description="Monitor step" />}
                    description={
                      <FormattedMessage
                        defaultMessage="Run scorers on live traffic to catch quality regressions in production."
                        description="Monitor step description"
                      />
                    }
                  />
                </div>
              </div>
            </div>
          ) : (
            <div
              css={{
                display: 'flex',
                flexDirection: 'column',
                overflow: 'visible',
                border: `1px solid ${theme.colors.border}`,
                borderRadius: theme.borders.borderRadiusLg,
                backgroundColor: theme.colors.backgroundPrimary,
                '& > :not(:last-child)': {
                  borderBottom: `1px solid ${theme.colors.border}`,
                },
                '& > article:first-of-type': {
                  borderTopLeftRadius: theme.borders.borderRadiusLg,
                  borderTopRightRadius: theme.borders.borderRadiusLg,
                },
                '& > article:last-of-type': {
                  borderBottomLeftRadius: theme.borders.borderRadiusLg,
                  borderBottomRightRadius: theme.borders.borderRadiusLg,
                },
              }}
            >
              <GenAIOverviewStageCard
                componentId="mlflow.genai-overview.stage.trace"
                icon={<ForkHorizontalIcon />}
                label={<FormattedMessage defaultMessage="Trace" description="Trace step" />}
                caption={traceCaption}
                description={tracePresentation.description}
                status={traceState}
                activityChart={
                  traceState.status === 'ready' ? (
                    <GenAIOverviewTraceActivityChart
                      activity={traceState.activity}
                      experimentId={experimentId}
                      onAskAssistant={setPromptChartContext}
                    />
                  ) : undefined
                }
                action={tracePresentation.action}
                artifactHref={tracesRoute}
                artifactLabel={<FormattedMessage defaultMessage="View traces" description="Trace artifacts action" />}
              />
              <div>
                <GenAIOverviewStageCard
                  componentId="mlflow.genai-overview.stage.analyze"
                  icon={<SparkleIcon />}
                  label={<FormattedMessage defaultMessage="Analyze" description="Analyze step" />}
                  caption={issueCaption}
                  description={
                    <FormattedMessage
                      defaultMessage="Review real traces to surface failure modes and quality issues worth fixing."
                      description="Analyze step description"
                    />
                  }
                  status={analyzeCardState}
                  headline={
                    analyzeState.status === 'completed' ? (
                      <>
                        <span>{analyzeState.issueCount.toLocaleString()}</span>
                        {analyzeState.issuesCreatedInLastSevenDays > 0 && (
                          <Typography.Text size="sm" color="error">
                            <FormattedMessage
                              defaultMessage="+{count, plural, one {# issue} other {# issues}}"
                              description="Number of issues created in the last seven days"
                              values={{ count: analyzeState.issuesCreatedInLastSevenDays }}
                            />
                          </Typography.Text>
                        )}
                      </>
                    ) : undefined
                  }
                  recommended={primaryStage === 'analyze'}
                  activityChart={
                    analyzeState.status === 'completed' ? (
                      <GenAIOverviewIssueSeverityChart
                        activity={analyzeState.activity}
                        onAskAssistant={setPromptChartContext}
                        issuesRoute={
                          latestIssueDetectionRunRoute ??
                          Routes.getExperimentPageTabRoute(experimentId, ExperimentPageTabName.EvaluationRuns)
                        }
                      />
                    ) : undefined
                  }
                  action={analyzeAction}
                  artifactHref={latestIssueDetectionRunRoute}
                  artifactLabel={
                    <FormattedMessage defaultMessage="View Issues" description="Action to view detected issues" />
                  }
                />
                <GenAIOverviewStaleIssueDetectionAlert
                  experimentId={experimentId}
                  state={staleIssueDetectionState}
                  issueDetectionPhase={issueDetectionPhase}
                  canStartDetection={canStartIssueDetection}
                  onDetectNewIssues={startIssueDetection}
                />
              </div>
              <GenAIOverviewStageCard
                componentId="mlflow.genai-overview.stage.eval"
                icon={<BeakerIcon />}
                label={<FormattedMessage defaultMessage="Eval/Test" description="Evaluation step" />}
                caption={<FormattedMessage defaultMessage="eval runs" description="Evaluation step caption" />}
                description={
                  <FormattedMessage
                    defaultMessage="Turn issues into a regression test and catch them before you ship."
                    description="Evaluation step description"
                  />
                }
                status={evalCardState}
                headline={
                  evalState.status === 'ready' ? (
                    <>
                      <span>{evalState.runCount.toLocaleString()}</span>
                      {evalRegressionCount > 0 && (
                        <Typography.Text size="sm" color="error">
                          <FormattedMessage
                            defaultMessage="↓ {count, plural, one {# regression} other {# regressions}}"
                            description="Number of evaluation scores that regressed beyond the overview margin"
                            values={{ count: evalRegressionCount }}
                          />
                        </Typography.Text>
                      )}
                    </>
                  ) : undefined
                }
                recommended={primaryStage === 'eval'}
                activityChart={
                  evalState.status === 'ready' ? (
                    <GenAIOverviewEvalScoresChart
                      experimentId={experimentId}
                      assessmentScoreNames={evalState.assessmentScoreNames}
                      scorePoints={evalState.scorePoints}
                      onAskAssistant={setPromptChartContext}
                    />
                  ) : undefined
                }
                action={
                  evalState.status === 'empty' && canUseAssistant ? (
                    <Button
                      componentId="mlflow.genai-overview.setup-eval"
                      type={primaryStage === 'eval' ? 'primary' : undefined}
                      size="small"
                      onClick={() => openAssistantWithPrompt(GENAI_OVERVIEW_EVAL_SETUP_PROMPT)}
                    >
                      <FormattedMessage defaultMessage="Set up eval" description="Action to set up an evaluation" />
                    </Button>
                  ) : undefined
                }
                artifactHref={
                  evalState.status === 'ready'
                    ? Routes.getExperimentPageTabRoute(experimentId, ExperimentPageTabName.EvaluationRuns)
                    : undefined
                }
                artifactLabel={
                  <FormattedMessage defaultMessage="View eval runs" description="Evaluation runs action" />
                }
              />
              <GenAIOverviewStageCard
                componentId="mlflow.genai-overview.stage.monitor"
                icon={<ChartLineIcon />}
                label={<FormattedMessage defaultMessage="Monitor" description="Monitor step" />}
                headline={
                  monitorState.status === 'ready' && monitorState.isTrendLoading ? (
                    <Spinner size="small" />
                  ) : (
                    <>
                      <span>{monitorTrendSummary?.valueLabel ?? '—'}</span>
                      {monitorTrendSummary?.deltaLabel && monitorTrendSummary.delta !== undefined && (
                        <Typography.Text size="sm" color={monitorTrendSummary.delta > 0 ? 'success' : 'error'}>
                          {monitorTrendSummary.deltaLabel}
                        </Typography.Text>
                      )}
                    </>
                  )
                }
                caption={intl.formatMessage({
                  defaultMessage: 'online scorers',
                  description: 'Monitor step caption before a scorer is selected',
                })}
                captionControl={
                  activeMonitorScorerName ? (
                    <div
                      css={{
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'flex-start',
                        gap: theme.spacing.xs,
                      }}
                    >
                      <GenAIOverviewMonitorScorerSelector
                        scorerNames={monitorState.status === 'ready' ? monitorState.onlineScorerNames : []}
                        activeScorerName={activeMonitorScorerName}
                        onScorerNameChange={setSelectedMonitorScorerName}
                      />
                      {monitorTrendSummary && (
                        <Typography.Text color="secondary" size="sm">
                          <FormattedMessage
                            defaultMessage="Last scored {date}"
                            description="Date of the latest online monitoring score"
                            values={{ date: monitorTrendSummary.latestDateLabel }}
                          />
                        </Typography.Text>
                      )}
                    </div>
                  ) : undefined
                }
                description={
                  <FormattedMessage
                    defaultMessage="Run scorers on live traffic to catch quality regressions in production."
                    description="Monitor step description"
                  />
                }
                status={monitorCardState}
                recommended={primaryStage === 'monitor'}
                activityChart={
                  monitorState.status === 'ready' ? (
                    <GenAIOverviewMonitorTrendChart
                      dashboardRoute={qualityDashboardRoute}
                      trendByScorerName={monitorState.trendByScorerName ?? new Map()}
                      isLoading={monitorState.isTrendLoading}
                      activeScorerName={activeMonitorScorerName}
                      onAskAssistant={setPromptChartContext}
                    />
                  ) : undefined
                }
                action={
                  monitorState.status !== 'ready' ? (
                    <Button
                      componentId="mlflow.genai-overview.add-monitor"
                      type={primaryStage === 'monitor' ? 'primary' : undefined}
                      size="small"
                      onClick={() => navigate(scorersRoute)}
                    >
                      <FormattedMessage defaultMessage="Add monitor" description="Monitor setup action" />
                    </Button>
                  ) : undefined
                }
                artifactHref={monitorState.status === 'ready' ? qualityDashboardRoute : undefined}
                artifactLabel={
                  <FormattedMessage
                    defaultMessage="View dashboard"
                    description="Action to view the quality monitoring dashboard"
                  />
                }
                artifactAriaLabel={monitorDashboardAriaLabel}
              />
            </div>
          )}
        </section>
      </div>
      {isIssueDetectionModalOpen && (
        <IssueDetectionModal
          experimentId={experimentId}
          onClose={() => setIsIssueDetectionModalOpen(false)}
          onJobStarted={registerIssueDetectionJob}
          availableTraceIds={issueDetectionTraces?.map((trace) => trace.trace_id) ?? []}
        />
      )}
    </main>
  );
};

export default ExperimentGenAIJourneyOverviewPage;
