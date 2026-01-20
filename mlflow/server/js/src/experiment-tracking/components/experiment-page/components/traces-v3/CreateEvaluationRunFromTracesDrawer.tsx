import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Alert,
  Button,
  Checkbox,
  CheckCircleIcon,
  Drawer,
  FormUI,
  Input,
  SearchIcon,
  Spinner,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { fetchAPI, getAjaxUrl } from '../../../../../common/utils/FetchUtils';
import { useMutation } from '../../../../../common/utils/reactQueryHooks';
import Utils from '../../../../../common/utils/Utils';
import {
  TrackingJobQueryResult,
  TrackingJobStatus,
  useGetTrackingServerJobStatus,
} from '../../../../../common/hooks/useGetTrackingServerJobStatus';
import { Progress } from '@databricks/design-system/development';
import { useGetScheduledScorers } from '../../../../pages/experiment-scorers/hooks/useGetScheduledScorers';
import { EndpointSelector } from '../../../EndpointSelector';
import { useEndpointsQuery } from '../../../../../gateway/hooks/useEndpointsQuery';
import { ExperimentPageTabName } from '../../../../constants';
import Routes from '../../../../routes';
import { useNavigate } from '../../../../../common/utils/RoutingUtils';
import { persistEvaluationRun } from './utils/evaluationRunStorage';

type EvaluateTracesToRunJobResult = {
  run_id?: string;
  experiment_id?: string;
  progress?: { completed: number; total: number };
};

const JOB_POLLING_INTERVAL = 1500; // polling interval for job status

const isJobRunning = (jobData?: TrackingJobQueryResult<EvaluateTracesToRunJobResult>): boolean => {
  return jobData?.status === TrackingJobStatus.RUNNING || jobData?.status === TrackingJobStatus.PENDING;
};

// Scorer category types
type ScorerCategory = 'prebuilt' | 'custom-llm' | 'custom-code' | 'third-party';
type ScorerLevel = 'trace' | 'session';
type ThirdPartyProvider = 'DeepEval' | 'Phoenix' | 'RAGAS';

// Built-in LLM judge templates (hardcoded list matching Python scorers)
const BUILTIN_SCORERS: Array<{
  name: string;
  description: string;
  level: ScorerLevel;
}> = [
  // Trace-level scorers
  { name: 'Correctness', description: 'Are the expected facts supported by the response?', level: 'trace' },
  { name: 'RelevanceToQuery', description: "Does app's response directly address the user's input?", level: 'trace' },
  {
    name: 'RetrievalGroundedness',
    description: "Is the app's response grounded in retrieved information?",
    level: 'trace',
  },
  {
    name: 'RetrievalRelevance',
    description: "Are retrieved documents relevant to the user's request?",
    level: 'trace',
  },
  {
    name: 'RetrievalSufficiency',
    description: 'Do retrieved documents contain all necessary information?',
    level: 'trace',
  },
  { name: 'Safety', description: "Does the app's response avoid harmful or toxic content?", level: 'trace' },
  { name: 'Guidelines', description: 'Does the response follow specified guidelines?', level: 'trace' },
  // Session-level scorers
  {
    name: 'ConversationCompleteness',
    description: "Did the conversation fully address the user's request?",
    level: 'session',
  },
  {
    name: 'KnowledgeRetention',
    description: 'Did the assistant remember context from earlier in the conversation?',
    level: 'session',
  },
  {
    name: 'UserFrustration',
    description: 'Did the conversation avoid causing user frustration?',
    level: 'session',
  },
];

// Third-party scorers from DeepEval, Phoenix, and RAGAS
const THIRD_PARTY_SCORERS: Array<{
  name: string;
  provider: ThirdPartyProvider;
  level: ScorerLevel;
  judgeType: 'deepeval' | 'phoenix' | 'ragas';
}> = [
  // DeepEval RAG Metrics
  { name: 'AnswerRelevancy', provider: 'DeepEval', level: 'trace', judgeType: 'deepeval' },
  { name: 'Faithfulness', provider: 'DeepEval', level: 'trace', judgeType: 'deepeval' },
  { name: 'ContextualRecall', provider: 'DeepEval', level: 'trace', judgeType: 'deepeval' },
  { name: 'ContextualPrecision', provider: 'DeepEval', level: 'trace', judgeType: 'deepeval' },
  { name: 'ContextualRelevancy', provider: 'DeepEval', level: 'trace', judgeType: 'deepeval' },
  // DeepEval Agentic Metrics
  { name: 'TaskCompletion', provider: 'DeepEval', level: 'trace', judgeType: 'deepeval' },
  { name: 'ToolCorrectness', provider: 'DeepEval', level: 'trace', judgeType: 'deepeval' },
  { name: 'ArgumentCorrectness', provider: 'DeepEval', level: 'trace', judgeType: 'deepeval' },
  { name: 'StepEfficiency', provider: 'DeepEval', level: 'trace', judgeType: 'deepeval' },
  { name: 'PlanAdherence', provider: 'DeepEval', level: 'trace', judgeType: 'deepeval' },
  { name: 'PlanQuality', provider: 'DeepEval', level: 'trace', judgeType: 'deepeval' },
  // DeepEval Conversational Metrics (session-level)
  { name: 'TurnRelevancy', provider: 'DeepEval', level: 'session', judgeType: 'deepeval' },
  { name: 'RoleAdherence', provider: 'DeepEval', level: 'session', judgeType: 'deepeval' },
  { name: 'ConversationCompleteness', provider: 'DeepEval', level: 'session', judgeType: 'deepeval' },
  { name: 'GoalAccuracy', provider: 'DeepEval', level: 'session', judgeType: 'deepeval' },
  { name: 'ToolUse', provider: 'DeepEval', level: 'session', judgeType: 'deepeval' },
  { name: 'TopicAdherence', provider: 'DeepEval', level: 'session', judgeType: 'deepeval' },
  // DeepEval Safety Metrics
  { name: 'Bias', provider: 'DeepEval', level: 'trace', judgeType: 'deepeval' },
  { name: 'Toxicity', provider: 'DeepEval', level: 'trace', judgeType: 'deepeval' },
  { name: 'NonAdvice', provider: 'DeepEval', level: 'trace', judgeType: 'deepeval' },
  { name: 'Misuse', provider: 'DeepEval', level: 'trace', judgeType: 'deepeval' },
  { name: 'PIILeakage', provider: 'DeepEval', level: 'trace', judgeType: 'deepeval' },
  { name: 'RoleViolation', provider: 'DeepEval', level: 'trace', judgeType: 'deepeval' },
  // DeepEval General Metrics
  { name: 'Hallucination', provider: 'DeepEval', level: 'trace', judgeType: 'deepeval' },
  { name: 'Summarization', provider: 'DeepEval', level: 'trace', judgeType: 'deepeval' },
  { name: 'JsonCorrectness', provider: 'DeepEval', level: 'trace', judgeType: 'deepeval' },
  { name: 'PromptAlignment', provider: 'DeepEval', level: 'trace', judgeType: 'deepeval' },
  // DeepEval Deterministic Metrics
  { name: 'ExactMatch', provider: 'DeepEval', level: 'trace', judgeType: 'deepeval' },
  { name: 'PatternMatch', provider: 'DeepEval', level: 'trace', judgeType: 'deepeval' },

  // Phoenix Metrics
  { name: 'Hallucination', provider: 'Phoenix', level: 'trace', judgeType: 'phoenix' },
  { name: 'Relevance', provider: 'Phoenix', level: 'trace', judgeType: 'phoenix' },
  { name: 'Toxicity', provider: 'Phoenix', level: 'trace', judgeType: 'phoenix' },
  { name: 'QA', provider: 'Phoenix', level: 'trace', judgeType: 'phoenix' },
  { name: 'Summarization', provider: 'Phoenix', level: 'trace', judgeType: 'phoenix' },

  // RAGAS RAG Metrics
  { name: 'ContextPrecision', provider: 'RAGAS', level: 'trace', judgeType: 'ragas' },
  { name: 'NonLLMContextPrecisionWithReference', provider: 'RAGAS', level: 'trace', judgeType: 'ragas' },
  { name: 'ContextRecall', provider: 'RAGAS', level: 'trace', judgeType: 'ragas' },
  { name: 'NonLLMContextRecall', provider: 'RAGAS', level: 'trace', judgeType: 'ragas' },
  { name: 'ContextEntityRecall', provider: 'RAGAS', level: 'trace', judgeType: 'ragas' },
  { name: 'NoiseSensitivity', provider: 'RAGAS', level: 'trace', judgeType: 'ragas' },
  { name: 'Faithfulness', provider: 'RAGAS', level: 'trace', judgeType: 'ragas' },
  // RAGAS Comparison Metrics
  { name: 'FactualCorrectness', provider: 'RAGAS', level: 'trace', judgeType: 'ragas' },
  { name: 'NonLLMStringSimilarity', provider: 'RAGAS', level: 'trace', judgeType: 'ragas' },
  { name: 'BleuScore', provider: 'RAGAS', level: 'trace', judgeType: 'ragas' },
  { name: 'CHRFScore', provider: 'RAGAS', level: 'trace', judgeType: 'ragas' },
  { name: 'RougeScore', provider: 'RAGAS', level: 'trace', judgeType: 'ragas' },
  { name: 'StringPresence', provider: 'RAGAS', level: 'trace', judgeType: 'ragas' },
  { name: 'ExactMatch', provider: 'RAGAS', level: 'trace', judgeType: 'ragas' },
  // RAGAS General Purpose
  { name: 'AspectCritic', provider: 'RAGAS', level: 'trace', judgeType: 'ragas' },
  { name: 'RubricsScore', provider: 'RAGAS', level: 'trace', judgeType: 'ragas' },
  { name: 'InstanceRubrics', provider: 'RAGAS', level: 'trace', judgeType: 'ragas' },
  { name: 'SummarizationScore', provider: 'RAGAS', level: 'trace', judgeType: 'ragas' },
];

// Unified scorer type for display
interface DisplayScorer {
  name: string;
  description: string;
  category: ScorerCategory;
  level: ScorerLevel;
  // For submission: 'builtin', 'registered', or third-party types
  type: 'builtin' | 'registered' | 'deepeval' | 'ragas' | 'phoenix';
  // Provider name for third-party scorers (used in hint text)
  provider?: ThirdPartyProvider;
  // Unique key for selection (handles duplicate names across providers)
  key: string;
}

// Pill control component matching Dubois style
const PillControl = ({
  children,
  isActive,
  onClick,
}: {
  children: React.ReactNode;
  isActive: boolean;
  onClick: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  return (
    <button
      type="button"
      onClick={onClick}
      css={{
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        height: 28,
        padding: '0 12px',
        fontSize: theme.typography.fontSizeSm,
        fontWeight: 400,
        fontFamily: 'inherit',
        lineHeight: 1,
        borderRadius: 14,
        cursor: 'pointer',
        transition: 'all 0.1s ease',
        backgroundColor: theme.colors.backgroundPrimary,
        color: theme.colors.textPrimary,
        border: isActive
          ? `2px solid ${theme.colors.actionPrimaryBackgroundDefault}`
          : `1px solid ${theme.colors.border}`,
        // Adjust padding to account for border width difference
        ...(isActive && { padding: '0 11px' }),
        '&:hover': {
          backgroundColor: theme.colors.actionDefaultBackgroundHover,
        },
        '&:focus-visible': {
          outline: `2px solid ${theme.colors.actionPrimaryBackgroundDefault}`,
          outlineOffset: 1,
        },
      }}
    >
      {children}
    </button>
  );
};

// Selected scorer chip component
const SelectedScorerChip = ({ name, onRemove }: { name: string; onRemove: () => void }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      css={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: theme.spacing.xs,
        backgroundColor: theme.colors.tagDefault,
        borderRadius: 12,
        padding: `2px ${theme.spacing.xs}px 2px ${theme.spacing.sm}px`,
        fontSize: theme.typography.fontSizeSm,
      }}
    >
      <span>{name}</span>
      <button
        type="button"
        onClick={onRemove}
        css={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          width: 16,
          height: 16,
          borderRadius: '50%',
          border: 'none',
          backgroundColor: 'transparent',
          cursor: 'pointer',
          color: theme.colors.textSecondary,
          '&:hover': {
            backgroundColor: theme.colors.actionDefaultBackgroundHover,
            color: theme.colors.textPrimary,
          },
        }}
        aria-label={`Remove ${name}`}
      >
        ×
      </button>
    </div>
  );
};

export const CreateEvaluationRunFromTracesDrawer = ({
  experimentId,
  traceIds,
  onClose,
}: {
  experimentId: string;
  traceIds: string[];
  onClose: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const navigate = useNavigate();
  const formatCount = useMemo(() => new Intl.NumberFormat(), []);

  const [runName, setRunName] = useState<string>('');
  const [selectedJudges, setSelectedJudges] = useState<Record<string, boolean>>({});
  const [selectedEndpoint, setSelectedEndpoint] = useState<string>('');
  const [jobId, setJobId] = useState<string | undefined>(undefined);
  const [createdRunId, setCreatedRunId] = useState<string | undefined>(undefined);

  // Filter states - default to built-in category and trace level selected
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [categoryFilter, setCategoryFilter] = useState<ScorerCategory | null>('prebuilt');
  const [levelFilter, setLevelFilter] = useState<ScorerLevel | null>('trace');

  const scheduledScorersResult = useGetScheduledScorers(experimentId);
  const registeredScorers = scheduledScorersResult.data?.scheduledScorers ?? [];

  // Fetch endpoints for auto-selection
  const { data: endpoints } = useEndpointsQuery();

  // Combine built-in, registered, and third-party scorers into a unified list
  const allScorers: DisplayScorer[] = useMemo(() => {
    const builtinList: DisplayScorer[] = BUILTIN_SCORERS.map((scorer) => ({
      name: scorer.name,
      description: scorer.description,
      category: 'prebuilt' as ScorerCategory,
      level: scorer.level,
      type: 'builtin' as const,
      key: `builtin-${scorer.name}`,
    }));

    const registeredList: DisplayScorer[] = registeredScorers.map((scorer) => ({
      name: scorer.name,
      description: '',
      category: scorer.type === 'llm' ? ('custom-llm' as ScorerCategory) : ('custom-code' as ScorerCategory),
      level: scorer.isSessionLevelScorer ? ('session' as ScorerLevel) : ('trace' as ScorerLevel),
      type: 'registered' as const,
      key: `registered-${scorer.name}`,
    }));

    const thirdPartyList: DisplayScorer[] = THIRD_PARTY_SCORERS.map((scorer) => ({
      name: scorer.name,
      description: '',
      category: 'third-party' as ScorerCategory,
      level: scorer.level,
      type: scorer.judgeType,
      provider: scorer.provider,
      key: `${scorer.judgeType}-${scorer.name}`,
    }));

    return [...builtinList, ...registeredList, ...thirdPartyList];
  }, [registeredScorers]);

  // Apply filters
  const filteredScorers = useMemo(() => {
    return allScorers.filter((scorer) => {
      // Apply search filter
      if (searchQuery) {
        const query = searchQuery.toLowerCase();
        if (!scorer.name.toLowerCase().includes(query) && !scorer.description.toLowerCase().includes(query)) {
          return false;
        }
      }
      // Apply category filter
      if (categoryFilter && scorer.category !== categoryFilter) {
        return false;
      }
      // Apply level filter
      if (levelFilter && scorer.level !== levelFilter) {
        return false;
      }
      return true;
    });
  }, [allScorers, searchQuery, categoryFilter, levelFilter]);

  // Get selected scorers with their type info for submission
  const selectedScorersList = useMemo(() => {
    return allScorers.filter((scorer) => selectedJudges[scorer.key]);
  }, [allScorers, selectedJudges]);

  const selectedJudgeNames = useMemo(() => selectedScorersList.map((s) => s.name), [selectedScorersList]);

  // Check if any judges requiring an endpoint are selected (built-in or third-party LLM judges)
  const hasBuiltinJudges = useMemo(() => selectedScorersList.some((s) => s.type === 'builtin'), [selectedScorersList]);
  const hasThirdPartyJudges = useMemo(
    () => selectedScorersList.some((s) => s.type === 'deepeval' || s.type === 'ragas' || s.type === 'phoenix'),
    [selectedScorersList],
  );
  const needsEndpoint = hasBuiltinJudges || hasThirdPartyJudges;

  // Auto-select first endpoint when judges requiring endpoint are selected
  useEffect(() => {
    if (needsEndpoint && !selectedEndpoint && endpoints.length > 0) {
      setSelectedEndpoint(endpoints[0].name);
    }
  }, [needsEndpoint, selectedEndpoint, endpoints]);

  const hasCreatedRunId = Boolean(createdRunId);
  const resolvedRunId = createdRunId;

  const { jobResults } = useGetTrackingServerJobStatus<EvaluateTracesToRunJobResult>(jobId ? [jobId] : undefined, {
    enabled: Boolean(jobId),
    refetchInterval: (data) => {
      if (data?.some((job) => isJobRunning(job))) {
        return JOB_POLLING_INTERVAL;
      }
      return false;
    },
    refetchOnWindowFocus: false,
    refetchOnMount: false,
  });

  const currentJob = jobId ? jobResults?.[jobId] : undefined;
  const jobResultPayload = currentJob && 'result' in currentJob ? (currentJob.result as any) : undefined;
  const evalJobProgress = jobResultPayload?.progress;
  const evalCompletedRaw = evalJobProgress?.completed ?? 0;
  const evalTotalRaw = evalJobProgress?.total ?? traceIds.length;
  const evalTotal = Math.max(0, evalTotalRaw);
  const evalCompleted = Math.max(0, Math.min(evalCompletedRaw, evalTotal));
  const evalPct = evalTotal ? Math.min(100, Math.floor((evalCompleted / evalTotal) * 100)) : 0;
  const evalUnitLabel =
    evalJobProgress?.total && evalJobProgress.total !== traceIds.length ? 'evaluation items' : 'traces';
  const evalJobIsRunning = isJobRunning(currentJob);
  const evalJobSucceeded = currentJob?.status === TrackingJobStatus.SUCCEEDED;

  const evaluationRunsLink = useMemo(() => {
    const base = Routes.getExperimentPageTabRoute(experimentId, ExperimentPageTabName.EvaluationRuns);
    if (!resolvedRunId) {
      return base;
    }
    return `${base}?selectedRunUuid=${encodeURIComponent(resolvedRunId)}`;
  }, [experimentId, resolvedRunId]);

  const { mutateAsync: startJob, isLoading: isStartingJob } = useMutation<
    { job_id: string; run_id?: string },
    Error,
    { run_name?: string; judges: { type: string; name: string }[]; endpoint_name?: string }
  >({
    mutationFn: async ({ run_name, judges, endpoint_name }) => {
      return fetchAPI(getAjaxUrl(`ajax-api/3.0/mlflow/evaluations/traces/run`), 'POST', {
        experiment_id: experimentId,
        trace_ids: traceIds,
        run_name: run_name || undefined,
        judges,
        endpoint_name: endpoint_name || undefined,
      });
    },
  });

  const isBusy = isStartingJob;
  const canSubmit =
    selectedJudgeNames.length > 0 && traceIds.length > 0 && !isBusy && !jobId && (!needsEndpoint || selectedEndpoint); // Require endpoint when builtin or third-party judges selected

  const onSubmit = useCallback(() => {
    const runNameToUse = runName.trim() || undefined;
    // Build judges payload with correct type (builtin, registered, or third-party types)
    const judgesPayload = selectedScorersList.map((scorer) => ({
      type: scorer.type,
      name: scorer.name,
    }));
    // Pass endpoint_name when builtin or third-party judges are selected
    const endpointToUse = needsEndpoint ? selectedEndpoint : undefined;

    (async () => {
      try {
        const jobResponse = await startJob({
          run_name: runNameToUse,
          judges: judgesPayload,
          endpoint_name: endpointToUse,
        });
        const newRunId = jobResponse.run_id;
        if (newRunId) {
          setCreatedRunId(newRunId);
        }
        persistEvaluationRun(experimentId, {
          jobId: jobResponse.job_id,
          total: traceIds.length,
          startedAt: Date.now(),
          runName: runNameToUse,
          runId: newRunId,
        });
        setJobId(jobResponse.job_id);
      } catch (e: any) {
        Utils.displayGlobalErrorNotification(
          intl.formatMessage(
            { defaultMessage: 'Failed to start evaluation run: {message}', description: 'Eval run start failure' },
            { message: e?.message ?? String(e) },
          ),
        );
      }
    })();
  }, [startJob, selectedScorersList, runName, experimentId, traceIds.length, intl, needsEndpoint, selectedEndpoint]);

  /*
   * Prototype (two-call) approach — kept around for easy toggling:
   *
   * 1) POST ajax-api/2.0/mlflow/runs/create -> get run_id
   * 2) POST ajax-api/3.0/mlflow/evaluations/traces/run with run_id
   *
   * This yields the best UX (immediate run link), but we reverted to the
   * single-call flow where the backend creates the run and returns run_id.
   */
  // const { mutateAsync: createRun, isLoading: isCreatingRun } = useMutation<
  //   CreateRunResponse,
  //   Error,
  //   { run_name?: string }
  // >({
  //   mutationFn: async ({ run_name }) => {
  //     return fetchAPI(getAjaxUrl(`ajax-api/2.0/mlflow/runs/create`), 'POST', {
  //       experiment_id: experimentId,
  //       run_name: run_name || undefined,
  //     });
  //   },
  // });
  //
  // const { mutateAsync: startJob, isLoading: isStartingJob } = useMutation<
  //   { job_id: string },
  //   Error,
  //   { run_id: string; judges: { type: string; name: string }[] }
  // >({
  //   mutationFn: async ({ run_id, judges }) => {
  //     return fetchAPI(getAjaxUrl(`ajax-api/3.0/mlflow/evaluations/traces/run`), 'POST', {
  //       experiment_id: experimentId,
  //       trace_ids: traceIds,
  //       run_id,
  //       judges,
  //     });
  //   },
  // });

  const onNavigateToRuns = useCallback(() => {
    navigate(evaluationRunsLink);
    onClose();
  }, [navigate, evaluationRunsLink, onClose]);

  const title = (
    <div>
      <Typography.Title level={2} withoutMargins>
        <FormattedMessage defaultMessage="Create evaluation run" description="Create eval run drawer title" />
      </Typography.Title>
      <Typography.Text color="secondary" css={{ marginTop: theme.spacing.xs }}>
        <FormattedMessage
          defaultMessage="Evaluate traces using various AI judges"
          description="Create eval run drawer subtitle"
        />
      </Typography.Text>
    </div>
  );

  return (
    <Drawer.Root
      modal
      open
      onOpenChange={(open) => {
        if (!open) {
          // Keep the drawer open while we are creating the run and starting evaluation.
          if (!isBusy) {
            onClose();
          }
        }
      }}
    >
      <Drawer.Content
        componentId="mlflow.traces.create_evaluation_run_drawer"
        width="520px"
        title={title}
        expandContentToFullHeight
        footer={
          <div css={{ display: 'flex', gap: theme.spacing.sm, justifyContent: 'flex-end' }}>
            <Button
              componentId="mlflow.traces.create_evaluation_run_drawer.cancel"
              onClick={onClose}
              type="tertiary"
              disabled={isBusy}
            >
              <FormattedMessage defaultMessage="Cancel" description="Cancel button" />
            </Button>

            {hasCreatedRunId && resolvedRunId ? (
              <Button
                componentId="mlflow.traces.create_evaluation_run_drawer.view_run"
                onClick={onNavigateToRuns}
                type="primary"
              >
                <FormattedMessage defaultMessage="View evaluation run" description="View eval run button" />
              </Button>
            ) : (
              <Button
                componentId="mlflow.traces.create_evaluation_run_drawer.create"
                onClick={onSubmit}
                type="primary"
                disabled={!canSubmit}
                loading={isBusy}
              >
                <FormattedMessage defaultMessage="Start run" description="Start run button" />
              </Button>
            )}
          </div>
        }
      >
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md, height: '100%' }}>
          {isStartingJob && (
            <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
              <Spinner size="small" />
              <Typography.Text>
                <FormattedMessage defaultMessage="Starting evaluation..." description="Spinner while starting eval" />
              </Typography.Text>
            </div>
          )}

          {jobId && (evalJobIsRunning || evalJobSucceeded) && (
            <div
              css={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                flex: 1,
                textAlign: 'center',
                padding: theme.spacing.lg,
              }}
            >
              {evalJobSucceeded ? (
                <CheckCircleIcon
                  css={{
                    color: theme.colors.textValidationSuccess,
                    fontSize: 24,
                    marginBottom: theme.spacing.sm,
                  }}
                />
              ) : (
                <Spinner
                  size="default"
                  css={{
                    marginBottom: theme.spacing.sm,
                  }}
                />
              )}

              <Typography.Title level={3} withoutMargins>
                {evalJobSucceeded ? (
                  <FormattedMessage
                    defaultMessage="Evaluation complete"
                    description="Title for evaluation status in traces eval run drawer when complete"
                  />
                ) : (
                  <FormattedMessage
                    defaultMessage="Evaluation in progress"
                    description="Title for evaluation status in traces eval run drawer when in progress"
                  />
                )}
              </Typography.Title>

              <div css={{ width: '100%', maxWidth: 300, marginTop: theme.spacing.md }}>
                <Progress.Root value={evalJobSucceeded ? 100 : evalPct}>
                  <Progress.Indicator />
                </Progress.Root>
                <Typography.Text color="secondary" css={{ marginTop: theme.spacing.xs, display: 'block' }}>
                  {evalJobSucceeded
                    ? `${formatCount.format(evalTotal)} ${evalUnitLabel} evaluated`
                    : `${evalPct}% • ${formatCount.format(evalCompleted)} / ${formatCount.format(evalTotal)} ${evalUnitLabel}`}
                </Typography.Text>
              </div>
            </div>
          )}

          {jobId && currentJob?.status === TrackingJobStatus.FAILED && (
            <Alert
              componentId="mlflow.traces.create_evaluation_run_drawer.failed"
              type="error"
              message={
                <FormattedMessage defaultMessage="Failed to run evaluation." description="Eval run failed in drawer" />
              }
              description={currentJob.result ? String(currentJob.result) : undefined}
            />
          )}

          {/* Only show the configuration form before job creation */}
          {!jobId && (
            <>
              {/* Selected traces count info */}
              <div
                css={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: theme.spacing.xs,
                  padding: theme.spacing.sm,
                  backgroundColor: theme.colors.backgroundSecondary,
                  borderRadius: theme.borders.borderRadiusMd,
                }}
              >
                <Typography.Text>
                  <FormattedMessage
                    defaultMessage="{count, plural, one {# trace} other {# traces}} selected for evaluation"
                    description="Selected traces count"
                    values={{ count: traceIds.length }}
                  />
                </Typography.Text>
              </div>

              <div>
                <FormUI.Label>
                  <FormattedMessage defaultMessage="Run name (optional)" description="Run name label" />
                </FormUI.Label>
                <Input
                  componentId="mlflow.traces.create_evaluation_run_drawer.run_name"
                  value={runName}
                  onChange={(e) => setRunName(e.target.value)}
                  placeholder={intl.formatMessage({
                    defaultMessage: 'Evaluation 1',
                    description: 'Run name placeholder',
                  })}
                />
              </div>

              <div
                css={{
                  display: 'flex',
                  flexDirection: 'column',
                  gap: theme.spacing.sm,
                  flex: 1,
                  minHeight: 0,
                  marginTop: theme.spacing.sm,
                }}
              >
                {/* Selected scorers display - always visible */}
                <div
                  css={{
                    display: 'flex',
                    flexDirection: 'column',
                    gap: theme.spacing.xs,
                    padding: theme.spacing.sm,
                    backgroundColor: theme.colors.backgroundSecondary,
                    borderRadius: theme.borders.borderRadiusMd,
                    minHeight: 52, // Ensures consistent height even when empty
                  }}
                >
                  <Typography.Text size="sm" color="secondary">
                    <FormattedMessage
                      defaultMessage="Selected ({count})"
                      description="Selected scorers count"
                      values={{ count: selectedScorersList.length }}
                    />
                  </Typography.Text>
                  {selectedScorersList.length > 0 && (
                    <div css={{ display: 'flex', flexWrap: 'wrap', gap: theme.spacing.xs }}>
                      {selectedScorersList.map((scorer) => (
                        <SelectedScorerChip
                          key={`selected-${scorer.key}`}
                          name={scorer.name}
                          onRemove={() => setSelectedJudges((prev) => ({ ...prev, [scorer.key]: false }))}
                        />
                      ))}
                    </div>
                  )}
                </div>

                {/* Search input */}
                <Input
                  componentId="mlflow.traces.create_evaluation_run_drawer.search"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder={intl.formatMessage({
                    defaultMessage: 'Search scorer',
                    description: 'Search scorer placeholder',
                  })}
                  prefix={<SearchIcon />}
                />

                {/* Filter pills - all in one line */}
                <div css={{ display: 'flex', gap: theme.spacing.xs, flexWrap: 'wrap' }}>
                  <PillControl
                    isActive={categoryFilter === 'prebuilt'}
                    onClick={() => setCategoryFilter(categoryFilter === 'prebuilt' ? null : 'prebuilt')}
                  >
                    <FormattedMessage defaultMessage="Built-in" description="Built-in judge filter" />
                  </PillControl>
                  <PillControl
                    isActive={categoryFilter === 'custom-llm'}
                    onClick={() => setCategoryFilter(categoryFilter === 'custom-llm' ? null : 'custom-llm')}
                  >
                    <FormattedMessage defaultMessage="Custom" description="Custom judge filter" />
                  </PillControl>
                  <PillControl
                    isActive={categoryFilter === 'third-party'}
                    onClick={() => setCategoryFilter(categoryFilter === 'third-party' ? null : 'third-party')}
                  >
                    <FormattedMessage defaultMessage="Third-party" description="Third-party judge filter" />
                  </PillControl>
                  <PillControl
                    isActive={levelFilter === 'trace'}
                    onClick={() => setLevelFilter(levelFilter === 'trace' ? null : 'trace')}
                  >
                    <FormattedMessage defaultMessage="Trace level" description="Trace level filter" />
                  </PillControl>
                  <PillControl
                    isActive={levelFilter === 'session'}
                    onClick={() => setLevelFilter(levelFilter === 'session' ? null : 'session')}
                  >
                    <FormattedMessage defaultMessage="Session level" description="Session level filter" />
                  </PillControl>
                </div>

                {/* Available count */}
                <Typography.Text size="sm" color="secondary">
                  <FormattedMessage
                    defaultMessage="{count} available"
                    description="Number of available scorers"
                    values={{ count: filteredScorers.length }}
                  />
                </Typography.Text>

                {/* Scorer list */}
                <div
                  css={{
                    border: `1px solid ${theme.colors.border}`,
                    borderRadius: theme.borders.borderRadiusMd,
                    maxHeight: 360,
                    overflowY: 'auto',
                  }}
                >
                  {scheduledScorersResult.isLoading ? (
                    <div css={{ padding: theme.spacing.md }}>
                      <Typography.Text>
                        <FormattedMessage defaultMessage="Loading scorers..." description="Loading scorers" />
                      </Typography.Text>
                    </div>
                  ) : filteredScorers.length === 0 ? (
                    <div css={{ padding: theme.spacing.md }}>
                      <Typography.Text color="secondary">
                        <FormattedMessage
                          defaultMessage="No scorers found matching your filters."
                          description="No scorers empty state"
                        />
                      </Typography.Text>
                    </div>
                  ) : (
                    filteredScorers.map((scorer) => (
                      <div
                        key={scorer.key}
                        css={{
                          padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
                          borderBottom: `1px solid ${theme.colors.borderDecorative}`,
                          cursor: 'pointer',
                          '&:hover': {
                            backgroundColor: theme.colors.actionDefaultBackgroundHover,
                          },
                          '&:last-child': {
                            borderBottom: 'none',
                          },
                        }}
                        onClick={() => setSelectedJudges((prev) => ({ ...prev, [scorer.key]: !prev[scorer.key] }))}
                      >
                        <div css={{ display: 'flex', alignItems: 'flex-start', gap: theme.spacing.sm }}>
                          <Checkbox
                            componentId={`mlflow.traces.create_evaluation_run_drawer.judge.${scorer.key}`}
                            isChecked={Boolean(selectedJudges[scorer.key])}
                            onChange={(checked) => setSelectedJudges((prev) => ({ ...prev, [scorer.key]: checked }))}
                          />
                          <div css={{ flex: 1 }}>
                            <Typography.Text bold>{scorer.name}</Typography.Text>
                            <Typography.Text color="secondary" css={{ display: 'block', fontSize: 12 }}>
                              {scorer.category === 'third-party' && scorer.provider
                                ? scorer.provider
                                : scorer.category === 'prebuilt'
                                  ? intl.formatMessage({
                                      defaultMessage: 'Built-in',
                                      description: 'Built-in scorer type',
                                    })
                                  : intl.formatMessage({
                                      defaultMessage: 'Custom',
                                      description: 'Custom scorer type',
                                    })}{' '}
                              |{' '}
                              {scorer.level === 'trace'
                                ? intl.formatMessage({
                                    defaultMessage: 'Trace level',
                                    description: 'Trace level label',
                                  })
                                : intl.formatMessage({
                                    defaultMessage: 'Session level',
                                    description: 'Session level label',
                                  })}
                            </Typography.Text>
                          </div>
                        </div>
                      </div>
                    ))
                  )}
                </div>

                {/* Endpoint selector - shown when built-in or third-party judges are selected */}
                {needsEndpoint && (
                  <div css={{ marginTop: theme.spacing.sm }}>
                    <FormUI.Label>
                      <FormattedMessage defaultMessage="Endpoint" description="Endpoint label for judges" />
                    </FormUI.Label>
                    <FormUI.Hint>
                      <FormattedMessage
                        defaultMessage="Select the endpoint that judges will use for LLM calls."
                        description="Endpoint hint text"
                      />
                    </FormUI.Hint>
                    <div css={{ marginTop: theme.spacing.xs }}>
                      <EndpointSelector
                        currentEndpointName={selectedEndpoint}
                        onEndpointSelect={setSelectedEndpoint}
                        componentIdPrefix="mlflow.traces.create_evaluation_run_drawer.endpoint"
                      />
                    </div>
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      </Drawer.Content>
    </Drawer.Root>
  );
};
