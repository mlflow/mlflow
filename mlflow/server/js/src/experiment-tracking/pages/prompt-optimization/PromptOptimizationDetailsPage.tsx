import { ScrollablePageWrapper } from '@mlflow/mlflow/src/common/components/ScrollablePageWrapper';
import {
  Alert,
  Breadcrumb,
  Button,
  Header,
  Spacer,
  Spinner,
  StopIcon,
  Tag,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { Link } from '../../../common/utils/RoutingUtils';
import Routes from '../../routes';
import { ExperimentPageTabName } from '../../constants';
import { SELECTED_DATASET_ID_QUERY_PARAM_KEY } from '../experiment-evaluation-datasets/hooks/useSelectedDatasetBySearchParam';
import { withErrorBoundary } from '../../../common/utils/withErrorBoundary';
import ErrorUtils from '../../../common/utils/ErrorUtils';
import { useQuery, useMutation } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { PromptOptimizationApi } from './api';
import {
  getJobStatusName,
  getOptimizerTypeName,
  isJobRunning,
  isJobFinalized,
  getJobProgress,
  GetOptimizationJobResponse,
  OptimizerType,
  JobStatus,
} from './types';
import { IntermediateCandidatesSection } from './components/IntermediateCandidatesSection';
import { EvalScoreChart } from './components/EvalScoreChart';
import { ExpandablePromptSection } from './components/ExpandablePromptSection';
import { PROMPT_VERSION_QUERY_PARAM } from '../prompts/utils';
import { DetailsOverviewMetadataTable } from '../../components/DetailsOverviewMetadataTable';
import { DetailsOverviewMetadataRow } from '../../components/DetailsOverviewMetadataRow';
import { Progress } from '@mlflow/mlflow/src/common/components/Progress';
import Utils from '../../../common/utils/Utils';
import { useIntl } from 'react-intl';
import { useDatasetNamesLookup } from './hooks/useDatasetNamesLookup';

interface PromptOptimizationDetailsPageProps {
  experimentId: string;
  jobId: string;
}

const PromptOptimizationDetailsPageImpl = ({ experimentId, jobId }: PromptOptimizationDetailsPageProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { getDatasetName } = useDatasetNamesLookup({ experimentId });

  const {
    data: jobData,
    error,
    isLoading,
    refetch,
  } = useQuery<GetOptimizationJobResponse, Error>(['optimization_job', jobId], {
    queryFn: () => PromptOptimizationApi.getJob(jobId),
    retry: false,
    refetchInterval: (data) => {
      // Poll every 30s if job is running
      if (data?.job && isJobRunning(data.job.state?.status)) {
        return 30000;
      }
      return false;
    },
  });

  const job = jobData?.job;

  const cancelMutation = useMutation({
    mutationFn: PromptOptimizationApi.cancelJob,
    onSuccess: () => refetch(),
  });

  if (isLoading) {
    return (
      <ScrollablePageWrapper css={{ display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
        <Spinner />
      </ScrollablePageWrapper>
    );
  }

  if (error?.message) {
    return (
      <ScrollablePageWrapper>
        <Alert
          type="error"
          message={error.message}
          componentId="mlflow.prompt-optimization.details.error"
          closable={false}
        />
      </ScrollablePageWrapper>
    );
  }

  if (!job) {
    return (
      <ScrollablePageWrapper>
        <Alert
          type="error"
          message="Job not found"
          componentId="mlflow.prompt-optimization.details.not-found"
          closable={false}
        />
      </ScrollablePageWrapper>
    );
  }

  // Calculate progress - always show progress bar (default to 0 if not available)
  const progress = getJobProgress(job);
  const isFinished = isJobFinalized(job.state?.status);
  const displayProgress = isFinished ? 1.0 : (progress ?? 0);
  const progressPercent = Math.round(displayProgress * 100);

  // Get status tag color (using valid TagColors: lime, pink, purple, turquoise, charcoal)
  const getStatusTagColor = (): 'lime' | 'pink' | 'turquoise' | 'charcoal' | undefined => {
    switch (job.state?.status) {
      case JobStatus.COMPLETED:
        return 'lime'; // green/success
      case JobStatus.FAILED:
        return 'pink'; // red/error
      case JobStatus.CANCELED:
        return 'charcoal'; // gray
      case JobStatus.IN_PROGRESS:
        return 'turquoise'; // blue/info
      case JobStatus.PENDING:
        return 'charcoal';
      default:
        return undefined;
    }
  };

  // Parse optimizer config for display
  const getOptimizerConfigEntries = (): Array<{ key: string; value: string }> => {
    if (!job.config?.optimizer_config_json) return [];
    try {
      const config = JSON.parse(job.config.optimizer_config_json);
      return Object.entries(config).map(([key, value]) => ({
        key,
        value: typeof value === 'object' ? JSON.stringify(value) : String(value),
      }));
    } catch {
      return [];
    }
  };

  const optimizerConfigEntries = getOptimizerConfigEntries();

  // Parse prompt URI and create a link to the prompt details page
  const formatPromptUriDisplay = (uri: string | undefined) => {
    if (!uri) return '-';

    // Parse prompt URI format: prompts:/name/version or prompts:/name/version@alias
    const match = uri.match(/^prompts:\/([^/]+)\/(\d+)(?:@.*)?$/);
    if (!match) {
      // Can't parse, just display as text
      return <span css={{ fontFamily: 'monospace', fontSize: theme.typography.fontSizeSm }}>{uri}</span>;
    }

    const [, promptName, version] = match;
    const basePath = Routes.getPromptDetailsPageRoute(encodeURIComponent(promptName), experimentId);
    const searchParams = new URLSearchParams();
    searchParams.set(PROMPT_VERSION_QUERY_PARAM, version);

    return (
      <Link
        to={`${basePath}?${searchParams.toString()}`}
        css={{ fontFamily: 'monospace', fontSize: theme.typography.fontSizeSm }}
      >
        {uri}
      </Link>
    );
  };

  // Format dataset display with link to dataset page
  const formatDatasetDisplay = () => {
    const datasetId = job.config?.dataset_id;
    if (!datasetId) return '-';

    // Build the link to the dataset page with the dataset selected
    const pathname = Routes.getExperimentPageTabRoute(experimentId, ExperimentPageTabName.Datasets);
    const searchParams = new URLSearchParams();
    searchParams.set(SELECTED_DATASET_ID_QUERY_PARAM_KEY, datasetId);

    // Use dataset name if available, otherwise fall back to ID
    const datasetName = getDatasetName(datasetId) || datasetId;
    const displayName = datasetName.length > 30 ? `${datasetName.slice(0, 30)}...` : datasetName;

    return (
      <Link
        to={{
          pathname,
          search: searchParams.toString(),
        }}
        css={{ fontFamily: 'monospace', fontSize: theme.typography.fontSizeSm }}
      >
        {displayName}
      </Link>
    );
  };

  return (
    <ScrollablePageWrapper css={{ overflow: 'auto' }}>
      <Spacer shrinks={false} />
      <Breadcrumb>
        <Breadcrumb.Item>
          <Link to={Routes.getPromptOptimizationPageRoute(experimentId)}>
            <FormattedMessage defaultMessage="Optimization" description="Breadcrumb for optimization list" />
          </Link>
        </Breadcrumb.Item>
        <Breadcrumb.Item>{jobId}</Breadcrumb.Item>
      </Breadcrumb>
      <Spacer shrinks={false} />
      <Header
        title={
          <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
            <span>
              <FormattedMessage defaultMessage="Optimization Job" description="Job details page title" />
            </span>
            <Tag componentId="mlflow.prompt-optimization.details.status-tag" color={getStatusTagColor()}>
              {getJobStatusName(job.state?.status)}
            </Tag>
          </div>
        }
        buttons={
          <div css={{ display: 'flex', gap: theme.spacing.sm }}>
            <Button
              componentId="mlflow.prompt-optimization.details.cancel"
              onClick={() => cancelMutation.mutate(jobId)}
              loading={cancelMutation.isLoading}
              disabled={!isJobRunning(job.state?.status)}
              icon={<StopIcon />}
              type="tertiary"
            >
              <FormattedMessage defaultMessage="Cancel Job" description="Cancel job button" />
            </Button>
            {job.run_id && (
              <Link to={Routes.getRunPageRoute(experimentId, job.run_id)} target="_blank">
                <Button componentId="mlflow.prompt-optimization.details.view-run">
                  <FormattedMessage defaultMessage="View MLflow Run" description="View run button" />
                </Button>
              </Link>
            )}
          </div>
        }
      />

      {/* Progress Bar Section - always show for running/pending jobs, show 100% for finished */}
      <>
        <Spacer shrinks={false} />
        <div
          css={{
            padding: theme.spacing.md,
            backgroundColor: theme.colors.backgroundSecondary,
            borderRadius: theme.borders.borderRadiusMd,
            maxWidth: 500,
          }}
        >
          <Typography.Text color="secondary" size="sm" css={{ marginBottom: theme.spacing.xs, display: 'block' }}>
            <FormattedMessage defaultMessage="Progress" description="Progress label" />
          </Typography.Text>
          <Progress percent={progressPercent} format={(p) => `${p}%`} />
        </div>
      </>

      <Spacer size="lg" shrinks={false} />

      {/* Job Details and Configuration - Side by Side */}
      <div
        css={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))',
          gap: theme.spacing.lg,
        }}
      >
        {/* Job Details Section */}
        <div>
          <Typography.Title level={4}>
            <FormattedMessage defaultMessage="Job Details" description="Job details section title" />
          </Typography.Title>
          <Spacer shrinks={false} />
          <DetailsOverviewMetadataTable>
            <DetailsOverviewMetadataRow
              title={<FormattedMessage defaultMessage="Job ID" description="Job ID label" />}
              value={
                <span css={{ fontFamily: 'monospace', fontSize: theme.typography.fontSizeSm }}>
                  {job.job_id || '-'}
                </span>
              }
            />
            <DetailsOverviewMetadataRow
              title={<FormattedMessage defaultMessage="MLflow Run ID" description="Run ID label" />}
              value={job.run_id ? <Link to={Routes.getRunPageRoute(experimentId, job.run_id)}>{job.run_id}</Link> : '-'}
            />
            <DetailsOverviewMetadataRow
              title={<FormattedMessage defaultMessage="Source Prompt" description="Source prompt label" />}
              value={formatPromptUriDisplay(job.source_prompt_uri)}
            />
            {job.optimized_prompt_uri && (
              <DetailsOverviewMetadataRow
                title={<FormattedMessage defaultMessage="Optimized Prompt" description="Optimized prompt label" />}
                value={formatPromptUriDisplay(job.optimized_prompt_uri)}
              />
            )}
            <DetailsOverviewMetadataRow
              title={<FormattedMessage defaultMessage="Created" description="Created timestamp label" />}
              value={job.creation_timestamp_ms ? Utils.formatTimestamp(job.creation_timestamp_ms, intl) : '-'}
            />
            {job.completion_timestamp_ms && (
              <DetailsOverviewMetadataRow
                title={<FormattedMessage defaultMessage="Completed" description="Completed timestamp label" />}
                value={Utils.formatTimestamp(job.completion_timestamp_ms, intl)}
              />
            )}
          </DetailsOverviewMetadataTable>
        </div>

        {/* Configuration Section */}
        <div>
          <Typography.Title level={4}>
            <FormattedMessage defaultMessage="Configuration" description="Configuration section title" />
          </Typography.Title>
          <Spacer shrinks={false} />
          <DetailsOverviewMetadataTable>
            <DetailsOverviewMetadataRow
              title={<FormattedMessage defaultMessage="Optimizer Type" description="Optimizer type label" />}
              value={getOptimizerTypeName(job.config?.optimizer_type)}
            />
            {/* Distillation-specific fields */}
            {job.config?.optimizer_type === OptimizerType.DISTILLATION && job.config?.teacher_prompt_uri && (
              <DetailsOverviewMetadataRow
                title={<FormattedMessage defaultMessage="Teacher Prompt" description="Teacher prompt label" />}
                value={formatPromptUriDisplay(job.config.teacher_prompt_uri)}
              />
            )}
            {job.config?.optimizer_type === OptimizerType.DISTILLATION && job.config?.student_model_config_json && (
              <DetailsOverviewMetadataRow
                title={<FormattedMessage defaultMessage="Student Model" description="Student model label" />}
                value={(() => {
                  try {
                    const config = JSON.parse(job.config.student_model_config_json);
                    return `${config.provider}/${config.model_name}`;
                  } catch {
                    return job.config.student_model_config_json;
                  }
                })()}
              />
            )}
            {/* Non-distillation fields */}
            {job.config?.optimizer_type !== OptimizerType.DISTILLATION && (
              <>
                <DetailsOverviewMetadataRow
                  title={<FormattedMessage defaultMessage="Dataset" description="Dataset label" />}
                  value={formatDatasetDisplay()}
                />
                <DetailsOverviewMetadataRow
                  title={<FormattedMessage defaultMessage="Scorer(s)" description="Scorers label" />}
                  value={job.config?.scorers?.join(', ') || '-'}
                />
              </>
            )}
            {optimizerConfigEntries.map(({ key, value }) => (
              <DetailsOverviewMetadataRow key={key} title={key} value={value} />
            ))}
          </DetailsOverviewMetadataTable>
        </div>
      </div>

      {/* Prompts Section - Expandable view for source and optimized prompts */}
      {job.source_prompt_uri && (
        <>
          <Spacer size="lg" shrinks={false} />
          <Typography.Title level={4}>
            <FormattedMessage defaultMessage="Prompts" description="Prompts section title" />
          </Typography.Title>
          <Spacer shrinks={false} />
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
            <ExpandablePromptSection
              promptUri={job.source_prompt_uri}
              experimentId={experimentId}
              title={<FormattedMessage defaultMessage="Source Prompt" description="Source prompt section title" />}
            />
            {job.optimized_prompt_uri && (
              <ExpandablePromptSection
                promptUri={job.optimized_prompt_uri}
                experimentId={experimentId}
                title={
                  <FormattedMessage defaultMessage="Optimized Prompt" description="Optimized prompt section title" />
                }
              />
            )}
          </div>
        </>
      )}

      {/* Evaluation Score History Chart - for GEPA jobs */}
      {job.config?.optimizer_type === OptimizerType.GEPA && job.run_id && (
        <>
          <Typography.Title level={4}>
            <FormattedMessage defaultMessage="Score History" description="Score history chart section title" />
          </Typography.Title>
          <Spacer shrinks={false} />
          <EvalScoreChart runId={job.run_id} scorerNames={job.config.scorers} />
          <Spacer size="lg" shrinks={false} />
        </>
      )}

      {/* Evaluation Scores Summary - for GEPA jobs */}
      {job.config?.optimizer_type === OptimizerType.GEPA && (job.initial_eval_scores || job.final_eval_scores) && (
        <>
          <Typography.Title level={4}>
            <FormattedMessage defaultMessage="Evaluation Scores" description="Scores section title" />
          </Typography.Title>
          <Spacer shrinks={false} />
          <div
            css={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
              gap: theme.spacing.lg,
            }}
          >
            {/* Initial Scores Card */}
            {job.initial_eval_scores && Object.keys(job.initial_eval_scores).length > 0 && (
              <div
                css={{
                  border: `1px solid ${theme.colors.borderDecorative}`,
                  borderRadius: theme.borders.borderRadiusMd,
                  overflow: 'hidden',
                }}
              >
                <div
                  css={{
                    backgroundColor: theme.colors.yellow100,
                    padding: theme.spacing.sm,
                    borderBottom: `1px solid ${theme.colors.borderDecorative}`,
                  }}
                >
                  <Typography.Text bold css={{ color: theme.colors.yellow800 }}>
                    <FormattedMessage defaultMessage="Initial Scores (Baseline)" description="Initial scores header" />
                  </Typography.Text>
                </div>
                <div css={{ padding: theme.spacing.md }}>
                  {Object.entries(job.initial_eval_scores).map(([scorer, score]) => (
                    <div
                      key={scorer}
                      css={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        padding: `${theme.spacing.xs}px 0`,
                        borderBottom:
                          scorer !== Object.keys(job.initial_eval_scores!).slice(-1)[0]
                            ? `1px solid ${theme.colors.borderDecorative}`
                            : 'none',
                      }}
                    >
                      <Typography.Text color="secondary">
                        {scorer === 'aggregate' ? 'Aggregate' : scorer}
                      </Typography.Text>
                      <Typography.Text bold>{(score * 100).toFixed(1)}%</Typography.Text>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Final Scores Card */}
            {job.final_eval_scores && Object.keys(job.final_eval_scores).length > 0 && (
              <div
                css={{
                  border: `1px solid ${theme.colors.borderDecorative}`,
                  borderRadius: theme.borders.borderRadiusMd,
                  overflow: 'hidden',
                }}
              >
                <div
                  css={{
                    backgroundColor: theme.colors.green100,
                    padding: theme.spacing.sm,
                    borderBottom: `1px solid ${theme.colors.borderDecorative}`,
                  }}
                >
                  <Typography.Text bold css={{ color: theme.colors.green800 }}>
                    <FormattedMessage defaultMessage="Final Scores (Optimized)" description="Final scores header" />
                  </Typography.Text>
                </div>
                <div css={{ padding: theme.spacing.md }}>
                  {Object.entries(job.final_eval_scores).map(([scorer, score]) => {
                    const initialScore = job.initial_eval_scores?.[scorer];
                    const improvementPct = initialScore !== undefined ? (score - initialScore) * 100 : undefined;

                    return (
                      <div
                        key={scorer}
                        css={{
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center',
                          padding: `${theme.spacing.xs}px 0`,
                          borderBottom:
                            scorer !== Object.keys(job.final_eval_scores!).slice(-1)[0]
                              ? `1px solid ${theme.colors.borderDecorative}`
                              : 'none',
                        }}
                      >
                        <Typography.Text color="secondary">
                          {scorer === 'aggregate' ? 'Aggregate' : scorer}
                        </Typography.Text>
                        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
                          <Typography.Text bold css={{ color: theme.colors.textValidationSuccess }}>
                            {(score * 100).toFixed(1)}%
                          </Typography.Text>
                          {improvementPct !== undefined && (
                            <Typography.Text
                              size="sm"
                              css={{
                                color:
                                  improvementPct >= 0
                                    ? theme.colors.textValidationSuccess
                                    : theme.colors.textValidationDanger,
                              }}
                            >
                              ({improvementPct >= 0 ? '+' : ''}
                              {improvementPct.toFixed(1)}%)
                            </Typography.Text>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
          <Spacer size="lg" shrinks={false} />
        </>
      )}

      {/* Intermediate Candidates Section for GEPA */}
      {job.config?.optimizer_type === OptimizerType.GEPA && job.run_id && (
        <>
          <Typography.Title level={4}>
            <FormattedMessage
              defaultMessage="Intermediate Candidates"
              description="Intermediate candidates section title"
            />
          </Typography.Title>
          <Spacer shrinks={false} />
          <IntermediateCandidatesSection runId={job.run_id} />
          <Spacer size="lg" shrinks={false} />
        </>
      )}

      {/* Error Message Section */}
      {job.state?.error_message && (
        <>
          <Typography.Title level={4}>
            <FormattedMessage defaultMessage="Error Details" description="Error details section title" />
          </Typography.Title>
          <Spacer shrinks={false} />
          <Alert
            type="error"
            message={job.state.error_message}
            componentId="mlflow.prompt-optimization.details.job-error"
            closable={false}
          />
          <Spacer size="lg" shrinks={false} />
        </>
      )}
    </ScrollablePageWrapper>
  );
};

export const PromptOptimizationDetailsPage = withErrorBoundary(
  ErrorUtils.mlflowServices.EXPERIMENTS,
  PromptOptimizationDetailsPageImpl,
);

export default PromptOptimizationDetailsPage;
