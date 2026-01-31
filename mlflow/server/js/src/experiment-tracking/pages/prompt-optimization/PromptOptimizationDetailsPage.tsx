import { ScrollablePageWrapper } from '@mlflow/mlflow/src/common/components/ScrollablePageWrapper';
import {
  Alert,
  Breadcrumb,
  Button,
  Header,
  Spacer,
  Spinner,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { Link } from '../../../common/utils/RoutingUtils';
import Routes from '../../routes';
import { withErrorBoundary } from '../../../common/utils/withErrorBoundary';
import ErrorUtils from '../../../common/utils/ErrorUtils';
import { useQuery, useMutation } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { PromptOptimizationApi } from './api';
import {
  getJobStatusName,
  getOptimizerTypeName,
  isJobRunning,
  GetOptimizationJobResponse,
  OptimizerType,
} from './types';
import { IntermediateCandidatesSection } from './components/IntermediateCandidatesSection';

interface PromptOptimizationDetailsPageProps {
  experimentId: string;
  jobId: string;
}

const PromptOptimizationDetailsPageImpl = ({ experimentId, jobId }: PromptOptimizationDetailsPageProps) => {
  const { theme } = useDesignSystemTheme();

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

  const deleteMutation = useMutation({
    mutationFn: PromptOptimizationApi.deleteJob,
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

  return (
    <ScrollablePageWrapper>
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
        title={jobId}
        buttons={
          <div css={{ display: 'flex', gap: theme.spacing.sm }}>
            {isJobRunning(job.state?.status) && (
              <Button
                componentId="mlflow.prompt-optimization.details.cancel"
                onClick={() => cancelMutation.mutate(jobId)}
                loading={cancelMutation.isLoading}
              >
                <FormattedMessage defaultMessage="Cancel" description="Cancel job button" />
              </Button>
            )}
            {job.run_id && (
              <Button
                componentId="mlflow.prompt-optimization.details.view-run"
                onClick={() => window.open(Routes.getRunPageRoute(experimentId, job.run_id!), '_blank')}
              >
                <FormattedMessage defaultMessage="View MLflow Run" description="View run button" />
              </Button>
            )}
          </div>
        }
      />
      <Spacer shrinks={false} />

      {/* Summary Section */}
      <div
        css={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: theme.spacing.md,
          padding: theme.spacing.md,
          backgroundColor: theme.colors.backgroundSecondary,
          borderRadius: theme.borders.borderRadiusMd,
        }}
      >
        <div>
          <Typography.Text color="secondary" size="sm">
            <FormattedMessage defaultMessage="Status" description="Status label" />
          </Typography.Text>
          <Typography.Text>{getJobStatusName(job.state?.status)}</Typography.Text>
        </div>
        <div>
          <Typography.Text color="secondary" size="sm">
            <FormattedMessage defaultMessage="Optimizer" description="Optimizer label" />
          </Typography.Text>
          <Typography.Text>{getOptimizerTypeName(job.config?.optimizer_type)}</Typography.Text>
        </div>
        <div>
          <Typography.Text color="secondary" size="sm">
            <FormattedMessage defaultMessage="Source Prompt" description="Source prompt label" />
          </Typography.Text>
          <Typography.Text>{job.source_prompt_uri || '-'}</Typography.Text>
        </div>
        {job.optimized_prompt_uri && (
          <div>
            <Typography.Text color="secondary" size="sm">
              <FormattedMessage defaultMessage="Optimized Prompt" description="Optimized prompt label" />
            </Typography.Text>
            <Typography.Text>{job.optimized_prompt_uri}</Typography.Text>
          </div>
        )}
        <div>
          <Typography.Text color="secondary" size="sm">
            <FormattedMessage defaultMessage="Dataset" description="Dataset label" />
          </Typography.Text>
          <Typography.Text>{job.config?.dataset_id || '-'}</Typography.Text>
        </div>
        <div>
          <Typography.Text color="secondary" size="sm">
            <FormattedMessage defaultMessage="Scorer(s)" description="Scorers label" />
          </Typography.Text>
          <Typography.Text>{job.config?.scorers?.join(', ') || '-'}</Typography.Text>
        </div>
      </div>

      <Spacer size="lg" shrinks={false} />

      {/* Scores Section */}
      {(job.initial_eval_scores || job.final_eval_scores) && (
        <>
          <Typography.Title level={4}>
            <FormattedMessage defaultMessage="Evaluation Scores" description="Scores section title" />
          </Typography.Title>
          <Spacer shrinks={false} />
          <div
            css={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
              gap: theme.spacing.md,
            }}
          >
            {job.initial_eval_scores &&
              Object.entries(job.initial_eval_scores).map(([key, value]) => (
                <div key={`initial-${key}`}>
                  <Typography.Text color="secondary" size="sm">
                    Initial {key}
                  </Typography.Text>
                  <Typography.Text>{value.toFixed(3)}</Typography.Text>
                </div>
              ))}
            {job.final_eval_scores &&
              Object.entries(job.final_eval_scores).map(([key, value]) => (
                <div key={`final-${key}`}>
                  <Typography.Text color="secondary" size="sm">
                    Final {key}
                  </Typography.Text>
                  <Typography.Text css={{ color: theme.colors.textValidationSuccess }}>
                    {value.toFixed(3)}
                  </Typography.Text>
                </div>
              ))}
          </div>
        </>
      )}

      <Spacer size="lg" shrinks={false} />

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

      <Typography.Title level={4}>
        <FormattedMessage defaultMessage="Configuration" description="Configuration section title" />
      </Typography.Title>
      <Spacer shrinks={false} />
      {job.config?.optimizer_config_json && (
        <pre
          css={{
            backgroundColor: theme.colors.backgroundSecondary,
            padding: theme.spacing.md,
            borderRadius: theme.borders.borderRadiusMd,
            overflow: 'auto',
            fontSize: theme.typography.fontSizeSm,
          }}
        >
          {JSON.stringify(JSON.parse(job.config.optimizer_config_json), null, 2)}
        </pre>
      )}
    </ScrollablePageWrapper>
  );
};

export const PromptOptimizationDetailsPage = withErrorBoundary(
  ErrorUtils.mlflowServices.EXPERIMENTS,
  PromptOptimizationDetailsPageImpl,
);

export default PromptOptimizationDetailsPage;
