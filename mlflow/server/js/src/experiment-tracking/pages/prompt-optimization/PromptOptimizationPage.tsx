import { ScrollablePageWrapper } from '@mlflow/mlflow/src/common/components/ScrollablePageWrapper';
import { useOptimizationJobsQuery } from './hooks/useOptimizationJobsQuery';
import { useDatasetNamesLookup } from './hooks/useDatasetNamesLookup';
import {
  Alert,
  Button,
  DangerModal,
  Header,
  RefreshIcon,
  Spacer,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { useState, useCallback } from 'react';
import { OptimizationJobsListTable } from './components/OptimizationJobsListTable';
import { withErrorBoundary } from '../../../common/utils/withErrorBoundary';
import ErrorUtils from '../../../common/utils/ErrorUtils';
import { useMutation } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { PromptOptimizationApi } from './api';
import { useCreateOptimizationModal } from './hooks/useCreateOptimizationModal';

interface PromptOptimizationPageProps {
  experimentId: string;
}

const PromptOptimizationPageImpl = ({ experimentId }: PromptOptimizationPageProps) => {
  const { theme } = useDesignSystemTheme();
  const { data, error, isLoading, refetch } = useOptimizationJobsQuery({ experimentId });
  const { getDatasetName } = useDatasetNamesLookup({ experimentId });

  const [jobToDelete, setJobToDelete] = useState<string | null>(null);

  const { CreateOptimizationModal, openModal: openCreateModal } = useCreateOptimizationModal({
    experimentId,
    onSuccess: refetch,
  });

  const cancelMutation = useMutation({
    mutationFn: PromptOptimizationApi.cancelJob,
    onSuccess: () => refetch(),
  });

  const deleteMutation = useMutation({
    mutationFn: PromptOptimizationApi.deleteJob,
    onSuccess: () => {
      refetch();
      setJobToDelete(null);
    },
  });

  const handleCancelJob = useCallback(
    (jobId: string) => {
      cancelMutation.mutate(jobId);
    },
    [cancelMutation],
  );

  const handleDeleteJob = useCallback((jobId: string) => {
    setJobToDelete(jobId);
  }, []);

  const handleConfirmDelete = useCallback(() => {
    if (jobToDelete) {
      deleteMutation.mutate(jobToDelete);
    }
  }, [deleteMutation, jobToDelete]);

  const handleCancelDelete = useCallback(() => {
    setJobToDelete(null);
    deleteMutation.reset();
  }, [deleteMutation]);

  return (
    <ScrollablePageWrapper css={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
      <Spacer shrinks={false} />
      <Header
        title={
          <FormattedMessage defaultMessage="Optimization" description="Header title for the prompt optimization page" />
        }
        buttons={
          <div css={{ display: 'flex', gap: theme.spacing.sm }}>
            <Button
              componentId="mlflow.prompt-optimization.list.refresh"
              icon={<RefreshIcon />}
              onClick={() => refetch()}
              loading={isLoading}
            >
              <FormattedMessage defaultMessage="Refresh" description="Label for refresh button" />
            </Button>
            <Button componentId="mlflow.prompt-optimization.list.create" type="primary" onClick={openCreateModal}>
              <FormattedMessage
                defaultMessage="Create new optimization"
                description="Label for the create optimization button"
              />
            </Button>
          </div>
        }
      />
      <Typography.Text color="secondary" css={{ marginTop: theme.spacing.xs }}>
        <FormattedMessage
          defaultMessage="Prompt optimization automatically rewrites your prompts to help you find the best one based on your metrics and data."
          description="Description text for the prompt optimization page"
        />
      </Typography.Text>
      <Spacer shrinks={false} />
      <div css={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        {error?.message && (
          <>
            <Alert
              type="error"
              message={error.message}
              componentId="mlflow.prompt-optimization.list.error"
              closable={false}
            />
            <Spacer />
          </>
        )}
        <OptimizationJobsListTable
          jobs={data}
          isLoading={isLoading}
          experimentId={experimentId}
          onCancelJob={handleCancelJob}
          onDeleteJob={handleDeleteJob}
          getDatasetName={getDatasetName}
        />
      </div>
      {CreateOptimizationModal}

      {/* Delete Confirmation Modal */}
      <DangerModal
        componentId="mlflow.prompt-optimization.delete-modal"
        title={
          <FormattedMessage
            defaultMessage="Delete Optimization Job"
            description="Title for the delete optimization job confirmation modal"
          />
        }
        visible={!!jobToDelete}
        onCancel={handleCancelDelete}
        onOk={handleConfirmDelete}
        confirmLoading={deleteMutation.isLoading}
      >
        <>
          <FormattedMessage
            defaultMessage="Are you sure you want to delete this optimization job? This action cannot be undone."
            description="Confirmation message for deleting an optimization job"
          />
          {deleteMutation.error && (
            <Alert
              componentId="mlflow.prompt-optimization.delete-modal.error"
              type="error"
              message={(deleteMutation.error as Error).message}
              closable={false}
              css={{ marginTop: theme.spacing.md }}
            />
          )}
        </>
      </DangerModal>
    </ScrollablePageWrapper>
  );
};

export const PromptOptimizationPage = withErrorBoundary(
  ErrorUtils.mlflowServices.EXPERIMENTS,
  PromptOptimizationPageImpl,
);

export default PromptOptimizationPage;
