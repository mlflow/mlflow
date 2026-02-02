import { Button, ChevronDownIcon, DropdownMenu, Modal } from '@databricks/design-system';
import type { RowSelectionState } from '@tanstack/react-table';
import { useCallback, useMemo, useState } from 'react';
import { FormattedMessage } from 'react-intl';
import { useDeleteRuns } from '../../components/experiment-page/hooks/useDeleteRuns';
import { ErrorWrapper } from '@mlflow/mlflow/src/common/utils/ErrorWrapper';

export const ExperimentEvaluationRunsTableActions = ({
  rowSelection,
  setRowSelection,
  refetchRuns,
}: {
  rowSelection: RowSelectionState;
  setRowSelection: (selection: RowSelectionState) => void;
  refetchRuns: () => void;
}) => {
  const [deleteModalVisible, setDeleteModalVisible] = useState(false);

  const selectedRunUuids = useMemo(
    () =>
      Object.entries(rowSelection)
        .filter(([_, value]) => value)
        .map(([key]) => key),
    [rowSelection],
  );

  const { mutate, isLoading } = useDeleteRuns({
    onSuccess: () => {
      refetchRuns();
      setRowSelection({});
      setDeleteModalVisible(false);
    },
  });

  const handleDelete = useCallback(() => {
    mutate({ runUuids: selectedRunUuids });
  }, [mutate, selectedRunUuids]);

  const noRunsSelected = selectedRunUuids.length === 0;

  return (
    <>
      <DropdownMenu.Root>
        <DropdownMenu.Trigger asChild>
          <Button
            type="primary"
            componentId="mlflow.eval-runs.actions-button"
            disabled={noRunsSelected}
            endIcon={<ChevronDownIcon />}
          >
            <FormattedMessage defaultMessage="Actions" description="Experiment evaluation runs table actions button" />
          </Button>
        </DropdownMenu.Trigger>
        <DropdownMenu.Content>
          <DropdownMenu.Item componentId="mlflow.eval-runs.actions.delete" onClick={() => setDeleteModalVisible(true)}>
            <FormattedMessage defaultMessage="Delete runs" description="Delete evaluation runs action" />
          </DropdownMenu.Item>
        </DropdownMenu.Content>
      </DropdownMenu.Root>
      <Modal
        componentId="mlflow.eval-runs.runs-delete-modal"
        visible={deleteModalVisible}
        onOk={handleDelete}
        okButtonProps={{ danger: true, loading: isLoading }}
        okText={<FormattedMessage defaultMessage="Delete" description="Delete evaluation runs modal button text" />}
        onCancel={() => {
          setDeleteModalVisible(false);
        }}
        cancelText={
          <FormattedMessage defaultMessage="Cancel" description="Delete evaluation runs cancel button text" />
        }
        confirmLoading={isLoading}
        title={
          <FormattedMessage
            defaultMessage="Delete {numRuns, plural, =1 {1 run} other {# runs}}"
            description="Delete evaluation runs modal title"
            values={{ numRuns: selectedRunUuids.length }}
          />
        }
      >
        <FormattedMessage
          defaultMessage="Are you sure you want to delete these runs?"
          description="Delete evaluation runs modal confirmation text"
        />
      </Modal>
    </>
  );
};
