import { Button, ChevronDownIcon, DropdownMenu, Modal } from '@databricks/design-system';
import type { RowSelectionState } from '@tanstack/react-table';
import { useCallback, useMemo, useState } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import { useDeleteRuns } from '../../components/experiment-page/hooks/useDeleteRuns';
import { ErrorWrapper } from '@mlflow/mlflow/src/common/utils/ErrorWrapper';

export const ExperimentEvaluationRunsTableActions = ({
  rowSelection,
  setRowSelection,
  refetchRuns,
  onCompare,
  selectedRunUuid,
  compareToRunUuid,
}: {
  rowSelection: RowSelectionState;
  setRowSelection: (selection: RowSelectionState) => void;
  refetchRuns: () => void;
  onCompare: (runUuid1: string, runUuid2: string) => void;
  selectedRunUuid?: string;
  compareToRunUuid?: string;
}) => {
  const intl = useIntl();
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

  const handleCompare = useCallback(() => {
    if (selectedRunUuids.length === 2) {
      onCompare(selectedRunUuids[0], selectedRunUuids[1]);
    }
  }, [onCompare, selectedRunUuids]);

  const isCompareEnabled = selectedRunUuids.length === 2;

  // Disable compare if already comparing the same 2 runs
  const isAlreadyComparingSelectedRuns = useMemo(() => {
    if (!selectedRunUuid || !compareToRunUuid || selectedRunUuids.length !== 2) {
      return false;
    }
    const compareSet = new Set([selectedRunUuid, compareToRunUuid]);
    return selectedRunUuids.every((uuid) => compareSet.has(uuid));
  }, [selectedRunUuid, compareToRunUuid, selectedRunUuids]);

  const isCompareDisabled = !isCompareEnabled || isAlreadyComparingSelectedRuns;
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
          <DropdownMenu.Item
            componentId="mlflow.eval-runs.actions.compare"
            onClick={handleCompare}
            disabled={isCompareDisabled}
            disabledReason={
              <FormattedMessage
                defaultMessage="Please select 2 runs to compare"
                description="Tooltip for disabled compare action in evaluation runs table actions"
              />
            }
          >
            <FormattedMessage defaultMessage="Compare" description="Compare evaluation runs action" />
          </DropdownMenu.Item>
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
