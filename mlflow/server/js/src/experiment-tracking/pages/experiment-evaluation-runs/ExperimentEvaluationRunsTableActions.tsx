import { Button, ChevronDownIcon, DropdownMenu, Modal } from '@databricks/design-system';
import { useIntl } from '@databricks/i18n';
import type { RowSelectionState } from '@tanstack/react-table';
import { useCallback, useMemo, useState } from 'react';
import { FormattedMessage } from 'react-intl';
import { useDeleteRuns } from '../../components/experiment-page/hooks/useDeleteRuns';
import { ErrorWrapper } from '@mlflow/mlflow/src/common/utils/ErrorWrapper';

export const ExperimentEvaluationRunsTableActions = ({
  rowSelection,
  setRowSelection,
  refetchRuns,
  onCompare,
  selectedRunUuid,
  compareToRunUuid,
  enableImprovedComparison,
  setIsComparisonMode,
  onSetBaseline,
  baselineRunUuid,
  isSavingBaseline,
}: {
  rowSelection: RowSelectionState;
  setRowSelection: (selection: RowSelectionState) => void;
  refetchRuns: () => void;
  onCompare?: (runUuid1: string, runUuid2: string) => void;
  selectedRunUuid?: string;
  compareToRunUuid?: string;
  enableImprovedComparison?: boolean;
  setIsComparisonMode?: (isComparisonMode: boolean) => void;
  /** Omitted when the baseline feature is off, which hides the menu item. */
  onSetBaseline?: (runUuid: string) => void;
  baselineRunUuid?: string;
  isSavingBaseline?: boolean;
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

  // Original Compare logic (used when flag is OFF)
  const handleCompare = useCallback(() => {
    if (selectedRunUuids.length === 2 && onCompare) {
      onCompare(selectedRunUuids[0], selectedRunUuids[1]);
      setIsComparisonMode?.(true);
    }
  }, [onCompare, selectedRunUuids, setIsComparisonMode]);

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

  // Exactly one run, the same discipline Compare applies with two. The run has to
  // be unambiguous for "the baseline" to mean anything.
  const isExactlyOneSelected = selectedRunUuids.length === 1;
  const isAlreadyBaseline = isExactlyOneSelected && selectedRunUuids[0] === baselineRunUuid;
  const [promoteModalVisible, setPromoteModalVisible] = useState(false);

  const handleSetBaseline = useCallback(() => {
    if (isExactlyOneSelected) {
      onSetBaseline?.(selectedRunUuids[0]);
    }
    setPromoteModalVisible(false);
  }, [isExactlyOneSelected, onSetBaseline, selectedRunUuids]);

  return (
    <>
      <DropdownMenu.Root>
        <DropdownMenu.Trigger asChild disabled={noRunsSelected}>
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
          {onSetBaseline && (
            <DropdownMenu.Item
              componentId="mlflow.eval-runs.actions.set-baseline"
              onClick={() => (baselineRunUuid ? setPromoteModalVisible(true) : handleSetBaseline())}
              disabled={!isExactlyOneSelected || isAlreadyBaseline || isSavingBaseline}
              disabledReason={
                isAlreadyBaseline ? (
                  <FormattedMessage
                    defaultMessage="This run is already the baseline"
                    description="Tooltip explaining that the selected run is already the experiment baseline"
                  />
                ) : (
                  <FormattedMessage
                    defaultMessage="Please select exactly 1 run to set as the baseline"
                    description="Tooltip for the disabled set-as-baseline action in the evaluation runs table"
                  />
                )
              }
            >
              {baselineRunUuid ? (
                <FormattedMessage
                  defaultMessage="Replace baseline"
                  description="Action that promotes the selected run over the existing experiment baseline"
                />
              ) : (
                <FormattedMessage
                  defaultMessage="Set as baseline"
                  description="Action that marks the selected run as the experiment baseline"
                />
              )}
            </DropdownMenu.Item>
          )}
          {/* Original Compare option in dropdown (when flag is OFF) */}
          {!enableImprovedComparison && (
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
          )}
          {/*
            The Trigger above is disabled when no runs are selected, so this item is normally
            unreachable in that state. We keep the item-level disabled guard (and tooltip, matching
            Compare) as defense-in-depth: it stays correct if the Trigger is ever re-enabled, and it
            prevents opening a "Delete 0 runs" modal.
          */}
          <DropdownMenu.Item
            componentId="mlflow.eval-runs.actions.delete"
            onClick={() => setDeleteModalVisible(true)}
            disabled={noRunsSelected}
            disabledReason={
              <FormattedMessage
                defaultMessage="Please select at least 1 run to delete"
                description="Tooltip for disabled delete action in evaluation runs table actions"
              />
            }
          >
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
      {/*
        Replacing the baseline is confirmed because the baseline is an experiment
        tag: it is shared by everyone viewing the experiment, so silently swapping
        it moves someone else's comparison anchor out from under them.
      */}
      <Modal
        componentId="mlflow.eval-runs.promote-baseline-modal"
        visible={promoteModalVisible}
        onOk={handleSetBaseline}
        okButtonProps={{ loading: isSavingBaseline }}
        okText={
          <FormattedMessage
            defaultMessage="Replace baseline"
            description="Confirm button for replacing the shared experiment baseline"
          />
        }
        onCancel={() => setPromoteModalVisible(false)}
        cancelText={
          <FormattedMessage defaultMessage="Cancel" description="Cancel button for the replace-baseline modal" />
        }
        title={
          <FormattedMessage
            defaultMessage="Replace the baseline?"
            description="Title of the modal confirming replacement of the shared experiment baseline"
          />
        }
      >
        <FormattedMessage
          defaultMessage="The baseline is shared with everyone viewing this experiment. Replacing it changes what every run is compared against, for all viewers."
          description="Body of the modal confirming replacement of the shared experiment baseline"
        />
      </Modal>
    </>
  );
};
