import type { RowSelectionState } from '@tanstack/react-table';
import { useCallback, useMemo, useState } from 'react';

import { Button, Tooltip, DropdownMenu, ChevronDownIcon } from '@databricks/design-system';
import { useIntl } from '@databricks/i18n';
import type { ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import { displayErrorNotification, displaySuccessNotification } from '@databricks/web-shared/model-trace-explorer';

import type { SessionTableRow } from './types';
import { GenAiDeleteTraceModal } from '../components/GenAiDeleteTraceModal';
import type { TraceActions } from '../types';

interface GenAIChatSessionsActionsProps {
  experimentId: string;
  selectedSessions: SessionTableRow[];
  traceActions?: TraceActions;
  setRowSelection?: React.Dispatch<React.SetStateAction<RowSelectionState>>;
}

export const GenAIChatSessionsActions = (props: GenAIChatSessionsActionsProps) => {
  const { experimentId, selectedSessions, traceActions, setRowSelection } = props;
  const intl = useIntl();
  const [showDeleteModal, setShowDeleteModal] = useState(false);

  const handleDeleteSessions = useCallback(() => {
    setShowDeleteModal(true);
  }, []);

  // Convert selected sessions to traces for deletion
  const tracesToDelete = useMemo(() => {
    const allTracesInSelectedSessions: ModelTraceInfoV3[] = [];

    selectedSessions.forEach((session) => {
      allTracesInSelectedSessions.push(...session.traces);
    });

    return allTracesInSelectedSessions.map((trace) => ({
      evaluationId: trace.trace_id,
      requestId: trace.client_request_id || trace.trace_id,
      inputsId: trace.trace_id,
      inputs: {},
      outputs: {},
      targets: {},
      overallAssessments: [],
      responseAssessmentsByName: {},
      metrics: {},
      traceInfo: trace,
    }));
  }, [selectedSessions]);

  const deleteTraces = useCallback(
    async (experimentId: string, traceIds: string[]) => {
      try {
        await traceActions?.deleteTracesAction?.deleteTraces?.(experimentId, traceIds);
        setRowSelection?.({});
        setShowDeleteModal(false);

        const sessionMessage = intl.formatMessage(
          {
            defaultMessage: '{count, plural, one {Session} other {{count} sessions}} deleted successfully',
            description: 'Success message after deleting sessions - session count',
          },
          { count: selectedSessions.length },
        );

        const traceMessage = intl.formatMessage(
          {
            defaultMessage: '{count, plural, one {# trace was} other {# traces were}} removed',
            description: 'Success message after deleting sessions - trace count',
          },
          { count: traceIds.length },
        );

        displaySuccessNotification(`${sessionMessage}. ${traceMessage}.`);
      } catch (error) {
        displayErrorNotification(
          intl.formatMessage(
            {
              defaultMessage: 'Failed to delete sessions. Error: {error}',
              description: 'Error message when deleting sessions fails',
            },
            {
              error: error instanceof Error ? error.message : String(error),
            },
          ),
        );
        throw error;
      }
    },
    [setRowSelection, traceActions, selectedSessions.length, intl],
  );

  const hasDeleteAction = Boolean(traceActions?.deleteTracesAction);
  const noSessionsSelected = selectedSessions.length === 0;
  const noActionsAvailable = !hasDeleteAction;

  if (noActionsAvailable) {
    return null;
  }

  const ActionButton = (
    <Button
      componentId="mlflow.chat-sessions.actions-dropdown"
      endIcon={<ChevronDownIcon />}
      disabled={noSessionsSelected || traceActions?.deleteTracesAction?.isDisabled}
    >
      {intl.formatMessage(
        {
          defaultMessage: 'Actions ({count})',
          description: 'Actions dropdown button label with count of selected sessions',
        },
        { count: selectedSessions.length },
      )}
    </Button>
  );

  return (
    <>
      <DropdownMenu.Root modal={false}>
        <Tooltip
          componentId="mlflow.chat-sessions.actions-dropdown-tooltip"
          content={
            noSessionsSelected
              ? intl.formatMessage({
                  defaultMessage: 'Select at least one session to enable actions',
                  description: 'Tooltip for disabled actions button when no sessions are selected',
                })
              : traceActions?.deleteTracesAction?.disabledReason
          }
        >
          <DropdownMenu.Trigger asChild>{ActionButton}</DropdownMenu.Trigger>
        </Tooltip>
        <DropdownMenu.Content align="end">
          {hasDeleteAction && (
            <>
              <DropdownMenu.Item
                componentId="mlflow.chat-sessions.delete-sessions"
                onClick={handleDeleteSessions}
                disabled={traceActions?.deleteTracesAction?.isDisabled}
                disabledReason={traceActions?.deleteTracesAction?.disabledReason}
              >
                {intl.formatMessage({
                  defaultMessage: 'Delete sessions',
                  description: 'Delete sessions action',
                })}
              </DropdownMenu.Item>
            </>
          )}
        </DropdownMenu.Content>
      </DropdownMenu.Root>

      <GenAiDeleteTraceModal
        experimentIds={[experimentId]}
        visible={showDeleteModal}
        selectedTraces={tracesToDelete}
        handleClose={() => setShowDeleteModal(false)}
        deleteTraces={deleteTraces}
      />
    </>
  );
};
