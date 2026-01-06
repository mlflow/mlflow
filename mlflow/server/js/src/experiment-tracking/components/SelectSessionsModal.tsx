import { Button, Empty, Modal, Tooltip } from '@databricks/design-system';
import { useMemo, useState } from 'react';
import { FormattedMessage } from 'react-intl';
import { useParams } from '../../common/utils/RoutingUtils';
import { GenAiTraceTableRowSelectionProvider } from '@databricks/web-shared/genai-traces-table/hooks/useGenAiTraceTableRowSelection';
import { GenAIChatSessionsTable, useSearchMlflowTraces } from '@databricks/web-shared/genai-traces-table';
import { getChatSessionsFilter } from '../pages/experiment-chat-sessions/utils';
import { TracesV3DateSelector } from './experiment-page/components/traces-v3/TracesV3DateSelector';
import {
  MonitoringFilters,
  MonitoringFiltersUpdateContext,
  useMonitoringFiltersTimeRange,
} from '../hooks/useMonitoringFilters';

interface SelectSessionsModalProps {
  onClose?: () => void;
  onSuccess?: (sessionIds: string[]) => void;
  maxSessionCount?: number;
  initialSessionIdsSelected?: string[];
}

const SelectSessionsModalImpl = ({
  onClose,
  onSuccess,
  maxSessionCount,
  initialSessionIdsSelected = [],
}: SelectSessionsModalProps) => {
  const { experimentId } = useParams();

  const [searchQuery, setSearchQuery] = useState('');

  const timeRange = useMonitoringFiltersTimeRange();

  const [rowSelection, setRowSelection] = useState<Record<string, boolean>>(() =>
    initialSessionIdsSelected.reduce((acc, sessionId) => {
      acc[sessionId] = true;
      return acc;
    }, {} as Record<string, boolean>),
  );

  const handleOk = async () => {
    const selectedSessionIds = Object.entries(rowSelection)
      .filter(([_, isSelected]) => isSelected)
      .map(([traceId]) => traceId);
    onSuccess?.(selectedSessionIds);
  };

  const selectedCount = useMemo(() => {
    return Object.values(rowSelection).filter((isSelected) => isSelected).length;
  }, [rowSelection]);

  const isMaxSessionCountReached = useMemo(() => {
    if (!maxSessionCount) {
      return false;
    }

    return selectedCount > maxSessionCount;
  }, [maxSessionCount, selectedCount]);

  const filters = useMemo(() => getChatSessionsFilter({ sessionId: null }), []);

  const { data: traceInfos, isLoading } = useSearchMlflowTraces({
    locations: [{ mlflow_experiment: { experiment_id: experimentId ?? '' }, type: 'MLFLOW_EXPERIMENT' as const }],
    disabled: !experimentId,
    filters,
    searchQuery,
    timeRange,
  });

  if (!experimentId) {
    return null;
  }

  return (
    <Modal
      visible
      title={<FormattedMessage defaultMessage="Select sessions" description="Title for the select sessions modal" />}
      componentId="mlflow.experiment-scorers.form.select-sessions-modal"
      onCancel={onClose}
      css={{ width: '90% !important' }}
      size="wide"
      verticalSizing="maxed_out"
      footer={
        <>
          <Button componentId="mlflow.experiment-scorers.form.select-sessions-modal.cancel" onClick={onClose}>
            <FormattedMessage defaultMessage="Cancel" description="Cancel button in the select sessions modal" />
          </Button>
          <Tooltip
            componentId="mlflow.experiment-scorers.form.select-sessions-modal.ok-tooltip"
            content={
              isMaxSessionCountReached ? (
                <FormattedMessage
                  defaultMessage="Maximum of {max} sessions can be selected"
                  description="Tooltip shown when too many sessions are selected"
                  values={{ max: maxSessionCount }}
                />
              ) : undefined
            }
          >
            <Button
              componentId="mlflow.experiment-scorers.form.select-sessions-modal.ok"
              type="primary"
              onClick={handleOk}
              disabled={selectedCount === 0 || isMaxSessionCountReached}
            >
              <FormattedMessage
                defaultMessage="Select ({count})"
                description="Confirm button in the select sessions modal showing number of selected sessions"
                values={{ count: selectedCount }}
              />
            </Button>
          </Tooltip>
        </>
      }
    >
      <div css={{ height: '100%', display: 'flex', overflow: 'hidden' }}>
        <GenAiTraceTableRowSelectionProvider rowSelection={rowSelection} setRowSelection={setRowSelection}>
          <GenAIChatSessionsTable
            experimentId={experimentId}
            traces={traceInfos ?? []}
            isLoading={isLoading}
            searchQuery={searchQuery}
            setSearchQuery={setSearchQuery}
            enableRowSelection
            enableLinks={false}
            empty={<EmptySessionsList />}
            // TODO: Move date selector to the toolbar in all callsites permanently
            toolbarAddons={<TracesV3DateSelector />}
          />
        </GenAiTraceTableRowSelectionProvider>
      </div>
    </Modal>
  );
};

const EmptySessionsList = () => {
  return (
    <div css={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}>
      <Empty
        title={
          <FormattedMessage
            defaultMessage="No sessions found"
            description="Title for the empty sessions list in the select sessions modal"
          />
        }
        description={null}
      />
    </div>
  );
};

export const SelectSessionsModal = (props: SelectSessionsModalProps) => {
  const [monitoringFilters, setMonitoringFilters] = useState<MonitoringFilters>({
    startTimeLabel: 'LAST_7_DAYS',
    startTime: undefined,
    endTime: undefined,
  });
  const contextValue = useMemo(
    () => ({ params: monitoringFilters, setParams: setMonitoringFilters, disableAutomaticInitialization: true }),
    [monitoringFilters, setMonitoringFilters],
  );
  return (
    <MonitoringFiltersUpdateContext.Provider value={contextValue}>
      <SelectSessionsModalImpl {...props} />
    </MonitoringFiltersUpdateContext.Provider>
  );
};
