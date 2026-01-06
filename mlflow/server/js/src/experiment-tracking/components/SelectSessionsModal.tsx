import { Empty, Modal } from '@databricks/design-system';
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
  initialSessionIdsSelected?: string[];
}

const SelectSessionsModalImpl = ({ onClose, onSuccess, initialSessionIdsSelected = [] }: SelectSessionsModalProps) => {
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
      okText={<FormattedMessage defaultMessage="Select" description="Confirm button in the select sessions modal" />}
      okButtonProps={{
        type: 'primary',
        disabled: Object.values(rowSelection).every((isSelected) => !isSelected),
      }}
      onOk={handleOk}
      cancelText={<FormattedMessage defaultMessage="Cancel" description="Cancel button in the select sessions modal" />}
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
