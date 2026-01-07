import { Modal } from '@databricks/design-system';
import { useMemo, useState } from 'react';
import { FormattedMessage } from 'react-intl';
import { useParams } from '../../common/utils/RoutingUtils';
import { TracesV3Logs } from './experiment-page/components/traces-v3/TracesV3Logs';
import { GenAiTraceTableRowSelectionProvider } from '@databricks/web-shared/genai-traces-table/hooks/useGenAiTraceTableRowSelection';
import {
  TRACE_ID_COLUMN_ID,
  TracesTableColumn,
  TracesTableColumnType,
} from '@databricks/web-shared/genai-traces-table';
import { INPUTS_COLUMN_ID, RESPONSE_COLUMN_ID } from '@databricks/web-shared/genai-traces-table/hooks/useTableColumns';
import { TracesV3DateSelector } from './experiment-page/components/traces-v3/TracesV3DateSelector';
import {
  MonitoringFilters,
  MonitoringFiltersUpdateContext,
  useMonitoringFilters,
  useMonitoringFiltersTimeRange,
} from '../hooks/useMonitoringFilters';

/**
 * Default columns to be visible when selecting traces.
 */
const defaultCustomDefaultSelectedColumns = (column: TracesTableColumn) => {
  if (column.type === TracesTableColumnType.ASSESSMENT || column.type === TracesTableColumnType.EXPECTATION) {
    return true;
  }
  return [TRACE_ID_COLUMN_ID, INPUTS_COLUMN_ID, RESPONSE_COLUMN_ID].includes(column.id);
};

interface SelectTracesModalProps {
  onClose?: () => void;
  onSuccess?: (traceIds: string[]) => void;
  maxTraceCount?: number;
  customDefaultSelectedColumns?: (column: TracesTableColumn) => boolean;
  initialTraceIdsSelected?: string[];
}

const SelectTracesModalImpl = ({
  onClose,
  onSuccess,
  maxTraceCount,
  customDefaultSelectedColumns = defaultCustomDefaultSelectedColumns,
  initialTraceIdsSelected = [],
}: SelectTracesModalProps) => {
  const { experimentId } = useParams();
  const [monitoringFilters] = useMonitoringFilters();

  const timeRange = useMonitoringFiltersTimeRange();

  const [rowSelection, setRowSelection] = useState<Record<string, boolean>>(
    initialTraceIdsSelected.reduce((acc, traceId) => {
      acc[traceId] = true;
      return acc;
    }, {} as Record<string, boolean>),
  );

  const handleOk = async () => {
    const selectedTraceIds = Object.entries(rowSelection)
      .filter(([_, isSelected]) => isSelected)
      .map(([traceId]) => traceId);
    onSuccess?.(selectedTraceIds);
  };

  const isMaxTraceCountReached = useMemo(() => {
    if (!maxTraceCount) {
      return false;
    }

    return Object.values(rowSelection).filter((isSelected) => isSelected).length > maxTraceCount;
  }, [maxTraceCount, rowSelection]);

  if (!experimentId) {
    return null;
  }

  return (
    <Modal
      visible
      title={<FormattedMessage defaultMessage="Select traces" description="Title for the select traces modal" />}
      componentId="mlflow.experiment-scorers.form.select-traces-modal"
      onCancel={onClose}
      css={{ width: '90% !important' }}
      size="wide"
      verticalSizing="maxed_out"
      okText={<FormattedMessage defaultMessage="Select" description="Confirm button in the select traces modal" />}
      okButtonProps={{
        type: 'primary',
        disabled: Object.values(rowSelection).every((isSelected) => !isSelected) || isMaxTraceCountReached,
      }}
      onOk={handleOk}
      cancelText={<FormattedMessage defaultMessage="Cancel" description="Cancel button in the select traces modal" />}
    >
      <GenAiTraceTableRowSelectionProvider rowSelection={rowSelection} setRowSelection={setRowSelection}>
        <TracesV3Logs
          disableActions
          experimentId={experimentId}
          timeRange={timeRange}
          customDefaultSelectedColumns={customDefaultSelectedColumns}
          // TODO: Move date selector to the toolbar in all callsites permanently
          toolbarAddons={<TracesV3DateSelector />}
        />
      </GenAiTraceTableRowSelectionProvider>
    </Modal>
  );
};

export const SelectTracesModal = (props: SelectTracesModalProps) => {
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
      <SelectTracesModalImpl {...props} />
    </MonitoringFiltersUpdateContext.Provider>
  );
};
