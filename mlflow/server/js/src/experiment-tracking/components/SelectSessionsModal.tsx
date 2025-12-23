import { Modal, TableSkeleton, TableSkeletonRows } from '@databricks/design-system';
import { useState } from 'react';
import { FormattedMessage } from 'react-intl';
import { useParams } from '../../common/utils/RoutingUtils';
import { TracesV3Logs } from './experiment-page/components/traces-v3/TracesV3Logs';
import { GenAiTraceTableRowSelectionProvider } from '@databricks/web-shared/genai-traces-table/hooks/useGenAiTraceTableRowSelection';
import {
  GenAIChatSessionsTable,
  TRACE_ID_COLUMN_ID,
  TracesTableColumn,
  TracesTableColumnType,
  useSearchMlflowTraces,
} from '@databricks/web-shared/genai-traces-table';
import { INPUTS_COLUMN_ID, RESPONSE_COLUMN_ID } from '@databricks/web-shared/genai-traces-table/hooks/useTableColumns';

/**
 * Default columns to be visible when selecting traces.
 */
const defaultCustomDefaultSelectedColumns = (column: TracesTableColumn) => {
  if (column.type === TracesTableColumnType.ASSESSMENT || column.type === TracesTableColumnType.EXPECTATION) {
    return true;
  }
  return [TRACE_ID_COLUMN_ID, INPUTS_COLUMN_ID, RESPONSE_COLUMN_ID].includes(column.id);
};

export const SelectSessionsModal = ({
  onClose,
  onSuccess,
  // customDefaultSelectedColumns = defaultCustomDefaultSelectedColumns,
  initialSessionIdsSelected = [],
}: {
  onClose?: () => void;
  onSuccess?: (traceIds: string[]) => void;
  // customDefaultSelectedColumns?: (column: TracesTableColumn) => boolean;
  initialSessionIdsSelected?: string[];
}) => {
  const { experimentId } = useParams();

  // const [rowSelection, setRowSelection] = useState<Record<string, boolean>>(
  //   initialTraceIdsSelected.reduce((acc, traceId) => {
  //     acc[traceId] = true;
  //     return acc;
  //   }, {} as Record<string, boolean>),
  // );

  const [rowSelection, setRowSelection] = useState<Record<string, boolean>>({});

  const handleOk = async () => {
    const selectedSessionIds = Object.entries(rowSelection)
      .filter(([_, isSelected]) => isSelected)
      .map(([traceId]) => traceId);
    onSuccess?.(selectedSessionIds);
  };

  const { data: traceInfos, isLoading } = useSearchMlflowTraces({
    locations: [{ mlflow_experiment: { experiment_id: experimentId ?? '' }, type: 'MLFLOW_EXPERIMENT' as const }],
    pageSize: 500,
    limit: 500,
    disabled: !experimentId,
  });

  if (!experimentId) {
    return null;
  }

  return (
    <Modal
      visible
      title="Select sessions" // TODO
      componentId="TODO"
      onCancel={onClose}
      css={{ width: '90% !important' }}
      size="wide"
      verticalSizing="maxed_out"
      okText={<FormattedMessage defaultMessage="Select" description="Confirm button in the select traces modal" />}
      okButtonProps={{
        type: 'primary',
        disabled: Object.values(rowSelection).every((isSelected) => !isSelected),
      }}
      onOk={handleOk}
      cancelText={<FormattedMessage defaultMessage="Cancel" description="Cancel button in the select traces modal" />}
    >
      <GenAiTraceTableRowSelectionProvider rowSelection={rowSelection} setRowSelection={setRowSelection}>
        {isLoading || !traceInfos ? (
          <TableSkeleton lines={3} />
        ) : (
          <GenAIChatSessionsTable
            experimentId={experimentId}
            traces={traceInfos}
            isLoading={false}
            searchQuery=""
            setSearchQuery={() => {}}
            forceEnableRowSelection
          />
        )}
      </GenAiTraceTableRowSelectionProvider>
    </Modal>
  );
};
